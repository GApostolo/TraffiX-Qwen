# TraffiX-Qwen Multi-GPU Batch Inference Script
# This script performs batch inference on TUMTraf VideoQA dataset.

from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import argparse
from typing import Dict
import tempfile
import os
import json
import torch
import copy
import numpy as np
from tqdm import tqdm
import math
from decord import VideoReader, cpu
from llava.train.train import DataArguments, LazySupervisedDataset, DataCollatorForSupervisedDataset, \
    preprocess_multimodal, preprocess, _add_speaker_and_signal, _tokenize_fn, _mask_targets
from llava.utils import process_video_with_decord
import transformers
from transformers import HfArgumentParser
from llava import conversation as conversation_lib
from multiprocessing import Process, Lock
import re
import time


def parse_tuples(entry):
    """
    Parse the coordinate tuple from a given entry string.
    Returns a list of parsed coordinates or an empty list if parsing fails.
    """
    matches = re.findall(r'[\[\(]?(?:c\d+,)?\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)[\]\)]?', entry)
    if len(matches) > 0:
        return list(set([float(match[1]) for match in matches]))
    else:
        return []


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False):
    """
    Preprocess Qwen-format data for inference, extracting key frames.
    """
    question = sources[0][0]['value']
    conv = copy.deepcopy(conv_templates["qwen_traffic"])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    targets = copy.deepcopy(input_ids)
    key_frames = []
    for source in sources:
        question = source[0]["value"]
        key_time = parse_tuples(question)
        key_frames.append(key_time)
    return dict(input_ids=input_ids, labels=targets, key_frames=key_frames)


class LazySupervisedDatasetEval(LazySupervisedDataset):
    """
    Dataset class for lazy loading and preprocessing of evaluation data.
    """

    def __init__(self, data_path, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.video_list = []
        self.list_data_dict = []
        if isinstance(data_path, list):
            self.list_data_dict = data_path
        else:
            with open(data_path, "r") as f:
                self.list_data_dict = json.load(f)
        for sample in self.list_data_dict:
            video_file = sample.get("video", None)
            video_path = os.path.join(self.data_args.video_folder, video_file)
            sample["gt_answer"] = sample["conversations"][1]['value']
            sample["conversations"] = [convo for convo in sample["conversations"] if convo["from"] == "human"]
            if video_path not in self.video_list:
                self.video_list.append(video_path)

    def _get_item(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            if not os.path.exists(video_file):
                print(f"File {video_file} not exist!")
            try:
                video, video_time, frame_time, num_frames_to_sample, video_frame_idx, frame_time_norm = process_video_with_decord(
                    video_file, self.data_args)
                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                image = [(image, [video[0].size] * image.shape[0], ["video"] * image.shape[0])]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess_qwen(sources, self.tokenizer, has_image=has_image)
        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0],
                             key_frames=data_dict["key_frames"][0])
        data_dict["graphs"] = None
        if "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        data_dict["id"] = self.list_data_dict[i].get("id", i)
        return data_dict


def read_processed_ids(output_file):
    """
    Read already processed IDs from the output file to support resuming.
    """
    processed_ids = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    result_data = json.loads(line)
                    processed_ids.add(result_data['id'])
                except json.JSONDecodeError:
                    print(f"Warning: Line in {output_file} is not valid JSON.")
    return processed_ids


def process_batch(data_batch, data_args, model, tokenizer, device, data_collator):
    """
    Process a batch of data and run inference.
    """
    dataset = LazySupervisedDatasetEval(
        tokenizer=tokenizer,
        data_path=data_batch,
        data_args=data_args
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=len(data_batch),
        collate_fn=data_collator,
        num_workers=0
    )
    results = []
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_tensors = [img_tensor.half().to(device) for img_tensor in batch["images"]]
        image_sizes = [item for sublist in batch["image_sizes"] for item in sublist]
        modalities = [item for sublist in batch["modalities"] for item in sublist]
        key_frames = batch["key_frames"]
        generated_texts = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=128,
            modalities=modalities,
            output_scores=True,
            return_dict_in_generate=True,
            key_frames=key_frames
        )
        decoded_texts = tokenizer.batch_decode(generated_texts['sequences'], skip_special_tokens=True)
        for idx, data_item in enumerate(data_batch):
            print("Question:", data_item['conversations'][0]['value'], " Answer: ", decoded_texts[idx])
            results.append({
                'id': data_item['id'],
                'question': data_item['conversations'][0]['value'],
                'answer': decoded_texts[idx],
                'gt_answer': data_item["gt_answer"],
                'question_type': data_item['type']
            })
    return results


def main(args):
    """
    Main function to launch multi-GPU inference.
    """
    args_dict = vars(args)
    data_arg_fields = {field for field in DataArguments.__dataclass_fields__}
    filtered_args_dict = {k: v for k, v in args_dict.items() if k in data_arg_fields}
    parser = HfArgumentParser(DataArguments)
    data_args = parser.parse_dict(filtered_args_dict)[0]
    setattr(data_args, "mm_use_im_start_end", args_dict.get("mm_use_im_start_end", False))
    torch.cuda.set_device(0)
    device = torch.device(f"cuda:0")
    llava_model_args = {"multimodal": True}
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if args.lora_path is not None:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.lora_path,
            args.model_path,
            "llava_qwen_lora",
            device_map=None,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            **llava_model_args
        )
    else:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.model_path,
            None,
            "llava_qwen",
            device_map=None,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            **llava_model_args
        )
    model.to(device)
    model.eval()
    data_args.image_processor = image_processor
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    result = []
    for i in tqdm(range(0, len(args.prompt), args.batch_size), desc="Processing"):
        try:
            data_batch = args.prompt[i:i + args.batch_size]
            print(data_batch)
            batch_results = process_batch(data_batch, data_args, model, tokenizer, device, data_collator)
            result.append(batch_results[0])
        except Exception as e:
            print(f"Error executing rompt")
        finally:
            torch.cuda.empty_cache()
        return result


if __name__ == '__main__':
    import argparse

    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="./saved_weights/traffix-qwen-google_siglip-so400m-patch14-384-Qwen-Qwen2-0.5B-Instruct-mm_language_model-None")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--video-folder", type=str, default="./TUMTrafficQA/raw_videos")
    parser.add_argument("--frames-upbound", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--is-multimodal", type=bool, default=True)
    parser.add_argument("--use-graph", type=bool, default=False)
    args = parser.parse_args()
    # decode JSON → dict
    with open(args.prompt, "r") as f:
        args.prompt = json.load(f)

    result = main(args)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(result, f)
        print(f.name)

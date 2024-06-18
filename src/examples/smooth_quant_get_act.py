import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
        )
        scale_dict["q_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
        )
        scale_dict["k_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
        )
        scale_dict["v_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
        )
        scale_dict["out_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
        )
        scale_dict["fc1_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
        )
        scale_dict["fc2_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict


import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse


def build_model_and_tokenizer(model_name):

    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512, trust_remote_code=True)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--output-path', type=str, default='act_scales/opt-1.3b.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args

if __name__ == '__main__':


 

    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)



 
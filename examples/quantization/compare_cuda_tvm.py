import os
import sys
sys.path.append("/workspace/v-leiwang3/AutoGPTQ_nf4")

from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import gradio as gr  
import torch
import torch.nn as nn
import time
from logging import getLogger
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np

enable_quant = False
export_tvm = True

pretrained_model_dir = "facebook/opt-125m"
tvm_quantized_model_dir = "save/opt125m_nf4_tvm"
cuda_quantized_model_dir = "save/opt125m_nf4_cuda"
# pretrained_model_dir = "/workspace/v-leiwang3/mlc-llm/dist/models/vicuna-v1-7b"
# quantized_model_dir = "save/vicuna_7b_nf4_ng_tvm"

# pretrained_model_dir = "meta-llama/Llama-2-7b-hf"
# quantized_model_dir = "save/llama2-7b_nf4_g128_tvm"

# pretrained_model_dir = "meta-llama/Llama-2-13b-hf"
# quantized_model_dir = "save/llama2-13b_nf4_g128"

# pretrained_model_dir = "/workspace/v-leiwang3/AutoGPTQ_nf4/models/Llama-2-70b-hf"
# quantized_model_dir = "save/llama2-70b_nf4_g128"



def main():
    logger = getLogger(__name__)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    # traindataset,testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
   
    tvm_model = AutoGPTQForCausalLM.from_quantized(tvm_quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=True,
        inject_fused_attention=False, inject_fused_mlp=False
    )
    cuda_model = AutoGPTQForCausalLM.from_quantized(cuda_quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=False,
        inject_fused_attention=False, inject_fused_mlp=False
    )
    # model.model.model.decoder.layers = model.model.model.decoder.layers[:1]
    use_single_layer = False
    if use_single_layer:
        model = model.model.model
    # export 2 onnx
    shape_list = [
        # [1, 1],
        # [1, 16]
        # [1, 64]
        [1, 55]
    ]
    for batch_size, seq_length in shape_list:
        input_shape = (batch_size, seq_length)
        input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
        tvm_outputs = tvm_model(input_ids)
        cuda_outputs = cuda_model(input_ids)
        tvm_np = tvm_outputs.logits.cpu().detach().numpy()
        cuda_np = cuda_outputs.logits.cpu().detach().numpy()
        np.testing.assert_allclose(tvm_np, cuda_np, rtol=5e-2, atol=5e-2)
        # print the max abs diff
        print("max abs diff is ", np.max(np.abs(tvm_np - cuda_np)))
        max_abs_diff = np.max(np.abs(tvm_np - cuda_np))
        max_abs_diff_index = np.argmax(np.abs(tvm_np - cuda_np))
        print("max abs diff index is ", max_abs_diff_index)
        # print the value of the max abs diff
        print("tvm value is ", tvm_np.flatten()[max_abs_diff_index])
        print("cuda value is ", cuda_np.flatten()[max_abs_diff_index])
        

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

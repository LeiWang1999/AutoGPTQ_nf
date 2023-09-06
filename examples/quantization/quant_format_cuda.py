import os
import sys
sys.path.append("/workspace/v-leiwang3/AutoGPTQ_nf4")

from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
import torch
import torch.nn as nn
import time
from logging import getLogger

enable_quant = False
export_nnfusion = True

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "save/opt125m_nf4_cuda"

pretrained_model_dir = "/workspace/v-leiwang3/mlc-llm/dist/models/vicuna-v1-7b"
quantized_model_dir = "save/vicuna_7b_nf4_ng_cuda"

# pretrained_model_dir = "meta-llama/Llama-2-7b-hf"
# quantized_model_dir = "save/llama2-7b_nf4_g128"

# pretrained_model_dir = "meta-llama/Llama-2-13b-hf"
# quantized_model_dir = "save/llama2-13b_nf4_g128"

# pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/AutoGPTQ_nf4/models/Llama-2-70b-hf"
# quantized_model_dir = "save/llama2-70b_nf4_g128"

def main():
    logger = getLogger(__name__)

    if enable_quant:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
        examples = [
            tokenizer(
                "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
            )
        ]
        quantize_config = BaseQuantizeConfig(
            bits=4,  # quantize model to 4-bit
            format='nf', # quantize model to int / nf / fp
            group_size=-1,  # it is recommended to set the value to 128
            desc_act=False,  # desc_act and group size only works on triton
        )

        # load un-quantized model, the model will always be force loaded into cpu
        model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, cache_dir='/gptq_hub')
        # model.model.model.layers = model.model.model.layers[:1]
        # model.model.model.decoder.layers = model.model.model.decoder.layers[:1]

        # quantize model
        time_start = time.time()
        model.quantize(examples, use_triton=False)
        logger.info('quant time: %ds' % (time.time() - time_start))

        model.save_quantized(quantized_model_dir)

    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, 
        inject_fused_attention=False, inject_fused_mlp=False
    )
    # model.model.model.decoder.layers = model.model.model.decoder.layers[:1]
    # model = model.model.model.layers[0]
    # export 2 onnx
    shape_list = [
        [1, 1],
        # [1, 16]
        # [1, 64]
    ]
    for batch_size, seq_length in shape_list:
        input_shape = (batch_size, seq_length)
        input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
        
    #     start = time.time()
    #     for i in range(1):
    #         outputs = model(input_ids=input_ids)
    #     end = time.time()
    #     print("output", outputs[0])
    #     print("time", end - start)
        outputs = model(input_ids=input_ids)
        print("output", outputs[0])
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    print(pipeline("hello. ", max_length=100)[0]["generated_text"])

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

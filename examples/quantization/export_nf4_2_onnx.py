import os
import sys
sys.path.append("/workspace/v-leiwang3/lowbit_workspace/AutoGPTQ_nf4")

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
# quantized_model_dir = "save/opt125m_nf4"

# pretrained_model_dir = "meta-llama/Llama-2-7b-hf"
# quantized_model_dir = "save/llama2-7b_nf4_g128"

# pretrained_model_dir = "meta-llama/Llama-2-13b-hf"
# quantized_model_dir = "save/llama2-13b_nf4_g128"

pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/AutoGPTQ_nf4/models/Llama-2-70b-hf"
quantized_model_dir = "save/llama2-70b_nf4_g128"


def main():
    logger = getLogger(__name__)
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    # traindataset,testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
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
            group_size=128,  # it is recommended to set the value to 128
            desc_act=False,  # desc_act and group size only works on triton
        )

        # load un-quantized model, the model will always be force loaded into cpu
        model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, cache_dir='/gptq_hub')
        model.model.model.layers = model.model.model.layers[:1]

        # quantize model
        time_start = time.time()
        model.quantize(examples, use_triton=False)
        logger.info('quant time: %ds' % (time.time() - time_start))

        model.save_quantized(quantized_model_dir)

    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, 
        inject_fused_attention=False, inject_fused_mlp=False
    )
    configs = [
        "qweight", "scales"
    ]
    quant_layers = [
        "k_proj", "v_proj", "q_proj", "o_proj",
        "down_proj", "up_proj"
    ]
    qlayer_params = {}
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        print(k, v.shape)
        for qlayer in quant_layers:
            # model.model.layers.0.mlp.down_proj.weight -> model.model.layers.0.mlp.down_proj.weight
            layer_name = ".".join(k.split(".")[:-1])
            attr_name = k.split(".")[-1]
            print(layer_name, attr_name)
            if qlayer in k:
                if attr_name not in configs:
                    continue
                if layer_name not in qlayer_params:
                    qlayer_params[layer_name] = {}
                qlayer_params[layer_name][attr_name] = v
    
    print(qlayer_params.keys())
    
    lut = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
    # combined each layer's params into binary file
    for layer_name, layer_params in qlayer_params.items():
        print(layer_name, layer_params.keys())
        binary_file = os.path.join("/workspace/v-leiwang3/xbox_workspace/llama2_70b_nf4/single_layer_single/data_storage", layer_name + ".bin")
        # binary data = [qweight, scales, g_idx, lut], condense them into one binary file
        with open(binary_file, "wb") as f:
            data = []
            data.append(layer_params["qweight"].cpu().numpy())
            data.append(layer_params["scales"].cpu().numpy())
            # data.append(layer_params["g_idx"].cpu().numpy())
            data.append(lut.cpu().numpy())
            np.save(f, np.array(data))    


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

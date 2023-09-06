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

pretrained_model_dir = "/workspace/v-leiwang3/lowbit_workspace/AutoGPTQ_nf4/models/Llama-2-70b-hf"
quantized_model_dir = "save/llama2-70b_nf4_g128"


def main():
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir)
    # model.model.layers = model.model.layers[:1]
    
    # export 2 onnx
    shape_list = [
        [1, 1],
        [16, 1],
        [32, 1],
        [64, 1],
        [1, 128],
        [64, 128],
        [1, 1024],
        [1, 4096],
    ]
    for batch_size, seq_length in shape_list:
        input_shape = (batch_size, seq_length)
        onnx_dir = os.path.join(quantized_model_dir, f"model_b{batch_size}s{seq_length}")
        if not os.path.exists(onnx_dir):
            os.makedirs(onnx_dir)
        onnx_name = f"model.onnx"
        output_path = os.path.join(onnx_dir, onnx_name)
        input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
    
        import onnx
        from onnxsim import simplify
        # model.model.model.layers = model.model.model.layers[:4]
        _model = model.half().cuda()
        torch.onnx.export(      
            _model,  
            input_ids,  
            f=output_path,  
            opset_version=13, 
            export_params=True,
        )
        # load your predefined ONNX model
        _model = onnx.load(output_path)
        # convert model
        model_simp, check = simplify(_model)
        sim_output_path = os.path.join(onnx_dir, f"model_sim.onnx")
        onnx.save(model_simp, sim_output_path)
    

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

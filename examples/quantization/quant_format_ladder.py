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

enable_quant = False
export_tvm = True

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "save/opt125m_nf4_tvm"

# pretrained_model_dir = "/workspace/v-leiwang3/mlc-llm/dist/models/vicuna-v1-7b"
# quantized_model_dir = "save/vicuna_7b_nf4_ng_tvm"

# pretrained_model_dir = "meta-llama/Llama-2-7b-hf"
# quantized_model_dir = "save/llama2-7b_nf4_g128_tvm"

# pretrained_model_dir = "meta-llama/Llama-2-13b-hf"
# quantized_model_dir = "save/llama2-13b_nf4_g128"

pretrained_model_dir = "/workspace/v-leiwang3/AutoGPTQ_nf4/models/Llama-2-70b-hf"
quantized_model_dir = "save/llama2-70b_nf4_g128"

class WrapperModelSingle(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModelSingle, self).__init__()
        self._model = model

    def forward(self, input_ids, attention_mask, past_key_values, cur_index):
        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=None,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            cur_index=cur_index
            )
        return out


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
        # model.model.model.decoder.layers = model.model.model.decoder.layers[:1]

        # quantize model
        time_start = time.time()
        model.quantize(examples, use_triton=False, use_tvm=export_tvm)
        logger.info('quant time: %ds' % (time.time() - time_start))

        model.save_quantized(quantized_model_dir)

    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=export_tvm,
        inject_fused_attention=False, inject_fused_mlp=False
    )
    # model.model.model.decoder.layers = model.model.model.decoder.layers[:1]
    use_single_layer = True
    if use_single_layer:
        model = model.model.model
    # export 2 onnx
    shape_list = [
        [1, 1],
        [1, 16],
        [1, 32],
        [1, 64],
        [1, 128],
        [1, 2048],
        [1, 4096],
        [1, 8192],
    ]
    for batch_size, seq_length in shape_list:
        hid = 8192
        if use_single_layer:
            input_shape = (batch_size, seq_length)
        else:
            input_shape = (batch_size, seq_length)
        onnx_dir = os.path.join(quantized_model_dir, f"qmodel_b{batch_size}s{seq_length}")
        if not os.path.exists(onnx_dir):
            os.makedirs(onnx_dir)
        onnx_name = f"qmodel.onnx"
        output_path = os.path.join(onnx_dir, onnx_name)
        input_ids = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones(input_shape, dtype=torch.long, device="cuda:0")
        iter_time = 100
        profile_iter_time = 10
        start = time.time()
        torch.cuda.synchronize()
        for i in range(iter_time):
            model(input_ids=input_ids)
        torch.cuda.synchronize()
        end = time.time()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("record_function"):
                for _ in range(profile_iter_time):
                    model(input_ids)
        print(
            "Total CUDA time %.3f us"
            % (prof.key_averages().total_average().cuda_time_total / profile_iter_time)
        )
        print(prof.key_averages().table(sort_by=f"cuda_time_total", row_limit=10))
        outputs = model(input_ids)
        print("output", outputs[0])
        print("time", (end - start) / iter_time, " s")


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    tokenizer.padding_side = 'left'
    
    generate_config = {
        "bos_token_id": 1,
        "do_sample": True,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "temperature": 0.6,
        "max_length": 256,
        "top_p": 0.9,
        "transformers_version": "4.31.0.dev0"
    }
    
    # for prompt in prompts:
    #     print('p:')
    #     print(prompt)
    #     print('a:')
    #     input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()  

    #     # 4. 使用配置的生成参数生成文本  
    #     output_ids = model.generate(input_ids=input_ids, **generate_config, num_beams=4)  
    #     # 5. 对生成的输出进行解码  
    #     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  
    #     print("Generated text:", output_text)   

    # def generate_response(user_input):  
    #     input_ids = tokenizer.encode(user_input, return_tensors="pt").cuda()  

    #     # 4. 使用配置的生成参数生成文本  
    #     output_ids = model.generate(input_ids=input_ids, **generate_config)  
    #     # 5. 对生成的输出进行解码  
    #     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)  

    #     return output_text  

    # # iface = gr.Interface(  
    # #     fn=generate_response,  
    # #     inputs=gr.inputs.Textbox(lines=5, placeholder="Type your message here..."),  
    # #     outputs=gr.outputs.Textbox(),  
    # #     title="Chatbot Demo",  
    # #     description="This is an AI chatbot demo using DialoGPT-medium.",
    # #     live=True,  
    # # )  
    
    # # iface.launch()  
    
    # print("Type 'exit' or 'quit' to end the session.")
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         break
    #     model_output = generate_response(user_input)
    #     print(f"Model: {model_output}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()

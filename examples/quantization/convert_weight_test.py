from transformers import AutoModelForCausalLM  
import torch  
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "facebook/opt-125m"  
quantized_model_dir = "save/opt125m_nf4_tvm"  
quantized_model_dir_permute = "save/opt125m_nf4_ladder"  
  
def permute_qlinear_layer(model):  
    # Assuming the model has a qlinear layer named 'quantized_qlinear'  
    layers = model.get_layers()
  
    return model  
  
def main(): 
     
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False, use_tvm=True,
        inject_fused_attention=False, inject_fused_mlp=False
    )
  
    # Permute the qlinear layer  
    model = permute_qlinear_layer(model)  
  
    # Use the model for your tasks  
    model.save_quantized(quantized_model_dir_permute)
    
  
if __name__ == "__main__":  
    import logging  
  
    logging.basicConfig(  
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"  
    )  
  
    main()  

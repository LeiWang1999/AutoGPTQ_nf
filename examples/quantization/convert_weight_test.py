from transformers import AutoModelForCausalLM
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "save/opt125m_nf4_tvm"
quantized_model_dir_permute = "save/opt125m_nf4_ladder"

bits = 4
mask = (1 << bits) - 1


def weight_permuate(weight):
    assert weight.dim() == 2
    weight_shape = weight.shape
    N = weight_shape[0]
    QK = weight_shape[1]
    K = QK // 4 * 8

    weight_int8 = torch.zeros((N, K), dtype=torch.int8)
    for vi in range(N):
        for vj in range(K):
            weight_int8[vi][vj] = weight[vi][vj // 2] >> (vj % 2 * 4) & mask

    weight_prmt_int8 = torch.zeros((N, K), dtype=torch.int8)
    for i in range(N):
        for j in range(K):
            weight_prmt_int8[i][j] = weight_int8[i // 8 * 8 + i % 4 * 2 + j % 16 // 8][j // 16 * 16 + i % 8 // 4 * 8 + j % 8]
    weight_prmt_compress = torch.zeros((N, QK), dtype=torch.int8)
    for i in range(N):
        for j in range(K):
            weight_prmt_compress[i][j // 2] |= weight_prmt_int8[i][j] << (j % 2 * 4)
    # transform
    weight_prmt = torch.zeros((N // 16, K // 16, 16, 8), dtype=torch.int8)
    for i in range(N):
        for j in range(QK):
            weight_prmt[i // 16][j // 8][i % 16][j % 8] = weight_prmt_compress[i][j]
    
    return weight_prmt


def permute_qlinear_layer(model):
    # Assuming the model has a qlinear layer named 'quantized_qlinear'
    layers = model.model.model.decoder.layers
    inner_layers = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ]
    for i, layer in enumerate(layers):
        print("Permute layer {}".format(i))
        for _attr in inner_layers:
            if hasattr(layer, _attr):
                qweight = getattr(layer, _attr).qweight
                weight_prmt = weight_permuate(qweight)
                getattr(layer, _attr).qweight.data = weight_prmt 
    
    return model


def main():
    model = AutoGPTQForCausalLM.from_quantized(
        quantized_model_dir,
        device="cuda:0",
        use_triton=False,
        use_tvm=True,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )

    # Permute the qlinear layer
    model = permute_qlinear_layer(model)
    model.model.model.decoder.layers = model.model.model.decoder.layers[:1]
    # Use the model for your tasks
    model.save_quantized(quantized_model_dir_permute)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()

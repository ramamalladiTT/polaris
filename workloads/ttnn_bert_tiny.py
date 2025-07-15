# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pytest
import torch
from transformers import BertForQuestionAnswering
import ttsim.front.functional.sim_ttnn as ttnn


def bert_attention(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = ttnn.linear(
        hidden_states,
        parameters.self.query.weight,
        bias=parameters.self.query.bias,
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = ttnn.linear(
        hidden_states,
        parameters.self.key.weight,
        bias=parameters.self.key.bias,
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = ttnn.linear(
        hidden_states,
        parameters.self.value.weight,
        bias=parameters.self.value.bias,
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = ttnn.matmul(query, key)
    attention_scores = ttnn.to_device(attention_scores, device)
    #attention_scores = attention_scores * (1 / (head_size**0.5)) ## Need to determine if this is to be handled as mul, div ops
    if attention_mask is not None:
        attention_scores = ttnn.to_layout(attention_scores, ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)

        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    #context_layer = attention_probs @ value ## this is matmul
    context_layer = ttnn.matmul(attention_probs, value)  # Using ttnn.matmul

    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))

    self_output = context_layer
    self_output = ttnn.linear(
        self_output,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    attention_output = ttnn.layer_norm(
        hidden_states + self_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None, #ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return attention_output


def bert_intermediate(
    hidden_states,
    device=None,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        activation="gelu", ## what is impact of activation here in terms of perf stats?
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return output


def bert_output(
    config,
    hidden_states,
    residual,
    device=None,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight.T,
        bias=parameters.dense.bias,
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    output = ttnn.layer_norm(
        output + residual,
        weight=parameters.LayerNorm.weight,
        bias=parameters.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None, #ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return output


def bert_feedforward(
    config,
    hidden_states,
    device=None,
    *,
    parameters,
):
    intermediate = bert_intermediate(hidden_states, parameters=parameters.intermediate, device=device)
    hidden_states = bert_output(config, intermediate, hidden_states, parameters=parameters.output, device=device)
    return hidden_states


def bert_layer(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    attention_output = bert_attention(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.attention,
        device=device,
    )
    feedforward_output = bert_feedforward(
        config,
        attention_output,
        parameters=parameters,
        device=device,
    )

    return feedforward_output


def bert_encoder(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = bert_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
            device=device,
        )
        encoder_input = encoder_output
    return encoder_output


def bert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device=None,
    *,
    parameters,
):
    word_embeddings = ttnn.embedding(input_ids, parameters.embeddings.word_embeddings.weight)
    token_type_embeddings = ttnn.embedding(token_type_ids, parameters.embeddings.token_type_embeddings.weight)
    position_embeddings = ttnn.embedding(position_ids, parameters.embeddings.position_embeddings.weight)
    word_embeddings = ttnn.to_layout(word_embeddings, ttnn.TILE_LAYOUT)
    token_type_embeddings = ttnn.to_layout(token_type_embeddings, ttnn.TILE_LAYOUT)
    position_embeddings = ttnn.to_layout(position_embeddings, ttnn.TILE_LAYOUT)

    embeddings = word_embeddings + token_type_embeddings + position_embeddings ## overload __add__ in tensor.py doesn't account for instr, perf stats. Need to fix it.
    hidden_states = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=None, #ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = bert_encoder(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.encoder,
        device=device,
    )

    return hidden_states


def bert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device=None,
    *,
    parameters,
    name="bert",
):
    bert_output = bert(
        config,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        device=device,
        parameters=parameters#[name],
    )

    qa_outputs = bert_output
    qa_outputs = ttnn.linear(
        qa_outputs,
        parameters.qa_outputs.weight.T,
        bias=parameters.qa_outputs.bias,
        core_grid=None, #device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return qa_outputs


# Parameters structure for BERT model
class BertParameters:
    def __init__(self):
        # Embedding parameters
        self.embeddings = EmbeddingParameters()
        
        # Encoder parameters (contains multiple layers)
        self.encoder = EncoderParameters()
        
        # QA output layer parameters
        self.qa_outputs = QAOutputParameters()

class EmbeddingParameters:
    def __init__(self):
        self.word_embeddings = WordEmbeddingParameters()
        self.token_type_embeddings = TokenTypeEmbeddingParameters()
        self.position_embeddings = PositionEmbeddingParameters()
        self.LayerNorm = LayerNormParameters()

class WordEmbeddingParameters:
    def __init__(self):
        self.weight = None  # Shape: [vocab_size, hidden_size]

class TokenTypeEmbeddingParameters:
    def __init__(self):
        self.weight = None  # Shape: [type_vocab_size, hidden_size]

class PositionEmbeddingParameters:
    def __init__(self):
        self.weight = None  # Shape: [max_position_embeddings, hidden_size]

class LayerNormParameters:
    def __init__(self):
        self.weight = None  # Shape: [hidden_size]
        self.bias = None    # Shape: [hidden_size]

class EncoderParameters:
    def __init__(self, num_layers=2):  # bert-tiny has 2 layers
        self.layer = [BertLayerParameters() for _ in range(num_layers)]

class BertLayerParameters:
    def __init__(self):
        self.attention = AttentionParameters()
        self.intermediate = IntermediateParameters()
        self.output = OutputParameters()

class AttentionParameters:
    def __init__(self):
        self.self = SelfAttentionParameters()
        self.output = AttentionOutputParameters()

class SelfAttentionParameters:
    def __init__(self):
        self.query = LinearParameters()  # Query projection
        self.key = LinearParameters()    # Key projection
        self.value = LinearParameters()  # Value projection

class AttentionOutputParameters:
    def __init__(self):
        self.dense = LinearParameters()      # Output projection
        self.LayerNorm = LayerNormParameters()

class IntermediateParameters:
    def __init__(self):
        self.dense = LinearParameters()  # FFN first layer

class OutputParameters:
    def __init__(self):
        self.dense = LinearParameters()      # FFN second layer
        self.LayerNorm = LayerNormParameters()

class LinearParameters:
    def __init__(self):
        self.weight = None  # Shape: [out_features, in_features]
        self.bias = None    # Shape: [out_features]

class QAOutputParameters:
    def __init__(self):
        self.weight = None  # Shape: [2, hidden_size] for start/end positions
        self.bias = None    # Shape: [2]


def load_bert_parameters_from_huggingface(model):
    """Load parameters from a HuggingFace BERT model into our parameter structure."""
    
    parameters = BertParameters()
    
    # Load embeddings
    parameters.embeddings.word_embeddings.weight = model.bert.embeddings.word_embeddings.weight
    parameters.embeddings.token_type_embeddings.weight = model.bert.embeddings.token_type_embeddings.weight
    parameters.embeddings.position_embeddings.weight = model.bert.embeddings.position_embeddings.weight
    parameters.embeddings.LayerNorm.weight = model.bert.embeddings.LayerNorm.weight
    parameters.embeddings.LayerNorm.bias = model.bert.embeddings.LayerNorm.bias
    
    # Load encoder layers
    for i, layer in enumerate(model.bert.encoder.layer):
        # Attention parameters
        parameters.encoder.layer[i].attention.self.query.weight = layer.attention.self.query.weight
        parameters.encoder.layer[i].attention.self.query.bias = layer.attention.self.query.bias
        parameters.encoder.layer[i].attention.self.key.weight = layer.attention.self.key.weight
        parameters.encoder.layer[i].attention.self.key.bias = layer.attention.self.key.bias
        parameters.encoder.layer[i].attention.self.value.weight = layer.attention.self.value.weight
        parameters.encoder.layer[i].attention.self.value.bias = layer.attention.self.value.bias
        
        # Attention output
        parameters.encoder.layer[i].attention.output.dense.weight = layer.attention.output.dense.weight
        parameters.encoder.layer[i].attention.output.dense.bias = layer.attention.output.dense.bias
        parameters.encoder.layer[i].attention.output.LayerNorm.weight = layer.attention.output.LayerNorm.weight
        parameters.encoder.layer[i].attention.output.LayerNorm.bias = layer.attention.output.LayerNorm.bias
        
        # Feedforward parameters
        parameters.encoder.layer[i].intermediate.dense.weight = layer.intermediate.dense.weight.T
        parameters.encoder.layer[i].intermediate.dense.bias = layer.intermediate.dense.bias
        parameters.encoder.layer[i].output.dense.weight = layer.output.dense.weight
        parameters.encoder.layer[i].output.dense.bias = layer.output.dense.bias
        parameters.encoder.layer[i].output.LayerNorm.weight = layer.output.LayerNorm.weight
        parameters.encoder.layer[i].output.LayerNorm.bias = layer.output.LayerNorm.bias
    
    # QA outputs
    parameters.qa_outputs.weight = model.qa_outputs.weight
    parameters.qa_outputs.bias = model.qa_outputs.bias
    
    return parameters


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("model_name", ["mrm8488/bert-tiny-finetuned-squadv2"])
def test_perf_bert_tiny(
    device = None,
    batch_size = 8,
    sequence_size = 128,
    model_name = "mrm8488/bert-tiny-finetuned-squadv2",
):
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    config = hugging_face_reference_model.config
    parameters = load_bert_parameters_from_huggingface(hugging_face_reference_model)
    #model_name = str(model_location_generator(model_name, model_subdir="Bert"))
    torch_bert_input = torch.randint(0, 100, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size)

    ttnn_bert_inputs = ttnn.from_torch(torch_bert_input, dtype=ttnn.uint32, device=device)
    ttnn_token_type_ids = ttnn.from_torch(torch_token_type_ids, dtype=ttnn.uint32, device=device)
    ttnn_position_ids = ttnn.from_torch(torch_position_ids, dtype=ttnn.uint32, device=device)
    ttnn_attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat16, device=device)

    for i in range(1):
        ttnn_output = bert_for_question_answering(
            config,
            input_ids=ttnn_bert_inputs,
            token_type_ids=ttnn_token_type_ids,
            position_ids=ttnn_position_ids,
            attention_mask=ttnn_attention_mask,
            parameters=parameters,
            device=device,
        )
        output = ttnn.from_device(ttnn_output)


if __name__ == "__main__":
    test_perf_bert_tiny(device=None, batch_size=8, sequence_size=128,
                        model_name="mrm8488/bert-tiny-finetuned-squadv2")
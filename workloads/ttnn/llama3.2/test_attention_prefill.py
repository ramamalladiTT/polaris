#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from attention import Attention
from rope import get_prefill_rot_mat, get_rot_transformation_mat
from model_config import ModelArgs

def test_attention_inference():
    mesh_device = ttnn.open_device(device_id=0)#, device_type="ttnn", device_name=None)
    max_seq_len = 256
    paged_attention = False
    page_params = [{"page_block_size": 32, "page_max_num_blocks": 1024}]
    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1
    model_args = ModelArgs(mesh_device, batch_size, max_seq_len)

    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        mesh_device,
        max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)

    transformation_mat_torch = ttnn._rand(transformation_mat_torch.shape, device=mesh_device, dtype=dtype)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=None, #ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    # Setup page table
    page_table_tt = None
    paged_attention_config = None

    if paged_attention: # set to false for simulation
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    state_dict = None

    tt_model = Attention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
    )
    
    attention_input = ttnn._rand((1, 1, 256, 3072), device=mesh_device, dtype=ttnn.bfloat16)

    tt_out = tt_model(
        attention_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )
    print(f"TT Output Shape: {tt_out.shape}, dtype: {tt_out.dtype}")
    tt_out = ttnn.to_torch(
        tt_out,# mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape)
    )

    if (tt_out.shape == [1, 1, 256, 3072]):
        print(f"TT Output shape matches expected: {tt_out.shape} == [1, 1, 256, 3072]")
    else:
        print(f"TT Output shape mismatch: {tt_out.shape} != [1, 1, 256, 3072]")

    check_kv_cache = False  # set to false for simulation
    if check_kv_cache:
        # PyTorch output --------------------------------------------------------------------
        pytorch_layer_present = [
            reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
            reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
        ]
        # TT hardware execution -------------------------------------------------------------
        if paged_attention:
            tt_layer_present = [
                (
                    ttnn.to_torch(
                        cache,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device,
                            dims=(1, 3) if model_args.is_galaxy else (0, 1),
                            mesh_shape=model_args.cluster_shape,
                        ),
                    )[reverse_permutation][:, : model_args.n_kv_heads, :, : model_args.head_dim]
                    .reshape(
                        model_args.max_batch_size,
                        paged_attention_config.max_num_blocks // model_args.max_batch_size,
                        model_args.n_kv_heads,
                        paged_attention_config.block_size,
                        model_args.head_dim,
                    )
                    .transpose(1, 2)
                    .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
                        :batch_size, ...
                    ]
                )
                for cache in tt_model.layer_past
            ]
        else:
            tt_layer_present = [
                ttnn.to_torch(
                    cache,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device,
                        dims=(1, 0) if model_args.is_galaxy else (0, 1),
                        mesh_shape=model_args.cluster_shape,
                    ),
                )[:batch_size, :, :, :]
                for cache in tt_model.layer_past
            ]

        for i, (cache_pt, cache_tt) in enumerate(zip(pytorch_layer_present, tt_layer_present)):
            cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + generation_length + 1)
            cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
            cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
            does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
            if i == 0:
                logger.info(f"K cache output: {output_pcc}")
            else:
                logger.info(f"V cache output: {output_pcc}")

            if does_pass:
                logger.info(f"KV Cache Passed!")
            else:
                logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
                all_tests_pass = False

    print("\nGenerating computation graph...")
    g = mesh_device.get_graph()
    g.graph2onnx('test_attn_prefill.onnx', do_model_check=False)

if __name__ == "__main__":
    test_attention_inference()
    print("\n Attention test completed!")

from sched import scheduler
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
import numpy as np

from PIL import Image
# import torch
from datetime import datetime
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from src.diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler

class SDModel(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name           = name
        self.bs             = cfg.get('bs', 1)
        self.vae = AutoencoderKL(  # random weights
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(128, 256, 512, 512),
            latent_channels=4,
            norm_num_groups=32,
            sample_size=32,
        )

        self.unet = UNet2DConditionModel(  # random weights
            objname="unet",
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=512,
            attention_head_dim=8,
        )

        self.scheduler = UniPCMultistepScheduler(
            'unipcmultisched',
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            trained_betas=None,
            prediction_type="epsilon",
            use_karras_sigmas=False,
            timestep_spacing="linspace",
            steps_offset=1,
            rescale_betas_zero_snr=False,
        )

        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
                'txt_embeds': F._from_shape('txt_embeds',  [2, 77, 512], np_dtype=np.int64),
                'latents': F._from_shape('latents',  [1, 4, 64, 64], np_dtype=np.int64)
                }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):
        text_embeddings = self.input_tensors['txt_embeds']
        latents = self.input_tensors['latents']
        i = 0
        for t in tqdm(self.scheduler.timesteps):
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            if i == 0:
                break
        print(f'ttsim unet input - latents is of shape {latents.shape}')
        print(f'ttsim unet output - noise_pred shape is {noise_pred.shape}')
        image = self.vae.decode(noise_pred).sample
        return image

def run_standalone() -> None:
    sd_model = SDModel('StableDiffusion', {'bs': 1})
    sd_model.create_input_tensors()
    image = sd_model()
    if (image.shape == [1, 3, 512, 512]):
        print(f'ttsim SD output image shape is as expected: {image.shape}')
    else:
        print(f'Error: ttsim SD output image shape is NOT as expected: {image.shape}. Expected shape is [1, 3, 512, 512]')

    # print('Generating model graph...')
    # gg = sd_model.get_forward_graph()
    # onnx_ofilename = f'test_sd.onnx'
    # gg.graph2onnx(onnx_ofilename)

if __name__ == "__main__":
    run_standalone()

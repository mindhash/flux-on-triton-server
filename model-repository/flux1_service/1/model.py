import os
import json
import asyncio
from pathlib import Path
from functools import partial
import triton_python_backend_utils as pb_utils

import torch
import numpy as np
from diffusers import FluxPipeline, FluxTransformer2DModel


ROOT = Path(__file__).parent

class TritonPythonModel: 
    MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"
    
    def initialize(self,args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "output_image")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
 

        self.pipe = FluxPipeline.from_pretrained(
                    self.MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                )
        self.pipe.to("cuda")

    def execute(self,requests):
        """Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        logger = pb_utils.Logger

        prompts = [pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0][0].decode('utf-8') for request in requests]
        print(f"prompts: {prompts}")
        print(f"request length: {len(requests)}")
        
        images = self.pipe(
            prompt=prompts,
            guidance_scale      = 3.5,
            output_type         = "pil",
            num_inference_steps = 20,
            max_sequence_length = 512,
            # num_images_per_prompt=1,
            generator           = torch.Generator("cuda").manual_seed(0)
        ).images

        # Convert final images to numpy array
        final_images_array = np.array([np.array(image) for image in images])
    
        for n, request in enumerate(requests):
            # Create final output tensor
            final_images_array = np.array(images[n])
            final_output_tensor = pb_utils.Tensor("output_image", final_images_array)

            # Create final inference response
            final_response = pb_utils.InferenceResponse(
                output_tensors=[final_output_tensor])

            responses.append(final_response)
        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

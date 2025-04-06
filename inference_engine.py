import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class InferenceEngine:
    def __init__(self, config):
        self.config = config
        model_path = config.model_path
        
        # Get number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        print(f"Using {self.num_gpus} GPUs")
        if False: #self.num_gpus == 2:
            device_map = {'visual': 0, 'model.embed_tokens': 1, 'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.norm': 1, 'model.rotary_emb': 1, 'lm_head': 1}
        else:
            device_map = 'auto'
        
        # Load model on first GPU
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map  # let HF parallelize across GPUs
        )
        print(self.model.hf_device_map)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    def run(self, message, num_inference_attempts=1):
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        print("num_inference_attempts", num_inference_attempts)
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            num_return_sequences=num_inference_attempts
            )
        print("generated_ids", generated_ids)
        print("Generated IDs shape:", generated_ids.shape)
        
        # Check if generation was successful
        if generated_ids is None or len(generated_ids) == 0:
            print("Warning: Null generation")
            return ["Error: Generation failed"]
            
        prompt_ids = inputs.input_ids[0]
        generated_ids_trimmed = []
        for i, out_ids in enumerate(generated_ids):
            trimmed = out_ids[len(prompt_ids) :]
            generated_ids_trimmed.append(trimmed)

        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        print("generated_ids_trimmed", generated_ids_trimmed)
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("output_text", output_text)
        # Check if output is valid
        if output_text and any(text.strip() for text in output_text):
            return output_text
        else:
            print("Warning: Empty output")
            return ["Error: No valid output generated"]

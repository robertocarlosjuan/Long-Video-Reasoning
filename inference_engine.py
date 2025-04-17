import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

class InferenceEngine:
    def __init__(self, config):
        self.config = config
        model_path = config.model_path
        
        # Configure 4-bit quantization for both model and inputs
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Get number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        print(f"Using {self.num_gpus} GPUs")
        
        # Load model with 4-bit quantization
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        
        print(self.model.hf_device_map)
        
        # Initialize processor with 4-bit quantization
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
        )

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

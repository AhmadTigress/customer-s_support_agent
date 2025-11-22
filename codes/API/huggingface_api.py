

# Import the required modules
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Get the Hugging Face API key from the environment variables
HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY')
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

def huggingface_completion(prompt: str) -> dict:
    '''
    Call Hugging Face API for text completion
    Parameters:
        - prompt: user query (str)
    Returns:
        - dict
    '''
    # BUG FIX: Validate input
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        print("Hugging Face API call failed: Empty or invalid prompt")
        return {'status': 0, 'response': ''}
    
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        
        # BUG FIX: Set pad token to prevent crashes
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,  # BUG FIX: Use float16 to save memory
            token=HF_TOKEN,
        )
        
        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id  # BUG FIX: Prevent padding issues
        )
        
        # Generate response
        response = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,  # BUG FIX: Don't include prompt in output
        )
        
        # Extract the generated text
        output_text = response[0]["generated_text"].strip()

        # Print a success message with the response from the Hugging Face API call
        print(f"Hugging Face API call successful. Response: {output_text[:100]}...")  

        # BUG FIX: Clean up memory
        del model, tokenizer, pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Return a dictionary with the status and the content of the response
        return {
            'status': 1,
            'response': output_text
        }
        
    except torch.cuda.OutOfMemoryError:
        # BUG FIX: Handle GPU memory errors
        print("Hugging Face API call failed: CUDA out of memory")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {'status': 0, 'response': ''}
        
    except Exception as e:
        # Print any error that occurs during the Hugging Face API call
        print(f"Hugging Face API call failed. Error: {e}")  

        # BUG FIX: Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Return a dictionary with the status and an empty response
        return {'status': 0, 'response': ''}
# =================== MATRXI API ============================================

# Import the required modules
import os
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
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
            token=HF_TOKEN,
        )
        
        # Create a pipeline for text generation
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        # Generate response
        response = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        
        # Extract the generated text
        output_text = response[0]["generated_text"]
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        # Print a success message with the response from the Hugging Face API call
        print(f"Hugging Face API call successful. Response: {output_text[:100]}...")  

        # Return a dictionary with the status and the content of the response
        return {
            'status': 1,
            'response': output_text
        }
    except Exception as e:
        # Print any error that occurs during the Hugging Face API call
        print(f"Hugging Face API call failed. Error: {e}")  

        # Return a dictionary with the status and an empty response
        return {'status': 0, 'response': ''}
# codes/API/huggingface_api.py
import logging
from codes.initialize import model_pipeline  # Import the pre-loaded pipeline

logger = logging.getLogger(__name__)

def huggingface_completion(prompt: str) -> dict:
    '''
    Call the pre-loaded Hugging Face pipeline for text completion.
    '''
    # Validate input
    if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        logger.error("Hugging Face call failed: Empty or invalid prompt")
        return {'status': 0, 'response': ''}

    try:
        # Use the global pipeline from initialize.py
        # This is now nearly instantaneous because the model is already in RAM/VRAM
        response = model_pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )

        output_text = response[0]["generated_text"].strip()

        logger.info(f"Hugging Face completion successful. Response length: {len(output_text)}")

        return {
            'status': 1,
            'response': output_text
        }

    except Exception as e:
        logger.error(f"Inference failed. Error: {e}")
        return {'status': 0, 'response': ''}

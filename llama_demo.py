import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_llama_response(prompt, model_name, max_tokens=100):
    """
    Generate a response using a PyTorch Llama model with a custom autoregressive loop.
    
    :param prompt: The input prompt to send to the model
    :param model_name: Name or path of the Llama model
    :param max_tokens: Maximum number of tokens to generate (default: 100)
    :return: Generated response as a string
    """
    # Initialize the tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)

    # Move model to GPU if available
    model.to(DEVICE)

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    # Set up for generation
    generated = input_ids
    attention_mask = torch.ones(generated.shape, dtype=torch.long, device=DEVICE)
    past_key_values = None

    # Autoregressive generation loop
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # Get the next token logits
        next_token_logits = outputs.logits[:, -1, :]
        
        # Sample the next token (you can implement different sampling strategies here)
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # Append the new token to the generated sequence
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        
        # Update past_key_values for efficiency
        past_key_values = outputs.past_key_values

        # Check if we've generated an end-of-sequence token
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode and return the generated text
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    return response[len(prompt):]  # Return only the newly generated text

# Example usage
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"  # or path to your local model
    user_prompt = "Tell me a short joke about programming."
    
    response = generate_llama_response(user_prompt, model_name)
    print(f"Prompt: {user_prompt}")
    print(f"Response: {response}")


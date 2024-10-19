import os
import ctypes
import torch
import llama_cpp
import llama_cpp.llava_cpp as llava_cpp
from PIL import Image
import torchvision.transforms as transforms

# Convert an image to a PyTorch tensor
def image_to_tensor(image_path):
    try:
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to fit the model
            transforms.ToTensor()          # Convert image to tensor
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Evaluate tokens in batches
def eval_tokens(ctx_llama, tokens, n_batch, n_past):
    N = len(tokens)
    for i in range(0, N, n_batch):
        n_eval = min(N - i, n_batch)
        c_tokens = (ctypes.c_int * n_eval)(*tokens[i:i+n_eval])  # Create ctypes array
        batch = llama_cpp.llama_batch_get_one(c_tokens, n_eval, n_past, 0)
        
        if llama_cpp.llama_decode(ctx_llama, batch):  # Perform inference
            print(f"Error at token index {i}")
            return False
        
        n_past.value += n_eval
    return True

# Load and convert an image to embeddings
def prepare_embedding(image_path):
    tensor = image_to_tensor(image_path)
    if tensor is None:
        return None

    embd_image = tensor.squeeze().cpu().tolist()
    c_float_array = (ctypes.c_float * len(embd_image))(*embd_image)
    return llava_cpp.llava_image_embed(embed=c_float_array, n_image_pos=1)

# Initialize the LLaMA model
MODEL_PATH = os.getenv("MODEL", r"path_to_your_model")
llm = llama_cpp.Llama(
    model_path=MODEL_PATH,
    n_ctx=3768,
    n_batch=1024,
    n_ubatch=1024
)

# Run inference with the model
def evaluate_with_model(llm, prompt, image_path):
    n_past = ctypes.c_int(llm.n_tokens)
    n_past_p = ctypes.pointer(n_past)

    # Tokenize prompt
    tokens = llm.tokenize(prompt, add_bos=True, add_eos=True)

    # Evaluate prompt tokens
    if not eval_tokens(ctx_llama=llm.ctx, tokens=tokens, n_batch=llm.n_batch, n_past=n_past):
        print("Error evaluating prompt tokens.")
        return

    # Prepare image embedding
    embed = prepare_embedding(image_path)
    if embed is None:
        print("Error preparing embedding.")
        return

    # Evaluate image embedding
    llava_cpp.llava_eval_image_embed(llm.ctx, embed, llm.n_batch, n_past_p)

    # Generate model response
    output_tokens = llm.tokenize("OCR: <|im_end|><|im_start|>assistant", add_bos=False, add_eos=True)
    if not eval_tokens(ctx_llama=llm.ctx, tokens=output_tokens, n_batch=llm.n_batch, n_past=n_past):
        print("Error evaluating generated tokens.")
        return

    # Print generated response
    for token in llm.generate(output_tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.0):
        print(llm.detokenize([token]), end='')

    llm.close()

# Usage example
if __name__ == "__main__":
    prompt = """<|im_start|>system
    You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user"""
    image_path = "path/to/image.png"  # Replace with your image path
    evaluate_with_model(llm, promp

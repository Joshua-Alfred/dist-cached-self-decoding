from time import time
import argparse
from transformers import AutoTokenizer
from custom_modeling_llama import LlamaForCausalLM
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="codellama/CodeLlama-7b-hf")
args = parser.parse_args()

model = args.model

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token  # Set the pad_token to the eos_token

model = LlamaForCausalLM.from_pretrained(
    model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

prompts = ["public static void main(String[] args) {\n    //Fizzbuzz",
        "def fibonacci(n):\n    a, b = 0, 1",
        "SELECT * FROM users WHERE age > 18;",
        "class Car:\n    def __init__(self, name):",
        "#include <stdio.h>\nint main() {\n    printf(\"Hello World\");"]

total_time = 0
n = len(prompts)

for prompt in prompts:

    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding=True)
    inputs = inputs.to(model.device)

    start = time()

    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
        pad_token_id=tokenizer.eos_token_id,
        draft_skip_layers=[i for i in range(21, 29)],
        # draft_skip_layers=[],
        num_candidate_tokens=5,
        use_cache=True,
        return_dict_in_generate=True,
    )

    end = time()

    num_tokens = len(output["sequences"][0])
    time_per_token = (end - start)/num_tokens
    total_time += time_per_token

    generated_text = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Prompt: {prompt}")
    print("-" * 50)
    print(generated_text)
    print(f"Time per token: {(end - start)/num_tokens:.4f}s")

print(f"Average time taken for self-speculative decoding: {total_time/n:.4f}s")
# # Access the debugging information
# draft_tokens = output.draft_tokens
# accepted_tokens = output.accepted_tokens
# rejected_tokens = output.rejected_tokens

# print("\nDebugging Information:")
# print(f"Number of generation steps: {len(draft_tokens)}")
# print(f"Total draft tokens: {sum(len(tokens) for tokens in draft_tokens)}")
# print(f"Total accepted tokens: {sum(len(tokens) for tokens in accepted_tokens)}")
# print(f"Total rejected tokens: {sum(len(tokens) for tokens in rejected_tokens)}")


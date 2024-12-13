from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

# Create a repository on the Hugging Face Hub
repo_id = "hefajuddin/Rapunzel_Story_Gen_Llma"  # Format: <username>/<repo_name>
api.create_repo(repo_id=repo_id, token="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")

# Load the model and tokenizer (optional if already in memory)
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_llama")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_llama")

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_id, token="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")
tokenizer.push_to_hub(repo_id, token="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")

print(f"Model and tokenizer successfully pushed to: https://huggingface.co/{repo_id}")
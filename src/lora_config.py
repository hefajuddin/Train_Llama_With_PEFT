from peft import LoraConfig, get_peft_model
from pretrained_model import pretrained_model
# Configure and apply LoRA
config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(pretrained_model, config)
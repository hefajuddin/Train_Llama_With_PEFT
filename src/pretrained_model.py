from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pretrained_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
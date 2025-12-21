import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

base_model_name = "Unbabel/TowerInstruct-Mistral-7B-v0.2"
lora_path = "moore3930/tower-calibrated"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# model = base_model
model = PeftModel.from_pretrained(base_model, lora_path)

'''
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)
'''

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    num_beams=5,          # beam search，5 个 beam
    do_sample=False,      # 必须是 False（beam search 是确定性的）
    early_stopping=True,
    eos_token_id=terminators,
)

source = "“May I come in?” Ivory asked, remembering the time they went in without asking."

prompt = (
    f"Translate the following text from {{src_lang}} into {{tgt_lang}}.\n{{src_lang}}: {{src}}\n{{tgt_lang}}:"
)

messages = [
    {"role": "user", "content": prompt.format(src_lang="English", tgt_lang="Chinese", src=source)},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
print(outputs[0]["generated_text"])


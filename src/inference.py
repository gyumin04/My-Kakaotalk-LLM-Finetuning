from unsloth import FastLanguageModel
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

token = os.environ.get("HF_TOKEN")

login(token)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

model = FastLanguageModel.for_inference(model)
hf_repo = "gyumin040911/llama-3-my-kakaotalk-lora"

model.load_adapter(model_id=hf_repo, adapter_name="default")
model.set_adapter("default")

messages = []

while True:
    message = input("user: ")
    if message.strip() == "exit":
        break
        
    messages.append({"role": "user", "content": message})
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")

    input_length = inputs.shape[1]
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=64,
        temperature=0.53,
        top_p=0.85,
        top_k=50,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_tokens = outputs[0][input_length:]
    
    ai_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    print(f'assistant : {ai_response}')
    messages.append({"role": "assistant", "content": ai_response})
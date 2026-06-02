import json
import re

def clean_and_merge_session(messages):
    merged_messages = []
    
    for msg in messages:
        role = msg['role']
        content = msg['content'].strip()
        
        if "보이스톡" in content or "동영상" in content:
            continue
            
        if merged_messages and merged_messages[-1]['role'] == role:
            merged_messages[-1]['content'] += " " + content
        else:
            merged_messages.append({'role': role, 'content': content})
            
    final_messages = []
    for msg in merged_messages:
        content = msg['content']
        
        only_chosung = re.sub(r'[ㄱ-ㅎㅏ-ㅣ\s?.]', '', content)
        
        if len(only_chosung) == 0 and len(content) <= 3:
            continue
            
        final_messages.append(msg)
        
    while final_messages and final_messages[0]['role'] != 'user':
        final_messages.pop(0)
    while final_messages and final_messages[-1]['role'] != 'assistant':
        final_messages.pop()
        
    if len(final_messages) >= 2:
        return final_messages
    return None

input_file = "data/train_data.jsonl"
output_file = "data/train_data_final.jsonl"

cleaned_data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        
        if "messages" in data:
            refined_messages = clean_and_merge_session(data["messages"])
            if refined_messages:
                cleaned_data.append({"messages": refined_messages})

with open(output_file, "w", encoding="utf-8") as f:
    for item in cleaned_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
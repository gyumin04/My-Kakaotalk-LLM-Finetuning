import pandas as pd
import json

data = pd.read_csv("data/chat_history.csv")

history = data['history'].tolist()
time_diff = data['time_diff'].tolist()
speaker = data['speaker'].tolist()

multi_turn = []
messages = []
message = {"role": None, "content" : None}
role = ""

for i in range(len(history)):
    message = {"role": None, "content" : None}
    if len(history) > i+1:
        if time_diff[i+1] <= 240:
            if role == speaker[i]:
                print(speaker[i], history[i])
                break
            role = speaker[i]
            message['role'] = speaker[i]
            message['content'] = history[i]
            if len(messages) == 0 and speaker[i] == "user":
                messages.append(message)
            elif len(messages) != 0:
                messages.append(message)
        else:
            if speaker[i] == "assistant":
                message['role'] = speaker[i]
                message['content'] = history[i]
                messages.append(message)
            if len(messages) >= 4:
                multi_turn.append({"messages" : messages})
            role = ""
            messages = []

train_data = multi_turn



with open('data/train_data.jsonl', 'w', encoding='utf-8') as f:
    for js in train_data:
        json.dump(js, f, ensure_ascii=False)
        f.write('\n')

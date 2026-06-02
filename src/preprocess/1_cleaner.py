import pandas as pd
import re
import os
from dotenv import load_dotenv
import ast

load_dotenv()

data = pd.read_csv("data/conversation_history.csv")

data['timestamp'] = pd.to_datetime(
    data['time'].astype(str), format='%Y%m%d%H%M'
)

data['time_diff'] = data['timestamp'].diff()
diff = data['timestamp'].diff()
data['diff_minutes'] = data['time_diff'].dt.total_seconds() / 60

speaker_list = data['role'].tolist()
diff_minutes_list = data['diff_minutes'].tolist()
history_list = data['text'].tolist()
timestamp_list = data['timestamp'].tolist()


for i in range(len(speaker_list)):
    if len(speaker_list) > i+1:
        if speaker_list[i] == speaker_list[i+1]:
            if diff_minutes_list[i+1] <= 240:
                history_list[i+1] = str(history_list[i]) + " " + str(history_list[i+1])
                history_list[i] = None
                diff_minutes_list[i+1] = diff_minutes_list[i]

name_list = os.getenv('name_list')
id_list = os.getenv('id_list')
name_list = ast.literal_eval(name_list)
id_list = ast.literal_eval(id_list)

PHONE = r'010[-\s]?\d{3,4}[-\s]?\d{4}'
EMAIL = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
ACCOUNT = r'\d{3,6}[-\s]?\d{2,6}[-\s]?\d{2,6}'
URL = r'https?://[\w\-\.]+(?:/[\w\-\./?%&=]*)?'

chat_history = {
   'speaker' : speaker_list,
   'history' : history_list,
   'timestamp' : timestamp_list,
   'time_diff' : diff_minutes_list
}

chat_history = pd.DataFrame(chat_history)

chat_history = chat_history.dropna(subset=['history'])
chat_history = chat_history.reset_index(drop=True)

history_list = chat_history['history']

for i in range(len(history_list)):
    if history_list[i] != None:
        history_list[i] = re.sub(PHONE, "[PHONE]", history_list[i])
        history_list[i] = re.sub(EMAIL, "[EMAIL]", history_list[i])
        history_list[i] = re.sub(URL, "[URL]", history_list[i])
        history_list[i] = re.sub(ACCOUNT, "[ACCOUNT]", history_list[i])
        for name in name_list:
            if name in history_list[i]:
                history_list[i] = history_list[i].replace(name, '[NAME]')
        for id in id_list:
            if id in history_list[i]:
                history_list[i] = history_list[i].replace(id, '[ACCOUNT]')

chat_history['history'] = history_list

chat_history.to_csv('data/chat_history.csv')
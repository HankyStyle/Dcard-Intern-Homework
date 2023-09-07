# %% [markdown]
# ## Using BART for Like Prediction with Text Only Input

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Read File

# %%
import pandas as pd
import numpy as np
train = pd.read_csv('../../raw_data/intern_homework_train_dataset.csv')
test = pd.read_csv('../../raw_data/intern_homework_public_test_dataset.csv')
train.head()

# %%
from sklearn.model_selection import train_test_split

train, valid = train_test_split(train, random_state=777, train_size=0.9)
len(train), len(valid)

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# 挑選 Label

# %%
train_label = train['like_count_24h']
valid_label = valid['like_count_24h']
test_label = test['like_count_24h']
train_label

# %%
train_label = train_label.tolist()
valid_label = valid_label.tolist()
test_label = test_label.tolist()
train_label[0]

# %%
train_label = list(map(str, train_label))
valid_label = list(map(str, valid_label))
test_label = list(map(str, test_label))
train_label[0]

# %% [markdown]
# 因為 BART 為自然語言模型，label 也需要為文字的格式

# %% [markdown]
# 挑選 Input

# %% [markdown]
# 省略掉 作者 ID  因為我認為 BART 沒辦法了解這些 Feature, 會讓模型預測文字輸出帶來 noise.

# %%
# 指定要刪除的 column names，並使用 drop 函數將這些 column 刪除
drop_columns = ['author_id', 'like_count_24h']

train_input = train.drop(drop_columns, axis=1)
valid_input = valid.drop(drop_columns, axis=1)
test_input = test.drop(drop_columns, axis=1)

# %% [markdown]
# ### Data Transfromations

# %% [markdown]
# 處理 created_by Feature

# %%
# 將文章發佈時間 拆成 星期幾 與 小時 的函數
def split_date(df, date_column):

    # 將 created_by 欄位轉換成日期格式
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    
    # 新增 星期幾 和 小時 欄位
    df['weekday'] = df[date_column].dt.weekday
    df['hour'] = df[date_column].dt.hour

    # 移除 created_by 欄位
    df = df.drop(date_column, axis=1)

    # 回傳處理過的資料集
    return df

# %%
train_input = split_date(train_input, 'created_at')
valid_input = split_date(valid_input, 'created_at')
test_input = split_date(test_input, 'created_at')

# 顯示處理過的資料集
train_input.head()

# %%
train_input.columns

# %% [markdown]
# 自訂義 dict，將 weekday 轉換為中文星期幾

# %%
weekday_dict = {
    0: '星期一',
    1: '星期二',
    2: '星期三',
    3: '星期四',
    4: '星期五',
    5: '星期六',
    6: '星期日'
}

# 將 weekday 轉換為中文星期幾
train_input['weekday'] = train_input['weekday'].map(weekday_dict)
valid_input['weekday'] = valid_input['weekday'].map(weekday_dict)
test_input['weekday'] = test_input['weekday'].map(weekday_dict)

# %% [markdown]
# 將 Input 的數值與文字合併一篇文章來當作 BART Input

# %%
# 將 dataframe 內的 feature 合併成一串文字 的函數
def transfrom_to_text(df):

    passage = ""

    # combined title and post time
    passage += "這篇文章標題是 {} ，在{}{}點發佈。".format(df["title"], df["weekday"], df["hour"])

    # combined forum_id and forum_stats
    passage += "這篇發佈的看板 ID 為 {}， 看板資訊為 {} 。".format(df["forum_id"], df["forum_stats"])

    # combined likes_count
    passage += "文章在發佈後的第一小時累積到的愛心數有 {}，在第二小時累積到的愛心數有 {}，".format(df["like_count_1h"], df["like_count_2h"])
    passage += "第三小時累積到的愛心數有 {}，在第四小時累積到的愛心數有 {}，".format(df["like_count_3h"], df["like_count_4h"])
    passage += "第五小時累積到的愛心數有 {}，在第六小時累積到的愛心數有 {}。".format(df["like_count_5h"], df["like_count_6h"])

    # combined comment_count
    passage += "文章在發佈後的第一小時累積到的留言數有 {}，在第二小時累積到的留言數有 {}，".format(df["comment_count_1h"], df["comment_count_2h"])
    passage += "第三小時累積到的留言數有 {}，在第四小時累積到的留言數有 {}，".format(df["comment_count_3h"], df["comment_count_4h"])
    passage += "第五小時累積到的留言數有 {}，在第六小時累積到的留言數有 {}。".format(df["comment_count_5h"], df["comment_count_6h"])

    
    return passage

# %%
train_text = train_input.apply(transfrom_to_text, axis=1).tolist()
valid_text = valid_input.apply(transfrom_to_text, axis=1).tolist()
test_text = test_input.apply(transfrom_to_text, axis=1).tolist()
train_text[0]

# %%
print("Train Data 總共有 {} 筆".format(len(train_text)))
print("Valid Data 總共有 {} 筆".format(len(valid_text)))
print("Test Data 總共有 {} 筆".format(len(test_text)))

# %% [markdown]
# ### Tokenization

# %%
from transformers import BertTokenizer
import torch

# %%
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

# %%
train_encodings = tokenizer(train_text ,text_target = train_label, truncation=True, padding=True)
valid_encodings = tokenizer(valid_text ,text_target = valid_label, truncation=True, padding=True)
test_encodings = tokenizer(test_text ,text_target = test_label, truncation=True, padding=True)

# %%
train_encodings.keys()

# %% [markdown]
# ## Fine-tuning

# %%
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-base-chinese")

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = Dataset(train_encodings)
valid_dataset = Dataset(valid_encodings)
test_dataset = Dataset(test_encodings)

# %%
batch_size = 16
args = Seq2SeqTrainingArguments(
    output_dir = "./results",
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="MAPE",
    greater_is_better = False,
    weight_decay=0.01,
    predict_with_generate=True,
    eval_accumulation_steps = 1,
)

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# %%
import numpy as np
import re
from sklearn.metrics import mean_absolute_percentage_error
def compute_metrics(p):
    predictions, labels = p
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # store all sentences
    predicted = []
    true_label = []
    for k in range(len(decoded_labels)):

        match = re.search(r'\d+', decoded_preds[k])
        if match:
            pred = int(match.group())
        label = int(decoded_labels[k])

        predicted.append(pred)
        true_label.append(label)

    # evaluation metrics
    MAPE = mean_absolute_percentage_error(true_label, predicted)
    
    result = {'MAPE': MAPE}
    
    return result

# %%
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# %%
trainer.train()

# %% [markdown]
# ## Evaluation

# %% [markdown]
# Save Model

# %%
trainer.save_model('saved_model/bart-base-text2text-dcard-all')

# %% [markdown]
# ## Prediction

# %%
import pandas as pd
import numpy as np
private_test = pd.read_csv('../../raw_data/intern_homework_private_test_dataset.csv')

# %%
# 指定要刪除的 column names，並使用 drop 函數將這些 column 刪除
drop_columns = ['author_id']
private_test_input = private_test.drop(drop_columns, axis=1)

# %%
# 將文章發佈時間 拆成 星期幾 與 小時 的函數
def split_date(df, date_column):

    # 將 created_by 欄位轉換成日期格式
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    
    # 新增 星期幾 和 小時 欄位
    df['weekday'] = df[date_column].dt.weekday
    df['hour'] = df[date_column].dt.hour

    # 移除 created_by 欄位
    df = df.drop(date_column, axis=1)

    # 回傳處理過的資料集
    return df

# %%
private_test_input = split_date(private_test_input, 'created_at')

weekday_dict = {
    0: '星期一',
    1: '星期二',
    2: '星期三',
    3: '星期四',
    4: '星期五',
    5: '星期六',
    6: '星期日'
}

# 將 weekday 轉換為中文星期幾
private_test_input['weekday'] = private_test_input['weekday'].map(weekday_dict)

# %%
# 將 dataframe 內的 feature 合併成一串文字 的函數
def transfrom_to_text(df):

    passage = ""

    # combined title and post time
    passage += "這篇文章標題是 {} ，在{}{}點發佈。".format(df["title"], df["weekday"], df["hour"])

    # combined forum_id and forum_stats
    passage += "這篇發佈的看板 ID 為 {}， 看板資訊為 {} 。".format(df["forum_id"], df["forum_stats"])

    # combined likes_count
    passage += "文章在發佈後的第一小時累積到的愛心數有 {}，在第二小時累積到的愛心數有 {}，".format(df["like_count_1h"], df["like_count_2h"])
    passage += "第三小時累積到的愛心數有 {}，在第四小時累積到的愛心數有 {}，".format(df["like_count_3h"], df["like_count_4h"])
    passage += "第五小時累積到的愛心數有 {}，在第六小時累積到的愛心數有 {}。".format(df["like_count_5h"], df["like_count_6h"])

    # combined comment_count
    passage += "文章在發佈後的第一小時累積到的留言數有 {}，在第二小時累積到的留言數有 {}，".format(df["comment_count_1h"], df["comment_count_2h"])
    passage += "第三小時累積到的留言數有 {}，在第四小時累積到的留言數有 {}，".format(df["comment_count_3h"], df["comment_count_4h"])
    passage += "第五小時累積到的留言數有 {}，在第六小時累積到的留言數有 {}。".format(df["comment_count_5h"], df["comment_count_6h"])

    
    return passage

# %%
private_test_text = private_test_input.apply(transfrom_to_text, axis=1).tolist()
private_test_text[0]

# %%
print("預測資料總共有 {} 筆".format(len(private_test_text)))

# %%
from transformers import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
private_test_encodings = tokenizer(private_test_text, truncation=True, padding=True)

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

private_test_dataset = Dataset(private_test_encodings)

# %%
predictions, labels, metrics = trainer.predict(private_test_dataset)

# %%
metrics

# %%
predictions

# %%
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_preds

# %% [markdown]
# Save Prediction

# %%
df = pd.DataFrame(decoded_preds, columns=["like_count_24h"])
df.to_csv("result(without_author_id).csv", index=False)



---
title: Dcard
author: Han Cheng Yu
date: "2023-04-11"
CJKmainfont: "Microsoft YaHei Mono"
---

# Dcard 作業 ReadME


## 檔案說明
本專案包含以下資料夾及檔案：
```
余翰承_2023_dcard_ml_intern_homework/
.
├── result.csv
├── result.pdf
├── feature_analysis: 存放針對資料集進行分析的程式碼與結果
│   ├── time
│   │   ├── analysis_time.ipynb
│   │   ├── hours_likes_relationship.png 
│   │   └── weekday_likes_relationship.png
│   └── title
│       ├── analysis_title.ipynb
│       ├── dcard_gray.png
│       ├── dcard.jpg
│       ├── dcard_wordcloud.png : 關鍵字文字雲的圖片檔案
│       ├── dcard_wordcloud(考慮動詞).png
│       ├── histogram of title sentiment analysis.png
│       ├── NotoSansTC-Regular.otf
│       ├── title_wordcloud.png
│       └── title_wordcloud(考慮動詞).png
├── model : 本次作業存放模型訓練與預測 csv 的資料夾
│   ├── text_only <-- NLP Model 訓練模型的 Jupyter Notebook 與 python 檔
│   │   ├── bart_text2text_dcard_all.ipynb
│   │   ├── bart_text2text_dcard_all.py
│   │   ├── bart_text2text_dcard.ipynb
│   │   ├── bart_text2text_dcard.py
│   │   ├── bart_text2text_dcard(with forum_stats).ipynb
│   │   ├── bart_text2text_dcard_with forum_stats.py
│   │   ├── bert_text2text_dcard.ipynb
│   │   ├── result(with forum_stats).csv
│   │   └── saved_model: 訓練好的模型參數
│   │       ├── bart-base-text2text-dcard
│   │       │   ├── config.json
│   │       │   ├── generation_config.json
│   │       │   ├── pytorch_model.bin
│   │       │   ├── special_tokens_map.json
│   │       │   ├── tokenizer_config.json
│   │       │   ├── training_args.bin
│   │       │   └── vocab.txt
│   │       ├── bart-base-text2text-dcard-all
│   │       ├── bart-base-text2text-dcard-forum_stats
│   │       └── bert-base-dcard
│   ├── transformers_with_tabular_data <-- Multimodal Transformers 訓練模型的 Jupyter Notebook
│   │   ├── Mixing BERT with Numerical Features.ipynb
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   └── without_title: baseline AutoML 訓練模型的 Jupyter Notebook
│       ├── auto-sklearn.ipynb
│       └── pycaret.ipynb
└── raw_data: 存放原始的訓練、測試、預測資料集
    ├── intern_homework_private_test_dataset.csv: 訓練集資料
    ├── intern_homework_public_test_dataset.csv: 測試集資料
    └── intern_homework_train_dataset.csv: 預測集資料
```

## 模型訓練與評估
本次作業使用了多種機器學習模型來進行訓練，包括:

1. Auto-sklearn : 在 without_title 的資料夾內
2. Pycaret : 在 without_title 的資料夾內
3. BERT : 在 text_only 的資料夾內
4. BART : 在 text_only 的資料夾內
5. Multimodel Transformers : 在 transformers_with_tabular_data 的資料夾內

使用 5種不同方法進行預測。


其中 BART 有額外再做 2 種實驗
1. 檔名 bart_text2text_dcard(with forum_stats)
    * Input Feature 考慮 forum_stats  
2. 檔名 bart_text2text_dcard_all
    * Input Feature 考慮全部的 Feature (連同 author_id 也考慮)


## 模型訓練環境

執行環境
* Python 3.8.10
* torch 1.11.0+cu113

套件
* pandas 
* numpy
* matplotlib
* scikit-learn
* transformer
* multimodal-transformers


## How to Read

本次作業主要以 jupyter notebook 的方式進行 python 的程式撰寫，每個步驟以及程式碼在做什麼，皆有對應的 Markdown 與 範例解釋。另外，也有提供對應程式 python 檔，只需要輸入
```
python bart_text2text_dcard.py
```
程式會自動進行 資料前處理、模型建立、模型訓練、模型評估與預測 的功能。

## Authors
**Han Cheng Yu:** boy19990222@gmail.com
# LLM Bootcamp - NVIDIA NeMo 大型語言模型訓練實戰教學 🚀

歡迎來到 LLM Bootcamp！本教學將帶您完整體驗使用 [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) 進行大型語言模型（LLM）的完整流程，從零開始學會模型轉換、預訓練、微調到部署的實戰技巧。

## 🎯 學習目標

通過本 Bootcamp，您將學會：

1. **🔄 模型轉換技能**：掌握 Hugging Face 與 NeMo 格式間的轉換
2. **📚 預訓練實踐**：體驗大規模語言模型的持續預訓練
3. **🛠️ 微調技術**：學會針對特定任務進行模型微調、掌握 LoRA 等參數高效微調方法
4. **📊 模型評估**：學會評估和測試模型性能
5. **🚀 模型部署**：了解模型導出和部署流程

---

## 📂 教學大綱

- [🚀 開始之前：環境設定](#🛠️-環境準備)
  - [📖 詳細環境設定指南](setup/README.md) ⭐
- [📥 專案設置](#📥-專案設置)
- [📖 詳細教學步驟](#📖-詳細教學步驟)
  - [第一章：模型轉換基礎](#第一章模型轉換基礎)
  - [第二章：持續預訓練實戰](#第二章持續預訓練實戰)
  - [第三章：指令微調技術](#第三章指令微調技術)
  - [第四章：Reasoning 資料微調技術](#第四章reasoning-資料微調技術)
  - [第五章：模型評估與測試](#第五章模型評估與測試)
  - [第六章：模型部署與轉換](#第六章模型部署與轉換)
- [💡 實戰技巧](#💡-實戰技巧)
- [📚 進階學習資源](#📚-進階學習資源)

---

## 🛠️ 環境準備

### 📋 選擇您的環境

在開始訓練之前，您需要設定適合的 GPU 環境。我們提供詳細的環境設定指南：

#### 🌟 TWCC 雲端環境

**設定步驟**：詳見 **[📖 TWCC 環境設定指南](setup/README.md)**

#### 🐳 本地 Docker 環境

使用官方 NeMo 容器，包含所有必要的依賴套件：

```bash
docker run \
    --gpus all -it --rm --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $PWD:$PWD -w $PWD -p 8888:8888 \
    nvcr.io/nvidia/nemo:25.04
```

> 💡 **提示**：您可以在 [NGC NeMo Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) 查看最新版本

---

## 📥 專案設置

### 下載教學專案

```bash
git clone https://github.com/wcks13589/LLM-Tutorial.git
cd LLM-Tutorial
```

> 💡 **提示**：請確保在 `LLM-Tutorial` 專案目錄中執行後續指令。

### 🔑 設定 Hugging Face 權限

申請並設定您的 Hugging Face Token：

1. **申請 Token**：前往 [Hugging Face Settings](https://huggingface.co/settings/tokens) 建立新的 Access Token
2. **設定環境變數**：
   ```bash
   # 替換為您的實際 Token
   export HF_TOKEN="your_hf_token"
   huggingface-cli login --token $HF_TOKEN
   ```

> 📌 **重要**：請先在 [Hugging Face](https://huggingface.co/settings/tokens) 申請 Access Token

## 📖 詳細教學步驟

### 第一章：模型轉換基礎

#### 1.1 下載預訓練模型

```bash
# 下載 Llama 3.1 8B Instruct 模型
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir Llama-3.1-8B-Instruct \
    --exclude original/
```

#### 1.2 轉換為 NeMo 格式

```bash
# 設定變數
MODEL=llama3_8b
HF_MODEL_ID=Llama-3.1-8B-Instruct
OUTPUT_PATH=nemo_ckpt/Llama-3.1-8B-Instruct
OVERWRITE_EXISTING=false

# 執行轉換
nemo llm import -y \
    model=${MODEL} \
    source=hf://${HF_MODEL_ID} \
    output_path=${OUTPUT_PATH} \
    overwrite=${OVERWRITE_EXISTING}
```

> ✅ **檢查點**：確認 `nemo_ckpt/` 目錄下成功生成了 NeMo 格式的模型檔案

---

### 第二章：持續預訓練實戰

#### 2.1 準備訓練資料

**下載中文資料集**：

```bash
python data_preparation/download_pretrain_data.py \
    --dataset_name erhwenkuo/wikinews-zhtw \
    --output_dir data/custom_dataset/json/wikinews-zhtw.jsonl
```

#### 2.2 資料預處理

```bash
# 建立預處理目錄
mkdir -p data/custom_dataset/preprocessed

# 使用 NeMo 的資料預處理工具
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=data/custom_dataset/json/wikinews-zhtw.jsonl \
    --json-keys=text \
    --dataset-impl mmap \
    --tokenizer-library=huggingface \
    --tokenizer-type meta-llama/Llama-3.1-8B-Instruct \
    --output-prefix=data/custom_dataset/preprocessed/wikinews \
    --append-eod
```

#### 2.3 執行預訓練

##### 前置準備

設定基本參數：

```bash
JOB_NAME=llama31_pretraining
NUM_NODES=1
NUM_GPUS=1
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct

# 並行處理參數
TP=1  # Tensor Parallel
PP=1  # Pipeline Parallel  
CP=1  # Context Parallel

# 訓練參數
GBS=8          # Global Batch Size
MAX_STEPS=20  # 最大訓練步數(模型權重更新次數)
DATASET_PATH=data/custom_dataset/preprocessed/
```

##### 方法一：從頭開始預訓練模型

**適用情況**：當您想要從零開始訓練模型時使用。

**特點**：腳本會自動從基礎模型架構進行權重初始化

**執行指令**：
```bash
python pretraining/pretrain.py \
   --executor local \
   --experiment ${JOB_NAME} \
   --num_nodes ${NUM_NODES} \
   --num_gpus ${NUM_GPUS} \
   --model_size 8B \
   --hf_model_id ${HF_MODEL_ID} \
   --hf_token ${HF_TOKEN} \
   --max_steps ${MAX_STEPS} \
   --global_batch_size ${GBS} \
   --tensor_model_parallel_size ${TP} \
   --pipeline_model_parallel_size ${PP} \
   --context_parallel_size ${CP} \
   --dataset_path ${DATASET_PATH}
```

> **重要提醒**：本教學內容特別針對 V100 32GB GPU 進行配置優化
> 
> 由於本教學內容預計使用 V100 32GB 的 GPU 來實作，為確保可以順利執行模型的訓練，我們在 `pretrain.py` 中特地將模型的層數與維度大幅降低：
> 
> **模型配置對比**：
> - **原始 Llama3.1 8B 模型**：
>   - `num_layers = 32`
>   - `hidden_size = 4096`
>   - 參數量： 8B 個參數
> - **調整後配置**：
>   - `num_layers = 1`
>   - `hidden_size = 128`
>   - 參數量：大幅降低，適合單張 V100 GPU
> 
> **調整原因**：
> - 確保在 V100 32GB 記憶體限制下能順利執行
> - 降低訓練時間，提供更好的學習體驗
> - 保持完整的訓練流程，讓學習者理解整個預訓練過程
> 
> **程式碼位置**：這些配置調整位於 `pretrain.py` 中的 `configure_recipe` 函數內。
> 
> **具體修改的程式碼**：
> ```python
> recipe.model.config.num_layers = 1
> recipe.model.config.hidden_size = 128
> ```

> 📊 **監控訓練**：訓練過程中可以觀察 loss 變化來判斷模型學習狀況

##### 方法二：從預訓練模型開始繼續預訓練

**適用情況**：當您想要從現有的 NeMo 格式模型開始，進行持續預訓練時使用。

**前置條件**：
- 需要先將 Hugging Face 模型轉換為 NeMo 格式
- 確保 `${NEMO_MODEL}` 路徑下存在有效的 NeMo 模型檔案

**執行指令**：
```bash
NEMO_MODEL=nemo_ckpt/Llama-3.1-8B-Instruct

python pretraining/pretrain.py \
   --executor local \
   --experiment ${JOB_NAME} \
   --num_nodes ${NUM_NODES} \
   --num_gpus ${NUM_GPUS} \
   --model_size 8B \
   --hf_model_id ${HF_MODEL_ID} \
   --nemo_model ${NEMO_MODEL} \
   --hf_token ${HF_TOKEN} \
   --max_steps ${MAX_STEPS} \
   --global_batch_size ${GBS} \
   --tensor_model_parallel_size ${TP} \
   --pipeline_model_parallel_size ${PP} \
   --context_parallel_size ${CP} \
   --dataset_path ${DATASET_PATH}
```

---

### 第三章：指令微調技術

#### 3.1 準備微調資料

```bash
# 下載並準備 Alpaca 資料集
python data_preparation/download_sft_data.py
```

#### 3.2 執行 LoRA 微調

```bash
# 微調參數設定
JOB_NAME=llama31_finetuning
NUM_NODES=1
NUM_GPUS=1
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
NEMO_MODEL=nemo_ckpt/Llama-3.1-8B-Instruct
# LATEST_CHECKPOINT=$(find nemo_experiments/llama31_pretraining/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)
HF_TOKEN=$HF_TOKEN

# 並行處理參數
TP=1
PP=1
CP=1

# 微調參數
MAX_STEPS=20
GBS=8
DATASET_PATH=data/alpaca

# 執行 LoRA 微調
python finetuning/finetune.py \
    --executor local \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --model_size 8B \
    --hf_model_id ${HF_MODEL_ID} \
    --hf_token ${HF_TOKEN} \
    --nemo_model ${NEMO_MODEL} \
    --max_steps ${MAX_STEPS} \
    --global_batch_size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_PATH} \
    --peft lora
```

> 🎯 **全參數微調**：若要進行全參數的微調，請移除`--peft lora`

---

### 第四章：Reasoning 資料微調技術

#### 4.1 準備 Reasoning 資料集

```bash
# 建立 Reasoning 資料集目錄
mkdir -p data/reasoning_dataset/

# 下載 NVIDIA Llama-Nemotron 後訓練資料集
wget https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/resolve/main/SFT/chat/chat.jsonl -P data/reasoning_dataset/
```

#### 4.2 資料預處理與策展

```bash
# 執行資料策展與預處理
python data_preparation/curate_reasoning_data.py \
    --input-dir "data/reasoning_dataset" \
    --filename-filter "chat" \
    --remove-columns "category" "generator" "license" "reasoning" "system_prompt" "used_in_training" "version" \
    --json-files-per-partition 16 \
    --tokenizer "meta-llama/Llama-3.1-8B-Instruct" \
    --max-token-count 16384 \
    --max-completion-token-count 8192 \
    --output-dir data/reasoning_dataset/curated-data \
    --device "gpu" \
    --n-workers 1
```

#### 4.3 執行 Reasoning LoRA 微調

```bash
# Reasoning 微調參數設定
JOB_NAME=llama31_reasoning_finetuning
NUM_NODES=1
NUM_GPUS=1
HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
NEMO_MODEL=nemo_ckpt/Llama-3.1-8B-Instruct
HF_TOKEN=$HF_TOKEN

# 並行處理參數
TP=1
PP=1
CP=1

# 微調參數
MAX_STEPS=20
GBS=8
DATASET_PATH=data/reasoning_dataset/curated-data

# 執行 Reasoning LoRA 微調
python finetuning/finetune.py \
    --executor local \
    --experiment ${JOB_NAME} \
    --num_nodes ${NUM_NODES} \
    --num_gpus ${NUM_GPUS} \
    --model_size 8B \
    --hf_model_id ${HF_MODEL_ID} \
    --hf_token ${HF_TOKEN} \
    --nemo_model ${NEMO_MODEL} \
    --max_steps ${MAX_STEPS} \
    --global_batch_size ${GBS} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --dataset_path ${DATASET_PATH} \
    --peft lora
```

> 🧠 **Reasoning 微調特色**：使用高品質的推理資料集，提升模型的邏輯推理和複雜問題解決能力

---

### 第五章：模型評估與測試

#### 5.1 準備測試資料

```bash
# 從測試集中選取樣本進行快速評估
head -n 30 data/alpaca/test.jsonl > data/alpaca/test_subset.jsonl
```

#### 5.2 執行推理測試

```bash
# 使用微調後的模型進行推理
# 找到最新的檢查點資料夾
LATEST_CHECKPOINT=$(find nemo_experiments/llama31_finetuning/checkpoints/ -type d -name "*-last" | sort -r | head -n 1)

python evaluation/inference.py \
    --peft_ckpt_path ${LATEST_CHECKPOINT} \
    --input_dataset data/alpaca/test_subset.jsonl \
    --output_path data/alpaca/peft_prediction.jsonl
```

#### 5.3 計算評估指標

```bash
# 計算模型性能指標
python /opt/NeMo/scripts/metric_calculation/peft_metric_calc.py \
    --pred_file data/alpaca/peft_prediction.jsonl \
    --label_field "label" \
    --pred_field "prediction"
```

---

### 第六章：模型部署與轉換

#### 6.1 轉換回 Hugging Face 格式

```bash
# 設定轉換參數
OUTPUT_PATH=hf_ckpt

# 執行轉換
nemo llm export -y \
    path=${LATEST_CHECKPOINT} \
    output_path=${OUTPUT_PATH} \
    target=hf
```

## 📚 進階學習資源

### 官方文檔
- [NeMo 官方文件](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)

### 進階主題
1. **多模態模型訓練**
2. **分散式訓練優化**
3. **模型壓縮與量化**
4. **自定義資料載入器**

---

## 🎉 恭喜完成 LLM Bootcamp！

通過本教學，您已經掌握了：
- ✅ 大型語言模型的完整訓練流程
- ✅ NeMo 框架的核心功能
- ✅ 實際的 AI 模型開發技能
- ✅ 企業級 AI 應用開發基礎

**下一步建議**：
1. 嘗試使用自己的資料集
2. 探索不同的模型架構
3. 學習模型部署與服務化
4. 參與開源專案貢獻

---

> 💬 **需要幫助？** 歡迎在 [Issues](https://github.com/wcks13589/NeMo-Tutorial/issues) 中提出問題或建議！

**Happy Learning! 🚀**
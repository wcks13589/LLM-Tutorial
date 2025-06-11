# Llama 3.1-8B 微調腳本說明

此文件說明如何使用 `finetuning/finetune.py` 腳本進行 Llama 3.1-8B 模型的微調。該腳本基於 NVIDIA NeMo 平臺，並利用 PyTorch Lightning 與 NeMo 完成微調工作。

---

## 主要功能

- 自定義資料集的路徑與參數。
- 配置微調的模型參數與分佈式訓練配置。
- 支持使用 `torchrun` 作為啟動分佈式訓練的工具。
- 支持檢查點恢復與自動化。
- 支持 LoRA (Low-Rank Adaptation) 參數高效微調。

---

## 內容概要

### 資料集設定

資料集的設定通過 `configure_dataset` 函數完成，包括：

- **資料夾結構**：微調的資料應按照如下結構放置。
- **batch_size**：`global_batch_size` 與 `micro_batch_size`。
- **seq_length**：設定模型輸入的序列長度（預設 8192）。
- **tokenizer**：使用 Huggingface 的標記器，指定為 `meta-llama/Llama-3.1-8B-Instruct`。
- **prompt_template**：提供 prompt 格式化模板。

#### 資料集格式

微調的資料需使用 JSON Lines 格式，每行為一個 JSON 對象，包含 `input` 與 `output` 欄位。

示例資料：

```json
{"input":"請解釋 Python 的主要功能。","output":"Python 是一種高效能的程式語言，具有易於學習、強大的標準庫與多樣化的應用場景。"}
{"input":"如何在資料中搜尋字串？","output":"可以使用 Python 的 `in` 關鍵字或者 `re` 模組進行正則表達式匹配。"}
```

#### 資料夾結構

資料應按以下結構儲存：

```
data/alpaca/
├── gpt4-chinese-zhtw.jsonl
├── test.jsonl
├── training.jsonl
└── validation.jsonl
```

- `training.jsonl`：訓練資料集。
- `validation.jsonl`：驗證資料集。
- `test.jsonl`：測試資料集。

---

## 資料準備步驟

### 1. 準備 Alpaca 資料集

```bash
# 下載並準備 Alpaca 資料集
python data_preparation/download_sft_data.py
```

### 2. 準備 Reasoning 資料集（可選）

```bash
# 建立 Reasoning 資料集目錄
mkdir -p data/reasoning_dataset/

# 下載 NVIDIA Llama-Nemotron 後訓練資料集
wget https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/resolve/main/SFT/chat/chat.jsonl -P data/reasoning_dataset/

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

---

## 執行方法

### 1. Alpaca 資料微調

確保資料按照上述格式與結構準備好後，執行 `finetune.py`：

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
GBS=8
MAX_STEPS=20
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

### 2. Reasoning 資料微調

```bash
# Reasoning 微調參數設定
JOB_NAME=llama31_reasoning_finetuning

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
GBS=8
MAX_STEPS=20
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

---

## 參數說明

以下是 `finetune.py` 訓練腳本的參數說明。你可以透過指令行參數來設定模型訓練的方式，例如選擇執行環境、指定模型、設定批次大小等。

## **實驗執行方式**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--executor` | `str` | `local` | 選擇執行方式，可選 `slurm`（使用 Slurm 叢集）或 `local`（單機執行）。 |
| `-E, --experiment` | `str` | `llama31_finetuning` | 設定實驗名稱，這會影響輸出資料夾的命名。 |

## **Slurm 參數設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-a, --account` | `str` | `root` | Slurm 的帳戶名稱，適用於需要多帳戶管理的 HPC 環境。 |
| `-p, --partition` | `str` | `defq` | Slurm 叢集的 Partition 名稱，不同叢集可能有不同的 Partition。 |
| `-i, --container_image` | `str` | `nvcr.io/nvidia/nemo:dev` | 指定要執行的 NeMo Docker 容器映像檔（通常是 NVIDIA NGC 提供的映像）。 |

## **硬體設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-N, --num_nodes` | `int` | `1` | 設定要使用的計算節點數量（適用於多機環境）。 |
| `-G, --num_gpus` | `int` | `8` | 每個計算節點使用的 GPU 數量（如果是單機訓練，通常設定為可用 GPU 數）。 |

## **模型設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-M, --model_size` | `str` | `8B` | 設定欲訓練的模型大小，預設包含 `8B` 與 `70B` 兩種選項 |
| `--hf_model_id` | `str` | **(必填)** | 指定要使用的 Hugging Face 模型 ID，例如 `"meta-llama/Llama-3.1-8B-Instruct"`。 |
| `-n, --nemo_model` | `str` | `None` | 指定預訓練好的 NeMo 模型權重路徑。 |
| `--hf_token` | `str` | **(必填)** | Hugging Face 的 API Token，用以下載 tokenizer。 |
| `-s, --seq_length` | `int` | `8192` | 設定模型輸入的序列長度。 |
| `--fp8` | `action` | `False` | 啟用 FP8 訓練模式以提高性能和記憶體效率。 |

## **訓練參數**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--max_steps` | `int` | `None` | 設定最大訓練步數（如果不設定，則以資料集大小計算 1 epoch 來決定）。 |
| `-g, --global_batch_size` | `int` | `8` | 訓練時的全域 batch size，需為 `micro_batch_size * data_parallel_size` 的倍數。 |
| `-m, --micro_batch_size` | `int` | `1` | 設定微批次大小，通常取決於單個 GPU 可承受的記憶體大小。 |

## **模型平行化設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-T, --tensor_model_parallel_size` | `int` | `1` | 設定 Tensor Model Parallelism（張量模型平行化）。 |
| `-P, --pipeline_model_parallel_size` | `int` | `1` | 設定 Pipeline Model Parallelism（管線平行化）。 |
| `-C, --context_parallel_size` | `int` | `1` | 設定 Context Parallelism（上下文平行化）。 |

## **PEFT (Parameter Efficient Fine-Tuning) 設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--peft` | `str` | `None` | 指定 PEFT 方法，支持 `lora`（Low-Rank Adaptation）和 `dora`。 |

## **資料集路徑**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-D, --dataset_path` | `str` | **(必填)** | 設定訓練資料夾的路徑，此資料夾應包含 `training.jsonl`, `validation.jsonl` 與 `test.jsonl` 。 |

## **WandB 記錄設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--wandb` | `action` | `False` | 啟用 WandB 記錄功能。 |
| `--wandb_project` | `str` | `None` | WandB 專案名稱，預設使用實驗名稱。 |
| `--wandb_name` | `str` | `None` | WandB 執行名稱，預設使用自動生成的名稱。 |
| `--wandb_token` | `str` | `None` | WandB 個人 API Token，用於認證。 |

---

## 檢查點恢復

若訓練中斷，可通過以下參數自動恢復：

- **策略**：`resume_if_exists=True`。
- **恢復路徑**：可透過 `restore_config` 指定檢查點路徑（如 `nemo_experiments/llama31_finetuning/checkpoints/...`）。

重新執行訓練腳本時，會自動檢查是否存在檢查點並進行恢復。

---

## 訓練過程輸出與紀錄

訓練過程的輸出會被記錄於實驗目錄 `nemo_experiments` 下，並支持即時監控與自動化管理。

---


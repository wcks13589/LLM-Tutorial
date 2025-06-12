# Llama 3.1-8B 預訓練腳本說明

此文件說明如何使用 `pretraining.py` 腳本進行大型語言模型 Llama 3.1-8B 的預訓練。該腳本培於 NVIDIA NeMo 平臺，並利用 PyTorch Lightning 與 NeMo 進行預訓練工作。

## 主要功能

- 配置資料集的路徑與參數。
- 配置訓練的參數與分佈式訓練參數。
- 支持使用 `torchrun` 作為啟動分佈式訓練的工具。
- 允許恢復中斷的訓練。

---

## 內容概要

### 資料集設定

資料集的設定通過 `configure_dataset` 函數完成，包括：

- **paths**：預處理後的資料儲放位置（預設為 `data/custom_dataset/preprocessed/wikinews_text_document`）。
- **batch_size**：`global_batch_size`與`micro_batch_size`。
- **seq_length**：設定模型輸入的序列長度（預設 8192）。
- **分割比例**：分割訓練、驗證、測試資料的比例。
- **tokenizer**：使用 Huggingface 提供的分詞器。

#### 資料集格式

預訓練需要使用格式为 JSON Lines (每行一個 JSON 對象)的資料集。每個 JSON 對象應包含一個字段 `text`，用來儲存文本內容。

以 Wikinews 為例，資料格式如下：

```
{"text":"中華民國...."}
{"text":"日近有程式設計師...."}
```

若要替換成其他資料集，可以導入相同格式的 JSON Lines。例如：
##### 客製化資料集

```
{"text":"Python 是一個強大而簡潔的程式語言，它擁有很多工具和平台支援...."}
{"text":"在測試過程中，我們採用了優化的模型構廻以提高準確性...."}
```

### 資料夾結構

- **資料處理**：使用預處理工具對原始資料進行Tokenize的動作。
- **資料儲存**：前處理完成後，資料應儲存在 `.bin` 和 `.idx` 格式中，模型訓練前需確認資料夾結構。

```
data
└─ custom_dataset
    ├─ json
    │   └─ wikinews-zhtw.jsonl
    └─ preprocessed
        ├─ wikinews_text_document.bin
        └─ wikinews_text_document.idx
```

---

## 執行方法

1. 確保資料已下載並預處。

   預處理完畢後，資料目錄中必須包含 `.bin` 與 `.idx` 檔案，結構如上。

2. 執行 `pretrain.py`：

   ```bash
   JOB_NAME=llama31_pretraining

   NUM_NODES=1
   NUM_GPUS=8

   HF_MODEL_ID=Llama-3.1-8B-Instruct
   NEMO_MODEL=nemo_ckpt/${HF_MODEL_ID}
   HF_TOKEN=<HF_TOKEN>

   TP=2
   PP=1
   CP=1

   GBS=2048
   MAX_STEPS=100
   DATASET_PATH=data/custom_dataset/preprocessed/

   python pretraining/pretrain.py \
      --executor local \
      --experiment ${JOB_NAME} \
      --num_nodes ${NUM_NODES} \
      --num_gpus ${NUM_GPUS} \
      --model_size 8B \
      --hf_model_id meta-llama/${HF_MODEL_ID} \
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

## 參數說明

以下是 `pretrain.py` 訓練腳本的參數說明。你可以透過指令行參數來設定模型訓練的方式，例如選擇執行環境、指定模型、設定批次大小等。

## **實驗執行方式**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--executor` | `str` | `local` | 選擇執行方式，可選 `slurm`（使用 Slurm 叢集）或 `local`（單機執行）。 |
| `-E, --experiment` | `str` | `llama31_pretraining` | 設定實驗名稱，這會影響輸出資料夾的命名。 |

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

## **訓練參數**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--max_steps` | `int` | `None` | 設定最大訓練步數（如果不設定，則以資料集大小計算 1 epoch 來決定）。 |
| `-g, --global_batch_size` | `int` | `2048` | 訓練時的全域 batch size，需為 `micro_batch_size * data_parallel_size` 的倍數。 |
| `-m, --micro_batch_size` | `int` | `1` | 設定微批次大小，通常取決於單個 GPU 可承受的記憶體大小。 |

## **模型平行化設定**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-T, --tensor_model_parallel_size` | `int` | `1` | 設定 Tensor Model Parallelism（張量模型平行化）。 |
| `-P, --pipeline_model_parallel_size` | `int` | `1` | 設定 Pipeline Model Parallelism（管線平行化）。 |
| `-C, --context_parallel_size` | `int` | `1` | 設定 Context Parallelism（上下文平行化）。 |

## **資料集路徑**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `-D, --dataset_path` | `str` | **(必填)** | 設定訓練資料夾的路徑，此資料夾應包含 .bin 和 .idx 檔案。 |

---

## 檢查點恢復

若訓練中斷，可通過下列參數自動恢復：

- **策略**：`resume_if_exists=True`。

重新執行訓練腳本，會自動檢查是否存在檢查點並進行恢復。

---

## 訓練過程輸出與紀錄

訓練過程中的輸出會被記錄於實驗目錄`nemo_experiments`下，並支持即時監控。

---
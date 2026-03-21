# chunky

命令行知识库构建工具

## 特性

- 📄 **多格式支持**: PDF, DOCX, PPTX, TXT, Markdown
- ✂️ **智能切块**: 基于语义和长度的文本分块
- 🏷️ **LLM 打标**: 自动为文档块生成标签
- 📊 **主题建模**: LDA / BertTopic 主题分析
- 🔍 **混合搜索**: 向量搜索 + BM25 融合
- 🔄 **智能重排序**: 支持本地 CrossEncoder 和 vLLM API 重排序
- 💾 **双向量存储**: ChromaDB (轻量) / Milvus (大规模)

## 命令行演示

```
$ chunky --help

  Usage: chunky [OPTIONS] COMMAND [ARGS]...

    chunky -- Build local knowledge bases from documents.

  Options:
    --help  Show this message and exit.

  Commands:
    build        Build a knowledge base from a directory of documents.
    chroma       Manage ChromaDB vector store configuration.
    collections  List all collections in the vector store.
    config       Show or modify configuration settings.
    embedding    Manage embedding model configuration.
    init         Interactive setup for chunky configuration.
    milvus       Manage Milvus vector store configuration.
    models       Manage LLM model configuration.
    reranker     Manage reranker model configuration.
    search       Search the knowledge base using hybrid search (vector + BM25).
```

```
$ chunky config --list

            LLM Configuration
 Setting      Value
 API Type     openai
 API Base     https://api.minimaxi.com/v1
 API Key      sk-a****azsk
 Model        MiniMax-M2-8K
 Max Tokens   1024
 Temperature  0.3

    Embedding Configuration
 Setting     Value
 API Type    vllm
 Model Name  bge-reranker-v2-m3
 API Base    10.****:8996
 API Key     sk-h****8996
 Device      cpu
 Batch Size  4

     Reranker Configuration
 Setting     Value
 API Type    vllm
 Model Name  BAAI/bge-reranker-v2-m3
 API Base    http://localhost:8000
 API Key     sk-****8996

     Vector Store Configuration
 Setting             Value
 Type                milvus
 URI                 localhost:19530
 Lite Mode           False
 Default Collection  test_milvus
 Dimension           384

     General
 Setting    Value
 Test Mode  OFF
```

```
$ chunky collections

  Collections (2 found)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name                      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ test_chroma               │
│ test_milvus (default)     │
└───────────────────────────┘
```

```
$ chunky search "machine learning" --collection test_chroma -k 3 -v

  ==================================================
  chunky search -- hybrid search (vector + BM25)
  ==================================================

  Search Configuration:
    Query:         machine learning
    Collection:    test_chroma
    Top K:         3
    Vector Weight: 0.5
    BM25 Weight:   0.5
    Fusion Method: RRF

  Connecting to vector store...
  Building BM25 index...
  Searching 4 chunks...

  ============================================================
  Found 3 results:
  ============================================================

  >>> Result #1: test_docs/test1.txt (chunk-0) (vec:-0.043, bm25:1.110, combined:0.033)
  ----------------------------------------
    This is a test document about machine learning. Machine learning is a subset of artificial intelligence.
    Labels: machine, learning, test, document, about
    Topics: is, learning, machine, about, this
```

```
$ chunky build --dir ./test_docs

  Build Summary
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Parameter        ┃ Value                                          ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
┃ Source directory ┃ /path/to/test_docs                             ┃
┃ Collection       │ test_chroma                                    ┃
┃ Test mode        │ OFF                                            ┃
┃ Total files      │ 4                                              ┃
│   .txt           │ 3                                              ┃
│   .md            │ 1                                              ┃
└──────────────────┴────────────────────────────────────────────────┘

  🏗️  Building knowledge base: test_chroma
  📁 Source directory: ./test_docs
  🧪 Test mode: OFF

  Step 1/6 Discovering and parsing files ...
    Parsed 4 documents.

  Step 2/6 Splitting text into chunks ...
    Created 8 chunks.

  Step 3-6/6 Processing chunks ...

  📄 Chunking files ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (4/4) 0:00:01
  🔢 Embedding      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (8/8) 0:00:05
  🏷️ LLM Labeling   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (8/8) 0:00:02
  💾 Milvus Insert  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (8/8) 0:00:00
  🎯 LDA Topics     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (8/8) 0:00:00

  ✅ Done! Processed 8 chunks.
```

## 快速开始

### 1. 初始化配置

```bash
chunky init
```

选择 LLM、Embedding 模型和向量存储类型 (ChromaDB 或 Milvus)。

### 2. 构建知识库

```bash
# 使用默认配置
chunky build --dir ./documents

# 指定集合名称
chunky build --dir ./documents --collection my_kb
```

### 3. 搜索

```bash
# 默认搜索
chunky search "机器学习"

# 指定集合和结果数
chunky search "深度学习" -c my_kb -k 10

# 调整搜索权重
chunky search "NLP" -vw 0.7 -bw 0.3 -f rrf

# 显示详细分数
chunky search "transformer" -v

# 使用重排序优化结果
chunky search "machine learning" -r --rerank-top-k 20 -k 5
```

### 4. 管理集合

```bash
# 列出所有集合
chunky collections

# 切换默认集合
chunky chroma --collection new_collection
chunky milvus --collection new_collection

# 删除集合（会提示确认）
chunky chroma --collection collection_xxx --delete
chunky milvus --collection collection_xxx --delete

# 快速切换测试模式
chunky config --test-mode on   # 启用测试模式
chunky config --test-mode off  # 禁用测试模式
```

## 搜索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-k, --top-k` | 返回结果数量 | 5 |
| `-vw, --vector-weight` | 向量搜索权重 | 0.5 |
| `-bw, --bm25-weight` | BM25 权重 | 0.5 |
| `-f, --fusion` | 融合方法: rrf/weighted_sum/relative_score | rrf |
| `-r, --rerank` | 启用重排序 | False |
| `--rerank-top-k` | 重排序的初始结果数 | 20 |

## 模型配置与下载

### 模型下载源

chunky 支持从以下源下载模型：

| 源 | 适用场景 | 特点 |
|---|---------|------|
| **Hugging Face** | 海外用户或有 VPN | 官方源，模型最全 |
| **ModelScope (魔搭社区)** | 国内用户 | 阿里云镜像，无需 VPN |

**自动选择逻辑**：
1. 首先测试 Hugging Face CDN 实际下载能力
2. 如果 HF 可用且代理工作正常 → 使用 Hugging Face
3. 如果 HF 被墙或代理无效 → 提示切换到 ModelScope

```bash
# 下载模型时会自动提示选择源
chunky build --dir ./documents

# Switch to ModelScope (魔搭社区)? [y/n] (n): y
```

### 模型缓存机制

模型下载后缓存到 `~/.cache/chunky/models/`，支持**模糊匹配查找**：

| 输入 | 匹配到的缓存 |
|-----|-------------|
| `Qwen/Qwen3-Embedding-0.6B` | `Qwen--Qwen3-Embedding-0.6B` |
| `Qwen3-Embedding-0.6B` | `Qwen--Qwen3-Embedding-0.6B` |
| `bge-reranker-base` | `BAAI--bge-reranker-base` |

**特点**：
- 自动识别短名（如 `bge-reranker-base` 匹配 `BAAI/bge-reranker-base`）
- 大小写不敏感
- 避免重复下载同一模型

### 内存优化（避免 OOM）

如果遇到 `zsh: killed` 或内存不足错误，请尝试以下方案：

**方案 1: 降低 Batch Size**（推荐）
```bash
# 编辑配置文件
vim ~/.config/chunky/config.yaml

embedding:
  batch_size: 1  # 从 4 降到 1
```

**方案 2: 使用更小的 Embedding 模型**
```bash
chunky embedding config
# 会显示缓存中的模型列表：
#   Cached models (use short name or full HF ID):
#     • BAAI/bge-small-zh-v1.5
#     • Qwen/Qwen3-Embedding-0.6B
#
# 推荐: BAAI/bge-small-zh-v1.5 (128MB，适合 8GB 内存)
# 避免: Qwen3-Embedding-0.6B (1.1GB，需要 16GB+ 内存)
```

**方案 3: 关闭 Reranker**
```bash
chunky reranker config
# 选择禁用或使用 API 方式，节省 3GB+ 内存
```

**方案 4: 使用 Test Mode**（无模型加载）
```bash
chunky config --test-mode on
```

### LLM Labeling 并发配置

LLM 打标签阶段默认使用 5 并发线程加速处理。可根据 API 限流调整：

```yaml
# ~/.config/chunky/config.yaml
llm:
  max_concurrent: 5  # 并发数 (1-20)，默认 5
```

**建议值**:
- MiniMax/OpenAI: 5-10（官方限流较严）
- 自建 vLLM: 10-20（内部服务可更高）
- 免费/低额度账号: 1-3（避免触发限流）

**性能对比**（59 chunks）:
| 并发数 | 预计耗时 | 提升 |
|-------|---------|------|
| 1 (串行) | ~6分钟 | 基准 |
| 5 (默认) | ~1.2分钟 | **5x** |
| 10 | ~40秒 | 9x |

## Reranker 配置

支持三种重排序方式：

### 1. 本地 CrossEncoder (默认)
- 使用本地 sentence-transformers 模型
- 无需外部服务

```bash
chunky reranker config
# 选择: local
# Model: BAAI/bge-reranker-base
```

### 2. vLLM Reranker API
- 使用 vLLM 部署的 reranker 服务
- 支持高并发场景

```bash
# 启动 vLLM reranker 服务
vllm serve BAAI/bge-reranker-v2-m3 --runner pooling

# 配置 chunky
chunky reranker config
# 选择: vllm
# API Base: http://localhost:8000
```

### 3. OpenAI 兼容 API
- 支持 Jina AI、Cohere 等兼容 API

```bash
chunky reranker config
# 选择: openai
# API Base: https://api.example.com
```

## 测试模式

测试模式使用轻量级实现，无需配置外部 API：

```bash
# 快速启用测试模式
chunky config --test-mode on

# 禁用测试模式
chunky config --test-mode off
```

测试模式特点：
- **Embedding**: 使用 bag-of-words (TF-IDF + SVD)
- **LLM Labeling**: 使用关键词提取
- **Reranker**: 禁用

## 向量存储

### ChromaDB (默认)
- 轻量、离线可用
- 适合小规模知识库

### Milvus
- 支持 Docker Standalone 部署
- 适合大规模数据

```bash
# 配置 Milvus
chunky milvus config

# 启动 Milvus Docker
cd ~/Desktop/play_code/chunky
docker-compose up -d
```

## 项目结构

```
chunky/
├── src/chunky/
│   ├── cli/          # CLI 入口
│   ├── config/       # 配置管理
│   ├── parsers/      # 文档解析
│   ├── chunking/     # 文本切块
│   ├── embedding/    # Embedding 生成
│   ├── llm/          # LLM 标签器
│   ├── topics/       # 主题建模
│   ├── vectorstore/  # 向量存储
│   ├── search/       # 混合搜索
│   └── pipeline/     # 处理流程
├── tests/            # 测试
└── docker-compose.yml # Milvus Docker 配置
```

## 配置位置

- 配置文件: `~/.config/chunky/config.yaml`
- 模型缓存: `~/.cache/chunky/models/`
- ChromaDB 数据: `~/.chunky/chroma_db/`

### 清理模型缓存

```bash
# 查看缓存大小
du -sh ~/.cache/chunky/models/*

# 删除所有缓存（下次会自动重新下载）
rm -rf ~/.cache/chunky/models/*
```

## 开发

```bash
# 安装
pip install -e ".[test,embedding]"

# 测试
pytest tests/ -v

# 构建文档到 Milvus
chunky build --dir test_docs --collection final_test
```

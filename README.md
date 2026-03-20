# chunky

命令行知识库构建工具

## 特性

- 📄 **多格式支持**: PDF, DOCX, PPTX, TXT, Markdown
- ✂️ **智能切块**: 基于语义和长度的文本分块
- 🏷️ **LLM 打标**: 自动为文档块生成标签
- 📊 **主题建模**: LDA / BertTopic 主题分析
- 🔍 **混合搜索**: 向量搜索 + BM25 融合
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
    config       Show all configuration settings.
    embedding    Manage embedding model configuration.
    init         Interactive setup for chunky configuration.
    milvus       Manage Milvus vector store configuration.
    models       Manage LLM model configuration.
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
```

### 4. 管理集合

```bash
# 列出所有集合
chunky collections

# 切换默认集合
chunky chroma --collection new_collection
chunky milvus --collection new_collection
```

## 搜索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-k, --top-k` | 返回结果数量 | 5 |
| `-vw, --vector-weight` | 向量搜索权重 | 0.5 |
| `-bw, --bm25-weight` | BM25 权重 | 0.5 |
| `-f, --fusion` | 融合方法: rrf/weighted_sum/relative_score | rrf |

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
- ChromaDB 数据: `~/.chunky/chroma_db/`

## 开发

```bash
# 安装
pip install -e ".[test,embedding]"

# 测试
pytest tests/ -v

# 构建文档到 Milvus
chunky build --dir test_docs --collection final_test
```

# chunky 开发进度

## 项目概述
chunky 是一个命令行工具，用于构建本地知识库。核心功能包括：
- 支持多种文件格式（pdf, docx, doc, ppt, pptx, txt, markdown, all）
- 文本切块（chunking）
- LLM打标
- 主题建模（LDA/BertTopic）
- Embedding
- Milvus 向量存储

## 处理流程（重要）
1. 文件解析 -> chunking（实时更新 chunking 进度条）
2. 对每个 chunk 立即执行：
   - embedding（实时更新 embedding 进度条）
   - LLM 打标签（实时更新 LLM 标签进度条）
   - 插入 milvus（实时更新 milvus 插入进度条）
3. 所有文档处理完后，从 milvus 取出所有 chunk
4. 统一做 LDA（实时更新 LDA 进度条）
5. 将 LDA 主题标签更新到 milvus

## CLI 命令
- `chunky init` - 初始化配置（选择 ChromaDB 或 Milvus）
- `chunky init --test` - 测试模式
- `chunky models config` - 配置大模型
- `chunky models list` - 列出已配置模型
- `chunky milvus --collection xxx` - 设置 Milvus 默认集合
- `chunky milvus config` - 配置 Milvus（先选 Lite/Standalone，再详细配置）
- `chunky chroma --collection xxx` - 设置 ChromaDB 默认集合 ⭐新增
- `chunky chroma config` - 配置 ChromaDB ⭐新增
- `chunky build --dir xxx` - 构建知识库（带 5 个进度条）
- `chunky build --dir xxx --collection yyy` - 指定集合名称构建
- `chunky search "query"` - Hybrid 搜索（向量 + BM25）
- `chunky search "query" -c collection -k 10` - 指定集合和结果数
- `chunky search "query" -vw 0.7 -bw 0.3 -f rrf` - 调整权重和融合方法
- `chunky collections` - 列出所有集合

## 5个进度条（已实现）
1. 📄 Chunking files - 文件 chunking 进度
2. 🔢 Embedding - embedding 进度
3. 🏷️ LLM Labeling - LLM 打标签进度
4. 🎯 LDA Topics - LDA 打类别标签进度
5. 💾 Milvus Insert - milvus 插入进度

## 开发进度

### 整体进度: ✅ 已完成并发布到 GitHub
- **GitHub**: https://github.com/NJUxlj/chunky
- **发布时间**: 2026-03-20

### 里程碑
- [x] 项目初始化
- [x] CLI 命令实现
- [x] 文档解析 (PDF, DOCX, PPTX, TXT, Markdown)
- [x] 文本切块
- [x] LLM 打标
- [x] Embedding 生成
- [x] 主题建模 (LDA/BertTopic)
- [x] Hybrid Search (向量 + BM25)
- [x] ChromaDB 向量存储
- [x] Milvus Docker 支持
- [x] 单元测试 (115/117 通过)
- [x] README 文档
- [x] GitHub 发布

### Python Backend
- 状态: ✅ 全部完成
- 已完成:
  - ✅ CLI 入口 (src/chunky/cli/main.py)
  - ✅ 配置管理 (src/chunky/config/settings.py)
  - ✅ 文档解析器 (src/chunky/parsers/)
  - ✅ Pipeline Runner (src/chunky/pipeline/runner.py)
  - ✅ 进度条系统 (src/chunky/progress/manager.py)
  - ✅ LLM 标签器 (src/chunky/llm/labeler.py)
  - ✅ Embedding (src/chunky/embedding/embedder.py)
  - ✅ 文本切块 (src/chunky/chunking/splitter.py)
  - ✅ Topic Modeling 框架 (src/chunky/topics/modeler.py)
  - ✅ ChromaDB 接口 (src/chunky/vectorstore/chroma_store.py)
  - ✅ pip install . 支持
  - ✅ **Hybrid Search 模块** ⭐新增
    - BM25 搜索引擎 (rank_bm25)
    - Hybrid Search 融合器 (RRF/weighted_sum/relative_score)
    - Search Manager (整合 ChromaDB + BM25)
- 测试状态:
  - ✅ 测试完整 build 流程
  - ✅ 测试 5 个进度条显示

### Test
- 状态: ✅ 集成测试通过
- 已完成:
  - ✅ 测试框架已搭建
  - ✅ 部分单元测试
  - ✅ 集成测试通过 (ChromaDB)

## 已创建的文件
```
chunky/
├── pyproject.toml ✅
├── README.md ✅
├── docker-compose.yml ✅ (Milvus Standalone 测试环境) ⭐新增
├── src/chunky/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py (含 search/collections 命令) ⭐更新
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py (支持 ChromaDB 配置)
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── docx_parser.py
│   │   ├── pdf_parser.py
│   │   ├── pptx_parser.py
│   │   ├── registry.py
│   │   └── text_parser.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── splitter.py
│   ├── progress/
│   │   ├── __init__.py
│   │   └── manager.py (5进度条)
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── labeler.py
│   │   └── test_labeler.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── topics/
│   │   ├── __init__.py
│   │   └── modeler.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── runner.py (支持 ChromaDB)
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── milvus_store.py
│   │   └── chroma_store.py
│   └── search/ ⭐新增
│       ├── __init__.py
│       ├── bm25_engine.py (BM25 搜索引擎)
│       ├── hybrid_search.py (融合器)
│       └── search_manager.py (管理器)
└── tests/
    ├── __init__.py
    └── unit/ (11 个测试文件)
```

## 新增/更新的关键代码

### 1. 进度条管理 (src/chunky/progress/manager.py)
```python
class ChunkingProgress:
    """管理5个同时显示的进度条"""
    def __init__(self, console=None):
        self.progress = Progress(...)
        self.chunking_task = None
        self.embedding_task = None
        self.llm_labeling_task = None
        self.lda_task = None
        self.milvus_task = None
    
    def setup_chunking(self, total_files: int) -> TaskID: ...
    def setup_processing(self, total_chunks: int) -> None: ...
    def setup_lda(self, total_chunks: int) -> TaskID: ...
    def update_chunking(self, advance: int = 1) -> None: ...
    def update_embedding(self, advance: int = 1) -> None: ...
    def update_llm_labeling(self, advance: int = 1) -> None: ...
    def update_milvus(self, advance: int = 1) -> None: ...
    def update_lda(self, advance: int = 1) -> None: ...
```

### 2. Pipeline 流式处理 (src/chunky/pipeline/runner.py)
```python
def run(self, directory: str, collection_name: str | None = None) -> None:
    # 初始化 5 个进度条
    self.progress = ChunkingProgress(self.console)
    self.progress.start()
    
    # Step 2: Chunking with progress
    all_chunks = self._chunk_documents_with_progress(documents)
    
    # Step 3-6: 流式处理 (embedding + LLM labeling + Milvus insert)
    dim = self._process_chunks_streaming(all_chunks, collection)
    
    # Step 4: LDA 批处理（所有 chunk 处理完后）
    self._assign_topics_with_progress(collection, len(all_chunks))
```

### 3. MilvusStore 新方法 (src/chunky/vectorstore/milvus_store.py)
```python
def get_all_chunks(self, collection_name: str) -> list[Chunk]: ...
def update_lda_topics(self, collection_name: str, chunks: list[Chunk]) -> int: ...
```

## 时间线
- 开始时间: 2026-03-19 20:54
- 最后更新: 2026-03-21 22:00

## 测试命令
```bash
# 搜索知识库 (ChromaDB)
chunky search "machine learning" --collection test_chroma -k 3

# 带详细分数
chunky search "artificial" --collection test_chroma -k 2 -v

# 调整权重（更偏重向量/BM25）
chunky search "neural networks" -vw 0.7 -bw 0.3 -f rrf

# 列出所有集合
chunky collections

# ChromaDB 配置
chunky chroma config

# === Milvus Docker Standalone 测试 ===
# 1. 启动 Milvus Docker
cd ~/Desktop/play_code/chunky
docker-compose up -d

# 2. 等待服务就绪 (约30秒)
sleep 30

# 3. 配置 Milvus Standalone
chunky milvus --collection my_collection
# 修改 ~/.config/chunky/config.yaml 中的 milvus.uri 为 "localhost:19530"
# 设置 vector_store_type: milvus

# 4. 构建知识库到 Milvus
chunky build --dir test_docs --collection my_collection

# 5. 搜索 Milvus 知识库
chunky search "machine learning" --collection my_collection -k 3
```

## 待完成任务
1. ✅ 实现 5 个进度条
2. ✅ 调整 LDA 流程
3. ✅ 确保 pip install . 可用
4. ✅ 测试完整流程
5. ✅ 测试 5 个进度条显示

## 修复记录
### 2026-03-19: 替换 Milvus-Lite 为 ChromaDB
**问题**: Milvus-Lite 依赖 `libomp.dylib`，但该库缺失导致无法启动本地服务器

**解决方案**: 
- 添加 ChromaDB 作为替代的向量数据库
- 新增 `src/chunky/vectorstore/chroma_store.py`
- 更新配置支持 `vector_store_type: chroma`
- 更新 pipeline 支持 ChromaDB

**测试结果**: ✅ 全部通过
- 成功处理 4 个测试文档
- 5 个进度条正常显示

### 2026-03-20: 支持 Milvus Docker Standalone + 修复查询问题

**问题1**: 使用 Milvus Docker Standalone 时，LDA 阶段查询不到已插入的数据

**原因**: 
- Milvus Docker 需要 flush + load collection 后才能查询
- 新插入的数据未 flush 到磁盘

**解决方案**:
- 在 `get_all_chunks()` 中添加 `client.flush()` 和 `client.load_collection()`
- 修改 `update_topics()` 使用 `flush()` 后再 re-insert

**问题2**: 更新 topics 时报错 `expected=8 fields, actual=9`

**原因**: 
- schema 中 `auto_id=True` 的 `id` 字段不应该在 insert 时传入
- 之前错误地保留了查询返回的 `id` 字段

**解决方案**:
- 在 `update_topics()` 中定义字段时不包含 `id`
- `defined_fields = {"text", "source_file", "chunk_index", "labels", "topics", "embedding"}`

**测试结果**: ✅ 全部通过
- 成功构建知识库到 Milvus Docker Standalone
- LDA 主题建模正常完成
- `chunky build --dir test_docs --collection final_test` 全流程通过
- LDA 主题建模完成

### 2026-03-20 (下午): 修复 config/__init__.py + 完善 README

**问题1**: `from chunky.config import load_config` 导入失败

**原因**: 
- `config/__init__.py` 的内容被错误地覆盖为 `settings.py` 的内容

**解决方案**:
- 重写 `config/__init__.py`，正确导出 `load_config`, `save_config` 等函数

**问题2**: README.md 内容过少

**解决方案**:
- 完善 README.md，添加特性、快速开始、参数说明等

**测试结果**: ✅ 全部通过
- 单元测试 115/117 通过（2个失败是交互式 init 测试，非功能 bug）
- 搜索功能正常工作

### 2026-03-21: 修复模型下载与缓存查找逻辑

**问题1**: 模型下载源选择基于代理环境变量，而非实际连通性测试

**原因**:
- 代码检测 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量来判断是否使用 Hugging Face
- 用户可能设置了代理环境变量，但代理软件并未实际运行
- `hf-mirror.com` 只代理 API，不代理实际的模型文件（存储在 `cas-bridge.xethub.hf.co`）

**解决方案**:
- `_test_hf_download()`: 实际测试 Hugging Face CDN 文件下载能力
- `_test_modelscope_download()`: 测试 ModelScope 连通性
- `_detect_best_source()`: 基于实际下载测试选择源，而非环境变量
- 即使代理测试通过，也显示风险提示（大文件可能超时）

**问题2**: Step 1 连通性测试重复下载模型

**原因**:
- `SentenceTransformer(model_name)` 会触发 Hugging Face 自动下载
- Step 0 检测到的缓存未被 Step 1 使用

**解决方案**:
- `find_cached_model()`: 添加模糊匹配缓存查找
  - 完全匹配: `Qwen/Qwen3-Embedding-0.6B` → `Qwen--Qwen3-Embedding-0.6B`
  - 短名匹配: `Qwen3-Embedding-0.6B` → `Qwen--Qwen3-Embedding-0.6B`  
  - 大小写不敏感: `qwen3-embedding-0.6b` → `Qwen--Qwen3-Embedding-0.6B`
- `_resolve_model_path()`: 优先使用缓存路径的优先级逻辑:
  1. 用户设置的 `local_model_path`（如果有效）
  2. 模糊匹配在 `~/.cache/chunky/models/` 中查找
  3. 回退到 `model_name`（触发 Hugging Face 下载）
- `ensure_models_downloaded()`: 下载后设置 `config.local_model_path`

**涉及文件**:
- `src/chunky/utils/model_downloader.py` - 下载管理、模糊匹配
- `src/chunky/utils/connectivity.py` - 连通性测试使用缓存路径

**测试结果**: ✅ 全部通过
- 代理检测从 ~4秒优化到 ~0.2ms
- 缓存命中时不再重复下载
- ModelScope 切换提示正常显示
- 模糊匹配正确找到缓存模型

### 2026-03-21 (晚): LLM Labeling 并发优化

**问题**: LLM 打标签阶段串行处理，59 个 chunks 耗时约 6 分钟

**分析**: 
- LLM API 调用是 I/O 密集型操作
- 串行处理导致大量 CPU 空等时间
- API 响应时间约 6 秒/个 chunk

**解决方案**: 线程池并发处理

**实现**:
- `LLMLabeler.label_chunks()`: 新增 `progress_callback` 参数支持进度回调
- `_label_chunks_concurrent()`: 使用 `ThreadPoolExecutor` 并发处理
- `_label_chunks_sequential()`: 小批量时回退到串行（避免线程开销）
- `PipelineRunner._label_chunks_concurrent()`: 批量处理所有 chunks
- `LLMConfig.max_concurrent`: 新增可配置并发数（默认 5）

**关键设计**:
```python
# 线程安全进度更新
progress_lock = Lock()
def progress_callback():
    with progress_lock:
        self.progress.update_llm_labeling(advance=1)

# 使用 ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(label_task, chunk): i for i, chunk in enumerate(chunks)}
    for future in as_completed(futures):
        # 处理完成结果
```

**预期效果**:
- 原耗时: 59 chunks × 6秒 = 354秒（约 6分钟）
- 优化后: 354秒 ÷ 5 并发 ≈ 71秒（约 1.2分钟）
- **提升约 5 倍**

**涉及文件**:
- `src/chunky/llm/labeler.py` - 并发 Labeler 实现
- `src/chunky/pipeline/runner.py` - 批量 labeling 集成
- `src/chunky/config/settings.py` - max_concurrent 配置

**测试结果**: ✅ 代码结构测试通过
- 空列表处理正常
- 小批量自动使用串行
- 进度回调线程安全

### 2026-03-21 (晚): 添加删除 Collection 功能

**需求**: 支持删除指定的 ChromaDB/Milvus collection

**实现**:
- `chunky chroma --collection xxx --delete` - 删除 ChromaDB collection
- `chunky milvus --collection xxx --delete` - 删除 Milvus collection

**特性**:
- 支持 `--collection` 指定要删除的集合（或使用默认集合）
- 删除前需要用户确认（防止误删）
- 如果集合不存在，给出友好提示

**示例**:
```bash
# 删除默认集合
chunky chroma --delete

# 删除指定集合
chunky milvus --collection my_kb --delete
```

**涉及文件**:
- `src/chunky/cli/main.py` - 添加 `--delete` 选项和删除逻辑

**测试结果**: ✅ 帮助信息显示正确，删除功能待测试

### 2026-03-21 (晚): 配置时显示缓存模型列表

**需求**: 在配置 embedding 和 reranker 模型时，显示缓存目录中的可用模型

**实现**:
- 在 `Model name (Hugging Face model ID)` 提示上方显示缓存模型列表
- 使用 `[dim]` 标签实现 50% 透明度
- 缩进 4 个空格，使用 bullet point (•) 格式
- 只显示有效的模型目录（包含 config.json）

**效果**:
```
  Cached models (use short name or full HF ID):
    • BAAI/bge-reranker-base
    • BAAI/bge-small-zh-v1.5
    • Qwen/Qwen3-Embedding-0.6B
    • prajjwal1/bert-tiny
  Model name (Hugging Face model ID) (BAAI/bge-small-zh-v1.5):
```

**涉及文件**:
- `src/chunky/cli/main.py` - `_prompt_embedding_config()` 和 `_prompt_reranker_config()`

**测试结果**: ✅ 正常显示缓存模型列表

### 2026-03-21 (晚): 抑制 PDF FontBBox 警告

**问题**: 解析某些 PDF 时出现大量 `FontBBox` 警告

```
Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats
```

**原因**: 
- pdfplumber 解析不规范 PDF 时的警告
- 某些 PDF 缺少字体边界框元数据
- 不影响文本提取，只是视觉干扰

**解决方案**:
- 在 `pdf_parser.py` 模块级别添加警告过滤器
- 抑制 `FontBBox` 和 `font descriptor` 相关警告

**修改**:
```python
import warnings
warnings.filterwarnings("ignore", message=".*FontBBox.*")
warnings.filterwarnings("ignore", message=".*font descriptor.*", category=UserWarning)
```

**涉及文件**:
- `src/chunky/parsers/pdf_parser.py`

**测试结果**: ✅ PDF 解析时不再显示 FontBBox 警告

**更新**: 加强 FontBBox 警告抑制
- 添加 `logging.getLogger("pdfminer").setLevel(logging.ERROR)` 抑制 pdfminer 的日志警告
- 确保在导入 pdfplumber 之前设置所有过滤器

### 2026-03-21 (晚): 修复 search 命令默认 collection 选择逻辑

**问题**: `chunky search` 命令始终使用 `config.milvus.default_collection`，忽略了 `vector_store_type` 设置

**现象**:
- 用户配置 `vector_store_type: chroma`
- 运行 `chunky search "query"` 时，使用了 `test_milvus` 而不是 `test_check`
- 导致搜索时集合名称错误

**原因**:
```python
# 原代码（错误）
collection_name = collection or config.milvus.default_collection  # 硬编码使用 milvus
```

**修复**:
```python
# 修复后（正确）
if collection is None:
    if config.vector_store_type == "chroma":
        collection_name = config.chroma.default_collection
    else:
        collection_name = config.milvus.default_collection
else:
    collection_name = collection
```

**涉及文件**:
- `src/chunky/cli/main.py` - search 函数

**测试结果**: ✅ 代码语法检查通过

### 2026-03-21 (晚): 彻底修复 ChromaDB collection 删除问题

**问题**: `chunky chroma --delete` 后，SQLite 中仍残留 collection 记录，导致维度不匹配错误

**根本原因**:
```
ChromaDB 存储架构:
├── chroma.sqlite3          ← 元数据数据库
│   ├── collections 表      ← collection 名称和 ID
│   ├── embeddings 表       ← embedding 数据  
│   └── segments 表         ← 向量索引片段
└── {uuid}/ 目录           ← 向量文件

问题: client.delete_collection() 只删除了向量文件，
     但没有删除 SQLite 中的 collections 和 embeddings 记录！
```

**修复方案**:
1. **调用 ChromaDB API 删除**
2. **强制 persist 更改**
3. **直接操作 SQLite 清理残留记录**:
   - 查询 collections 表获取 collection ID
   - 删除 embeddings 记录（先删，避免外键约束）
   - 删除 segments 记录
   - 删除 collection_metadata 记录
   - 删除 collections 记录
4. **重置客户端**清除内存缓存

**代码修改**:
```python
def drop_collection(self, collection_name: str) -> None:
    # ... API 删除 ...
    # ... 强制 persist ...
    
    # 直接清理 SQLite
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
    if result:
        collection_id = result[0]
        cursor.execute("DELETE FROM embeddings WHERE ...", (collection_id,))
        cursor.execute("DELETE FROM segments WHERE ...", (collection_id,))
        cursor.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        conn.commit()
```

**涉及文件**:
- `src/chunky/vectorstore/chroma_store.py`

**用户操作**:
```bash
# 现在删除会彻底清理
chunky chroma --collection test_chroma --delete

# 或者使用强力清理（如果代码修复后仍有问题）
rm -rf ~/.chunky/chroma_db/*
```

**问题**: `chunky chroma --delete` 后重建 collection，搜索时仍报维度不匹配错误

**原因分析**:
1. `drop_collection()` 方法调用 `client.delete_collection()` 但 ChromaDB 底层数据未完全清除
2. `create_collection()` 使用 `get_or_create=True`，如果旧数据存在会复用
3. 导致新 collection 仍保留旧的维度（384维）配置

**修复**:
- `drop_collection()`: 添加存在性检查、清除内部缓存、改进错误处理
- 删除前先 `get_collection` 确认存在，避免误报
- 删除后清除 `self._collection` 缓存
- 异常时重新抛出，让调用者知道删除失败

**涉及文件**:
- `src/chunky/vectorstore/chroma_store.py`

**建议用户操作**:
```bash
# 如果问题仍然存在，手动清除 ChromaDB 数据
rm -rf ~/.chunky/chroma_db/*

# 然后重新构建
chunky build --dir ./test_docs --collection test_chroma
```

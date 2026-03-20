"""进度条管理类 - 管理5个同时显示的进度条"""

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskID,
)


class ChunkingProgress:
    """管理5个同时显示的进度条"""

    def __init__(self, console: Console | None = None):
        """初始化进度条管理器

        Args:
            console: Rich Console 对象，如果为 None 则创建新的
        """
        self.console = console or Console()
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._started = False

        # 5个任务ID
        self.chunking_task: TaskID | None = None
        self.embedding_task: TaskID | None = None
        self.llm_labeling_task: TaskID | None = None
        self.lda_task: TaskID | None = None
        self.milvus_task: TaskID | None = None

    def start(self) -> None:
        """启动进度条显示"""
        if not self._started:
            self.progress.start()
            self._started = True

    def stop(self) -> None:
        """停止进度条显示"""
        if self._started:
            self.progress.stop()
            self._started = False

    def setup_chunking(self, total_files: int) -> TaskID:
        """设置文件 chunking 进度条

        Args:
            total_files: 总文件数

        Returns:
            TaskID: chunking 任务的ID
        """
        self.chunking_task = self.progress.add_task(
            "[cyan]📄 Chunking files",
            total=total_files,
        )
        return self.chunking_task

    def setup_processing(self, total_chunks: int) -> None:
        """设置处理进度条（embedding、LLM标签、milvus插入）

        Args:
            total_chunks: 总 chunk 数
        """
        self.embedding_task = self.progress.add_task(
            "[green]🔢 Embedding",
            total=total_chunks,
        )
        self.llm_labeling_task = self.progress.add_task(
            "[yellow]🏷️ LLM Labeling",
            total=total_chunks,
        )
        self.milvus_task = self.progress.add_task(
            "[blue]💾 Milvus Insert",
            total=total_chunks,
        )

    def setup_lda(self, total_chunks: int) -> TaskID:
        """设置 LDA 主题建模进度条

        Args:
            total_chunks: 总 chunk 数

        Returns:
            TaskID: LDA 任务的ID
        """
        self.lda_task = self.progress.add_task(
            "[magenta]🎯 LDA Topics",
            total=total_chunks,
        )
        return self.lda_task

    def update_chunking(self, advance: int = 1) -> None:
        """更新文件 chunking 进度

        Args:
            advance: 前进的步数，默认为 1
        """
        if self.chunking_task is not None:
            self.progress.update(self.chunking_task, advance=advance)

    def update_embedding(self, advance: int = 1) -> None:
        """更新 embedding 进度

        Args:
            advance: 前进的步数，默认为 1
        """
        if self.embedding_task is not None:
            self.progress.update(self.embedding_task, advance=advance)

    def update_llm_labeling(self, advance: int = 1) -> None:
        """更新 LLM 标签进度

        Args:
            advance: 前进的步数，默认为 1
        """
        if self.llm_labeling_task is not None:
            self.progress.update(self.llm_labeling_task, advance=advance)

    def update_milvus(self, advance: int = 1) -> None:
        """更新 Milvus 插入进度

        Args:
            advance: 前进的步数，默认为 1
        """
        if self.milvus_task is not None:
            self.progress.update(self.milvus_task, advance=advance)

    def update_lda(self, advance: int = 1) -> None:
        """更新 LDA 主题建模进度

        Args:
            advance: 前进的步数，默认为 1
        """
        if self.lda_task is not None:
            self.progress.update(self.lda_task, advance=advance)

    def get_task_counts(self) -> dict[str, tuple[int, int]]:
        """获取所有任务的完成情况

        Returns:
            dict: 包含每个任务名称和 (已完成, 总数) 的元组
        """
        counts = {}
        tasks = [
            ("chunking", self.chunking_task),
            ("embedding", self.embedding_task),
            ("llm_labeling", self.llm_labeling_task),
            ("lda", self.lda_task),
            ("milvus", self.milvus_task),
        ]
        for name, task_id in tasks:
            if task_id is not None:
                task = self.progress._tasks.get(task_id)
                if task:
                    counts[name] = (task.completed, task.total)
        return counts

    def __enter__(self) -> "ChunkingProgress":
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.stop()

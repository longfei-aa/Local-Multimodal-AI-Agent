from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging
from pathlib import Path

# 全局Console对象
console = Console()

def setup_logger(log_file='./logs/app.log', level=logging.INFO):
    """
    配置日志记录器
    同时输出到文件和控制台
    """
    Path('./logs').mkdir(exist_ok=True)

    logger = logging.getLogger('AIAssistant')
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器（只显示warning以上）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def create_progress_bar(iterable, desc: str, **kwargs):
    """创建进度条"""
    return tqdm(iterable, desc=desc, **kwargs)

def log_success(message: str):
    """成功信息（绿色显示）"""
    console.print(f"[green]✓[/green] {message}")

def log_error(message: str, exception: Exception = None):
    """错误信息（红色显示）"""
    error_msg = f"[red]✗[/red] {message}"
    if exception:
        error_msg += f"\n  原因: {str(exception)}"
    console.print(error_msg)

def log_warning(message: str):
    """警告信息（黄色显示）"""
    console.print(f"[yellow]⚠[/yellow] {message}")

def log_info(message: str):
    """普通信息"""
    console.print(f"[blue]ℹ[/blue] {message}")

class ProgressTracker:
    """批量处理的进度跟踪器"""

    def __init__(self, total: int, task_name: str):
        self.total = total
        self.task_name = task_name
        self.success_count = 0
        self.failed_count = 0
        self.results = []
        self.pbar = tqdm(total=total, desc=task_name)

    def update(self, success: bool, message: str, details: dict = None):
        """更新进度"""
        if success:
            self.success_count += 1
        else:
            self.failed_count += 1

        self.results.append({
            'success': success,
            'message': message,
            'details': details
        })

        self.pbar.set_postfix({
            '成功': self.success_count,
            '失败': self.failed_count
        })
        self.pbar.update(1)

    def close(self):
        """关闭进度条"""
        self.pbar.close()

    def get_summary(self) -> dict:
        """获取统计摘要"""
        return {
            'total': self.total,
            'success': self.success_count,
            'failed': self.failed_count,
            'success_rate': self.success_count / self.total if self.total > 0 else 0
        }

    def print_summary(self):
        """打印美化的统计摘要"""
        summary = self.get_summary()

        table = Table(title=f"{self.task_name} - 处理完成")
        table.add_column("指标", style="cyan")
        table.add_column("数值", style="magenta")

        table.add_row("总数", str(summary['total']))
        table.add_row("成功", f"[green]{summary['success']}[/green]")
        table.add_row("失败", f"[red]{summary['failed']}[/red]")
        table.add_row("成功率", f"{summary['success_rate']:.1%}")

        console.print(table)

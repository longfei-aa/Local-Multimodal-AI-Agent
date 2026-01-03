import argparse
import sys
from pathlib import Path

# 导入模块
from models.text_embedder import TextEmbedder
from models.image_embedder import ImageEmbedder
from services.vector_store import VectorStore
from services.paper_manager import PaperManager
from services.image_manager import ImageManager
from config.settings import DEFAULT_CONFIG
from utils.progress import log_info, log_success, log_error, log_warning, setup_logger
from rich.console import Console
from rich.table import Table

console = Console()


def init_services(config):
    """初始化所有服务"""
    log_info("正在加载模型...")

    text_embedder = TextEmbedder(config.get('text_embedding_model'))
    image_embedder = ImageEmbedder(
        model_name=config.get('image_embedding_model'),
        pretrained='openai'
    )
    vector_store = VectorStore(config.get('vector_db_dir'))

    paper_manager = PaperManager(text_embedder, vector_store, config)
    image_manager = ImageManager(image_embedder, vector_store, config)

    log_success("模型加载完成")

    return paper_manager, image_manager


def display_search_results(results):
    """美化显示搜索结果"""
    if not results:
        log_warning("未找到相关论文")
        return

    table = Table(title="搜索结果", show_lines=True)
    table.add_column("排名", style="cyan", width=6)
    table.add_column("论文信息", style="white")
    table.add_column("相似度", style="green", width=8)
    table.add_column("页码", style="yellow", width=15)

    for i, result in enumerate(results, 1):
        # 论文信息
        info = f"[bold]{result['title']}[/bold]\n"
        if result.get('authors'):
            authors = result['authors'][:60]
            if len(result['authors']) > 60:
                authors += '...'
            info += f"作者: {authors}\n"
        info += f"主题: {result['topic']}\n"
        info += f"文件: {result['filename']}"

        # 页码信息
        pages_str = ', '.join(map(str, result['pages'][:5]))
        if len(result['pages']) > 5:
            pages_str += '...'

        table.add_row(
            str(i),
            info,
            f"{result['score']:.3f}",
            f"第 {pages_str} 页"
        )

    console.print(table)


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description='AI 智能文献与图像管理助手',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py add_paper paper.pdf --topics "CV,NLP,RL"
  python main.py search_paper "transformer architecture"
  python main.py organize_papers ./papers --topics "CV,NLP,RL"
  python main.py index_images ./images
  python main.py search_image "sunset by the sea"
  python main.py stats
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # add_paper 子命令
    parser_add = subparsers.add_parser('add_paper', help='添加并分类论文')
    parser_add.add_argument('pdf_path', type=str, help='PDF文件路径')
    parser_add.add_argument('--topics', type=str, required=True,
                          help='候选主题（逗号分隔），如: "CV,NLP,RL"')

    # search_paper 子命令
    parser_search = subparsers.add_parser('search_paper', help='搜索论文')
    parser_search.add_argument('query', type=str, help='搜索查询')
    parser_search.add_argument('--top_k', type=int, default=5,
                              help='返回结果数量（默认5）')

    # organize_papers 子命令
    parser_organize = subparsers.add_parser('organize_papers',
                                           help='批量整理论文')
    parser_organize.add_argument('source_dir', type=str, help='源文件夹路径')
    parser_organize.add_argument('--topics', type=str, required=True,
                                help='候选主题（逗号分隔）')

    # index_images 子命令
    parser_index = subparsers.add_parser('index_images', help='索引图像库')
    parser_index.add_argument('image_dir', type=str, help='图像文件夹路径')

    # search_image 子命令
    parser_img_search = subparsers.add_parser('search_image', help='以文搜图')
    parser_img_search.add_argument('query', type=str, help='图像描述')
    parser_img_search.add_argument('--top_k', type=int, default=5,
                                  help='返回结果数量')

    # stats 子命令
    parser_stats = subparsers.add_parser('stats', help='查看统计信息')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # 加载配置
    config = DEFAULT_CONFIG

    # 初始化服务
    try:
        paper_manager, image_manager = init_services(config)
    except Exception as e:
        log_error("模型初始化失败", e)
        sys.exit(1)

    # 执行命令
    try:
        if args.command == 'add_paper':
            topics = [t.strip() for t in args.topics.split(',')]
            result = paper_manager.add_paper(args.pdf_path, topics)

            if result['success']:
                log_success(f"成功添加论文到 {result['topic']} 类别")
                console.print(f"\n论文标题: {result['metadata']['title']}")
                if result['metadata']['authors']:
                    authors = ', '.join(result['metadata']['authors'][:3])
                    console.print(f"作者: {authors}")
            else:
                log_error("添加失败", Exception(result.get('error')))

        elif args.command == 'search_paper':
            results = paper_manager.search_papers(args.query, top_k=args.top_k)
            display_search_results(results)

        elif args.command == 'organize_papers':
            topics = [t.strip() for t in args.topics.split(',')]
            result = paper_manager.batch_organize(args.source_dir, topics)

        elif args.command == 'index_images':
            result = image_manager.index_images(args.image_dir)

        elif args.command == 'search_image':
            results = image_manager.search_images(args.query, top_k=args.top_k)

            if results:
                table = Table(title="图像搜索结果")
                table.add_column("排名", style="cyan", width=6)
                table.add_column("文件名", style="white")
                table.add_column("相似度", style="green", width=10)
                table.add_column("路径", style="blue")

                for i, r in enumerate(results, 1):
                    table.add_row(
                        str(i),
                        r['filename'],
                        f"{r['score']:.3f}",
                        r['file']
                    )

                console.print(table)
            else:
                log_warning("未找到相关图像")

        elif args.command == 'stats':
            paper_stats = paper_manager.vector_store.get_collection_stats('papers')
            image_stats = image_manager.vector_store.get_collection_stats('images')

            table = Table(title="系统统计信息")
            table.add_column("类型", style="cyan")
            table.add_column("数量", style="magenta")

            table.add_row("论文chunk数", str(paper_stats.get('count', 0)))
            table.add_row("图像数", str(image_stats.get('count', 0)))

            console.print(table)

    except Exception as e:
        log_error("执行命令时出错", e)
        logger.exception("详细错误信息")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]用户中断操作[/yellow]")
        sys.exit(0)
    except Exception as e:
        log_error("程序执行出错", e)
        sys.exit(1)

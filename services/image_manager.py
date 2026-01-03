import os
import glob
from pathlib import Path
from typing import List, Dict

from utils.progress import (
    log_info, log_success, log_error, log_warning,
    ProgressTracker, setup_logger
)


class ImageManager:
    """
    图像管理服务
    负责图像索引和以文搜图功能
    """

    def __init__(self, image_embedder, vector_store, config):
        """
        初始化图像管理器

        Args:
            image_embedder: 图像嵌入模型
            vector_store: 向量数据库
            config: 配置字典
        """
        self.image_embedder = image_embedder
        self.vector_store = vector_store
        self.config = config
        self.logger = setup_logger()

    def index_images(self, image_dir: str) -> dict:
        """
        索引图像库

        Args:
            image_dir: 图像文件夹路径

        Returns:
            索引结果统计
        """
        # 支持的格式
        formats = self.config.get('supported_image_formats',
                                 ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])

        # 查找所有图像文件
        image_files = []
        for fmt in formats:
            pattern = os.path.join(image_dir, '**', f'*{fmt}')
            image_files.extend(glob.glob(pattern, recursive=True))

        if not image_files:
            log_warning(f"未找到图像文件: {image_dir}")
            return {'success': False, 'message': '未找到图像'}

        log_info(f"找到 {len(image_files)} 张图像")

        # 使用进度跟踪器
        tracker = ProgressTracker(
            total=len(image_files),
            task_name="索引图像"
        )

        embeddings_list = []
        metadatas_list = []
        ids_list = []

        for i, img_path in enumerate(image_files):
            try:
                # 生成图像嵌入
                embedding = self.image_embedder.encode_image(img_path)

                embeddings_list.append(embedding)
                metadatas_list.append({
                    'file_path': img_path,
                    'filename': Path(img_path).name
                })
                ids_list.append(f"image_{i}")

                tracker.update(success=True, message=Path(img_path).name)

            except Exception as e:
                tracker.update(success=False, message=Path(img_path).name)
                self.logger.error(f"处理失败 {img_path}: {e}")

        # 批量存入数据库
        if embeddings_list:
            import numpy as np
            embeddings_array = np.array(embeddings_list)

            self.vector_store.add_documents(
                collection_name='images',
                embeddings=embeddings_array,
                metadatas=metadatas_list,
                ids=ids_list
            )

        tracker.close()
        tracker.print_summary()

        return {
            'success': True,
            'indexed': len(embeddings_list),
            'total': len(image_files)
        }

    def search_images(self, text_query: str, top_k: int = 5) -> List[dict]:
        """
        以文搜图

        Args:
            text_query: 文本描述
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        log_info(f"搜索图像: {text_query}")

        # 文本转CLIP嵌入
        query_embedding = self.image_embedder.encode_text(text_query)

        # 向量检索
        results = self.vector_store.search(
            collection_name='images',
            query_embedding=query_embedding,
            top_k=top_k
        )

        # 格式化结果
        formatted_results = []
        for r in results:
            formatted_results.append({
                'file': r['metadata']['file_path'],
                'filename': r['metadata']['filename'],
                'score': r['score']
            })

        return formatted_results

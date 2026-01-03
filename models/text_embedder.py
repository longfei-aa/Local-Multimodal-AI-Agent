from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class TextEmbedder:
    """
    文本嵌入模型封装
    使用SentenceTransformers生成文本向量
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        初始化文本嵌入模型

        Args:
            model_name: SentenceTransformers模型名称
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """懒加载模型（首次使用时加载）"""
        if self._model is None:
            print(f"正在加载文本嵌入模型: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("文本嵌入模型加载完成")
        return self._model

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条

        Returns:
            numpy数组，形状为 (n, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本

        Args:
            text: 待编码的文本

        Returns:
            一维numpy数组
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        return self.model.get_sentence_embedding_dimension()

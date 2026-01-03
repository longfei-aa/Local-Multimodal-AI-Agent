import torch
import open_clip
from PIL import Image
import numpy as np
from typing import List, Union


class ImageEmbedder:
    """
    图像嵌入模型封装
    使用CLIP模型实现图像和文本的统一嵌入空间
    """

    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        """
        初始化CLIP模型

        Args:
            model_name: CLIP模型架构名称
            pretrained: 预训练权重来源
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            print(f"正在加载图像嵌入模型: {self.model_name}...")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            print(f"图像嵌入模型加载完成 (设备: {self.device})")
        return self._model

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        编码单张图像

        Args:
            image_path: 图像文件路径

        Returns:
            图像嵌入向量 (numpy数组)
        """
        # 确保模型已加载
        _ = self.model

        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        # 生成嵌入
        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().flatten()

    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码图像

        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小

        Returns:
            图像嵌入矩阵 (n, embedding_dim)
        """
        _ = self.model
        all_features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(self._preprocess(image))
                except Exception as e:
                    print(f"处理图像失败 {path}: {e}")
                    continue

            if not batch_images:
                continue

            # 批量处理
            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                features = self._model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu().numpy())

        if all_features:
            return np.vstack(all_features)
        else:
            return np.array([])

    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本（用于以文搜图）

        Args:
            text: 单个文本或文本列表

        Returns:
            文本嵌入向量
        """
        _ = self.model

        if isinstance(text, str):
            text = [text]

        # Tokenize文本
        text_tokens = self._tokenizer(text).to(self.device)

        # 生成嵌入
        with torch.no_grad():
            text_features = self._model.encode_text(text_tokens)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        result = text_features.cpu().numpy()

        # 如果输入是单个文本，返回一维数组
        if len(text) == 1:
            return result.flatten()

        return result

    def get_embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        _ = self.model
        return self._model.visual.output_dim

import os
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PAPERS_DIR = DATA_DIR / 'papers'
IMAGES_DIR = DATA_DIR / 'images'
VECTOR_DB_DIR = DATA_DIR / 'vector_db'
METADATA_DIR = DATA_DIR / 'metadata'
LOGS_DIR = BASE_DIR / 'logs'

# 创建必要的目录
for dir_path in [PAPERS_DIR, IMAGES_DIR, VECTOR_DB_DIR, METADATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 默认配置
DEFAULT_CONFIG = {
    # 路径配置
    'papers_dir': str(PAPERS_DIR),
    'images_dir': str(IMAGES_DIR),
    'vector_db_dir': str(VECTOR_DB_DIR),
    'metadata_dir': str(METADATA_DIR),
    'logs_dir': str(LOGS_DIR),

    # 模型配置
    'text_embedding_model': 'all-MiniLM-L6-v2',
    'image_embedding_model': 'ViT-B-32',  # CLIP模型名称

    # 向量数据库配置
    'paper_collection_name': 'papers',
    'image_collection_name': 'images',

    # 论文分类主题
    'default_topics': ['CV', 'NLP', 'RL'],

    # 文本处理配置
    'chunk_size': 512,
    'chunk_overlap': 50,
    'max_pages_for_classification': 5,

    # OCR配置
    'enable_ocr': True,
    'ocr_engine': 'paddleocr',
    'ocr_threshold': 100,  # 少于100字符认为是扫描版

    # 搜索配置
    'default_top_k': 5,
    'similarity_threshold': 0.3,

    # 支持的文件格式
    'supported_image_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'supported_document_formats': ['.pdf'],
}

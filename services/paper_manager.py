import os
import shutil
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from utils.pdf_parser import extract_text_by_pages, chunk_text_with_pages
from utils.metadata_extractor import MetadataExtractor
from utils.progress import (
    log_info, log_success, log_error, log_warning,
    ProgressTracker, setup_logger
)


class PaperManager:
    """
    文献管理服务
    负责论文的添加、搜索、分类和批量整理
    """

    def __init__(self, text_embedder, vector_store, config):
        """
        初始化文献管理器

        Args:
            text_embedder: 文本嵌入模型
            vector_store: 向量数据库
            config: 配置字典
        """
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.config = config
        self.metadata_extractor = MetadataExtractor()
        self.logger = setup_logger()

    def add_paper(self, pdf_path: str, topics: List[str]) -> dict:
        """
        添加单篇论文并自动分类

        Args:
            pdf_path: PDF文件路径
            topics: 候选主题列表

        Returns:
            处理结果字典
        """
        log_info(f"开始处理: {Path(pdf_path).name}")

        try:
            # 步骤1: 提取文本
            pages_data = extract_text_by_pages(pdf_path, use_ocr=True)
            log_success(f"提取了 {len(pages_data)} 页文本")

            # 步骤2: 提取元数据
            metadata = self.metadata_extractor.extract_metadata(pdf_path, pages_data)
            log_info(f"论文标题: {metadata['title']}")
            if metadata['authors']:
                log_info(f"作者: {', '.join(metadata['authors'][:3])}")

            # 步骤3: 文本分块（保留页码）
            chunks = chunk_text_with_pages(
                pages_data,
                chunk_size=self.config.get('chunk_size', 512),
                overlap=self.config.get('chunk_overlap', 50)
            )
            log_info(f"生成了 {len(chunks)} 个文本块")

            if not chunks:
                raise ValueError("未能从PDF中提取有效文本")

            # 步骤4: 生成嵌入
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.text_embedder.encode(texts)

            # 步骤5: 自动分类（使用标题+摘要，更准确）
            classification_text = f"{metadata['title']}. {metadata['abstract']}"
            if not metadata['abstract']:
                # 如果没有摘要，使用前1000字符
                classification_text = chunks[0]['text'][:1000]
            topic = self.classify_paper(classification_text, topics)
            log_info(f"分类结果: {topic}")

            # 步骤6: 准备元数据
            metadatas = []
            ids = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'file_path': pdf_path,
                    'filename': Path(pdf_path).name,
                    'topic': topic,
                    'title': metadata['title'] or 'Unknown',
                    'authors': ', '.join(metadata['authors']) if metadata['authors'] else 'Unknown',
                    'abstract': metadata['abstract'][:200] if metadata['abstract'] else '',
                    'year': metadata.get('year') or 0,  # ChromaDB不接受None，用0表示未知
                    'page': chunk['page'],
                    'chunk_id': chunk['chunk_id'],
                    'text_snippet': chunk['text'][:100]
                }
                metadatas.append(chunk_metadata)
                ids.append(f"{Path(pdf_path).stem}_chunk_{i}")

            # 步骤7: 存入向量数据库
            self.vector_store.add_documents(
                collection_name='papers',
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                documents=texts
            )

            # 步骤8: 移动文件到分类文件夹
            target_dir = Path(self.config['papers_dir']) / topic
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(pdf_path).name

            if not target_path.exists():
                shutil.copy2(pdf_path, target_path)
                log_success(f"文件已分类到: {topic}/")
            else:
                log_warning(f"文件已存在: {target_path}")

            return {
                'success': True,
                'topic': topic,
                'chunks': len(chunks),
                'metadata': metadata,
                'target_path': str(target_path)
            }

        except Exception as e:
            log_error(f"处理失败: {Path(pdf_path).name}", e)
            self.logger.exception(f"详细错误信息")
            return {'success': False, 'error': str(e)}

    def search_papers(self, query: str, top_k: int = 5) -> List[dict]:
        """
        语义搜索论文

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        log_info(f"搜索: {query}")

        # 生成查询嵌入
        query_embedding = self.text_embedder.encode_single(query)

        # 向量检索（返回更多结果用于聚合）
        results = self.vector_store.search(
            collection_name='papers',
            query_embedding=query_embedding,
            top_k=top_k * 3
        )

        # 按文件聚合结果（去重）
        aggregated = self._aggregate_results(results, top_k)

        return aggregated

    def _aggregate_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        聚合搜索结果：同一论文的多个chunk合并

        Args:
            results: 原始搜索结果
            top_k: 最终返回数量

        Returns:
            聚合后的结果
        """
        # 按文件路径分组
        grouped = defaultdict(list)
        for r in results:
            file_path = r['metadata']['file_path']
            grouped[file_path].append(r)

        # 聚合
        aggregated = []
        for file_path, chunks in grouped.items():
            # 取最高分数
            best_score = max(c['score'] for c in chunks)
            # 收集所有相关页码
            pages = sorted(set(c['metadata']['page'] for c in chunks))
            # 取第一个chunk的元数据
            first_chunk = chunks[0]['metadata']

            aggregated.append({
                'file': file_path,
                'filename': first_chunk['filename'],
                'title': first_chunk.get('title', 'Unknown'),
                'authors': first_chunk.get('authors', ''),
                'abstract': first_chunk.get('abstract', ''),
                'topic': first_chunk['topic'],
                'score': best_score,
                'pages': pages,
                'best_page': pages[0] if pages else 1,
                'preview': chunks[0]['metadata']['text_snippet']
            })

        # 按分数排序并截取top_k
        aggregated.sort(key=lambda x: x['score'], reverse=True)
        return aggregated[:top_k]

    def classify_paper(self, text: str, candidate_topics: List[str]) -> str:
        """
        自动分类论文

        Args:
            text: 论文文本（通常是前几段）
            candidate_topics: 候选主题列表

        Returns:
            最佳匹配的主题
        """
        # 主题描述（用于语义匹配）
        topic_descriptions = {
            'CV': 'computer vision image recognition object detection image segmentation convolutional neural network ResNet VGG image classification visual',
            'NLP': 'natural language processing text understanding language model transformer BERT GPT sequence modeling machine translation sentiment analysis',
            'RL': 'reinforcement learning agent environment reward policy gradient Q-learning actor critic Markov decision process game playing'
        }

        paper_embedding = self.text_embedder.encode_single(text[:1000])

        max_similarity = -1
        best_topic = candidate_topics[0] if candidate_topics else 'Other'

        for topic in candidate_topics:
            if topic in topic_descriptions:
                topic_emb = self.text_embedder.encode_single(topic_descriptions[topic])

                # 计算余弦相似度
                similarity = np.dot(paper_embedding, topic_emb) / (
                    np.linalg.norm(paper_embedding) * np.linalg.norm(topic_emb)
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_topic = topic

        return best_topic

    def batch_organize(self, source_dir: str, topics: List[str]) -> dict:
        """
        批量整理文件夹

        Args:
            source_dir: 源文件夹路径
            topics: 候选主题列表

        Returns:
            处理结果统计
        """
        # 查找所有PDF文件
        pdf_files = glob.glob(os.path.join(source_dir, '*.pdf'))

        if not pdf_files:
            log_warning(f"未找到PDF文件: {source_dir}")
            return {'success': False, 'message': '未找到PDF文件'}

        log_info(f"找到 {len(pdf_files)} 个PDF文件")

        # 使用进度跟踪器
        tracker = ProgressTracker(
            total=len(pdf_files),
            task_name="批量整理论文"
        )

        results = []
        for pdf_path in pdf_files:
            result = self.add_paper(pdf_path, topics)
            tracker.update(
                success=result['success'],
                message=Path(pdf_path).name,
                details=result
            )
            results.append(result)

        tracker.close()
        tracker.print_summary()

        return {
            'success': True,
            'total': len(pdf_files),
            'results': results,
            'summary': tracker.get_summary()
        }

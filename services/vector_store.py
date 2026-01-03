import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional


class VectorStore:
    """
    向量数据库封装
    使用ChromaDB进行向量存储和检索
    """

    def __init__(self, persist_directory='./data/vector_db'):
        """
        初始化ChromaDB客户端

        Args:
            persist_directory: 数据持久化目录
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

    def create_collection(self, name: str, metadata: dict = None):
        """
        创建或获取collection

        Args:
            name: collection名称
            metadata: collection元数据

        Returns:
            ChromaDB collection对象
        """
        try:
            # 尝试获取已存在的collection
            collection = self.client.get_collection(name=name)
            print(f"Collection '{name}' 已存在，使用现有collection")
        except:
            # 创建新collection
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            print(f"创建新collection: '{name}'")

        return collection

    def get_or_create_collection(self, name: str):
        """获取或创建collection"""
        # 使用余弦相似度（cosine），返回值在-1到1之间
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦距离
        )

    def add_documents(self, collection_name: str, embeddings: np.ndarray,
                     metadatas: List[Dict], ids: List[str],
                     documents: List[str] = None):
        """
        添加文档到collection

        Args:
            collection_name: collection名称
            embeddings: 嵌入向量数组
            metadatas: 元数据列表
            ids: 文档ID列表
            documents: 可选的原始文档文本
        """
        collection = self.get_or_create_collection(collection_name)

        # 转换embeddings为列表格式
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings

        # 如果没有提供documents，使用空字符串
        if documents is None:
            documents = [""] * len(ids)

        # 批量添加
        collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids,
            documents=documents
        )

    def search(self, collection_name: str, query_embedding: np.ndarray,
              top_k: int = 5, where: dict = None) -> List[Dict]:
        """
        向量相似度搜索

        Args:
            collection_name: collection名称
            query_embedding: 查询向量
            top_k: 返回结果数量
            where: 元数据过滤条件

        Returns:
            搜索结果列表
        """
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            print(f"Collection '{collection_name}' 不存在")
            return []

        # 转换query_embedding为列表
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        # 格式化结果
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                # ChromaDB使用余弦距离时，distance范围是[0, 2]
                # 余弦相似度 = 1 - (余弦距离 / 2)，范围变为[0, 1]
                score = max(0, 1 - (distance / 2))

                formatted_results.append({
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': distance,
                    'score': score
                })

        return formatted_results

    def delete_document(self, collection_name: str, doc_id: str):
        """
        删除文档

        Args:
            collection_name: collection名称
            doc_id: 文档ID
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=[doc_id])
            print(f"已删除文档: {doc_id}")
        except Exception as e:
            print(f"删除文档失败: {e}")

    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        获取collection统计信息

        Args:
            collection_name: collection名称

        Returns:
            统计信息字典
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            return {
                'name': collection_name,
                'count': count,
                'exists': True
            }
        except:
            return {
                'name': collection_name,
                'count': 0,
                'exists': False
            }

    def delete_collection(self, collection_name: str):
        """
        删除整个collection

        Args:
            collection_name: collection名称
        """
        try:
            self.client.delete_collection(name=collection_name)
            print(f"已删除collection: {collection_name}")
        except Exception as e:
            print(f"删除collection失败: {e}")

    def list_collections(self) -> List[str]:
        """列出所有collection"""
        collections = self.client.list_collections()
        return [col.name for col in collections]

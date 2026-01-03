import re
import fitz  # PyMuPDF
from typing import List, Dict, Optional
from pathlib import Path


class MetadataExtractor:
    """
    论文元数据提取器
    提取标题、作者、摘要、年份等信息
    """

    def __init__(self):
        pass

    def extract_metadata(self, pdf_path: str, pages_text: List[Dict] = None) -> Dict:
        """
        提取论文元数据

        Args:
            pdf_path: PDF文件路径
            pages_text: 可选的预提取页面文本

        Returns:
            元数据字典
        """
        try:
            # 如果没有提供文本，先提取
            if pages_text is None:
                from utils.pdf_parser import extract_text_by_pages
                pages_text = extract_text_by_pages(pdf_path, use_ocr=False)

            # 获取前几页文本
            first_page = pages_text[0]['text'] if pages_text else ""
            full_text = '\n'.join([p['text'] for p in pages_text[:5]]) if len(pages_text) >= 5 else first_page

            # 提取各项元数据
            metadata = {
                'title': self.extract_title(first_page, pdf_path),
                'authors': self.extract_authors(first_page),
                'abstract': self.extract_abstract(full_text),
                'year': self.extract_year(first_page),
                'success': True
            }

            return metadata

        except Exception as e:
            print(f"元数据提取失败: {e}")
            return {
                'title': 'Unknown',
                'authors': [],
                'abstract': '',
                'year': None,
                'success': False
            }

    def extract_title(self, first_page_text: str, pdf_path: str = None) -> str:
        """
        提取论文标题

        Args:
            first_page_text: 首页文本
            pdf_path: PDF文件路径（用于提取元数据）

        Returns:
            标题字符串
        """
        # 方法1: 从PDF元数据获取
        if pdf_path:
            try:
                doc = fitz.open(pdf_path)
                title = doc.metadata.get('title', '').strip()
                doc.close()
                if title and len(title) > 10 and len(title) < 300:
                    # 验证标题不是arXiv编号或日期
                    if not re.search(r'arxiv:\d+\.\d+|^\d{4}$', title, re.IGNORECASE):
                        return title
            except:
                pass

        # 方法2: 从首页文本提取
        lines = first_page_text.strip().split('\n')
        # 过滤掉空行
        lines = [l.strip() for l in lines if l.strip()]

        if not lines:
            return "Unknown Title"

        # 取前20行中最合适的作为标题候选
        candidate_lines = []
        for i, line in enumerate(lines[:20]):
            # 过滤掉arXiv编号行（如 "arXiv:1409.3215v3  14 Dec 2014"）
            if re.search(r'arxiv:\d+\.\d+', line, re.IGNORECASE):
                continue

            # 过滤掉期刊信息行（如 "JournalofMachineLearningResearch21(2020)1-67"）
            if re.search(r'journal.*research|submitted|revised|published|proceedings|conference', line, re.IGNORECASE):
                continue

            # 过滤掉ORCID ID行（如 "Jie Hu[0000−0002−5150−1003]"）
            if re.search(r'\[\d{4}[-−]\d{4}[-−]\d{4}[-−]\d{4}\]', line):
                continue

            # 过滤掉包含日期格式的行（如 "Submitted1/20;Revised6/20;Published6/20"）
            if re.search(r'\d+/\d+;.*\d+/\d+', line):
                continue

            # 过滤掉可能是URL、邮箱、日期、版权声明的行
            if re.search(r'@|http|www|\.com|^\d{4}$|copyright|permission|provided|attribution|preprint', line, re.IGNORECASE):
                continue

            # 过滤掉纯日期行（如 "14 Dec 2014"）
            if re.search(r'^\d+\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$', line, re.IGNORECASE):
                continue

            # 过滤掉包含特殊符号的行（通常是作者或机构）
            if re.search(r'[∗†‡§¶]', line):
                continue

            # 过滤掉包含逗号的行（通常是作者列表）
            if ',' in line:
                continue

            # 检测是否是连写的作者名（如 "KaimingHe GeorgiaGkioxari"）
            # 特征：多个驼峰式单词连在一起
            words = line.split()
            camelcase_count = sum(1 for w in words if re.match(r'^[A-Z][a-z]+[A-Z]', w))
            # 如果有3个或以上的驼峰单词，可能是作者列表
            if camelcase_count >= 3:
                continue

            # 过滤掉全是驼峰式连写的行（如 "JournalofMachineLearningResearch"）
            # 特征：一个单词但包含多个大写字母
            if len(words) == 1 and sum(1 for c in line if c.isupper()) > 3:
                continue

            # 过滤掉以小写字母开头的行（通常是句子或摘要的一部分）
            if line and line[0].islower():
                continue

            # 过滤掉包含"our"、"we"、"this"等代词的行（通常是摘要）
            if re.search(r'\b(our|we|this|these|propose|present|introduce)\b', line, re.IGNORECASE):
                # 但如果行很短且在前3行，可能是标题的一部分，不过滤
                if not (len(line) < 50 and i < 3):
                    continue

            # 标题长度通常在15-200字符
            if 15 <= len(line) <= 200:
                # 计算标题得分：优先位置靠前、每个单词首字母大写
                title_case_words = sum(1 for w in words if w and w[0].isupper())
                # 标题得分 = 首字母大写的单词数 - 行号权重（越靠前越好）
                score = title_case_words - i * 0.15

                # 如果包含常见的标题关键词，增加分数
                title_keywords = ['learning', 'network', 'model', 'approach', 'method', 'system', 'deep', 'neural',
                                'attention', 'transformer', 'detection', 'recognition', 'classification', 'vision',
                                'policy', 'reinforcement', 'optimization', 'algorithms', 'actor', 'critic']
                if any(keyword in line.lower() for keyword in title_keywords):
                    score += 2

                # 如果行中所有单词都是首字母大写（标题格式），额外加分
                if all(w[0].isupper() for w in words if len(w) > 2):
                    score += 3

                # 如果行包含冒号（可能是副标题），略微加分
                if ':' in line:
                    score += 1

                # 如果位置在前3行，额外加分
                if i < 3:
                    score += 1

                candidate_lines.append((line, score, i))

        if candidate_lines:
            # 返回得分最高的一行
            best_candidate = max(candidate_lines, key=lambda x: x[1])
            return best_candidate[0]

        # 如果都不行，返回第一个长度合适的行
        for line in lines[:15]:
            if 15 < len(line) < 200 and not re.search(r'arxiv:\d+', line, re.IGNORECASE):
                return line

        return lines[0] if lines else "Unknown Title"

    def extract_authors(self, first_page_text: str) -> List[str]:
        """
        提取作者列表

        Args:
            first_page_text: 首页文本

        Returns:
            作者列表
        """
        authors = []

        # 先清理特殊符号（∗†‡§ 等）
        cleaned_text = re.sub(r'[∗†‡§¶]', '', first_page_text[:2000])

        # 方法1: 匹配连在一起的驼峰式人名（如 AshishVaswani NoamShazeer）
        # 匹配模式: 大写字母+小写字母+大写字母+小写字母
        camelcase_pattern = r'\b([A-Z][a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)*)\b'
        camelcase_matches = re.findall(camelcase_pattern, cleaned_text)

        # 方法2: 匹配正常的人名（如 John Doe）
        normal_pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        normal_matches = re.findall(normal_pattern, cleaned_text)

        # 合并两种匹配结果
        all_matches = camelcase_matches + normal_matches

        # 扩展的过滤词列表（包括常见标题词和技术术语）
        filter_words_list = [
            'abstract', 'introduction', 'keywords', 'university', 'computer science',
            'united states', 'google brain', 'googlebrain', 'googleresearch', 'google research',
            'attention is all you need', 'index terms', 'indexterms', 'towards real',
            'deep learning', 'neural network', 'convolutional', 'residual learning',
            'exploring the limits', 'unified text', 'unifiedtext', 'textto', 'texttransformer',
            'proximal policy', 'policy optimization', 'soft actor', 'maximum entropy',
            'stochastic actor', 'policy maximum'
        ]

        # 过滤词（用于部分匹配）
        filter_words = [
            'attention', 'all', 'you', 'need', 'learning', 'neural', 'network',
            'detection', 'proposal', 'region', 'object', 'deep', 'convolutional',
            'towards', 'real', 'time', 'index', 'terms', 'keyword', 'image',
            'super', 'resolution', 'biomedical', 'segmentation', 'using', 'based',
            'model', 'method', 'approach', 'system', 'framework', 'video',
            'classification', 'recognition', 'vision', 'transformer', 'transfer',
            'limits', 'exploring', 'unified', 'text', 'squeeze', 'excitation',
            'residual', 'networks', 'faster', 'region', 'proposal', 'mask',
            'generative', 'adversarial', 'going', 'deeper', 'convolutions',
            'worth', 'words', 'point', 'sets', 'rethinking', 'scaling',
            'proximal', 'policy', 'optimization', 'algorithms', 'soft', 'actor',
            'critic', 'maximum', 'entropy', 'stochastic', 'reinforcement'
        ]

        # 去重并限制数量
        seen = set()
        for match in all_matches[:40]:
            # 过滤掉一些常见的非人名词和标题（完全匹配）
            lower_match = match.lower()
            if lower_match in filter_words_list:
                continue

            # 过滤掉包含常见标题词、技术术语、关键词的（部分匹配）
            # 将驼峰式单词拆分后再检查
            # 例如 "UnifiedText" -> ["unified", "text"]
            words_in_match = re.findall(r'[A-Z][a-z]+', match)
            words_lower = [w.lower() for w in words_in_match] if words_in_match else [lower_match]

            # 如果任何拆分后的词在过滤列表中，跳过
            if any(word in filter_words for word in words_lower):
                continue

            # 过滤掉全部大写或全部小写的（通常不是人名）
            if match.isupper() or match.islower():
                continue

            # 过滤太短或太长的
            if not (5 < len(match) < 40):
                continue

            # 人名通常不会有太多连续大写字母（除了首字母）
            uppercase_count = sum(1 for c in match if c.isupper())
            if uppercase_count > len(match) * 0.5:  # 超过一半是大写，可能不是人名
                continue

            if match not in seen:
                authors.append(match)
                seen.add(match)

            if len(authors) >= 6:  # 最多6个作者
                break

        return authors if authors else ["Unknown"]

    def extract_abstract(self, full_text: str) -> str:
        """
        提取摘要

        Args:
            full_text: 完整文本（前几页）

        Returns:
            摘要文本
        """
        # 查找Abstract关键词后的段落
        patterns = [
            r'Abstract\s*[:：]?\s*(.*?)(?=\n\s*(?:1\.|Introduction|Keywords|©|\d+\s+[A-Z]))',
            r'ABSTRACT\s*[:：]?\s*(.*?)(?=\n\s*(?:1\.|INTRODUCTION|KEYWORDS))',
            r'摘\s*要\s*[:：]?\s*(.*?)(?=\n\s*(?:关键词|1\.|引言))'
        ]

        for pattern in patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # 清理并限制长度
                abstract = ' '.join(abstract.split())  # 移除多余空格
                # 限制长度
                if len(abstract) > 500:
                    abstract = abstract[:500] + '...'
                if len(abstract) > 50:  # 确保提取到的内容足够长
                    return abstract

        return ""

    def extract_year(self, text: str) -> Optional[int]:
        """
        提取发表年份

        Args:
            text: 文本

        Returns:
            年份（整数）或None
        """
        # 在前1500字符中查找年份（19xx或20xx）
        matches = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', text[:1500])

        if matches:
            years = [int(m) for m in matches]
            # 返回最近的合理年份
            valid_years = [y for y in years if 1990 <= y <= 2025]
            return max(valid_years) if valid_years else None

        return None

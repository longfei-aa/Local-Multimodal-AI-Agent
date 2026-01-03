import pdfplumber
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict
from tqdm import tqdm
from pathlib import Path


def extract_text_by_pages(pdf_path: str, use_ocr: bool = True) -> List[Dict]:
    """
    按页提取PDF文本，支持OCR fallback

    Args:
        pdf_path: PDF文件路径
        use_ocr: 是否启用OCR fallback

    Returns:
        页面数据列表: [{'page': 1, 'text': '...', 'method': 'pdfplumber'}, ...]
    """
    # 首先尝试使用pdfplumber
    try:
        pages_data = extract_with_pdfplumber(pdf_path)

        # 检查是否为扫描版PDF
        if use_ocr and is_scanned_pdf_pages(pages_data):
            print(f"检测到扫描版PDF，切换到OCR模式...")
            pages_data = extract_with_ocr(pdf_path)

        return pages_data

    except Exception as e:
        print(f"pdfplumber处理失败: {e}")
        if use_ocr:
            print("尝试使用OCR处理...")
            return extract_with_ocr(pdf_path)
        else:
            raise


def extract_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """
    使用pdfplumber提取文本（优先方法）

    Args:
        pdf_path: PDF文件路径

    Returns:
        页面数据列表
    """
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages_data.append({
                'page': page_num,
                'text': text,
                'method': 'pdfplumber'
            })

    return pages_data


def extract_with_ocr(pdf_path: str) -> List[Dict]:
    """
    使用OCR提取文本（fallback方案）

    Args:
        pdf_path: PDF文件路径

    Returns:
        页面数据列表
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("需要安装PaddleOCR: pip install paddleocr")

    # 初始化OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    doc = fitz.open(pdf_path)
    results = []

    # 使用进度条
    for page_num in tqdm(range(len(doc)), desc="OCR处理中"):
        page = doc[page_num]

        # 转换为高分辨率图像
        mat = fitz.Matrix(2, 2)  # 2倍放大以提高OCR质量
        pix = page.get_pixmap(matrix=mat)

        # 转换为numpy数组
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.height, pix.width, pix.n)

        # 如果是RGBA，转换为RGB
        if pix.n == 4:
            img_array = img_array[:, :, :3]

        # OCR识别
        try:
            result = ocr.ocr(img_array, cls=True)
            if result and result[0]:
                text = '\n'.join([line[1][0] for line in result[0]])
            else:
                text = ""
        except Exception as e:
            print(f"OCR识别第{page_num + 1}页失败: {e}")
            text = ""

        results.append({
            'page': page_num + 1,
            'text': text,
            'method': 'ocr'
        })

    doc.close()
    return results


def is_scanned_pdf_pages(pages_data: List[Dict], threshold: int = 100) -> bool:
    """
    检测是否为扫描版PDF

    Args:
        pages_data: 页面数据列表
        threshold: 判断阈值（平均每页字符数）

    Returns:
        是否为扫描版
    """
    if not pages_data:
        return True

    # 检查前3页的字符数
    total_chars = 0
    check_pages = min(3, len(pages_data))

    for i in range(check_pages):
        text = pages_data[i]['text'].strip()
        total_chars += len(text)

    avg_chars = total_chars / check_pages
    return avg_chars < threshold


def chunk_text_with_pages(pages_data: List[Dict],
                          chunk_size: int = 512,
                          overlap: int = 50) -> List[Dict]:
    """
    将长文本分块，保留页码信息

    Args:
        pages_data: 页面数据列表
        chunk_size: 每个chunk的词数
        overlap: 重叠词数

    Returns:
        分块列表: [{'text': '...', 'page': 1, 'chunk_id': 0}, ...]
    """
    chunks = []
    chunk_id = 0

    for page_data in pages_data:
        page_num = page_data['page']
        text = page_data['text']
        method = page_data.get('method', 'pdfplumber')

        # 简单的按词分割
        words = text.split()

        # 如果页面文本太短，直接作为一个chunk
        if len(words) < 50:
            if text.strip():
                chunks.append({
                    'text': text,
                    'page': page_num,
                    'chunk_id': chunk_id,
                    'method': method
                })
                chunk_id += 1
            continue

        # 滑动窗口分块
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            if len(chunk_text.strip()) > 50:  # 跳过过短的chunk
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'chunk_id': chunk_id,
                    'method': method
                })
                chunk_id += 1

    return chunks


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = False) -> str:
    """
    提取PDF完整文本（简化版本）

    Args:
        pdf_path: PDF文件路径
        use_ocr: 是否使用OCR

    Returns:
        完整文本
    """
    pages_data = extract_text_by_pages(pdf_path, use_ocr)
    full_text = '\n\n'.join([page['text'] for page in pages_data])
    return full_text

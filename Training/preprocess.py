# from datasets import load_dataset
# from py_vncorenlp import VnCoreNLP
# import os

# # Tạo thư mục lưu model VnCoreNLP nếu chưa có
# os.makedirs("vncorenlp", exist_ok=True)

# # Khởi tạo VnCoreNLP
# segmenter = VnCoreNLP(save_dir="vncorenlp", annotators=["wseg"])

# def preprocess_dataset():
#     # Tải dataset từ HuggingFace
#     dataset = load_dataset("sonlam1102/vihsd")
    
#     # Hàm tách từ
#     def word_segment(batch):
#         segmented_texts = []
#         for text in batch["text"]:
#             # VnCoreNLP trả về list các câu đã tách từ
#             segmented = segmenter.word_segment(text)
#             # Ghép các câu lại thành 1 đoạn
#             segmented_texts.append(" ".join(segmented))
#         return {"text": segmented_texts}
    
#     # Áp dụng tách từ cho toàn bộ dataset
#     dataset = dataset.map(word_segment, batched=True)
    
#     # Lưu dataset đã xử lý
#     dataset.save_to_disk("vihsd_preprocessed")
    
#     print("Tiền xử lý hoàn tất! Dataset đã lưu tại: vihsd_preprocessed")
#     return dataset

# if __name__ == "__main__":
#     preprocess_dataset()
  #############################################################  
# from datasets import load_dataset
# from py_vncorenlp import VnCoreNLP
# import os
# import logging

# # Cấu hình logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def setup_vncorenlp():
#     """Khởi tạo VnCoreNLP từ thư mục local"""
#     model_dir = os.path.abspath("vncorenlp")
    
#     # Kiểm tra file jar
#     if not os.path.exists(os.path.join(model_dir, "VnCoreNLP-1.2.jar")):
#         raise FileNotFoundError("Không tìm thấy file VnCoreNLP-1.2.jar")
    
#     logger.info("Khởi tạo VnCoreNLP...")
#     return VnCoreNLP(
#         save_dir=model_dir,
#         annotators=["wseg"],
#         max_heap_size="-Xmx2g"
#     )

# def preprocess_dataset():
#     try:
#         # 1. Khởi tạo segmenter
#         segmenter = setup_vncorenlp()
        
#         # 2. Tải dataset
#         logger.info("Đang tải dataset...")
#         dataset = load_dataset("sonlam1102/vihsd")
        
#         # 3. Kiểm tra cấu trúc dataset
#         if "free_text" not in dataset["train"].features:
#             raise ValueError("Dataset không có trường 'free_text'")
        
#         # 4. Hàm xử lý batch
#         def process_batch(batch):
#             texts = batch["free_text"] if isinstance(batch, dict) else batch
#             segmented_texts = []
            
#             for free_text in texts:
#                 try:
#                     # Xử lý từng văn bản
#                     if not isinstance(free_text, str):
#                         free_text = str(free_text)
#                     output = segmenter.word_segment(free_text)
#                     segmented_texts.append(" ".join(output[0]))  # Lấy câu đầu tiên
#                 except Exception as e:
#                     logger.warning(f"Lỗi khi xử lý free_text: {e}")
#                     segmented_texts.append(free_text)  # Giữ nguyên nếu có lỗi
            
#             # return {"text": segmented_texts}
#             batch["free_text"] = segmented_texts
        
#         # 5. Áp dụng xử lý
#         logger.info("Bắt đầu tiền xử lý...")
#         processed_dataset = dataset.map(
#             lambda batch: process_batch(batch),
#             batched=True,
#             batch_size=100,
#             remove_columns=[col for col in dataset["train"].column_names if col not in ["free_text", "label_id"]]

#         )
        
#         # 6. Lưu kết quả
#         output_dir = "vihsd_preprocessed"
#         processed_dataset.save_to_disk(output_dir)
#         logger.info(f"Tiền xử lý hoàn tất! Đã lưu tại: {output_dir}")
        
#         return processed_dataset
        
#     except Exception as e:
#         logger.error(f"Lỗi chính trong quá trình xử lý: {e}")
#         raise

# if __name__ == "__main__":
#     preprocess_dataset()
############################################################################################
from datasets import load_dataset
import os
import logging
import re
from functools import partial

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_vncorenlp():
    """Khởi tạo VnCoreNLP trong mỗi process riêng"""
    from py_vncorenlp import VnCoreNLP
    model_dir = os.path.abspath("./VnCoreNLP")
    jar_path = os.path.join(model_dir, "VnCoreNLP-1.2.jar")
    
    if not os.path.exists(jar_path):
        raise FileNotFoundError(f"Không tìm thấy VnCoreNLP jar tại: {jar_path}")
    return VnCoreNLP(
        save_dir=model_dir,
        annotators=["wseg"],
        max_heap_size="-Xmx2g"
    )

def process_text(text, segmenter=None):
    """Xử lý từng văn bản - Phiên bản đã sửa"""
    try:
        # Xử lý trường hợp text là None
        if text is None:
            return ""
            
        # Khởi tạo segmenter trong mỗi process nếu chưa có
        if segmenter is None:
            segmenter = setup_vncorenlp()
            
        # Chuẩn hóa từ toxic
        replacements = {
            r'\b(d[iịj]t)\b': 'địt',
            r'\b(clmm?)\b': 'cặc',
            r'\b(vl|vcl)\b': 'vãi lồn',
            r'\b(dm|dmn)\b': 'địt mẹ'
        }
        text = text.lower()
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Phân đoạn từ
        output = segmenter.word_segment(text)
        return " ".join(output[0]) if output else text
        
    except Exception as e:
        logger.warning(f"Lỗi xử lý văn bản: {str(e)}")
        return str(text)[:500]  # Đảm bảo luôn trả về string

def preprocess_dataset():
    try:
        # 1. Tải dataset
        logger.info("Đang tải dataset...")
        dataset = load_dataset("sonlam1102/vihsd")
        
        # 2. Kiểm tra cấu trúc dataset
        if "free_text" not in dataset["train"].features:
            raise ValueError("Dataset thiếu trường 'free_text'")

        # 3. Xử lý với multiprocessing
        logger.info("Bắt đầu tiền xử lý...")
        
        processed_dataset = dataset.map(
            lambda batch: {
                "free_text": [process_text(text) for text in batch["free_text"]]
            },
            batched=True,
            batch_size=256,
            num_proc=4,
            remove_columns=[col for col in dataset["train"].column_names
                if col not in ["free_text", "label_id"]]
        )

        # 4. Lọc dữ liệu không hợp lệ
        logger.info("Đang lọc dữ liệu...")
        processed_dataset = processed_dataset.filter(
            lambda x: x["free_text"] is not None  # Thêm điều kiện kiểm tra None
            and len(x["free_text"]) > 3 
            and x["label_id"] in {0, 1, 2}
        )

        # 5. Xử lý các giá trị null còn sót lại
        processed_dataset = processed_dataset.map(
            lambda x: {"free_text": x["free_text"] or ""},
            num_proc=4
        )

        # 6. Lưu kết quả
        output_dir = "vihsd_preprocessed"
        processed_dataset.save_to_disk(output_dir)
        logger.info(f" Tiền xử lý hoàn tất! Đã lưu tại: {output_dir}")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f" Lỗi tiền xử lý: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess_dataset()
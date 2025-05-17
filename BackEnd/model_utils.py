# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from vncorenlp import VnCoreNLP
# import torch

# # Khởi tạo VnCoreNLP
# from vncorenlp import VnCoreNLP
# # Tải model nếu chưa có (chỉ cần chạy 1 lần)
# # VnCoreNLP.download_model(save_dir='vncorenlp')
# segmenter = VnCoreNLP(address="http://127.0.0.1", port=9000) 

# # Load model đã huấn luyện
# model_dir = "phobert_vihsd_model"
# tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# model.eval()

# # Danh sách từ độc hại (có thể mở rộng)
# TOXIC_WORDS = {
#     "vãi", "đụ", "lồn", "địt", "cặc", "buồi", "đéo", "đếch", "đỉ", "đĩ", 
#     "chó", "súc vật", "thằng", "con", "ngu", "ngu ngốc", "óc chó", 
#     "khốn nạn", "đồ khốn", "đồ ngu", "đồ đần", "đồ chó", "đồ súc sinh",
#     "điên", "thần kinh", "mất dạy", "vô giáo dục", "rác rưởi"
# }

# def classify_text(text):
#     try:
#         # Tách từ tiếng Việt
#         segmented = segmenter.tokenize(text)
#         text_seg = " ".join(segmented[0])
        
#         # Tokenize và dự đoán
#         inputs = tokenizer(text_seg, return_tensors="pt", truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # Xử lý kết quả
#         scores = torch.softmax(outputs.logits, dim=-1).tolist()[0]
#         labels = ["CLEAN", "OFFENSIVE", "HATE"]
#         best_idx = int(outputs.logits.argmax(axis=-1))
        
#         # Tìm các từ độc hại
#         toxic_spans = []
#         for word in segmented[0]:
#             lower_word = word.lower()
#             for toxic_word in TOXIC_WORDS:
#                 if toxic_word in lower_word and word not in toxic_spans:
#                     toxic_spans.append(word)
        
#         return {
#             "label": labels[best_idx],
#             "score": float(scores[best_idx]),
#             "spans": toxic_spans,
#             "segmented_text": " ".join(segmented[0])
#         }
#     except Exception as e:
#         return {"error": str(e)}

#####################################################################################################

#########ver2


# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from py_vncorenlp import VnCoreNLP
# import torch
# import os

# # Khởi tạo VnCoreNLP - sử dụng chung model từ thư mục training
# VNCORENLP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../training/VnCoreNLP"))

# def initialize_components():
#     """Khởi tạo tất cả các thành phần cần thiết"""
#     # 1. Kiểm tra và khởi tạo VnCoreNLP
#     try:
#         if not os.path.exists(VNCORENLP_DIR):
#             raise FileNotFoundError(f"Không tìm thấy thư mục VnCoreNLP tại: {VNCORENLP_DIR}")
        
#         required_files = [
#             os.path.join(VNCORENLP_DIR, "VnCoreNLP-1.2.jar"),
#             os.path.join(VNCORENLP_DIR, "models", "wordsegmenter", "wordsegmenter.rdr")
#         ]
        
#         for f in required_files:
#             if not os.path.exists(f):
#                 raise FileNotFoundError(f"Thiếu file quan trọng: {f}")
        
#         segmenter = VnCoreNLP(save_dir=VNCORENLP_DIR, annotators=["wseg"])
#     except Exception as e:
#         raise RuntimeError(f"Lỗi khởi tạo VnCoreNLP: {str(e)}")

#     # 2. Khởi tạo PhoBERT model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     MODEL_PATH = os.path.join(os.path.dirname(__file__), "../training/phobert_vihsd_model/best_model")
    
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#         model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
#         model.eval()
#     except Exception as e:
#         raise RuntimeError(f"Lỗi khởi tạo PhoBERT: {str(e)}")
    
#     return segmenter, tokenizer, model, device

# # Khởi tạo các thành phần
# try:
#     segmenter, tokenizer, model, device = initialize_components()
# except Exception as e:
#     print(f"Lỗi khởi tạo hệ thống: {e}")
#     exit(1)

# def predict(text):
#     """Dự đoán loại văn bản"""
#     try:
#         # 1. Tiền xử lý văn bản
#         if not isinstance(text, str) or not text.strip():
#             return {"error": "Văn bản đầu vào không hợp lệ"}
            
#         # 2. Tách từ
#         segmented = segmenter.word_segment(text)
#         text_seg = " ".join(segmented[0]) if segmented else text
        
#         # 3. Tokenize
#         inputs = tokenizer(
#             text_seg,
#             return_tensors="pt",
#             truncation=True,
#             max_length=128,
#             padding=True
#         ).to(device)
        
#         # 4. Dự đoán
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # 5. Xử lý kết quả
#         probs = torch.softmax(outputs.logits, dim=-1).squeeze()
#         labels = ["CLEAN", "OFFENSIVE", "HATE"]
        
#         return {
#             "label": labels[probs.argmax().item()],
#             "confidence": round(probs.max().item(), 4),
#             "probabilities": {label: round(prob.item(), 4) for label, prob in zip(labels, probs)},
#             "segmented_text": text_seg
#         }
        
        
        
#     except Exception as e:
#         return {"error": f"Lỗi khi dự đoán: {str(e)}"}


###########################################################################################################
######ver3

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from py_vncorenlp import VnCoreNLP
import torch
import os

# Tắt cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['JAVA_TOOL_OPTIONS'] = '-Dfile.encoding=UTF8'

class TextClassifier:
    def __init__(self):
        """Khởi tạo tất cả components"""
        self.vncorenlp_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../training/VnCoreNLP"))
        self.phobert_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../training/phobert_vihsd_model/best_model"))
        self.segmenter, self.tokenizer, self.model, self.device = self._initialize_components()
    
    def _initialize_components(self):
        """Khởi tạo các thành phần xử lý ngôn ngữ"""
        # 1. Kiểm tra đường dẫn
        self._check_paths()
        
        # 2. Khởi tạo VnCoreNLP
        segmenter = VnCoreNLP(
            save_dir=self.vncorenlp_dir,
            annotators=["wseg"]
        )
        
        # 3. Khởi tạo PhoBERT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sử dụng đường dẫn local với from_pretrained (đã sửa tham số)
        # tokenizer = AutoTokenizer.from_pretrained(self.phobert_dir, local_files_only=True)
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     self.phobert_dir,
        #     local_files_only=True
        # ).to(device)
        # model.eval()
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")  # Luôn dùng tokenizer gốc
        model = AutoModelForSequenceClassification.from_pretrained(
        self.phobert_dir  # Chỉ model dùng local
        ).to(device)
        model.eval()
        return segmenter, tokenizer, model, device
    
    def _check_paths(self):
        """Kiểm tra đường dẫn model"""
        required_files = [
            (self.vncorenlp_dir, ["VnCoreNLP-1.2.jar", os.path.join("models", "wordsegmenter", "wordsegmenter.rdr")]),
            (self.phobert_dir, ["config.json", "model.safetensors"])  # Hoặc "pytorch_model.bin"
        ]
        
        for base_dir, files in required_files:
            if not os.path.exists(base_dir):
                raise FileNotFoundError(f"Không tìm thấy thư mục: {base_dir}")
            
            for file in files:
                if not os.path.exists(os.path.join(base_dir, file)):
                    raise FileNotFoundError(f"Không tìm thấy file: {os.path.join(base_dir, file)}")
    
    def predict(self, text):
        """Dự đoán loại văn bản"""
        try:
            if not isinstance(text, str) or not text.strip():
                return {"error": "Văn bản đầu vào không hợp lệ"}
                
            # Tiền xử lý
            segmented = self.segmenter.word_segment(text)
            text_seg = " ".join(segmented[0]) if segmented else text
            
            # Tokenize
            inputs = self.tokenizer(
                text_seg,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(self.device)
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Xử lý kết quả
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()
            labels = ["CLEAN","OFFENSIVE","HATE"]
            
            return {
                "text": text,
                "segmented": text_seg,
                "label": labels[probs.argmax().item()],
                "confidence": round(probs.max().item(), 4),
                "probabilities": {l: round(p.item(), 4) for l, p in zip(labels, probs)}
            }
                
        except Exception as e:
            return {"error": str(e), "input_text": text}

# Khởi tạo instance
try:
    classifier = TextClassifier()
    print(" Hệ thống khởi tạo thành công")
except Exception as e:
    print(f" Lỗi khởi tạo: {e}")
    classifier = None
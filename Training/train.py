# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     Trainer,
#     TrainingArguments,
# )
# from datasets import load_from_disk
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
# import torch
# import logging

# # Cấu hình logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     acc = accuracy_score(labels, preds)
#     f1 = f1_score(labels, preds, average="weighted")
#     return {"accuracy": acc, "f1_score": f1}

# def clean_text(free_text):
#     """Chuẩn hóa dữ liệu đầu vào"""
#     if isinstance(free_text, str):
#         return free_text.strip()
#     elif isinstance(free_text, (list, tuple)):
#         return [str(x).strip() for x in free_text]
#     else:
#         return str(free_text).strip()

# def main():
#     try:
#         ##. Kiểm tra GPU
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         logger.info(f"Đang sử dụng thiết bị: {device}")
#         # 1. Tải dataset đã tiền xử lý
#         logger.info("Đang tải dataset đã tiền xử lý...")
#         dataset = load_from_disk("VnCoreNLP/vihsd_preprocessed")
        
#         # 2. Kiểm tra cấu trúc dataset
#         required_columns = {"free_text", "label_id"}
#         if not required_columns.issubset(dataset["train"].features):
#             missing = required_columns - set(dataset["train"].features)
#             raise ValueError(f"Dataset thiếu các cột: {missing}")
        
#         # 3. Tải tokenizer và model
#         logger.info("Đang tải tokenizer và model...")
#         model_name = "vinai/phobert-base"
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
        
#         # 4. Tokenize dataset với xử lý lỗi
#         def tokenize_function(batch):
#             # Làm sạch dữ liệu đầu vào
#             texts = batch["free_text"]
#             cleaned_texts = []
            
#             for free_text in texts:
#                 try:
#                     cleaned = clean_text(free_text)
#                     if not cleaned:  # Nếu free_text rỗng
#                         cleaned = "[EMPTY]"
#                     cleaned_texts.append(cleaned)
#                 except Exception as e:
#                     logger.warning(f"Lỗi khi làm sạch free_text: {free_text[:50]}... Error: {e}")
#                     cleaned_texts.append("[ERROR]")
            
#             # Tokenize với xử lý lỗi
#             try:
#                 tokenized = tokenizer(
#                     cleaned_texts,
#                     padding="max_length",
#                     truncation=True,
#                     max_length=128,
#                     return_tensors="pt"
#                 )
#                 # Thêm labels
#                 tokenized["labels"] = batch["label_id"]
#                 return tokenized
#             except Exception as e:
#                 logger.error(f"Lỗi khi tokenize batch: {e}")
#                 raise
        
#         logger.info("Đang tokenize dataset...")
#         tokenized_dataset = dataset.map(
#             tokenize_function,
#             batched=True,
#             batch_size=1000,
#             remove_columns=["free_text"],
#             num_proc=4  # Sử dụng đa tiến trình
#         )
        
#         # 5. Lọc các samples bị lỗi
#         original_size = len(tokenized_dataset["train"])
#         tokenized_dataset = tokenized_dataset.filter(
#             lambda x: x["input_ids"] is not None
#         )
#         filtered_size = len(tokenized_dataset["train"])
        
#         if filtered_size < original_size:
#             logger.warning(f"Đã lọc {original_size - filtered_size} samples bị lỗi")
        
#         # 6. Định dạng dataset cho PyTorch
#         tokenized_dataset.set_format(
#             type="torch",
#             columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
#         )
        
#         # 7. Khởi tạo model
#         logger.info("Đang khởi tạo model...")
#         model = AutoModelForSequenceClassification.from_pretrained(
#             model_name,
#             num_labels=3,
#             ignore_mismatched_sizes=True
#         )
        
#         # 8. Thiết lập training
#         # training_args = TrainingArguments(
#         #     output_dir="phobert_vihsd_model",
#         #     num_train_epochs=3,
#         #     per_device_train_batch_size=16,
#         #     per_device_eval_batch_size=32,
#         #     evaluation_strategy="epoch",
#         #     save_strategy="epoch",
#         #     logging_dir="logs",
#         #     load_best_model_at_end=True,
#         #     metric_for_best_model="f1_score",
#         #     save_total_limit=2,
#         #     logging_steps=100,
#         #     fp16=True,  # Sử dụng mixed precision nếu có GPU
#         # )
#         training_args = TrainingArguments(
#         output_dir="./phobert_vihsd_model",
#         num_train_epochs=3,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=32,
#         save_steps=500,
#         save_total_limit=2,
#         logging_dir="./logs",
#         logging_steps=100,
#         fp16=torch.cuda.is_available(),
#         # evaluation_strategy="steps",  # Bỏ dòng này
#         # eval_steps=500,  # Bỏ dòng này
#         )
        
#         # 9. Khởi tạo Trainer
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=tokenized_dataset["train"],
#             eval_dataset=tokenized_dataset["validation"],
#             compute_metrics=compute_metrics,
#         )
        
#         # 10. Huấn luyện
#         logger.info("Bắt đầu huấn luyện...")
#         trainer.train()
        
#         # 11. Lưu model
#         logger.info("Đang lưu model...")
#         trainer.save_model("phobert_vihsd_model/best_model")
#         tokenizer.save_pretrained("phobert_vihsd_model/best_model")
        
#         logger.info(f"Huấn luyện hoàn tất! Model đã lưu tại: phobert_vihsd_model/best_model")
#         logger.info(f"Kết quả đánh giá: {trainer.evaluate()}")
        
#     except Exception as e:
#         logger.error(f"Lỗi trong quá trình huấn luyện: {e}")
#         raise

# if __name__ == "__main__":
#     main()
    
    
    ####################################################################################################

    
    
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import logging
from torch.nn import CrossEntropyLoss
from collections import Counter

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cấu hình phần cứng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Đang sử dụng thiết bị: {device}")
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # Tên GPU



# Cân bằng lớp
CLASS_WEIGHTS = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32).to(device)  # Đã sửa thành tensor

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            available_keys = ", ".join(inputs.keys())
            raise ValueError(f"Không tìm thấy labels trong inputs. Các key có sẵn: {available_keys}")
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Sử dụng trực tiếp CLASS_WEIGHTS đã được định nghĩa là tensor
        loss_fct = CrossEntropyLoss(weight=CLASS_WEIGHTS)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro")
    }
    
    logger.info("\n" + classification_report(labels, preds, target_names=["CLEAN", "OFFENSIVE", "HATE"]))
    return metrics

def main():
    try:
        # 1. Tải dataset
        logger.info(" Đang tải dataset...")
        dataset = load_from_disk("VnCoreNLP/vihsd_preprocessed")
        
        # 2. Kiểm tra phân phối nhãn
        logger.info(" Phân phối nhãn tập train: %s", Counter(dataset["train"]["label_id"]))
        logger.info(" Phân phối nhãn tập validation: %s", Counter(dataset["validation"]["label_id"]))
        
        # 3. Tải tokenizer
        logger.info(" Đang tải tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        # 4. Tokenize dataset
        def tokenize_function(batch):
            texts = []
            for text in batch["free_text"]:
                if text is None:
                    text = ""
                elif not isinstance(text, str):
                    text = str(text)
                texts.append(text)
            
            return tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
        logger.info(" Đang tokenize dữ liệu...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=512,
            num_proc=4,
            remove_columns=["free_text"]
        )
        
        # 5. Đổi tên cột và định dạng
        tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")
        tokenized_dataset.set_format(
            "torch",
            columns=["input_ids", "attention_mask", "token_type_ids", "labels"]
        )
        
        # 6. Khởi tạo model
        logger.info("Đang khởi tạo model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base",
            num_labels=3,
            id2label={0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
        ).to(device)
        
        # 7. Cấu hình training
        training_args = TrainingArguments(
    output_dir="./phobert_vihsd_model",
    num_train_epochs=5,  # Giữ nguyên nếu dữ liệu ổn
    per_device_train_batch_size=16,  # Giảm từ 32 -> 16 để ổn định hơn
    per_device_eval_batch_size=32,   # Giảm từ 64 -> 32
    eval_strategy="steps",
    eval_steps=250,  # Tăng tần suất đánh giá (500 -> 250)
    save_strategy="steps",
    save_steps=250,  # Lưu model thường xuyên hơn
    logging_dir="./logs",
    logging_steps=50,  # Log chi tiết hơn
    learning_rate=3e-5,  # Tăng từ 2e-5 -> 3e-5
    warmup_ratio=0.2,  # Tăng warmup
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # Thay f1_weighted bằng f1_macro
    remove_unused_columns=False,
    gradient_accumulation_steps=2,  # Thêm để ổn định training
    weight_decay=0.01,  # Thêm regularization
    
    logging_strategy="steps",
)
        
        # 8. Khởi tạo trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
        )
        
        # 9. Huấn luyện
        logger.info(" Bắt đầu huấn luyện...")
        trainer.train()
        
        # 10. Lưu model
        logger.info(" Đang lưu model...")
        trainer.save_model("./phobert_vihsd_model/best_model")
        tokenizer.save_pretrained("./phobert_vihsd_model/best_model")
        
        # 11. Đánh giá
        logger.info(" Kết quả đánh giá:")
        eval_results = trainer.evaluate()
        logger.info(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
        logger.info(f"Validation F1 (weighted): {eval_results['eval_f1_weighted']:.4f}")
        
    except Exception as e:
        logger.error(f" Lỗi huấn luyện: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
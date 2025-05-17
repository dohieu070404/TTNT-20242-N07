# """
# Toxicity Analysis API Server

# This Flask application provides an API endpoint for analyzing text toxicity using the Unitary Toxic-BERT model.
# It includes input validation, error handling, and proper logging mechanisms.

# Main Features:
# - Text toxicity analysis using transformer-based model
# - Input validation for text length and content
# - CORS support for cross-origin requests
# - Comprehensive error handling and logging
# - Configurable settings through Configuration class

# Author: Allan Niñal
# Date: 2024-12-12
# """

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import pipeline
# from typing import Dict, Any
# import logging

# # Configure logging with detailed format
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)

# class Configuration:
#     """
#     Application configuration settings.
    
#     Attributes:
#         TOXICITY_MODEL_NAME (str): The name/path of the pre-trained model to use
#         MAXIMUM_TEXT_LENGTH (int): Maximum allowed length for input text
#         ENABLE_DEBUG_MODE (bool): Flag to enable/disable Flask debug mode
#     """
#     TOXICITY_MODEL_NAME = "unitary/toxic-bert"
#     MAXIMUM_TEXT_LENGTH = 512  # Characters limit for input text
#     ENABLE_DEBUG_MODE = True   # Warning: Set to False in production

# # Initialize Flask application with CORS support
# app = Flask(__name__)
# CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# # Initialize the toxicity classification model
# try:
#     # Load the model using transformers pipeline
#     toxicity_classifier = pipeline("text-classification", model=Configuration.TOXICITY_MODEL_NAME)
#     logger.info(f"Model {Configuration.TOXICITY_MODEL_NAME} loaded successfully")
# except Exception as error:
#     # Log any errors during model initialization and re-raise
#     logger.error(f"Critical error: Failed to load model: {str(error)}")
#     raise

# def validate_input_text(input_text: str) -> tuple[bool, str]:
#     """
#     Validate the input text against defined criteria.
    
#     Args:
#         input_text (str): The text to be validated
        
#     Returns:
#         tuple[bool, str]: A tuple containing:
#             - bool: True if validation passes, False otherwise
#             - str: Error message if validation fails, empty string if passes
            
#     Validation Criteria:
#         1. Text must not be empty
#         2. Text length must not exceed MAXIMUM_TEXT_LENGTH
#     """
#     # Check for empty or whitespace-only input
#     if not input_text:
#         return False, "No text provided"
    
#     # Validate text length
#     if len(input_text) > Configuration.MAXIMUM_TEXT_LENGTH:
#         return False, f"Text exceeds maximum length of {Configuration.MAXIMUM_TEXT_LENGTH} characters"
    
#     return True, ""

# @app.route('/analyze', methods=['POST'])
# def analyze_text_toxicity() -> tuple[Dict[str, Any], int]:
#     """
#     API endpoint to analyze text toxicity.
    
#     Expected JSON Request Body:
#         {
#             "text": "string to analyze"
#         }
    
#     Returns:
#         tuple: Contains:
#             - JSON response with analysis results or error message
#             - HTTP status code
            
#     Response Format (Success):
#         {
#             "success": true,
#             "data": {
#                 "text": "original input text",
#                 "analysis": {
#                     "label": "toxic/non-toxic",
#                     "score": float
#                 },
#                 "model": "model name"
#             }
#         }
    
#     Response Format (Error):
#         {
#             "success": false,
#             "error": "error message",
#             "details": "detailed error information"
#         }
#     """
#     try:
#         # Extract and validate JSON payload
#         request_data = request.get_json()
#         if not request_data:
#             return jsonify({"error": "Invalid JSON payload"}), 400

#         # Extract and clean input text
#         input_text = request_data.get("text", "").strip()
        
#         # Validate input text
#         is_valid_input, validation_error = validate_input_text(input_text)
#         if not is_valid_input:
#             return jsonify({"error": validation_error}), 400

#         # Perform toxicity classification
#         classification_results = toxicity_classifier(input_text)
        
#         # Construct successful response
#         api_response = {
#             "success": True,
#             "data": {
#                 "text": input_text,
#                 "analysis": classification_results[0],  # First element contains classification results
#                 "model": Configuration.TOXICITY_MODEL_NAME
#             }
#         }
        
#         # Log successful analysis
#         logger.info(f"Successfully analyzed text: {input_text[:50]}...")
#         return jsonify(api_response), 200

#     except Exception as error:
#         # Log the full error details
#         logger.error(f"Error processing request: {str(error)}", exc_info=True)
        
#         # Return user-friendly error response
#         return jsonify({
#             "success": False,
#             "error": "Internal server error",
#             "details": str(error)
#         }), 500

# if __name__ == '__main__':
#     # Start the Flask application
#     logger.info(f"Starting application in {'debug' if Configuration.ENABLE_DEBUG_MODE else 'production'} mode")
#     app.run(debug=Configuration.ENABLE_DEBUG_MODE)


#########################################################################################################################
##ver2


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from model_utils import classifier  # Import từ file model_utils.py đã sửa
# import logging

# # Cấu hình logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)  # Cho phép CORS nếu frontend ở domain khác

# @app.route('/predict', methods=['POST'])
# def predict():
#     """API endpoint cho phân loại văn bản"""
#     try:
#         # 1. Nhận dữ liệu đầu vào
#         data = request.get_json()
#         text = data.get('text', '').strip()
        
#         # 2. Validate input
#         if not text:
#             logger.warning("Văn bản đầu vào rỗng")
#             return jsonify({"error": "Vui lòng nhập văn bản cần phân loại"}), 400
        
#         # 3. Gọi model dự đoán
#         logger.info(f"Đang xử lý văn bản: {text[:50]}...")  # Log 50 ký tự đầu
#         result = classifier.predict(text)
        
#         # 4. Xử lý kết quả trả về
#         if "error" in result:
#             logger.error(f"Lỗi khi dự đoán: {result['error']}")
#             return jsonify(result), 500
            
#         logger.info(f"Kết quả: {result['label']} (Độ tin cậy: {result['confidence']:.2f})")
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"Lỗi hệ thống: {str(e)}", exc_info=True)
#         return jsonify({
#             "error": "Lỗi server nội bộ",
#             "details": str(e)
#         }), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Endpoint kiểm tra tình trạng hệ thống"""
#     return jsonify({
#         "status": "healthy",
#         "model_loaded": True,
#         "ready": True
#     })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)  # Đặt debug=False khi chạy production


####################################################################################################
########Ver3

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import classifier
import logging
from typing import Dict, Union

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Bật CORS cho tất cả routes

@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Union[str, float, Dict]]:
    """API endpoint phân loại văn bản"""
    try:
        # Validate input
        if not request.is_json:
            return jsonify({"error": "Request phải là JSON"}), 400
            
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Văn bản không được để trống"}), 400
        
        # Kiểm tra model
        if classifier is None:
            logger.error("Model chưa được khởi tạo")
            return jsonify({"error": "Hệ thống đang bảo trì"}), 503
        
        # Xử lý và trả kết quả
        result = classifier.predict(text)
        
        if "error" in result:
            logger.error(f"Lỗi xử lý: {result['error']} | Input: '{text[:50]}...'")
            return jsonify(result), 500
            
        logger.info(
            f"Phân loại thành công | Kết quả: {result['label']} | "
            f"Độ tin cậy: {result['confidence']:.2f} | "
            f"Text: '{text[:30]}...'"
        )
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Lỗi server: {str(e)}", exc_info=True)
        return jsonify({"error": "Lỗi hệ thống"}), 500

@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, Union[str, bool]]:
    """Kiểm tra tình trạng hệ thống"""
    status = {
        "status": "running",
        "model_ready": False,
        "vncore_ready": False,
        "version": "1.0.0"
    }
    
    if classifier is not None:
        status.update({
            "model_ready": True,
            "vncore_ready": hasattr(classifier, 'segmenter') and classifier.segmenter is not None
        })
    
    return jsonify(status)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True  # Xử lý nhiều request đồng thời
    )
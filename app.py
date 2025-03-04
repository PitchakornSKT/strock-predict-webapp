from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# โหลดโมเดล
model = joblib.load("model/stroke_prediction_model.pkl")

# LINE API Credentials (ใส่ TOKEN ของคุณที่นี่)
LINE_CHANNEL_ACCESS_TOKEN = "117e61c03788c2884fbd08d78d2d0482"
LINE_CHANNEL_SECRET = "bb8a3f877011defb9ff8e07e5d65475e"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ตั้งค่า Logger เพื่อตรวจสอบค่าที่ส่งเข้าโมเดล
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([data['features']]).reshape(1, -1)
        
        # Log ค่าที่ส่งเข้าโมเดล
        app.logger.info(f"Received features: {features}")
        
        prediction = model.predict(features)[0]
        
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)})

# Webhook สำหรับ LINE Chatbot
@app.route('/webhook', methods=['POST']) 
def webhook():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400  

    return "OK", 200 


# ฟังก์ชันตอบกลับข้อความ
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    try:
        # แปลงข้อความเป็นฟีเจอร์
        features = text_to_features(user_text)
        
        # Log ค่าที่ส่งเข้าโมเดล
        app.logger.info(f"User input features: {features}")

        prediction = model.predict([features])[0]

        # แปลผลลัพธ์
        result_text = "มีความเสี่ยงเป็นโรคหลอดเลือดสมอง" if prediction == 1 else "ไม่มีความเสี่ยง"
    except Exception as e:
        result_text = f"เกิดข้อผิดพลาด: {str(e)}"
        app.logger.error(f"Error: {str(e)}")

    # ส่งข้อความกลับไปยัง LINE
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_text))

def text_to_features(user_text):
    """
    ฟังก์ชันแปลงข้อความจากผู้ใช้เป็นฟีเจอร์ที่ใช้พยากรณ์
    """
    try:
        features = list(map(float, user_text.split(",")))
        
        # ตรวจสอบว่าจำนวนฟีเจอร์ครบหรือไม่
        expected_features = 10  # ปรับตามจำนวนฟีเจอร์ที่โมเดลใช้
        if len(features) != expected_features:
            raise ValueError(f"จำนวนฟีเจอร์ไม่ถูกต้อง ({len(features)}/{expected_features}) กรุณาใส่ข้อมูลให้ครบ")

        return features
    except ValueError as e:
        raise ValueError(f"รูปแบบข้อมูลไม่ถูกต้อง: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

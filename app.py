import traceback
from flask import Flask, abort, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

model = joblib.load("model/stroke_prediction_model.pkl")

LINE_CHANNEL_ACCESS_TOKEN = "CdCARCAIynwrgbv5KL1X+8IqIjzkI77Jktx2DJ7ULuLuXh/N8V7XkF/1lfIfaRHwSjg3FZyiUfK5/QqUfuYfiZjl5f0DYHmky9MIuMXvkwDpKJqHA6cEm5so9VUrN17S8vk5ZklY99dqHMF40q2PbQdB04t89/1O/w1cDnyilFU="
LINE_CHANNEL_SECRET = "bb8a3f877011defb9ff8e07e5d65475e"

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

# Webhook Route
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

# Handle Message Event
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    try:
        features = text_to_features(user_text)
        app.logger.info(f"User input features: {features}")

        prediction = model.predict([features])[0]
        result_text = "มีความเสี่ยงเป็นโรคหลอดเลือดสมอง" if prediction == 1 else "ไม่มีความเสี่ยง"
    
    except Exception as e:
        result_text = f"เกิดข้อผิดพลาด: {str(e)}"
        app.logger.error(f"Error: {traceback.format_exc()}")

    try:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_text))
        app.logger.info(f"Replied to user: {result_text}")
    except Exception as e:
        app.logger.error(f"Failed to reply to user: {str(e)}")

def text_to_features(user_text):
    try:
        features = list(map(float, user_text.split(",")))
        if len(features) != 10:
            raise ValueError(f"จำนวนฟีเจอร์ไม่ถูกต้อง ({len(features)}/10) กรุณาใส่ข้อมูลให้ครบ")
        return features
    except ValueError as e:
        raise ValueError(f"รูปแบบข้อมูลไม่ถูกต้อง: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
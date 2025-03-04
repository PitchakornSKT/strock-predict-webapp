# การประเมินผลโมเดลการทำนายโรคหลอดเลือดสมอง

## 1. คำอธิบายข้อมูล (Data Description)
- **Dataset:** ชุดข้อมูล Healthcare สำหรับการทำนายโรคหลอดเลือดสมอง
- **คุณลักษณะ (Features):** เพศ, อายุ, โรคความดันโลหิตสูง, โรคหัวใจ, สถานะการแต่งงาน, สถานะการสูบบุหรี่ เป็นต้น
- **เป้าหมาย (Target):** โรคหลอดเลือดสมอง (1 = เป็นโรค, 0 = ไม่เป็นโรค)

## 2. การเตรียมข้อมูล (Data Preprocessing)
- จัดการข้อมูลที่หายไป (Missing values)
- แปลงข้อมูลที่เป็นประเภท (Categorical) เช่น `gender`, `work_type`, `smoking_status` เป็นค่าตัวเลข (Encoding)

## 3. การฝึกโมเดลและการประเมินผล (Model Training and Evaluation)
- โมเดลที่ฝึก ได้แก่: **Logistic Regression**, **kNN**, **Decision Tree**, **Random Forest**, **Gradient Boosting**, **AdaBoost**
- **ความแม่นยำ (Accuracy)** ของโมเดลแต่ละตัว:
  - Logistic Regression: 0.95
  - kNN: 0.94
  - Decision Tree: 0.92
  - Random Forest: 0.95
  - Gradient Boosting: 0.94
  - AdaBoost: 0.95

## 4. การใช้ Ensemble Learning
- ใช้ **Voting Classifier** แบบ **soft voting** เป็นเทคนิค Ensemble
- **ความแม่นยำ (Accuracy)** ของโมเดล Ensemble: 0.9470468431771895

## 5. การบันทึกโมเดล (Model Saving)
- โมเดลสุดท้ายที่ได้จากการใช้ **Voting Classifier** ได้ถูกบันทึกเป็นไฟล์ `stroke_prediction_model.pkl` ด้วยการใช้ `joblib`

## 6. สรุปผล (Conclusion)
- โมเดล Ensemble ให้ความแม่นยำสูงสุดที่ 86% เมื่อเปรียบเทียบกับโมเดลเดี่ยว

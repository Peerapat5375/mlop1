# ==========================================================
#  Base Image
# ==========================================================
FROM python:3.9-slim

# คำอธิบาย:
# - ใช้ python:3.9-slim เป็น base image
# - Slim version ช่วยลดขนาดลงแต่ยังมีฟีเจอร์เพียงพอสำหรับ MLflow/Sklearn


# ==========================================================
#  Set working directory
# ==========================================================
WORKDIR /app
# ทำให้ทุกคำสั่งถัดไปจะถูกรันใน path /app


# ==========================================================
#  Install system dependencies
# ==========================================================
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# เพิ่ม curl (ใช้กับ HEALTHCHECK)
# gcc จำเป็นสำหรับการ compile บาง package เช่น numpy, scipy


# ==========================================================
#  Install Python dependencies
# ==========================================================
COPY requirements.txt ./requirements.txt
COPY models/ ./models/

# ติดตั้ง dependencies ทั้งหมดจาก requirements.txt


# ==========================================================
#  Copy project code
# ==========================================================
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
# หากคุณมี processed_data/ หรือ mlruns/ ให้เพิ่ม COPY ตามนี้
# COPY processed_data/ ./processed_data/


# ==========================================================
#  Expose port
# ==========================================================
EXPOSE 5002
# Flask ของคุณใน app.py ใช้ port 5002


# ==========================================================
#  Environment variables
# ==========================================================
ENV FLASK_APP=api/app.py
ENV MODEL_URI=models:/cyberbullying-tweet-classifier@staging
ENV PYTHONUNBUFFERED=1

# หมายเหตุ:
# MODEL_URI = ใช้ MLflow Model Registry alias (@staging)
# PYTHONUNBUFFERED = ให้ Flask แสดง log ทันที ไม่เก็บบัฟเฟอร์ไว้


# ==========================================================
#  Health check
# ==========================================================
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5002/health || exit 1


# ==========================================================
#  Run application
# ==========================================================
CMD ["python", "api/app.py"]

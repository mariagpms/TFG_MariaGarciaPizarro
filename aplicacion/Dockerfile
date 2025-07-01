FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Asegurar que Python no buffer los outputs para ver logs inmediatamente
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]

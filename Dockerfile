    FROM python:3.13-slim

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install -r requirements.txt

    COPY . .

    EXPOSE 8080

    CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
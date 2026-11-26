# 1. Imagen base de Python
FROM python:3.14-slim

# 2. Directorio de trabajo
WORKDIR /app

# 3. Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar la aplicación, el modelo, los informes y el tema
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY .streamlit/ ./.streamlit/

# 5. Arrancar Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
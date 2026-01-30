FROM python:3.10
RUN pip install streamlit numpy
WORKDIR /app
ENTRYPOINT [ "streamlit", "run", "app.py" ]
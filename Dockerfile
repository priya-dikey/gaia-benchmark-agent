FROM python:3.9

# 2. Create a non-root user (required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 3. Set the working directory
WORKDIR $HOME/app

# 4. Install dependencies
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code
COPY --chown=user . $HOME/app

# 6. Start the application on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
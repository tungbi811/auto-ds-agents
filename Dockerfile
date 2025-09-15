FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl wget git vim nano \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    pandas numpy matplotlib seaborn \
    requests beautifulsoup4 ddgs \
    scikit-learn streamlit openai \
    python-dotenv crewai crewai-tools

# Create workspace
WORKDIR /app

# Security: non-root user
RUN useradd -m -s /bin/bash sandbox
USER sandbox

CMD ["/bin/bash"]
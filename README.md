# Mental Health Analysis API

A comprehensive API for analyzing mental health through text, audio, and video inputs. This system provides sentiment analysis, emotion detection, and personalized mental health recommendations.

## Features

- **Text Analysis**: Analyze journal entries and thoughts for sentiment, emotions, and mental health indicators
- **Audio Analysis**: Process voice recordings for emotional state and speech patterns
- **Video Analysis**: Analyze facial expressions and voice for comprehensive emotional assessment
- **Trend Analysis**: Track emotional and mental health trends over time
- **Personalized Interventions**: Generate customized mental health recommendations
- **Real-time Processing**: Fast and efficient analysis of user inputs
- **Secure Authentication**: OAuth2 with JWT token-based authentication
- **Scalable Architecture**: Built with FastAPI and PostgreSQL

## Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **Database**: PostgreSQL
- **Cache**: Redis
- **ML Models**: Transformers, PyTorch
- **Storage**: AWS S3
- **Deployment**: Docker, Kubernetes (optional)

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 13 or higher
- Redis 6 or higher
- FFmpeg
- AWS Account (for S3 storage)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mental-health-analysis-api.git
cd mental-health-analysis-api
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install system dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    python3-dev \
    build-essential \
    libpq-dev \
    redis-server \
    postgresql \
    postgresql-contrib

# Windows (using Chocolatey)
choco install ffmpeg
choco install redis-64
choco install postgresql
```

4. **Install Python dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. **Set up environment variables**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

6. **Initialize the database**
```bash
# Create database
createdb mental_health_db

# Run migrations
alembic upgrade head
```

7. **Download required models**
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader averaged_perceptron_tagger
python -m nltk.downloader wordnet
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mental_health_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys
HUGGINGFACE_API_KEY=your-huggingface-api-key
OPENAI_API_KEY=your-openai-api-key

# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=your-aws-region
S3_BUCKET_NAME=your-bucket-name

# Application Settings
MAX_FILE_SIZE=10485760  # 10MB in bytes
MAX_VIDEO_DURATION=300  # 5 minutes in seconds
MAX_AUDIO_DURATION=600  # 10 minutes in seconds
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

## Running the Application

### Development

```bash
# Start the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Build the Docker image
docker build -t mental-health-api .

# Run the container
docker run -d \
  --name mental-health-api \
  -p 8000:8000 \
  --env-file .env \
  mental-health-api
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=app tests/

# Run specific test file
pytest tests/test_analysis.py
```

## Deployment

### Docker Deployment

1. **Build the image**
```bash
docker build -t mental-health-api .
```

2. **Run the container**
```bash
docker run -d \
  --name mental-health-api \
  -p 8000:8000 \
  --env-file .env \
  mental-health-api
```

### Kubernetes Deployment

1. **Create namespace**
```bash
kubectl create namespace mental-health
```

2. **Apply configurations**
```bash
kubectl apply -f k8s/
```

## Monitoring

The application includes:
- Prometheus metrics endpoint
- Sentry error tracking
- Structured logging
- Health check endpoint

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, email support@example.com or open an issue in the GitHub repository.

## Acknowledgments

- FastAPI team for the amazing framework
- Hugging Face for the transformer models
- OpenCV for computer vision capabilities
- All contributors and maintainers 
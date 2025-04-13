# NeuroCare Backend

A comprehensive mental health analysis API that provides text, audio, and video analysis for mental well-being assessment.

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
- Git

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Tahadevv/NeuroCare-Backend.git
cd NeuroCare-Backend
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

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory with:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/mental_health_db
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
HUGGINGFACE_API_KEY=your-huggingface-api-key
```

5. **Initialize the database**
```bash
# Create database
createdb mental_health_db

# Run migrations
alembic upgrade head
```

6. **Start the server**
```bash
uvicorn main:app --reload
```

## API Endpoints

### Authentication

#### Sign Up
- **URL**: `/signup`
- **Method**: `POST`
- **Headers**: 
  - `Content-Type: application/json`
- **Body**:
```json
{
    "email": "user@example.com",
    "full_name": "John Doe",
    "password": "securepassword123"
}
```

#### Login
- **URL**: `/login`
- **Method**: `POST`
- **Headers**: 
  - `Content-Type: application/json`
- **Body**:
```json
{
    "email": "user@example.com",
    "password": "securepassword123"
}
```

### Text Analysis

#### Analyze Text
- **URL**: `/mental-health/analyze/text`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
  - `Content-Type: application/json`
- **Body**:
```json
{
    "content": "I've been feeling stressed at work lately. The deadlines are piling up and I'm having trouble sleeping.",
    "context": "Evening reflection"
}
```

### Audio Analysis

#### Analyze Audio
- **URL**: `/mental-health/analyze/audio`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Body**: 
  - `audio`: (binary) WAV file
  - `context`: (optional) string

### Video Analysis

#### Analyze Video
- **URL**: `/mental-health/analyze/video`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Body**: 
  - `video`: (binary) MP4 file
  - `context`: (optional) string

### Trend Analysis

#### Get Emotion Trends
- **URL**: `/mental-health/trends/emotions/{days}`
- **Method**: `GET`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Parameters**:
  - `days`: number of days (1-365)

#### Get Mental Health Trends
- **URL**: `/mental-health/trends/mental-health/{days}`
- **Method**: `GET`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
- **Parameters**:
  - `days`: number of days (1-365)

### User Profile

#### Create Profile
- **URL**: `/user/profile`
- **Method**: `POST`
- **Headers**: 
  - `Authorization: Bearer <access_token>`
  - `Content-Type: application/json`
- **Body**:
```json
{
    "name": "John Doe",
    "age": 30,
    "gender": "Male",
    "sleep_hours_actual": 7.5,
    "sleep_hours_target": 8.0,
    "goals": [
        "Reduce Stress & Anxiety",
        "Improve Sleep Quality"
    ]
}
```

#### Get Profile
- **URL**: `/user/profile`
- **Method**: `GET`
- **Headers**: 
  - `Authorization: Bearer <access_token>`

## Example Responses

### Text Analysis Response
```json
{
    "analysis_id": 1,
    "timestamp": "2024-03-15T10:30:00Z",
    "sentiment_analysis": {
        "sentiment": "NEGATIVE",
        "score": 0.85,
        "emotions": {
            "stress": 0.8,
            "anxiety": 0.6
        }
    },
    "recommendations": [
        "Practice deep breathing exercises",
        "Try time management techniques"
    ]
}
```

### Audio Analysis Response
```json
{
    "session_id": "uuid-string",
    "timestamp": "2024-03-15T10:30:00Z",
    "emotion_analysis": {
        "dominant_emotion": "stressed",
        "confidence": 0.82,
        "voice_characteristics": {
            "pitch": "higher than normal",
            "speech_rate": "faster than normal"
        }
    },
    "recommendations": [
        "Try progressive muscle relaxation",
        "Practice mindfulness meditation"
    ]
}
```

### Video Analysis Response
```json
{
    "facial_analysis": {
        "emotions": {
            "stress": 0.7,
            "tension": 0.6
        }
    },
    "recommendations": [
        "Take regular breaks during work",
        "Practice facial relaxation exercises"
    ]
}
```

## Error Responses

### Authentication Error
```json
{
    "detail": "Invalid credentials",
    "code": "AUTH_ERROR"
}
```

### Validation Error
```json
{
    "detail": [
        {
            "loc": ["body", "email"],
            "msg": "Invalid email format",
            "type": "value_error"
        }
    ]
}
```

### File Upload Error
```json
{
    "detail": "File size exceeds 10MB limit",
    "code": "FILE_TOO_LARGE"
}
```

## Support

For support, email support@example.com or open an issue in the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

## Acknowledgments

- FastAPI team for the amazing framework
- Hugging Face for the transformer models
- OpenCV for computer vision capabilities
- All contributors and maintainers 
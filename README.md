# Mental Health Analysis API - User Guide

## Introduction

Welcome to the Mental Health Analysis API! This guide will help you understand how to use our API to analyze mental health through text, audio, and video inputs. Our API provides personalized insights and recommendations to help you better understand and manage your mental well-being.

## Getting Started

### 1. Create an Account

First, you'll need to create an account to use the API:

```bash
# Sign up
curl -X POST "https://api.example.com/signup" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your.email@example.com",
    "full_name": "Your Name",
    "password": "your_secure_password"
  }'
```

### 2. Login and Get Access Token

After signing up, you'll need to login to get your access token:

```bash
# Login
curl -X POST "https://api.example.com/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your.email@example.com",
    "password": "your_secure_password"
  }'
```

The response will include your access token. Use this token in all subsequent API requests.

## Using the API

### Text Analysis

Analyze your thoughts and feelings through text:

```bash
# Analyze text
curl -X POST "https://api.example.com/mental-health/analyze/text" \
  -H "Authorization: Bearer your_access_token" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "I've been feeling stressed at work lately. The deadlines are piling up and I'm having trouble sleeping.",
    "context": "Evening reflection"
  }'
```

The response will include:
- Sentiment analysis
- Emotional state
- Key themes
- Personalized recommendations
- Follow-up questions

### Audio Analysis

Analyze your emotional state through voice recordings:

```bash
# Analyze audio
curl -X POST "https://api.example.com/mental-health/analyze/audio" \
  -H "Authorization: Bearer your_access_token" \
  -F "audio=@your_recording.wav"
```

The response will include:
- Voice analysis
- Emotional state
- Speech patterns
- Stress indicators
- Personalized recommendations

### Video Analysis

Analyze your emotional state through video:

```bash
# Analyze video
curl -X POST "https://api.example.com/mental-health/analyze/video" \
  -H "Authorization: Bearer your_access_token" \
  -F "video=@your_video.mp4"
```

The response will include:
- Facial expression analysis
- Voice analysis
- Overall emotional state
- Personalized recommendations

### View Your Trends

Track your mental health over time:

```bash
# Get emotion trends
curl -X GET "https://api.example.com/mental-health/trends/emotions/30" \
  -H "Authorization: Bearer your_access_token"

# Get mental health trends
curl -X GET "https://api.example.com/mental-health/trends/mental-health/30" \
  -H "Authorization: Bearer your_access_token"
```

## Understanding the Results

### Text Analysis Results

```json
{
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
  ],
  "follow_up_questions": [
    "What specific work tasks are causing the most stress?",
    "How has your sleep been affected?"
  ]
}
```

### Audio Analysis Results

```json
{
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

### Video Analysis Results

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

## Best Practices

1. **Regular Analysis**
   - Perform text analysis daily
   - Record audio weekly
   - Video analysis monthly

2. **Privacy**
   - Your data is encrypted and secure
   - You can delete your data at any time
   - We never share your data with third parties

3. **Getting the Best Results**
   - Be honest in your text entries
   - Record in a quiet environment
   - Ensure good lighting for video analysis
   - Use consistent recording times

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Check your access token
   - Ensure token hasn't expired
   - Verify your credentials

2. **File Upload Issues**
   - Check file size (max 10MB)
   - Verify file format
   - Ensure stable internet connection

3. **Analysis Errors**
   - Check input quality
   - Ensure sufficient content
   - Try again later

## Support

If you need help:
1. Check our [FAQ](https://example.com/faq)
2. Email support@example.com
3. Visit our [documentation](https://example.com/docs)

## Privacy and Security

- All data is encrypted
- Your privacy is our priority
- You control your data
- Regular security audits
- GDPR compliant

## Feedback

We value your feedback! Help us improve by:
1. Reporting issues
2. Suggesting features
3. Sharing your experience

Email: feedback@example.com 
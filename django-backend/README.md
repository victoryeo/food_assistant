# Food Assistant - Django Backend

This is the Django backend for the Food Assistant application, providing RESTful APIs for task management between parents and students.

## Features

- User authentication with JWT
- Google OAuth integration
- Task management (create, read, update, delete)
- Parent-student relationship management
- Task completion tracking
- Task summary and statistics

## Prerequisites

- Python 3.8+
- PostgreSQL
- pip
- virtualenv (recommended)

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd django-backend
   ```

2. **Set up Python virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.sample .env
   # Edit .env with your configuration
   ```

5. **Run database migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create a superuser (admin)**
   ```bash
   python manage.py createsuperuser
   ```

## Running the Development Server

```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/`

## API Documentation

### Authentication

- `POST /token/` - Obtain JWT token (email/password)
- `POST /token/refresh/` - Refresh JWT token
- `POST /auth/google/` - Google OAuth login

### Tasks

- `GET /tasks/` - List all tasks (parent view)
- `POST /tasks/` - Create a new task
- `GET /student/tasks/` - List student's tasks
- `POST /student/tasks/<id>/complete/` - Mark task as complete/incomplete
- `GET /summary/` - Get task summary and statistics

## Testing

To run the test suite:

```bash
python manage.py test
```

## Environment Variables

Copy `.env.sample` to `.env` and update the following variables:

- `SECRET_KEY`: Django secret key
- `DATABASE_URL`: Database connection URL
- `JWT_SECRET_KEY`: Secret key for JWT tokens
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `GOOGLE_CLIENT_SECRET`: Google OAuth client secret
- `GOOGLE_REDIRECT_URI`: Callback URL for Google OAuth

## Deployment

For production deployment, make sure to:

1. Set `DEBUG=False` in settings
2. Configure a production database
3. Set up a proper web server (e.g., Nginx + Gunicorn)
4. Configure HTTPS
5. Set proper CORS and ALLOWED_HOSTS

## License

[Your License Here]

## Test URL
curl http://localhost:8000/auth/google/callback/test -X GET 
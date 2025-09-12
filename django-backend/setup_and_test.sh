#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set up the database
echo "Setting up database..."
python manage.py makemigrations
python manage.py migrate

# Run tests
echo "Running tests..."
python manage.py test djangoapp.tests.test_authentication --verbosity=2
python manage.py test djangoapp.tests.test_tasks --verbosity=2
python manage.py test djangoapp.tests.test_parents --verbosity=2

echo "Test complete!"

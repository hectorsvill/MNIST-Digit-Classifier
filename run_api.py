# run_api.py
"""
Main script to run the Flask API server.
Use this instead of running app.py directly, especially with Gunicorn.
"""
import argparse
from src.api.app import create_app # Import the factory function
from src import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the MNIST Digit Guesser API")
    parser.add_argument("--host", type=str, default=config.API_HOST, help="Host to bind the API to")
    parser.add_argument("--port", type=int, default=config.API_PORT, help="Port to run the API on")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode (NOT for production)")

    args = parser.parse_args()

    # Create the app instance using the factory
    app = create_app()

    # Run using Flask's built-in server (good for development)
    # For production, use a WSGI server like Gunicorn:
    # gunicorn --workers 4 --bind 0.0.0.0:5000 run_api:app
    app.run(host=args.host, port=args.port, debug=args.debug)
#web: gunicorn main.app
web: gunicorn main:app --workers=2 --threads=2 --timeout=120 --max-requests=1000 --max-requests-jitter=50 --worker-class=gthread --worker-tmp-dir=/dev/shm --log-file=- --log-level=info --capture-output

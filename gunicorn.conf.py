import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
timeout = 120

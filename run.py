#!/usr/bin/env python3
import os
import sys

try:
    from app import app, price_model, recommendation_rules
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False
    )

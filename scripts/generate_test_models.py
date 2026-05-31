"""Test-model generation lives in the Colab notebook — open it from:

    https://colab.research.google.com/github/IGNODE-CONNECT/ignode-collab/blob/main/colab/generate_test_models.ipynb

Colab has torch + tensorflow pre-installed and downloads ZIPs straight to
your browser, so it's the canonical path. There is no local-Python
equivalent.
"""
import sys

URL = (
    "https://colab.research.google.com/github/IGNODE-CONNECT/"
    "ignode-collab/blob/main/colab/generate_test_models.ipynb"
)

if __name__ == "__main__":
    print("Test models are generated in Colab. Open:")
    print(f"  {URL}")
    sys.exit(0)

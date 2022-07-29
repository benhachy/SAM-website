#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

sys.path.append('..\\')
sys.path.append('..\\ML\\Full\\')

import nltk
nltk.data.path.append('../nltk_data/')

# Ignoring the CUDA warning from TensorFlow because we don't use GPU acceleration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myteam.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

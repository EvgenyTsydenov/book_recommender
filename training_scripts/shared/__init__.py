import os
import sys
from os.path import dirname

from dotenv import load_dotenv

# Append PYTHONPATH with the project path
sys.path.append(dirname(dirname(dirname(__file__))))

# Load environment variables
load_dotenv()
NEPTUNE_API_KEY = os.environ['NEPTUNE_API_KEY']
NEPTUNE_PROJECT = os.environ['NEPTUNE_PROJECT']
RANDOM_SEED = int(os.environ['RANDOM_SEED'])

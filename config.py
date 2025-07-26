import os

# ---------🔧 PATH CONFIGURATIONS -------------

#root data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

#os.path.join(...) makes this code OS-independent (Windows/Mac/Linux).

#default dataset to load 
DEFAULT_DATASET_FILENAME = "sample_dataset.csv"
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, DEFAULT_DATASET_FILENAME)

# --------- 🏷️ COLUMN NAMES -------------

#reuired columns like rag context

REQUIRED_COLUMNS = ["question", "ground_truth_answer"]

#optional columns like rag context

OPTIONAL_COLUMNS = ["context"]

# --------- 📂 SUPPORTED FILE TYPES ----------

SUPPORTED_FORMATS = [".csv", ".json"]

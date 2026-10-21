import sys
import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import ultralytics
import supervision as sv

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Ultralytics:", ultralytics.__version__)
print("Supervision:", sv.__version__)
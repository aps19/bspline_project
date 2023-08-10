import os
import sys
import numpy as np
import pandas as pd
import dill

from exception import CustomException
from logger import logging

def save_object(filepath,obj):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e,sys)
import gradio as gr
import warnings
import numpy as np
import sys
warnings.filterwarnings('ignore')

sys.path.append('/home/shanybiton/repos/RDS/parsing')
sys.path.append('/home/shanybiton/repos/RDS')
sys.path.append('/home/shanybiton/repos/RDS/utils')
# Relative imports
import src.models.metrics as metrics
import src.models.model_utils as model_utils
from parsing.db_loader import *
import data.data_loading as data_loading
from utils import consts as cts

def run(model, X):
    probas = model.predict_proba(X)[:, 1]
    return probas

if __name__ == '__main__':
    algo_py='ArNet2'
    feature_extractor_path = cts.path_models['ResNet']
    model_dict = model_utils.load_model(cts.path_models['ArNet2'], algo=algo_py,
                                        path_feature_extractor=feature_extractor_path)
    model = model_dict['classifier']
    outputs = gr.outputs.Textbox()

    app = gr.Interface(fn=run, #callable function
                       inputs=["dict", "pkl"], #input format
                       outputs=outputs, description="This is ArNet2") #output format
    #display the interface
    app.launch(share=True)



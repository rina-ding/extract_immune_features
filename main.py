# https://github.com/TissueImageAnalytics/tiatoolbox
import sys
sys.path.append('./tiatool_current')
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.visualization import overlay_prediction_contours

import numpy as np
import matplotlib.pyplot as plt
import joblib

from PIL import Image
import os
from glob import glob
import sys
import pandas as pd

from skimage.draw import polygon2mask
import shutil
from natsort import natsorted
import denTIL_and_tilab
import cv2
import time

def extract_denTIL_and_tilab(df, image_path):
    all_features = []
    isLymphocyte = df['nuclei_type'] == 'immune'
    isEpithelial = df['nuclei_type'] == 'tumor_epithelial'
    df_matrix = df[['y', 'x']].values
    I = cv2.imread(image_path)
    totLympArea = df.loc[isLymphocyte, 'nuclei_area'].sum()

    # Density TIL features
    denTIL_features = denTIL_and_tilab.getDenTILFeatures(I, df_matrix[isLymphocyte, :], df_matrix[~isLymphocyte, :], df_matrix[isEpithelial, :], totLympArea)
    all_features.extend(denTIL_features)
    # TIL abundance scores
    df_group1 = df[df['nuclei_type'] == 'immune'].values
    df_group2 = df[df['nuclei_type'] == 'tumor_epithelial'].values
    partitions = 5
    imgDim = I.shape[0]
    tilab_features_immune_vs_cancer = denTIL_and_tilab.get_tilab_features(df_group1, df_group2, partitions, imgDim)
    # When calculating immune cells vs rest
    nuclei_types = ['tumor_epithelial', 'connective', 'nontumor_epithelial', 'necrotic']
    df_rest = df[df['nuclei_type'].isin(nuclei_types)].values
    tilab_features_immune_vs_rest = denTIL_and_tilab.get_tilab_features(df_group1, df_rest, partitions, imgDim)
    all_features.extend(tilab_features_immune_vs_cancer)
    all_features.extend(tilab_features_immune_vs_rest)
    return all_features

def run_nuclei_classification(input_tile):

    # "input_tile" has to be in a list, where there's only 1 item in the list.    
    # Tile prediction
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model= 'hovernet_fast-pannuke',
        num_loader_workers=1,
        num_postproc_workers=1,
        batch_size=4,
    )

    tile_output = inst_segmentor.predict(
        input_tile,
        save_dir = output_dir,
        mode = 'tile',
        on_gpu = True,
        crash_on_exception=True,
    )

    tile_preds = joblib.load(os.path.join(output_dir, '0.dat'))
    color_dict = {
        0: ("neoplastic epithelial", (255, 0, 0)),
        1: ("Inflammatory", (255, 255, 0)),
        2: ("Connective", (0, 255, 0)),
        3: ("Dead", (0, 0, 0)),
        4: ("non-neoplastic epithelial", (0, 0, 255)),
    }

    # Create the overlay image
    tile_img = imread(input_tile[0])
    height, width, channels = tile_img.shape
    overlaid_predictions = overlay_prediction_contours(
        canvas=tile_img,
        inst_dict=tile_preds,
        draw_dot=False,
        type_colours=color_dict,
        line_thickness=2,
    )
    Image.fromarray(overlaid_predictions).save(os.path.join(output_dir, 'overlaid_predictions.png'))

    values = [(value['centroid'][1], value['centroid'][0], value['type'], np.sum(polygon2mask((height, width), np.asarray(value['contour'])))) for key, value in tile_preds.items()]
    df = pd.DataFrame(values, columns=['y', 'x', 'nuclei_type', 'nuclei_area'])
    value_mapping = {0: 'tumor_epithelial', 1: 'immune', 2: 'connective', 3:'necrotic', 4: 'nontumor_epithelial'}
    df['nuclei_type'] = df['nuclei_type'].map(value_mapping)
    return df

if __name__ == "__main__":
    input_tile_path = '/workspace/home/ruiwending/lung_project/features/extract_immune_features/example_input/image.png'
    output_dir = os.path.join(os.path.dirname(input_tile_path), 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
  
    all_tile_features = []
    input_tile = [input_tile_path] # This is a required format for TIATOOLBOX. A list with only 1 item, which is your tile path.
    
    print('Starting nuclei classification ')

    df_nuclei_classification = run_nuclei_classification(input_tile)
    df_nuclei_classification.to_csv(os.path.join(output_dir, 'nuclei_classification_results.csv'), index = None)
  
    # Extract immune features
    print('Starting feature extraction ')
    this_tile_features = extract_denTIL_and_tilab(df_nuclei_classification, input_tile_path)
    all_tile_features.append(this_tile_features)

    df_all_tile_features = pd.DataFrame(all_tile_features, columns = ['#Lymp/TissueArea', 'LympTotalArea/TissueArea', '#Lymp/#TotalNuclei', '#Lymp/#TotalTumorEpithelial', 'MaxLympGroupingFactor', 'MinLympGroupingFactor',
                                                    'AvgLympGroupingFactor',  'StdLympGroupingFactor', 'MedianLympGroupingFactor', 'ModeLympGroupingFactor', 'NumHighlyGroupedLymp', '#Lymp/TotalConvHullArea', 
                                                    '#Lymp/TotalEpithelialConvHullArea','LympConvHullArea/TotalConvHullArea', 'LympConvHullArea/TotalEpithelialConvHullArea',
                                                    'IntersectedAreaConvHullLymp&NonLymp',  'IntersectedAreaConvHullLymp&Epithelial', 'MaxDensityMatrixVal', 'MinDensityMatrixVal', 
                                                    'AvgDensityMatrixVal', 'StdDensityMatrixVal', 'MedianDensityMatrixVal', 'ModeDensityMatrixVal',
                                                    'coloc_score_immune_vs_epi', 'tilab_score_immune_vs_epi', 'coloc_score_immune_vs_rest', 'tilab_score_immune_vs_rest'
                                                    ])
    
    df_all_tile_features.insert(0, 'tile_name', os.path.basename(input_tile_path))

    df_all_tile_features.to_csv(os.path.join(output_dir, 'tile_features.csv'), index = None)


            

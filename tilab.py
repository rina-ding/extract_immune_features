import numpy as np
import pandas as pd

def calculate_til_abundance(global_num_g1, global_num_g2):
    # Calculate the immune infiltration score
    # https://github.com/TissueImageAnalytics/TILAb-Score/blob/db95fbdcbc9a974d717078e03ad1a55f9f386ba0/src/til_quantification.py
    l = np.asarray(global_num_g1)
    t = np.asarray(global_num_g2)
    # Morisita-Horn Index based colocalization socre
    coloc_score = (2 * sum(t*l)) / (sum(t**2) + sum(l**2))
    if np.sum(t) == 0:
        l2t_ratio = np.sum(l)
        tilab_score = 1 # when only lymphocytes are present
    else:
        l2t_ratio = np.sum(l) / np.sum(t)  # lymphocyte to tumour ratio
        tilab_score = 0.5 * coloc_score * l2t_ratio  # final TILAb-score
        
    print('Final TILAb score ', tilab_score)
    return coloc_score, l2t_ratio, tilab_score

def get_tilab_features(centroids_yx_g1, centroids_yx_g2, partitions, imgDim):
    # Assume group1 cells are immune cells
    tileDim = imgDim / partitions
    global_num_g1 = []
    for i in np.arange(1, imgDim, tileDim):
        for j in np.arange(1, imgDim, tileDim):
            coords_g1 = centroids_yx_g1[(centroids_yx_g1[:, 0] >= i) & (centroids_yx_g1[:, 0] < i + tileDim) & 
                               (centroids_yx_g1[:, 1] >= j) & (centroids_yx_g1[:, 1] < j + tileDim)]
            global_num_g1.append(len(coords_g1))
    
    global_num_g2 = []
    for i in np.arange(1, imgDim, tileDim):
        for j in np.arange(1, imgDim, tileDim):
            coords_g2 = centroids_yx_g2[(centroids_yx_g2[:, 0] >= i) & (centroids_yx_g2[:, 0] < i + tileDim) & 
                               (centroids_yx_g2[:, 1] >= j) & (centroids_yx_g2[:, 1] < j + tileDim)]
            global_num_g2.append(len(coords_g2))

    # print('global_num_g1 ', global_num_g1)
    # print('global_num_g2 ', global_num_g2)
    # If there's no tumor cells and also no lymphocytes
    if sum(global_num_g1) == 0 and sum(global_num_g2) == 0:
        coloc_score, l2t_ratio, tilab_score = 0, 0, 0
    else:
        coloc_score, l2t_ratio, tilab_score = calculate_til_abundance(global_num_g1, global_num_g2)
    # Calculate mean, SD, max, min of immune cell count
    # mean_l = np.mean(np.asarray(global_num_g1))
    # sd_l = np.std(np.asarray(global_num_g1))
    # max_l = np.max(np.asarray(global_num_g1))
    # min_l = np.min(np.asarray(global_num_g1))
    features = [coloc_score, tilab_score]
    return features

if __name__ == "__main__":
    df = pd.read_csv('/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/this_tile_centroids.csv')
    df_matrix = df[['y', 'x']]
    df_group1 = df[df['nuclei_type'] == 'immune'].values
    df_group2 = df[df['nuclei_type'] == 'tumor_epithelial'].values
    partitions = 5
    imgDim = 2048
    features = get_tilab_features(df_group1, df_group2, partitions, imgDim)
    # When calculating immune cells vs rest
    nuclei_types = ['tumor_epithelial', 'connective', 'nontumor_epithelial', 'necrotic']
    df_rest = df[df['nuclei_type'].isin(nuclei_types)].values
    features_immune_vs_rest = get_tilab_features(df_group1, df_rest, partitions, imgDim)
    features.extend(features_immune_vs_rest)
    features = np.asarray(features).reshape(1, -1)
    df_to_save = pd.DataFrame(features, columns = ['coloc_score_immune_vs_epi', 'tilab_score_immune_vs_epi', 'coloc_score_immune_vs_rest', 'tilab_score_immune_vs_rest'])
    df_to_save.to_csv('/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/this_tile_tilab_features.csv', index=False)




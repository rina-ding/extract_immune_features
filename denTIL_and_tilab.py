import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import cv2
from scipy.spatial.distance import pdist, squareform
import os
import pandas as pd
import statistics

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

def getDensityMatrixCore(imgDim, partitions, centroids):
    tileDim = imgDim // partitions
    M = []
    for i in np.arange(1, imgDim, tileDim):
        for j in np.arange(1, imgDim, tileDim):
            coords = centroids[(centroids[:, 0] >= i) & (centroids[:, 0] < i + tileDim) & 
                               (centroids[:, 1] >= j) & (centroids[:, 1] < j + tileDim)]
            M.append(len(coords))
    return np.array(M)

def getIntersectedArea(pointsShape1, pointsShape2):
    poly1 = Polygon(pointsShape1)
    poly2 = Polygon(pointsShape2)
    intersection = poly1.intersection(poly2)
    area = intersection.area
    return area

def getTissueArea(I):
    gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    binary = np.array(binary, dtype=bool)
    area = np.sum(~binary)
    return area

def getSumNodeWeightsThreshold(feature, distance, threshold):
    dist = pdist(feature, distance)
    dist[dist == 0] = 1
    dist = dist ** -1

    dist[dist < threshold] = 0

    vect = np.sum(squareform(dist), axis=0)
    return vect

def getClusterCentroids(members, nodes):
    numMembers = len(members)
    centroids = []
    polygons = []
    counter = 0

    for i in range(numMembers):
        member = members[i]
        col = nodes['centroid_c'][member]
        row = nodes['centroid_r'][member]
        numNodes = len(col)

        if numNodes > 2:
            k = ConvexHull(np.column_stack((col, row)))
            cx = np.mean(col[k.vertices])
            cy = np.mean(row[k.vertices])
            centroids.append([cx, cy])
            polygons.append(np.column_stack((col[k.vertices], row[k.vertices])))
            counter += 1

    return centroids, polygons

def normalizeVector(vect, inverted=False):
    if not inverted:
        arr = (vect - np.min(vect)) / (np.max(vect) - np.min(vect))
    else:
        arr = (np.max(vect) - vect) / (np.max(vect) - np.min(vect))
    return arr

def getDenTILFeatures(image, lympCentroids, nonLympCentroids, EpithelialCentroids, totLympArea):
    numLymp = len(lympCentroids)
    totNuclei = numLymp + len(nonLympCentroids)

    # Regular-density-based measures
    A = getTissueArea(image)
    densLymp = numLymp / A
    densAreaLymp = totLympArea / A
    if totNuclei == 0:
        ratioLymp = 0
    else:
        ratioLymp = numLymp / totNuclei
    if len(EpithelialCentroids) == 0:
        ratioLympToEpithelial = 0
    else:
        ratioLympToEpithelial = numLymp / len(EpithelialCentroids) # Not in the original denTIL features

    # Grouping-based measures

    if len(lympCentroids) == 0:
        maxGr = 0
        minGr = 0
        avgGr = 0
        stdGr = 0
        medGr = 0
        modeGr = 0
        numHighlyGroupedLymp = 0
    elif len(lympCentroids) != 0:
        groupingVector = getSumNodeWeightsThreshold(lympCentroids, 'euclidean', 0.005)
        normVect = normalizeVector(groupingVector, 0)

        maxGr = np.max(groupingVector)
        minGr = np.min(groupingVector)
        avgGr = np.mean(groupingVector)
        stdGr = np.std(groupingVector)
        medGr = np.median(groupingVector)
        modeGr = statistics.mode(groupingVector)
        numHighlyGroupedLymp = len(normVect[normVect > 0.5])

    # Convex hull measures
    if len(lympCentroids) > 2 and len(EpithelialCentroids) > 2:
        allCentroids = np.concatenate((lympCentroids, nonLympCentroids), axis=0)
        areaConvHull = ConvexHull(allCentroids).volume
        epithelialAreaConvHull = ConvexHull(EpithelialCentroids).volume # Not in the original denTIL features

        hullLymp = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hullLymp.vertices]

        if len(nonLympCentroids) > 2:
            hullNonLymp = ConvexHull(nonLympCentroids)
            convHullNonLymp = nonLympCentroids[hullNonLymp.vertices]
            intersArea = getIntersectedArea(convHullLymp, convHullNonLymp)
        else:
            intersArea = 0

        # Not in the original denTIL features
        if len(EpithelialCentroids) > 2:
            hullEpithelial = ConvexHull(EpithelialCentroids)
            convHullEpithelial = EpithelialCentroids[hullEpithelial.vertices]
            intersAreaWithEpithelial = getIntersectedArea(convHullLymp, convHullEpithelial)
        else:
            intersAreaWithEpithelial = 0

        densLympConvHull = numLymp / areaConvHull
        densLympEpithelialConvHull = numLymp / epithelialAreaConvHull # Not in the original denTIL features

        hull = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hull.vertices]
        areaConvHullLymp = hull.volume
        ratioConvHulls = areaConvHullLymp / areaConvHull
        ratioEpithelialConvHulls = areaConvHullLymp / epithelialAreaConvHull # Not in the original denTIL features

    else:
        densLympConvHull = 0
        densLympEpithelialConvHull = 0
        ratioConvHulls = 0
        ratioEpithelialConvHulls = 0
        intersArea = 0
        intersAreaWithEpithelial = 0

    # Density-Matrix-based measures
    if len(lympCentroids) == 0:
        maxM = 0
        minM = 0
        avgM = 0
        stdM = 0
        medM = 0
        modeM = 0
    elif len(lympCentroids) != 0:
        M = getDensityMatrixCore(len(image), 5, lympCentroids)
        M = M[M != 0]

        try:
            maxM = np.max(M)
            minM = np.min(M)
            avgM = np.mean(M)
            stdM = np.std(M)
            medM = np.median(M)
            modeM = statistics.mode(M)
        except:
            maxM = 0
            minM = 0
            avgM = 0
            stdM = 0
            medM = 0
            modeM = 0

    # Compiling features
    features = [densLymp, densAreaLymp, ratioLymp, ratioLympToEpithelial, maxGr, minGr, avgGr,
                stdGr, medGr, modeGr, numHighlyGroupedLymp, densLympConvHull, densLympEpithelialConvHull,
                ratioConvHulls, ratioEpithelialConvHulls, intersArea, intersAreaWithEpithelial, maxM, minM, avgM, stdM, medM, modeM]
    return features

def getDenTILFeaturesv2(image, lympCentroids, nonLympCentroids, EpithelialCentroids, totLympArea):
    partitions = 3
    numLymp = len(lympCentroids)
    totNuclei = numLymp + len(nonLympCentroids)

    # Regular-density-based measures
    A = getTissueArea(image)
    densLymp = numLymp / A
    densAreaLymp = totLympArea / A

    if totNuclei == 0:
        ratioLymp = 0
    else:
        ratioLymp = numLymp / totNuclei
    if len(EpithelialCentroids) == 0:
        ratioLympToEpithelial = 0
    else:
        ratioLympToEpithelial = numLymp / len(EpithelialCentroids) # Not in the original denTIL features

    # Grouping-based measures

    if len(lympCentroids) == 0:
        maxGr = 0
        minGr = 0
        avgGr = 0
        stdGr = 0
        medGr = 0
        modeGr = 0
        numHighlyGroupedLymp = 0
    elif len(lympCentroids) != 0:
        groupingVector = getSumNodeWeightsThreshold(lympCentroids, 'euclidean', 0.005)
        normVect = normalizeVector(groupingVector, 0)

        maxGr = np.max(groupingVector)
        minGr = np.min(groupingVector)
        avgGr = np.mean(groupingVector)
        stdGr = np.std(groupingVector)
        medGr = np.median(groupingVector)
        modeGr = statistics.mode(groupingVector)
        numHighlyGroupedLymp = len(normVect[normVect > 0.5])

    # Convex hull measures
    if len(lympCentroids) > 2 and len(EpithelialCentroids) > 2:
        allCentroids = np.concatenate((lympCentroids, nonLympCentroids), axis=0)
        areaConvHull = ConvexHull(allCentroids).volume
        epithelialAreaConvHull = ConvexHull(EpithelialCentroids).volume # Not in the original denTIL features

        hullLymp = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hullLymp.vertices]

        if len(nonLympCentroids) > 2:
            hullNonLymp = ConvexHull(nonLympCentroids)
            convHullNonLymp = nonLympCentroids[hullNonLymp.vertices]
            intersArea = getIntersectedArea(convHullLymp, convHullNonLymp)
        else:
            intersArea = 0

        # Not in the original denTIL features
        if len(EpithelialCentroids) > 2:
            hullEpithelial = ConvexHull(EpithelialCentroids)
            convHullEpithelial = EpithelialCentroids[hullEpithelial.vertices]
            intersAreaWithEpithelial = getIntersectedArea(convHullLymp, convHullEpithelial)
        else:
            intersAreaWithEpithelial = 0

        densLympConvHull = numLymp / areaConvHull
        densLympEpithelialConvHull = numLymp / epithelialAreaConvHull # Not in the original denTIL features

        hull = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hull.vertices]
        areaConvHullLymp = hull.volume
        ratioConvHulls = areaConvHullLymp / areaConvHull
        ratioEpithelialConvHulls = areaConvHullLymp / epithelialAreaConvHull # Not in the original denTIL features

    else:
        densLympConvHull = 0
        densLympEpithelialConvHull = 0
        ratioConvHulls = 0
        ratioEpithelialConvHulls = 0
        intersArea = 0
        intersAreaWithEpithelial = 0

    # Density-Matrix-based measures
    if len(lympCentroids) <= 1:
        maxM = 0
        minM = 0
        avgM = 0
        stdM = 0
        medM = 0
        modeM = 0
    elif len(lympCentroids) > 1:
        M = getDensityMatrixCore(len(image), partitions, lympCentroids)
        M = M[M != 0]
        try:
            maxM = np.max(M)
            minM = np.min(M)
            avgM = np.mean(M)
            stdM = np.std(M)
            medM = np.median(M)
            modeM = statistics.mode(M)
        except:
            maxM = 0
            minM = 0
            avgM = 0
            stdM = 0
            medM = 0
            modeM = 0

    # Compiling features
    features = [densLymp, densAreaLymp, ratioLymp, maxGr, avgGr,
                stdGr, medGr, numHighlyGroupedLymp, densLympConvHull, 
                ratioConvHulls, intersArea, intersAreaWithEpithelial, maxM, avgM, stdM, medM, modeM]
    return features

def getDenTILFeaturesv2WithoutTissueArea(tile_size, lympCentroids, nonLympCentroids, EpithelialCentroids, totLympArea):
    partitions = 3
    numLymp = len(lympCentroids)
    totNuclei = numLymp + len(nonLympCentroids)

    if totNuclei == 0:
        ratioLymp = 0
    else:
        ratioLymp = numLymp / totNuclei
    if len(EpithelialCentroids) == 0:
        ratioLympToEpithelial = 0
    else:
        ratioLympToEpithelial = numLymp / len(EpithelialCentroids) # Not in the original denTIL features

    # Grouping-based measures

    if len(lympCentroids) == 0:
        maxGr = 0
        minGr = 0
        avgGr = 0
        stdGr = 0
        medGr = 0
        modeGr = 0
        numHighlyGroupedLymp = 0
    elif len(lympCentroids) != 0:
        groupingVector = getSumNodeWeightsThreshold(lympCentroids, 'euclidean', 0.005)
        normVect = normalizeVector(groupingVector, 0)

        maxGr = np.max(groupingVector)
        minGr = np.min(groupingVector)
        avgGr = np.mean(groupingVector)
        stdGr = np.std(groupingVector)
        medGr = np.median(groupingVector)
        modeGr = statistics.mode(groupingVector)
        numHighlyGroupedLymp = len(normVect[normVect > 0.5])

    # Convex hull measures
    if len(lympCentroids) > 2 and len(EpithelialCentroids) > 2:
        allCentroids = np.concatenate((lympCentroids, nonLympCentroids), axis=0)
        areaConvHull = ConvexHull(allCentroids).volume
        epithelialAreaConvHull = ConvexHull(EpithelialCentroids).volume # Not in the original denTIL features

        hullLymp = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hullLymp.vertices]

        if len(nonLympCentroids) > 2:
            hullNonLymp = ConvexHull(nonLympCentroids)
            convHullNonLymp = nonLympCentroids[hullNonLymp.vertices]
            intersArea = getIntersectedArea(convHullLymp, convHullNonLymp)
        else:
            intersArea = 0

        # Not in the original denTIL features
        if len(EpithelialCentroids) > 2:
            hullEpithelial = ConvexHull(EpithelialCentroids)
            convHullEpithelial = EpithelialCentroids[hullEpithelial.vertices]
            intersAreaWithEpithelial = getIntersectedArea(convHullLymp, convHullEpithelial)
        else:
            intersAreaWithEpithelial = 0

        densLympConvHull = numLymp / areaConvHull
        densLympEpithelialConvHull = numLymp / epithelialAreaConvHull # Not in the original denTIL features

        hull = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hull.vertices]
        areaConvHullLymp = hull.volume
        ratioConvHulls = areaConvHullLymp / areaConvHull
        ratioEpithelialConvHulls = areaConvHullLymp / epithelialAreaConvHull # Not in the original denTIL features

    else:
        densLympConvHull = 0
        densLympEpithelialConvHull = 0
        ratioConvHulls = 0
        ratioEpithelialConvHulls = 0
        intersArea = 0
        intersAreaWithEpithelial = 0

    # Density-Matrix-based measures
    if len(lympCentroids) == 0:
        maxM = 0
        minM = 0
        avgM = 0
        stdM = 0
        medM = 0
        modeM = 0
    elif len(lympCentroids) > 0:
        M = getDensityMatrixCore(tile_size, partitions, lympCentroids)
        M = M[M != 0]
        try:
            maxM = np.max(M)
            minM = np.min(M)
            avgM = np.mean(M)
            stdM = np.std(M)
            medM = np.median(M)
            modeM = statistics.mode(M)
        except:
            maxM = 0
            minM = 0
            avgM = 0
            stdM = 0
            medM = 0
            modeM = 0

    # Compiling features
    features = [ratioLymp, maxGr, avgGr,
                stdGr, medGr, numHighlyGroupedLymp, densLympConvHull, 
                ratioConvHulls, intersArea, intersAreaWithEpithelial, maxM, avgM, stdM, medM, modeM]
    return features

# if __name__ == "__main__":
#     image_path = '/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/image.png'
#     df = pd.read_csv('/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/this_tile_centroids.csv')
#     isLymphocyte = df['nuclei_type'] == 'immune'
#     isEpithelial = df['nuclei_type'] == 'tumor_epithelial'
#     df_matrix = df[['y', 'x']].values
#     I = cv2.imread(image_path)
#     totLympArea = df.loc[isLymphocyte, 'nuclei_area'].sum()

#     # Density TIL features
#     denTIL_features = getDenTILFeatures(I, df_matrix[isLymphocyte, :], df_matrix[~isLymphocyte, :], df_matrix[isEpithelial, :], totLympArea)
#     denTIL_features = np.asarray(denTIL_features).reshape(1, -1)
#     df_denTIL_features = pd.DataFrame(denTIL_features, columns = ['#Lymp/TissueArea', 'LympTotalArea/TissueArea', '#Lymp/#TotalNuclei', '#Lymp/#TotalTumorEpithelial', 'MaxLympGroupingFactor', 'MinLympGroupingFactor',
#                                                                  'AvgLympGroupingFactor',  'StdLympGroupingFactor', 'MedianLympGroupingFactor', 'ModeLympGroupingFactor', 'NumHighlyGroupedLymp', '#Lymp/TotalConvHullArea', 
#                                                                  '#Lymp/TotalEpithelialConvHullArea','LympConvHullArea/TotalConvHullArea', 'LympConvHullArea/TotalEpithelialConvHullArea',
#                                                                   'IntersectedAreaConvHullLymp&NonLymp',  'IntersectedAreaConvHullLymp&Epithelial', 'MaxDensityMatrixVal', 'MinDensityMatrixVal', 
#                                                                   'AvgDensityMatrixVal', 'StdDensityMatrixVal', 'MedianDensityMatrixVal', 'ModeDensityMatrixVal'])

#     # TIL abundance scores
#     df_group1 = df[df['nuclei_type'] == 'immune'].values
#     df_group2 = df[df['nuclei_type'] == 'tumor_epithelial'].values
#     partitions = 5
#     imgDim = 2048
#     features = get_tilab_features(df_group1, df_group2, partitions, imgDim)
#     # When calculating immune cells vs rest
#     nuclei_types = ['tumor_epithelial', 'connective', 'nontumor_epithelial', 'necrotic']
#     df_rest = df[df['nuclei_type'].isin(nuclei_types)].values
#     features_immune_vs_rest = get_tilab_features(df_group1, df_rest, partitions, imgDim)
#     features.extend(features_immune_vs_rest)
#     features = np.asarray(features).reshape(1, -1)
#     df_tilab = pd.DataFrame(features, columns = ['coloc_score_immune_vs_epi', 'tilab_score_immune_vs_epi', 'coloc_score_immune_vs_rest', 'tilab_score_immune_vs_rest'])

#     df_all = pd.concat([df_denTIL_features, df_tilab], axis = 1)
    
#     df_all.to_csv('/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/this_tile_denTIL_and_tilab_features.csv', index=False)




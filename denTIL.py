import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import cv2
from scipy.spatial.distance import pdist, squareform
import os
import pandas as pd
import statistics

def getDensityMatrixCore(imgDim, partitions, centroids):
    tileDim = imgDim / partitions
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

def getDenTILFeatures(image, lympCentroids, nonLympCentroids, totLympArea):
    numLymp = len(lympCentroids)
    totNuclei = numLymp + len(nonLympCentroids)

    # Regular-density-based measures
    A = getTissueArea(image)
    densLymp = numLymp / A
    densAreaLymp = totLympArea / A
    ratioLymp = numLymp / totNuclei

    # Grouping-based measures
    groupingVector = getSumNodeWeightsThreshold(lympCentroids, 'euclidean', 0.005)
    normVect = normalizeVector(groupingVector, 0)

    maxGr = np.max(groupingVector)
    minGr = np.min(groupingVector)
    avgGr = np.mean(groupingVector)
    stdGr = np.std(groupingVector)
    medGr = np.median(groupingVector)
    modeGr = statistics.mode(groupingVector)
    numHighlyGroupedLymp = len(normVect[normVect > 0.5])

    if len(lympCentroids) > 2:
        allCentroids = np.concatenate((lympCentroids, nonLympCentroids), axis=0)
        areaConvHull = ConvexHull(allCentroids).volume

        hullLymp = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hullLymp.vertices]

        if len(nonLympCentroids) > 2:
            hullNonLymp = ConvexHull(nonLympCentroids)
            convHullNonLymp = nonLympCentroids[hullNonLymp.vertices]
            intersArea = getIntersectedArea(convHullLymp, convHullNonLymp)
        else:
            intersArea = 0

        densLympConvHull = numLymp / areaConvHull

        hull = ConvexHull(lympCentroids)
        convHullLymp = lympCentroids[hull.vertices]
        areaConvHullLymp = hull.volume
        ratioConvHulls = areaConvHullLymp / areaConvHull
    else:
        densLympConvHull = 0
        ratioConvHulls = 0
        intersArea = 0

    # Density-Matrix-based measures
    M = getDensityMatrixCore(len(image), 5, lympCentroids)
    M = M[M != 0]

    maxM = np.max(M)
    minM = np.min(M)
    avgM = np.mean(M)
    stdM = np.std(M)
    medM = np.median(M)
    modeM = statistics.mode(M)

    # Compiling features
    features = [densLymp, densAreaLymp, ratioLymp, maxGr, minGr, avgGr,
                stdGr, medGr, modeGr, numHighlyGroupedLymp, densLympConvHull,
                ratioConvHulls, intersArea, maxM, minM, avgM, stdM, medM, modeM]
    return features

    # featureNames = ['#Lymp/TissueArea', 'LympTotalArea/TissueArea',
    #                 '#Lymp/#TotalNuclei', 'MaxLympGroupingFactor',
    #                 'MinLympGroupingFactor', 'AvgLympGroupingFactor',
    #                 'StdLympGroupingFactor', 'MedianLympGroupingFactor',
    #                 'ModeLymp


if __name__ == "__main__":
    image_path = '/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/image.png'
    df = pd.read_csv('/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/this_tile_centroids.csv')
    isLymphocyte = df['nuclei_type'] == 'immune'
    df_matrix = df[['y', 'x']].values
    I = cv2.imread(image_path)
    totLympArea = df.loc[isLymphocyte, 'nuclei_area'].sum()

    # Density TIL features
    denTIL_features = getDenTILFeatures(I, df_matrix[isLymphocyte, :], df_matrix[~isLymphocyte, :], totLympArea)
    pd.DataFrame(denTIL_features).to_csv('/workspace/home/ruiwending/lung_project/features/tiatool_current/sample_tile_results/this_tile_denTIL_features_python.csv', index=False)




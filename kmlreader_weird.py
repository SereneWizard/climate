import os
import sys
import re
import glob
import pickle
import numpy as np 
import pandas as pd 
import geopandas as gpd 
import argparse
from shapely.geometry import Point, Polygon 
from xml.etree import ElementTree as ET 

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(2, os.path.join(sys.path[1], '..'))

from utils.database_utils import get_prod_connection, set_AIID_MUID_column, set_MUID_column


def kml2polygon(kmlFpath, bounds=[120, 42, 120.5, 42.5], mask=True, buffer=False):
    if kmlFpath:
        tree = ET.parse(kmlFpath)
        root = tree.getroot().tag
        kmlns = re.split('[{}]', root)[1]
        elements = tree.findall(".//{%s}coordinates" % kmlns)
        for el in elements:
            shapeCoordinates = []
            for coords in el.text.split(' '):
                x, y = coords.split(',')
                x, y = float(x), float(y)
                x = (360+x) if x<0 else x
                shapeCoordinates.append((x,y))
        maskPolygon = Polygon(shapeCoordinates)
        returnPolygon = maskPolygon
        if not mask:
            llx, lly, urx, ury = maskPolygon.bounds
            rectBoundsPolygon = Polygon(((llx, lly), (urx, lly), (urx, ury), 
                            (llx, ury), (llx, lly)))
            returnPolygon = rectBoundsPolygon
    else: 
        llx, lly, urx, ury = bounds
        boundsPolygon = Polygon(((llx, lly), (urx, lly), (urx, ury), 
                            (llx, ury), (llx, lly)))
        returnPolygon = boundsPolygon
    if buffer:
        returnPolygon = returnPolygon.buffer(0.25)
        
    return returnPolygon


def intersection(maskPolygon):
    with open("grid.p", 'rb') as gridSrc:
        gdf = pickle.load(gridSrc)
    gridSpIndex = gdf.sindex 
    possible_matches_index = list(gridSpIndex.intersection(maskPolygon.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(maskPolygon)]
    precise_matches['longitude'] = precise_matches['geometry'].x
    precise_matches['latitude'] = precise_matches['geometry'].y
    precise_matches.reset_index(drop=True, inplace=True)
    precise_matches.index.name = 'ID'

    return (precise_matches)

def savetocsv(precise_matches, saveFileName, signedX=True, **kwargs):
    if signedX:
        precise_matches['longitude'] = precise_matches['longitude'].apply(lambda x: x-360 if x>180 else x) 
    precise_matches.to_csv(saveFileName, **kwargs)

def saveasKML(precise_matches, saveFileKML):
    import simplekml
    kml = simplekml.Kml()
    for key, row in precise_matches.iterrows():
        kml.newpoint(coords=[(row.longitude, row.latitude)])
    kml.save(saveFileKML)



def get_ids(df=[],filePath=False, getAI_ID=False):
    if filePath:
        df = pd.read_csv(filepath)
    df['longitude'] = df['longitude'].apply(lambda x: x-360 if x>180 else x )
    conn = get_prod_connection()
    if getAI_ID:
        df_new = set_AIID_MUID_column(df, conn)
    else:
        df_new = set_MUID_column(df, conn.cursor)
    print(df_new)

    print("Done")


if __name__ == "__main__":
<<<<<<< HEAD
    kmlFname = glob.glob("./data/kmls/e*.kml")[0]
    print(kmlFname)
=======
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kmlFname', '-k', help='The kml file', required=True)
    parser.add_argument(
        '--saveFile', '-s', help='Filepath for storing intersected coordinates', 
        required=False
    )
    args = parser.parse_args()
    kmlFname = args.kmlFname
>>>>>>> 82ea3e3872547d13a1f9cfbca9070dca768cadc1
    maskPolygon = kml2polygon(kmlFname, buffer=True)
    print("The bounding box of the polygon is: ", maskPolygon.bounds)
    intersected = intersection(maskPolygon)
<<<<<<< HEAD
    intersectedWithIDs = get_ids(intersected)
    print(intersectedWithIDs)
    savetocsv(intersectedWithIDs, "easternidahoPoints3.csv", signedX=True, index=True)
    saveasKML(intersected, "./data/mypoints2.kml")
    #intersected.to_csv("intersected_points.csv", index=False)
=======
    #intersected = get_ids(intersected)
    if args.saveFile:
        saveFile = args.saveFile
    else:
        saveFile = os.path.join(os.getcwd(), 'data', 'intersectedPoints.csv')
    savetocsv(intersected, saveFile, signedX=True, index=True)
    #saveasKML(intersected, "./data/kmls/intersectedPoints.kml")

>>>>>>> 82ea3e3872547d13a1f9cfbca9070dca768cadc1

    
    

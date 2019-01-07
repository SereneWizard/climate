import os
import glob
import re
import json
import requests
import itertools
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
import warnings
from shapely.geometry import Point
warnings.simplefilter(action='ignore', category=FutureWarning)


from get_daylight import *



class Climate:
    # Array of all lat and long coordinates 
    # .. of the NASA climate data
    latarray = np.linspace(-89.875, 89.875, 720)
    lonarray = np.linspace(0.125, 359.875, 1440)


    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.datalen = coordinates.shape[0]
        self.currentpoint = 0
    

    # Convert the given dates to the repository timepoints
    def date_to_timepoint(self, startyear, endyear):
        beginyear = 2006
        startpoint = (startyear-2006)*365 
        endpoint = (endyear+1-2006)*365 
        return (startpoint, endpoint)

    def get_dates(self, startyear, endyear, noleap=False):
        startdate = datetime(startyear, 1, 1)
        enddate = datetime(endyear, 12, 31)
        dt = enddate - startdate
        daterange = [startdate+timedelta(days=i) for i in range(dt.days+1)]
        if noleap:
            daterange = [x  for x in daterange  if not (x.month==2 and x.day==29)]
        return daterange



    def index2coordinates(self, xid, yid):
        return (
            Climate.lonarray[xid], 
            Climate.latarray[yid]
        )

    def coordinates2index(self, lon, lat):
        lon = 360+lon if lon<0 else lon
        return(
            (np.abs(Climate.lonarray- lon)).argmin(), 
            (np.abs(Climate.latarray- lat)).argmin()
        )



    def _download_data(self, x, y, mintime, maxtime, rcp, params=None):
        if params == None:
            params = ['pr', 'tasmax', 'tasmin']
        else:
            params = [params]
        timesteps = np.arange(mintime, maxtime, step=1500, dtype='int64')
        timesteps = np.append(timesteps, maxtime)
        outputDict = {}
        for param in params:
            print(param)
            i = 0
            starttime = timesteps[i]
            dataset = np.zeros(maxtime-mintime, dtype=float)
            n = 0
            while True:
                try:
                    endtime = timesteps[i+1]-1
                    i += 1
                    baseUrlPart = "https://dataserver.nccs.nasa.gov/thredds/dodsC/bypass/NEX-GDDP/bcsd/" 
                    modelUrlPart = "{}/r1i1p1/{}/CSIRO-Mk3-6-0.ncml.ascii?{}".format(rcp, param, param)
                    timeLocProperties =  "[{}:1:{}][{}:1:{}][{}:1:{}]".format(starttime, endtime, y, y, x, x)
                    finalurl = baseUrlPart + modelUrlPart + timeLocProperties
                    response = requests.get(finalurl)
                    print (response.status_code, " : ", finalurl)
                    dataText = response.text
                    regexSearch = '{}\.{}([^;]*){}\.time'.format(param, param, param)
                    #regexSearch = r'pr\.pr([^;]*)pr\.time'
                    data = re.search(regexSearch, dataText).group(1)
                    data = data.split(',')[1:]
                    data = [x.split('\n')[0] for x in data]
                    #data = [float(x) for x in data]
                    data = list(map(float, data))
                    for value in data:
                        dataset[n] = value
                        n += 1
                    starttime = endtime + 1
                except:
                    break
            if param == 'pr':
                outputDict[param] = dataset * 86400
            else :
                outputDict[param] = dataset - 273
        climateDF = pd.DataFrame.from_dict(outputDict, orient='columns')
        return (climateDF)

                
    def extract_data(self, startyear, endyear, rcp, fromlen=0, param=None):
        outdir = 'Output'
        if param == None: 
            p = ''
        else:
            p = param
        outpath = os.path.join(os.getcwd(), "data",'Climate_Data', p)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        startpoint, endpoint = self.date_to_timepoint(startyear, endyear)
        n = 0
        for index, row in self.coordinates.iterrows():
            if n < fromlen:
                n += 1 
                continue 
            lon, lat = row.x, row.y
            print(index, lat,lon)
            ix, iy = self.coordinates2index(lon, lat)
            elevation = self.get_elevation(lon, lat)
            df = self._download_data(ix, iy, startpoint, endpoint, rcp, param)
            df['Day_Date'] = self.get_dates(startyear, endyear, noleap=True)
            df['Day_Length'] = daylight(lat, lon, startyear, endyear)
            print(elevation)
            df['Longitude'] = lon
            df['Latitude'] = lat
            df['AI_ID'] = row.AI_ID
            nameMapper = {'pr': 'Rainfall', 
                'tasmax':'Max_Temp', 'tasmin':'Min_Temp'}
            df.rename(mapper=nameMapper, axis=1, inplace=True)
            el = float('%.2f' %(elevation))
            outfname = "{}_{}_{}_{}_{}_{}.csv".format(lon,lat, el, startyear, endyear, rcp, p)
            df.to_csv(os.path.join(outpath, outfname), 
                        index=False, float_format='%.3f')
            self.currentpoint += 1

    ### Other derivatives from the climate are calculated from the static methods

    @staticmethod
    def get_elevation(lon, lat):
        """
        This static method returns elevation when the latitude and longitude of a place is provided
        Arguments:
        lon = longitude in decimal degrees
        lat = latitude in decimal degrees
        Returns: 
        elevation = elevation in m
        """
        lon = lon-360 if lon >180 else lon
        apikey = 'AIzaSyB3QU0UeNQcQl0sGDGrlWbJErV7_wvp5hg'
        urlstring = "https://maps.googleapis.com/maps/api/elevation/json?locations={},{}&key={}".format(lat, lon, apikey)
        #print(urlstring)
        try:
            results = json.loads(requests.get(urlstring).text)
            results = results.get('results')
            if 0<len(results):
                elevation = results[0].get('elevation')
                resolution = results[0].get('resolution')
                #print("The resolution of the request is: {}".format(resolution))
                #print ("The elevation is: ", elevation)
                return elevation
        except ValueError as e:
            print ('JSON decode failed: {}'.format(e))


    @staticmethod
    def get_radiation(df, elevation):
        print("Calculating Solar Radiation...")
        df['Avg_Temp'] = (df['Max_Temp'] + df['Min_Temp'])/2
        df['dT'] = (df['Max_Temp'] - df['Min_Temp'])
        df['Dayofyear'] = df['Day_Date'].apply(lambda x: x.timetuple().tm_yday)
        df['HypsometricRatio'] = df.apply(lambda x: 
                                    1/((elevation*0.0065)/(x.Avg_Temp+273)+1)**5.257, 
                                        axis=1)

        df['DNI_Wm2'] = df.apply(lambda x: 1366.1 * (1 + 0.033 * np.cos(2*np.pi * x.Dayofyear/365)), 
                        axis=1)
        # This convesion from W/m2 to mm/day is taken from the following source: 
        # https://www.kimberly.uidaho.edu/water/fao56/fao56.pdf
        # FAO Irrigation and Drainage Paper No. 56, Page 212
        df['DNI_mmday'] = df['DNI_Wm2']*0.0864*0.408
        df['DNI_Whm2'] = df.apply(lambda x: x['DNI_Wm2']*24/np.pi, axis=1)
        df['Solar'] = df.apply(lambda x: 
                                0.20 * x['HypsometricRatio']**0.5 * x['DNI_Whm2'] * 
                                x['dT']**0.5, axis=1)
        df['ET'] = df.apply(lambda x: 
                                0.0135 * 0.20 * x['HypsometricRatio']**0.5 * x['DNI_mmday'] * 
                                x['dT']**0.5 * (x['Avg_Temp']+17.8), axis=1) 
        df.drop(['HypsometricRatio', 'DNI_Wm2', 'DNI_mmday', 'DNI_Whm2'], axis=1, inplace=True)

    # def _rh(self, t, td):
    #     rh = 100 * np.exp((17.625*td)/(243.04+td)) / np.exp((17.625*t)/(243.04+t))
    #     return rh

    @staticmethod
    def get_rh(dframe, n):
        print ("Calculating Relative Humidity...")
        _rh = lambda t, td: 100 * np.exp((17.625*td)/(243.04+td)) / np.exp((17.625*t)/(243.04+t))
        for row in range(dframe.shape[0]+1):
            if row >n:
                dframe.loc[row, 'FaoIndex'] = dframe.loc[row-n-1:row,:].apply(lambda x: x['Rainfall']/x['ET'], 
                                            axis=1).sum()
            else:
                dframe.loc[row, 'FaoIndex'] = 1
        dframe['DewPoint'] = dframe.apply(lambda x: x['Min_Temp']-2 if x['FaoIndex']>=3 else x['Min_Temp'], 
                                    axis=1)
        dframe['Min_RH'] = dframe.apply(lambda x: _rh(x['Max_Temp'], x['DewPoint']), axis=1)
        dframe['Max_RH'] = dframe.apply(lambda x: _rh(x['Min_Temp'], x['DewPoint']), axis=1)
        dframe.drop(['ET', 'FaoIndex', 'DewPoint', 'dT'], inplace=True, axis=1)



if __name__ == "__main__":
    df = pd.read_csv('easternidahoPoints2.csv')
    print(df.head())
    derivaties = 0
    if not derivaties:
        climate = Climate(df)
        climate.extract_data(2036, 2065, 'rcp45', fromlen=118)
    else: 
        fname = glob.glob(os.path.join("data", "Climate_Data", "*.csv"))[0]
        print(fname)
        df = pd.read_csv(fname) 
        print(df.shape)
        df['Day_Date'] = df['Day_Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        x, y = df.loc[0, 'Longitude'], df.loc[0, 'Latitude']
        elevation = Climate.get_elevation(x, y)
        Climate.get_radiation(df, elevation)
        Climate.get_rh(df, 10)
        print(df.head())



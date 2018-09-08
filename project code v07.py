import datetime
import  numpy as np, pandas as pd
import ast, os
import urllib
from bs4 import BeautifulSoup
#%matplotlib inline
import matplotlib.pyplot as plt
import gpxpy.geo

## creating all the needed functions first


# given an url, get list of holidays
def getHolidays (url):
    page=urllib.request.urlopen(url)
    soup=BeautifulSoup(page, 'html.parser')
    holidays=[]
    out=soup.findAll('div',attrs={'class':'col-md-3 black-link'})
    for i in out:
        holidays.append(str(i.text))

    return holidays


# flag as 1 the non-working days (public holidays of weekends)
def flagHolidays(x):
    # if it's saturday or sunday, it's automatically a "holiday"
    if datetime.datetime.isoweekday(x)==6 or  datetime.datetime.isoweekday(x)==7:
        return 1
    # otherwise, check if it's a public holiday
    if x in holidays:
        return 1
    else:
        return 0


def calc_duration(x):
    return (len(x)-1)*15

def get_hour(x):
    return int(str(x)[-8:-6])

def round_coordinates(x):
    temp=[]
    decimals=3
    # iterating over all the points
    for i in range(len(x)):

    # for each point, rounding both x and y coordinates
        first=round(x[i][0],decimals)
        second=round(x[i][1],decimals)
        temp.append([first,second])

    return temp



def to_binary(x):
    x=str(x)
    if x=='False':
        return 0
    elif x=='True':
        return 1
    else:
        # flag for error
        return -1

def start(x):
    return x[0]
def end(x):
    return x[-1]



def bin_hour_of_day(x):
    holiday=x['holiday']
    hour=x['hour']

    # bin for holidays
    if holiday==1:
        return 4

    if hour>6 and hour <=10:
        return 1
    elif hour>10 and  hour <=16:
        return 2
    elif hour >16 and hour <=20:
        return 1
    else:
        return 3


def start_same_end(x):
    if x[0]==x[-1]:
        return 1
    return 0



#########################
# getting public holidays for 2013 and 2014
url_2013='https://www.feiertagskalender.ch/index.php?geo=3516&jahr=2013&hl=en'
url_2014='https://www.feiertagskalender.ch/index.php?geo=3516&jahr=2014&hl=en'

holidays=getHolidays(url_2013)
holidays+=getHolidays(url_2014)


# converting into the same date format as the original dataset
temp=[]
for date in holidays:
    day=datetime.datetime.date(datetime.datetime.strptime(date, '%B %d %Y'))
    temp.append(day)
holidays=temp
#########################


## to load the entire train dataset 
df = pd.read_csv('train.csv')


## to work on a sample of the train set (for faster runtime)
#df = df.sample(frac=0.01)
#df.to_csv('C:/Users/chiar/Documents/Heavy files/494 python project/train_sampled.csv', sep=',')
df = pd.read_csv('train_sampled.csv')


## to load the test set. It will be treated as the train set (so in the variable df) since the test set has to go through all the 
## same pre-processing phases as the train set. You can run this code either on the train set to generate the optimized routes table 
## or on the test set to find the predictions
df = pd.read_csv('test.csv')

print("done loading file")


# initializing new columns to populate later
df['duration'] = np.zeros(len(df))
df['date'] = np.zeros(len(df))
df['hour'] = np.zeros(len(df))
df['datetime'] = np.zeros(len(df))
df['route'] = np.zeros(len(df))
df['route_string'] = np.zeros(len(df))
df['start'] = np.empty((len(df), 0)).tolist()
df['end'] =  np.empty((len(df), 0)).tolist()
df['holiday'] = np.zeros(len(df))
df['start_same_end'] = np.zeros(len(df))


# converting timestamp into date and hour
df['datetime'] = df['TIMESTAMP'].apply(datetime.datetime.fromtimestamp)
df['date'] = df['TIMESTAMP'].apply(datetime.date.fromtimestamp)
df['hour'] = df['datetime'].apply(get_hour)
# removing helper column
df.drop('datetime', axis=1, inplace=True)


# flagging holidays
df['holiday'] = df['date'].apply(flagHolidays)


# converting missing colum into 0/1
df['MISSING_DATA'] = df['MISSING_DATA'].apply(to_binary)


# removing rows with missing coordinates
df.drop(df[df['MISSING_DATA'] >0].index, inplace=True)



# converting polyline into an array of arrays
print("started converting polyline")
df['POLYLINE'] = df['POLYLINE'].apply(ast.literal_eval)
print("done converting polyline")


# computing duration in seconds
df['duration'] = df['POLYLINE'].apply(calc_duration)


# removing the rows that have duration <0 (their POLYLINE is empty or has only one point, so wrong rows)
df.drop(df[df['duration'] <=15].index, inplace=True)


# rounding coordinates
print("started rounding coordinates")
df['route'] = df['POLYLINE'].apply(round_coordinates)
print("done rounding coordinates")


# creating helper columns with end and start points of the trip
df['start'] = df['route'].apply(start)
df['end'] = df['route'].apply(end)


# checking and removing the rows where the taxi didn't move, meaning that start==end
df['start_same_end'] = df['route'].apply(start_same_end)
df.drop(df[df['start_same_end'] ==1].index, inplace=True)


## creating bins for hour of the day
print("started binning")
df['time_bin'] = np.zeros(len(df))
#df['time_bin'] = df.apply(lambda row: bin_hour_of_day(row['hour'], row['holiday']))
df['time_bin'] = df.apply(bin_hour_of_day, axis=1)
print("done binning")


#creating helper column with route converted to string
df['route_string'] = df['route'].apply(str)
df['start'] = df['start'].apply(str)
df['end'] = df['end'].apply(str)


##################
# to save the dataset as obtained so far, keeping all data types and colum names
#print ("saving pickle")
#df.to_pickle('output.pkl')
#print ("done saving pickle")
#
## to read back the complete file previously saved
#df=pd.read_pickle('output.pkl')

# to see where it saved the file
#print os.getcwd()
################


#computing the average duration by hour of the day and by route (output is a separate table)
print("started averaging")
#filtering only the hour/route pairs that have a number of occurrences above a threshold
min_count=2
avgtime_byhour_byroute=df.groupby(['time_bin','route_string','holiday','start','end'],as_index=False )['duration'].agg(['mean','count'],as_index=False)
avgtime_byhour_byroute=avgtime_byhour_byroute.drop(avgtime_byhour_byroute[avgtime_byhour_byroute['count']<min_count].index)
avgtime_byhour_byroute.to_csv('final.csv')

print("done averaging")


###########################
## lookup in table to make predictions/recommendations
## to run this code, the df must have been loaded as the Test set, not the Train set

# loading the table just created
table=pd.read_csv('final.csv')


# for the same start, end, day type, time bim, keep only the lowest duration one (most optimal route)
table=table.loc[table.groupby(['time_bin','holiday','start','end'], as_index=False)["mean"].idxmin()]

# do a join between test set and table with optimal routes to have the recommended route for the trips 
#that have a optimized route that was considered "reliable enough" to be included in the averages table
output = pd.merge(df, table, on=['time_bin','holiday','start','end'], how='inner', suffixes=('','_'))
output.to_csv('output.csv')



#######################################

#plotting the polylines on map


df = pd.read_csv('se_big.csv')


df['start'] = np.empty((len(df), 0)).tolist()
df['end'] =  np.empty((len(df), 0)).tolist()

df['POLYLINE'] = df['POLYLINE'].apply(ast.literal_eval)

 
    

df['start'] = df['POLYLINE'].apply(start)
df['end'] = df['POLYLINE'].apply(end)
se = df.filter(['start','end'], axis=1)
se.to_csv('se_big.csv')
df.drop('POLYLINE', axis=1, inplace=True)

pol_original = [[-8.61084, 41.145714], [-8.61084, 41.145705], [-8.610858, 41.145714], [-8.610561, 41.146092], [-8.609607, 41.146722], [-8.608788, 41.147208], [-8.609445, 41.147568], [-8.610228, 41.147586], [-8.610264, 41.147757], [-8.610291, 41.147811], [-8.611218, 41.147937], [-8.612118, 41.148063], [-8.613819, 41.148423], [-8.614287, 41.148342], [-8.614638, 41.147721], [-8.615412, 41.147289], [-8.616618, 41.147244], [-8.617689, 41.147235], [-8.617734, 41.147208], [-8.617959, 41.147208], [-8.617995, 41.147226], [-8.617986, 41.147343], [-8.618265, 41.148459], [-8.619102, 41.148693], [-8.619669, 41.148279], [-8.619741, 41.148207], [-8.620038, 41.14782], [-8.620407, 41.147433], [-8.620416, 41.147424], [-8.621181, 41.14746], [-8.621388, 41.147487], [-8.621469, 41.147469], [-8.621856, 41.147559], [-8.622198, 41.147613], [-8.622306, 41.147442], [-8.622315, 41.147442], [-8.622324, 41.147424], [-8.622333, 41.147433], [-8.622369, 41.147397], [-8.622423, 41.147172], [-8.622441, 41.147154]]
pol1_predicted = [[-8.61084, 41.145714], [-8.61084, 41.145705], [-8.610858, 41.145714], [-8.610561, 41.146092], [-8.609607, 41.146722], [-8.608788, 41.147208], [-8.609445, 41.147568], [-8.610228, 41.147586], [-8.610264, 41.147757], [-8.610291, 41.147811], [-8.611218, 41.147937], [-8.612118, 41.148063], [-8.613819, 41.148423], [-8.614287, 41.148342], [-8.614638, 41.147721], [-8.615412, 41.147289], [-8.616618, 41.147244], [-8.617689, 41.147235], [-8.617734, 41.147208], [-8.617959, 41.147208], [-8.617995, 41.147226], [-8.617986, 41.147343], [-8.618265, 41.148459], [-8.619102, 41.148693], [-8.619669, 41.148279], [-8.619741, 41.148207], [-8.620038, 41.14782], [-8.620407, 41.147433], [-8.620416, 41.147424], [-8.621181, 41.14746], [-8.621388, 41.147487], [-8.621469, 41.147469], [-8.621856, 41.147559], [-8.622198, 41.147613], [-8.622306, 41.147442], [-8.622315, 41.147442], [-8.622324, 41.147424], [-8.622333, 41.147433], [-8.622369, 41.147397], [-8.622423, 41.147172], [-8.622441, 41.147154]]
pol2_original = [[-8.644383, 41.175351], [-8.644842, 41.175864], [-8.644131, 41.17689], [-8.64252, 41.177871], [-8.641278, 41.177232], [-8.640477, 41.177619], [-8.639928, 41.178375], [-8.638983, 41.178933], [-8.638146, 41.178888], [-8.638146, 41.178699], [-8.638173, 41.178483], [-8.638173, 41.178438], [-8.638182, 41.17851]]
pol2_predicted = [[-8.64432, 41.175486], [-8.645031, 41.176053], [-8.643798, 41.176998], [-8.641998, 41.177745], [-8.6409, 41.177169], [-8.640333, 41.177907], [-8.639289, 41.178726], [-8.638146, 41.178879], [-8.638164, 41.178501]]
lat = list()
lon = list()
for i in range(len(pol)):
    lat.append(pol[i][0])
    lon.append(pol[i][1])

lat1 = list()
lon1 = list()
for i in range(len(pol1)):
    lat1.append(pol1[i][0])
    lon1.append(pol1[i][1])
    

df['start_lat'], df['start_long'] = df['start'].str.split(',', 1).str
df['end_lat'], df['end_long'] = df['end'].str.split(',', 1).str

print df


df.start_lat = df.start_lat.str.strip('[')
df.start_long = df.start_long.str.strip(']')
df.end_lat = df.end_lat.str.strip('[')
df.end_long = df.end_long.str.strip(']')
df.start_lat = df.start_lat.astype(float)
df.start_long = df.start_long.astype(float)
df.end_lat = df.end_lat.astype(float)
df.end_long = df.end_long.astype(float)

gmap = gmplot.GoogleMapPlotter(41.1496100, -8.6109900, len(pol))

#gmap.scatter(df.start_long,df.start_lat, 'r', 100, marker=False)
#gmap.scatter(df.end_long,df.end_lat, 'b', 100, marker=False)
gmap.plot(lon,lat, 'b', 1, marker=False)
gmap.plot(lon1,lat1, 'g', 1, marker=False)
gmap.draw('map.html')
gmap.draw('map1.html')


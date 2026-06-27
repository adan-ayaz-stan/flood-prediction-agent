## Data Decision Exploration
Exploring datasets available so I can effectively make an AI ML project with a clear goal in mind.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

## Earthquake Dataset


```python
eq_df = pd.read_csv('../data/earthquake_data.csv')
print("Earthquake DataFrame Columns:")
print(eq_df.columns)
print("Total Earthquake Records:", len(eq_df))
print(eq_df.info())
```

    Earthquake DataFrame Columns:
    Index(['title', 'magnitude', 'date_time', 'cdi', 'mmi', 'alert', 'tsunami',
           'sig', 'net', 'nst', 'dmin', 'gap', 'magType', 'depth', 'latitude',
           'longitude', 'location', 'continent', 'country'],
          dtype='str')
    Total Earthquake Records: 782
    <class 'pandas.DataFrame'>
    RangeIndex: 782 entries, 0 to 781
    Data columns (total 19 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   title      782 non-null    str    
     1   magnitude  782 non-null    float64
     2   date_time  782 non-null    str    
     3   cdi        782 non-null    int64  
     4   mmi        782 non-null    int64  
     5   alert      415 non-null    str    
     6   tsunami    782 non-null    int64  
     7   sig        782 non-null    int64  
     8   net        782 non-null    str    
     9   nst        782 non-null    int64  
     10  dmin       782 non-null    float64
     11  gap        782 non-null    float64
     12  magType    782 non-null    str    
     13  depth      782 non-null    float64
     14  latitude   782 non-null    float64
     15  longitude  782 non-null    float64
     16  location   777 non-null    str    
     17  continent  206 non-null    str    
     18  country    484 non-null    str    
    dtypes: float64(6), int64(5), str(8)
    memory usage: 185.1 KB
    None


## Airports Dataset


```python
ap_df = pd.read_csv('../data/airports.csv')
print("Airports DataFrame Columns:")
print(ap_df.columns)
print("Total Airports Records:", len(ap_df))
print(ap_df.info())
```

    Airports DataFrame Columns:
    Index(['id', 'ident', 'type', 'name', 'latitude_deg', 'longitude_deg',
           'elevation_ft', 'continent', 'iso_country', 'iso_region',
           'municipality', 'scheduled_service', 'icao_code', 'iata_code',
           'gps_code', 'local_code', 'home_link', 'wikipedia_link', 'keywords'],
          dtype='str')
    Total Airports Records: 85549
    <class 'pandas.DataFrame'>
    RangeIndex: 85549 entries, 0 to 85548
    Data columns (total 19 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 85549 non-null  int64  
     1   ident              85549 non-null  str    
     2   type               85549 non-null  str    
     3   name               85549 non-null  str    
     4   latitude_deg       85549 non-null  float64
     5   longitude_deg      85549 non-null  float64
     6   elevation_ft       70695 non-null  float64
     7   continent          45968 non-null  str    
     8   iso_country        85246 non-null  str    
     9   iso_region         85549 non-null  str    
     10  municipality       80843 non-null  str    
     11  scheduled_service  85549 non-null  str    
     12  icao_code          10164 non-null  str    
     13  iata_code          9056 non-null   str    
     14  gps_code           44289 non-null  str    
     15  local_code         36098 non-null  str    
     16  home_link          4719 non-null   str    
     17  wikipedia_link     16702 non-null  str    
     18  keywords           21358 non-null  str    
    dtypes: float64(3), int64(1), str(15)
    memory usage: 19.0 MB
    None


## Cities Dataset


```python
city_df = pd.read_csv('../data/worldcities.csv')
print("World Cities DataFrame Columns:")
print(city_df.columns)
print("Total World Cities Records:", len(city_df))
print(city_df.info())
```

    World Cities DataFrame Columns:
    Index(['city', 'city_ascii', 'lat', 'lng', 'country', 'iso2', 'iso3',
           'admin_name', 'capital', 'population', 'id'],
          dtype='str')
    Total World Cities Records: 49992
    <class 'pandas.DataFrame'>
    RangeIndex: 49992 entries, 0 to 49991
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   city        49992 non-null  str    
     1   city_ascii  49990 non-null  str    
     2   lat         49992 non-null  float64
     3   lng         49992 non-null  float64
     4   country     49992 non-null  str    
     5   iso2        49956 non-null  str    
     6   iso3        49992 non-null  str    
     7   admin_name  49814 non-null  str    
     8   capital     16086 non-null  str    
     9   population  48778 non-null  float64
     10  id          49992 non-null  int64  
    dtypes: float64(3), int64(1), str(7)
    memory usage: 6.3 MB
    None


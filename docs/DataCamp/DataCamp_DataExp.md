---
layout: default
title: Data Exploration
parent: DataCamp Works
nav_order: 3
---

# Data Exploration
## Analyzing Crime in Los Angeles

Los Angeles, California 😎. The City of Angels. Tinseltown. The Entertainment Capital of the World! 

Known for its warm weather, palm trees, sprawling coastline, and Hollywood, along with producing some of the most iconic films and songs. 
However, as with any highly populated city, it isn't always glamorous and there can be a large volume of crime. 

We have been asked to support the Los Angeles Police Department (LAPD) by analyzing crime data to identify patterns in criminal behavior. 
They plan to use our insights to allocate resources effectively to tackle various crimes in different areas.

### The Data

They have provided us with a single dataset to use. A summary and preview are provided below.
It is a modified version of the original data, which is publicly available from Los Angeles Open Data.

crimes.csv

| Column     | Description              |
|------------|--------------------------|
| `'DR_NO'` | Division of Records Number: Official file number made up of a 2-digit year, area ID, and 5 digits. |
| `'Date Rptd'` | Date reported - MM/DD/YYYY. |
| `'DATE OCC'` | Date of occurrence - MM/DD/YYYY. |
| `'TIME OCC'` | In 24-hour military time. |
| `'AREA NAME'` | The 21 Geographic Areas or Patrol Divisions are also given a name designation that references a landmark or the surrounding community that it is responsible for. For example, the 77th Street Division is located at the intersection of South Broadway and 77th Street, serving neighborhoods in South Los Angeles. |
| `'Crm Cd Desc'` | Indicates the crime committed. |
| `'Vict Age'` | Victim's age in years. |
| `'Vict Sex'` | Victim's sex: `F`: Female, `M`: Male, `X`: Unknown. |
| `'Vict Descent'` | Victim's descent:<ul><li>`A` - Other Asian</li><li>`B` - Black</li><li>`C` - Chinese</li><li>`D` - Cambodian</li><li>`F` - Filipino</li><li>`G` - Guamanian</li><li>`H` - Hispanic/Latin/Mexican</li><li>`I` - American Indian/Alaskan Native</li><li>`J` - Japanese</li><li>`K` - Korean</li><li>`L` - Laotian</li><li>`O` - Other</li><li>`P` - Pacific Islander</li><li>`S` - Samoan</li><li>`U` - Hawaiian</li><li>`V` - Vietnamese</li><li>`W` - White</li><li>`X` - Unknown</li><li>`Z` - Asian Indian</li> |
| `'Weapon Desc'` | Description of the weapon used (if applicable). |
| `'Status Desc'` | Crime status. |
| `'LOCATION'` | Street address of the crime. |

```python
# Re-run this cell
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
crimes = pd.read_csv("crimes.csv", parse_dates=["Date Rptd", "DATE OCC"], dtype={"TIME OCC": str})
crimes.head()
```

|   | DR_NO      | Date Rptd  | DATE OCC   | TIME OCC | AREA NAME    | Crm Cd Desc         | Vict Age | Vict Sex | Vict Descent | Weapon Desc | Status Desc  | LOCATION                            |
| - | ---------- | ---------- | ---------- | -------- | ------------ | -------------------- | -------- | -------- | ------------ | ----------- | ------------ | ----------------------------------- |
| 0 | 220314085  | 2022-07-22 | 2020-05-12 | 1110     | Southwest    | THEFT OF IDENTITY    | 27       | F        | B            | NaN         | Invest Cont  | 2500 S  SYCAMORE AV                  |
| 1 | 222013040  | 2022-08-06 | 2020-06-04 | 1620     | Olympic      | THEFT OF IDENTITY    | 60       | M        | H            | NaN         | Invest Cont  | 3300 SAN MARINO ST                  |
| 2 | 220614831  | 2022-08-18 | 2020-08-17 | 1200     | Hollywood    | THEFT OF IDENTITY    | 28       | M        | H            | NaN         | Invest Cont  | 1900 TRANSIENT                       |
| 3 | 231207725  | 2023-02-27 | 2020-01-27 | 0635     | 77th Street  | THEFT OF IDENTITY    | 37       | M        | H            | NaN         | Invest Cont  | 6200 4TH AV                          |
| 4 | 220213256  | 2022-07-14 | 2020-07-14 | 0900     | Rampart      | THEFT OF IDENTITY    | 79       | M        | B            | NaN         | Invest Cont  | 1200 W 7TH ST                        |


Which hour has the highest frequency of crimes? 

```python
crimes['HOUR OCC'] = pd.to_datetime(crimes['TIME OCC'], format="%H%M").dt.hour
peak_crime_hour = crimes['HOUR OCC'].value_counts().index[0]
print(peak_crime_hour)
```

    12

```python
sns.countplot(data=crimes, x='HOUR OCC')
plt.show()
```

![HoursCrime](../../../docs/DataCamp/images/ny_crimes.png)


Which area has the largest frequency of night crimes (crimes committed between 10pm and 3:59am)? 

```python
crimes['TIME OCC converted'] = pd.to_datetime(crimes['TIME OCC'], format="%H%M")
crimes_time = crimes.set_index('TIME OCC converted')
night_time = crimes_time.between_time('22:00:00', '03:59:00', inclusive='both')
peak_night_crime_location = night_time['AREA NAME'].value_counts().index[0]
peak_night_crime_location
```




    'Central'



Identify the number of crimes committed against victims of different age groups. 

```python
bins = [0, 17, 25, 34, 44, 54, 64, float('inf')]
labels = ["0-17", "18-25", "26-34", "35-44", "45-54", "55-64", "65+"]
crimes['Age Group'] = pd.cut(crimes['Vict Age'], bins=bins, labels=labels)
victim_ages = crimes['Age Group'].value_counts()
victim_ages
```




    26-34    47470
    35-44    42157
    45-54    28353
    18-25    28291
    55-64    20169
    65+      14747
    0-17      4528
    Name: Age Group, dtype: int64




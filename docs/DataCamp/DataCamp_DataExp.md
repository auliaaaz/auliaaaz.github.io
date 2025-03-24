---
layout: default
title: Data Exploration
parent: DataCamp Works
nav_order: 3
---

# Data Exploration
{: .no_toc }
<br/>

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }


1. TOC
{:toc}
</details>
---

## Analyzing Crime in Los Angeles

Los Angeles, California 😎. The City of Angels. Tinseltown. The Entertainment Capital of the World! 

Known for its warm weather, palm trees, sprawling coastline, and Hollywood, along with producing some of the most iconic films and songs. 
However, as with any highly populated city, it isn't always glamorous and there can be a large volume of crime. 

The Los Angeles Police Department (LAPD) has requested support in analyzing crime data to identify patterns in criminal behavior. The insights gained will assist in effectively allocating resources to address various crimes across different areas.

They have provided a single dataset to use. A summary and preview are provided below.
It is a modified version of the original data, which is publicly available from Los Angeles Open Data.

| **Column**           | **Description**                                                                                                                                                                                                                                               |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `'DR_NO'`            | Division of Records Number: Official file number consisting of a 2-digit year, area ID, and 5 digits.                                                                                                                 |
| `'Date Rptd'`        | Date reported in MM/DD/YYYY format.                                                                                                                                                                                      |
| `'DATE OCC'`         | Date of occurrence in MM/DD/YYYY format.                                                                                                                                                                                  |
| `'TIME OCC'`         | Time of occurrence in 24-hour military time.                                                                                                                                                                              |
| `'AREA NAME'`        | The 21 Geographic Areas or Patrol Divisions, named based on landmarks or surrounding communities. For example, the 77th Street Division is responsible for neighborhoods in South Los Angeles.                           |
| `'Crm Cd Desc'`      | Description of the crime committed.                                                                                                                                                                                       |
| `'Vict Age'`         | Victim's age in years.                                                                                                                                                                                                    |
| `'Vict Sex'`         | Victim's sex (`F`: Female, `M`: Male, `X`: Unknown).                                                                                                                                                                      |
| `'Vict Descent'`     | Victim's descent:                                                                                                                                                                                                         |
|                      | - `A` - Other Asian                                                                                                                                                                                                       |
|                      | - `B` - Black                                                                                                                                                                                                             |
|                      | - `C` - Chinese                                                                                                                                                                                                           |
|                      | - `D` - Cambodian                                                                                                                                                                                                         |
|                      | - `F` - Filipino                                                                                                                                                                                                          |
|                      | - `G` - Guamanian                                                                                                                                                                                                         |
|                      | - `H` - Hispanic/Latin/Mexican                                                                                                                                                                                            |
|                      | - `I` - American Indian/Alaskan Native                                                                                                                                                                                    |
|                      | - `J` - Japanese                                                                                                                                                                                                          |
|                      | - `K` - Korean                                                                                                                                                                                                            |
|                      | - `L` - Laotian                                                                                                                                                                                                           |
|                      | - `O` - Other                                                                                                                                                                                                             |
|                      | - `P` - Pacific Islander                                                                                                                                                                                                  |
|                      | - `S` - Samoan                                                                                                                                                                                                            |
|                      | - `U` - Hawaiian                                                                                                                                                                                                          |
|                      | - `V` - Vietnamese                                                                                                                                                                                                        |
|                      | - `W` - White                                                                                                                                                                                                             |
|                      | - `X` - Unknown                                                                                                                                                                                                           |
|                      | - `Z` - Asian Indian                                                                                                                                                                                                      |
| `'Weapon Desc'`      | Description of the weapon used, if applicable.                                                                                                                                                                           |
| `'Status Desc'`      | Status of the crime investigation.                                                                                                                                                                                       |
| `'LOCATION'`         | Street address where the crime occurred.                                                                                                                                                                                 |



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

## Customer Analytics: Preparing Data for Modeling

A common problem when creating models to generate business value from data is that the datasets can be so large that it can take days for the model to generate predictions. Ensuring that your dataset is stored as efficiently as possible is crucial for allowing these models to run on a more reasonable timescale without having to reduce the size of the dataset.

Training Data Ltd. has provided access to customer_train.csv, a subset of their customer dataset containing anonymized student information and indicators of whether students were seeking new job opportunities during training. The objective is to create a proof-of-concept for a more efficient storage solution, enabling improved data management and supporting predictive modeling to connect students with prospective recruiters.

| Column                   | Description                                                                      |
|------------------------- |--------------------------------------------------------------------------------- |
| `student_id`             | A unique ID for each student.                                                    |
| `city`                   | A code for the city the student lives in.                                        |
| `city_development_index` | A scaled development index for the city.                                         |
| `gender`                 | The student's gender.                                                            |
| `relevant_experience`    | An indicator of the student's work relevant experience.                          |
| `enrolled_university`    | The type of university course enrolled in (if any).                              |
| `education_level`        | The student's education level.                                                   |
| `major_discipline`       | The educational discipline of the student.                                       |
| `experience`             | The student's total work experience (in years).                                  |
| `company_size`           | The number of employees at the student's current employer.                       |
| `company_type`           | The type of company employing the student.                                       |
| `last_new_job`           | The number of years between the student's current and previous jobs.             |
| `training_hours`         | The number of hours of training completed.                                       |
| `job_change`             | An indicator of whether the student is looking for a new job (`1`) or not (`0`). |


```python
# Import necessary libraries
import pandas as pd

# Load the dataset
ds_jobs = pd.read_csv("customer_train.csv")

# View the dataset
ds_jobs.head()
```

| student_id | city      | city_development_index | gender | relevant_experience     | enrolled_university   | education_level | major_discipline  | experience | company_size | company_type     | last_new_job | training_hours | job_change |
|-------------|-----------|------------------------|--------|--------------------------|------------------------|-----------------|-------------------|------------|--------------|------------------|--------------|----------------|------------|
| 8949        | city_103  | 0.920                  | Male   | Has relevant experience  | no_enrollment          | Graduate        | STEM              | >20        | NaN          | NaN              | 1            | 36             | 1.0        |
| 29725       | city_40   | 0.776                  | Male   | No relevant experience   | no_enrollment          | Graduate        | STEM              | 15         | 50-99        | Pvt Ltd          | >4           | 47             | 0.0        |
| 11561       | city_21   | 0.624                  | NaN    | No relevant experience   | Full time course       | Graduate        | STEM              | 5          | NaN          | NaN              | never        | 83             | 0.0        |
| 33241       | city_115  | 0.789                  | NaN    | No relevant experience   | NaN                    | Graduate        | Business Degree   | <1         | NaN          | Pvt Ltd          | never        | 52             | 1.0        |
| 666         | city_162  | 0.767                  | Male   | Has relevant experience  | no_enrollment          | Masters         | STEM              | >20        | 50-99        | Funded Startup   | 4            | 8              | 0.0        |


```python
# Create a copy of ds_jobs for transforming
ds_jobs_transformed = ds_jobs.copy()
```

```python
# Checking data types
ds_jobs_transformed.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19158 entries, 0 to 19157
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   student_id              19158 non-null  int64  
 1   city                    19158 non-null  object 
 2   city_development_index  19158 non-null  float64
 3   gender                  14650 non-null  object 
 4   relevant_experience     19158 non-null  object 
 5   enrolled_university     18772 non-null  object 
 6   education_level         18698 non-null  object 
 7   major_discipline        16345 non-null  object 
 8   experience              19093 non-null  object 
 9   company_size            13220 non-null  object 
 10  company_type            13018 non-null  object 
 11  last_new_job            18735 non-null  object 
 12  training_hours          19158 non-null  int64  
 13  job_change              19158 non-null  float64
dtypes: float64(2), int64(2), object(10)
memory usage: 2.0+ MB
```

```python
# print unique value for object type
for i in ds_jobs_transformed.columns[ds_jobs_transformed.dtypes=="object"]:
    print(f"{i} : \n{ds_jobs_transformed[i].value_counts()}")
```

```
city : 
city_103    4355
city_21     2702
city_16     1533
city_114    1336
city_160     845
            ... 
city_129       3
city_111       3
city_121       3
city_140       1
city_171       1
Name: city, Length: 123, dtype: int64
gender : 
Male      13221
Female     1238
Other       191
Name: gender, dtype: int64
relevant_experience : 
Has relevant experience    13792
No relevant experience      5366
Name: relevant_experience, dtype: int64
enrolled_university : 
no_enrollment       13817
Full time course     3757
Part time course     1198
Name: enrolled_university, dtype: int64
education_level : 
Graduate          11598
Masters            4361
High School        2017
Phd                 414
Primary School      308
Name: education_level, dtype: int64
major_discipline : 
STEM               14492
Humanities           669
Other                381
Business Degree      327
Arts                 253
No Major             223
Name: major_discipline, dtype: int64
experience : 
>20    3286
5      1430
4      1403
3      1354
6      1216
2      1127
7      1028
10      985
9       980
8       802
15      686
11      664
14      586
1       549
<1      522
16      508
12      494
13      399
17      342
19      304
18      280
20      148
Name: experience, dtype: int64
company_size : 
50-99        3083
100-499      2571
10000+       2019
10-49        1471
1000-4999    1328
<10          1308
500-999       877
5000-9999     563
Name: company_size, dtype: int64
company_type : 
Pvt Ltd                9817
Funded Startup         1001
Public Sector           955
Early Stage Startup     603
NGO                     521
Other                   121
Name: company_type, dtype: int64
last_new_job : 
1        8040
>4       3290
2        2900
never    2452
4        1029
3        1024
Name: last_new_job, dtype: int64
```

Columns containing categories with only two factors must be stored as Booleans (bool).

```python
# relevant_experience and job_change only have 2 factors
ds_jobs_transformed['relevant_experience'] = ds_jobs_transformed['relevant_experience'].str.strip()
ds_jobs_transformed['relevant_experience'] = ds_jobs_transformed['relevant_experience'].astype('bool')
ds_jobs_transformed['relevant_experience'] = ds_jobs_transformed['relevant_experience'].replace({'Has relevant experience': True, 'No relevant experience': False})

cat_job = {1: True, 0: False}
ds_jobs_transformed['job_change'] = ds_jobs_transformed['job_change'].map(cat_job)
ds_jobs_transformed['job_change'] = ds_jobs_transformed['job_change'].astype('bool')
```

Columns containing integers only must be stored as 32-bit integers (int32).

```python
# student_id and training_hours stored as int64
ds_jobs_transformed['student_id'] = ds_jobs_transformed['student_id'].astype('int32')
ds_jobs_transformed['training_hours'] = ds_jobs_transformed['training_hours'].astype('int32')
```

Columns containing floats must be stored as 16-bit floats (float16).

```python
# city_development_index and job_change is in float64
ds_jobs_transformed['city_development_index'] = ds_jobs_transformed['city_development_index'].astype('float16')
```

Columns containing nominal categorical data must be stored as the category data type.

```python
for i in ds_jobs_transformed[['city', 'gender', 'major_discipline', 'company_type']]:
    ds_jobs_transformed[i] = ds_jobs_transformed[i].astype('category')
ds_jobs_transformed[['city', 'gender', 'major_discipline', 'company_type']].info()
```

Columns containing ordinal categorical data must be stored as ordered categories, and not mapped to numerical values, with an order that reflects the natural order of the column.

```python
for i in ds_jobs_transformed[['education_level', 'enrolled_university' ,'experience', 'company_size', 'last_new_job']]:
    ds_jobs_transformed[i] = ds_jobs_transformed[i].astype('category')
    categories = ds_jobs_transformed[i].cat.categories.tolist()
    ds_jobs_transformed[i] = ds_jobs_transformed[i].cat.reorder_categories(new_categories=categories, ordered=True)
    
ds_jobs_transformed[['education_level', 'enrolled_university', 'experience', 'company_size', 'last_new_job']].info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19158 entries, 0 to 19157
Data columns (total 5 columns):
 #   Column               Non-Null Count  Dtype   
---  ------               --------------  -----   
 0   education_level      18698 non-null  category
 1   enrolled_university  18772 non-null  category
 2   experience           19093 non-null  category
 3   company_size         13220 non-null  category
 4   last_new_job         18735 non-null  category
dtypes: category(5)
memory usage: 95.3 KB
```

The DataFrame should be filtered to only contain students with 10 or more years of experience at companies with at least 1000 employees, as their recruiter base is suited to more experienced professionals at enterprise companies.

```python
ds_jobs_transformed['experience'].unique()
```

```
['>20', '15', '5', '<1', '11', ..., '6', '9', '8', '20', NaN]
Length: 23
Categories (22, object): ['1' < '10' < '11' < '12' ... '8' < '9' < '<1' < '>20']
```

```python
ds_jobs_transformed = ds_jobs_transformed[(ds_jobs_transformed['experience'].isin(['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'])) & ds_jobs_transformed['company_size'].isin(['1000-4999', '5000-9999', '10000+'])]
```

```python
ds_jobs_transformed.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2201 entries, 9 to 19143
Data columns (total 14 columns):
 #   Column                  Non-Null Count  Dtype   
---  ------                  --------------  -----   
 0   student_id              2201 non-null   int32   
 1   city                    2201 non-null   category
 2   city_development_index  2201 non-null   float16 
 3   gender                  1821 non-null   category
 4   relevant_experience     2201 non-null   bool    
 5   enrolled_university     2185 non-null   category
 6   education_level         2184 non-null   category
 7   major_discipline        2097 non-null   category
 8   experience              2201 non-null   category
 9   company_size            2201 non-null   category
 10  company_type            2144 non-null   category
 11  last_new_job            2184 non-null   category
 12  training_hours          2201 non-null   int32   
 13  job_change              2201 non-null   bool    
dtypes: bool(2), category(9), float16(1), int32(2)
memory usage: 69.5 KB
```

The memory usage is reduced from 2 MB to 69.5 KB.

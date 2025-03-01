---
layout: default
title: Data Manipulation
parent: DataCamp Works
nav_order: 1
---
## Investigating Netflix Movies
**Netflix**! What started in 1997 as a DVD rental service has since exploded into one of the largest entertainment and media companies.

Perform exploratory data analysis on the netflix_data.csv data to understand more about movies from the 1990s decade.

### **netflix_data.csv**
| Column | Description |
|:-------|:------------|
| `show_id` | The ID of the show |
| `type` | Type of show |
| `title` | Title of the show |
| `director` | Director of the show |
| `cast` | Cast of the show |
| `country` | Country of origin |
| `date_added` | Date added to Netflix |
| `release_year` | Year of Netflix release |
| `duration` | Duration of the show in minutes |
| `description` | Description of the show |
| `genre` | Show genre |


```python
# Importing pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Read in the Netflix CSV as a DataFrame
netflix_df = pd.read_csv("netflix_data.csv")
```


```python
# Start coding here! Use as many cells as you like
netflix_df.head(4)
```
|   | show_id | type  | title | director           | cast                                      | country       | date_added        | release_year | duration | description                                      | genre          |
|:--|:--------|:------|:------|:-------------------|:-------------------------------------------|:-------------|:------------------|:-------------|:---------|:-------------------------------------------------|:--------------|
| 0 | s2      | Movie | 7:19  | Jorge Michel Grau  | Demián Bichir, Héctor Bonilla, Oscar Serrano, ... | Mexico       | December 23, 2016 | 2016         | 93       | After a devastating earthquake hits Mexico City... | Dramas        |
| 1 | s3      | Movie | 23:59 | Gilbert Chan      | Tedd Chan, Stella Chung, Henley Hii, Lawrence ... | Singapore    | December 20, 2018 | 2011         | 78       | When an army recruit is found dead, his fellow...  | Horror Movies |
| 2 | s4      | Movie | 9     | Shane Acker       | Elijah Wood, John C. Reilly, Jennifer Connelly... | United States | November 16, 2017 | 2009         | 80       | In a postapocalyptic world, rag-doll robots hi... | Action        |
| 3 | s5      | Movie | 21    | Robert Luketic    | Jim Sturgess, Kevin Spacey, Kate Bosworth, Aar... | United States | January 1, 2020   | 2008         | 123      | A brilliant group of students become card-coun... | Dramas        |

What was the most frequent movie duration in the 1990s? Save an approximate answer as an integer called duration (use 1990 as the decade's start year).
```python
duration = netflix_df[(netflix_df["release_year"] >= 1990) & (netflix_df["release_year"] <= 1999)]["duration"].value_counts(ascending=False)
duration = duration.index[0]
duration
```
    94
  -> The most frequent movie duration in the 1990s is 1 hour 34 minutes

A movie is considered short if it is less than 90 minutes. Count the number of short action movies released in the 1990s and save this integer as short_movie_count.
```python
short_movie_count = netflix_df[
    (netflix_df["release_year"] >= 1990) & (netflix_df["release_year"] <= 1999)]
short_movie_count = short_movie_count[(short_movie_count["duration"]<90) & (short_movie_count["genre"]=="Action")].value_counts()
short_movie_count = len(short_movie_count)
short_movie_count
```
    7
  -> The number of short action movies released in the 1990s is 7

## Exploring NYC Public School Test Result Scores
Every year, American high school students take SATs, which are standardized tests intended to measure literacy, numeracy, and writing skills. There are three sections - reading, math, and writing, each with a **maximum score of 800 points**. 
These tests are extremely important for students and colleges, as they play a pivotal role in the admissions process.

```python
# Re-run this cell 
import pandas as pd

# Read in the data
schools = pd.read_csv("schools.csv")

# Preview the data
schools.head()
```
|   | school_name                                          | borough   | building_code | average_math | average_reading | average_writing | percent_tested |
|:--|:-----------------------------------------------------|:----------|:-------------|:-------------|:---------------|:---------------|:---------------|
| 0 | New Explorations into Science, Technology and ...   | Manhattan | M022         | 657          | 601            | 601            | NaN            |
| 1 | Essex Street Academy                                | Manhattan | M445         | 395          | 411            | 387            | 78.9           |
| 2 | Lower Manhattan Arts Academy                        | Manhattan | M445         | 418          | 428            | 415            | 65.1           |
| 3 | High School for Dual Language and Asian Studies    | Manhattan | M445         | 613          | 453            | 463            | 95.9           |
| 4 | Henry Street School for International Studies      | Manhattan | M056         | 410          | 406            | 381            | 59.7           |

1. Which NYC schools have the best math results? The best math results are at least 80% of the *maximum possible score of 800* for math.

```python
best_math_schools = schools[["school_name", "average_math"]]
best_math_schools = best_math_schools[best_math_schools["average_math"] >= (0.8) * (800)]
best_math_schools = best_math_schools.sort_values(by="average_math", ascending=False)
best_math_schools.reset_index()
```

|   | index | school_name                                                 | average_math |
|:--|:------|:------------------------------------------------------------|:-------------|
| 0 | 88    | Stuyvesant High School                                      | 754          |
| 1 | 170   | Bronx High School of Science                                | 714          |
| 2 | 93    | Staten Island Technical High School                         | 711          |
| 3 | 365   | Queens High School for the Sciences at York College        | 701          |
| 4 | 68    | High School for Mathematics, Science, and Engineering      | 683          |
| 5 | 280   | Brooklyn Technical High School                              | 682          |
| 6 | 333   | Townsend Harris High School                                 | 680          |
| 7 | 174   | High School of American Studies at Lehman College          | 669          |
| 8 | 0     | New Explorations into Science, Technology and Math         | 657          |
| 9 | 45    | Eleanor Roosevelt High School                               | 641          |


2. What are the top 10 performing schools based on the combined SAT scores? top_10_schools containing the "school_name" and a new column named "total_SAT", with results ordered by "total_SAT" in descending order ("total_SAT" being the sum of math, reading, and writing scores).

```python
schools["total_SAT"] = schools["average_math"] + schools["average_reading"] + schools["average_writing"]
top_10_schools = schools[["school_name", "total_SAT"]]
top_10_schools = (top_10_schools.sort_values(by="total_SAT", ascending=False))
top_10_schools = (top_10_schools.reset_index(drop=True)).iloc[0:10, :]
top_10_schools
```

|       | school_name                                                    | total_SAT |
|:------|:---------------------------------------------------------------|:----------|
| 0     | Stuyvesant High School                                        | 2144      |
| 1     | Bronx High School of Science                                  | 2041      |
| 2     | Staten Island Technical High School                           | 2041      |
| 3     | High School of American Studies at Lehman College            | 2013      |
| 4     | Townsend Harris High School                                   | 1981      |
| 5     | Queens High School for the Sciences at York College          | 1947      |
| 6     | Bard High School Early College                                | 1914      |
| 7     | Brooklyn Technical High School                                | 1896      |
| 8     | Eleanor Roosevelt High School                                 | 1889      |
| 9     | High School for Mathematics, Science, and Engineering        | 1889      |


3. Which single borough has the largest standard deviation in the combined SAT score?

```python
schools["num_schools"] = schools.groupby("borough")["school_name"].transform('count')
schools["average_SAT"] = round(schools.groupby("borough")["total_SAT"].transform('mean'), 2)
schools["std_SAT"] = round(schools.groupby("borough")["total_SAT"].transform('std'), 2)
largest_std_dev = (schools.sort_values(by="std_SAT", ascending=False)
.drop_duplicates(subset="borough").iloc[0])
largest_std_dev = pd.DataFrame({"borough":[largest_std_dev["borough"]],
                                'num_schools':[largest_std_dev["num_schools"]],
                               'average_SAT':[largest_std_dev["average_SAT"]],
                               'total_SAT':[largest_std_dev["total_SAT"]],
                               'std_SAT':[largest_std_dev["std_SAT"]]})
largest_std_dev
```
|   | borough   | num_schools | average_SAT | total_SAT | std_SAT |
|:--|:----------|:-----------|:------------|:----------|:--------|
| 0 | Manhattan | 89         | 1340.13     | 1859      | 230.29  |


Manhattan is a borough in NYC that have the largest standard deviation of total SAT score.

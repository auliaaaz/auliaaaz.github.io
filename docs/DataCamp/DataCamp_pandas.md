---
layout: default
title: Data Manipulation
parent: DataCamp Works
nav_order: 1
---

# Data Manipulation with Pandas 
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

## Investigating Netflix Movies
**Netflix**! What started in 1997 as a DVD rental service has since exploded into one of the largest entertainment and media companies.

Perform exploratory data analysis on the netflix_data.csv data to understand more about movies from the 1990s decade.

### **netflix_data.csv**
```
| Column         | Description                     |
|----------------|---------------------------------|
| `show_id`      | The ID of the show              |
| `type`         | Type of show                    |
| `title`        | Title of the show               |
| `director`     | Director of the show            |
| `cast`         | Cast of the show                |
| `country`      | Country of origin               |
| `date_added`   | Date added to Netflix           |
| `release_year` | Year of Netflix release         |
| `duration`     | Duration of the show in minutes |
| `description`  | Description of the show         |
| `genre`        | Show genre                      |
```



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
<div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>duration</th>
      <th>description</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s2</td>
      <td>Movie</td>
      <td>7:19</td>
      <td>Jorge Michel Grau</td>
      <td>Demián Bichir, Héctor Bonilla, Oscar Serrano, ...</td>
      <td>Mexico</td>
      <td>December 23, 2016</td>
      <td>2016</td>
      <td>93</td>
      <td>After a devastating earthquake hits Mexico Cit...</td>
      <td>Dramas</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s3</td>
      <td>Movie</td>
      <td>23:59</td>
      <td>Gilbert Chan</td>
      <td>Tedd Chan, Stella Chung, Henley Hii, Lawrence ...</td>
      <td>Singapore</td>
      <td>December 20, 2018</td>
      <td>2011</td>
      <td>78</td>
      <td>When an army recruit is found dead, his fellow...</td>
      <td>Horror Movies</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s4</td>
      <td>Movie</td>
      <td>9</td>
      <td>Shane Acker</td>
      <td>Elijah Wood, John C. Reilly, Jennifer Connelly...</td>
      <td>United States</td>
      <td>November 16, 2017</td>
      <td>2009</td>
      <td>80</td>
      <td>In a postapocalyptic world, rag-doll robots hi...</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s5</td>
      <td>Movie</td>
      <td>21</td>
      <td>Robert Luketic</td>
      <td>Jim Sturgess, Kevin Spacey, Kate Bosworth, Aar...</td>
      <td>United States</td>
      <td>January 1, 2020</td>
      <td>2008</td>
      <td>123</td>
      <td>A brilliant group of students become card-coun...</td>
      <td>Dramas</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s6</td>
      <td>TV Show</td>
      <td>46</td>
      <td>Serdar Akar</td>
      <td>Erdal Beşikçioğlu, Yasemin Allen, Melis Birkan...</td>
      <td>Turkey</td>
      <td>July 1, 2017</td>
      <td>2016</td>
      <td>1</td>
      <td>A genetics professor experiments with a treatm...</td>
      <td>International TV</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4807</th>
      <td>s7779</td>
      <td>Movie</td>
      <td>Zombieland</td>
      <td>Ruben Fleischer</td>
      <td>Jesse Eisenberg, Woody Harrelson, Emma Stone, ...</td>
      <td>United States</td>
      <td>November 1, 2019</td>
      <td>2009</td>
      <td>88</td>
      <td>Looking to survive in a world taken over by zo...</td>
      <td>Comedies</td>
    </tr>
    <tr>
      <th>4808</th>
      <td>s7781</td>
      <td>Movie</td>
      <td>Zoo</td>
      <td>Shlok Sharma</td>
      <td>Shashank Arora, Shweta Tripathi, Rahul Kumar, ...</td>
      <td>India</td>
      <td>July 1, 2018</td>
      <td>2018</td>
      <td>94</td>
      <td>A drug dealer starts having doubts about his t...</td>
      <td>Dramas</td>
    </tr>
    <tr>
      <th>4809</th>
      <td>s7782</td>
      <td>Movie</td>
      <td>Zoom</td>
      <td>Peter Hewitt</td>
      <td>Tim Allen, Courteney Cox, Chevy Chase, Kate Ma...</td>
      <td>United States</td>
      <td>January 11, 2020</td>
      <td>2006</td>
      <td>88</td>
      <td>Dragged from civilian life, a former superhero...</td>
      <td>Children</td>
    </tr>
    <tr>
      <th>4810</th>
      <td>s7783</td>
      <td>Movie</td>
      <td>Zozo</td>
      <td>Josef Fares</td>
      <td>Imad Creidi, Antoinette Turk, Elias Gergi, Car...</td>
      <td>Sweden</td>
      <td>October 19, 2020</td>
      <td>2005</td>
      <td>99</td>
      <td>When Lebanon's Civil War deprives Zozo of his ...</td>
      <td>Dramas</td>
    </tr>
    <tr>
      <th>4811</th>
      <td>s7784</td>
      <td>Movie</td>
      <td>Zubaan</td>
      <td>Mozez Singh</td>
      <td>Vicky Kaushal, Sarah-Jane Dias, Raaghav Chanan...</td>
      <td>India</td>
      <td>March 2, 2019</td>
      <td>2015</td>
      <td>111</td>
      <td>A scrappy but poor boy worms his way into a ty...</td>
      <td>Dramas</td>
    </tr>
  </tbody>
</table>
<p>4812 rows × 11 columns</p>
</div>

**What was the most frequent movie duration in the 1990s?** 
```python
duration = netflix_df[(netflix_df["release_year"] >= 1990) & (netflix_df["release_year"] <= 1999)]["duration"].value_counts(ascending=False)
duration = duration.index[0]
duration
```
    94

The most frequent movie duration in the 1990s is 1 hour 34 minutes

A movie is considered short if it is less than 90 minutes. **Count the number of short action movies released in the 1990s and save this integer as short_movie_count.**
```python
short_movie_count = netflix_df[
    (netflix_df["release_year"] >= 1990) & (netflix_df["release_year"] <= 1999)]
short_movie_count = short_movie_count[(short_movie_count["duration"]<90) & (short_movie_count["genre"]=="Action")].value_counts()
short_movie_count = len(short_movie_count)
short_movie_count
```
    7
    
The number of short action movies released in the 1990s is 7

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
<div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>school_name</th>
      <th>borough</th>
      <th>building_code</th>
      <th>average_math</th>
      <th>average_reading</th>
      <th>average_writing</th>
      <th>percent_tested</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Explorations into Science, Technology and ...</td>
      <td>Manhattan</td>
      <td>M022</td>
      <td>657</td>
      <td>601</td>
      <td>601</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Essex Street Academy</td>
      <td>Manhattan</td>
      <td>M445</td>
      <td>395</td>
      <td>411</td>
      <td>387</td>
      <td>78.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lower Manhattan Arts Academy</td>
      <td>Manhattan</td>
      <td>M445</td>
      <td>418</td>
      <td>428</td>
      <td>415</td>
      <td>65.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>High School for Dual Language and Asian Studies</td>
      <td>Manhattan</td>
      <td>M445</td>
      <td>613</td>
      <td>453</td>
      <td>463</td>
      <td>95.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Henry Street School for International Studies</td>
      <td>Manhattan</td>
      <td>M056</td>
      <td>410</td>
      <td>406</td>
      <td>381</td>
      <td>59.7</td>
    </tr>
  </tbody>
</table>
</div>

**Which NYC schools have the best math results?** The best math results are at least 80% of the *maximum possible score of 800* for math.

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


**What are the top 10 performing schools based on the combined SAT scores?** top_10_schools containing the "school_name" and a new column named "total_SAT", with results ordered by "total_SAT" in descending order ("total_SAT" being the sum of math, reading, and writing scores).

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


**Which single borough has the largest standard deviation in the combined SAT score**

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
<div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>borough</th>
      <th>num_schools</th>
      <th>average_SAT</th>
      <th>total_SAT</th>
      <th>std_SAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manhattan</td>
      <td>89</td>
      <td>1340.13</td>
      <td>1859</td>
      <td>230.29</td>
    </tr>
  </tbody>
</table>
</div>

Manhattan is a borough in NYC that have the largest standard deviation of total SAT score.

---
layout: default
title: Data Visualization 
parent: DataCamp Works
nav_order: 2
---

# Data Visualization with Matplotlib (Seaborn)
The Nobel Prize has been among the most prestigious international awards since 1901. Each year, awards are bestowed in chemistry, literature, physics, physiology or medicine, economics, and peace. In addition to the honor, prestige, and substantial prize money, the recipient also gets a gold medal with an image of Alfred Nobel (1833 - 1896), who established the prize.

The Nobel Foundation has made a dataset available of all prize winners from the outset of the awards from 1901 to 2023. The dataset used in this project is from the Nobel Prize API and is available in the `nobel.csv` file in the `data` folder.

In this project, we will explore and answer several questions related to this prizewinning data. And to explore further questions.

```python
# Loading in required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import re
```


```python
df = pd.read_csv('data/nobel.csv', parse_dates=['birth_date', 'death_date'])
```


```python
df.head()
```

|   | year | category   | prize                                          | motivation                                         | prize\_share | laureate\_id | laureate\_type | full\_name                   | birth\_date | birth\_city       | birth\_country   | sex  | organization\_name | organization\_city | organization\_country | death\_date | death\_city | death\_country |
| - | ---- | ---------- | ---------------------------------------------- | -------------------------------------------------- | ------------ | ------------ | -------------- | ---------------------------- | ----------- | ----------------- | ---------------- | ---- | ------------------ | ------------------ | --------------------- | ----------- | ----------- | -------------- |
| 0 | 1901 | Chemistry  | The Nobel Prize in Chemistry 1901              | "in recognition of the extraordinary services ..." | 1/1          | 160          | Individual     | Jacobus Henricus van 't Hoff | 1852-08-30  | Rotterdam         | Netherlands      | Male | Berlin University  | Berlin             | Germany               | 1911-03-01  | Berlin      | Germany        |
| 1 | 1901 | Literature | The Nobel Prize in Literature 1901             | "in special recognition of his poetic composit..." | 1/1          | 569          | Individual     | Sully Prudhomme              | 1839-03-16  | Paris             | France           | Male | NaN                | NaN                | NaN                   | 1907-09-07  | Châtenay    | France         |
| 2 | 1901 | Medicine   | The Nobel Prize in Physiology or Medicine 1901 | "for his work on serum therapy, especially its..." | 1/1          | 293          | Individual     | Emil Adolf von Behring       | 1854-03-15  | Hansdorf (Lawice) | Prussia (Poland) | Male | Marburg University | Marburg            | Germany               | 1917-03-31  | Marburg     | Germany        |
| 3 | 1901 | Peace      | The Nobel Peace Prize 1901                     | NaN                                                | 1/2          | 462          | Individual     | Jean Henry Dunant            | 1828-05-08  | Geneva            | Switzerland      | Male | NaN                | NaN                | NaN                   | 1910-10-30  | Heiden      | Switzerland    |
| 4 | 1901 | Peace      | The Nobel Peace Prize 1901                     | NaN                                                | 1/2          | 463          | Individual     | Frédéric Passy               | 1822-05-20  | Paris             | France           | Male | NaN                | NaN                | NaN                   | 1912-06-12  | Paris       | France         |


```python
df = pd.read_csv('data/nobel.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 18 columns):
     #   Column                Non-Null Count  Dtype 
    ---  ------                --------------  ----- 
     0   year                  1000 non-null   int64 
     1   category              1000 non-null   object
     2   prize                 1000 non-null   object
     3   motivation            912 non-null    object
     4   prize_share           1000 non-null   object
     5   laureate_id           1000 non-null   int64 
     6   laureate_type         1000 non-null   object
     7   full_name             1000 non-null   object
     8   birth_date            968 non-null    object
     9   birth_city            964 non-null    object
     10  birth_country         969 non-null    object
     11  sex                   970 non-null    object
     12  organization_name     736 non-null    object
     13  organization_city     735 non-null    object
     14  organization_country  735 non-null    object
     15  death_date            596 non-null    object
     16  death_city            579 non-null    object
     17  death_country         585 non-null    object
    dtypes: int64(2), object(16)
    memory usage: 140.8+ KB


The most commonly awarded gender and birth country


```python
top_gender = df['sex'].value_counts().idxmax()
top_country = df['birth_country'].value_counts().idxmax()
print(f'The most awarded gender: {top_gender} \nThe most awarded contry: {top_country}' )
```

    The most awarded gender: Male 
    The most awarded contry: United States of America


Which decade had the highest ratio of US-born Nobel Prize winners to total winners in all categories?


```python
# create decade column
df['decade'] = (df['year'] // 10) * 10

# count winner each decade
total_winner_decade = df.groupby('decade')['year'].count()

# count USA winner
us_winner_decade = df[df['birth_country']=='United States of America'].groupby('decade')['year'].count()

ratio = (us_winner_decade/total_winner_decade)
max_decade_usa = ratio.idxmax()
print(f"Decade with highest ration: {max_decade_usa}'s")

# create a new DataFrame for plotting
plot_data = pd.DataFrame({
    'decade': total_winner_decade.index,
    'total_winner_decade': total_winner_decade.values,
    'us_winner_decade': us_winner_decade.reindex(total_winner_decade.index, fill_value=0).values
})

sns.relplot(data=plot_data, x='total_winner_decade', y='us_winner_decade', kind='line')
plt.show()
```

    Decade with highest ration: 2000's
    PNG


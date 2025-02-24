---
layout: default
title: Google Play Store Recommendation
parent: Exploratory Data Analysis
nav_order: 2
---

## Google Play Store Recommendation Apps
The Google Play Store dataset used here consists of two main files:
1. googleplaystore.csv: This dataset contains a list of Android apps along with details such as genre, user ratings, and more.
2. googleplaystore_users_reviews.csv: This dataset includes user reviews (and some sentiment scores) which are associated with the apps listed in the first dataset.
Our goal with this data is to perform various analysis techniques to ultimately recommend the top 5 apps for download.

```python
# Import libraries
import pandas as pd
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Read dataset
google_play_store = pd.read_csv('/content/drive/MyDrive/my_cv/googleplaystore - googleplaystore.csv')
google_play_store.head(3)
```

| App                                              | Category       | Rating | Reviews | Size  | Installs   | Type  | Price | Content Rating | Genres                     | Last Updated     | Current Ver | Android Ver     |
|--------------------------------------------------|--------------|--------|---------|------|-----------|------|------|----------------|----------------------------|------------------|-------------|----------------|
| Photo Editor & Candy Camera & Grid & ScrapBook  | ART_AND_DESIGN | 4.1    | 159     | 19M  | 10,000+   | Free  | 0    | Everyone       | Art & Design               | January 7, 2018  | 1.0.0       | 4.0.3 and up   |
| Coloring book moana                              | ART_AND_DESIGN | 3.9    | 967     | 14M  | 500,000+  | Free  | 0    | Everyone       | Art & Design; Pretend Play | January 15, 2018 | 2.0.0       | 4.0.3 and up   |
| U Launcher Lite – FREE Live Cool Themes, Hide   | ART_AND_DESIGN | 4.7    | 87510   | 8.7M | 5,000,000+ | Free  | 0    | Everyone       | Art & Design               | August 1, 2018   | 1.2.4       | 4.0.3 and up   |



## Data Cleaning

In this section, we clean the Google Play Store dataset to prepare it for analysis.

```python
# Remove additional spaces
clean_google_play_store = google_play_store.applymap(
    lambda x: x.strip() if isinstance(x, str) else x)

# Sort row based on the Last Updated to remove duplicated later based on that
clean_google_play_store = clean_google_play_store.sort_values(by='Last Updated')

# Drop duplicates for the App column
# This step ensures that only one entry per app is kept, discarding older versions
clean_google_play_store = google_play_store.drop_duplicates(keep="last")
clean_google_play_store = clean_google_play_store.reset_index(drop=True)

# Drop duplicates especially for App column, to only keep the unique updated App data
clean_google_play_store = google_play_store.drop_duplicates(subset='App', keep="last")
clean_google_play_store = clean_google_play_store.reset_index(drop=True)

clean_google_play_store.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9660 entries, 0 to 9659
    Data columns (total 13 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   App             9660 non-null   object 
     1   Category        9659 non-null   object 
     2   Rating          8197 non-null   float64
     3   Reviews         9660 non-null   int64  
     4   Size            9660 non-null   object 
     5   Installs        9660 non-null   object 
     6   Type            9659 non-null   object 
     7   Price           9660 non-null   object 
     8   Content Rating  9660 non-null   object 
     9   Genres          9659 non-null   object 
     10  Last Updated    9660 non-null   object 
     11  Current Ver     9652 non-null   object 
     12  Android Ver     9658 non-null   object 
    dtypes: float64(1), int64(1), object(11)
    memory usage: 981.2+ KB

```python
# Remove additional character in object types that supposed to be integer type
# Converting kb size into mb
def convert_kb_to_mb(size):
  if 'k' in size:
    kb_size = float(size.replace('k', ''))
    mb_size = kb_size / 1024
  if 'M' in size:
    mb_size = float(size.replace('M', ''))
  else:
    mb_size = 'nan'
  return mb_size

# Applying the convert kb to mb function to Size column
clean_google_play_store['Size'] = clean_google_play_store['Size'].apply(convert_kb_to_mb)

# Removing additional character with replacing it with empty string in Installs column
clean_google_play_store['Installs'] = clean_google_play_store['Installs'].replace(r'[+, ,]', '', regex=True)
clean_google_play_store['Installs'] = clean_google_play_store['Installs'].astype(int)

# Removing additional character with replacing it with empty string in Price column
clean_google_play_store['Price'] = clean_google_play_store['Price'].replace(r'[$]', '', regex=True)
clean_google_play_store['Price'] = clean_google_play_store['Price'].astype(float)
```

## Data Exploration and Visualization

We start by grouping the data by category to analyze the number of installations per category.
This helps us understand which categories have the most installed apps.

Then, we create a bar chart to visually represent the total number of installs by category.

```python
# Group by category according to number of instalation
installs_avg_category = clean_google_play_store.groupby('Category')['Installs'].sum().reset_index()
installs_avg_category = installs_avg_category.sort_values(by='Installs', ascending=False)

# Create a bar chart to visualize the total installs by category
num_category = px.bar(installs_avg_category, x='Category',  y='Installs', title='Total Installs by Category')
num_category.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Total%20Installs%20by%20Category.html" 
        width="700" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>

The chart shows that the categories of 'Communication,' 'Games,' 'Family,' and 'Tools' boast the highest installation rates among users. Since our goal is to recommend applications that offer unique experiences and aren't overly mainstream, we'll factor in the number of installations to filter our list of applications. This will serve as our starting point to select a niche or explore new possibilities for users to explore and install.

Next, we will also visualize the rating of each category because we wanted those list application are applications that user most like with reputable rating.

```python
# Group by category according to ratings
ratings_avg_category = clean_google_play_store.groupby('Category')['Rating'].mean().reset_index()
ratings_avg_category = ratings_avg_category.sort_values(by='Rating', ascending=False)

# Create a box plot to visualize the distribution of ratings for each category
rating_category = px.box(clean_google_play_store, x='Category',  y='Rating', title='Ratings Distribution by Category')
rating_category.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Ratings%20Distribution%20by%20Category.html" 
        width="700" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>

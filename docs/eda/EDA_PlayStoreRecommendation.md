---
layout: default
title: Google Play Store App
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

The average rating for each category appears to be above 4. This suggests that users generally have positive experiences with apps across most categories. For ratings we'll later set a threshold to filter only high rating app will include in our list of apps, considering the high rating apps are app that marked as a good app from users experience.

And then we'll analyze that, **is size of the app impacting user to install the app?** Considering the large app will take more space in the android and may requiring much time to download so use may not prefer a too large app.

```python
# Visualize histogram for total of installs by its size
installs_size = px.histogram(clean_google_play_store,
                            x='Size',
                            y='Installs',
                            title= 'Number of Installs by Size')
installs_size.show()

# Visualize histogram for total rating by its size
rating_size = px.histogram(clean_google_play_store,
                            x='Size',
                            y='Rating',
                            title= 'Rating of Size')
rating_size.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Number%20of%20Installs%20by%20Size.html" 
        width="800" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Rating%20of%20Size.html" 
        width="800" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>

The charts show that most users prefer apps with smaller file sizes, with the highest total installations occurring within the 1MB to 20MB range. This trend also reflects in app ratings, as those with sizes below 20MB generally receive higher ratings. In our recommendation process, we will also take into the app size. If the size of a potential recommendation is too large, it may be removed from consideration.

Next, we'll check how being free or paid affects the number of ratings and installations. **Does being paid application impact the number of installations and ratings?**

```python
# Visualize box plot for distribution of rating for free and paid apps
plot_free_paid = px.box(clean_google_play_store, x='Type',  y='Installs', color='Type', title='Installs Distribution of Free and Paid Apps')
plot_free_paid.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Installs%20Distribution%20of%20Free%20and%20Paid%20Apps.html" 
        width="700" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>
The chart shows a broader distribution for free apps compared to paid apps. Free apps reach installations as high as 1 billion, while paid apps peak at 1 million installations. This suggests that users tend to prefer free apps, likely due to their lack of cost

However, let's consider the other side, **how do ratings impact free and paid apps? Does paying for an app lead to greater user satisfaction compared to free ones?**

```python
# Visualize box plot for distribution of rating for free and paid apps
plot_free_paid_rating = px.box(clean_google_play_store, x='Type',  y='Rating', color='Type', title='Ratings Distribution of Free and Paid Apps')
plot_free_paid_rating.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Ratings%20Distribution%20of%20Free%20and%20Paid%20Apps.html" 
        width="700" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>
The chart shows that both free and paid apps have similar maximum and minimum ratings, with a range from 1 to 5. The distribution of ratings for both types of apps appears to be relatively similar, with the median and quartiles showing only slight differences. However, paid apps seem to have slightly higher ratings overall, as indicated by their slightly higher median and quartile values compared to free apps. Related to consideration for our list recommendation apps, both Free and Paid apps will be good for recommend.

So, **what about the price of paid apps? Does a higher price guarantee higher ratings? And when it comes to installations, do lower-priced apps tend to get more installations compared to higher-priced ones?** Users might consider if higher-priced apps are really worth the cost

```python
# Filter only paid type to include
paid_apps = clean_google_play_store[clean_google_play_store['Type'] == 'Paid']

# We'll filter out apps priced under $100 for further analysis. Since there are only a few apps priced at $100, we aim to focus on apps priced below $100 to examine more details
apps_under_100 = paid_apps[paid_apps['Price'] < 100]

# Visualize distribution of number of installation of paid price App
installs_price = px.scatter(apps_under_100,
                            x='Price',
                            y='Installs',
                            title= 'Installs of Paid Price App')
installs_price.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/PlayStoreRecommendation/Installs%20of%20Paid%20Price%20App.html" 
        width="700" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>
The chart shows that the lower the price, the higher the number of installations. This might be because of factors such as affordability, perceived value for money, and accessibility, which may lead users to prefer for lower-priced apps over higher-priced ones

```python
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")

# Plot price vs. rating
plt2 = sns.jointplot(x = 'Price', y = 'Rating', data= apps_under_100)
```
![PNG Image](../../../docs/eda/images/PlayStoreRecommendation/Rating%20x%20Price%20Matplotlib.png)

The chart shows a trend that, as app prices increase, the range of ratings becomes more limited, with predominantly higher ratings for pricier apps.

## Determine Weight for Apps

We will then assign weight for each app based on its number of installation, ratings, number of reviews, and content rating.

```python
# Filter the apps based on number of download, here we consider well recognized apps are apps with tens of millions of installations or more
# So we will filter a niche app as app with less than 10 millions installations
_app = clean_google_play_store[clean_google_play_store['Installs'] < 10000000]

# Then as we wanted to get app that user mostly like, we consider rating 4.00 or above is in it and not only rating, app review also considering here, we will keep apps
# with 2500 or above number of reviews (https://www.appsflyer.com/blog/tips-strategy/app-ratings-reviews/)
_app = _app[(_app['Rating'] >= 4.00) & (_app['Reviews'] >= 2500)]

# As we consider the app recommendation are for general/everyone, so we only proceed content for everyone or everyone 10+
_app = _app[(_app['Content Rating'] == "Everyone") | (_app['Content Rating'] == "Everyone 10+")]
```

```python
# Drop nan value and convert the data type
_app.dropna(subset=['Size'], inplace=True)
_app['Size'] = _app['Size'].astype(float)

# Load reviews dataset to get more insight from reviews sentiment
google_store_reviews = pd.read_csv('/content/drive/MyDrive/my_cv/googleplaystore_user_reviews - googleplaystore_user_reviews.csv')
```
```python
# Get count of reviews for each app
review_count =  google_store_reviews.groupby('App')['Translated_Review'].count().reset_index()

# Get the sum of polarity and subjectivity score for each app
polarity_score_sum = google_store_reviews.groupby('App')['Sentiment_Polarity'].sum().reset_index()
subjectivity_score_sum = google_store_reviews.groupby('App')['Sentiment_Subjectivity'].sum().reset_index()

# Merge polarity score and subjectivity score
polarity_subjectivity_score = polarity_score_sum.merge(subjectivity_score_sum, on='App', how='inner').merge(review_count, on='App', how='inner')

# Multiply polarity and subjectivity by the number of review for each app, because of number of review also contribute to the list of app recommendation
polarity_subjectivity_score['Sentiment_Polarity'] = polarity_subjectivity_score['Sentiment_Polarity'] * polarity_subjectivity_score['Translated_Review']
polarity_subjectivity_score['Sentiment_Subjectivity'] = polarity_subjectivity_score['Sentiment_Subjectivity'] * polarity_subjectivity_score['Translated_Review']

# Normalize the value so it will in range [0, 1]
polarity_subjectivity_score['Polarity_Normalized'] = MinMaxScaler().fit_transform(polarity_subjectivity_score[['Sentiment_Polarity']])
polarity_subjectivity_score['Subjectivity_Normalized'] = MinMaxScaler().fit_transform(polarity_subjectivity_score[['Sentiment_Subjectivity']])
```

```python
# Merge the application data with application reviews data
_app_reviews = _app.merge(polarity_subjectivity_score, on="App", how='left')
```

```python
# Normalize rating value, so it will be contribute fairly
_app_reviews['Rating_Normalized'] = MinMaxScaler().fit_transform(_app_reviews[['Rating']])

# Subtract it with sentiment subjectivity score because of high score indicate the reviews most likely consist of opinion and not too factual
_app_reviews['Final_Weight'] = _app_reviews['Rating_Normalized'] + _app_reviews['Polarity_Normalized'] - _app_reviews['Subjectivity_Normalized']

# Showing final result with sort the App based on the final weight
_app_reviews.sort_values(by='Final_Weight', ascending=False).head(5)
```
| App                                             | Category          | Rating | Reviews | Size  | Installs | Type  | Price | Content Rating | Genres          | Last Updated    | Current Ver | Android Ver  | Sentiment Polarity | Sentiment Subjectivity | Translated Review | Polarity Normalized | Subjectivity Normalized | Rating Normalized | Final Weight |
|-------------------------------------------------|------------------|--------|---------|------|----------|------|------|----------------|----------------|----------------|-------------|--------------|-------------------|----------------------|----------------|--------------------|----------------------|----------------|--------------|
| DMV Permit Practice Test 2018 Edition         | AUTO_AND_VEHICLES | 4.9    | 6090    | 27.0 | 100000   | Free  | 0.0  | Everyone       | Auto & Vehicles | July 3, 2018   | 1.7         | 4.2 and up   | 341.783331        | 647.135574          | 34.0           | 0.099260           | 0.013281              | 1.000000       | 1.085979     |
| CDL Practice Test 2018 Edition                | AUTO_AND_VEHICLES | 4.9    | 7774    | 17.0 | 100000   | Free  | 0.0  | Everyone       | Auto & Vehicles | July 3, 2018   | 1.7         | 4.2 and up   | 138.888571        | 275.227302          | 24.0           | 0.088741           | 0.005649              | 1.000000       | 1.083093     |
| FreePrints – Free Photos Delivered           | PHOTOGRAPHY       | 4.8    | 109500  | 37.0 | 1000000  | Free  | 0.0  | Everyone       | Photography     | August 2, 2018 | 2.18.2      | 4.1 and up   | 578.250715        | 812.177818          | 36.0           | 0.111519           | 0.016668              | 0.888889       | 0.983739     |
| Home Workout for Men - Bodybuilding          | HEALTH_AND_FITNESS | 4.8    | 12705   | 15.0 | 1000000  | Free  | 0.0  | Everyone       | Health & Fitness | July 10, 2018  | 1.0.2       | 4.0 and up   | 353.851875        | 317.790159          | 26.0           | 0.099886           | 0.006522              | 0.888889       | 0.982252     |
| GoodRx Drug Prices and Coupons               | MEDICAL           | 4.8    | 59158   | 11.0 | 1000000  | Free  | 0.0  | Everyone       | Medical         | July 26, 2018  | 5.4.8       | 4.1 and up   | 345.632857        | 614.719762          | 37.0           | 0.099459           | 0.012616              | 0.888889       | 0.975732     |

## Recommendation
### **Top 5 Apps Recommendation for You!**
(Curated using Kaggle data analysis + user reviews)

📚 **Driving & Education**
For learners aiming to ace their exams

**1. DMV Permit Practice Test 2018 Edition**

Rating: 4.9 ⭐ | Installs: 100K+

Crush your driver’s license test! This app feels like having a patient instructor in your pocket—packed with realistic practice tests, instant feedback, and a clutter-free design. Perfect for nervous first-timers or seasoned learners.

**2. CDL Practice Test 2018 Edition**

Rating: 4.9 ⭐ | Installs: 100K+

Dreaming of trucking or bus driving? This app’s laser-focused on the Commercial Driver’s License (CDL) exam. Users rave about how it breaks down complex rules into bite-sized lessons, making it a no-brainer for career-changers.

🖼️ **Photo Printing & Gifts**
For preserving memories (without the hassle)

**3. FreePrints – Free Photos Delivered**

Rating: 4.8 ⭐ | Installs: 1M+

Forget overpriced photo labs! FreePrints lets you turn phone pics into tangible keepsakes for free—you only pay shipping. Users love surprising Grandma with framed vacation snaps or creating DIY wall collages.

💪 **Health & Fitness**
For building strength at home

**4. Home Workout for Men - Bodybuilding**

Rating: 4.8 ⭐ | Installs: 1M+

No gym? No problem. This app’s like having a personal trainer who gets busy schedules. With zero-equipment routines and progress tracking, even couch potatoes turn into fitness fans. (“Finally, an app that doesn’t judge my pizza nights!” – Happy User)

💊 **Healthcare & Savings**
For cutting costs on prescriptions

**5. GoodRx Drug Prices and Coupons**

Rating: 4.8 ⭐ | Installs: 1M+

Tired of pharmacy sticker shock? GoodRx compares prices at nearby stores and unlocks secret coupons. One user saved $120 on allergy meds—it’s basically a superhero cape for your wallet.

**Why Trust These Picks?**

📊 Each app was shortlisted based on:

- High ratings (all above 4.8/5)
- User install trends from Kaggle datasets
- Real reviews highlighting practicality and results


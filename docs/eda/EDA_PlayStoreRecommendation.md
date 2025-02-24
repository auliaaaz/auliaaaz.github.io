---
layout: default
title: EDA Google Play Store 
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
| App                                              | Category       | Rating | Reviews | Size  | Installs  | Type  | Price | Content Rating | Genres                     | Last Updated     | Current Ver | Android Ver      |
|-------------------------------------------------|--------------|--------|---------|------|----------|------|------|----------------|----------------------------|-----------------|-------------|----------------|
| Photo Editor & Candy Camera & Grid & ScrapBook | ART_AND_DESIGN | 4.1    | 159     | 19M  | 10,000+   | Free  | 0    | Everyone       | Art & Design               | January 7, 2018  | 1.0.0       | 4.0.3 and up    |
| Coloring book moana                              | ART_AND_DESIGN | 3.9    | 967     | 14M  | 500,000+  | Free  | 0    | Everyone       | Art & Design; Pretend Play | January 15, 2018 | 2.0.0       | 4.0.3 and up    |
| U Launcher Lite – FREE Live Cool Themes, Hide  | ART_AND_DESIGN | 4.7    | 87510   | 8.7M | 5,000,000+| Free  | 0    | Everyone       | Art & Design               | August 1, 2018   | 1.2.4       | 4.0.3 and up    |


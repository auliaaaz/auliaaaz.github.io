---
layout: default
title: UK Online Retail
parent: Exploratory Data Analysis
nav_order: 1
---

## Introduction

Problem Statement:
* The Online Retail dataset encompasses transactions from a UK-based online retailer between 01/12/2009 and 09/12/2011.
* The goal is to extract actionable insights to optimize customer retention, improve sales strategies, and enhance business performance.

Objectives:
1. Understand the customer base through segmentation and behavior analysis.
2. Identify key trends and patterns in sales data over time.
3. Recommend strategies to improve customer experience and boost revenue.


## Load and Preprocessing Data
```python
! python --version
```

    Python 3.10.12

```python
pip install ucimlrepo
```

    /usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning:
    
    `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.


    Requirement already satisfied: ucimlrepo in /usr/local/lib/python3.10/dist-packages (0.0.7)
    Requirement already satisfied: pandas>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ucimlrepo) (2.2.2)
    Requirement already satisfied: certifi>=2020.12.5 in /usr/local/lib/python3.10/dist-packages (from ucimlrepo) (2024.12.14)
    Requirement already satisfied: numpy>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.0->ucimlrepo) (1.26.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.0->ucimlrepo) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.0->ucimlrepo) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.0.0->ucimlrepo) (2024.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->ucimlrepo) (1.17.0)

```python
import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning)

```

    /usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning:
    
    `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.


```python
# fetch dataset
online_retail = fetch_ucirepo(id=352)

# metadata
print(online_retail.metadata)

# variable information
print(online_retail.variables)
```

    {'uci_id': 352, 'name': 'Online Retail', 'repository_url': 'https://archive.ics.uci.edu/dataset/352/online+retail', 'data_url': 'https://archive.ics.uci.edu/static/public/352/data.csv', 'abstract': 'This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.', 'area': 'Business', 'tasks': ['Classification', 'Clustering'], 'characteristics': ['Multivariate', 'Sequential', 'Time-Series'], 'num_instances': 541909, 'num_features': 6, 'feature_types': ['Integer', 'Real'], 'demographics': [], 'target_col': None, 'index_col': ['InvoiceNo', 'StockCode'], 'has_missing_values': 'no', 'missing_values_symbol': None, 'year_of_dataset_creation': 2015, 'last_updated': 'Mon Oct 21 2024', 'dataset_doi': '10.24432/C5BW33', 'creators': ['Daqing Chen'], 'intro_paper': {'ID': 361, 'type': 'NATIVE', 'title': 'Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining', 'authors': 'Daqing Chen, Sai Laing Sain, Kun Guo', 'venue': 'Journal of Database Marketing and Customer Strategy Management, Vol. 19, No. 3', 'year': 2012, 'journal': None, 'DOI': '10.1057/dbm.2012.17', 'URL': 'https://www.semanticscholar.org/paper/e43a5a90fa33d419df42e485099f8f08badf2149', 'sha': None, 'corpus': None, 'arxiv': None, 'mag': None, 'acl': None, 'pmid': None, 'pmcid': None}, 'additional_info': {'summary': 'This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.', 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': "InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. \nStockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.\nDescription: Product (item) name. Nominal.\nQuantity: The quantities of each product (item) per transaction. Numeric.\t\nInvoiceDate: Invoice Date and time. Numeric, the day and time when each transaction was generated.\nUnitPrice: Unit price. Numeric, Product price per unit in sterling.\nCustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.\nCountry: Country name. Nominal, the name of the country where each customer resides. ", 'citation': None}}
              name     role         type demographic  \
    0    InvoiceNo       ID  Categorical        None   
    1    StockCode       ID  Categorical        None   
    2  Description  Feature  Categorical        None   
    3     Quantity  Feature      Integer        None   
    4  InvoiceDate  Feature         Date        None   
    5    UnitPrice  Feature   Continuous        None   
    6   CustomerID  Feature  Categorical        None   
    7      Country  Feature  Categorical        None   
    
                                             description     units missing_values  
    0  a 6-digit integral number uniquely assigned to...      None             no  
    1  a 5-digit integral number uniquely assigned to...      None             no  
    2                                       product name      None             no  
    3  the quantities of each product (item) per tran...      None             no  
    4  the day and time when each transaction was gen...      None             no  
    5                             product price per unit  sterling             no  
    6  a 5-digit integral number uniquely assigned to...      None             no  
    7  the name of the country where each customer re...      None             no  

```python
# convert the fetch dataset into dataframe format to make it easy to analyze
data_url = online_retail.metadata['data_url']
df = pd.read_csv(data_url)

# display the first few rows of the DataFrame
print(df.head())
```

      InvoiceNo StockCode                          Description  Quantity  \
    0    536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6   
    1    536365     71053                  WHITE METAL LANTERN         6   
    2    536365    84406B       CREAM CUPID HEARTS COAT HANGER         8   
    3    536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6   
    4    536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6   
    
          InvoiceDate  UnitPrice  CustomerID         Country  
    0  12/1/2010 8:26       2.55     17850.0  United Kingdom  
    1  12/1/2010 8:26       3.39     17850.0  United Kingdom  
    2  12/1/2010 8:26       2.75     17850.0  United Kingdom  
    3  12/1/2010 8:26       3.39     17850.0  United Kingdom  
    4  12/1/2010 8:26       3.39     17850.0  United Kingdom  

```python
# check the data type and other information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype  
    ---  ------       --------------   -----  
     0   InvoiceNo    541909 non-null  object 
     1   StockCode    541909 non-null  object 
     2   Description  540455 non-null  object 
     3   Quantity     541909 non-null  int64  
     4   InvoiceDate  541909 non-null  object 
     5   UnitPrice    541909 non-null  float64
     6   CustomerID   406829 non-null  float64
     7   Country      541909 non-null  object 
    dtypes: float64(2), int64(1), object(5)
    memory usage: 33.1+ MB

### Handling Data Quality

```python
# check null values
df.isnull().sum()
```

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>InvoiceNo</th>
      <td>0</td>
    </tr>
    <tr>
      <th>StockCode</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Description</th>
      <td>1454</td>
    </tr>
    <tr>
      <th>Quantity</th>
      <td>0</td>
    </tr>
    <tr>
      <th>InvoiceDate</th>
      <td>0</td>
    </tr>
    <tr>
      <th>UnitPrice</th>
      <td>0</td>
    </tr>
    <tr>
      <th>CustomerID</th>
      <td>135080</td>
    </tr>
    <tr>
      <th>Country</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>


```python
# drop null rows where CustomerID or Description have null value
df = df.dropna(subset=['CustomerID', 'Description'])

# drop duplicated rows
df = df.drop_duplicates()

# separate canceled orders
df['IsCanceled'] = df['InvoiceNo'].str.contains('C', na=False)

# convert data type
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype('float')

# for further analysis only non-canceled product will be included
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
```
### Add Features

```python
# add time based features
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
df['Hour'] = df['InvoiceDate'].dt.hour

# add revenue feature
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# add column for canceled orders
df['IsCanceled'] = df['InvoiceNo'].str.contains('C', na=False)
```
```python
# statistic summary
df[['Quantity', 'UnitPrice', 'Revenue']].describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Quantity</th>
      <th>UnitPrice</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>392692.000000</td>
      <td>392692.000000</td>
      <td>392692.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>13.119702</td>
      <td>3.125914</td>
      <td>22.631500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>180.492832</td>
      <td>22.241836</td>
      <td>311.099224</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.001000</td>
      <td>0.001000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.250000</td>
      <td>4.950000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>1.950000</td>
      <td>12.450000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12.000000</td>
      <td>3.750000</td>
      <td>19.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80995.000000</td>
      <td>8142.750000</td>
      <td>168469.600000</td>
    </tr>
  </tbody>
</table>

## Analysis

### Customer Analysis RFM

```python
def analyze_customers(df):
    # RFM Analysis
    today = df['InvoiceDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (today - x.max()).days,
        'InvoiceNo': 'count',
        'Revenue': 'sum'
        }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Revenue': 'Monetary'
    })

    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=['5', '4', '3', '2', '1'])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=['1', '2', '3', '4', '5'])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=['1', '2', '3', '4', '5'])

    rfm['RFM_Segment_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    segments = {
    '555': 'Champions', '554': 'Champions', '544': 'Champions', '545': 'Champions',
    '454': 'Champions', '455': 'Champions', '445': 'Champions',

    '543': 'Loyal', '444': 'Loyal', '443': 'Loyal', '355': 'Loyal',
    '354': 'Loyal', '345': 'Loyal', '344': 'Loyal', '335': 'Loyal',

    '553': 'Potential_Loyalist', '551': 'Potential_Loyalist', '552': 'Potential_Loyalist',
    '541': 'Potential_Loyalist', '542': 'Potential_Loyalist', '533': 'Potential_Loyalist',
    '532': 'Potential_Loyalist', '531': 'Potential_Loyalist', '452': 'Potential_Loyalist',
    '451': 'Potential_Loyalist', '442': 'Potential_Loyalist', '441': 'Potential_Loyalist',
    '453': 'Potential_Loyalist', '433': 'Potential_Loyalist', '432': 'Potential_Loyalist',
    '423': 'Potential_Loyalist', '353': 'Potential_Loyalist', '352': 'Potential_Loyalist',
    '351': 'Potential_Loyalist', '342': 'Potential_Loyalist', '341': 'Potential_Loyalist',
    '333': 'Potential_Loyalist', '323': 'Potential_Loyalist',

    '512': 'Recent_Customers', '511': 'Recent_Customers', '422': 'Recent_Customers',
    '421': 'Recent_Customers', '412': 'Recent_Customers', '411': 'Recent_Customers',
    '311': 'Recent_Customers',

    '525': 'Promising', '524': 'Promising', '523': 'Promising', '522': 'Promising',
    '521': 'Promising', '515': 'Promising', '514': 'Promising', '513': 'Promising',
    '425': 'Promising', '424': 'Promising', '413': 'Promising', '414': 'Promising',
    '415': 'Promising', '315': 'Promising', '314': 'Promising', '313': 'Promising',

    '535': 'Need_Attention', '534': 'Need_Attention', '443': 'Need_Attention',
    '434': 'Need_Attention', '343': 'Need_Attention', '334': 'Need_Attention',
    '325': 'Need_Attention', '324': 'Need_Attention',

    '331': 'About_to_Sleep', '321': 'About_to_Sleep', '312': 'About_to_Sleep',
    '221': 'About_to_Sleep', '213': 'About_to_Sleep', '231': 'About_to_Sleep',
    '241': 'About_to_Sleep', '251': 'About_to_Sleep',

    '255': 'At_Risk', '254': 'At_Risk', '245': 'At_Risk', '244': 'At_Risk',
    '253': 'At_Risk', '252': 'At_Risk', '243': 'At_Risk', '242': 'At_Risk',
    '235': 'At_Risk', '234': 'At_Risk', '225': 'At_Risk', '224': 'At_Risk',
    '133': 'At_Risk', '152': 'At_Risk', '154': 'At_Risk', '143': 'At_Risk',
    '142': 'At_Risk', '135': 'At_Risk', '134': 'At_Risk', '125': 'At_Risk', '124': 'At_Risk',

    '155': 'Cannot_Lose', '154': 'Cannot_Lose', '144': 'Cannot_Lose',
    '214': 'Cannot_Lose', '215': 'Cannot_Lose', '115': 'Cannot_Lose',
    '114': 'Cannot_Lose', '113': 'Cannot_Lose',

    '332': 'Hibernating', '322': 'Hibernating', '231': 'Hibernating',
    '241': 'Hibernating', '253': 'Hibernating', '233': 'Hibernating',
    '232': 'Hibernating', '223': 'Hibernating', '222': 'Hibernating',
    '132': 'Hibernating', '123': 'Hibernating', '122': 'Hibernating',
    '212': 'Hibernating', '211': 'Hibernating',

    '111': 'Lost', '112': 'Lost', '121': 'Lost', '131': 'Lost', '141': 'Lost', '151': 'Lost',
}

    rfm['Segment'] = rfm['RFM_Segment_Score'].map(segments)
    return rfm
```

```python
rfm = analyze_customers(df)
segment_distribution = rfm['Segment'].value_counts().reset_index()
segment_distribution.columns = ['Segment', 'Number of Customer']

segment_order = [
    "Champions", "Loyal", "Potential_Loyalist", "Recent_Customers",
    "Promising", "Need_Attention", "About_to_Sleep", "At_Risk",
    "Cannot_Lose", "Hibernating", "Lost"
]

colors = sns.color_palette("RdYlGn", n_colors=len(segment_order))[::-1]
color_map = {segment: colors[i] for i, segment in enumerate(segment_order)}

plt.figure(figsize=(10, 6))
sns.barplot(
    data=segment_distribution,
    x="Number of Customer",
    y="Segment", hue='Segment',
    order=segment_order,
    palette=color_map, legend=False
)

plt.title("RFM Segment Distribution")
plt.xlabel("Number of Customer")
plt.ylabel("Segment")
for i, bar in enumerate(plt.gca().patches):
    value = bar.get_width()
    plt.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.0f}",
        va='center',
        ha='left',
        fontsize=10
    )
plt.tight_layout()
plt.show()
```
<img src="https://raw.githubusercontent.com/auliaaaz/auliaaaz.github.io/main/docs/eda/images/2024-02-19-blog-post/2024-02-19-blog-post_19_0.png" alt="EDA Image" width="500">





<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/2024-02-19-blog-post/rfm.html" 
        width="100%" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>


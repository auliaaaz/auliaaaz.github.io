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

| Column       | Missing Values |
|-------------|---------------|
| InvoiceNo   | 0             |
| StockCode   | 0             |
| Description | 1454          |
| Quantity    | 0             |
| InvoiceDate | 0             |
| UnitPrice   | 0             |
| CustomerID  | 135080        |
| Country     | 0             |


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
<div class="code-example" markdown="1">

|       | Quantity       | UnitPrice     | Revenue       |
|-------|--------------:|-------------:|-------------:|
| count | 392692.000000 | 392692.000000 | 392692.000000 |
| mean  | 13.119702     | 3.125914      | 22.631500     |
| std   | 180.492832    | 22.241836     | 311.099224    |
| min   | 1.000000      | 0.001000      | 0.001000      |
| 25%   | 2.000000      | 1.250000      | 4.950000      |
| 50%   | 6.000000      | 1.950000      | 12.450000     |
| 75%   | 12.000000     | 3.750000      | 19.800000     |
| max   | 80995.000000  | 8142.750000   | 168469.600000 |

</div>

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
![EDA Image](../../../docs/eda/images/2024-02-19-blog-post/2024-02-19-blog-post_18_0.png)

```python
avg_r = (rfm.groupby('Segment').agg({'Recency': 'mean'}).round(0)).reset_index()
avg_r.columns = ["Segment", "Average Recency (Days)"]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=avg_r,
    x="Average Recency (Days)",
    y="Segment",
    hue = "Segment",
    order=segment_order,
    palette=color_map, legend=False
)

plt.title("Average Recency Customer Segmentation")
plt.xlabel("Average Recency (Days)")
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
![EDA Image2](../../../docs/eda/images/2024-02-19-blog-post/2024-02-19-blog-post_19_0.png)

```python
df_r = rfm.groupby("R_Score", observed=False).agg({'Recency':'mean'}).sort_values(by="R_Score").reset_index().rename(columns={'Recency': 'Avg_R_Value'})
df_f = rfm.groupby("F_Score", observed=False).agg({'Frequency':'mean'}).sort_values(by="F_Score").reset_index().rename(columns={'Frequency': 'Avg_F_Value'})
df_m = rfm.groupby("M_Score", observed=False).agg({'Monetary':'mean'}).sort_values(by="M_Score",).reset_index().rename(columns={'Monetary': 'Avg_M_Value'})

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        "Avg. R Value by R Score",
        "Avg. F Value by F Score",
        "Avg. M Value by M Score"])

fig.add_trace(
    go.Bar(x=df_r['R_Score'], y=df_r['Avg_R_Value'], marker_color='olive', name="Avg_R_Value"),
    row=1, col=1)

fig.add_trace(
    go.Bar(x=df_f['F_Score'], y=df_f['Avg_F_Value'], marker_color='teal', name="Avg. F Value"),
    row=1, col=2)

fig.add_trace(
    go.Bar(x=df_m['M_Score'], y=df_m['Avg_M_Value'], marker_color='purple', name="Avg. M Value"),
    row=1, col=3)

fig.update_layout(
    height=400, width=1200,
    title_text="RFM Metrics",
    showlegend=False)
fig.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/2024-02-19-blog-post/rfm.html" 
        width="100%" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>

```python
total_revenue = rfm.reset_index().groupby("Segment", observed=False).agg({"Monetary":"sum", "Recency":"mean", "CustomerID":"count"}).reset_index()
fig = px.scatter(
    total_revenue,
    x="Recency",
    y="Monetary",
    size="CustomerID",
    color="Segment",
    hover_name="Segment",
    title="Recency and Monetary of each Segment",
    labels={"Recency": "AVG Days Since Last Transaction",
            "Monetary": "Total Revenue"})
fig.update_layout(
    xaxis_title="AVG Days Since Last Transaction",
    yaxis_title="Total Revenue",
    legend_title="Segment",
    title_font_size=16)
fig.show()
```
<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/2024-02-19-blog-post/RM%20per%20Segment.html" 
        width="100%" 
        style="border: none; overflow: hidden;"></iframe>
</div>

#### Insights and Recommendation

```python
from IPython.display import display, HTML
segments_df = pd.DataFrame({
    'Segment': [
        'Champions',
        'Loyal Customers',
        'Potential Loyalists',
        'Recent Customers',
        'Promising Customers',
        'Needing Attention',
        'About to Sleep',
        'At Risk',
        "Can't Lose Them",
        'Hibernating',
        'Lost'],
    'Description': [
        'High-value frequent buyers',
        'Regular consistent buyers',
        'Promising new customers',
        'New accounts',
        'Small initial purchases',
        'Declining activity',
        'Reducing purchase frequency',
        'Previously high-value, now declining',
        'Former top accounts',
        'Long-inactive accounts',
        'No recent activity'],
    'Recommended Strategy': [
        'VIP wholesale program or loyalty program for online shopping',
        'Wholesale bundles like Christmas Gift/Thanks-giving season, volume-based benefits, premium services',
        'Trade credit options, graduated discounts, loyalty program promotions',
        'Welcome pack, sample products',
        'Starter packs to order, merchandising tips, easy reorder process',
        'Comeback discounts, reorder reminders',
        'Customer survey about our product',
        'Account review meetings, flexible payments, custom assortments',
        'Special pricing',
        'New product updates, restart packages like new-member, re-engagement',
        'Annual reactivation campaigns to their email, keep in promotional database'
    ]
})

styled_df = segments_df.style.hide(axis='index').set_properties(**{
    'padding': '10px',
    'text-align': 'left'
})

display(styled_df)
```
<style>
table {
    width: 100%;
    border-collapse: collapse;
    background-color: #222;
    color: white;
}
th, td {
    border: 1px solid #555;
    padding: 10px;
    text-align: left;
}
th {
    background-color: #333;
    font-weight: bold;
}
</style>

<table>
    <tr>
        <th>Segment</th>
        <th>Description</th>
        <th>Recommended Strategy</th>
    </tr>
    <tr>
        <td>Champions</td>
        <td>High-value frequent buyers</td>
        <td>VIP wholesale program or loyalty program for online shopping</td>
    </tr>
    <tr>
        <td>Loyal Customers</td>
        <td>Regular consistent buyers</td>
        <td>Wholesale bundles like Christmas Gift/Thanksgiving season, volume-based benefits, premium services</td>
    </tr>
    <tr>
        <td>Potential Loyalists</td>
        <td>Promising new customers</td>
        <td>Trade credit options, graduated discounts, loyalty program promotions</td>
    </tr>
    <tr>
        <td>Recent Customers</td>
        <td>New accounts</td>
        <td>Welcome pack, sample products</td>
    </tr>
    <tr>
        <td>Promising Customers</td>
        <td>Small initial purchases</td>
        <td>Starter packs to order, merchandising tips, easy reorder process</td>
    </tr>
    <tr>
        <td>Needing Attention</td>
        <td>Declining activity</td>
        <td>Comeback discounts, reorder reminders</td>
    </tr>
    <tr>
        <td>About to Sleep</td>
        <td>Reducing purchase frequency</td>
        <td>Customer survey about our product</td>
    </tr>
    <tr>
        <td>At Risk</td>
        <td>Previously high-value, now declining</td>
        <td>Account review meetings, flexible payments, custom assortments</td>
    </tr>
    <tr>
        <td>Can't Lose Them</td>
        <td>Former top accounts</td>
        <td>Special pricing</td>
    </tr>
    <tr>
        <td>Hibernating</td>
        <td>Long-inactive accounts</td>
        <td>New product updates, restart packages like new-member, re-engagement</td>
    </tr>
    <tr>
        <td>Lost</td>
        <td>No recent activity</td>
        <td>Annual reactivation campaigns to their email, keep in promotional database</td>
    </tr>
</table>


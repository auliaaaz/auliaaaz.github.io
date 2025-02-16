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
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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

    <ipython-input-59-73cfaded9c91>:2: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    <ipython-input-59-73cfaded9c91>:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    



```python
# statistic summary
df[['Quantity', 'UnitPrice', 'Revenue']].describe()
```





  <div id="df-6584d0bb-0a7e-4154-b998-0fb83691c830" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6584d0bb-0a7e-4154-b998-0fb83691c830')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6584d0bb-0a7e-4154-b998-0fb83691c830 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6584d0bb-0a7e-4154-b998-0fb83691c830');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9c608f5e-ccb2-41a9-846f-c06a1ae53d52">
  <button class="colab-df-quickchart" onclick="quickchart('df-9c608f5e-ccb2-41a9-846f-c06a1ae53d52')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9c608f5e-ccb2-41a9-846f-c06a1ae53d52 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
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


    
![png](EDA_Retail_files/EDA_Retail_18_0.png)
    



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


    
![png](EDA_Retail_files/EDA_Retail_19_0.png)
    



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


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="03643294-4b41-496f-878e-b2f96cfdb06a" class="plotly-graph-div" style="height:400px; width:1200px;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("03643294-4b41-496f-878e-b2f96cfdb06a")) {                    Plotly.newPlot(                        "03643294-4b41-496f-878e-b2f96cfdb06a",                        [{"marker":{"color":"olive"},"name":"Avg_R_Value","x":["5","4","3","2","1"],"y":[5.168202764976958,22.055309734513273,51.315850815850816,115.2491103202847,267.60346820809247],"type":"bar","xaxis":"x","yaxis":"y"},{"marker":{"color":"teal"},"name":"Avg. F Value","x":["1","2","3","4","5"],"y":[7.676375404530744,21.805256869773,41.968122786304605,84.41849710982659,300.1809744779582],"type":"bar","xaxis":"x2","yaxis":"y2"},{"marker":{"color":"purple"},"name":"Avg. M Value","x":["1","2","3","4","5"],"y":[152.57853686635946,357.55860553633215,684.3383433179724,1399.6088927335638,7646.65993202765],"type":"bar","xaxis":"x3","yaxis":"y3"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.2888888888888889]},"yaxis":{"anchor":"x","domain":[0.0,1.0]},"xaxis2":{"anchor":"y2","domain":[0.35555555555555557,0.6444444444444445]},"yaxis2":{"anchor":"x2","domain":[0.0,1.0]},"xaxis3":{"anchor":"y3","domain":[0.7111111111111111,1.0]},"yaxis3":{"anchor":"x3","domain":[0.0,1.0]},"annotations":[{"font":{"size":16},"showarrow":false,"text":"Avg. R Value by R Score","x":0.14444444444444446,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Avg. F Value by F Score","x":0.5,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{"size":16},"showarrow":false,"text":"Avg. M Value by M Score","x":0.8555555555555556,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"title":{"text":"RFM Metrics"},"height":400,"width":1200,"showlegend":false},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('03643294-4b41-496f-878e-b2f96cfdb06a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



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


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="12271eff-3cca-4fc9-8e39-ac25b63e0965" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("12271eff-3cca-4fc9-8e39-ac25b63e0965")) {                    Plotly.newPlot(                        "12271eff-3cca-4fc9-8e39-ac25b63e0965",                        [{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=About_to_Sleep\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["About_to_Sleep"],"legendgroup":"About_to_Sleep","marker":{"color":"#636efa","size":[168],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"About_to_Sleep","orientation":"v","showlegend":true,"x":[80.41666666666667],"xaxis":"x","y":[50108.03],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=At_Risk\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["At_Risk"],"legendgroup":"At_Risk","marker":{"color":"#EF553B","size":[395],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"At_Risk","orientation":"v","showlegend":true,"x":[151.8860759493671],"xaxis":"x","y":[601538.65],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Cannot_Lose\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Cannot_Lose"],"legendgroup":"Cannot_Lose","marker":{"color":"#00cc96","size":[93],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Cannot_Lose","orientation":"v","showlegend":true,"x":[220.50537634408602],"xaxis":"x","y":[300785.331],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Champions\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Champions"],"legendgroup":"Champions","marker":{"color":"#ab63fa","size":[804],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Champions","orientation":"v","showlegend":true,"x":[10.531094527363184],"xaxis":"x","y":[5481175.85],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Hibernating\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Hibernating"],"legendgroup":"Hibernating","marker":{"color":"#FFA15A","size":[824],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Hibernating","orientation":"v","showlegend":true,"x":[149.5254854368932],"xaxis":"x","y":[328877.542],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Lost\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Lost"],"legendgroup":"Lost","marker":{"color":"#19d3f3","size":[434],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Lost","orientation":"v","showlegend":true,"x":[276.9700460829493],"xaxis":"x","y":[77328.65],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Loyal\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Loyal"],"legendgroup":"Loyal","marker":{"color":"#FF6692","size":[404],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Loyal","orientation":"v","showlegend":true,"x":[38.227722772277225],"xaxis":"x","y":[953861.77],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Need_Attention\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Need_Attention"],"legendgroup":"Need_Attention","marker":{"color":"#B6E880","size":[231],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Need_Attention","orientation":"v","showlegend":true,"x":[31.83116883116883],"xaxis":"x","y":[341065.78],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Potential_Loyalist\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Potential_Loyalist"],"legendgroup":"Potential_Loyalist","marker":{"color":"#FF97FF","size":[500],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Potential_Loyalist","orientation":"v","showlegend":true,"x":[28.272],"xaxis":"x","y":[297563.721],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Promising\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Promising"],"legendgroup":"Promising","marker":{"color":"#FECB52","size":[138],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Promising","orientation":"v","showlegend":true,"x":[17.891304347826086],"xaxis":"x","y":[327793.49],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eSegment=Recent_Customers\u003cbr\u003eAVG Days Since Last Transaction=%{x}\u003cbr\u003eTotal Revenue=%{y}\u003cbr\u003eCustomerID=%{marker.size}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Recent_Customers"],"legendgroup":"Recent_Customers","marker":{"color":"#636efa","size":[313],"sizemode":"area","sizeref":2.06,"symbol":"circle"},"mode":"markers","name":"Recent_Customers","orientation":"v","showlegend":true,"x":[28.45367412140575],"xaxis":"x","y":[65738.93],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"AVG Days Since Last Transaction"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Total Revenue"}},"legend":{"title":{"text":"Segment"},"tracegroupgap":0,"itemsizing":"constant"},"title":{"text":"Recency and Monetary of each Segment","font":{"size":16}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('12271eff-3cca-4fc9-8e39-ac25b63e0965');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


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


<style type="text/css">
#T_5fdec_row0_col0, #T_5fdec_row0_col1, #T_5fdec_row0_col2, #T_5fdec_row1_col0, #T_5fdec_row1_col1, #T_5fdec_row1_col2, #T_5fdec_row2_col0, #T_5fdec_row2_col1, #T_5fdec_row2_col2, #T_5fdec_row3_col0, #T_5fdec_row3_col1, #T_5fdec_row3_col2, #T_5fdec_row4_col0, #T_5fdec_row4_col1, #T_5fdec_row4_col2, #T_5fdec_row5_col0, #T_5fdec_row5_col1, #T_5fdec_row5_col2, #T_5fdec_row6_col0, #T_5fdec_row6_col1, #T_5fdec_row6_col2, #T_5fdec_row7_col0, #T_5fdec_row7_col1, #T_5fdec_row7_col2, #T_5fdec_row8_col0, #T_5fdec_row8_col1, #T_5fdec_row8_col2, #T_5fdec_row9_col0, #T_5fdec_row9_col1, #T_5fdec_row9_col2, #T_5fdec_row10_col0, #T_5fdec_row10_col1, #T_5fdec_row10_col2 {
  padding: 10px;
  text-align: left;
}
</style>
<table id="T_5fdec" class="dataframe">
  <thead>
    <tr>
      <th id="T_5fdec_level0_col0" class="col_heading level0 col0" >Segment</th>
      <th id="T_5fdec_level0_col1" class="col_heading level0 col1" >Description</th>
      <th id="T_5fdec_level0_col2" class="col_heading level0 col2" >Recommended Strategy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_5fdec_row0_col0" class="data row0 col0" >Champions</td>
      <td id="T_5fdec_row0_col1" class="data row0 col1" >High-value frequent buyers</td>
      <td id="T_5fdec_row0_col2" class="data row0 col2" >VIP wholesale program or loyalty program for online shopping</td>
    </tr>
    <tr>
      <td id="T_5fdec_row1_col0" class="data row1 col0" >Loyal Customers</td>
      <td id="T_5fdec_row1_col1" class="data row1 col1" >Regular consistent buyers</td>
      <td id="T_5fdec_row1_col2" class="data row1 col2" >Wholesale bundles like Christmas Gift/Thanks-giving season, volume-based benefits, premium services</td>
    </tr>
    <tr>
      <td id="T_5fdec_row2_col0" class="data row2 col0" >Potential Loyalists</td>
      <td id="T_5fdec_row2_col1" class="data row2 col1" >Promising new customers</td>
      <td id="T_5fdec_row2_col2" class="data row2 col2" >Trade credit options, graduated discounts, loyalty program promotions</td>
    </tr>
    <tr>
      <td id="T_5fdec_row3_col0" class="data row3 col0" >Recent Customers</td>
      <td id="T_5fdec_row3_col1" class="data row3 col1" >New accounts</td>
      <td id="T_5fdec_row3_col2" class="data row3 col2" >Welcome pack, sample products</td>
    </tr>
    <tr>
      <td id="T_5fdec_row4_col0" class="data row4 col0" >Promising Customers</td>
      <td id="T_5fdec_row4_col1" class="data row4 col1" >Small initial purchases</td>
      <td id="T_5fdec_row4_col2" class="data row4 col2" >Starter packs to order, merchandising tips, easy reorder process</td>
    </tr>
    <tr>
      <td id="T_5fdec_row5_col0" class="data row5 col0" >Needing Attention</td>
      <td id="T_5fdec_row5_col1" class="data row5 col1" >Declining activity</td>
      <td id="T_5fdec_row5_col2" class="data row5 col2" >Comeback discounts, reorder reminders</td>
    </tr>
    <tr>
      <td id="T_5fdec_row6_col0" class="data row6 col0" >About to Sleep</td>
      <td id="T_5fdec_row6_col1" class="data row6 col1" >Reducing purchase frequency</td>
      <td id="T_5fdec_row6_col2" class="data row6 col2" >Customer survey about our product</td>
    </tr>
    <tr>
      <td id="T_5fdec_row7_col0" class="data row7 col0" >At Risk</td>
      <td id="T_5fdec_row7_col1" class="data row7 col1" >Previously high-value, now declining</td>
      <td id="T_5fdec_row7_col2" class="data row7 col2" >Account review meetings, flexible payments, custom assortments</td>
    </tr>
    <tr>
      <td id="T_5fdec_row8_col0" class="data row8 col0" >Can't Lose Them</td>
      <td id="T_5fdec_row8_col1" class="data row8 col1" >Former top accounts</td>
      <td id="T_5fdec_row8_col2" class="data row8 col2" >Special pricing</td>
    </tr>
    <tr>
      <td id="T_5fdec_row9_col0" class="data row9 col0" >Hibernating</td>
      <td id="T_5fdec_row9_col1" class="data row9 col1" >Long-inactive accounts</td>
      <td id="T_5fdec_row9_col2" class="data row9 col2" >New product updates, restart packages like new-member, re-engagement</td>
    </tr>
    <tr>
      <td id="T_5fdec_row10_col0" class="data row10 col0" >Lost</td>
      <td id="T_5fdec_row10_col1" class="data row10 col1" >No recent activity</td>
      <td id="T_5fdec_row10_col2" class="data row10 col2" >Annual reactivation campaigns to their email, keep in promotional database</td>
    </tr>
  </tbody>
</table>



### Seasonality Sales Behaviour Analysis


```python
monthly_sales = df.groupby(['Year', 'Month'], observed=False).agg({"Revenue":"sum", "CustomerID":"count"}).reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
monthly_sales = monthly_sales[monthly_sales["Date"]!= "2011-12-01"]

fig = px.bar(monthly_sales,
             x='Date', y='Revenue', color='Revenue',
             color_continuous_scale = 'Teal', title="Total Revenue per Month")

fig.update(layout_coloraxis_showscale=False,)
fig.show()

fig = px.bar(monthly_sales,
             x='Date', y='CustomerID', color='CustomerID',
             color_continuous_scale = 'Teal', title="Number of Customer per Month")

fig.update(layout_coloraxis_showscale=False)
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="04f9b1c2-e3d1-488a-808e-5be4d18e23e0" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("04f9b1c2-e3d1-488a-808e-5be4d18e23e0")) {                    Plotly.newPlot(                        "04f9b1c2-e3d1-488a-808e-5be4d18e23e0",                        [{"alignmentgroup":"True","hovertemplate":"Date=%{x}\u003cbr\u003eRevenue=%{marker.color}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":[570422.73,568101.31,446084.92,594081.76,468374.331,677355.15,660046.05,598962.901,644051.04,950690.202,1035642.45,1156205.61],"coloraxis":"coloraxis","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["2010-12-01T00:00:00","2011-01-01T00:00:00","2011-02-01T00:00:00","2011-03-01T00:00:00","2011-04-01T00:00:00","2011-05-01T00:00:00","2011-06-01T00:00:00","2011-07-01T00:00:00","2011-08-01T00:00:00","2011-09-01T00:00:00","2011-10-01T00:00:00","2011-11-01T00:00:00"],"xaxis":"x","y":[570422.73,568101.31,446084.92,594081.76,468374.331,677355.15,660046.05,598962.901,644051.04,950690.202,1035642.45,1156205.61],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Revenue"}},"coloraxis":{"colorbar":{"title":{"text":"Revenue"}},"colorscale":[[0.0,"rgb(209, 238, 234)"],[0.16666666666666666,"rgb(168, 219, 217)"],[0.3333333333333333,"rgb(133, 196, 201)"],[0.5,"rgb(104, 171, 184)"],[0.6666666666666666,"rgb(79, 144, 166)"],[0.8333333333333334,"rgb(59, 115, 143)"],[1.0,"rgb(42, 86, 116)"]],"showscale":false},"legend":{"tracegroupgap":0},"title":{"text":"Total Revenue per Month"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('04f9b1c2-e3d1-488a-808e-5be4d18e23e0');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="991993f9-6777-4069-a8f3-ea4c93bfb046" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("991993f9-6777-4069-a8f3-ea4c93bfb046")) {                    Plotly.newPlot(                        "991993f9-6777-4069-a8f3-ea4c93bfb046",                        [{"alignmentgroup":"True","hovertemplate":"Date=%{x}\u003cbr\u003eCustomerID=%{marker.color}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":[25670,20988,19706,26870,22433,28073,26926,26580,26790,39669,48793,63168],"coloraxis":"coloraxis","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["2010-12-01T00:00:00","2011-01-01T00:00:00","2011-02-01T00:00:00","2011-03-01T00:00:00","2011-04-01T00:00:00","2011-05-01T00:00:00","2011-06-01T00:00:00","2011-07-01T00:00:00","2011-08-01T00:00:00","2011-09-01T00:00:00","2011-10-01T00:00:00","2011-11-01T00:00:00"],"xaxis":"x","y":[25670,20988,19706,26870,22433,28073,26926,26580,26790,39669,48793,63168],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"CustomerID"}},"coloraxis":{"colorbar":{"title":{"text":"CustomerID"}},"colorscale":[[0.0,"rgb(209, 238, 234)"],[0.16666666666666666,"rgb(168, 219, 217)"],[0.3333333333333333,"rgb(133, 196, 201)"],[0.5,"rgb(104, 171, 184)"],[0.6666666666666666,"rgb(79, 144, 166)"],[0.8333333333333334,"rgb(59, 115, 143)"],[1.0,"rgb(42, 86, 116)"]],"showscale":false},"legend":{"tracegroupgap":0},"title":{"text":"Number of Customer per Month"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('991993f9-6777-4069-a8f3-ea4c93bfb046');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


*  Monthly sales and the number of buyers showed a sharp increase between September 2011 and November 2011 which are near to the holiday season. Maintain a good application system will be good to handle the large amount of customers in this peak season.
*  While the outlook appears positive, longer-term data is needed to determine whether this trend is driven by winter seasonality or reflects an overall improvement in performance.

### Product Analysis


```python
product_performance = df.groupby('Description').agg({
        'Quantity': 'sum',
        'Revenue': 'sum',
        'CustomerID': 'nunique'
    }).rename(columns={'CustomerID': 'Unique_Customers'})
product_performance = product_performance.sort_values(by="Quantity", ascending=False).head(5).reset_index()

fig = px.bar(product_performance,
             x='Quantity', y='Description', color='Quantity',
             color_continuous_scale = 'Teal', title="Top 5 the Most Ordered Product")

fig.update(layout_coloraxis_showscale=False)
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="c4295f02-7412-479d-80df-9373aa1ae4db" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c4295f02-7412-479d-80df-9373aa1ae4db")) {                    Plotly.newPlot(                        "c4295f02-7412-479d-80df-9373aa1ae4db",                        [{"alignmentgroup":"True","hovertemplate":"Quantity=%{marker.color}\u003cbr\u003eDescription=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":[80995,77916,54319,46078,36706],"coloraxis":"coloraxis","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"h","showlegend":false,"textposition":"auto","x":[80995,77916,54319,46078,36706],"xaxis":"x","y":["PAPER CRAFT , LITTLE BIRDIE","MEDIUM CERAMIC TOP STORAGE JAR","WORLD WAR 2 GLIDERS ASSTD DESIGNS","JUMBO BAG RED RETROSPOT","WHITE HANGING HEART T-LIGHT HOLDER"],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Quantity"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Description"}},"coloraxis":{"colorbar":{"title":{"text":"Quantity"}},"colorscale":[[0.0,"rgb(209, 238, 234)"],[0.16666666666666666,"rgb(168, 219, 217)"],[0.3333333333333333,"rgb(133, 196, 201)"],[0.5,"rgb(104, 171, 184)"],[0.6666666666666666,"rgb(79, 144, 166)"],[0.8333333333333334,"rgb(59, 115, 143)"],[1.0,"rgb(42, 86, 116)"]],"showscale":false},"legend":{"tracegroupgap":0},"title":{"text":"Top 5 the Most Ordered Product"},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c4295f02-7412-479d-80df-9373aa1ae4db');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


To increase more the product selling, it can be conducted with programs like product bundling or cross-selling product to suggest related product to customer. It can be do with Apriori Algorithm


```python
def basket_analysis(df):
    basket = (
        df.groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
    )

    basket = basket > 0

    # apply Apriori Algorithm
    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=len(frequent_itemsets))

    rules = rules.sort_values('lift', ascending=False)
    return rules
```


```python
rules = basket_analysis(df)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5)
styled_df = rules.style.hide(axis='index').set_properties(**{
    'padding': '10px',
    'text-align': 'left',
})

display(styled_df)
```


<style type="text/css">
#T_81650_row0_col0, #T_81650_row0_col1, #T_81650_row0_col2, #T_81650_row0_col3, #T_81650_row0_col4, #T_81650_row1_col0, #T_81650_row1_col1, #T_81650_row1_col2, #T_81650_row1_col3, #T_81650_row1_col4, #T_81650_row2_col0, #T_81650_row2_col1, #T_81650_row2_col2, #T_81650_row2_col3, #T_81650_row2_col4, #T_81650_row3_col0, #T_81650_row3_col1, #T_81650_row3_col2, #T_81650_row3_col3, #T_81650_row3_col4, #T_81650_row4_col0, #T_81650_row4_col1, #T_81650_row4_col2, #T_81650_row4_col3, #T_81650_row4_col4 {
  padding: 10px;
  text-align: left;
}
</style>
<table id="T_81650" class="dataframe">
  <thead>
    <tr>
      <th id="T_81650_level0_col0" class="col_heading level0 col0" >antecedents</th>
      <th id="T_81650_level0_col1" class="col_heading level0 col1" >consequents</th>
      <th id="T_81650_level0_col2" class="col_heading level0 col2" >support</th>
      <th id="T_81650_level0_col3" class="col_heading level0 col3" >confidence</th>
      <th id="T_81650_level0_col4" class="col_heading level0 col4" >lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_81650_row0_col0" class="data row0 col0" >frozenset({'PINK REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row0_col1" class="data row0 col1" >frozenset({'ROSES REGENCY TEACUP AND SAUCER ', 'GREEN REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row0_col2" class="data row0 col2" >0.021045</td>
      <td id="T_81650_row0_col3" class="data row0 col3" >0.701439</td>
      <td id="T_81650_row0_col4" class="data row0 col4" >24.027846</td>
    </tr>
    <tr>
      <td id="T_81650_row1_col0" class="data row1 col0" >frozenset({'ROSES REGENCY TEACUP AND SAUCER ', 'GREEN REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row1_col1" class="data row1 col1" >frozenset({'PINK REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row1_col2" class="data row1 col2" >0.021045</td>
      <td id="T_81650_row1_col3" class="data row1 col3" >0.720887</td>
      <td id="T_81650_row1_col4" class="data row1 col4" >24.027846</td>
    </tr>
    <tr>
      <td id="T_81650_row2_col0" class="data row2 col0" >frozenset({'GREEN REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row2_col1" class="data row2 col1" >frozenset({'ROSES REGENCY TEACUP AND SAUCER ', 'PINK REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row2_col2" class="data row2 col2" >0.021045</td>
      <td id="T_81650_row2_col3" class="data row2 col3" >0.564399</td>
      <td id="T_81650_row2_col4" class="data row2 col4" >23.989564</td>
    </tr>
    <tr>
      <td id="T_81650_row3_col0" class="data row3 col0" >frozenset({'ROSES REGENCY TEACUP AND SAUCER ', 'PINK REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row3_col1" class="data row3 col1" >frozenset({'GREEN REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row3_col2" class="data row3 col2" >0.021045</td>
      <td id="T_81650_row3_col3" class="data row3 col3" >0.894495</td>
      <td id="T_81650_row3_col4" class="data row3 col4" >23.989564</td>
    </tr>
    <tr>
      <td id="T_81650_row4_col0" class="data row4 col0" >frozenset({'PINK REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row4_col1" class="data row4 col1" >frozenset({'GREEN REGENCY TEACUP AND SAUCER'})</td>
      <td id="T_81650_row4_col2" class="data row4 col2" >0.024822</td>
      <td id="T_81650_row4_col3" class="data row4 col3" >0.827338</td>
      <td id="T_81650_row4_col4" class="data row4 col4" >22.188466</td>
    </tr>
  </tbody>
</table>



Metrics:

* Antecedents: Items that trigger the rule.
* Consequents: Items that are likely to be purchased together with the antecedents.
* Support: The frequency of the itemsets in the dataset.
* Confidence: The likelihood of purchasing the consequents when the antecedents are bought.
* Lift: The strength of the association between antecedents and consequents.

From the sample of product bundling or cross-selling:
70% of customers who buy 'PINK REGENCY TEACUP AND SAUCER' also buy 'ROSES REGENCY TEACUP AND SAUCER' and 'GREEN REGENCY TEACUP AND SAUCER'.

This suggests offering discounts for bundling these products or promoting this pair in marketing emails.

It would be even better to promote the least-ordered products by upselling them through product bundling or discounts.

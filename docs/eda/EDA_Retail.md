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


<div style="width: 100%; overflow: hidden;">
    <iframe src="https://auliaaaz.github.io/docs/eda/images/2024-02-19-blog-post/rfm.html" 
        width="100%" 
        height="600px" 
        style="border: none; overflow: hidden;"></iframe>
</div>


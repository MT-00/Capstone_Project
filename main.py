# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:32:39 2021

@author: Meng Tian
"""
# from hcaa import HierarchicalClusteringAssetAllocation
# from testHCAA import TestHCAA

import os
import numpy as np
import pandas as pd
from portfoliolab.clustering import HierarchicalEqualRiskContribution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

project_path = os.path.dirname(__file__)
data_path = project_path + '/stock_prices.csv'
data = pd.read_csv(data_path, parse_dates=True, index_col="Date")

# In[1]:
# reading in our data
stock_prices = data.sort_values(by='Date')
stock_prices.resample('M').last().plot(figsize=(17,7))
plt.ylabel('Price', size=15)
plt.xlabel('Dates', size=15)
plt.title('Asset Prices Overview', size=15)
plt.show()

# In[2]:
    
herc = HierarchicalEqualRiskContribution()
herc.allocate(asset_names=stock_prices.columns, 
              asset_prices=stock_prices, 
              risk_measure="equal_weighting", 
              linkage="ward")
# plotting our optimal portfolio
herc_weights = herc.weights
y_pos = np.arange(len(herc_weights.columns))
plt.figure(figsize=(25,7))
plt.bar(list(herc_weights.columns), herc_weights.values[0])
plt.xticks(y_pos, rotation=45, size=10)
plt.xlabel('Assets', size=20)
plt.ylabel('Asset Weights', size=20)
plt.title('HERC Portfolio Weights', size=20)
plt.show()    

# In[3]

plt.figure(figsize=(17,7))
herc.plot_clusters(assets=stock_prices.columns)
plt.title('HERC Dendrogram', size=18)
plt.xticks(rotation=45)
plt.show()

# In[4]:
    
print("Optimal Number of Clusters: " + str(herc.optimal_num_clusters))

# In[5]:
    
























# Instantiating HERC Class
# herc = HierarchicalEqualRiskContribution()
# herc.allocate(asset_prices=stock_prices, risk_measure='equal_weighting')
# herc_weights = herc.weights
# # y_pos = np.arange(len(herc_weights.columns))
# # plt.figure(figsize=(25,7))
# plt.bar(list(herc_weights.columns), herc_weights.values[0])
# # plt.xticks(y_pos, rotation=45, size=10)
# # plt.xlabel('Assets', size=20)
# # plt.ylabel('Asset Weights', size=20)
# # plt.title('HERC Portfolio Weights', size=20)
# # plt.show()
# # Instantiating HERC Class with weight constraints
# # constraints = {'A': (0, 0.3), 'B': (0, 0.4), 'XLU': (None, None)}
# # herc.apply_weight_constraints(constraints=constraints)
# # herc_weights = herc.weights

# # # Increase the max number of iterations for constrained weights calculation
# # herc.apply_weight_constraints(n_iter=1000, precision=2)
# # herc_weights = herc.weights

# # Plot Dendrogram
# herc.plot_clusters(assets=stock_prices.columns)


# hcaa = HierarchicalClusteringAssetAllocation()

# test = TestHCAA()
# test.setUp()
# test.test_hcaa_equal_weight()

# hcaa = HierarchicalClusteringAssetAllocation()
# hcaa.allocate(asset_prices=data,
#               asset_names=data.columns,
#               optimal_num_clusters=5,
#               allocation_metric='equal_weighting')
# weights = hcaa.weights.values[0]
# assert (weights >= 0).all()
# assert len(weights) == data.shape[1]
# np.testing.assert_almost_equal(np.sum(weights), 1)

# constraints = {'A': (0, 0.3), 'B': (0, 0.4), 'XLU': (None, None)}
#hcaa.apply_weight_constraints(constraints=constraints)
#hcaa_weights = hcaa.weights

# hcaa.plot_clusters(assets=data.columns)
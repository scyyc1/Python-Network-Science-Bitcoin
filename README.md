# 0. Preparation
## 0.1 Libraries Use
Here I use 
1. **networkX** to perform basic graph manipulation. 
1. **numpy** for numerical calculation. 
1. **matplotlib** for data visualization.
1. **zipfile** for data import.
1. **pandas** to deliver table data.


```python
# libraries
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import zipfile as zp
import pandas as pd

# Tool libraries to assist computation (To accelerate some process or verify some values). 
# Does not influence the result of exercise
import random
import math
import itertools
```

## 0.2 Data
The data presented and used in this ipython notebook file may not be complete. Some of the data are pre-calculated as the runtime of some network algorithms is extremely long. The python kernel of notebook is fragile, so the instant variables might lose due to the kernel crash. As a result, I stored the backups of these data in a separate folder in .csv format. They are intelligibly named and the director is:
````
./data/.csv
````

## 0.3 Data Preprocessing

The **zipfile** library is used to import the graph data. The general process:
1. Obtain the name list of graphs in the zip file
1. Span the graph with built-in function of **networkX** and store them in a list

<br>I also sort the data of graph in ascending order of time</br>


```python
# list of graphs
graphs = []
# list of graph's name (the file name)
names = []

prefix = 'bitcoin_data_coursework1/'
suffix = '.graphml'
with zp.ZipFile('bitcoin_data_coursework1.zip') as arc:
        all_name_list=arc.namelist()
        data_list_temp=list(filter(lambda f: str(f).startswith(prefix), all_name_list))
        name_list=list(filter(lambda f: str(f).endswith(suffix), data_list_temp))
        
        # Sort the list according to the alphabet order
        name_list.sort()
        # read the graphs
        for data_name in name_list:
            with arc.open(data_name) as file:
                graphs.append(nx.read_graphml(file))
                data_name=data_name.lstrip(prefix)
                data_name=data_name.rstrip(suffix)
                names.append(data_name)
                print("Graph loaded: "+ data_name) 
```

    Graph loaded: 2013-09-09_to_2013-09-15
    Graph loaded: 2013-09-16_to_2013-09-22
    Graph loaded: 2013-09-23_to_2013-09-29
    Graph loaded: 2013-09-30_to_2013-10-06
    Graph loaded: 2013-10-07_to_2013-10-13
    Graph loaded: 2013-10-14_to_2013-10-20
    Graph loaded: 2013-10-21_to_2013-10-27
    Graph loaded: 2013-10-28_to_2013-11-03
    Graph loaded: 2013-11-04_to_2013-11-10
    Graph loaded: 2013-11-11_to_2013-11-17
    Graph loaded: 2013-11-18_to_2013-11-24
    Graph loaded: 2013-11-25_to_2013-12-01
    Graph loaded: 2013-12-02_to_2013-12-08
    


```python
# redundance codes to check the data integrity
for index in range(0,len(graphs)):
    print(names[index] + " is weakly connected? ", nx.is_weakly_connected(graphs[index]))
```

    2013-09-09_to_2013-09-15 is weakly connected?  False
    2013-09-16_to_2013-09-22 is weakly connected?  False
    2013-09-23_to_2013-09-29 is weakly connected?  False
    2013-09-30_to_2013-10-06 is weakly connected?  False
    2013-10-07_to_2013-10-13 is weakly connected?  False
    2013-10-14_to_2013-10-20 is weakly connected?  False
    2013-10-21_to_2013-10-27 is weakly connected?  False
    2013-10-28_to_2013-11-03 is weakly connected?  False
    2013-11-04_to_2013-11-10 is weakly connected?  False
    2013-11-11_to_2013-11-17 is weakly connected?  False
    2013-11-18_to_2013-11-24 is weakly connected?  False
    2013-11-25_to_2013-12-01 is weakly connected?  False
    2013-12-02_to_2013-12-08 is weakly connected?  False
    

# PART I 
## Exercise 1.1

First to obtain the Giant Strongly Connected Component (GSCC) of this graphs: 


```python
GSCC = []
def getGSCC(graphs):
    for graph in graphs:
        components = nx.strongly_connected_components(graph) 
        G_strong = nx.subgraph(graph, max(components, key = len)) 
        GSCC.append(G_strong)

getGSCC(graphs)
```


```python
# redundance codes to verify the correctness of data manipulation
for index in range(0,len(GSCC)):
    print(names[index] + " is strongly connected? ", nx.is_strongly_connected(GSCC[index]))
```

    2013-09-09_to_2013-09-15 is strongly connected?  True
    2013-09-16_to_2013-09-22 is strongly connected?  True
    2013-09-23_to_2013-09-29 is strongly connected?  True
    2013-09-30_to_2013-10-06 is strongly connected?  True
    2013-10-07_to_2013-10-13 is strongly connected?  True
    2013-10-14_to_2013-10-20 is strongly connected?  True
    2013-10-21_to_2013-10-27 is strongly connected?  True
    2013-10-28_to_2013-11-03 is strongly connected?  True
    2013-11-04_to_2013-11-10 is strongly connected?  True
    2013-11-11_to_2013-11-17 is strongly connected?  True
    2013-11-18_to_2013-11-24 is strongly connected?  True
    2013-11-25_to_2013-12-01 is strongly connected?  True
    2013-12-02_to_2013-12-08 is strongly connected?  True
    

To ease the difficulty of calculation I separate the easy-to-calculate data and the time consuming ones:


```python
def getBasicStats(G):
    N = nx.number_of_nodes(G)
    #     print(N)
    
    nodeNum = G.number_of_nodes()
    linkNum = G.number_of_edges()
    Density = nx.density(G)
    
    list_inDegree = np.array(G.in_degree())[:,1].astype(float)
    list_outDegree = np.array(G.out_degree())[:,1].astype(float)
    list_totalDegree = np.array(G.degree())[:,1].astype(float)
    
    avg_inDegree = list_inDegree.sum() / N
    avg_outDegree = list_outDegree.sum() / N
    avg_degree = list_totalDegree.sum() / N
    
    max_inDegree = np.amax(list_inDegree)
    max_outDegree = np.amax(list_outDegree)
    max_degree = np.amax(list_totalDegree)
    
    avg_inStrength = list_inStrength = np.array(G.in_degree(weight="qty")).astype(float).sum() / N
    avg_outStrength = list_outStrength = np.array(G.out_degree(weight="qty")).astype(float).sum() / N 
    avg_strength = list_totalStrength = np.array(G.degree(weight="qty"))[:,1].astype(float).sum() / N

    return [nodeNum, linkNum, Density,
           avg_inDegree, avg_outDegree, avg_degree,
           max_inDegree, max_outDegree, max_degree,
           avg_inStrength, avg_outStrength, avg_strength]

# function to compute the time-wasted items
def getComplexStats(G):
    N = nx.number_of_nodes(G)
    shortestPathList=nx.shortest_path_length(G)
    shortestPathArray=np.array(shortestPathList)[:,1]
    
    avg_clusteringCoefficient = nx.average_clustering(G)
    # The diameter is the longest shortest path length of all nodes of a graph
    # To decrease the repeated calculations, I use nx.shortest_path_length function
    avg_pathLength = shortestPathArray.sum()/N
    diameter = np.amax(shortestPathArray)
    
    return [avg_pathLength, diameter, avg_clusteringCoefficient]
```


```python
data = []
data_basic = []
for G in GSCC:
    data_basic.append(getBasicStats(G))
```


```python
data_complex = []
for G in GSCC:
    data_complex.append(getComplexStats(G))
```


```python
data=list(np.concatenate([np.array(data_basic), np.array(data_complex)], axis=1))
```


```python
col_labels=['Number of nodes', 'Number of links', 'Density'
            ,'Average in-degree', 'Average out-degree','Average total degree'
            ,'Maximum in-degree', 'Maximum out-degree', 'Maximum total degree'
            ,'Average in-strength', 'Average out-strength', 'Average total strength'
            ,' Average path length', 'Diameter','Average clustering coefficient'
           ]

df = pd.DataFrame(
    data,
    index=names,
    columns=col_labels
)
pd.set_option('display.width', 1000)
pd.set_option("display.max_columns", None)

df
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
      <th>Number of nodes</th>
      <th>Number of links</th>
      <th>Density</th>
      <th>Average in-degree</th>
      <th>Average out-degree</th>
      <th>Average total degree</th>
      <th>Maximum in-degree</th>
      <th>Maximum out-degree</th>
      <th>Maximum total degree</th>
      <th>Average in-strength</th>
      <th>Average out-strength</th>
      <th>Average total strength</th>
      <th>Average shortest path length</th>
      <th>Diameter</th>
      <th>Average clustering coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-09-09_to_2013-09-15</th>
      <td>59574</td>
      <td>148959</td>
      <td>0.000042</td>
      <td>2.500403</td>
      <td>2.500403</td>
      <td>5.000806</td>
      <td>29021</td>
      <td>15264</td>
      <td>44285</td>
      <td>23.995667</td>
      <td>23.995667</td>
      <td>47.991334</td>
      <td>7.570300706</td>
      <td>291</td>
      <td>0.114752</td>
    </tr>
    <tr>
      <th>2013-09-16_to_2013-09-22</th>
      <td>60774</td>
      <td>153809</td>
      <td>0.000042</td>
      <td>2.530836</td>
      <td>2.530836</td>
      <td>5.061671</td>
      <td>29778</td>
      <td>15198</td>
      <td>44976</td>
      <td>26.439220</td>
      <td>26.439220</td>
      <td>52.878441</td>
      <td>7.48833107</td>
      <td>230</td>
      <td>0.116856</td>
    </tr>
    <tr>
      <th>2013-09-23_to_2013-09-29</th>
      <td>60619</td>
      <td>153914</td>
      <td>0.000042</td>
      <td>2.539039</td>
      <td>2.539039</td>
      <td>5.078078</td>
      <td>31086</td>
      <td>15965</td>
      <td>47051</td>
      <td>29.573710</td>
      <td>29.573710</td>
      <td>59.147420</td>
      <td>Na</td>
      <td>146</td>
      <td>0.121982</td>
    </tr>
    <tr>
      <th>2013-09-30_to_2013-10-06</th>
      <td>65944</td>
      <td>166822</td>
      <td>0.000038</td>
      <td>2.529753</td>
      <td>2.529753</td>
      <td>5.059505</td>
      <td>30185</td>
      <td>15972</td>
      <td>46157</td>
      <td>30.706182</td>
      <td>30.706182</td>
      <td>61.412364</td>
      <td>7.681334487</td>
      <td>187</td>
      <td>0.111434</td>
    </tr>
    <tr>
      <th>2013-10-07_to_2013-10-13</th>
      <td>64295</td>
      <td>163760</td>
      <td>0.000040</td>
      <td>2.547010</td>
      <td>2.547010</td>
      <td>5.094020</td>
      <td>30048</td>
      <td>16136</td>
      <td>46184</td>
      <td>27.575940</td>
      <td>27.575940</td>
      <td>55.151879</td>
      <td>10.09391679</td>
      <td>274</td>
      <td>0.117625</td>
    </tr>
    <tr>
      <th>2013-10-14_to_2013-10-20</th>
      <td>69933</td>
      <td>176285</td>
      <td>0.000036</td>
      <td>2.520770</td>
      <td>2.520770</td>
      <td>5.041540</td>
      <td>33142</td>
      <td>19229</td>
      <td>52371</td>
      <td>28.992855</td>
      <td>28.992855</td>
      <td>57.985710</td>
      <td>Na</td>
      <td>266</td>
      <td>0.108987</td>
    </tr>
    <tr>
      <th>2013-10-21_to_2013-10-27</th>
      <td>80556</td>
      <td>214668</td>
      <td>0.000033</td>
      <td>2.664829</td>
      <td>2.664829</td>
      <td>5.329659</td>
      <td>38376</td>
      <td>22074</td>
      <td>60450</td>
      <td>27.780600</td>
      <td>27.780600</td>
      <td>55.561201</td>
      <td>Na</td>
      <td>271</td>
      <td>0.122053</td>
    </tr>
    <tr>
      <th>2013-10-28_to_2013-11-03</th>
      <td>71727</td>
      <td>185615</td>
      <td>0.000036</td>
      <td>2.587798</td>
      <td>2.587798</td>
      <td>5.175596</td>
      <td>33397</td>
      <td>17952</td>
      <td>51349</td>
      <td>23.529886</td>
      <td>23.529886</td>
      <td>47.059773</td>
      <td>Na</td>
      <td>177</td>
      <td>0.110815</td>
    </tr>
    <tr>
      <th>2013-11-04_to_2013-11-10</th>
      <td>97739</td>
      <td>259483</td>
      <td>0.000027</td>
      <td>2.654856</td>
      <td>2.654856</td>
      <td>5.309713</td>
      <td>43200</td>
      <td>24946</td>
      <td>68146</td>
      <td>25.209999</td>
      <td>25.209999</td>
      <td>50.419997</td>
      <td>Na</td>
      <td>Na</td>
      <td>0.105403</td>
    </tr>
    <tr>
      <th>2013-11-11_to_2013-11-17</th>
      <td>96291</td>
      <td>257001</td>
      <td>0.000028</td>
      <td>2.669003</td>
      <td>2.669003</td>
      <td>5.338007</td>
      <td>42679</td>
      <td>24588</td>
      <td>67267</td>
      <td>35.354349</td>
      <td>35.354349</td>
      <td>70.708698</td>
      <td>Na</td>
      <td>Na</td>
      <td>0.101211</td>
    </tr>
    <tr>
      <th>2013-11-18_to_2013-11-24</th>
      <td>147046</td>
      <td>405149</td>
      <td>0.000019</td>
      <td>2.755253</td>
      <td>2.755253</td>
      <td>5.510507</td>
      <td>66194</td>
      <td>33168</td>
      <td>99362</td>
      <td>41.730604</td>
      <td>41.730604</td>
      <td>83.461207</td>
      <td>Na</td>
      <td>Na</td>
      <td>0.095868</td>
    </tr>
    <tr>
      <th>2013-11-25_to_2013-12-01</th>
      <td>165768</td>
      <td>456183</td>
      <td>0.000017</td>
      <td>2.751936</td>
      <td>2.751936</td>
      <td>5.503873</td>
      <td>88358</td>
      <td>37205</td>
      <td>125563</td>
      <td>45.653230</td>
      <td>45.653230</td>
      <td>91.306461</td>
      <td>Na</td>
      <td>Na</td>
      <td>0.106394</td>
    </tr>
    <tr>
      <th>2013-12-02_to_2013-12-08</th>
      <td>165772</td>
      <td>463363</td>
      <td>0.000017</td>
      <td>2.795183</td>
      <td>2.795183</td>
      <td>5.590365</td>
      <td>80395</td>
      <td>36819</td>
      <td>117214</td>
      <td>34.163976</td>
      <td>34.163976</td>
      <td>68.327953</td>
      <td>Na</td>
      <td>Na</td>
      <td>0.096236</td>
    </tr>
  </tbody>
</table>
</div>



As there are some data missing in average shortest path length and diameter, the statstics of them were calculated with the data available.
Given the data above, summary statistics could be calculated as follow:


```python
data_matrix = np.array(data)
stats_names = ['mean', 'Median', 'Maximum', 'Minimum', 'Standard deviation']
# Data buffer
data_stats = []

for counter in range(0, len(col_labels)):
    mean=np.mean(data_matrix[:,counter])
    median=np.median(data_matrix[:,counter])
    maximum=np.max(data_matrix[:,counter])
    minimum=np.min(data_matrix[:,counter])
    std=np.std(data_matrix[:,counter])
    data_stats.append([mean, median, maximum, minimum, std])
    
temp = np.array(data_stats)
data_stats=list(temp.T)

df_stats = pd.DataFrame(
    data_stats,
    index=stats_names,
    columns=col_labels
)

df_stats
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
      <th>Number of nodes</th>
      <th>Number of links</th>
      <th>Density</th>
      <th>Average in-degree</th>
      <th>Average out-degree</th>
      <th>Average total degree</th>
      <th>Maximum in-degree</th>
      <th>Maximum out-degree</th>
      <th>Maximum total degree</th>
      <th>Average in-strength</th>
      <th>Average out-strength</th>
      <th>Average total strength</th>
      <th>Average shortest path length</th>
      <th>Diameter</th>
      <th>Average clustering coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>92772.15385</td>
      <td>246539.3077</td>
      <td>0.000032</td>
      <td>2.618975</td>
      <td>2.618975</td>
      <td>5.237949</td>
      <td>44296.84615</td>
      <td>22655.076920</td>
      <td>66951.92308</td>
      <td>30.823555</td>
      <td>30.823555</td>
      <td>61.647111</td>
      <td>8.208471</td>
      <td>230.250000</td>
      <td>0.109970</td>
    </tr>
    <tr>
      <th>Median</th>
      <td>71727.00000</td>
      <td>185615.0000</td>
      <td>0.000036</td>
      <td>2.587798</td>
      <td>2.587798</td>
      <td>5.175596</td>
      <td>33397.00000</td>
      <td>19229.000000</td>
      <td>52371.00000</td>
      <td>28.992855</td>
      <td>28.992855</td>
      <td>57.985710</td>
      <td>7.625818</td>
      <td>248.000000</td>
      <td>0.110815</td>
    </tr>
    <tr>
      <th>Maximum</th>
      <td>165772.00000</td>
      <td>463363.0000</td>
      <td>0.000042</td>
      <td>2.795183</td>
      <td>2.795183</td>
      <td>5.590365</td>
      <td>88358.00000</td>
      <td>37205.000000</td>
      <td>125563.00000</td>
      <td>45.653230</td>
      <td>45.653230</td>
      <td>91.306461</td>
      <td>10.093917</td>
      <td>291.000000</td>
      <td>0.122053</td>
    </tr>
    <tr>
      <th>Minimum</th>
      <td>59574.00000</td>
      <td>148959.0000</td>
      <td>0.000017</td>
      <td>2.500403</td>
      <td>2.500403</td>
      <td>5.000806</td>
      <td>29021.00000</td>
      <td>15198.000000</td>
      <td>44285.00000</td>
      <td>23.529886</td>
      <td>23.529886</td>
      <td>47.059773</td>
      <td>7.488331</td>
      <td>146.000000</td>
      <td>0.095868</td>
    </tr>
    <tr>
      <th>Standard deviation</th>
      <td>38674.53532</td>
      <td>112990.0500</td>
      <td>0.000009</td>
      <td>0.098123</td>
      <td>0.098123</td>
      <td>0.196246</td>
      <td>19664.31781</td>
      <td>7884.613355</td>
      <td>27411.83800</td>
      <td>6.479883</td>
      <td>6.479883</td>
      <td>12.959766</td>
      <td>1.259450</td>
      <td>53.914615</td>
      <td>0.008760</td>
    </tr>
  </tbody>
</table>
</div>



## Exercise 1.2
In this exercise, the **first week** of data is used, which is form 2013-09-09 to 2013-09-15.The reason for using this sort of data is because:
1. **Far away from the price bubble period**. While the price bubble may affect people's decisions and behaviour on bitcoin transactions, these data are believed to contain the purest characteristics of the network.
1. **Contains the least nodes and edges**. It eases the difficulty for computation and allows to focus on the data analyse.

The following graphs illustrate the degree distribution:


```python
# Use data between 2013-09-09 an 2013-09-15, week1
G=GSCC[0]
```


```python
bins = np.logspace(np.log10(1), np.log10(1e5), 25)

def plot_hist(degrees):
    plt.hist([deg for _, deg in degrees], bins)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Fraction')

plt.figure(figsize = (15, 3))

plt.subplot(1, 3, 1)
plt.title('In-degree distribution')
plot_hist(G.in_degree())

plt.subplot(1, 3, 2)
plt.title('Out-degree distribution')
plot_hist(G.out_degree())

plt.subplot(1, 3, 3)
plt.title('Total-degree distribution')
plot_hist(G.degree())

plt.show()
```


    
![png](./pic/output_21_0.png)
    



```python
def plot_scatter(degrees):
    distribution_dict=dict([i, test.count(i)] for i in [deg for _, deg in degrees])
    distribution_array=np.array(list(distribution_dict.items()))
    plt.scatter(distribution_array[:,0], distribution_array[:,1]/distribution_array[:,1].sum())
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('p(k)')

plt.figure(figsize = (15, 3))
    
plt.subplot(1, 3, 1)
plt.title('In-degree distribution')
plot_scatter(G.in_degree())

plt.subplot(1, 3, 2)
plt.title('Out-degree distribution')
plot_scatter(G.out_degree())

plt.subplot(1, 3, 3)
plt.title('Total-degree distribution')
plot_scatter(G.degree())
```


    
![png](./pic/output_22_0.png)
    


The graphs demostrate that the bitcoin transaction network presents a clear curve of **power-law distribution**. As a result, it is reasonable to assert that the bitcoin transaction network is a **scale-free network**. The reasons are:
### Degree Distribution
The scale-free network refers to the networks that degree distribution follows a power law. That is $p(k)$ of nodes in the network having $k$ connections to other nodes goes for large values of $k$ as:
$$p(k)\approx k^{-\gamma}$$
$$\gamma\approx-\log_k{p(k)}$$


```python
k_w1=np.array([degree for _, degree in G.degree()])
k=np.median(k_w1)
pk=np.sum(k_w1==k)/np.sum(k_w1)
gamma=-math.log(pk, k)
print("gamma= ",gamma)
```

    gamma=  2.1664521570364217
    


```python
estimate_gamma=3
gamma_error=abs(gamma-estimate_gamma)/gamma
print("The error of gamma is: ", gamma_error)
```

    The error of gamma is:  0.3847524812658787
    

In this case, I use the median values to derive the relation of the distribution $p(k)$ and the degree $k$. It gives the relationship: $p(k)\approx k^{-2.16}$. The degree distribution of **BA (Barabasi-Albert) network** is roughly $p(k)\approx k^{-3}$ as the error of gamma is rather small. So there is a probability that the bitcoin transaction network is a BA network with a **classical fitness function (base on the degree of the node) for preferential attachment** in terms of degree distribution. 

As for the random network, specifically **ER (Erdős–Rényi) network** in this case, the degree distribution should present **binomial distribution**, which gives $$p(k)={N \choose k}p^k(1-p)^{N-1-k}$$ The graph should be like:


![](./pic/1.png)

It shows a significantly difference with bitcoin traction network. 

As for the **small world network (Watts-Strogatz)**, each node would have the same amount of links (Connect a node to its $k$ Nearest neighbours). The distribution of degree should be a straight line since every nodes would have the same number of degree $k$, which is contradict to the result.

### Clustering Coefficient

I also researh into the distribution of clustering coefficient $C(k)$. Here I plot the distribution of node's degree $k$ and the corresponding clustering coefficient $C(k)$


```python
ck=nx.clustering(G)
ck_value=np.array(list(ck.values()))
```

Again, the median value of $k$ is used to estimate the relationship of $C(k)$ and $k$.


```python
plt.scatter(k_w1, ck_value)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k')
plt.ylabel('C(k)')
```




    Text(0, 0.5, 'C(k)')




    
![png](./pic/output_31_1.png)
    


It can be learned from the graph that the clustering coefficient of bitcoin network also rougly converges in power-law. The relationship of BA network generally follows: $$C(k)\approx k^{-1}$$

To verify the property, again we use the median $k$ for estimation:


```python
median_nodes=[node_id for node_id, degree in G.degree() if(degree == k)]
median_node=random.choice(median_nodes)
median_node
```




    '4224160'




```python
median_ck=nx.clustering(G, median_node)
print("The median k = ", k)
print("The median c(k) = ", median_ck)
```

    The median k =  4.0
    The median c(k) =  0.16666666666666666
    


```python
estimate_ck=1/k
ck_error=(median_ck-estimate_ck)/median_ck
print('The error of clustering coefficient is: ', ck_error)
```

    The error of clustering coefficient is:  -0.5000000000000001
    

As the absolute value of estimated error $|error_{c(k)}|<1$, which seems tolerable. The analyse of clustering coefficient also support that the bitcoin transaction network is a scale-free network.

For the random network (Erdős–Rényi), the clustering coefficient is $$C(k) = p$$ It means that it will converage around a certain probability value, which is not in line with the data.

As for the small world network (Watts-Strogatz), the clustering coefficient is roughly around $C(k) = \dfrac{k}{N}$. As the algorithm feature decide that the $k$ is a fixed number. This means that initially, each node generates a fixed amount of links with neighbours. After that, these links are randomly disturbed with nodes. Theoretically, every nodes are exposed to equal opportunity to form triangles. However, the bitcoin transaction network shows a difference on the distribution of clustering coefficient.

### Diameter & Average shortest path length

The data presented in Exercise 1 gives that the diameter for week1 is:
$$d(week1)=297$$
And the average shortest path langth (ASPL) is:
$$l(week1)=7.570300706$$
The definition of diameter is the maximum shortest paths of all pair of nodes in a network. These pieces of data provide insight into the efficiency of information or mass transport on a network  (Mao  and  Zhang,  2013). Specifically in the bitcoin transaction network, the data describe the efficiency of bitcoin transference amount traders. In this case, I also research in the relationship of diamter and the network size $N$ with three different network.


```python
N_week1=G.number_of_nodes()
ASPL=data[0][12]
```

In BA network, the relation of diameter and network size roughly follows: 
$$l\approx\dfrac{\ln N}{\ln \ln N}$$


```python
estimate_ASPL=math.log(N_week1)/math.log(math.log(N_week1))
ASPL_error=(ASPL-estimate_ASPL)/estimate_ASPL
print('The error of ASPL of week 1 is: ', ASPL_error)
```

    The error of ASPL of week 1 is:  0.6506931303506265
    

The error of ASPL is large in this case. However, in consideration of the network size, the estimated value and the actual value are generally in the same order of magnitude. So it is reasonable to convince that the bitcoin transaction network is a BA network.

For random network (Erdős–Rényi), the shortest path length $l$ of a certain node satisfies that $$k(k − 1)^{l-1} ≈ k ^{l}$$
This means that the diameter of the network should be $$l_{max}=\log_{k}N$$


```python
estimate_ER_diameter=math.log(N_week1, k)
print('The estimated diameter of ER network is: ', estimate_ER_diameter)
```

    The estimated diameter of ER network is:  7.931197604655606
    

The number is far smaller than the actual diameter value $d(week1)=297$. The diameter fomula remains the same for small world network (Watts-Strogatz), because both of the algorithms do not contain any priority strategy (preferential attachment) to add new nodes. As a result, the bitcoin transaction network is not a small world network.

## Exercise 1.3
Given the data above, the graphs the 9 quantities in temporal evolution diagrams would be (Some data were missing):


```python
weekNum=len(names)
time_scale=range(1,weekNum+1)
                      
plt.figure(figsize = (20, 16))

graph_num=1;
for counter in range(15):
    # Except the in and out data
    if counter+1 in [4,5,7,8,10,11]:
        continue
    plt.subplot(3,3,graph_num)
    graph_num+=1
    plt.bar(time_scale, np.array(data)[:,counter])
    plt.title(col_labels[counter])
    plt.xlabel('week')
    plt.ylabel(col_labels[counter])
```


    
![png](./pic/output_46_0.png)
    


### General tendency (Whether these quantities evolved in the way that you expected, and why)
The quantities generally evolved as my expectation except few abnormal changes during the price bubble period. 

The bitcoin is a kind of decentralized currency and every transactions were recorded. This means that the nodes and links in a transaction network would keep expending. The graphs generally illustrate the tendancy except the abnormal decline saw in week 8. This might due to the cyber attacks which cause the split of bitcoin block chain and the bitcoin loss.

As it is evaluated in Exercise 1.2, the bitcoin transaction network might be a scale-free network, which means new nodes were added to the network evenly according to a specific rule (fitness function). Usually, the old nodes would have great advantage over the later-joined nodes. This might account for the almost constant average-total-degree and the increasing maximum-total-degree, because the oldest nodes keep connecting to the new nodes with higher possibility.

### Any signal that might have predicted the bubble
The price bubble began in week 5, in this week, an anomalous decrease on average-total-strength and increase on density can be spotted. This might infer that less quantity of bitcoins were used averagely among transactions. Suppose the total transaction demand remain unchanged. It means that each transaction is conducted in a lower amount of bitcoin. This indicated the rise of bitcoin value.

### Any signiﬁcant change during the bubble
During the bubble period, notably in week 8, it saw an unusual reduction on both the nodes and links. As it is mentioned above, in information on a bitcoin transaction track is not likely to lose. This might due to the attack which cause the evaporation of bitcoins. Accordingly, the unit price of bitcoin raised significantly and eventually led to the price bubble.

etc.


### Any signiﬁcant change after the bubble
The price bubble ends in week 11, 

# Part II
## Exercise 2.1

The three weeks I choose:
1. Before bubble, **week 1**: The data of this week is the most far away from the price bubble, which could provide an insight on the characteristics of bitcoin network.
1. During bubble, **week 8**: In this week, the total number of nodes and links abnormally reduced compared to the last few weeks. As the total amount of information in the block chain can not be reduced due to its design, the anomalous decline might be caused by a cyber attack. As a result, the changes of network properties is vital for victimization analyse.
1. After bubble, **week 13**: The week after the price bubble period. To see the changes of network properties. 


```python
G1 = GSCC[0]
G2 = GSCC[7]
G3 = GSCC[12]
```

In this part of exercise, I use three all three centrality concepts to measure the significance of nodes, which is:
1. **degree centrality**: The number of degree of one node. It measure directly the connectivity of one node. In a BA network, this ability is affected by the preferential attachment directly. The changes of degree centrality provide direct insight into how new node are added to the network.
1. **eigenvector centrality**: centrality computes the centrality for a node based on the centrality of its neighbors. Accordingly, a node with a high eigenvector score is one that is adjacent to nodes that also have high eigenvector scores(Borgatti, 2005). This measure is essential for discovering central hubs such as exchanges, miners, or “laundry services” that are important nodes in the Bitcoin network (Baumann and Fabian etc., 2014).

In my opinion, bitcoin transaction network belongs to financial network. The weight of edge represent the quantity of transactions (the `qty` attribute). Since the bitcoin is a kind of decentralized currency, which means the transactions happen in a bitcoin network are more likely between parties directly without any medium, instead of researching on the betweenness of nodes, I focus more on the transaction behavior itself, which is closely related to security issues.

As it is showed at degree distribution diagram in Exercise 1.2, there exist few nodes with higher degree than others. I believe that these nodes play a pivotal role in the transaction network and through which we can have a insight on the cause of price bubble. Therefore, I measured the closeness centrality, which illustrate how close one node is to other nodes, and tryed to find the **hubs** of the network.

For each of the centrality, I pick up the 10 best performance nodes

Here is the necessary tool functions for centrality measurement


```python
# Obtian the top nodes with its degree
def top_ten(deg_dist):
    return [[id,degree] for id, degree in sorted(deg_dist, key = lambda p: -p[1])][:10]

def logloghist(dist, xmin, xmax, bins = 25):
    plt.hist(list(dist), bins = np.geomspace(xmin, xmax, bins))
    plt.xscale('log')
    plt.yscale('log')

def calculateStats(G, eigen_centrality):
    top10_degree=np.array(top_ten(G.degree()))
    top10_degree_strength=np.array(top_ten(G.degree(weight="qty")))
    top10_degree_complete=np.concatenate([top10_degree, top10_degree_strength], axis=1)

    top10_eigen=np.array(top_ten(eigen_centrality.items()))
    top10_eigen_degree=np.array([degree for _, degree in G.degree(top10_eigen[:,0])]).T
    top10_eigen_strength=np.array([degree for _, degree in G.degree(top10_eigen[:,0], weight="qty")])
    top10_eigen_temp=np.concatenate([top10_eigen, np.array([top10_eigen_degree]).T], axis=1)
    top10_eigen_complete=np.concatenate([top10_eigen_temp, np.array([top10_eigen_strength]).T], axis=1)
    
    return [top10_degree_complete, top10_eigen_complete]

def plot_table(degree, eigen):
    rank=list(range(1,11))
    df_=pd.DataFrame({
        'Node (Degree centrality)': list(degree[:,0]),
        'Degree centrality':list(degree[:,1]),
        'Strength (Degree centrality)': list(degree[:,2]),
        'Node (Eigen centrality)':list(eigen[:,0]),
        'Eigen centrality':list(eigen[:,1]),
        'Degree (Eigen centrality)': list(eigen[:,2]),
        'Strength (Eigen centrality)': list(eigen[:,3])
        },
        index=rank
    )
    return df_
```


```python
# Katz centrality -- out of memory, not work
katz_centrality_G1=nx.katz_centrality_numpy(G1, weight="qty")
katz_centrality_G2=nx.katz_centrality_numpy(G2, weight="qty")
katz_centrality_G3=nx.katz_centrality_numpy(G3, weight="qty")
```


```python
# Eigen centrality
eigen_centrality_G1=nx.eigenvector_centrality_numpy(G1, weight="qty")
eigen_centrality_G2=nx.eigenvector_centrality_numpy(G2, weight="qty")
eigen_centrality_G3=nx.eigenvector_centrality_numpy(G3, weight="qty")
```

### Before the bubble


```python
[top10_degree_week1, top10_eigen_week1]=calculateStats(G1, eigen_centrality_G1)

df_before=plot_table(top10_degree_week1, top10_eigen_week1)
df_before
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
      <th>Node (Degree centrality)</th>
      <th>Degree centrality</th>
      <th>Strength (Degree centrality)</th>
      <th>Node (Eigen centrality)</th>
      <th>Eigen centrality</th>
      <th>Degree (Eigen centrality)</th>
      <th>Strength (Eigen centrality)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24778</td>
      <td>44285</td>
      <td>24778</td>
      <td>1056959</td>
      <td>0.6527250342066321</td>
      <td>1604</td>
      <td>48820.33962495007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1056959</td>
      <td>1604</td>
      <td>1056959</td>
      <td>4224108</td>
      <td>0.5516199740736305</td>
      <td>8</td>
      <td>21423.20975519</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3941484</td>
      <td>964</td>
      <td>4224108</td>
      <td>2696272</td>
      <td>0.3238269084029462</td>
      <td>2</td>
      <td>11815.643285400001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3483370</td>
      <td>243</td>
      <td>46606</td>
      <td>4191029</td>
      <td>0.26305325828229476</td>
      <td>5</td>
      <td>10883.25242798</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3375464</td>
      <td>230</td>
      <td>4086705</td>
      <td>2216968</td>
      <td>0.1828566872871267</td>
      <td>22</td>
      <td>8747.0498</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4163907</td>
      <td>229</td>
      <td>2121345</td>
      <td>24778</td>
      <td>0.14461401611480132</td>
      <td>44285</td>
      <td>415860.2060226785</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3183164</td>
      <td>224</td>
      <td>2696272</td>
      <td>2121345</td>
      <td>0.08232392861487074</td>
      <td>69</td>
      <td>12108.401467929998</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4158569</td>
      <td>221</td>
      <td>2300615</td>
      <td>4191803</td>
      <td>0.052444879973456406</td>
      <td>9</td>
      <td>1681.6416294599999</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2508019</td>
      <td>213</td>
      <td>4191029</td>
      <td>4225205</td>
      <td>0.05025142321858903</td>
      <td>3</td>
      <td>7999.9995</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3697450</td>
      <td>189</td>
      <td>3163935</td>
      <td>2881501</td>
      <td>0.04545275534255438</td>
      <td>20</td>
      <td>7235.167611999999</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (5, 5))
plt.suptitle('Eigenvalue Centrality Distributions (week1)')
logloghist(eigen_centrality_G1.values(), 1e-10, 1)
plt.xlabel('Eigenvalue centrality score')
plt.ylabel('Node frequency')
plt.show()
```


    
![png](./pic/output_58_0.png)
    


### During bubble


```python
[top10_degree_week8, top10_eigen_week8]=calculateStats(G2, eigen_centrality_G2)

df_during=plot_table(top10_degree_week8, top10_eigen_week8)
df_during
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
      <th>Node (Degree centrality)</th>
      <th>Degree centrality</th>
      <th>Strength (Degree centrality)</th>
      <th>Node (Eigen centrality)</th>
      <th>Eigen centrality</th>
      <th>Degree (Eigen centrality)</th>
      <th>Strength (Eigen centrality)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24778</td>
      <td>51349</td>
      <td>24778</td>
      <td>46606</td>
      <td>0.707337774873297</td>
      <td>65</td>
      <td>146130.63405291998</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1056959</td>
      <td>1698</td>
      <td>46606</td>
      <td>3456919</td>
      <td>0.7060748751584912</td>
      <td>2</td>
      <td>135000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4614011</td>
      <td>1105</td>
      <td>3456919</td>
      <td>24778</td>
      <td>0.03205967014470368</td>
      <td>51349</td>
      <td>346941.08640882315</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4726253</td>
      <td>638</td>
      <td>4220239</td>
      <td>4739938</td>
      <td>0.007233886586728299</td>
      <td>5</td>
      <td>2745.69159802</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3183164</td>
      <td>461</td>
      <td>4726604</td>
      <td>3233393</td>
      <td>0.0033187705138717207</td>
      <td>6</td>
      <td>13997.9955</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3941484</td>
      <td>419</td>
      <td>4459315</td>
      <td>4293378</td>
      <td>0.002956175278664835</td>
      <td>32</td>
      <td>11799.91739405</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4045065</td>
      <td>414</td>
      <td>4373184</td>
      <td>4733887</td>
      <td>0.002180901586487859</td>
      <td>4</td>
      <td>9199.027326489999</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4373184</td>
      <td>370</td>
      <td>3233393</td>
      <td>4754288</td>
      <td>0.001793723067246086</td>
      <td>3</td>
      <td>349.96900822</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4688560</td>
      <td>317</td>
      <td>4293378</td>
      <td>2665227</td>
      <td>0.0016955405593750896</td>
      <td>11</td>
      <td>336.38234554999997</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4158737</td>
      <td>274</td>
      <td>4463024</td>
      <td>4661163</td>
      <td>0.0016795938939987598</td>
      <td>9</td>
      <td>6667.1510885</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (5, 5))
plt.suptitle('Eigenvalue Centrality Distributions (week8)')
logloghist(eigen_centrality_G2.values(), 1e-10, 1)
plt.xlabel('Eigenvalue centrality score')
plt.ylabel('Node frequency')
plt.show()
```


    
![png](./pic/output_61_0.png)
    


### After bubble


```python
[top10_degree_week13, top10_eigen_week13]=calculateStats(G3, eigen_centrality_G3)

df_after=plot_table(top10_degree_week13, top10_eigen_week13)
df_after
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
      <th>Node (Degree centrality)</th>
      <th>Degree centrality</th>
      <th>Strength (Degree centrality)</th>
      <th>Node (Eigen centrality)</th>
      <th>Eigen centrality</th>
      <th>Degree (Eigen centrality)</th>
      <th>Strength (Eigen centrality)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>24778</td>
      <td>117214</td>
      <td>24778</td>
      <td>4987284</td>
      <td>0.8603170868071799</td>
      <td>540</td>
      <td>194808.4679811002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4614011</td>
      <td>1767</td>
      <td>4987284</td>
      <td>4086210</td>
      <td>0.2498419360093388</td>
      <td>9</td>
      <td>40233.002100000005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1056959</td>
      <td>1736</td>
      <td>3547244</td>
      <td>24778</td>
      <td>0.24225930611221774</td>
      <td>117214</td>
      <td>735700.1440940073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5314986</td>
      <td>870</td>
      <td>4086210</td>
      <td>5580241</td>
      <td>0.10750566250631072</td>
      <td>6</td>
      <td>5993.69457104</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4195093</td>
      <td>865</td>
      <td>5413973</td>
      <td>5443585</td>
      <td>0.071740053010437</td>
      <td>5</td>
      <td>3999.6778782399997</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4987284</td>
      <td>540</td>
      <td>46606</td>
      <td>5595338</td>
      <td>0.07169077478305512</td>
      <td>4</td>
      <td>3996.16266</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4045065</td>
      <td>513</td>
      <td>3456919</td>
      <td>5580221</td>
      <td>0.07154498978576773</td>
      <td>4</td>
      <td>3987.4122248000003</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3697902</td>
      <td>485</td>
      <td>717202</td>
      <td>5506636</td>
      <td>0.07149832438911337</td>
      <td>3</td>
      <td>3986.20093336</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3183164</td>
      <td>455</td>
      <td>4655026</td>
      <td>5596462</td>
      <td>0.0714944789895615</td>
      <td>3</td>
      <td>3985.9865431999997</td>
    </tr>
    <tr>
      <th>10</th>
      <td>309431</td>
      <td>429</td>
      <td>2781928</td>
      <td>5519473</td>
      <td>0.07143300524489174</td>
      <td>4</td>
      <td>3977.57023586</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize = (5, 5))
plt.suptitle('Eigenvalue Centrality Distributions (week13)')
logloghist(eigen_centrality_G3.values(), 1e-10, 1)
plt.xlabel('Eigenvalue centrality score')
plt.ylabel('Node frequency')
plt.show()
```


    
![png](./pic/output_64_0.png)
    


I believe my chosen centrality method could effectively measure the significance of the nodes. As it is discussed previously, the **betweenness centrality** focus on analyzing the connections between multiple groups in the network, which might not be suitable for this bitcoin transaction network. 

Besides, the **closeness centrality** measure how close a node is to others. In a bitcoin network, the weight of graph represents the volume of trade, which is irrelevant to the idea of distance. The closeness centrality might be helpful when measuring social networks since the relationship between individuals can be quantified as distance.

Finally, the top 10 nodes highlighted by the eigenvalue centrality drastically change over the few weeks, especially during the bubble period. The changes of these hubs might reveal the contributing factors of price bubble of the network.

## Exercise 2.2
These nodes represent the **hubs** of the network.As the bitcoin transaction network is a BA network analyzed in Exercise 1.2. These nodes are more likely to have the following characters:

1. Higher degree. As new nodes are connected to nodes in the BA network according to the its degree, the old nodes have advantage over other nodes
1. Large influence to the network. The eigenvector centrality measures is calculated according to the centrality of surrounding nodes. This means that the nodes with high score of centrality are more likely to directly connect to other high score nodes. As a result, these nodes would have great influence to the overall network.

# Part III
## Exercise 3.1
As it is measured in Exercise 1.2, the bitcoin transaction network is more likely to be a scale-free network. It means that the network is robust to **random failures** instead of **target attacks**. Again, I take week1 (2013-09-09 to 2013-09-15) as an example, two scenarios  can be distinguished

### Attack to hubs
In this situation, the attack might cause a price crash or major disruption. As it is analyzed in Exercise 1, the bitcoin transaction network is a scale-free network, which means that most of the nodes in the network are sparsely connected while few of them have large amount of links. This means that there is large amount of bitcoins "flow" through these nodes. Once thery are attacked, large amount of bitcoins may loss. This might distory the bitcoin market as the bitcoins in circulation decrease and eventually cause a price bubble.

As it is measure in Exercise 2, the nodes with high score of centrality usually are the nodes with high strength, while the strenth in a the network actually represent the bitcoins that circulate in the market. If multiple attacks happen in a same time toward the hubs, it might cause the disruption of the whole network

### Attack to non-hub nodes
In this situation, as it is mentioned previously, only limited amount of bitcoin "flow" through these nodes. The major function of bitcoin network won't be greatly affected in a short time.

## Exercise 3.2

In this part, I track the nodes with **highest score of eigenvalue centrality in all three weeks**. The nodes are:


```python
target_nodes=top10_eigen_week1[:,0]
F=[]
for G in GSCC:
    S_in=np.array([value for _, value in G.in_degree(target_nodes, weight="qty")])
    S_out=np.array([value for _, value in G.out_degree(target_nodes, weight="qty")])
    S_tot=np.array([value for _, value in G.degree(target_nodes, weight="qty")])
    F_local=np.divide((S_out-S_in),S_tot)
    F.append(F_local)
    
df_flow=pd.DataFrame(
    F,
    index=names,
    columns=list(range(1,11))
)
df_flow
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-09-09_to_2013-09-15</th>
      <td>0.022488</td>
      <td>-0.002644</td>
      <td>0.033332</td>
      <td>-0.000024</td>
      <td>-0.087365</td>
      <td>0.065325</td>
      <td>-0.105757</td>
      <td>-0.119089</td>
      <td>-6.250000e-08</td>
      <td>-0.000123</td>
    </tr>
    <tr>
      <th>2013-09-16_to_2013-09-22</th>
      <td>-0.028067</td>
      <td>0.003831</td>
      <td>0.133433</td>
      <td>0.058316</td>
      <td>0.042684</td>
      <td>-0.274246</td>
      <td>0.143569</td>
      <td>-0.000012</td>
      <td>-2.858881e-03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-23_to_2013-09-29</th>
      <td>-0.014613</td>
      <td>0.030253</td>
      <td>0.012832</td>
      <td>-0.116209</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-30_to_2013-10-06</th>
      <td>-0.016825</td>
      <td>0.010410</td>
      <td>0.062953</td>
      <td>-0.021533</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-07_to_2013-10-13</th>
      <td>-0.006380</td>
      <td>-0.121095</td>
      <td>0.028514</td>
      <td>0.015942</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-14_to_2013-10-20</th>
      <td>-0.050263</td>
      <td>0.090697</td>
      <td>0.070483</td>
      <td>-0.010132</td>
      <td>-0.133035</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-21_to_2013-10-27</th>
      <td>-0.028448</td>
      <td>0.989139</td>
      <td>-0.000092</td>
      <td>0.055026</td>
      <td>0.400481</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-28_to_2013-11-03</th>
      <td>0.004757</td>
      <td>-0.274212</td>
      <td>0.100900</td>
      <td>-0.579983</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-04_to_2013-11-10</th>
      <td>0.031664</td>
      <td>0.051245</td>
      <td>-0.522741</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-11_to_2013-11-17</th>
      <td>0.076620</td>
      <td>-0.098466</td>
      <td>0.069711</td>
      <td>0.905273</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-18_to_2013-11-24</th>
      <td>-0.102049</td>
      <td>0.254944</td>
      <td>0.048172</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-25_to_2013-12-01</th>
      <td>0.373018</td>
      <td>0.061364</td>
      <td>-0.023835</td>
      <td>0.485425</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-12-02_to_2013-12-08</th>
      <td>0.009801</td>
      <td>0.000051</td>
      <td>-0.021342</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
target_nodes=top10_eigen_week8[:,0]
F=[]
for G in GSCC:
    S_in=np.array([value for _, value in G.in_degree(target_nodes, weight="qty")])
    S_out=np.array([value for _, value in G.out_degree(target_nodes, weight="qty")])
    S_tot=np.array([value for _, value in G.degree(target_nodes, weight="qty")])
    F_local=(S_out-S_in)/S_tot
    F.append(F_local)
    
df_flow=pd.DataFrame(
    F,
    index=names,
    columns=list(range(1,11))
)
df_flow
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-09-09_to_2013-09-15</th>
      <td>0.008172</td>
      <td>-0.130435</td>
      <td>0.065325</td>
      <td>-1.833037e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-16_to_2013-09-22</th>
      <td>0.023608</td>
      <td>-0.281690</td>
      <td>0.042684</td>
      <td>-8.573568e-02</td>
      <td>-0.004867</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-23_to_2013-09-29</th>
      <td>-0.008714</td>
      <td>0.083172</td>
      <td>0.012832</td>
      <td>-2.018064e-01</td>
      <td>-0.000436</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-30_to_2013-10-06</th>
      <td>0.016615</td>
      <td>-0.186312</td>
      <td>0.062953</td>
      <td>-2.459723e-01</td>
      <td>-0.001351</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-07_to_2013-10-13</th>
      <td>0.011689</td>
      <td>-0.161793</td>
      <td>0.028514</td>
      <td>1.649264e-01</td>
      <td>-0.000860</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-14_to_2013-10-20</th>
      <td>-0.003830</td>
      <td>0.538462</td>
      <td>-0.010132</td>
      <td>-2.496877e-01</td>
      <td>-0.000337</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-21_to_2013-10-27</th>
      <td>-0.021287</td>
      <td>-0.120000</td>
      <td>0.055026</td>
      <td>8.997891e-02</td>
      <td>-0.026743</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-28_to_2013-11-03</th>
      <td>-0.018874</td>
      <td>0.000000</td>
      <td>0.100900</td>
      <td>-1.456828e-07</td>
      <td>-0.000143</td>
      <td>-0.056822</td>
      <td>-0.000104</td>
      <td>-0.002835</td>
      <td>-0.002346</td>
      <td>-0.99538</td>
    </tr>
    <tr>
      <th>2013-11-04_to_2013-11-10</th>
      <td>-0.021071</td>
      <td>0.372578</td>
      <td>0.051245</td>
      <td>-7.186376e-02</td>
      <td>0.040880</td>
      <td>-0.003685</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-11_to_2013-11-17</th>
      <td>0.031436</td>
      <td>-0.529412</td>
      <td>0.069711</td>
      <td>-4.972325e-02</td>
      <td>-0.006018</td>
      <td>-0.995239</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-18_to_2013-11-24</th>
      <td>-0.001793</td>
      <td>0.443609</td>
      <td>0.048172</td>
      <td>-6.009191e-03</td>
      <td>-0.993964</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-25_to_2013-12-01</th>
      <td>-0.083538</td>
      <td>0.478261</td>
      <td>-0.023835</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-12-02_to_2013-12-08</th>
      <td>0.050513</td>
      <td>0.360000</td>
      <td>-0.021342</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
target_nodes=top10_eigen_week13[:,0]
F=[]
for G in GSCC:
    S_in=np.array([value for _, value in G.in_degree(target_nodes, weight="qty")])
    S_out=np.array([value for _, value in G.out_degree(target_nodes, weight="qty")])
    S_tot=np.array([value for _, value in G.degree(target_nodes, weight="qty")])
    F_local=(S_out-S_in)/S_tot
    F.append(F_local)
    
df_flow=pd.DataFrame(
    F,
    index=names,
    columns=list(range(1,11))
)
df_flow
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
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-09-09_to_2013-09-15</th>
      <td>-0.202552</td>
      <td>0.065325</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-16_to_2013-09-22</th>
      <td>0.042684</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-23_to_2013-09-29</th>
      <td>0.012832</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-09-30_to_2013-10-06</th>
      <td>0.062953</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-07_to_2013-10-13</th>
      <td>0.028514</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-14_to_2013-10-20</th>
      <td>-0.010132</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-21_to_2013-10-27</th>
      <td>0.055026</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-10-28_to_2013-11-03</th>
      <td>0.100900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-04_to_2013-11-10</th>
      <td>-0.556492</td>
      <td>0.051245</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-11_to_2013-11-17</th>
      <td>-0.008254</td>
      <td>-0.380037</td>
      <td>0.069711</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-18_to_2013-11-24</th>
      <td>-0.002176</td>
      <td>0.011533</td>
      <td>0.048172</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-11-25_to_2013-12-01</th>
      <td>-0.001295</td>
      <td>-0.023835</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-12-02_to_2013-12-08</th>
      <td>-0.001349</td>
      <td>-0.229488</td>
      <td>-0.021342</td>
      <td>-2.502630e-07</td>
      <td>-1.250101e-07</td>
      <td>-0.000192</td>
      <td>-0.000349</td>
      <td>-1.254327e-07</td>
      <td>-1.254395e-07</td>
      <td>-0.001254</td>
    </tr>
  </tbody>
</table>
</div>



The result demostrates that before and after the price bubble, except the most important nodes, the hubs changes sharply. 

## Exercise 3.3

### Ocation 1
One might happen in week 8. Because in this week, the strenths of the top 10 nodes are all inputs ($S^{in}>S^{out}$). But none of these nodes keeps high value of strenths after the weeks. These nodes might be the evidence of attackers that took the advantage of hacked crypto exchanges to earn profits.

### Ocation 2

# Reference
Annika, B, Benjamin, F and Matthias, L (2014): Exploring the Bitcoin Network. Institute of Information Systems, Humboldt University Berlin, Spandauer Str. 1, 10178 Berlin, Germany 


Borgatti, S (2005): Centrality and Network Flows. Social Networks 27(1): 55-71. 

Mao, G; Zhang, N (2013): Analysis of Average Shortest-Path  Length  of  Scale-Free  Network.  Journal  of Applied Mathematics, Vol. 2013, Article ID 865643


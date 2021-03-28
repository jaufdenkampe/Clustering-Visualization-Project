#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as mp
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set
mp.set_printoptions(threshold=10000)
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
df_segmentation = pd.read_csv('segmentation data.csv',index_col = 0)
import pickle


# In[ ]:





# In[48]:


df_segmentation.head()


# In[49]:


df_segmentation.describe()


# In[50]:


df_segmentation.corr()


# In[51]:


plt.figure(figsize = (12,9))
s = sns.heatmap(df_segmentation.corr(),
                annot = True,
                cmap = 'RdBu',
                vmin = -1,
                vmax = 1)
s.set_xticklabels(s.get_xticklabels(),rotation = 90,fontsize = 12)
s.set_yticklabels(s.get_yticklabels(),rotation = 0,fontsize = 12)
plt.title('Correlation Heatmap')
plt.show()


# In[52]:


plt.figure(figsize = (12,9))
plt.scatter(df_segmentation.iloc[:,2],df_segmentation.iloc[:,4])
plt.xlabel('age')
plt.ylabel('income')
plt.title('visualization of raw data')


# In[53]:


scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)


# In[54]:


hier_clust = linkage(segmentation_std, method = 'ward')


# In[58]:


plt.figure(figsize = (12,9))
plt.title('Heiarchical Clustering Dendrogram')
plt.xlabel('observations')
plt.ylabel('distance')
dendrogram(hier_clust,
           truncate_mode = 'level',
           p=5,
           show_leaf_counts = False,
           no_labels=True,
           )


# In[41]:





# In[61]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)
    


# In[ ]:





# In[63]:


plt.figure(figsize = (10,8))
plt.plot(range(1,11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('kmeans clustering')
plt.show()


# In[64]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)


# In[65]:


kmeans.fit(segmentation_std)


# In[136]:


df_segm_kmeans=df_segmentation.copy()
df_segm_kmeans['segment kmeans']=kmeans.labels_
df_segm_kmeans


# In[68]:


df_segm_analysis = df_segm_kmeans.groupby(['segment kmeans']).mean()
df_segm_analysis


# In[123]:


df_segm_analysis['n obs']=df_segm_kmeans[['segment kmeans', 'Sex']].groupby(['segment kmeans']).count()


# In[75]:


df_segm_analysis['prop ops'] = df_segm_analysis['n obs']/df_segm_analysis['n obs'].sum()


# In[76]:


df_segm_analysis


# In[79]:


df_segm_analysis.rename({0:'well off',
                         1: 'limited opportunity',
                        2:'standard',
                        3:'career focused'})


# In[81]:


df_segm_kmeans['labels']=df_segm_kmeans['segment kmeans'].map({0:'well off',
                                                                 1: 'limited opportunity',
                        2:'standard',
                        3:'career focused'})


# In[85]:


x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize=(10,8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_kmeans['labels'], palette = ['g', 'r', 'c', 'm'])
plt.title('segmentation kmeans')
plt.show()


# In[90]:


pca = PCA()


# In[91]:


pca.fit(segmentation_std)


# In[92]:


pca.explained_variance_ratio_


# In[95]:


plt.figure(figsize = (12,9))
plt.plot(range(1,8),pca.explained_variance_ratio_.cumsum(),marker = 'o', linestyle = '--')
plt.title('explained variance by components')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[96]:


pca = PCA(n_components = 3)
pca.fit(segmentation_std)


# In[97]:


pca.components_


# In[99]:


df_pca_comp = pd.DataFrame(data = pca.components_, 
                           columns = df_segmentation.columns.values,
                           index = ['component 1', 'component 2', 'component3'])
df_pca_comp


# In[101]:


sns.heatmap(df_pca_comp,
           vmin = -1,
           vmax = 1,
            cmap = 'RdBu',
           annot = True)
plt.yticks([0,1,2],['component1','component2','component3'],
          rotation = 45,
          fontsize = 9)


# In[102]:


scores_pca = pca.transform(segmentation_std)


# In[103]:


wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)


# In[104]:


plt.figure(figsize = (10,8))
plt.plot(range(1,11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('kmeans clustering w PCA')
plt.show()


# In[107]:


kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
kmeans_pca.fit(scores_pca)


# In[143]:


df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop = True),pd.DataFrame(scores_pca)], axis = 1)
df_segm_pca_kmeans.columns.values[-3:] = ['component1', 'component2', 'component3']
df_segm_pca_kmeans['Segment kmeans pca'] = kmeans_pca.labels_


# In[144]:


df_segm_pca_kmeans


# In[145]:


def_seg_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment kmeans pca']).mean()
def_seg_pca_kmeans_freq


# In[146]:


def_seg_pca_kmeans_freq['n obs']=df_segm_pca_kmeans[['Sex','Segment kmeans pca']].groupby(['Segment kmeans pca']).count()
def_seg_pca_kmeans_freq['prop ops'] = def_seg_pca_kmeans_freq['n obs']/def_seg_pca_kmeans_freq['n obs'].sum()
def_seg_pca_kmeans_freq=def_seg_pca_kmeans_freq.rename({0:'standard',
                                                        1:'career focused',
                                                        2:'fewer opportunities',
                                                        3:'well off'})
def_seg_pca_kmeans_freq


# In[150]:


df_segm_pca_kmeans['legend'] = df_segm_pca_kmeans['Segment kmeans pca'].map({0:'standard',
                                                                            1:'career focused',
                                                                            2:'fewer opportunities',
                                                                            3:'well off'})


# In[151]:


x_axis = df_segm_pca_kmeans['component1']
y_axis = df_segm_pca_kmeans['component2']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis, y_axis, df_segm_pca_kmeans['legend'], palette = ['g','r','c','m'])
plt.title = ('clusters by pca components')
plt.show()


# In[152]:


x_axis = df_segm_pca_kmeans['component1']
y_axis = df_segm_pca_kmeans['component3']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis, y_axis, df_segm_pca_kmeans['legend'], palette = ['g','r','c','m'])
plt.title = ('clusters by pca components')
plt.show()


# In[153]:


x_axis = df_segm_pca_kmeans['component2']
y_axis = df_segm_pca_kmeans['component3']
plt.figure(figsize = (10,8))
sns.scatterplot(x_axis, y_axis, df_segm_pca_kmeans['legend'], palette = ['g','r','c','m'])
plt.title = ('clusters by pca components')
plt.show()


# In[157]:


pickle.dump(scaler,open('scalar.pickle','wb'))
pickle.dump(pca,open('pca.pickle','wb'))
pickle.dump(kmeans_pca,open('kmeans_pca.pickle','wb'))


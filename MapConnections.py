# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:58:55 2022

@author: ajsie

Maps adapted from:
https://github.com/tuangauss/DataScienceProjects/blob/master/Python/flights_networkx.py
https://tuangauss.github.io/projects/networkx_basemap/networkx_basemap.html
"""
import os
import pandas as pd
import numpy as np
from pysankey import sankey #NB: not original, which has bugs, but Beta: https://pypi.org/project/pySankeyBeta/
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap as Basemap
import matplotlib.lines as mlines
import math

import seaborn as sns
sns.set_style("white", {'font.sans-serif':['Open Sans']})

#%% Load data
os.chdir(r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy')
dfRaw = pd.read_csv('AuthorLocations_output_Filtered.csv', encoding='utf-8')
dfRaw.loc[dfRaw['Code'].isin(['CAN', 'MEX', 'USA']), 'UN continental'] = 'North America'
dfRaw.loc[dfRaw['UN continental'] == 'Americas', 'UN continental'] = 'South America'
dfRaw['Region'] = dfRaw['UN continental']

df = dfRaw.copy()
#Split up Americas again

#We need to know how often each "connection" occurs with a source and a destination
#Take first authors as the source so we can use the geo id of all non-first authors as destinations
df[['source', 'source_country', 'source_region']] = None

for doc_id in df['doc_id'].unique():
    source = df[(df['doc_id'] == doc_id) & (df['position'] == 1)]
    if len(source) == 0:
        source_geo = None
        source_cou = None
        source_reg = None
    else:
        #Set the geoname id as the source for all non primary authors. Take first if multiple are given
        source_geo = source['geonameid'].iloc[0]
        source_cou = source['country_code3'].iloc[0]
        source_reg = source['Region'].iloc[0]
        
        # #Add the lat-long to all non-primary authors
        # #Don't use this in practice as it is easier to take lat-long info from a separate df 
        # try:
        #     lat = source['lat'].iloc[0]
        #     lon = source['lon'].iloc[0]
        #     df.loc[(df['doc_id'] == doc_id) & (df['position'] != 1), 'source_lat'] = lat
        #     df.loc[(df['doc_id'] == doc_id) & (df['position'] != 1), 'source_lon'] = lon
        # except:
        #     print(f'No lat-long for doc {doc_id}')
        
    #add the source (incl. None) to all non-primary authors
    df.loc[(df['doc_id'] == doc_id) & (df['position'] != 1), 'source'] = source_geo
    df.loc[(df['doc_id'] == doc_id) & (df['position'] != 1), 'source_country'] = source_cou
    df.loc[(df['doc_id'] == doc_id) & (df['position'] != 1), 'source_region'] = source_reg
    

#Also create a position df with the lat-long info of every node
#This is used later to create a dict
dfPosFromData = dfRaw.drop_duplicates(subset=['geonameid'])
dfPos = dfPosFromData[['geonameid', 'lat', 'lon']]

#Do the same but with a file that has country info 
dfPosCountry = pd.read_csv(r'data/CountryMappingTable.csv', encoding='utf-8')
dfPosCountry.rename(columns = {'Latitude (average)': 'lat', 
                               'Longitude (average)': 'lon'}, inplace=True)




#%% Count how often each connection occurs
def createCountGraph(df, mincount = 1, deduplicate='country', souCol = 'source', desCol = 'geonameid',
                     graph=True, #Drops collaborations where source == destination & then calculates graph
                     graphType = nx.DiGraph()): #No paralel edges but directional; nx.MultiDiGraph() allows for paralel edges
    
    if deduplicate!=False: #Remove authors with multiple institutions (hurts North-South collaboration nrs)
        df.drop_duplicates(subset=["position" , "doc_id"], inplace=True)
        if deduplicate=='country': #Also remove collaborations within same country
            df.drop_duplicates(subset=['doc_id', 'country_code3'], inplace=True)
        elif deduplicate=='countryKeepAuthors': #Multiple authors from same country are kept
            df.drop_duplicates(subset=['doc_id', 'country_code3','position'], inplace=True)
        
    #Now count
    dfCount = df.groupby([souCol,desCol]).size().reset_index()
    dfCount.rename(columns = {0:'count', desCol: 'destination'}, inplace=True)
        
    #Threshold
    dfCount = dfCount[dfCount['count'] > mincount]
    
    if graph: #Network graph for country plotting
        #Drop collaborations within same institution as they will not have a place in the graph
        #(For Sankey we need those still, so )
        dfCount = dfCount[dfCount[souCol] != dfCount['destination']]
        graph = nx.from_pandas_edgelist(dfCount, source = souCol, target = 'destination',
        		                        edge_attr = 'count',
                                        create_using = graphType
                                        )
    
        return(dfCount, graph)
    else: return(dfCount)

dfCount, graph = createCountGraph(df, souCol = 'source_country', desCol='country_code3')
    
#Plotting it is fairly useless without locations, except that there clearly is a "periphery"
#Also takes a while but you can call it with:
#nx.draw_networkx(graph, with_labels = False, node_size = 5, width=0.5)

#%% Create a sankey diagram

def plotSankey(df, sourceCol = 'source_region', desCol = 'destination', weightCol = 'count',
               colors = 'default', groups = 'continents',
               fig = None, ax=None,
               save = False):
    if fig == None:
        fig, ax = plt.subplots(figsize=(12, 7),dpi=300)
        
    if colors == 'default':
        colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6nf']
    
    if groups == 'continents':
        groups = ['Africa','Asia','Europe','North America','South America', 'Oceania']
    elif type(groups) != list:
        groups = df[sourceCol].unique()
        print(f"Using as groups: {groups}")
    pd.options.display.max_rows = 8
    colorDict = {}
    for i, c in enumerate(groups):
        colorDict[c] = colors[i]
    
    ax = sankey(df[sourceCol], df[desCol],  
             leftWeight = df[weightCol],
             aspect=100, colorDict=colorDict,
             leftLabels=list(reversed(groups)),
             rightLabels=list(reversed(groups)),
             fontsize=15, #figure_name="fruit"
             ax = ax
            )
    
    if save: 
        plt.savefig(f'Images/{save}.svg', bbox_inches="tight", facecolor='white', edgecolor='none', dp=300)
        
    return(ax)
#%% Plot sankey
fig, ax = plt.subplots(figsize=(12, 10),dpi=300)
dfc = createCountGraph(df, souCol = 'source_region', desCol = 'Region', graph=False)
ax = plotSankey(dfc, fig = fig, ax=ax)
#%% different de-duplication

#Remove single-entries as they are not collaborations
dfs = dfRaw[dfRaw.duplicated(subset = ['doc_id'], keep = False)]
#Also remove collaborations within same country
dfs.drop_duplicates(subset = ['doc_id', 'country_code3'], keep = False, inplace=True)
dfs['left_weight'] = pd.Series(dtype=float)
dfs['right_weight'] = pd.Series(dtype=float)
for doc_id in dfs['doc_id'].unique():
    tdf = dfs[dfs['doc_id'] == doc_id]
    if len(tdf[tdf['position'] == 1]) == len(tdf): #Single author paper with multiple locations listed
        dfs.drop(index=tdf[tdf['position'] == 1].index, inplace=True)
        continue
    lw = len(tdf[tdf['position'] == 1])
    if lw>0:
        dfs.loc[(dfs['doc_id'] == doc_id) & (dfs['position'] ==1), 'left_weight'] = 1/lw
        for pos in tdf['position'].unique(): #NB: within if => only add co-author weights if first author 
            if pos != 1:
                rw = len(tdf[tdf['position'] == pos])
                if rw >0:
                    dfs.loc[(dfs['doc_id'] == doc_id) & (dfs['position'] ==pos), 'right_weight'] = 1/rw
    
#%%
fig, ax = plt.subplots(figsize=(12, 7),dpi=300)    
leftDf = dfs[dfs['position'] ==1].dropna(subset = ['Region', 'left_weight'])
rightDf = dfs[dfs['position'] !=1].dropna(subset = ['Region', 'right_weight'])
#groups = ['Africa','Asia','Europe','North America','South America', 'Oceania']
groups = ['South America', 'Africa', 'Oceania', 'Asia', 'North America', 'Europe']
ax = sankey(leftDf['Region'], rightDf['Region'],  
         leftWeight = leftDf['left_weight'],
         rightWeight= rightDf['right_weight'],
         leftLabels=list(reversed(groups)),
         rightLabels=list(reversed(groups)),
         ax = ax,
         fontsize=12
        )

ax.text(-45,220, 'First author location', weight='bold', fontsize=12, ha = 'right', rotation = 90)
ax.text(195,220, 'Collaborator location', weight='bold', fontsize=12, ha = 'right', rotation = 270)
ax.set_title("Cross-Country Collaborations by Continent", weight='bold', fontsize=16)

fig.tight_layout()

fig.savefig('Images/221114_Sankey.png', dpi=300)

#%%Put graph on the world map
def setUpMap(dfPos = dfPos, fig=None, ax=None, lon=0, locCol = 'geonameid'):
    #Set up our basemap
    if fig == None:
        fig, ax = plt.subplots(figsize=(12, 7),dpi=300)
        
    m = Basemap(projection='eck4',lon_0=lon,resolution='c') #Change lon_0 to -60 to better highligh paucity of research in Africa
    
    #Create a dictionary that Basemap can use to look up the coordinates in the projection
    mx, my = m(dfPos['lon'].values, dfPos['lat'].values)
    pos = {}
    for i, elem in enumerate (dfPos[locCol]):
        pos[elem] = (mx[i], my[i])
    
    m.drawcountries(linewidth = 1.5)
    m.drawstates(linewidth = 0.2)
    m.drawcoastlines(linewidth=1.5)
    
    return(m, pos)

m, pos = setUpMap(dfPos=dfPosCountry, locCol = 'Alpha-3')


#Scale the width of the edges with the counts
counts = nx.get_edge_attributes(graph,'count').values()
baseWidth = 0.25
#widths = [baseWidth + math.log(i)*baseWidth for i in list(counts)]
widths = [i*baseWidth for i in list(counts)]
baseAlpha = 0.4
alphas = [0.01 + math.log(i)*baseAlpha for i in list(counts)]

#Change the color based on counts
#Only has an appreciable effect if they aren't crowded out by 

#color_lookup = {k:v for v, k in enumerate(sorted(set(graph.nodes())))}
norm=mpl.colors.LogNorm(vmin=dfCount['count'].min(), vmax=dfCount['count'].max())
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.magma_r)

arcs = nx.draw_networkx_edges(graph, pos = pos,
                       #edgelist = dfCount[['source', 'destination']].values.tolist(),
                       #edge_cmap = plt.cm.viridis,
                       alpha=baseAlpha, 
                       #edge_color=[mapper.to_rgba(i) 
                       #                     for i in color_lookup.values()], 
                       width = widths,
                       arrows = False)

#This doesn't work, despite the code coming from the docs: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx_edges.html
# for i, arc in enumerate(arcs):  # change alpha values of arcs
#     arc.set_alpha(alphas[i])

plt.show()


#%% Let's create a few variations looking at North-South collaboration

dfSouthSouth = pd.DataFrame(columns = df.columns)
for doc_id in df['doc_id'].unique():
    tdf = df[df['doc_id'] == doc_id]
    if len(tdf) >1: #More than one author
        if tdf['Annex I or II'].isin(['Non-Annex I']).all():
            dfSouthSouth = pd.concat([dfSouthSouth, tdf])

        
dfNorthNorth = pd.DataFrame(columns = df.columns)
for doc_id in df['doc_id'].unique():
    tdf = df[df['doc_id'] == doc_id]
    if len(tdf) >1:
        if tdf['Annex I or II'].isin(['Annex I']).all():
            dfNorthNorth = pd.concat([dfNorthNorth, tdf])

dfNorthSouth = pd.DataFrame(columns = df.columns)
for doc_id in df['doc_id'].unique():
    tdf = df[df['doc_id'] == doc_id]
    if len(tdf) >1:
        if tdf['Annex I or II'].isin(['Annex I']).any() and tdf['Annex I or II'].isin(['Non-Annex I']).any():
            dfNorthSouth = pd.concat([dfNorthSouth, tdf])

#%% Plot North-South variations together in one map

fig, ax = plt.subplots(figsize=(12, 7),dpi=300)
m, pos = setUpMap(fig = fig, ax=ax, lon = -205, dfPos = dfPosCountry, locCol = 'Alpha-3')

for df1, col in zip([dfNorthSouth, dfNorthNorth, dfSouthSouth], ['#b03700' , '#7f88ff', '#57af34']):
    df1Count, graph1 = createCountGraph(df1, souCol = 'source_country', desCol='country_code3')
    print(f"{len(df1Count)} unique and {df1Count['count'].sum()} total")
    
    counts1 = nx.get_edge_attributes(graph1,'count').values()
    widths = [i*baseWidth for i in list(counts1)]
   
    nx.draw_networkx_edges(graph1, pos = pos,
                           ax = ax,
                           edge_color= col,
                           width = widths,
                           alpha = 0.3,
                           arrows = False)

plt.show()


#%% Combine North-South and South-South

fig, ax = plt.subplots(figsize=(10, 7),dpi=300)
m, pos = setUpMap(fig = fig, ax=ax, lon = -203, dfPos = dfPosCountry, locCol = 'Alpha-3')

#Scale the width of the edges with the counts
baseWidth = 0.2
#widths = [baseWidth + math.log(i)*baseWidth for i in list(counts)]


for df1, col in zip([dfNorthNorth, pd.concat([dfSouthSouth, dfNorthSouth])], ['blue' , 'red']):
    #Remove intra-country collaborations & only count multiple authors from the same country once
    df1Count, graph1 = createCountGraph(df1, deduplicate='country', souCol = 'source_country', desCol='country_code3',
                                        #mincount = 3
                                        )
    print(f"{len(df1Count)} unique and {df1Count['count'].sum()} total")
    
    counts1 = nx.get_edge_attributes(graph1,'count').values()
    widths = [i*baseWidth for i in list(counts1)]
    
    nx.draw_networkx_edges(graph1, pos = pos,
                           ax = ax,
                           edge_color= col,
                           width = widths,
                           alpha = 0.2,
                           arrows = False)
blu_patch = mpatches.Patch(color='blue', label='Only Annex I')
red_patch = mpatches.Patch(color='red', label='At least 1 Non-Annex I')
fig.legend(handles = [blu_patch, red_patch], loc=(0.02, 0.13))
plt.title("Cross-country collaborations", size=14, weight='bold', pad=10)
fig.tight_layout()

#fig.savefig(r"Images/221104_collaborations.png", dpi=300)

plt.show()

#%% Same but with nodes showing & multi graph
#MulitGraph with low alpha is an approximation for having the alpha vary by frequency
fig, ax = plt.subplots(figsize=(10, 7),dpi=300)
m, pos = setUpMap(fig = fig, ax=ax, lon = -203, dfPos = dfPosCountry, locCol = 'Alpha-3')

for df1, col in zip([dfNorthNorth, pd.concat([dfSouthSouth, dfNorthSouth])], ['blue' , 'red']):
    #Remove intra-country collaborations & only count multiple authors from the same country once
    df1Count, graph1 = createCountGraph(df1, deduplicate='country', 
                                        souCol = 'source_country', desCol='country_code3',
                                        graphType= nx.MultiDiGraph(),
                                        mincount = 2)
    print(f"{len(df1Count)} unique and {df1Count['count'].sum()} total")
    
    counts1 = nx.get_edge_attributes(graph1,'count').values()
    widths = [i*baseWidth for i in list(counts1)]
    
    #Have to draw nodes separately to control alpha separately
    nx.draw_networkx_edges(graph1, pos = pos,
                     ax = ax,
                     edge_color= col,
                     width = 2,
                     alpha = 0.1,
                     arrows = False)
    nx.draw_networkx_nodes(graph1, pos = pos,
                     ax = ax,
                     node_size = 30,
                     node_color = 'k',
                     alpha = 0.5
                     )
    
blu_patch = mpatches.Patch(color='blue', label='Only Annex I')
red_patch = mpatches.Patch(color='red', label='At least 1 Non-Annex I')
fig.legend(handles = [blu_patch, red_patch], loc=(0.02, 0.13))
plt.title("Cross-country collaborations", size=14, weight='bold')
fig.tight_layout()

plt.show()

#%% Try majority/minority instead
dfMajSouth = pd.DataFrame(columns = df.columns)
dfMajNorth = pd.DataFrame(columns = df.columns)
for doc_id in df['doc_id'].unique():
    tdf = df[df['doc_id'] == doc_id]
    if len(tdf) > 1: #more than one author
        nan1 = tdf['Annex I or II'].isin(['Non-Annex I']).sum()
        an1 = tdf['Annex I or II'].isin(['Annex I']).sum()
        if nan1 == an1: #In equal cases, use first author as tie breaker
            if tdf['Annex I or II'][tdf['position'] == 1].isin(['Annex I']).all():
                dfMajNorth = pd.concat([dfMajNorth, tdf])
            else:
                dfMajNorth = pd.concat([dfMajNorth, tdf]) 
        elif nan1 > an1:
            dfMajSouth = pd.concat([dfMajSouth, tdf])
        else: 
           dfMajNorth = pd.concat([dfMajNorth, tdf]) 
           
fig, ax = plt.subplots(figsize=(12, 7),dpi=300)
m, pos = setUpMap(fig = fig, ax=ax, lon = -205, dfPos = dfPosCountry, locCol = 'Alpha-3')

baseWidth = 0.4
counts = nx.get_edge_attributes(graph,'count').values()
widths = [baseWidth + math.log(i)*baseWidth for i in list(counts)]

for df1, col in zip([dfMajNorth, dfMajSouth], ['red' , 'blue']):
    df1Count, graph1 = createCountGraph(df1, souCol = 'source_country', desCol='country_code3')
    print(f"{len(df1Count)} unique and {df1Count['count'].sum()} total")
    
    counts1 = nx.get_edge_attributes(graph1,'count').values()
    widths = [baseWidth + math.log(i*baseWidth) for i in list(counts1)]
   
    nx.draw_networkx_edges(graph1, pos = pos,
                           ax = ax,
                           edge_color= col,
                           width = widths,
                           alpha = 0.25,
                           arrows = False)
        


#%% TODO: More connections, stricter N/S differentiation
#The above plots the connections between Northern countries as North-South 
#even if it included even just 1 Southern author
#And it only draws connections between 1st author and all co-authors

#Starting from the dfRaw again, we could expand this







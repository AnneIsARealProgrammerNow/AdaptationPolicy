import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from adjustText import adjust_text, get_renderer, get_bboxes
import seaborn as sns
import palettable
import math

from datetime import datetime
today = datetime.today().strftime('%Y%m%d')

os.chdir(r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\Topic models\topic model 2')
savepath = r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\Topic models\topic model 2\Outputs'

#%%
##_______________________________________
##Read in data, create a workable dataframe

df = pd.read_csv(r'Outputs\DocTopicsID.csv', encoding= 'UTF-8')

#Names of topics
#topics = ['Mekong delta', 'Migration and displacement', 'Drought', 'Legislation', 'Spatial planning', 'Economic effect', 'Local community', 'Des', 'Literature review', 'Recent years', 'United states', 'Building resilience', 'Qualitative research', 'Education & New Zealand', 'Climate change', 'REDD+ and forestry', 'Local governance', 'Climate information', 'Adaptation measures', 'Difference in analysis', 'China', 'CDM projects', 'Issue', 'Challenges and opportunities', 'Stormwater management', 'Energy', 'Risk management', 'Africa', 'Cities', 'SIDS', 'System transformation', 'Emergency response', 'Assessment framework', 'Policy integration', 'Resource management', 'International agreements', 'Climate finance', 'Regional level', 'Mitigation and co-benefits', 'Institutional capacity', 'Groundwater', 'Learning', 'Forest and ecosystem services', 'Drinking water', 'Housing intervention', 'India', 'Municipality and Scandinavia', 'Marine fisheries', 'Global level', 'General adaptation', 'Food security', 'European emissions', 'Agriculture', 'Land use change', 'Stakeholder engagement', 'Effect', 'Disaster risk reduction', 'Decision makers and wildfire', 'Climate need', 'National mainstreaming', 'Flood and crop insurance', 'Public health', 'Climate variability', 'Case study', 'Governance', 'Irrigation', 'Perception', 'Extreme event', 'Indigenous communities', 'Flood risk management', 'Potential benefits', 'New approach', 'Future uncertainty', 'Coastal issues', 'Social environment', 'Green infrastructure', 'River basin', 'Action plan', 'Public participation', 'Australia', 'Best practice and Canada', 'Paper and methodolgy', 'Wetlands', 'Developing country', 'Adaptation policy', 'Nature conservation', 'Sustainable development', 'Water management', 'Vulnerability assessment and reduction', 'Public and private finance', 'Gender programme', 'Ecosystem-based adaptation', 'Population growth', 'Junk', 'Climate impact', 'Natural disaster', 'Political discourse', 'Role of organisation', 'Scenario and model', 'Asia and technology', 'Implentation and national park', 'Climate policy', 'Adaptation strategy', 'Adaptation option', 'Rural livelihood']
#Shortened names
topics = ['Sustainable', 'Role', 'Precipitation', 'Stakeholder', 'Legislation', 'Effect', 'Dam', 'Study', 'Flood', 'City', 'Challenge', 'Capacity', 'SDGs', 'decision making', 'System', 'GHG emissions', 'Strategy', 'Programme', 'Innovation', 'Land use', 'Europe', 'Social', 'Public-private', 'Framework', 'Disaster risk', 'SIDS', 'Discourse', 'Groundwater', 'Africa', 'Energy', 'Watershed', 'Collaboration', 'Urban', 'USA and fire', 'Environment', 'Coast', 'Agriculture', 'Institutional', 'Review', 'Sector', 'Municipal', 'Livelihood', 'Conservation', 'Modelling', 'Project', 'Plan', 'Barrier', 'Measurement', 'Migration', 'Heat', 'Health', 'Politics', 'Marine', 'Storm', 'Adaptation1', 'Indigenous', 'Water', 'Infrastructure', 'Mitigation', 'Insurance', 'International', 'Australia', 'Community', 'Level', 'Resilience', 'Information', 'Response', 'Region', 'Research', 'Air quality', 'Canada', 'Explore', 'Farm', 'Education', 'case study', 'Risk', 'Governance', 'Terrestrial', 'Vulnerability', 'Finance', 'Initiative', 'Practice', 'Assessment', 'Policy', 'Problems', 'Terrestrial', 'Local', 'Management', 'Awareness', 'Climate', 'Extreme event', 'Gender', 'Adaptation2', 'Perception', 'Forest', 'River', 'Fishery', 'Investment', 'Economy', 'Global', 'National', 'Action', 'Wetland', 'South America', 'Impact']
df.columns = ['id']+topics
hm = df.drop(columns='id').to_numpy()

rawResults = pd.read_csv(r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\data\withPredictions_minimal.csv',
                         encoding = 'UTF-8')

df_results = pd.merge(rawResults, df, on='id', how='right')

#%% Set which categories will be plotted here
natoCols = ['4 - 1. Nodality','4 - 3. Treasure','4 - 4. Authority', '4 - 4. Organisation']
levelCols = ['7 - International', '7 - National', '7 - Subnational ']
evidenceCols = ['9 - Ex-ante', '9 - Ex-post']

chosenCategories = evidenceCols


#%%
## Do t-SNE or load an old one

#Set to false to calculate a new t-SNE or to a pickle of existing t-SNE
Use_old_tSNE = False #"20220927_tsne_50.pickle"
perp = 50 #perplexity. 20-50 is a good range to start
if Use_old_tSNE ==False: newtSNE = f"{today}_tsne_{perp}" #Name of file


if Use_old_tSNE == False:
    tsne = TSNE(n_components = 2, perplexity=perp, early_exaggeration=20,
                n_iter = 1000, n_iter_without_progress= 500,
                init = 'pca', 
                random_state=200816, verbose = 1)
    embedding = tsne.fit_transform(hm)
    
    with open(os.path.join(savepath, "tsne", f'{newtSNE}.pickle'), 'wb') as file:
        pickle.dump(embedding, file)
        
else: 
    with open(os.path.join(savepath, "tsne", Use_old_tSNE), 'rb') as file:
        embedding = pickle.load(file)
        print(f'Using previously calculated t-SNE \n{Use_old_tSNE.split(".")[0]}')
        

# hmtest = df.head(5000).to_numpy()
# tsne_test = TSNE(n_components = 2, perplexity=50, early_exaggeration=20,
#             n_iter_without_progress= 100,
#             random_state=200816, verbose = 1)

# embedding_test = tsne_test.fit_transform(hmtest)

#%% Prep data for plotting, including scanning for clusters

#Create df for easier plotting
embedding_df = pd.DataFrame(embedding, columns=['x','y'])
embedding_df['Topic number'] = hm.argmax(axis=1) #add dominant topic 

#Add the category with the highest score
def addFields(cats, embedding_df=embedding_df, df_results=df_results, threshold=0.3):
    ResultsOverThreshold = df_results[df_results[cats].gt(threshold).any(axis=1)]
    df_results['highestField'] = df_results[cats].idxmax(axis=1)
    embedding_df['highestField'] = df_results['highestField']
    embedding_df['field_nr'] = np.nan
    #Add the fields as a number bc that's easier to plot with
    fieldnrs = {}
    for n, field in enumerate(embedding_df['highestField'].unique()):
        embedding_df.loc[embedding_df['highestField'] == field, 'field_nr'] = n
        if  type(field) == float:
            if math.isnan(field):
                continue
        
        fieldnrs[n] = field
    
    return(embedding_df, fieldnrs)

embedding_df, fieldnrs = addFields(chosenCategories)



def doDBScan(embedding, eps, min_samples, n_jobs = 8, dif_thresh = 15):
    """ Uses DB scan to create a df with positions of clusters with labels
        dif_thresh determines the distance between labels to be considered
        a duplicate and therefore for them to be removed; can be set to False
    """
    #Scan for clusters to label
    #May need to play with eps. an min_samples to get good results
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs = n_jobs).fit(embedding)
    
    
    #Now db.labels_ is a list of the points where -1 represents an outlier
    # and numbers represent the clusters => calculate mean cluster positions
    embedding_df['cluster'] = db.labels_
    cluster_df = embedding_df[embedding_df['cluster'] > 0] #all assigned to a cluster
    if len(cluster_df) == 0:
        print(f"No clusters found with eps = {eps} and samples = {min_samples}")
    else:
        cluster_df['AvgX'] = cluster_df.groupby('cluster')['x'].transform('mean')
        cluster_df['AvgY'] = cluster_df.groupby('cluster')['y'].transform('mean')
        #Get the most common topic per cluster (though they should ~all have the same)
        #TODO: get this to work again. Get out of bounds error
        cluster_df = cluster_df.groupby(['cluster']).agg(lambda x:x.value_counts().index[0])

        #Match topic numbers to topic names (generates a warning you can ignore)
        cluster_df['topic'] = [topics[n] for n in cluster_df['Topic number']]
        
        #Remove double labels close together 
        #First attempt: round x & y to nearest 'base' & remove if they are still duplicates
        #Doesn't actually work if the values happen to fall on different sides of a base multiple
        # base = 20
        # def custom_round(x, base=5):
        #     """ returns rounded nr -- i.e. here, how close numbers can be together"""
        #     return int(base * math.floor(math.floor(x)/base)) #technicall floor, not round
        # cluster_df['RoundX'] = cluster_df['AvgX'].apply(lambda x: custom_round(x, base=base))
        # cluster_df['RoundY'] = cluster_df['AvgY'].apply(lambda x: custom_round(x, base=base))
        # cluster_df.drop_duplicates(subset=['RoundX', 'RoundY', 'topic'], inplace = True)
        
        #Second attempt: calculate means per topic, remove if too small of a difference
        if dif_thresh:
            cluster_df['TopicX'] = cluster_df.groupby('topic')['AvgX'].transform('mean')
            cluster_df['TopicY'] = cluster_df.groupby('topic')['AvgY'].transform('mean')
            cluster_df['SmallDifX'] = cluster_df['TopicX'] - cluster_df['AvgX']
            cluster_df['SmallDifY'] = cluster_df['TopicY'] - cluster_df['AvgY']
            cluster_df['SmallDifX'][cluster_df['SmallDifX'] < dif_thresh] = 'yes'
            cluster_df['SmallDifY'][cluster_df['SmallDifY'] < dif_thresh] = 'yes'
            cluster_df.drop_duplicates(subset = ['topic', 'SmallDifX', 'SmallDifY'], inplace = True)
            
            #This does not work for topics that are split and/or that have many outliers
            #Since it now probably has a relatively small number, it's not expensive to loop though
            dropClusters = set()
            for topic in cluster_df['topic'].unique():
                cdf = cluster_df[cluster_df['topic'] == topic]
                cdf['AvgXY'] = cdf['AvgX'] + cdf['AvgY']
                cdf.sort_values('AvgXY', inplace=True)
                smallDif = cdf[cdf[['AvgX', 'AvgY']].diff().abs().sum(axis=1) <1.414*dif_thresh]
                dropClusters = dropClusters | set(smallDif.index[1:]) #All except first row as that obviously has no difference
        cluster_df.drop(index=dropClusters, inplace=True)
        
    return(cluster_df)

# #To get an idea of the kinds of settings that will lead to a legible plot
# eps = [0.5, 1, 1.5, 2, 3, 4, 6]
# sams = [5, 10, 15, 25, 50] #Nr to be considered cluster; default is 5 but try at least up to about half of (nDocuments/nTopics)
# res = []
# for ep in eps:
#     for sam in sams:
#         for thresh in [15, 25, 50]:
#             cdf = doDBScan(embedding, ep, sam, dif_thresh=thresh)
#             res.append([ep, sam, thresh, cdf.shape[0]])
# res = pd.DataFrame(res, columns = ["eps", "sample", "threshold", "nr"])
        

cluster_df = doDBScan(embedding, eps = 1.5, min_samples= 15, dif_thresh=30) #3 15 30 worked OK; 2 10 30 seems too much. For high perplexity values, generally lower eps (100: 0.5, 8, 10)

#%%
#simpler aternative to dbscan (doesn't work well if topics are split on the map or have large outliers)
#To add simple labels for each topic, calculate their average position
embedding_df['AvgX'] = embedding_df.groupby('Topic number')['x'].transform('mean')
embedding_df['AvgY'] = embedding_df.groupby('Topic number')['y'].transform('mean')
averages = embedding_df.drop_duplicates('Topic number')
averages = averages[['Topic number', 'AvgX', 'AvgY']]
#Match topic numbers to topic names (generates a warning you can ignore)
averages['topic'] = [topics[n] for n in averages['Topic number']]
 

#%% Default layout options
sns.set()
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Poppins']})
#cm =  'magma' 
#cm = palettable.cmocean.diverging.Delta_20_r.mpl_colormap
cm = palettable.colorbrewer.qualitative.Set3_4.mpl_colormap

def PlotLayout(fig, ax, xlabel=str, ylabel=str, 
               LegendTitle = str, LegendNames = None, #names as list of str
               title=str, subtitle=None, figsize=(10,6)):
    """ Fixes the layout of input figure and axes (matplotlib) and returns
        the fig and ax with title, subtitle and legend
    """
    
    ax.tick_params(
    reset=True,
    axis='both',          # changes apply to the x-axis
    which=u'both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    right=False,
    labelleft = False,
    length = 0) # labels along the bottom edge are off

    #Only show lines at 
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    ax.set_xticks([-50,0, 50], [])
    ax.set_yticks([-50,0,50],[])
    
    ax.margins(x=0.05, y=0.05)
    
    ax.set_xlabel(xlabel, fontsize=12, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14, weight='bold')
    if LegendNames:
        ax.legend(LegendNames, title=LegendTitle,
                  facecolor = 'white')
    #else:
        #ax.legend(title= LegendTitle, facecolor = 'white')
    
    ax.text(x=0.5, y=1.05, s=title, fontsize=16, weight='bold', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(x=0.5, y=1.01, s=subtitle, fontsize=14, alpha=0.8, ha='center', va='bottom', transform=ax.transAxes)
    
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig, ax

#%% Plot the t-SNE with colours per topic
fig = plt.figure(figsize=(9,9),dpi=150)
ax = fig.add_subplot(1,1,1)
ax = embedding_df.plot.scatter(x='x',
                             y='y',
                             c='Topic number', 
                             colormap= cm, alpha = 0.5,
                             colorbar=False,
                             ax = ax)

#Add labels 
texts = []
#for index, row in averages.iterrows():
for index, row in cluster_df.iterrows():
    ax.plot(row['AvgX'], row['AvgY'], '.')
    texts.append(ax.annotate(row['topic'], (row['AvgX'], row['AvgY']),
             bbox=dict(facecolor='white', alpha=0.6) #edgecolor=)
             ))
# adjust_text(texts, ax = ax, 
#             arrowprops=dict(arrowstyle="-", color='w', lw=1))


fig, ax = PlotLayout(fig, ax, 't-SNE 1', 't-SNE 2', 
                     LegendNames=None, LegendTitle= None, 
                     title = 'Overview of topic model',
                     subtitle = 'Dimensionality reduction using t-SNE',
                     figsize = (8,8))
#%% Plot with the field selected above
cm = 'magma'
# cm = palettable.cartocolors.qualitative.Safe_4_r.mpl_colormap

fig2 = plt.figure(figsize=(8,8),dpi=150)
ax2 = fig2.add_subplot(1,1,1)
#To add a legend, we need to loop add each, but I'm losing the overview my mind so...
def plotone(embedding_df, color, label, ax=ax2):
        ax = embedding_df.plot.scatter(x='x',
                             y='y',
                             c=color,
                             label = label,
                             s = 15,
                             colormap= cm, alpha = 0.6,
                             edgecolor = 'w',  #palettable.cartocolors.qualitative.Safe_4_r.mpl_colors,
                             linewidths = 0.35,
                             colorbar=False,
                             ax = ax)
        
        return (ax)
    
#And then I found out everything gets plotted on top of the first one now
#So shuffle the df, then plot  samples in order
shuffle_df = embedding_df.sample(frac=1)
list_dfs = np.array_split(shuffle_df, 20)
colors = ["#000004", "#721f81", "#fcfdbf", "#f1605d" ]
for part_df in list_dfs:
    for nr, color in zip(fieldnrs, colors):
        ax2 = plotone(part_df[part_df['field_nr'] == nr], color, fieldnrs[nr])
    # ax2 = plotone(part_df[part_df['field_nr'] == 0], "#000004", 'Natural')
    # ax2 = plotone(part_df[part_df['field_nr'] == 1], "#721f81", 'Agricultural')
    # ax2 = plotone(part_df[part_df['field_nr'] == 2], "#f1605d", 'Social')
    # ax2 = plotone(part_df[part_df['field_nr'] == 3], "#fcfdbf", 'Health')
    

handles, labels = ax2.get_legend_handles_labels()

lgnd = plt.legend(handles[0:len(fieldnrs)], labels[0:len(fieldnrs)], 
                  loc = 'upper left', facecolor = 'white', 
                  fontsize= 10, markerscale=2)

#Add labels 
texts = [] #Need to feed into a list to use in adjustText 

#for index, row in averages.iterrows(): #position based on average
for index, row in cluster_df.iterrows(): #position based on db-scan/locally dominant topics
    ax2.plot(row['AvgX'], row['AvgY'], 'X', color = '0.95', markersize = 4.5)
    texts.append(ax2.annotate(row['topic'], (row['AvgX'], row['AvgY']),
                 fontsize = 10,
                 zorder=50,
                 bbox=dict(facecolor='white', alpha=0.75, pad = 0.1) #edgecolor=)
             ))
    
fig2, ax2 = PlotLayout(fig2, ax2, 't-SNE 1', 't-SNE 2', 
                     LegendNames=None, LegendTitle= None, 
                     title = 'Overview of topic model using t-SNE',
                     subtitle = f'Coloured by evidence type', # - perp: {perp} sam: {sampl} eps: {eps}',
                     figsize = (12,8))
adjust_text(texts, ax = ax2,
            add_objects = [lgnd],
            #autoalign= 'x', 
            arrowprops=dict(arrowstyle="-", color='0.95', lw=2, alpha = 0.8, zorder=49))


fig2.savefig(f"{savepath}\\Figures\\TSNE_evidence.png")

#%%  
  
#plot by relevance
#Categories of topics - NB: zero indexed!
methods = [4, 5, 10, 11, 13, 14, 20, 25, 31, 32, 36, 39, 45, 47, 53, 55, 58, 61, 65, 74]
problem = [2, 3, 6, 7, 8, 12, 15, 16, 17, 18, 22, 26, 28, 30, 33, 35, 38, 40, 41, 43, 46, 48, 49, 52, 56, 57, 59, 62, 63, 64, 68, 69, 71, 72]
inbetween = [0, 9, 19, 21, 24, 27, 44, 51, 54, 66, 70, 73]
solution = [1, 23, 29, 34, 37, 42, 50, 60, 67]

cm = palettable.matplotlib.Magma_4_r.hex_colors
cats = [(methods, cm[0], 'Methods/other'),
         (problem, cm[1], 'Problem space'),
         (inbetween, cm[2], 'Mixed'),
         (solution, cm[3], 'Solution space')]

fig3 = plt.figure(figsize=(7,6),dpi=150)
ax3 = fig3.add_subplot(1,1,1)

for cat, col, lab in cats:
    ax3 = embedding_df[embedding_df['Topic number'].isin(cat)].plot.scatter(x='x',
                             y='y',
                             label = lab,
                             s = 8.5,
                             c=col,
                             alpha = 0.7,
                             edgecolor = '0.5',  #palettable.cartocolors.qualitative.Safe_4_r.mpl_colors,
                             linewidths = 0.4,
                             colorbar=False,
                             ax = ax3)
    
# #Add labels 
# texts = [] #Need to feed into a list to avoid 
# for index, row in cluster_df.iterrows():
#     ax3.plot(row['AvgX'], row['AvgY'], 'o', color = '0.9', markersize = 4.5)
#     texts.append(ax3.annotate(row['topic'], (row['AvgX'], row['AvgY']),
#                  fontsize = 8,
#              bbox=dict(facecolor='white', alpha=0.7, pad = 0.15) #edgecolor=)
#              ))

#Use alternative labels (average for each topic)
texts = []
for index, row in averages.iterrows():
    if row['topic'] != '?? junk':
        ax3.plot(row['Average_X'], row['Average_Y'], 
             'o', markersize = 4.5, color="0.9")
        texts.append(ax3.annotate(row['topic'], 
                              (row['Average_X'], row['Average_Y']),
                 fontsize = 8,
                 bbox=dict(facecolor='white', alpha=0.7, pad = 0.15) #edgecolor=)
                 ))
    
fig3, ax3 = PlotLayout(fig3, ax3, 't-SNE 1', 't-SNE 2', 
                     LegendNames=None, LegendTitle= None, 
                     title = 'Overview of topic model using t-SNE',
                     subtitle = 'Coloured by type of dominant topic',
                     figsize = (8,8))
adjust_text(texts, ax = ax3, 
            autoalign= 'x', 
            arrowprops=dict(arrowstyle="-", color='0.95', lw=2, alpha = 0.8))

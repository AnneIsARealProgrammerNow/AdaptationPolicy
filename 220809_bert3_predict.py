# -*- coding: utf-8 -*-
"""
Created on Tue May 24 21:45:06 2022

@author: ajsie
"""
import pandas as pd
import numpy as np
import os


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

import torch
import optuna

from transformers import EarlyStoppingCallback
from transformers import RobertaTokenizerFast
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed
from transformers import logging
logging.set_verbosity(40) #Errors only

import time
import pickle
import regex as re
from collections import OrderedDict


#%% Testing set to true for small df, shorter encoding, fewer epochs
testing = False
atHome =False #Change data load/save locations
randomise = True #Randomise order of input df
calculateInner = True #If inner loop is already completed, set to false to reload pickles from file. NB: classes need to be identical!
calculateOuter = True #Idem for outer loop. Set to "resume" to try to load an existing file and calculate for the missing ones
makePrediction = True
priorRelevant = 'JulyPredictions.csv' #File containing predicted-relevant from old run; set to None to re-calculate


np.random.seed(2022)
set_seed(2022)

#%% Model & data parameters
if atHome == False:
    dfDir = "ForBERT_2022-06-26.csv"#Place of csv with labelled documents
    outputDir=r'/nobackup/eeajs/out6-lvl' #Model output
    loggingDir = r'/nobackup/eeajs/log6-lvl' #Model logging
    saveDir = r'/nobackup/eeajs/save6-lvl' #saving bestParams and steps along the way as pickle files
    gsteps = 1 #No gradient accumulation
else:
    dfDir = r"C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\data\ForBERT_2022-06-26.csv"
    outputDir=r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\examples\ClimateBertExample\results2'
    loggingDir = r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\examples\ClimateBertExample\logs2'
    saveDir = r'C:\Users\ajsie\OneDrive - University of Leeds\Adaptation policy\Cluster files\Predictions2'
    gsteps = 4 #sets gradient_accumulation_steps to overcome Out of Memory issues

if testing:
    n_trails = 3
    testSize = None #use the max as it's a small dataset anyway
    maxLength = 150
    innerLoops = 2
    outerLoops = 2
else:
    n_trails = 120 #60
    testSize = 400 #Disregarded if testing is False
    maxLength = 350 #420 for full run, 350 for finding likely hyperparameter values
    innerLoops = 3 #2 is enough for relevance, but with few positive examples and some witheld for testing, higher is bettter
    outerLoops = 4 #4
    #NB: batch size below

    
early_stop = EarlyStoppingCallback(2)

def my_hp_space(trial):
    if testing:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 2),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4]),
        }
    else:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.04, 0.305, step=0.15), #Allowing no weight decay seems to lead to over-fitting for the relevance class
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8), #BERT authors recommend 2-4, but slightly higher values seem to sometimes work better without over-fitting
            #"seed": trial.suggest_int("seed", 1, 42),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]), #Generally, pick highest GPU can take & adjust learning_rate instead but BERT authors themselves value batch size too: https://github.com/google-research/bert
        }




#%% Load data
#Set seeds again to make it easier to debug
np.random.seed(2022)
set_seed(2022)

fullDf = pd.read_csv(dfDir, encoding='utf-8')

if randomise:
    fullDf = fullDf.sample(frac = 1, random_state=np.random.randint(2020)
                           ).reset_index(drop=True)
    
# df1 = fullDf.loc[fullDf["seen"] ==1].head(1000).reset_index(drop=True) #Reset index as pytorch works with sequences, not named indeces
# fullDf = fullDf.head(250)
# fullDf = pd.merge(df1, fullDf, how='outer')

print(f"Full df shape: {fullDf.shape}")

#For ease of reference, let's create one column with all text
fullDf['text'] = fullDf['title'] + " " + fullDf['content']


#%% Prepare the df -- different between multi-label and single class, so set these variables first
#The prepareDf function also adds the 'labels'column. In hindsight the dataset function would be more efficient but ¯\_(ツ)_/¯.
def setVars(classes):
    global num_classes #how many classes
    global multiLabel #boolean, if we are doing multilabel
    global y_prefix #number prefix of class, used for saving
    global inclusion #if we are doing inclusion (i.e. relevant/not relevant
    
    inclusion = False
    if classes == 'relevant' or classes == ['relevant']: 
        num_classes = 2
        multiLabel = False
        y_prefix = 1
        inclusion = True
    elif len(classes) == 1 and type(classes) == list:
        print("NB-- only one input class detected -- treating as one-hot encoded \n")
        num_classes = 2
        multiLabel = False
        y_prefix = 1.1
    elif type(classes) != list:
        raise ValueError(f'Please input your classes as a list, not as a {type(classes)}')
    else:
        num_classes = len(classes)
        multiLabel=True
        y_prefix = re.search(r'[0-9]+',classes[0]).group() #OLD:re.findall(r'[0-9]+',classes[0])

def prepareDf(fullDf, classes, numExtra = 50):
    #For the relevant column, we have either None or 1
    #Encoder wants numbers or boolean, so we convert to true/false
    if multiLabel == False:
        if type(classes) == list: classes = classes[0]
        fullDf['labels'] = None
        fullDf.loc[(fullDf[classes].isna()) & (fullDf["seen"] ==1), 'labels'] = 0
        fullDf.loc[(fullDf[classes] == 1) & (fullDf["seen"] ==1), 'labels'] = 1
        
    
    #For multi-label, we want new column containing a list with all labels
    #Set the NAs to zero for the selected classes
    if multiLabel:
        fillValues = {key: 0 for key in classes}
        fullDf.fillna(value=fillValues, inplace=True)
        #Create 'labels' column
        fullDf['labels'] =  fullDf[classes].values.astype(int).tolist()
    
    
    print(fullDf.columns)
    
    #Create the final df with only the seen (=labelled) docs
    # reduce size if testing
    if testing:
        df = fullDf.loc[fullDf["seen"] ==1].head(250).reset_index(drop=True) #Reset index as pytorch works with sequences, not named indeces
        fullDf = fullDf.head(250)
        #Cannot merge with a list, so make tuples
        if multiLabel == True:
            fullDf['labels'] = fullDf['labels'].apply(tuple)
            df['labels'] = df['labels'].apply(tuple)
        fullDf = pd.merge(df, fullDf, how='outer') #Some seen docs likely in this, so df should be slightly smaller than 500
        
        #Change back to list (though I doubt it matters)
        if multiLabel == True:
            fullDf['labels'] = fullDf['labels'].apply(list)
            df['labels'] = df['labels'].apply(list)
        
        print(f"\nNB: TESTING - full df length reduced to: {fullDf.shape[0]}")
        global reducedFullDf
        reducedFullDf = fullDf
        
    else:
        df = fullDf.loc[fullDf["seen"] ==1].reset_index(drop=True)
        
        #All documents from other projects used from other projects have nan for seen
        #Add a sample of these documents if we have any
        if inclusion == False:
            mdf = fullDf[fullDf['random'] == -1].sample(frac=1, random_state = 42)
            for col in classes:
                if mdf[mdf[col] == 1].shape[0] >= numExtra:
                    if col == classes[0]:
                        mdf_col = mdf[mdf[col] ==1].head(50)
                        mdf_add = mdf_col
                    else:
                        includedAlready = mdf_add[mdf_add[col] == 1].shape[0]
                        print(includedAlready)
                        if includedAlready <50:
                            mdf_col = mdf[(mdf[col] ==1) & (~mdf['id'].isin(mdf_add['id']))].head(numExtra-includedAlready)
                            mdf_add = pd.concat([mdf_add, mdf_col]).drop_duplicates(subset='id')
                else:
                    mdf_add = mdf[mdf[col] == 1]
                
        print(f"DF size prior to adding docs from other projects: {df.shape}")
        try: 
            df = pd.concat([df, mdf_add]).reset_index(drop=True)
        except:
            pass
        print(f"DF size after adding docs from other projects: {df.shape}")
            
    
    relevantIndex = df[df['relevant'] == 1].index
    
    print(f"Relevant documents in dataframe: {df['relevant'].value_counts()}\n")
    
    return(df, relevantIndex)

#%% Define a pytorch dataset class & evaluation metrics

class TCFDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels #torch.from_numpy(labels).long()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)
    
#Alternative for multi-label
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, labels, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation = True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            #'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            #'targets': torch.tensor(self.targets[index], dtype=torch.float)
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }


    
#By default, only loss is provided; want to use additional evaluation metrics
#For single-class, this is relatively easy
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids

    if multiLabel == False:
        preds = pred.predictions.argmax(-1)
    else: #Multilabel: not mutually exclusive so the above does not work; instead, positive if over threshold
        pred_proba = torch.sigmoid(torch.from_numpy(pred.predictions))
        preds = np.zeros(pred.predictions.shape)
        preds[np.where(pred_proba >= 0.5)] = 1
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    
    if multiLabel == False:
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
            }
    
    else: 
        precisionPerLabel, recallPerLabel, f1PerLabel, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        return { #per group as list (np array doesn't play nice with JSON)
            'accuracy': acc,
            'f1_perlabel': f1PerLabel.tolist(),
            'precision_perlabel': precisionPerLabel.tolist(),
            'recall_perlabel': recallPerLabel.tolist(),
            # Use the weighted average -- these can be set as metrics for best model
            'f1': f1,
            'precision': precision,
            'recall': recall
            }
    
def compute_objective(pred): #For ray, return only f1 as the objective
    f1 = pred['eval_f1']
    return(f1)


#%% Class weights to counteract imbalances
#Fed to the algorithm with a custom loss function in the trainer below
def classWeights(classes):
    global class_weight
    global scorer
    
    if multiLabel == False:
        if inclusion == True: y_var = 'relevant'
        else: y_var = classes[0]
        
        df.fillna(value = {y_var: 0, 'relevant':0}, inplace=True)
        cw = df[(df['random']==1) & (df[y_var]==0)].shape[0] / df[(df['random']==1) & (df[y_var]==1)].shape[0] #"How many more times do we have a random non-relevant sample for every relevant sample?"
        class_weight={0:1, 1:cw} #NB: the loss function uses the order (dict.values)
        scorer = "f1"
    elif classes == ['9 - Ex-ante', '9 - Ex-post']:
        print("NB - Running study type with custom class weights")
        for i, t in enumerate(classes):
            df.fillna(value = {t: 0, 'relevant':0}, inplace=True)
        class_weight = {0:5, 1:1} 
    else:
        scorer = "f1_macro"
        class_weight = {}
        for i, t in enumerate(classes):
            df.fillna(value = {t: 0, 'relevant':0}, inplace=True)
            #weight based on random samples: nr of irrelevant samples per relevant sample
            #cw = df[(df['random']==1) & (df[t]==0) & (df['relevant'] ==1)].shape[0] / df[(df['random']==1) & (df[t]==1) & (df['relevant'] ==1)].shape[0]
            cw = df[(df['random']==1) & (df[t]==0)].shape[0] / df[(df['random']==1) & (df[t]==1)].shape[0]
            class_weight[i] = cw
            
        #normalise
        lowestWeight = class_weight[min(class_weight, key = class_weight.get)]
        for k, v in zip(class_weight, class_weight.values()): 
             class_weight[k] = v/lowestWeight
             
    print(f"{classes} weights set at {class_weight}")
            
    return(class_weight)


#Weighted loss adapted from Max
class CWTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        device = torch.device('cuda:0')
        labels = labels.to(device)
        
        outputs = model(**inputs)
        logits = outputs.logits#[:,0]
        try:
            logits.get_device()
        except:
            logits = logits.to(device)
        if self.class_weight is not None:
            cw = torch.tensor(list(self.class_weight.values()))
            try:
                cw.get_device()
            except:
                cw = cw.to(device)
        else:
            cw = None
        
        # #Added - unsqueeze single class to make dimensions match - only needed if num_classes ==1, which actually sets the trainer to regression mode anyway
        # try: 
        #     if labels.shape[1] >= 1:
        #         pass
        # except IndexError:
        #     labels = labels.unsqueeze(1)
            
        if multiLabel == True:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = cw.to(device), reduction='none')
            loss = loss_fct(logits.to(device),labels.float().to(device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=cw.to(device))
            loss = loss_fct(logits.to(device),labels.long().view(-1).to(device))
            
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss

#%% Get train and test set indices -- used to create subsets

def splitSetsWitholdRandom(df, innerSplits = innerLoops, outerSplits = outerLoops, testSize = None):
    #Split the data: create a dict with { train, [train, evaluation]} for nested cross-validation
    #This will use an inner loop for hyperparamter search and an outer loop for testing/calculating scores
    #Train and eval are a list with length innerSplits
    #The test sets are based only on randomly selected documents
    #For categories, we will only predict on the predicted to be relevant documents anyway
    #Therefore, a more representative random set are those marked as relevant
    
    if testSize == None:
        testSize = np.floor((1/(outerSplits+1))*len(df)) #If none provided, make outer loop equally big to inner loop
    
    if inclusion:
        randomIndex = df[df['random'] == 1].index
    else:
        randomIndex = df[df['relevant'] == 1].index
    
    #Test if we have enoug random docs
    if len(randomIndex) <= testSize:
        print(f"Cannot create a test set for the outer loop with size {testSize}")
        testSize = len(randomIndex)-1
        print(f"Using test size of {testSize} instead")
    
    nontest, testDf = train_test_split(df.loc[randomIndex], test_size=int(testSize), shuffle=False) #Order already randomised at load
    
    remainderDf = df.drop(testDf.index, errors='ignore')
    
    setIndices = {}
    
    kfi = KFold(n_splits=innerSplits)
    kfo = KFold(n_splits=outerSplits)
    for i, (nonTest, test) in enumerate(kfo.split(testDf)):
        #test is now the test for outer, remainder needs to be further split up
        setIndices[f'test_{i}'] = testDf.iloc[test].index.values
        
        innerDf = testDf.iloc[nonTest].append(remainderDf)
        for j, (train, evaluation) in enumerate(kfi.split(innerDf)):
            setIndices[f'train_{i}_{j}'] = innerDf.iloc[train].index.values
            setIndices[f'eval_{i}_{j}'] = innerDf.iloc[evaluation].index.values
            
    #Save the numbers too, in case they weren't set
    setIndices['innerSplits'] = innerSplits
    setIndices['outerSplits'] = outerSplits
    setIndices['testSize'] = testSize
            
    return(setIndices)



#%% Tokenize and use above dataset functions to create a torch dataset 
def tokenizeAndDataset(df, multilabel):
    global tokenizer
    
    if testing:
        max_length=150
    else:
        max_length=maxLength #420 - Only affects ~2.2% of documents, a third of which are over the max of 512 anyway, and reduces computation times

    tokenizer = RobertaTokenizerFast.from_pretrained('climatebert/distilroberta-base-climate-f')#'distilroberta-base')
    
    labels = np.array(df["labels"].values.tolist(), dtype = 'float16')
    
    if multiLabel == False:
        texts = df['text'].to_numpy(dtype = 'str')
        encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        dataset = TCFDDataset(encodings, labels)
    else:
        dataset = MultiLabelDataset(df, tokenizer, labels, max_length)
         
    return(dataset)



#%% Trainer parameters
#Adapted from https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
#Using hyperopt and optuna
#This keeps running out of memory on my home pc 

def model_init():
    if multiLabel==False:
        return RobertaForSequenceClassification.from_pretrained('climatebert/distilroberta-base-climate-f', num_labels=2)
    else:
        #Sets different loss function -- superfluous as we define our own loss function
        return RobertaForSequenceClassification.from_pretrained('climatebert/distilroberta-base-climate-f', num_labels=num_classes,
                                                                problem_type="multi_label_classification")
    
def cleanUp():
    try: del trainer
    except: pass
    try: del best_run
    except: pass
    try: del classPrediction
    except: pass
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


#%% Now run all
if __name__ == "__main__":
    classesGroup = [
                    #['relevant'],    #This is the only mutually exclusive one!
                    #['2 - Adaptation', '2 - Mitigation', '2 - Other'], 
                    #['4 - 1. Nodality', '4 - 3. Treasure', '4 - 4. Authority', '4 - 4. Organisation'],
                    ['7 - International', '7 - National', '7 - Subnational '],
                    #['8 - Coastal','8 - Rivers', '8 - Terrestrial', '8 - Human'],
                    ['9 - Ex-ante', '9 - Ex-post']
                    ]
    
    #We will save all the test scores in a variable called testDicts
    if calculateOuter == "resume": 
        try: #Load previous if we are resuming
            with open(os.path.join(saveDir, "testDicts.pickle"), 'rb') as f:
                testDicts = pickle.load(f)
        except:
            print("Could not resume because no testDicts file could be found. Starting with an empty one.")
            testDicts = {}
    else:
        testDicts = {} #Start with a clean file also if set to True; will overwrite existing testDicts on file
    
    for classes in classesGroup: 
        
        print(f"\n\nStarting with classes: {classes}\n\n")
        
        setVars(classes)
        df, relevantIndex = prepareDf(fullDf, classes)
        setsIndexDict = splitSetsWitholdRandom(df, testSize= testSize)
        relevantDataset = tokenizeAndDataset(df, multilabel = multiLabel)
        
        
        
        #Set up the outer loop
        for i in range(setsIndexDict['outerSplits']):
            testIndex = setsIndexDict[f'test_{i}']
            testSubset = torch.utils.data.Subset(relevantDataset, testIndex)
            #to get a full index of all not in the test in this fold, combine any of the train+eval pairs
            nonTestIndex = np.sort(np.append(setsIndexDict[f'train_{i}_0'], setsIndexDict[f'eval_{i}_0']))
            nonTestSubset = torch.utils.data.Subset(relevantDataset, nonTestIndex)
            
            #Inner loop
            for j in range(setsIndexDict['innerSplits']):
                
                trainIndex = setsIndexDict[f'train_{i}_{j}']
                evalIndex = setsIndexDict[f'eval_{i}_{j}']
                
                trainSubset = torch.utils.data.Subset(relevantDataset, trainIndex)
                evalSubset = torch.utils.data.Subset(relevantDataset, evalIndex)
                
                training_args = TrainingArguments(
                                    output_dir= outputDir,
                                    overwrite_output_dir=True,
                                    #num_train_epochs=nepochs,        # total number of training epochs - now in optimisation params
                                    #per_device_train_batch_size=6,  # batch size per device during training - now in optimisation params
                                    per_device_eval_batch_size=64,   # batch size for evaluation 
                                    warmup_steps=250,                # number of warmup steps for learning rate scheduler
                                    #weight_decay=0.01,               # strength of weight decay - now in optimisation params
                                    logging_dir= loggingDir,            # directory for storing logs
                                    logging_steps=5,
                                    #fp16=True,                       # enable mixed precision training if supported by GPU (especially good if you have tensor cores)
                                    gradient_accumulation_steps=gsteps,
                                    load_best_model_at_end=True,
                                    metric_for_best_model= 'f1',
                                    evaluation_strategy='steps',
                                    eval_steps = 350, #Evaluate a bit more often
                                    save_strategy='steps',
                                    save_steps= 1400,
                                    seed=2022
                                    )
                
                
                trainer = CWTrainer(
                                    model_init=model_init,
                                    args=training_args,
                                    train_dataset= trainSubset,
                                    eval_dataset= evalSubset,
                                    tokenizer=tokenizer,
                                    compute_metrics=compute_metrics,
                                    callbacks=[early_stop]
                                    )
                
                class_weight = classWeights(classes)
                trainer.class_weight = class_weight 
                
                if calculateInner == True:
                    if classes[0] == '9 - Ex-ante' and i == 3: #Temporary insertion to resume here
                        t0 = time.time()
                        
                        study = optuna.create_study()
            
                        best_run = trainer.hyperparameter_search(n_trials=n_trails, 
                                                                 compute_objective=compute_objective, direction="maximize",
                                                                 #default sampler: TPE
                                                                 pruner = optuna.pruners.HyperbandPruner(),
                                                                 hp_space=my_hp_space)

                        if testing == False:
                            with open(os.path.join(saveDir, f'bestRun_cat{y_prefix}_run{i}_{j}.pickle'), 'wb') as f:
                             	pickle.dump(best_run, f)
                         
                        t1=time.time()
                        print("\n\n__________________________________")
                        print(f"\n Parameter search for category {y_prefix} - outer loop {i} - inner loop {j} completed in {(t1-t0)//60} minutes and {(t1-t0) % 60} seconds")
                        print(f"F1: {best_run[1]} \nParams: {best_run[2]}")
                    else:
                        print(f"\nNB: loading prior inner loop results from disk. Now at category {y_prefix} - outer loop {i} - inner loop {j}\n")
                        #Note that new results would also be written to disk and loaded below
                    cleanUp()
                
                
                
            #Get best of the inner parameters
            innerDict = OrderedDict() #Python >3.6 maintains dict order but still
            for file in os.listdir(saveDir):
                if file.startswith(f"bestRun_cat{y_prefix}_run{i}"):
                    with open(os.path.join(saveDir, file), "rb") as f:
                        innerDict[file.split(".")[0]] = pickle.load(f)
            runOutcomes = [innerDict[n][1] for n in innerDict]
            try: 
                runNr = runOutcomes.index(max(runOutcomes))
            except ValueError:
                runNr = 0
            
            testRunParams = innerDict[f"bestRun_cat{y_prefix}_run{i}_{runNr}"].hyperparameters
            
            #To enable testing on my small pc, as we don't care about the actual results then anyway
            if atHome == True:
                testRunParams['per_device_train_batch_size'] = 4
                testRunParams['per_device_eval_batch_size'] = 4
            
            #train with these best parameters on all not in test set
            trainer = CWTrainer(
                                    model_init=model_init,
                                    args=training_args,
                                    train_dataset= nonTestSubset,
                                    eval_dataset= testSubset,
                                    tokenizer=tokenizer,
                                    compute_metrics=compute_metrics,
                                    callbacks=[early_stop]
                                    )
            
            for n, v in testRunParams.items():
                setattr(trainer.args, n, v) 
            
            class_weight = classWeights(classes)
            trainer.class_weight = class_weight
            if calculateOuter:               
                if f'cat{y_prefix}_run{i}' not in testDicts.keys(): #Will start with empty dict if calculateOuter set to False
                    trainer.train()
                    testScores = trainer.evaluate() #outputs a dict
                    testScores['Parameters'] = testRunParams
                    testDicts[f'cat{y_prefix}_run{i}'] = testScores
                
                if testing == False:
                        with open(os.path.join(saveDir, f"testDicts.pickle"), 'wb') as f:
                                pickle.dump(testDicts, f)

            
            
            cleanUp()
            
          
    #We now should have outerSplits times the scores for all the categories saved to one dict
    #Let's save the final dict (or load if calculated before) and then start predicting
    if calculateOuter == False:
        print("\nNB: loading outer results from disk")
        with open(os.path.join(saveDir, "testDicts.pickle"), 'rb') as f:
            testDicts = pickle.load(f)
    
    print("\n------\nNested hyper-parameter search completed.\nValues on test set:")
    for k, d in zip(testDicts, testDicts.values()):
        print(k)
        print(f'F1: {round(d["eval_f1"], 2)} -- Precision: {round(d["eval_precision"], 2)} -- Recall: {round(d["eval_recall"], 2)}')
        print()
        
#%% Moving on to predictions    
 
    #For all classes, we predict on the predicted relevant docs only => predict relevance first
    #That prediction is on the unseen doc using a trainer trained on the seen docs
    classes = ["relevant"]
    setVars("relevant") #sets y_prefix to 1, among others
    df, relevantIndex = prepareDf(fullDf, "relevant")
    if testing: fullDf = reducedFullDf
    fullDataset = tokenizeAndDataset(fullDf, multilabel = False)
    seenIndex = fullDf[fullDf['seen'] == 1].index.values
    seenSubset = torch.utils.data.Subset(fullDataset, seenIndex)
    unseenSubset = torch.utils.data.Subset(fullDataset, [i for i in fullDf.index.values if i not in seenIndex])
    #trainer.train() wants an evaluate set. Let's just use all the combined test sets as these were least involved in hyperparameter selection
    allTestIndices =  np.concatenate([v for k,v in setsIndexDict.items() if k.startswith('test_')])
    allTestSubset = torch.utils.data.Subset(fullDataset, allTestIndices)
    #Select best hyperparameters for the relevance trainer 
    runOutcomes = [ v['eval_f1'] for k,v in testDicts.items() if k.startswith('cat1')]
    try: 
        runNr = runOutcomes.index(max(runOutcomes))
    except ValueError:
        runNr = 0
        print(f"\nNB!Run outcomes on the testset for inclusion do not have a clear max F1.\n Outcomes: {runOutcomes}\n")
    fullRunParams = testDicts[f"cat1_run{runNr}"]["Parameters"]
    
    cleanUp()
    
    #train with these best parameters on the seen subset
    trainer = CWTrainer(
                            model_init=model_init,
                            args=training_args,
                            train_dataset= seenSubset, #specify eval dataset below
                            eval_dataset=allTestSubset,
                            tokenizer=tokenizer,
                            compute_metrics=compute_metrics,
                            callbacks=[early_stop]
                            )
    
    for n, v in fullRunParams.items():
        setattr(trainer.args, n, v) 
    
    if priorRelevant == None:
        class_weight = classWeights(["relevant"])
        trainer.class_weight = class_weight
        trainer.train()
        relevantPrediction  = trainer.predict(test_dataset = unseenSubset) 
        
        print("\n-----\nRelevance prediction complete")
        
        #Dump the df, unseenSubset (which has the indices) and predictions -- this is a biiiig file
        toSave = [fullDf, unseenSubset.indices, relevantPrediction]
        if testing == False:
            with open(os.path.join(saveDir, "relevance.pickle"), 'wb') as f:
                pickle.dump(toSave, f)
                
        cleanUp()
        
        relevance =  pd.DataFrame(columns = ['exclude_score', 'include_score'],
                            index = unseenSubset.indices,
                            data = softmax(relevantPrediction.predictions, axis=1)) #Softmax as CrossEntropyLoss is used for binary prediction
        relevance['relevant_high'] = 0
        relevance.loc[relevance['include_score'] > relevance['exclude_score'], 'relevant_high' ] = 1
        
        #Merge it onto the main df
        fullDf = fullDf.merge(relevance, how="outer", left_index = True, right_index=True)
    else:
        print("Loading relevance from file")
        fullDf = pd.read_csv(priorRelevant, encoding='utf-8')
        
    
    #Now create a new dataset for predicted relevant predict the remaining classes on for those
    #If we have a low number of predicted relevant documents, go for threshold approach
    #Threshold is pretty low but we record both and can always filter later
    nHighestRelevant = fullDf['relevant_high'].sum()
    nThreshRelevant = fullDf[fullDf['include_score'] >=.35].shape[0]    
    print(f"Number of documents where relevance score is highest: {fullDf['relevant_high'].sum()}")
    print(f"Number of documents where relevance score is above threshold of .35: {nThreshRelevant}") 
    #Because I'm curious
    print("Average non-BERT predictions values for those sets:")
    print(f"Highest score: {fullDf['0 - relevance - prediction'][fullDf['relevant_high'] ==1].mean()}")
    print(f"Threshold: {fullDf['0 - relevance - prediction'][fullDf['include_score'] >=.35].mean()}")
    if nHighestRelevant >= nThreshRelevant:
        predictedIndex = fullDf[fullDf['relevant_high'] ==1].index.values
        print("Predicting categories on 'highest score' subset")
    else:
        predictedIndex = fullDf[fullDf['include_score'] >=.35].index.values
        print("Predicting categories on 'above threshold' subset")
    
    if testing == False:
        fullDf.to_csv(os.path.join(saveDir, "finalDf.csv"), encoding = "UTF-8")

                 

#%% Now predict classes
    if makePrediction:    
        #Remove "relevant" from classes (if it was there) so we don't calculate it again
        classesGroup  = [g for g in classesGroup if g not in ("relevant", ["relevant"])]  
        for classes in classesGroup:
            setVars(classes)
            print(f"Started prediction on class: {y_prefix}") 
            df, relevantIndex = prepareDf(fullDf, classes)
            
            #Need to create new datasets too to make sure the labels match
            fullDataset = tokenizeAndDataset(fullDf, multilabel = multiLabel)
            predictedSubset = torch.utils.data.Subset(fullDataset, predictedIndex)
            seenSubset = torch.utils.data.Subset(fullDataset, seenIndex)
            allTestSubset = torch.utils.data.Subset(fullDataset, allTestIndices)
            
            #Select the best run on the test sets
            runOutcomes = [ v['eval_f1'] for k,v in testDicts.items() if k.startswith(f'cat{y_prefix}')]
            try: 
                runNr = runOutcomes.index(max(runOutcomes))
            except ValueError:
                runNr = 0
                print(f"\nNB!Run outcomes on the testset for {y_prefix} do not have a clear max F1. Using first run.\n Outcomes: {runOutcomes}\n")
            classRunParams = testDicts[f"cat{y_prefix}_run{runNr}"]["Parameters"]
            
            #Train with these new parameters
            trainer = CWTrainer(
                                model_init=model_init,
                                args=training_args,
                                train_dataset= seenSubset,
                                eval_dataset=allTestSubset,
                                tokenizer=tokenizer,
                                compute_metrics=compute_metrics,
                                callbacks=[early_stop]
                                )
        
            for n, v in classRunParams.items():
                setattr(trainer.args, n, v) 
            
            class_weight = classWeights(classes)
            trainer.class_weight = class_weight
            trainer.train()
            classPrediction  = trainer.predict(test_dataset = predictedSubset)
            
            if multiLabel == True:
                y_pred = torch.sigmoid(torch.from_numpy(classPrediction.predictions))
            else:
                y_pred = softmax(classPrediction.predictions, axis=1)
            
            classPredictionDf =  pd.DataFrame(index = predictedSubset.indices,
                            data = y_pred, 
                            columns = [f"{c}_pred" for c in classes])
            
            fullDf = fullDf.merge(classPredictionDf, how ="outer", left_index = True, right_index=True)
        
            #Save at every step -- just overwrite each run b/c if it gets to here, we're basically done
            if testing == False:
                fullDf.to_csv(os.path.join(saveDir, "finalDf.csv"), encoding = "UTF-8")
                
            cleanUp()
                 
    
        
        
        

































library("quanteda")
library("stm")
library(tidyverse)
library(igraph)
library(splines)
library(tm)
library(RColorBrewer)
library(stminsights)
library(ggplot2)

#___
#GET THE DATA
setwd("C:\\Users\\ajsie\\OneDrive - University of Leeds\\Adaptation policy\\Topic models\\topic model 2")
data <- read.csv("ForSTM_new.csv", header = TRUE, stringsAsFactors = FALSE)

#Lowercase, remove copyright texts (or most of them anyway -- specific keywords removed later)
data$text <- tolower(data$text)
data <- data %>% separate(text, c("text","copyright"), sep = "\\(c\\)\\s*\\d+", extra="merge", remove = TRUE) #removes (c) followed by numbers - this covers most copyright messages
data$text<-gsub("all rights reserved"," ",as.character(data$text)) 

#Since we're using publication year as a co-variate, 0-entries need to be removed
before = length(data) #8691
data = data[data$PY >1,]
#data = data[data$PY <2022,]
cat(length(data), " documents from ", before, " prior to removing missing years") #8691

#Add Pre- and Post-Paris agreement as categorical
data[ , 'Paris'] <- NA
data[data$PY >2016,]$Paris <- "Post-Paris"
data[data$PY <=2016,]$Paris <- "Pre-Paris"

#____
#CREATE A DFM
#Using quanteda as STM does not allow bigrams natively 
#First, simple one-word tokens, then combine with frequently occurring bi-grams/skipgrams
corpus <- corpus(data, text_field = "text")
docnames(corpus) <- data$id
customStopwords = c("copyright", "eslevier", "wiley", "ltd", "taylor", "francis", "llc", "GmbH", "inc")
toks <- tokens(corpus, remove_punct = TRUE, remove_symbols = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(pattern = stopwords("en"), #remove stopwords
                      min_nchar = 3) %>% #remove short words
  tokens_remove(pattern = phrase(customStopwords), valuetype = 'fixed') %>% #Custom stopwords
  tokens_wordstem(language = "en") #Using snowball stemmer
dfm_single = dfm(toks)%>%
  dfm_trim(min_termfreq = 10, termfreq_type = "count", max_docfreq=0.95, docfreq_type = "prop")

#We only want to keep frequent bigrams as otherwise it's mostly just duplicating information
toks_bigram <- tokens_ngrams(toks, n = 2, skip = 1:2)
dfm_bigram <- dfm(toks_bigram) %>%
  dfm_trim(min_termfreq = 100, termfreq_type = "count", max_docfreq=0.95, docfreq_type = "prop")

#Combine
dfm_both = cbind(dfm_bigram, dfm_single)
#To show most-frequent bigrams:
topfeatures(dfm_both, 15)

#Convert to STM format
dfm_stm <- quanteda::convert(dfm_both, to = c("stm"),docvars=docvars(dfm_both))
#output will have object meta, documents, and vocab
dfm_stm$meta$PY <- as.numeric(dfm_stm$meta$PY)
docs <- dfm_stm$documents
vocab <- dfm_stm$vocab
meta <-dfm_stm$meta


#_______
#START MODELLING

#I ran k = 50-75-85-100-125 single models to get a first idea

#To run a single stm
model110 <- stm(docs, vocab, data=meta, 
              K=110,
              prevalence =~ UN.continental + bs(as.numeric(PY), df=10, degree =2),
              init.type = "LDA", seed=42, max.em.its=250,
              control = list(alpha = 0.5, #Lower than default of 50/k to allow documents to have more topics -> makes small topics more likely to show up
                             eta = 0.1)) # Higher than default of 0.01 to create topics composed of more words -> more clarity for complex topics


#Based on the content of the early results, the I re-ran manyTopics 100-110-120 to zoom in on this range

#Run multiple for a range of k-s
storage <- manyTopics(docs, vocab, data=meta, 
             K=c(100, 110, 120), 
             runs = 15, # ~15% of runs are ran until convergion, the others are discarded early on
             prevalence =~ UN.continental + bs(as.numeric(PY), df=10, degree =2),
             init.type = "LDA", seed=42, max.em.its=250,
             control = list(alpha = 0.5, #Lower than default of 50/k to allow documents to have more topics -> makes small topics more likely to show up
                            eta = 0.1)) # Higher than default of 0.01 to create topics composed of more words -> more clarity for complex topics

#The output is a bit confusing, so let's collect the details we need
modelDetails = data.frame(
  K = numeric(),
  coherence = numeric(),
  exclusivity = numeric()
)

for(i in 1:length(storage)){
  modi = data.frame(
    coherence = storage$semcoh[[i]],
    exclusivity = storage$exclusivity[[i]]
  )
  K = storage$out[[i]]$settings$dim$K
  modi$K = K
  modelDetails <- rbind(modelDetails, modi)
  cat(K, 
        " - Exclusiviy:", mean(modi$exclusivity),
        " - Semantic Coherence:", mean(modi$coherence),
      "\n"
        )
}

ggplot(modelDetails, aes(exclusivity, coherence)) +
  geom_point(aes(color=factor(K)), size = 4, alpha=0.75) +
  scale_color_brewer(palette = "Paired")
model <- storage$out[[3]]

#Scores are similar, but based on top words, optimum seems to be between 100 and 110, let's optimise 105
#Below lets you compare a number of models side by side. Will evaluate after a few (2 I think?) runs and keep ones that converge quickest 
ModelSelect <- selectModel(docs, vocab, data=meta, 
                           K=110,
                           runs=60, 
                           prevalence =~ UN.continental + bs(as.numeric(PY), df=10, degree =2),
                           init.type = "LDA", seed=420, max.em.its=250,
                           control = list(alpha = 0.5, #Lower than default of 50/k to allow documents to have more topics -> makes small topics more likely to show up
                                          eta = 0.1))

#plot the different models that make the cut along exclusivity and semantic coherence of their topics
plotModels(ModelSelect)

#pick the one that looks best and give it the name model
model<-ModelSelect$runout[[11]]

#______
#INTERPRETATION OF MODEL

#see proportion of topic in corpus with top n words
dev.new(width=5, height=12, cex=0.5) #Open in new window
plot.STM(model, type="summary", n = 3, xlim=c(0,.12))

#Print top words for selected topics (or all here)
labelTopics(model, c(1:model$settings$dim$K))

#wordcloud for a topic (requires wordcloud package from CRAN)
cloud(model, topic=10)

#Plot the closeness (document overlap) of the topics using iGraph
mod.out.corr<-topicCorr(ModelPrevFit)
plot.topicCorr(mod.out.corr)

#plot comparisson of frequent terms between selected topics. I don't know what the y-axis specifies here
plot.STM(ModelPrevFit,type="perspectives", topics=c(2, 4))

#______
#INFLUENCE OF META DATA

##See CORRELATIONS BTWN METADATA & TOPIC PREVALANCE in documents
## We are estimating the expected proportion of a document that belongs to a topic as a function of a covariate
#First though we need to tell R to convert the variables in the meta file from string to categorical
meta[meta$UN.continental== "Antartica",]$UN.continental <- 0 #Only have 4, so no valid results anyway
meta$UN.continental <-as.factor(meta$UN.continental)
meta$Annex.I.or.II <-as.factor(meta$Annex.I.or.II)
meta$PY<-as.numeric(meta$PY)

#since we're preparing these covariates by estimating their effects we call these estimated effects 'prep'
#we're estimating Effects across all topics. We're using author country and normalized publishing year (PY), using the topic model poliblogPrevFit. 
#The meta data file we call meta. We are telling it to generate the model while accounting for all possible uncertainty. Note: when estimating effects of one covariate, others are held at their mean
prep_annex <- estimateEffect(1:model$settings$dim$K ~ Annex.I.or.II, 
                       model,meta=meta, uncertainty = "Global")
#Annex status and region have massive overlap, so adding them as one formula leads to singular covariate matrix
prep_region <- estimateEffect(1:model$settings$dim$K ~ UN.continental,
                             model,meta=meta, uncertainty = "Global")
#Continuous
prep_cont <- estimateEffect(1:model$settings$dim$K ~ bs(as.numeric(PY), df=10, degree =2),
                              model,meta=meta, uncertainty = "Global")

#Can combine region + year in one without getting an error
#=> controls for interaction
prep <- estimateEffect(1:model$settings$dim$K ~ UN.continental * bs(as.numeric(PY), df = 10, degree = 2),
                        model110,meta=meta, uncertainty = "Global")

topicnames <- c('Sustainable', 'Role', 'Precipitation', 'Stakeholder', 'Legislation', 'Effect', 'Dam', 'Study', 'Flood', 'City', 'Challenge', 'Capacity', 'SDGs', 'decision making', 'System', 'GHG emissions', 'Strategy', 'Programme', 'Innovation', 'Land use', 'Europe', 'Social', 'Public-private', 'Framework', 'Disaster risk', 'SIDS', 'Discourse', 'Groundwater', 'Africa', 'Energy', 'Watershed', 'Collaboration', 'Urban', 'USA and fire', 'Environment', 'Coast', 'Agriculture', 'Institutional', 'Review', 'Sector', 'Municipal', 'Livelihood', 'Conservation', 'Modelling', 'Project', 'Plan', 'Barrier', 'Measurement', 'Migration', 'Heat', 'Health', 'Politics', 'Marine', 'Storm', 'Adaptation1', 'Indigenous', 'Water', 'Infrastructure', 'Mitigation', 'Insurance', 'International', 'Australia', 'Community', 'Level', 'Resilience', 'Information', 'Response', 'Region', 'Research', 'Air quality', 'Canada', 'Explore', 'Farm', 'Education', 'case study', 'Risk', 'Governance', 'Terrestrial', 'Vulnerability', 'Finance', 'Initiative', 'Practice', 'Assessment', 'Policy', 'Problems', 'Terrestrial', 'Local', 'Management', 'Awareness', 'Climate', 'Extreme event', 'Gender', 'Adaptation2', 'Perception', 'Forest', 'River', 'Fishery', 'Investment', 'Economy', 'Global', 'National', 'Action', 'Wetland', 'South America', 'Impact')
summary(prep_annex)
summary(prep_cont)

#Most interesting sub-set
sel = c(10, 12, 13, 21, 26, 47)
selLabels = c("Urban", "Capacity", "SDGs", "Europe", "SIDS", "Barrier") #c("Migration", "Legislation", "Climate information", "CDM", "Cities", "SIDS", "Policy integration", "International Agreements", "Finance", "Stakeholder", "Health", "Action plan", "Plan", "Strategy", "Option")


#See how prevalence of selected topics differs across values of a categorical covariate -- in this case country_predicted
#Would be good if I can plot multiple in one go, (using add= True) but I can't specify the colour of the second plot
par(bty="n",col="darkslateblue",lwd=2)
plot.estimateEffect(prep_annex, add = F, covariate = "Annex.I.or.II", topics = sel,
                    model=model, method="difference", 
                    cov.value1="Non-Annex I",cov.value2="Annex I",
                    xlab="More Annex I ... More Non-Annex I",
                    main="Effect of country in abstract and country of institution (Annex I or Non-Annex I) on topics",
                    xlim=c(-.015,.015), labeltype = "custom", verbose.labels = F,
                    #topic.names = topicnames,
                    custom.labels = selLabels)

plot

#For more than 1 values (the above substracts the first from the second; this one just plots a point for each possible covariate)
plot.estimateEffect(prep, add = F, covariate = "UN.continental", topics = sel,
                    model=model, method="pointestimate", 
                    #xlab="More Annex I ... More Non-Annex I",
                    main="Effect of country in abstract (UN continental) on topics",
                    verbose.labels = T,)
#labeltype = "number", custom.labels = c(1:75))

#Same for continuous covariate -- in this case the spline of published year. Need to play with lay-out. 
plot.estimateEffect(prep_cont, "PY", method="continuous", topics=sel, model=model,
                    xaxt="n", xlab="Time (publication year)", ci.level = 0,#0.95,
                    ylim = c(-0.01, 0.05), #currently still plots conf. interval under x axis
                    main= "Change in prevalence for topics (0.90 conf.)",
                    linecol= brewer.pal(n = 6, name = "Dark2"),
                    printlegend = F,labeltype = "custom", cex = 0.75,
                    custom.labels = selNames)
#Adding years to the axis and then a legend NB: order/label colour should match with above
yearseq <- seq(from=as.Date("1988-01-01"),
               to=as.Date("2019-01-01"), by="year")
axis(1, at = seq(1988, 2022, by = 2), las=2)
legend("topright", legend = selNames, 
       fill= brewer.pal(n = length(selNames), name = "Dark2"), cex=0.7)

#______
#NICER PLOTTING

#trying it with the stminsights package to plot nicer looking graphs in ggplot2. First, get effects
effects <- get_effects(estimates = prep_cont,variable ='PY',type ='continuous')
effects_region <- get_effects(estimates = prep_region, variable = 'UN.continental', type = 'pointestimate')

#set colour palette
pal <- brewer.pal(n = 6, name = "Dark2")

#selected <- dplyr::filter(topic == c(3, 43, 12, 14, 31, 41)) #had to specify dplyr as otherwise it would not recognise topic as the data column. Think subset is the pipe operator does not work for ggplot2
timetopics = c(10, 12, 13, 21, 26, 47)
timelabels =c("Urban", "Capacity", "SDGs", "Europe", "SIDS", "Barrier")

p <- ggplot(subset(effects, topic %in% timetopics), 
            aes(x = value, y = proportion, ymin = lower, ymax = upper,
                color = topic, fill = topic)) +
  theme_light() +
  geom_line(size=1) +
  geom_ribbon(alpha = 0.05, colour = NA, show.legend =F)  +  #legend is added below as ggplot does not like having multiple types in one legend
  labs(x ='Year', y ='Topic Proportion', title = "Variation in topic proportion over time" )


#formatting
p + scale_color_manual(values = pal,  #NB: labels in alphabetical order of number -- 1, 10, 11, 12, ..., 2, 20 etc.
                       labels = timelabels) +
  scale_fill_manual(values= pal) + 
  coord_cartesian(xlim = c(1985, 2022), ylim = c(-0.01, 0.04),expand = F) +
  theme(text = element_text(size = 18),
        plot.title = element_text(size = 20, face = "bold"),
        legend.title = element_text(size = 18, face = "bold"))


#___
#Pre- and Post-Paris Agreement
prep_paris <- estimateEffect(1:model$settings$dim$K ~ Paris, 
                             model,meta=meta, uncertainty = "Global")

par(bty="n",col="darkslateblue",lwd=2)
plot.estimateEffect(prep_paris, add = F, covariate = "Paris", topics = 1:105,
                    model=model, method="difference", 
                    cov.value1="Post-Paris",cov.value2="Pre-Paris",
                    xlab="More Pre-Paris ... More Post-Paris",
                    main="Shifts in topics pre- and post- Paris Agreement",
                    xlim=c(-.01,.01), labeltype = "custom", verbose.labels = F,
                    topic.names = topicnames,
                    custom.labels = topicnames
                    )

plot

effects_paris <- get_effects(estimates = prep_paris, variable = 'Paris', 
                        type = 'difference',
                        cov_val1 = 'Post-Paris', cov_val2 = 'Pre-Paris')




#_____
#EXPORT
#Add the document IDs to the document topic matrix
write.csv(data$id,'Outputs/DocIdsUsed.csv', fileEncoding='UTF-8', row.names = FALSE)
theta <- data.frame(model$theta)
thetaOut <- cbind(data$id, theta)
#Using tidyverse writer; default somehow leaves decimal point & leading zeroes for values <1e-5
write_csv(thetaOut, 'Outputs/DocTopics.csv')#, fileEncoding='UTF-8', row.names = FALSE)


region <- data.frame(effects_region)
write_csv(region, 'Outputs/Region.csv')#, fileEncoding='UTF-8', row.names = FALSE)

paris<- data.frame(effects_paris)
write_csv(paris, 'Outputs/Paris.csv')#, fileEncoding='UTF-8', row.names = FALSE)


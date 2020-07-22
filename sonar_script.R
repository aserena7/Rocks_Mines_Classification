#-----------LOADING DATA - DATA PREPARATION - EXPLORATORY ANALYSIS-----------

#load useful libraries
library(corrplot)
library(ppcor)
library(tibble)
library(dplyr)
library(GGally)
library(tseries)
library(purrr)
library(tidyr)
library(readxl)
library(recipes)
library(mlr)
library(mlbench)
library(e1071)
library(kknn)
library(rpart)
library(rpart.plot)
library(kernlab)
library(nnet)
library(unbalanced)
library(DiscriMiner)
library(FSelectorRcpp)
library(praznik)
library(randomForest)
library(ada)
library(RWeka)
library(Hmisc)
library(tidyverse)
library(GGally) 
library(ggplot2)
library(superml)
library(apcluster)
library(mclust)
library(cluster)
library(stats)
library(factoextra)
library(plotrix)
library(bios2mds)

#load the dataset
sonar <- read.csv("C:/Users/Serena/Desktop/MACHINE LEARNING/assigment1/sonar.all-data", header=FALSE)
View(sonar)

#check NA values
sum(is.na(sonar))
summarizeColumns(sonar)
describe(sonar)

colnames(sonar)
names(sonar)[names(sonar) == "V61"] <- "Label"

#count occurencies in class values 
sonar %>%
  group_by(Label)  %>%
  count()
#no imbalanced class

#no exploratory analysis needed 

#target variable from string to integer

lbl <- LabelEncoder$new()
lbl$fit(sonar$Label)
sonar$Label <- lbl$fit_transform(sonar$Label)

sonar$Label <-as.factor(sonar$Label)

#define the task
task1 <- makeClassifTask(id = "task", data = sonar, target = "Label")
task1
getTaskFeatureNames(task1)

# Use 80% of the observations for training and 20% for test
# Random sample indexes
train_index <- sample(1:nrow(sonar), 0.8 * nrow(sonar))
test_index <- setdiff(1:nrow(sonar), train_index)

# Build the sets for supervised 
train <- sonar[train_index,1:61]
test <- sonar[test_index, 1:61]

#count occurencies in class values in training set
train %>%
  group_by(Label) %>%
  count()

####################################################SUPERVISED################################################################



#---------------------------------FIRST ANALYSIS-----------------------------------------------------------------------------------------------------------

#PROBABILISTIC
#1-------KNN-----

#learner
getParamSet("classif.kknn")
learner.knn <- makeLearner("classif.kknn",
                           predict.type = "response")
learner.knn$par.set
learner.knn

#Train
mod.knn <- mlr::train(learner.knn, task1, subset = train_index)
getLearnerModel(mod.knn)

#Predict
predict.knn <- predict(mod.knn, task = task1)
head(as.data.frame(predict.knn))
calculateConfusionMatrix(predict.knn)
predict.knn
pre<-as.data.frame(predict.knn)
pre

#Performance
listMeasures(task1)
performance(predict.knn, task1, measures = list(acc, mmce))

#Resampling
RCV.knn <- repcv(learner.knn, task1, folds = 5, reps = 2,
                 measures = list(acc, mmce), stratify = TRUE)
RCV.knn$aggr

#2-------RI----

#learner
getParamSet("classif.JRip")
learner.ri <- makeLearner("classif.JRip", 
                          predict.type = "response")
learner.ri$par.set

#Train
mod.ri <- mlr::train(learner.ri, task1, subset = train_index)
summary(mod.ri$learner.model)

#Predict
predict.ri <- predict(mod.ri, task = task1 )
predict.ri.test <- predict(mod.ri, newdata = test )
head(as.data.frame(predict.ri))
calculateConfusionMatrix(predict.ri)


#Performance
performance(predict.ri, measures = list(acc, mmce, kappa))
performance(predict.ri.test, measures = list(acc, mmce, kappa))

#Resampling
RCV.ri <- repcv(learner.ri, task1, folds = 5, reps=2,
                measures = list(acc, mmce, kappa), stratify = TRUE)

RCV.ri$aggr

#3-------DECISION TREE----

#learner
getParamSet("classif.rpart")
learner.dt <- makeLearner("classif.rpart", 
                          predict.type = "response")
learner.dt$par.set #same as getparamset

#Train
mod.dt <- mlr::train(learner.dt, task1, subset = train_index)
getLearnerModel(mod.dt)

#Predict
predict.dt <- predict(mod.dt, task = task1)
head(as.data.frame(predict.dt))
calculateConfusionMatrix(predict.dt)

predict.test.dt <- predict(mod.dt, newdata = test)
predict.test.dt
predict.dt
#Performance
listMeasures(task1)
performance(predict.dt, measures = list(acc, mmce, kappa))
performance(predict.test.dt, measures = list(acc, mmce, kappa))

#Resampling
RCV.dt <- repcv(learner.dt, task1, folds = 5, reps = 2, 
                measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.dt$aggr
rpart.plot(mod.dt$learner.model,  box.palette="RdBu", shadow.col="gray", nn=TRUE)

#4-------SVM----

#learner
getParamSet("classif.ksvm")
learner.svm <- makeLearner("classif.ksvm",
                           predict.type = "response")

#Train
mod.svm <- mlr::train (learner.svm, task1, subset = train_index)
getLearnerModel(mod.svm)

#Prediction
predict.svm <- predict(mod.svm, task1)
calculateConfusionMatrix(predict.svm)

#Performance
performance(predict.svm, measures = list(acc, mmce, kappa))

#Resampling
RCV.svm <- repcv(learner.svm, task1 , folds = 5,reps=2,
                 measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.svm$aggr

#5-------NN----

#learner
getParamSet("classif.nnet")
learner.nn <- makeLearner("classif.nnet", 
                          predict.type = "response")
learner.nn$par.set

#Train
mod.nn <- mlr::train(learner.nn, task1, subset = train_index)
getLearnerModel(mod.nn)

#Predict
predict.nn <- predict(mod.nn, task = task1)
head(as.data.frame(predict.nn))
calculateConfusionMatrix(predict.nn)

predict.test.nn <- predict(mod.nn, newdata = test)

#Performance
performance(predict.nn, measures = list(acc, mmce, kappa))
performance(predict.test.nn, measures = list(acc, mmce, kappa))

#Resampling
RCV.nn <- repcv(learner.nn, task1, folds = 5,reps=2,
                measures = list(acc, mmce, kappa), stratify = TRUE)

RCV.nn$aggr

#NO PROBABILISTIC 
#6-------LOGISTIC----

#learner
getParamSet("classif.logreg")
learner.lr <- makeLearner("classif.logreg",
                          predict.type = "response")

#Train
mod.lr <- mlr::train (learner.lr, task1, subset = train_index)
getLearnerModel(mod.lr)
summary(mod.lr$learner.model)


#Prediction
predict.lr <- predict(mod.lr, task1)
calculateConfusionMatrix(predict.lr)

#Performance
performance(predict.lr, measures = list(acc, mmce, kappa))

#Resampling
RCV.lr <- repcv(learner.lr, task1 , folds = 5,reps=2,
                measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.lr$aggr


#7-------NB----

#learner
getParamSet("classif.naiveBayes")
learner.nb <- makeLearner("classif.naiveBayes",
                          predict.type = "response")

#Train
mod.nb <- mlr::train(learner.nb, task1, subset = train_index)
getLearnerModel(mod.nb)

#Prediction
predict.nb <- predict(mod.nb, task1)
calculateConfusionMatrix(predict.nb)

#Performance
performance(predict.nb, task = task1, measures = list(acc, mmce, kappa), simpleaggr = TRUE)

#Resampling
RCV.nb <- repcv(learner.nb, task1, folds = 5, reps=2,
                measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.nb$aggr


#8-------DA----

#learner
getParamSet("classif.linDA")
learner.da <- makeLearner("classif.linDA", 
                          predict.type = "response")
learner.da$par.set

#Train
mod.da <- mlr::train(learner.da, task1, subset = train_index)
getLearnerModel(mod.da)

#Predict
predict.da <- predict(mod.da, task = task1)
head(as.data.frame(predict.da))
calculateConfusionMatrix(predict.da)

#Performance
performance(predict.da, measures = list(acc, mmce, kappa))

#Resampling
RCV.da <- repcv(learner.da, task1, folds = 5, reps=2,
                measures = list(acc, mmce, kappa), stratify = TRUE)

RCV.da$aggr

#--------Benchmark-----------

lrns <- list(learner.knn,learner.ri, learner.dt,learner.svm, learner.nn, 
             learner.lr, learner.nb, learner.da )
rdesc <- makeResampleDesc("RepCV", folds = 5, reps = 2) 
#Choose the resampling strategy
bmr <- benchmark(lrns, task1, rdesc, measures = list(acc, mmce))
getBMRPerformances(bmr, as.df = TRUE)
#sort
b1<-as.data.frame(getBMRAggrPerformances(bmr, as.df = TRUE))
b1<-b1[order(-b1$acc.test.mean),]
rownames(b1) <- NULL
b1

#---------------------------------SECOND ANALYSIS----------------------------------
#1-------KNN-------

fv.knn <- generateFilterValuesData(task1,
                                   method = "FSelectorRcpp_information.gain")
fv.knn$data

lrn.fss.knn <- makeFilterWrapper(learner = "classif.kknn", 
                                 fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
rdesc <- makeResampleDesc("RepCV", folds = 5, reps = 2)
r.knn.fss = resample(learner = lrn.fss.knn, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.knn.fss$aggr

mod.knn.fss <- mlr::train(lrn.fss.knn, task1, subset=train_index)
predict.knn.fss <- predict(mod.knn.fss, task = task1)
performance(predict.knn.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.knn)
getLearnerModel(mod.knn.fss)

#2-------RI-----------

lrn.fss.ri<- makeFilterWrapper(learner = "classif.JRip", 
                               fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.ri.fss = resample(learner = lrn.fss.ri, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.ri.fss$aggr

mod.ri.fss <- mlr::train(lrn.fss.ri, task1, subset=train_index)
predict.ri.fss <- predict(mod.ri.fss, task = task1)
performance(predict.ri.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.ri.fss)
getLearnerModel(mod.ri)

#3-------TREES-----

lrn.fss.dt <- makeFilterWrapper(learner = "classif.rpart", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.dt.fss = resample(learner = lrn.fss.dt, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.dt.fss$aggr

mod.dt.fss <- mlr::train(lrn.fss.dt, task1, subset=train_index)
predict.dt.fss <- predict(mod.dt.fss, task = task1)
performance(predict.dt.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.dt.fss)
getLearnerModel(mod.dt)

#4-------SVM--------

lrn.fss.svm <- makeFilterWrapper(learner = "classif.ksvm", 
                                 fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
rdesc.svm <- makeResampleDesc("Holdout")
r.svm.fss = resample(learner = lrn.fss.svm, task = task1, resampling = rdesc.svm, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.svm.fss$aggr

mod.svm.fss <- mlr::train(lrn.fss.svm, task1, subset=train_index)
predict.svm.fss <- predict(mod.svm.fss, task = task1)
performance(predict.svm.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.svm.fss)
getLearnerModel(mod.svm)

#5-------ANN--------

lrn.fss.nn <- makeFilterWrapper(learner = "classif.nnet", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.nn.fss = resample(learner = lrn.fss.nn, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.nn.fss$aggr

mod.nn.fss <- mlr::train(lrn.fss.nn, task1, subset=train_index)
predict.nn.fss <- predict(mod.nn.fss, task = task1)
performance(predict.nn.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.nn.fss)
getLearnerModel(mod.nn)

#6-------LOGISTIC-----

lrn.fss.lr <- makeFilterWrapper(learner = "classif.logreg", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.lr.fss = resample(learner = lrn.fss.lr, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.lr.fss$aggr

mod.lr.fss <- mlr::train(lrn.fss.lr, task1, subset=train_index)
predict.lr.fss <- predict(mod.lr.fss, task = task1)
performance(predict.lr.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.lr.fss)
getLearnerModel(mod.lr)

#7-------NB-----------

lrn.fss.nb <- makeFilterWrapper(learner = "classif.naiveBayes", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.nb.fss = resample(learner = lrn.fss.nb, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.nb.fss$aggr

mod.nb.fss <- mlr::train(lrn.fss.nb, task1, subset=train_index)
predict.nb.fss <- predict(mod.nb.fss, task = task1)
performance(predict.nb.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.nb.fss)
getLearnerModel(mod.nb)

#8-------DA--------

lrn.fss.da <- makeFilterWrapper(learner = "classif.linDA", 
                                fw.method = "FSelectorRcpp_information.gain", fw.perc = 0.6)
r.da.fss = resample(learner = lrn.fss.da, task = task1, resampling = rdesc, show.info = FALSE, models = TRUE, measures = mlr::acc)
r.da.fss$aggr

mod.da.fss <- mlr::train(lrn.fss.da, task1, subset=train_index)
predict.da.fss <- predict(mod.da.fss, task = task1)
performance(predict.da.fss, measures = list(acc, mmce, kappa))

getLearnerModel(mod.nn.fss)
getLearnerModel(mod.nn)

#---------Benchmark-----------

lrns.fss <- list(lrn.fss.knn, lrn.fss.ri, lrn.fss.dt, lrn.fss.svm,lrn.fss.nn,
                 lrn.fss.lr, lrn.fss.nb,lrn.fss.da)
bmr.fss <- benchmark(lrns.fss, task1, rdesc, measures = list(acc, mmce))
getBMRAggrPerformances(bmr.fss, as.df = TRUE)
b2<-as.data.frame(getBMRAggrPerformances(bmr.fss, as.df = TRUE))
b2<-b2[order(-b2$acc.test.mean),]
rownames(b2) <- NULL
b2

#---------------------------------THIRD ANALYSIS----------------------------------

#1------ KNN--------

lrn.wra.knn <- makeFeatSelWrapper(learner = "classif.knn",
                                  resampling = rdesc, control = 
                                    makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
lr.knn.wra <- resample(lrn.wra.knn, task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
lr.knn.wra$aggr
getParamSet(lrn.wra.knn)
mod.knn.wra <- mlr::train(lrn.wra.knn, task1, subset=train_index)
getFeatSelResult(mod.knn.wra)
predict.knn.wra <- predict(mod.knn.wra, task = task1)
performance(predict.knn.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.knn.wra)
lapply(mod.knn.wra$models, getFeatSelResult)
#2-------RI----------

lrn.wra.ri <- makeFeatSelWrapper(learner = "classif.JRip",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.ri.wra <- resample(lrn.wra.ri, task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.ri.wra$aggr
mod.ri.wra <- mlr::train(lrn.wra.ri, task1, subset=train_index)
predict.ri.wra <- predict(mod.ri.wra, task = task1)
performance(predict.ri.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.ri.wra)

#3-------TREES-------

lrn.wra.dt <- makeFeatSelWrapper(learner = "classif.rpart",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.dt.wra <- resample(lrn.wra.dt, task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.dt.wra$aggr
mod.dt.wra <- mlr::train(lrn.wra.dt, task1, subset=train_index)
predict.dt.wra <- predict(mod.dt.wra, task = task1)
performance(predict.dt.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.dt.wra)

#4-------SVM--------

lrn.wra.svm <- makeFeatSelWrapper(learner = "classif.ksvm",
                                  resampling = rdesc.svm, control = 
                                    makeFeatSelControlRandom(maxit = 1), show.info = FALSE)
r.svm.wra <- resample(lrn.wra.svm, task1, resampling = rdesc.svm, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.svm.wra$aggr
mod.svm.wra <- mlr::train(lrn.wra.svm, task1, subset=train_index)
predict.svm.wra <- predict(mod.svm.wra, task = task1)
performance(predict.svm.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.svm.wra)

#5-------ANN--------

lrn.wra.nn <- makeFeatSelWrapper(learner = "classif.nnet",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.nn.wra <- resample(lrn.wra.nn, task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.nn.wra$aggr
mod.nn.wra <- mlr::train(lrn.wra.nn, task1, subset=train_index)
predict.nn.wra <- predict(mod.nn.wra, task = task1)
performance(predict.nn.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.nn.wra)

#6-------LOGISTIC---------

lrn.wra.lr <- makeFeatSelWrapper(learner = "classif.logreg",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.lr.wra <- resample(lrn.wra.lr, task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.lr.wra$aggr
mod.lr.wra <- mlr::train(lrn.wra.lr, task1, subset=train_index)
predict.lr.wra <- predict(mod.lr.wra, task = task1)
performance(predict.lr.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.lr.wra)

#7-------NB-------

lrn.wra.nb <- makeFeatSelWrapper(learner = "classif.naiveBayes",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.nb.wra <- resample(lrn.wra.nb, task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.nb.wra$aggr
mod.nb.wra <- mlr::train(lrn.wra.nb, task1, subset=train_index)
predict.nb.wra <- predict(mod.nb.wra, task = task1)
performance(predict.nb.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.nb.wra)
#8-------DA------

lrn.wra.da <- makeFeatSelWrapper(learner = "classif.linDA",
                                 resampling = rdesc, control = 
                                   makeFeatSelControlRandom(maxit = 3), show.info = FALSE)
r.da.wra <- resample(lrn.wra.da,task1, resampling = rdesc, models = TRUE, show.info = FALSE, measures = mlr::acc)
r.da.wra$aggr
mod.da.wra <- mlr::train(lrn.wra.da, task1, subset=train_index)
predict.da.wra <- predict(mod.da.wra, task = task1)
performance(predict.da.wra, measures = list(acc, mmce, kappa))
getLearnerModel(mod.da.wra)

#---------Benchmark-----------

lrns.wra <- list(lrn.wra.knn, lrn.wra.ri, lrn.wra.dt, lrn.wra.svm, lrn.wra.nn,
                 lrn.wra.lr, lrn.wra.nb,lrn.wra.da)
bmr.wra <- benchmark(lrns.wra, task1, rdesc, measures = list(acc, mmce))
getBMRAggrPerformances(bmr.wra, as.df = TRUE)
b3<-as.data.frame(getBMRAggrPerformances(bmr.wra, as.df = TRUE))
b3<-b3[order(-b3$acc.test.mean),]
rownames(b3) <- NULL
b3



#######################METACLASSIFIER#############################
#9--------Bagging-------

learner.nn.bagg <- makeLearner("classif.nnet")
learner.nn.bagging <- makeBaggingWrapper(learner.nn.bagg, bw.iters = 20, bw.replace = TRUE)
r.nn.bagging = resample(learner.nn.bagging, task1, resampling = rdesc, show.info = FALSE)
r.nn.bagging$aggr

mod.nn.bagg <- mlr::train(learner.nn.bagging, task1, subset=train_index)
predict.nn.bagg <- predict(mod.nn.bagg, task1)
performance(predict.nn.bagg, measures = list(acc, mmce))
calculateConfusionMatrix(predict.nn.bagg)

RCV.bagg <- repcv(learner.nn.bagg, task1, folds = 5, reps = 2, 
                  measures = list(acc, mmce), stratify = TRUE)
RCV.bagg$aggr

getLearnerModel(mod.nn.bagg)

#10-------RandomForest-----------

#learner
getParamSet("classif.randomForest")
learner.randomf <- makeLearner("classif.randomForest", 
                               predict.type = "response", ntree=100)
learner.randomf$par.set

#Train
mod.randomf <- mlr::train(learner.randomf,task1, subset=train_index)
getLearnerModel(mod.randomf)



#Predict
predict.randomf <- predict(mod.randomf, task = task1)
head(as.data.frame(predict.randomf))
calculateConfusionMatrix(predict.randomf)

#Performance
performance(predict.randomf, measures = list(acc, mmce))

RCV.randomf <- repcv(learner.randomf, task1, folds = 5, reps = 2, 
                     measures = list(acc, mmce), stratify = TRUE)
RCV.randomf$aggr

#11-------AdaBoosting---------

#learner
getParamSet("classif.ada")
learner.ada <- makeLearner("classif.ada", 
                           predict.type = "response")
learner.ada$par.set

#Train
mod.ada <- mlr::train(learner.ada, task1, subset=train_index)
getLearnerModel(mod.ada)

#Predict
predict.ada <- predict(mod.ada, task =task1)
head(as.data.frame(predict.ada))
calculateConfusionMatrix(predict.ada)

#Performance
performance(predict.ada, measures = list(acc, mmce, kappa))

RCV.ada <- repcv(learner.ada, task1, folds = 5, reps = 2, 
                 measures = list(acc, mmce, kappa), stratify = TRUE)
RCV.ada$aggr

getLearnerModel(mod.ada)

#----------Benchmark-----------

meta_lrns <- list(learner.nn.bagg, learner.randomf, learner.ada)
meta_rdesc <- makeResampleDesc("RepCV", folds = 5, reps = 2) 
#Choose the resampling strategy
meta_bmr <- benchmark(meta_lrns, task1, meta_rdesc, measures = list(acc, mmce))
getBMRPerformances(meta_bmr, as.df = TRUE)
getBMRAggrPerformances(meta_bmr, as.df = TRUE)
Bmeta<-as.data.frame(getBMRAggrPerformances(meta_bmr, as.df = TRUE))
Bmeta<-Bmeta[order(-Bmeta$acc.test.mean),]
rownames(Bmeta) <- NULL
Bmeta



##################################################UNSUPERVISED########################################

#removing the Label
sonar$Label<- NULL


#12-------Kmeans------
sonar <- as.data.frame(scale(sonar[, 1:60]))
result<- kmeans(sonar, 3)
result$size
table(result)
result$cluster
table(result$cluster, sonar$Label)
fviz_cluster(result, data = sonar)

#which k to choose?
k.max <- 10
data <- as.data.frame(scale(sonar[, 1:60]))
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 10 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


#13-------Hierarchical Cluster-----

dist_mat <- dist(sonar[, 1:60], method = 'euclidean')
clusters <- hclust(dist(sonar[, 1:60]))
clusters <- hclust(dist_mat,method = 'complete')
plot(clusters)
clusterCut <- cutree(clusters, 2)
table(clusterCut, sonar$Label)
rect.hclust(clusters , k = 2, border = 2:6)


#14-------Gaussian Mixtures------

sonar$Label<-as.numeric(sonar$Label)
BIC <- mclustBIC(sonar[, 1:60])
summary(BIC)
mod1 <- Mclust(sonar[, 1:60], x = BIC)
summary(mod1)
table(mod1$classification, sonar$Label)

#other way
d_clust <- Mclust(as.matrix(data), G=1:15, 
                  modelNames = mclust.options("emModelNames"))
d_clust$BIC
plot(d_clust)

###########################################conclusions###################
algorithm = c("KNN","RI","TREES","SVM","ANN","LOGISTIC","NB","DA","BAGGING","RF","ADA","KMEANS","HIERARC","GMM")
accuracy = c(0.85,0.70,0.72,0.82,0.78,0.73,0.68,0.75,0.76,0.83,0.82,0.54,0.49,0.47)
percentage = data$accuracy*100
data<-data.frame(algorithm,accuracy,percentage)

my_bar <- barplot(data$accuracy , border=F , names.arg=data$algorithm , 
                  las=2 , 
                  #col=rgb(0.3,0.1,0.4) , 
                  ylim=c(0,1) , 
                  main="" )
abline(h=0.7, col="blue")
text(my_bar, 0.7, accuracy,cex=1,pos=3) 

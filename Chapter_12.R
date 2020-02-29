#  ==========================================================================================
# Libraries ---------------------------------------------------------------
#  ==========================================================================================
# This section uses the following R packages: AppliedPredictiveModeling, caret, glmnet, MASS, 
# pamr, pls, pROC, rms, sparseLDA, and subselect. As indicated in the textbook, the seed value 
# of 476 was randomly chosen for this chapter.

library(AppliedPredictiveModeling)
library(caret)
library(glmnet)
library(MASS)
library(pamr)
library(pls)
library(pROC)
library(rms)
library(sparseLDA)
library(subselect)
library(parallel)
library(pROC)

options(mc.cores = detectCores())
#  ==========================================================================================
# Exercise 12.1 -----------------------------------------------------------
#  ==========================================================================================
# The hepatic injury data set contains 281 unique compounds, each of which has been classified 
# as causing no liver damage, mild damage, or severe damage. These compounds were analyzed with
# 184 biological screens (i.e. experiments) to assess each compound’s effect on a particular 
# biologically relevant target in the body. The larger the value of each of these predictors, 
# the higher the activity of the compound. In addition to biological screens, 192 chemical
# fingerprint predictors were determined for these compounds. Each of these predictors represent
# a substructure (i.e. an atom or combination of atoms within the compound) and are either 
# counts of the number of substructures or an indicator of presence or absence of the particular
# substructure. The objective of this data set is to build a predictive model for hepatic injury
# so that other compounds can be screened for the likelihood of causing hepatic injury.
data(hepatic)

dim(bio) # Biological assay predictors
dim(chem) # Chemical fingerprint predictors
table(injury) # Liver damage classification

# (a) Given the classification imbalance in hepatic injury status, describe how you would create
# a training and testing set.

# Generate an initial seed:
# initialSeed <- as.integer(Sys.time())
# theSeed <- initialSeed %% 100
# theSeed # 77

set.seed(77)

# Modify liver injury categories into binary indicators:
CausesDamage <- as.character(injury)
CausesDamage[CausesDamage == "Severe"] = "Yes"
CausesDamage[CausesDamage == "Mild"] = "Yes"
CausesDamage[CausesDamage == "None"] = "No"

# Generate a data partition index:
LiverInjuryIndex <- createDataPartition(CausesDamage, p = .8, 
                                        list = FALSE, times = 1)

# Check proportions:
# the population proportions...
table(CausesDamage) / sum(table(CausesDamage))

# proportions from one resampling...
table(CausesDamage[LiverInjuryIndex]) / sum(table(CausesDamage[LiverInjuryIndex]))
table(CausesDamage[-LiverInjuryIndex]) / sum(table(CausesDamage[-LiverInjuryIndex]))
 
# Convert response to a factor:
CausesDamage <- as.factor(CausesDamage)

# Create training and test set:
trainingYInjury <- CausesDamage[LiverInjuryIndex]
testingYInjury <- CausesDamage[-LiverInjuryIndex]

# (b) Which classification statistic would you choose to optimize for this exercise and why?

# A matrix of variable importance. The problem at hand wants to optimise screening for the 
# likelihood of causing hepatic injury by identifying the variable with high likelihoods of a 
# positive outcome. One does not need to maximise sensivity or specifity. The problem wants a
# solution to find the best predictors of injury.

# (c) Split the data into a training and a testing set, pre-process the data, and build models
# described in this chapter for the biological predictors and separately for the chemical
# fingerprint predictors. Which model has the best predictive ability for the biological 
# predictors and what is the optimal performance? Which model has the best predictive ability 
# for the chemical predictors and what is the optimal performance? Based on these results, which
# set of predictors contains the most information about hepatic toxicity?

# Train Control -----------------------------------------------------------
tCtrl <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary, 
                     classProbs = TRUE, 
                     savePredictions = TRUE)

#  ==========================================================================================
# Bio data models ----------------------------------------------------------------
#  ==========================================================================================
set.seed(77)

# Bio Data ----------------------------------------------------------------
# Remove uninformative predictors:
bioZeroVar <- nearZeroVar(bio)

bioData <- bio[, -bioZeroVar]
dim(bioData)

# Remove Lienar combos:
findLinearCombos(bioData)
# ... there are none.

# Bind outcome Y variable to data set and split data into training and test sets:
trainingBioData <- cbind(trainingYInjury, bioData[LiverInjuryIndex, ])
dim(trainingBioData)

testingBioData <- cbind(testingYInjury, bioData[-LiverInjuryIndex, ])
dim(testingBioData)

# Logistic Bio Model ----------------------------------------------------------
# ... check levels:
levels(trainingBioData$trainingYInjury)

# Run standard glm model:
bioLogit <- glm(trainingYInjury ~ ., data = trainingBioData, family = "binomial")
summary(bioLogit)
# The glm function treats the second factor level as the event of interest. Since the slope is
# positive for the a positive outcome, we do not need to subtract the compliment 1 - p...
positivePred <- predict(bioLogit,
                        newdata = testingBioData, type = "response")
positivePred[1:5]
head(varImp(bioLogit))

# For a large set of predictors, the formula method for specifying models can be cumbersome. As
# in previous chapters, the train function can efficiently fit and validate models. For logistic 
# regression, train provides an interface to the glm function that bypasses a model formula,
# directly produces class predictions, and calculates the area under the ROC curve and other 
# metrics:
set.seed(77)
bioTrainLogit <- train(x = trainingBioData[, -1], y = trainingYInjury, 
                       method = "glm", metric = "ROC", trControl = tCtrl)
summary(bioTrainLogit)
table(bioTrainLogit$pred$pred, bioTrainLogit$pred$obs)
sensitivity(bioTrainLogit$pred$pred, bioTrainLogit$pred$obs)
specificity(bioTrainLogit$pred$pred, bioTrainLogit$pred$obs)
posPredValue(bioTrainLogit$pred$pred, bioTrainLogit$pred$obs)
negPredValue(bioTrainLogit$pred$pred, bioTrainLogit$pred$obs)

ctrainingMatrix <- confusionMatrix(data = bioTrainLogit$pred$pred, 
                                reference = bioTrainLogit$pred$obs, positive = "Yes")
ctrainingMatrix
ctrainingMatrix$overall

trainingROC <- roc(response = bioTrainLogit$pred$obs, predictor = bioTrainLogit$pred$Yes)
plot(trainingROC)

# Prediction testing set: 
bioTestLogit <- predict(bioTrainLogit, testingBioData[, -1])
table(bioTestLogit, testingBioData$testingYInjury)
LogitTestProbs <- predict(bioTrainLogit, newdata = testingBioData[, -1],
                        type = "prob")

ctestingMatrix <- confusionMatrix(bioTestLogit, testingBioData$testingYInjury, 
                                positive = "Yes")
ctestingMatrix

# ... although expected, the testing set does significantly more poorly than the 
# training set.

SummaryStatistics <- list(TrainingStats = ctestingMatrix$overall, 
                          TestingStats = ctrainingMatrix$overall)
SummaryStatistics

# LDA Bio model ---------------------------------------------------------------------
# It is important to first centre and scale the data in a LDA model.
set.seed(77)
# I can do this easily from within inside the caret train() function:
bioLDA <- train(x = trainingBioData[, -1], y = trainingBioData$trainingYInjury,
                method = "lda", 
                preProcess = c("center", "scale"),
                metric = "ROC",
                trControl = tCtrl)

cTrainingLDAMatrix <- confusionMatrix(data = bioLDA$pred$pred, 
                                     reference = bioLDA$pred$obs, positive = "Yes")
cTrainingLDAMatrix

LDAroc <- roc(response = bioLDA$pred$obs, predictor = bioLDA$pred$Yes, 
              levels = bioLDA$levels)
plot(LDAroc)

# Test set:
LDApredict <- predict(bioLDA, newdata = testingBioData[, -1])
table(LDApredict, testingBioData$testingYInjury)

LDATestProbs <- predict(bioLDA, newdata = testingBioData[, -1],
                        type = "prob")
plot(density(LDATestProbs$Yes))
plot(density(LDATestProbs$No))

cTestingLDAMatrix <- confusionMatrix(LDApredict, 
                                     reference = testingBioData$testingYInjury, 
                                     positive = "Yes")
cTestingLDAMatrix
# Improved prediction with the testing set. However, with a wider CI and other trade-off's.
# Does not indicate a balanced prediction model.
LDASummaryStats <- list(TrainingcTable = cTrainingLDAMatrix$table, 
                        TestingcTable = cTestingLDAMatrix$table,
                        TrainingStats = cTrainingLDAMatrix$overall,
                        TestingStats = cTestingLDAMatrix$overall)
LDASummaryStats

# Partial Least Sqaures Discriminant Analysis Bio Model -----------------------
# Here again we can use the train function in caret in order to wrap up all pre-processing
# and model specification in one go:
set.seed(77)
bioPLS <- train(x = trainingBioData[, -1], y = trainingBioData$trainingYInjury,
                method = "pls", tuneGrid = expand.grid(.ncomp = 1:10),
                preProcess = c("center", "scale"),
                metric = "ROC",
                trControl = tCtrl)
bioPLS
plsImpBio <- varImp(bioPLS, scale = FALSE)
plsImpBio
plot(plsImpBio, top = 20, scales = list(y = list(cex = .95)),
     ylab = "Predictor variables", xlab = "Variable Importance",
     main = "Component variable importance")


plsPred <- predict(bioPLS,
                   newdata = testingBioData[, -1])
plsProb <- predict(bioPLS, newdata = testingBioData[, -1],
                   type = "prob")
head(plsProb)

PLSTestingcMatrix<- confusionMatrix(data = bioPLS$pred$pred, 
                                    reference = bioPLS$pred$obs, 
                                    positive = "Yes")
PLSTestingcMatrix

plsROC <- roc(response = bioPLS$pred$obs, predictor = bioPLS$pred$Yes)
plot.roc(plsROC)

#  ==========================================================================================
# Chem data models --------------------------------------------------------
#  ==========================================================================================
# So it seems to be a bad idea to split the chem data set. The data split creates GIGO. 
# Therefore, despite perfoming relatively well, it is important to be skeptical of model 
# performance relative to the bio set.
set.seed(77)

# Chem data processing ---------------------------------------------------------------
# ...remove uninformative predictors:
chemZeroVar <- nearZeroVar(chem)
chemData <- chem[, -chemZeroVar]
dim(chemData)

# Remove linear combos:
chemLinearCombos <- findLinearCombos(chemData)
chemLinearCombos$remove
# ... NULL...

# Bind outcome Y variable to data set and split data into training and test sets:
chemData <- cbind(CausesDamage, chemData)
dim(chemData)

# check proportions:
dim(trainingBioData)
dim(testingBioData)

dim(chemData)
# ... same number of rows.

# Chem Logistic Model -----------------------------------------------------
chemLogit <- train(x = chemData[, -1], y = chemData$CausesDamage,
                   method = "glm", metric = "ROC", trControl = tCtrl)
summary(chemLogit)

table(chemLogit$pred$pred, chemLogit$pred$obs)
sensitivity(chemLogit$pred$pred, chemLogit$pred$obs)
specificity(chemLogit$pred$pred, chemLogit$pred$obs)
posPredValue(chemLogit$pred$pred, chemLogit$pred$obs)
negPredValue(chemLogit$pred$pred, chemLogit$pred$obs)

cMatrixChem <- confusionMatrix(data = chemLogit$pred$pred, reference = chemLogit$pred$obs, 
                               positive = "Yes")
cMatrixChem

chemLogitROC <- roc(response = chemLogit$pred$obs, predictor = chemLogit$pred$Yes)
plot(chemLogitROC)

# Chem LDA Model ----------------------------------------------------------
chemLDA <- train(x = chemData[, -1], y = chemData$CausesDamage,
                 method = "lda", preProcess = c("center", "scale"), metric = "ROC",
                 trControl = tCtrl)
summary(chemLDA)
cMatrixChemLDA <- confusionMatrix(data = chemLDA$pred$pred, 
                                  reference = chemLDA$pred$obs, positive = "Yes")
cMatrixChemLDA

# Chem PLSDA Model --------------------------------------------------------
# Note that PLSDA is preferavle to LDA, as it simultaneously reduces dimensions while minimising
# missclassification - the vice versa of PLS for continuous response. Additionally, the PLSDA 
# model encodes response as 0/1 binary outcomes. Post-processing of the output is required if 
# one desires class probabilities.
set.seed(77)

chemPLSDA <- train(x = chemData[, -1], y = chemData$CausesDamage,
                   method = "pls", tuneGrid = expand.grid(.ncomp = 1:10),
                   preProcess = c("center", "scale"),
                   metric = "ROC",
                   trControl = tCtrl)
plot(chemPLSDA)
# ... maximises at 7 components...
chemVarImpPLS <- varImp(chemPLSDA, scale = FALSE)
chemVarImpPLS

plot(chemVarImpPLS, top = 20, scale = list(y = list(cex = .80)),
     ylab = "Predictor variables", xlab = "Variable Importance",
     main = "Component variable importance")

chemPLSDAcMatrix <- confusionMatrix(data = chemPLSDA$pred$pred,
                                    reference = chemPLSDA$pred$obs,
                                    positive = "Yes")

chemPLSDAcMatrix

chemPLSroc <- roc(response = chemPLSDA$pred$obs, predictor = chemPLSDA$pred$Yes)
plot(chemPLSroc)

chemPLSDA$finalModel$loadings
components <- chemPLSDA$finalModel$loadings[, 7]
# ... the predictor variables with the highest cumulative variance, in the 7th component are:
names(components)

# I am not going to complete the other models as they are rarely used, if ever, in my field of expertise.

# End file ----------------------------------------------------------------

# This chapter will reference functions from the caret, earth, kernlab, and nnet packages.

# ===============================================================================================
# Libraries ---------------------------------------------------------------
# ===============================================================================================
library(caret)
library(earth)
library(kernlab)
library(nnet)
library(mlbench)
library(e1071)
library(AppliedPredictiveModeling)

# ===============================================================================================
# Neural Networks ---------------------------------------------------------
# ===============================================================================================
# R has a number of packages and functions for creating neural networks. Relevant packages 
# include nnet, neural, and RSNNS. The nnet package is the focus here since it supports the basic 
# neural network models outlined in this chapter (i.e., a single layer of hidden units) and 
# weight decay and has simple syntax. RSNNS supports a wide array of neural networks.

# To fit a regression model, the nnet function takes both the formula and non-formula interfaces.
# For regression, the linear relationship between the hidden units and the prediction can be used
# with the option linout = TRUE. A basic neural network function call would be:
nnetFit <- nnet(predictors, outcome,
                size = 5,
                decay = .01, 
                lineout = TRUE,
                # Reduce the amount of printed output:
                trace = FALSE,
                # Expand the number of iterations to find
                # parameter estimates:
                maxit = 500,
                # and the number of parameters used by the
                # model are set to be:
                MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)
# This would create a single model with 5 hidden units. Note, this assumes that the data in 
# predictors have been standardized to be on the same scale.

# To use model averaging, the avNNet function in the caret package has nearly identical syntax:
nnetAvg <- avNNet(predictors, outcome,
                  size = 5, 
                  decay = .01,
                  # Specify how many models to average:
                  repeats = 5, 
                  lineout = TRUE,
                  # Reduce the amount of printed output:
                  trace = FALSE,
                  # Expand the number of iterations to find
                  # parameter estimates:
                  maxit = 500,
                  # and the number of parameters used by the
                  # model are set to be:
                  MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)
# Again, new samples are processed using:
predict(nnetFit, newData)
# or
predict(nnetAvg, newData)

# To mimic the earlier approach of choosing the number of hidden units and the amount of weight 
# decay via resampling, the train function can be applied using either method = "nnet" or 
# method = "avNNet". First, we remove correlated predictors to ensure that the maximum absolute
# pairwise correlation between the predictors is less than 0.75. Note that the findCorrelation 
# takes a correlation matrix and determines the column numbers that should be removed to keep all
# pair-wise correlations below a threshold:
tooHigh <- findCorrelation(cor(trainingXdataTrans), cutoff = .75)
trainXnnet <- trainingXDataTrans[, -tooHigh]
testXnnet <- testingDataTrans[, -tooHigh]
# Then create a specific candidate set of models to evaluate:
nnetGrid <- expand.grid(.decay = c(0, .01, .1),
                        .size = c(1:10),
                        # The next option is to use bagging instead
                        # of different random seeds
                        .bag = FALSE)
set.seed(100)
nnetTune <- train(trainingXdataTrans, trainingYData,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = tCtrl,
                  # Automatically standardise data prior to modelling
                  # and prediction:
                  preProcess = c("center", "scale"),
                  lineout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
                  maxit = 500)

# ===============================================================================================
# Multivariate Adaptive Regression Splines --------------------------------
# ===============================================================================================
# MARS models are in several packages, but the most extensive implementation is in the earth 
# package. The MARS model using the nominal forward pass and pruning step can be called simply
# by:
marsFit <- earth(trainingXDataTrans, trainingYdata)
marsFit
# Note, however, that since this model used the internal GCV technique for model selection, the
# details of this model are different than the one used previously in the book chapter. The 
# summary method generates more extensive output:
summary(marsFit)

# The plotmo function in the earth package can be used to produce plots for MARS models. To tune
# the model using external resampling, the train function can be used.

# First, you need to define the candidate models to test:
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
# Note that the selection for pruning is excluding the intercept and the rest of the model terms
# used in the original marsFit model.

# Then fix the seed so that the results can be reproduced:
set.seed(100)
marsTune <- train(trainingXDataTrans, trainingYData,
                  method = "earth",
                  # Explicitly declare the candidate models 
                  # to test:
                  tuneGrid = marsGrid,
                  trControl = trainControl(method = "cv"))
marsTune
head(predict(marsTune, testingXData))
# However, note that the train function will accept a formula coding of y ~ x1... etc.

# There are two functions that estimate the importance of each predictor in the MARS model: 
# evimp in the earth package and varImp in the caret package (although the latter calls the 
# former):
varImp(marsTune)

# ===============================================================================================
# Support Vector Machines -------------------------------------------------
# ===============================================================================================
# There are a number of R packages with implementations of support vector machine models. The 
# svm function in the e1071 package has an interface to the LIBSVM library for regression. A 
# more comprehensive implementation of SVM models for regression is the kernlab package. In that
# package, the ksvm function is available for regression models and a large number of kernel
# functions. The radial basis function is the default kernel function. If appropriate values of
# the cost and kernel parameters are known, this model can be fit as:
svmFit <- ksvm(x = trainingXdataTrans, y = trainingYData,
               kernel = "rbfdt", kpar = "automatic",
               C = 1, epsilon = .1)
# The function automatically uses the analytical approach to estimate the kernel parameter. For
# this example, y is a numeric vector, and thus the function knows to fit a regression model 
# (instead of a classification model). Other kernel functions can be used, including the 
# polynomial (using kernel = "polydot") and linear (kernel = "vanilladot").

# If the values are unknown, they can be estimated through resampling. In train, the method 
# values of "svmRadial", "svmLinear", or "svmPoly" fit different kernels:
svmTune <- train(trainingXDataTrans, trainingYData,
                 method = "svmRadial",
                 preProcess = c("center", "scale"),
                 tuneLength = 14,
                 trControl = trainControl(method = "cv"))
# The tuneLength argument will use the default grid search of 14 cost values −2−1 11 between 
# 2^-2, 2^-1, . . . , 2^11. Again, the kernel parameter is estimated analytically by default. 
# The svm model can be inspected by calling:
svmTune
# The sub-object named finalModel contains the model created by the ksvm function:
svmTune$finalModel

# kernlab has an implementation of the RVM model for regression in the function rvm. The syntax
# is very similar to the example shown for ksvm.

# ===============================================================================================
# K-Nearest Neighbours ----------------------------------------------------
# ===============================================================================================
# The knnreg function in the caret package fits the KNN regression model; train tunes the model
# over K:

# First, it is best practice to investigate the need to remove sparse and unbalanced predictors
# first, for example:
knnDescr <- trainingXDataTrans[, -nearZeroVar(trainingXDataTrans)]

set.seed(100)
knnTune <- train(knnDescr,trainingYData,
                 method = "knn",
                 # Center and scaling will occur for new 
                 # predictions too
                 preProcess = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = trainControl(method = "cv"))
# When predicting new samples using this object, the new samples are automatically centered and
# scaled using the values determined by the training set.

# ===============================================================================================
# Exercise 7.1 ---------------------------------------------------------------
# ===============================================================================================
# Simulate a single predictor and a nonlinear relationship, such as a sin wave, and investigate 
# the relationship between the cost, error, and kernel parameters for a support vector machine
# model:
set.seed(100)
x <- runif(100, min = 2, max = 10)
y <- sin(x) + rnorm(length(x)) * .25
sinData <- data.frame(x = x, y = y)

plot(x, y)
# Create a grid of x values to use for prediction:
dataGrid <- data.frame(x = seq(2, 10, length.out = 100))

# a) Now fit different models using a radial basis function and different values of the cost
# (the C parameter) and error. Plot the fitted curve. For example:
rbfSVM <- ksvm(x = x, y = y, data = sinData,
               kernel = "rbfdot", kpar = "automatic",
               C = 1, epsilon = .1)
svmPred <- predict(rbfSVM, newdata = dataGrid)
# This is a matrix with one column. We can plot the model predictions by adding points to the 
# previous plot:
points(x = dataGrid$x, y = svmPred[, 1],
       type = "l", col = "blue")

# b) The kernel parameter can be adjusted using the kpar argument, such as kpar = list(sigma = 1).
# Try different values of the kernel parameter to understand how this parameter changes the 
# model fit. How do the cost, error, and kernel parameter values affect the model?
rbfSVM_1 <- ksvm(x = x, y = y, 
               data = sinData,
               kernel = "polydot",
               kpar = list(degree = 1),
               C = 1, epsilon = .1)
svmPred_poly <- predict(rbfSVM_1, newdata = dataGrid)
points(x = dataGrid$x, y = svmPred_poly[, 1],
       type = "line", col = "red")


rbfSVM_2 <- ksvm(x = x, y = y, 
               newdata = sinData, 
               kernel = "vanilladot",
               kpar = "automatic",
               C = 2, epsilon = .2)
svmPred_vanilla <- predict(rbfSVM_2, newdata = dataGrid)
points(x = dataGrid$x, y = svmPred_vanilla[, 1],
       type = "line", col = "yellow")
# Kernel parameters change the shape of the relationship between predictors and outcome.

# ===============================================================================================
# Exercise 7.2 ------------------------------------------------------------
# ===============================================================================================
# Friedman (1991) introduced several benchmark data sets create by simulation. One of these 
# simulations used the following nonlinear equation to create data:

# y=10sin(πx1x2)+20(x3 −0.5)2 +10x4 +5x5 +N(0,σ2)

# where the x values are random variables uniformly distributed between [0, 1] (there are also 5
# other non-informative variables also created in the simulation). The package mlbench contains 
# a function called mlbench.friedman1 that simulates these data:

set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
# We convert the 'x' data from a matrix to a data frame. One reason is that this will give the 
# columns names.
trainingData$x <- data.frame(trainingData$x)

# Look at the data using:
featurePlot(trainingData$x, trainingData$y)
# ... you can, however, use other methods.

# This creates a list with a vector 'y' and a matrix of predictors 'x'. Also simulate a large 
# test set to estimate the true error rate with good precision:
testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)

# Now tune several models on these data. For example, first is a KNN model:
set.seed(100)
knnModel <- train(x = trainingData$x, y = trainingData$y,
                  method = "knn",
                  preProcess = c("center", "scale"),
                  tuneLength = 10,
                  trControl = trainControl(method = "cv"))
knnModel

knnPred <- predict(knnModel$finalModel, testData$x)

plot(knnPred, testData$y)
plot(resid(knnModel), predict(knnModel))

# The function 'postResample' can be used to get the test set perforamnce values:
knnSummaryStats <- postResample(pred = knnPred, obs = testData$y)
knnSummaryStats

# MARS --------------------------------------------------------------------
# Basic fit:
marsFit <- earth(trainingData$x, trainingData$y)
summary(marsFit)

plotmo(marsFit)

# Tuning the model:
# Remember... there are two tuning parameters associated with the MARS model: the degree of the 
# features that are added to the model and the number of retained terms.

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:12)

set.seed(100)
marsTune <- train(trainingData$x, trainingData$y,
                  method = "earth",
                  tuneGrid = marsGrid,
                  trControl = trainControl(method = "cv"))
marsTune
varImp(marsTune)
# MARS does select the informative predictors...

marsPred <- predict(marsTune, testData$x)
plot(marsPred, testData$y, 
     asp = .28,
     xlab = "Predicted",
     ylab = "Observed")
abline(0, 1, 
       col = "red")             

plot(resid(marsTune), predict(marsTune))

marsSummaryStats <- postResample(pred = marsPred, obs = testData$y)
marsSummaryStats

# SVM ---------------------------------------------------------------------
set.seed(100)
svmTune = train(trainingData$x, y = trainingData$y,
                method = "svmRadial",
                preProcess = c("center", "scale"),
                trControl = trainControl(method = "cv", repeats = 10))
svmTune$finalModel

svmPred <- predict(svmTune$finalModel, testData$x)
plot(svmPred, testData$y)

svmSummaryStats <- postResample(pred = svmPred, obs = testData$y)
svmSummaryStats

# Comparing the different models ------------------------------------------
SummaryStats <- list(SVM = svmSummaryStats, MARS = marsSummaryStats, KNN = knnSummaryStats)
SummaryStats

# Comparing the three cross-validated models based on their quantiled cross-validation 
# statistics:
resampStats <- resamples(list(SVM = svmTune, kNN = knnModel, MARS = marsTune))
summary(resampStats)

# ===============================================================================================
# Exercise 7.3 ------------------------------------------------------------
# ===============================================================================================
# For the Tecator data described in the last chapter, build SVM, neural network, MARS, and KNN 
# models. Since neural networks are especially sensitive to highly correlated predictors, does
# pre-processing using PCA help the model? Here I will not use Neural Nets as I do not 
# understand them well enough.

data(tecator)
?tecator

dim(absorp)
dim(endpoints)

# Split training/test set & tControl -------------------------------------------------
is.data.frame(absorp)
absorpData <- as.data.frame(absorp)
Yfat <- endpoints[, 2]

trainingRows <- createDataPartition(Yfat, p = .75, list = FALSE)
dim(trainingRows)

trainingXData <- absorpData[trainingRows, ]
trainingYData <- Yfat[trainingRows]

testingXData <- absorpData[-trainingRows, ]
testingYData <- Yfat[-trainingRows]

str(trainingXData[1:5])
str(testingXData[1:5])

str(trainingYData)
str(testingYData)

# SVM Radial Kernel ---------------------------------------------------------------------
set.seed(1)
svmTune <- train(trainingXData, trainingYData,
                 method = "svmRadial",
                 preProcess = c("center", "scale"),
                 tuneLength = 10,
                 trControl = trainControl(method = "cv"))
svmTune
svmTune$finalModel

svmPred <- predict(svmTune, testingXData)

plot(svmPred, testingYData)
abline(0, 1, 
       col = "red")

svmStats <- postResample(pred = svmPred, obs = testingYData)
svmStats

# MARS --------------------------------------------------------------------
# Remember... there are two tuning parameters associated with the MARS model: the degree of the 
# features that are added to the model and the number of retained terms.
marsGrid <- expand.grid(.degree = 1:3, .nprune = 2:15)

set.seed(1)
marsTune <- train(trainingXData, trainingYData,
                  method = "earth", 
                  tuneGrid = marsGrid,
                  trControl = trainControl(method = "cv"))
# Inspect:
marsTune$bestTune
varImp(marsTune)
plot(marsTune)

# Test:
marsPred <- predict(marsTune$finalModel, testingXData)

# Visualise tests:
plot(marsPred, testingYData,
     xlab = "Predicted",
     ylab = "Observed",
     main = "Predicted vs Observed of MARS model")
abline(0, 1,
       col = "red",
       lty = "dashed")

plot(resid(marsTune$finalModel), predict(marsTune$finalModel))

# Summary statistics:
marsSummStats <- postResample(pred = marsPred, obs = testingYData)
marsSummStats

# kNN Model ---------------------------------------------------------------
nearZeroVar(trainingXData)

set.seed(1)
knnTune <- train(trainingXData, trainingYData,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 20,
                 trControl = trainControl(method = "cv"))
knnTune
plot(knnTune)

knnPred <- predict(knnTune, testingXData)

plot(knnPred, testingYData,
     xlab = "Predicted",
     ylab = "Observed",
     main = "Observed verse Predicted for kNN model")
abline(0, 1, 
       col = "red")

plot(resid(knnTune), predict(knnTune),
     xlab = "Residuals",
     ylab = "Predicted")

knnSummStats <- postResample(pred = knnPred, obs = testingYData)
knnSummStats

# Comparing Models: -------------------------------------------------------
svmStats
marsSummStats
knnSummStats

# Cross-validation averages:
resampModStats <- resamples(list(SVM = svmTune, kNN = knnTune, MARS = marsTune))
summary(resampModStats)

# It seems the MARS model technique has the highest predictive ability between the three models.

# ===============================================================================================
# Exercise 7.4 ------------------------------------------------------------
# ===============================================================================================
# Return to the permeability problem outlined in Exercise 6.2. Train several nonlinear regression
# models and evaluate the resampling and test set performance.
data("permeability")
dim(permeability)
dim(fingerprints)

is.data.frame(permeability)
permeabilityData <- as.data.frame(permeability)
is.data.frame(permeabilityData)

# Here, I will train using SVM, kNN, and MARS modelling techniques. Again, I refrain from using
# neural nets due to only being able to superfificually understand the modelling process.

# Partitioning the Data ---------------------------------------------------

# Generate a random seed setting:
initial_seed <- as.integer(Sys.time())
print(initial_seed)
# ... take the last 5 digits:
theSeed <- initial_seed %% 100000
print(theSeed) # 50751

trainingRows <- createDataPartition(permeability, p = .8, list = FALSE)

trainingXdata <- fingerprints[trainingRows, ]
trainingYdata <- permeabilityData[trainingRows]

testingXdata <- fingerprints[-trainingRows, ]
testingYdata <- permeabilityData[-trainingRows]

# Check proportions of partitioned sets:
str(trainingXdata)
str(trainingYdata)

str(testingXdata)
str(testingYdata)

# Set trainControl for cross-validation:
tCtrl <- trainControl(method = "cv", number = 10)

# kNN  --------------------------------------------------------------------
# Remove unbalanced and noisy predictors:
kNNXfiltered <- trainingXdata[, -nearZeroVar(trainingXdata)]

set.seed(50751)
knnTune <- train(kNNXfiltered, trainingYdata,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneGrid = data.frame(.k = 1:20),
                 trControl = tCtrl)
# Inspect:
knnTune
plot(knnTune)
# The optimal number of neighbours is indicated to be k = 9.

# Test:
knnPred <- predict(knnTune$finalModel, testingXdata[, -nearZeroVar(trainingXdata)])

plot(knnPred, testingYdata,
     xlab = "Predicted",
     ylab = "Observed",
     main = "kNN Model: Predicted vs. Observed")
abline(0, 1,
       col = "red")

# Plot residuals:
plot(resid(knnTune), predict(knnTune),
     xlab = "Residuals",
     ylab = "Predicted")
abline(2, -1,
       col = "red")
# Some concern re slight presence of patterns in the residual plot output.

# Summary statistics:
kNNSummaryStats <- postResample(pred = knnPred, obs = testingYdata)
kNNSummaryStats

# SVM ---------------------------------------------------------------------
# As above, filter noisy, zero variance predictors:
svmXfiltered <- trainingXdata[, -nearZeroVar(trainingXdata)]

set.seed(50751)
# Note: initially implemented using a linear kernal as per rule of thumb for Occam'z razor, 
# however, radial kernel found to have significantly increased performance.
svmTune <- train(svmXfiltered, trainingYdata, 
                 method = "svmRadial",
                 preProcess = c("center", "scale"),
                 # setting Cost values:
                 tuneLength = 20,
                 trControl = tCtrl)
# Inspect:
svmTune
svmTune$finalModel

# Test:
svmPred <- predict(svmTune$finalModel, testingXdata[, -nearZeroVar(trainingXdata)])

plot(svmPred, testingYdata)
abline(0, 1,
       col = "red")

# Summary statistics:
svmSummaryStats <- postResample(pred = svmPred, obs = testingYdata)
svmSummaryStats

# MARS --------------------------------------------------------------------
# First fit a standard mars model using earth in order to get a rough idea of two tuning
# parameters associated with the MARS model: the degree of the features that are added to the 
# model and the number of retained terms.
marsXfiltered <- trainingXdata[, -nearZeroVar(trainingXdata)]

set.seed(50751)
marsFit <- earth(marsXfiltered, trainingYdata)
summary(marsFit)

plotmo(marsFit)

# Create tuning Grid.
marsGrid <- expand.grid(.degree = 1:21, .nprune = 2:21)

set.seed(50751)
marsTune <- train(marsXfiltered, trainingYdata, 
                  method = "earth",
                  tuneGrid = marsGrid,
                  preProcess = c("center", "scale"),
                  trControl = tCtrl)
beepr::beep(sound = 2)

# Inspect:
marsTune
marsTune$bestTune
# The final values used for the model were nprune = 3 and degree = 2.
summary(marsTune)
varImp(marsTune)

# Test:
marsPred <- predict(marsTune$finalModel, testingXdata[, -nearZeroVar(trainingXdata)])
plot(marsPred, testingYdata)
plot(resid(marsTune), predict(marsTune))

# Summary statistics:
marsSummaryStats <- postResample(pred = marsPred, obs = testingYdata)
marsSummaryStats


# Comparing Model Performance: --------------------------------------------
list(MARS = marsSummaryStats, SVM = svmSummaryStats, kNN = kNNSummaryStats)

# k-fold CV performance statistic averages across all models:
resampleAvgs <- resamples(list(kNN = knnTune, SVM = svmTune, MARS = marsTune))
summary(resampleAvgs)

# The averaged statistics indicate that the SVM model has the most significant predictive 
# ability versus the other two models. For the SVM model:

# Mean Rsqaured = 60.81%
# Mean RMSE = 10.20 units
# Mean MAE = 7.24 units

# b) Do any of the nonlinear models outperform the optimal linear model you previously developed
# in Exercise 6.2? If so, what might this tell you about the underlying relationship between the
# predictors and the response?

# Yes. The SVM slighlty outperforms all models. This indicates that there are several 
# significant outliers and possible significant leverage variables. Thus, an SVM model is able
# to trade for an increase in bias with a resulting decrease in the structural variance.

# c) Would you recommend any of the models you have developed to replace the permeability 
# laboratory experiment?

# Yes. The SVM model would be a more predictive model for the permeability experiment compared 
# to all other models, albeit slight.

# ===============================================================================================
# Exercise 7.5 ------------------------------------------------------------
# ===============================================================================================
# Exercise 6.3 describes data for a chemical manufacturing process. Use the same data imputation, 
# data splitting, and pre-processing steps as before and train several nonlinear regression 
# models.
data("ChemicalManufacturingProcess")
chemicalProcessData <- as.data.frame(impute(x = ChemicalManufacturingProcess, what = "median"))
# Generate wrangle sets: 
chemicalXData <- chemicalProcessData[, -1]
chemicalYData <- chemicalProcessData$Yield

# Generate Train/Test Partition ---------------------------------------------
trainingRows <- createDataPartition(chemicalYData, p = .8, list = FALSE)

trainingXData <- chemicalXData[trainingRows, ]
testingXData <- chemicalXData[-trainingRows, ]

trainingYData <- chemicalYData[trainingRows]
testingYData <- chemicalYData[-trainingRows]

# Check data proportions:
str(trainingXData[1:5])
str(trainingYData)

str(testingXData[1:5])
str(testingYData)

# Generate reproducible seed setting:
# initial_seed <- as.integer(Sys.time())
# print(initial_seed)
# ... take the last 2 digits:
# theSeed <- initial_seed %% 100
# theSeed # 57

# Set train Control:
tCtrl <- trainControl(method = "cv", number = 10)

# I will train a kNN, SVM, and MARS model; again, excluding a neural net on the basis of my own
# limited understanding and use of such modelling methods.

# kNN ---------------------------------------------------------------------
# Filter noisy predictors:
knnXDataFiltered <- trainingXData[, -nearZeroVar(trainingXData)]

set.seed(57)
knnTune <- train(y = trainingYData, x = knnXDataFiltered,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 20,
                 trControl = tCtrl)
# Inspect
knnTune
plot(knnTune)
# Optimal k = 7.

# Test:
knnPred <- predict(knnTune, testingXData[, -nearZeroVar(testingXData)])

plot(knnPred, testingYData,
     xlab = "Predicted",
     ylab = "Observed",
     main = "kNN-Model: Predicted vs Observed")
abline(0, 1,
       lty = "dashed",
       col = "red")

plot(resid(knnTune), predict(knnTune),
     xlab = "Residuals",
     ylab = "Predicted",
     main = "kNN-Model: Residuals vs Predicted plot")

# Summary statistics:
knnSummaryStats <- postResample(pred = knnPred, obs = testingYData)
knnSummaryStats


# SVM ---------------------------------------------------------------------

set.seed(57)
# Note: initially implemented using a linear kernal as per rule of thumb for Occam's razor, 
# however, radial kernel found to have significantly increased performance.
svmTune <- train(x = trainingXData[, -nearZeroVar(trainingXData)], y = trainingYData,
                 method = "svmRadial",
                 preProcess = c("center", "scale"),
                 # Setting Cost values:
                 tuneLength = 20,
                 trControl = tCtrl)
# Inspect:
svmTune
svmTune$finalModel

# Test:
svmPred <- predict(svmTune, testingXData[, -nearZeroVar(testingXData)])

plot(svmPred, testingYData,
     xlab = "Predicted",
     ylab = "Observed",
     main = "SVM-Model:Predicted vs. Observed")
abline(0, 1,
       lty = "dashed",
       col = "red")

# Summary Stats:
svmSummaryStats <- postResample(pred = svmPred, obs = testingYData)
svmSummaryStats

# MARS --------------------------------------------------------------------
# Basic MARS model to get rough idea of ideal degree of features and the number of retained
# terms.
set.seed(57)
marsFit <- earth(trainingXData[, -nearZeroVar(trainingXData)], trainingYData)
marsFit
marsFit$selected.terms
# degree = 1:15;
# terms = 1:22
plotmo(marsFit)

# Generate tuningGrid:
marsGrid <- expand.grid(.degree = 1:15, .nrpune = 2:22)

set.seed(57)
marsTune <- train(x = trainingXData[, -nearZeroVar(trainingXData)], y = trainingYData,
                  method = "earth",
                  preProcess = c("center", "scale"),
                  trControl = tCtrl)
# Inspect:
summary(marsTune)
varImp(marsTune)
# The final values used for the model were a selection of 10 terms, and 1:9 degrees of 
# interaction.

# Test:
marsPred <- predict(marsTune, testingXData[, -nearZeroVar(testingXData)])

plot(marsPred, testingYData,
     xlab = "Predicted",
     ylab = "Observed",
     main = "MARS-Model:Predicted vs. Observed")

plot(resid(marsTune$finalModel), predict(marsTune$finalModel),
     xlab = "Residuals",
     ylab = "Predicted",
     main = "MARS-Model:Residuals of Predicted")

# Summary Stats:
marsSummaryStats <- postResample(pred = marsPred, obs = testingYData)
marsSummaryStats

# Comparing Non-Linear Models on the Chemical Process Data ----------------
# Final model summary stats:
list(kNN = knnSummaryStats, SVM = svmSummaryStats, MARS = marsSummaryStats)

# Cross-validation averages across models:
resampleAvgs <- resamples(list(kNN = knnTune, SVM = svmTune, MARS = marsTune))
summary(resampleAvgs)

# Summary stats inidcate SVM to provide the highest predictive ability for Yield, out of all 
# implemented models.

# End file ----------------------------------------------------------------
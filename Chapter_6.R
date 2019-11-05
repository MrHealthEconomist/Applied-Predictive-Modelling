# ============================================================================================
# Load Libraries ----------------------------------------------------------
# ============================================================================================
# The R packages elasticnet, caret, lars, MASS, pls and stats will be referenced. 
library(elasticnet)
library(caret)
library(lars)
library(MASS)
library(pls)
library(stats)
library(beepr)
library(tidyverse)
library(corrplot)
library(e1071)
library(AppliedPredictiveModeling)

# ============================================================================================
# Wrangling the Solubility Data -------------------------------------------
# ============================================================================================
# The solubility data can be obtained from the AppliedPredictiveModeling R package.
# The predictors for the training and test sets are contained in data frames called solTrainX 
# and solTestX, respectively.
data(solubility)

# The data objects begin with "sol":
ls(pattern = "^solT")

# Each column of the data corresponds to a predictor (i.e., chemical descriptor) and the rows 
# correspond to compounds. There are 228 columns in the data. A random sample of column names
# is:
set.seed(2)
sample(names(solTrainX), 8)
# The “FP” columns correspond to the binary 0/1 fingerprint predictors that are associated with
# the presence or absence of a particular chemical structure. Alternate versions of these data 
# that have been Box–Cox transformed are contained in the data frames solTrainXtrans and 
# solTestXtrans. These modified versions were used in the analyses in this and subsequent 
# chapters.

# The solubility values for each compound are contained in numeric vectors named solTrainY and 
# solTestY.

# ============================================================================================
# Ordinary Least Squares Regression ---------------------------------------
# ============================================================================================
# The primary function for creating linear regression models using simple least squares is lm. 
# This function takes a formula and data frame as input. Because of this, the training set 
# predictors and outcome should be contained in the same data frame. We can create a new data 
# frame for this purpose:
trainingData <- solTrainXtrans
# Add the solubility outcome:
trainingData$Solubility <- solTrainY
# To fit a linear model with all the predictors entering in the model as simple, independent 
# linear terms, the formula shortcut Solubility ∼ . can be used:
lmFit_AllPred <- lm(Solubility ~ ., data = trainingData)

# An intercept term is automatically added to the model. The summary method displays model 
# summary statistics, the parameter estimates, their standard errors, and p-values for testing 
# whether each individual coefficient is different than 0:
summary(lmFit_AllPred)
# The simple estimates of the RMSE and R2 were 0.55 and 0.945, respec- tively. Note that these 
# values are likely to be highly optimistic as they have been derived by re-predicting the 
# training set data.

# To compute the model solubility values for new samples, the predict method is used:
lmPred_1 <- predict(lmFit_AllPred, solTestXtrans)
head(lmPred_1)

plot(trainingData$Solubility, predict(lmFit_AllPred))
plot(predict(lmFit_AllPred), residuals(lmFit_AllPred))

# We can collect the observed and predicted values into a data frame, then use the caret 
# function defaultSummary to estimate the test set performance:
lmFit_Values_1 <- data.frame(obs = solTestY, pred = lmPred_1)
defaultSummary(lmFit_Values_1)
# Based on the test set, the summaries produced by the summary function for lm were 
# optimistic.

# If we wanted a robust linear regression model, then the robust linear model function (rlm) 
# from the MASS package could be used, which by default employs the Huber approach. Similar 
# to the lm function, rlm is called as follows:
rlmFit_Allpred <- rlm(Solubility ~ ., data = trainingData)

# The train function generates a resampling estimate of performance. Because the training set 
# size is not small, 10-fold cross-validation should produce reasonable estimates of model 
# performance. The function trainControl specifies the type of resampling:
tCtrl <- trainControl(method = "cv", number = 10)
# train will accept a model formula or a non-formula interface. The non-formula interface is:
set.seed(100)
lmFit_1 <- train(x = solTrainXtrans, y = solTrainY, method = "lm", trControl = tCtrl)
# The random number seed is set prior to modeling so that the results can be reproduced. The
# results are:
lmFit_1
# For models built to explain, it is important to check model assumptions, such as the 
# residual distribution. For predictive models, some of the same diagnostic techniques can 
# shed light on areas where the model is not predicting well.

# For example, we could plot the residuals versus the predicted values for the model. If the 
# plot shows a random cloud of points, we will feel more comfortable that there are no major 
# terms missing from the model (such as quadratic terms, etc.) or significant outliers. 
# Another important plot is the predicted values versus the observed values to assess how 
# close the predictions are to the actual values. Two methods of doing this (using the 
# training set samples are:
xyplot(solTrainY ~ predict(lmFit_1),
       # plot the points (type = "p") and a background grid ("g")
       type = c("c", "g"),
       xlab = "Predicted",
       ylab = "Observed")

xyplot(resid(lmFit_1) ~ predict(lmFit_1),
       type = c("p", "g"),
       xlab = "Predicted",
       ylab = "Observed")       
# Note that the resid function generates the model residuals for the training set and that
# using the predict function without an additional data argument returns the predicted values
# for the training set. For this model, there are no obvious warning signs in the diagnostic 
# plots.

# To build a smaller model without predictors with extremely high correlations, we can methods 
# to identify correlations above a chosen threshold in order to o reduce the number of 
# predictors such that there are no absolute pairwise correlations above 0.9:
corThresh <- .9
tooHigh <- findCorrelation(cor(solTrainXtrans), corThresh)
corrPred <- names(solTrainXtrans)[tooHigh]
trainXfiltered <- solTrainXtrans[, - tooHigh]
testXfiltered <- solTestXtrans[, - tooHigh]

set.seed(100)
lmFiltered <- train(solTrainXtrans, solTrainY, method = "lm",
                    trControl = tCtrl)

plot(trainingData$Solubility, predict(lmFiltered))
abline(0, 1, col = "red")
plot(predict(lmFiltered), resid(lmFiltered))


# Robust linear regression can also be performed using the train function which employs the 
# rlm function. However, it is important to note that rlm does not allow the covariance matrix
# of the predictors to be singular (unlike the lm function). To ensure that predictors are not
# singular, we will pre-process the predictors using PCA. Using the filtered set of 
# predictors, the robust regression model performance is:
set.seed(100)
rlmPCA <- train(solTrainXtrans, solTrainY, 
                method = "rlm",
                preProcess = "pca",
                trControl = tCtrl)
rlmPCA

# ============================================================================================
# Partial Least Squares ---------------------------------------------------
# ============================================================================================
#  By default, the pls package uses the first Dayal and MacGregor kernel algorithm while the 
# other algorithms can be specified using the method argument using the values "oscorespls", 
# "simpls", or "widekernelpls". The plsr function, like the lm function, requires a model 
# formula:
plsFit <- plsr(Solubility ~ ., data = trainingData)
# The number of components can be fixed using the ncomp argument or, if left to the default, 
# the maximum number of components will be calculated. Predictions on new samples can be 
# calculated using the predict function. Predictions can be made for a specific number of 
# components or for several values at a time. For example:
predict(plsFit, solTestXtrans[1:5, ], ncomp = 1:2)

# The plsr function has options for either K-fold or leave-one-out cross-validation (via the
# validation argument) or the PLS algorithm to use, such as SIMPLS (using the method argument).
# There are several helper functions to extract the PLS components (in the function loadings),
# the PLS scores (scores), and other quantities. The plot function has visualizations for many
# aspects of the model.

# train can also be used with method values of pls, such as "oscorespls", "simpls", or
# "widekernelpls". For example:
set.seed(100)
plsTune <- train(solTrainXtrans, solTrainY,
                 method = "pls",
                 # The default tuning grid evaluates the components 1... tuneLength
                 tuneLength = 20,
                 trControl = tCtrl,
                 preProcess = c("center", "scale"))

# Let's compare the tuning component results to standard PCR:
pcrTune <- train(solTrainXtrans, solTrainY, 
                 method = "pcr",
                 tuneLength = 20,
                 trControl = tCtrl,
                 preProcess = c("center", "scale"))

par(new = TRUE)
plot(plsTune, col = "red")
plot(pcrTune, add = TRUE)

# ============================================================================================
# Penalised Regression Models ---------------------------------------------
# ============================================================================================
# Ridge-regression models can be created using the lm.ridge function in the MASS package or
# the enet function in the elasticnet package. When calling the enet function, the lambda 
# argument specifies the ridge-regression penalty:
ridgeMod <- enet(x = as.matrix(solTrainXtrans), y = solTrainY,
                 lambda = .001)
# Recall that the elastic net model has both ridge penalties and lasso penalties and, at this 
# point, the R object ridgeModel has only fixed the ridge penalty value. The lasso penalty 
# can be computed efficiently for many values of the penalty. The predict function for enet 
# objects generates predictions for one or more values of the lasso penalty simultaneously 
# using the s and mode arguments. For ridge regression, we only desire a single lasso penalty
# of 0, so we want the full solution. To produce a ridge-regression solution we define s = 1 
# with mode = "fraction". This last option specifies how the amount of penalization is 
# defined; in this case, a value of 1 corresponds to a faction of 1, i.e., the full 
# solution:
ridgePred <- predict(ridgeMod, newx = as.matrix(solTestXtrans), 
                     s = 1, mode = "fraction",
                     type = "fit")
head(ridgePred$fit)

# To tune over the penalty, train can be used with a different method. First, define the set 
# of values,
ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))

set.seed(100)
ridgeRegFit <- train(solTrainXtrans, solTrainY, 
                     method = "ridge",
                     # then fit the model over many penalty values
                     tuneGrid = ridgeGrid,
                     trControl = tCtrl,
                     # then put the predictors on the same scale
                     preProcess = c("center", "scale"))
beep(sound = 6)

ridgeRegFit
# As the model indicates: we see that a lambda value of .02857143 minimises the RMSE (the 
# standard deviation (error) of the residuals) between observed and predicted values. Remember
# that the MSE is the average variance of the model's residuals. MAE is the average distance 
# between the predicted hyperplane and the observed point. Therefore, it is the average error
# in the model between the predicted and the observed values.  At the validated lambda value,
# MAE is minimised to be .5251866.  The coefficicent of determination, Rsquared, can be 
# thought of as the total area explained by the model, or interpreted as the proportion of
# information in the data that is explained by the model; this is maximised for the given 
# lambda value to be .8878666, or 88.79% of the information contained in the given data is 
# explained by the model.

# The lasso model can be estimated using a number of different functions. The lars package 
# contains the lars function, the elasticnet package has enet, and the glmnet package has a 
# function of the same name. The syntax for these functions is very similar. For the enet 
# function, the usage would be:
enetModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY,
                  lambda = .01, normalize = TRUE)
# The predictor data must be a matrix object, so the data frame solTrainXtrans needs to be
# converted for the enet function. The predictors should be centered and scaled prior to
# modeling. The normalize argument will do this standardization automatically. The parameter 
# lambda controls the ridge-regression penalty and, setting this value to 0, fits the lasso
# model. The lasso penalty does not need to be specified until the time of prediction:
enetPred <- predict(enetModel, newx = as.matrix(solTestXtrans),
                    s = .1, mode = "fraction",
                    type = "fit")
# A lsit is returned with several items:
names(enetPred)
# Here, the "fit" component has the predicted values:
head(enetPred$fit)
# To determine which predictors are used in the model, the predict method is used with 
# type = "coefficients":
enetCoef <- predict(enetModel, newx = as.matrix(solTestXtrans),
                    s = .1, mode = "fraction",
                    type = "coefficients")
tail(enetCoef$coefficients)
# More than one value of s can be used with the predict function to generate predictions from
# more than one model simultaneously.

# Other packages to fit the lasso model or some alternate version of the model are biglars 
# (for large data sets), FLLat (for the fused lasso), grplasso (the group lasso), penalized, 
# relaxo (the relaxed lasso), and others.

# To tune the elastic net model using train, we specify method = "enet". Here, we tune the 
# model over a custom set of penalties:
enetGrid <- expand.grid(.lambda = c(0, .01, .1),
                        .fraction = seq(.05, 1, length.out = 20))

set.seed(100)
enetTune <- train(solTrainXtrans, solTrainY,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = tCtrl,
                  preProcess = c("center", "scale"))
beep(sound = 6)
enetTune
plot(enetTune)

# ============================================================================================
# Exercise 6.1 ---------------------------------------------------------------
# ============================================================================================
# 6.1) 
# A Tecator Infratec Food and Feed Analyzer instrument was used to analyze 215 samples of meat
# across 100 frequencies. In addition to an IR profile, analytical chemistry determined the 
# percent content of water, fat, and protein for each sample. If we can establish a predictive
# relationship between IR spectrum and fat content, then food scientists could predict a 
# sample’s fat content with IR instead of using analytical chemistry. This would provide 
# costs savings, since analytical chemistry is a more expensive, time-consuming process.

# a) Load data:
data(tecator)
?tecator
# The matrix absorp contains the 100 absorbance values for the 215 samples, while matrix 
# endpoints contains the percent of moisture, fat, and protein in columns 1–3, respectively.
dim(absorp)
dim(endpoints)

# b) In this example the predictors are the measurements at the individual frequencies. 
# Because the frequencies lie in a systematic order (850–1,050 nm), the predictors have a
# high degree of correlation. Hence, the data lie in a smaller dimension than the total number
# of predictors (215). Use PCA to determine the effective dimension of these data. What is 
# the effective dimension?
set.seed(1)
pcaAbsorp <- prcomp(absorp, 
                    center = TRUE, scale. = TRUE)
# The variance explained by each principal component is obtained by squaring these:
pca_var <- pcaAbsorp$sdev ^ 2
# # To compute the proportion of variance explained by each principal component, we simply 
# divide the variance explained by each principal component by the total variance explained 
# by all of the principal components:
pve <- pca_var / sum(pca_var)
# # We see that the first principal component explains 98% of the variance in the data, the 
# next principal component explains 0.009% of the variance, and so forth. 

# We can plot the PVE explained by each component, as well as the cumulative PVE, as follows:
par(mfrow = c(1, 2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained",
     ylim = c(0, 1))
plot(cumsum(pve), xlab = "Principle Component", ylab = "Cum % of Variance Explained",
     ylim = c(0, 1))
# It seems that the effective dimension would be to use the first three predictor variables.

# c) Split the data into a training and a test set, pre-process the data, and build each 
# variety of models described in this chapter. For those models with tuning parameters, what 
# are the optimal values of the tuning parameter(s)?

# First combine and create the matrix with the response and predictors:
is.data.frame(absorp)
# Need to convert absorp to a data frame:
predictors <- as.data.frame(absorp)
is.data.frame(predictors)

# Only take the fat[, 2] column as that is the outcome of interest
outcome <- endpoints[, 2]
outcome

# Before creating a partition, set the seed to make it reproducible:
set.seed(1)

trainingRows <- createDataPartition(outcome, p = .8, list = FALSE)
head(trainingRows)

# Now we subset the data into objects for training, using integer subsetting:
trainingData <- predictors[trainingRows, ]
trainingData$Fat <- outcome[trainingRows]

# Now we can do the same for the test set, specifying an omition of the training data using 
# the negative sign in th subsetting process:
testData <- predictors[-trainingRows, ]
testData$Fat <- outcome[-trainingRows]

str(trainingData[1:5])
str(testData[1:5])

# Setup up trainControl function prior to modeling.
tCtrl <- trainControl(method = "cv", number = 10)

# OLS Regression ----------------------------------------------------------
# Explore the data to see if transformations are needed when using the linear model:
dev.off()

FAT <- trainingData$Fat
multiplot <- function(x) {
 for (i in seq_along(x)) {
  out <- plot(FAT, x[, i])
  }
}

multiplot(trainingData)

# So from the plots it does seem that the data will need to be transformed. However, the above
# models will be run through as on the solubility data.

# OLS Regression ----------------------------------------------------------

# ... using the lm() function:
lmFit <- lm(Fat ~ ., data = trainingData)
summary(lmFit)

# Assessing RMSE of the model for both training and test sets:
lmTrainPred <- predict(lmFit, trainingData)
lmTestPred <- predict(lmFit, testData)

RMSE(trainingData$Fat, lmTrainPred)
RMSE(testData$Fat, lmTestPred)

MSE <- sum((lmFit$residuals)^2)
MSE

# Plot the training results for the model:
par(mfrow = c(1, 2))
plot(trainingData$Fat, predict(lmFit),
     xlab = "Observed",
     ylab = "Predicted")
abline(0, 1, col = "red")

plot(predict(lmFit), resid(lmFit))
plot(lmFit)

# Although there are no obvious assumptions broken in this lm model, let us try reduce the
# dimension complexity. The afore PCA indicated that the first 3 variables capture
# the majority of information. Let's see what happens when only the first three are used!

lmFit_pca <- lm(Fat ~ V1 + V2 + V3, data = trainingData)
summary(lmFit_pca)

# Assessing RMSE of the model for both training and test sets:
#  ... remember that the RMSE is basically the squared deviation of the model's residuals.
lmTrainPred_pca <- predict(lmFit_pca, trainingData)
lmTestPred_pca <- predict(lmFit_pca, testData)

trainingRMSE <- RMSE(trainingData$Fat, lmTrainPred_pca)
trainingRMSE

testingRMSE <- RMSE(testData$Fat, lmTestPred_pca)
testingRMSE

# MSE/Variance of the residuals:
MSE_pca <- sum((lmFit_pca$residuals) ^ 2)
MSE_pca

plot(trainingData$Fat, predict(lmFit_pca),
     xlab = "Observed",
     ylab = "Predicted")
abline(0, 1, col = "red")

plot(predict(lmFit_pca), resid(lmFit_pca),
     xlab = "Predicted",
     ylab = "Residuals")

# Using the default function defaultSummary in caret to estimate the test set 
# performance:
lmpcaFitValues <- data.frame(obs = testData$Fat, pred = lmTestPred_pca)
defaultSummary(lmpcaFitValues)
# ...remember that the mean absolute error is basically the distance between the observed 
# points and the predicted hyperplane. Although the model indicates decreased performance, it is 
# better to have a simple model than a model that overfits the data. However, this model still
# indicates to poorly be able to fit and predict the data.

# The summary statistics indicate this to be a weak model for predicting fat content...

# ... robust linear regression using using the rlm function:
set.seed(1)

rlmFit <- train(Fat ~ ., data = trainingData, 
                method = "rlm",
                preProcess = "pca",
                trControl = tCtrl)
rlmFit$finalModel

plot(rlmFit$finalModel)

xyplot(testData$Fat ~ predict(rlmFit),
       type = c("p", "g"),
       xlab = "Predicted",
       ylab = "Observed",
       col.line = "red")

xyplot(resid(rlmFit) ~ predict(rlmFit),
       type = c("p", "g"),
       xlab = "Predicted",
       ylab = "Observed",
       col.line = "red")
# For this model, the plots clearly show a weak ability to predict the data; the observed versus
# predicted pbservations are highly dispersed.

# robust linear modelling seems to only be able to capture a neglible amount of 
# informatiion from the data. Let's double check the summary statistics:
rlmPred <- predict(rlmFit, testData)

plot(rlmPred, testData$Fat, 
     xlab = "Predicted",
     ylab = "Observed")
abline(0, 1,
       col = "red")

plot(resid(rlmFit), predict(rlmFit))
abline(0, 1,
       col = "red")

rlmPred_Values <- data.frame(obs = testData$Fat, pred = rlmPred)
defaultSummary(rlmPred_Values)
# As indicated by the summary test statistics, rlm is a weak prediction technique for fat 
# content.

# Partial Least Squares ---------------------------------------------------

# For the PLS and lasso functions, seperating the Y and X variables into seperate matrices 
# makes computation more efficient:
set.seed(1)

trainingXdata <- predictors[trainingRows, ]
trainingYdata <- outcome[trainingRows]

testXdata <- predictors[-trainingRows, ]
testYdata <- outcome[-trainingRows]

str(trainingXdata[1:5])
str(testXdata[1:5])

str(trainingYdata)
str(testYdata)

# Tune the model using cross-validation:
set.seed(1)
plsTune <- train(trainingXdata, trainingYdata,
                  method = "pls",
                  tuneLength = 20,
                  trControl = tCtrl,
                  preProcess = c("center", "scale"))
beep(sound = 2)
plsTune
summary(plsTune)
plsTune$bestTune
# The 14th component (linear combination of predictors) is found to maximise the variance 
# explained in the data (Rsquared) and have the best varince/bias trade-off with Rsquared and 
# RMSE.

plot(plsTune)

# Now we predict on test data:
plsPredict <- predict(plsTune, testData)

plot(plsPredict,testData$Fat,
     xlab = "Observed",
     ylab = "Predicted",
     main = "PLS Model: Observed vs Predicted values")
abline(0, 1,
       col = "red")

plsFit_summary <- data.frame(obs = testData$Fat, pred = plsPredict)
defaultSummary(plsFit_summary)
# PLS shows a to be a strong model for predicting fat.

# Lasso Regression model ---------------------------------------------------

# A custom search function grid:
enetGrid <- expand.grid(.lambda = c(0, 0.01, .1),
                       .fraction = seq(.05, 1, length = 20))
set.seed(1)
enetTune <- train(x = trainingXdata, y = trainingYdata,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = tCtrl,
                  preProcess = c("center", "scale"))
beep(sound = 1)

enetTune$bestTune
plot(enetTune)

# CV found a fraction (s) value of .05 and a lambda penalty value of 0 to the best tuning 
# parameters for this model.
enetFit <- enet(x = as.matrix(trainingXdata), y = trainingYdata,
                lambda = 0, normalize = TRUE)
plot(enetFit)

enetPred <- predict(enetFit, newx = as.matrix(testXdata),
                    s = .05, mode = "fraction",
                    type = "fit")

plot(enetPred$fit, testYdata,
     xlab = "Predicted",
     ylab = "Observed",
     main = "Lasso Model Predicted vs Observed")
abline(0, 1, col = "red")

enetFit_summary <- data.frame(obs = testYdata, pred = enetPred$fit)
defaultSummary(enetFit_summary)

# Comparing models ----------------------------------------------

defaultSummary(lmpcaFitValues)
defaultSummary(rlmPred_Values)
defaultSummary(plsFit_summary)
defaultSummary(enetFit_summary)
# To compare the three cross-validated models based on their cross-validation statistics:
resamp <- resamples(list(RLM = rlmFit, PLS = plsTune, LASSO = enetTune))
summary(resamp)

# d) Both the Lasso and the PLS models are competitive models for predicting fat content.

# ============================================================================================
# Exercise 6.2 ---------------------------------------------------------------
# ============================================================================================

# Developing a model to predict permeability ould save significant resources for a
# pharmaceutical company, while at the same time more rapidly identifying molecules that have a 
# sufficient permeability to become a drug:
data("permeability")
# The matrix fingerprints contains the 1,107 binary molecular predictors for the 165 compounds, 
# while permeability contains permeability response.
dim(fingerprints)
dim(permeability)
# b) The fingerprint predictors indicate the presence or absence of substructures of a molecule
# and are often sparse meaning that relatively few of the molecules contain each substructure. 
# Filter out the predictors that have low frequencies using the nearZeroVar function from the 
# caret package. How many predictors are left for modeling?
fingerprintsZeroVar <- nearZeroVar(fingerprints)
# Number of variables left:
length(fingerprintsZeroVar)
fingerprintsData <- as.data.frame(fingerprints[, -fingerprintsZeroVar])

Pbility <- as.data.frame(permeability)

dim(fingerprintsData)
dim(Pbility)

set.seed(1)
trainingRows <- createDataPartition(permeability, p = .8, list = FALSE)

trainingData <- fingerprintsData[trainingRows, ]
trainingData$Pbility <- Pbility[trainingRows]

testData <- fingerprintsData[-trainingRows, ]
testData$Pbility <- Pbility[-trainingRows]

str(trainingData[1:5])
str(testData[1:5])

tCtrl <- trainControl(method = "cv", number = 10)
# Tune the model using cross-validation. Split the data into X matrix and Y outcomes:
set.seed(1)
plsTune <- train(Pbility ~ ., data = trainingData,
                 method = "pls",
                 tuneLength = 20,
                 trControl = tCtrl,
                 preProcess = c("center", "scale"))
summary(plsTune)
plot(plsTune)

#  How many latent variables are optimal and what is the corresponding resampled estimate of R2?
plsTune$results
plsTune$bestTune

# The 14th component (or a 14 linear combination of predictors) is found to maximise the 
# variance explained in the data (Rsquared) and have the best varince/bias trade-off with 
# Rsquared and RMSE. The training Rsquared is 52.39%.

# Now we predict on test data:
plsPredict <- predict(plsTune, testData)

plot(plsPredict,testData$Pbility,
     xlab = "Observed",
     ylab = "Predicted",
     main = "Observed vs Predicted observations for PLS model")
abline(0, 1,
       col = "red")

plsTune_summary <- data.frame(obs = testData$Pbility, pred = plsPredict)
defaultSummary(plsTune_summary)
# What is the test set estimate of R2?
# 28.09%.
# The PLS technique is therefore a poor predictive model for permeability.

# e) Try building other models discussed in this chapter. Do any have better predictive 
# performance?

# OLS Regression ----------------------------------------------------------
lmFit <- lm(Pbility ~ ., data = trainingData)
summary(lmFit)
# Terrible model for permeability.

# Lasso -------------------------------------------------------------------

enetGrid <- expand.grid(.lambda = c(0, .01, .1),
                        .fraction = seq(.05, 1, length.out = 20))

set.seed(1)
enetTune <- train(Pbility ~ ., data = trainingData,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = tCtrl,
                  preProcess = c("center", "scale"),
                  na.action = na.omit)

enetTune$bestTune
plot(enetTune)

enetPred <- predict(enetTune, newdata = testData)

plot(testData$Pbility, enetPred)
abline(0, 1)

enetFit_summary <- data.frame(obs = testData$Pbility, pred = enetPred)
defaultSummary(enetFit_summary)
# Slight improvement over the PLS model. However, still a poor technique for prediction due to
# the binary nature of the predictors.

# (f) Would you recommend any of your models to replace the permeability laboratory experiment?
# Employing a classfication technique would likely lead to improvement in performance for 
# classifying each predictor, such as a MARS modelling technique or Knn.

# ============================================================================================
# Exercise 6.3 ------------------------------------------------------------
# ============================================================================================
# In this problem, the objective is to understand the relationship between biological 
# measurements of the raw materials (predictors), measurements of the manufacturing process
# (predictors), and the response of product yield. Biological predictors cannot be changed but 
# can be used to assess the quality of the raw material before processing. On the other hand, 
# manufacturing process predictors can be changed in the manufacturing process. Improving 
# product yield by 1% will boost revenue by approximately one hundred thousand dollars per 
# batch.
data("ChemicalManufacturingProcess")
# The set contains the 57 predictors (12 describing the input biological material and 45 
# describing the process predictors) for the 176 manufacturing runs. $yield contains the 
# percent yield for each run and is the outcome.

# b) A small percentage of cells in the predictor set contain missing values. Use an imputation 
# function to fill in these missing values:
chemicalProcessData <- as.data.frame(impute(x = ChemicalManufacturingProcess, what = "median"))


# c) Split the data into a training and a test set, pre-process the data, and tune a model of 
# your choice from this chapter. What is the optimal value of the performance metric?
set.seed(1)
trainingRows <- createDataPartition(y = chemicalProcessData$Yield, p = .8, list = FALSE)
trainingRows

set.seed(1)
trainingData <- chemicalProcessData[trainingRows, ]
testingData <- chemicalProcessData[-trainingRows, ]

# Check proportions:
str(trainingData[1:5])
str(testingData[1:5])

# Although summary statistics indicate a valid model, the significance of the predictive ability
# of the model is wanting; the ability to explain the relationship between responses and 
# predictors is weak. Furthermore, there are too many predictors leading to too much noise and
# a rank-deficicient fit.

# PLS model:

# Set up tControl:
tCtrl <- trainControl(method = "cv",
                      number = 10)

# PLS: training and tuning the model
set.seed(1)
plsTune <- train(Yield ~ ., data = trainingData,
                 method = "pls",
                 tuneLength = 20,
                 trControl = tCtrl,
                 preProcess = c("center", "scale", "pca"))
beep(sound = 2)

plot(plsTune)
plot(plsTune$finalModel, line = TRUE)
plot(plsTune$finalModel$fitted.values, plsTune$finalModel$residuals, 
     xlab = "Predicted",
     ylab = "Residuals",
     asp = .2)
# The optimal performance metric is a linear combination of four components of the predictors.

# (d) Predict the response for the test set. What is the value of the performance metric and how
# does this compare with the resampled performance metric on the training set?
plsPred <- predict(plsTune, testingData)
plot(testingData$Yield, plsPred,
     xlab = "Observed",
     ylab = "Predicted")
abline(0, 1, col = "red",
       lty = 'dashed')

plot(plsTune$finalModel$fitted.values, plsTune$finalModel$residuals,
     xlab = "Predicted",
     ylab = "Residuals", 
     asp = .2)

plsFitSummary <- data.frame(obs = testingData$Yield, pred = plsPred)
defaultSummary(plsFitSummary)

# Training set:
# RMSE = 1.437738
# Rsquared = 0.5778942
# MAE = 1.106473

# Test set:
# RMSE = 1.0352332
# Rsquared = 0.6298845
# MAE = 0.7842615

# There is improved performance on the test set compared to the training set.

# (e) Which predictors are most important in the model you have trained? Do either the 
plot(plsTune$finalModel)
plot(plsTune$finalModel$loadings)
  
# Do biological or process predictors dominate the list? 

set.seed(1)
pcaChemProc <- prcomp(chemicalProcessData, 
                      center = TRUE, scale. = TRUE)

pcaVariance <- pcaChemProc$sdev ^ 2
pveChemProc <- pcaVariance / sum(pcaVariance) * 100
pveChemProc

plot(pveChemProc, xlab = "Principal Component", ylab = "Proportion of Variance Explained")
plot(cumsum(pveChemProc), xlab = "Principle Component", ylab = "Cum % of Variance Explained")
# End file ----------------------------------------------------------------
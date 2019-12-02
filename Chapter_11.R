# ====================================================================================
# Load Libraries  ---------------------------------------------------------
# ====================================================================================
library(AppliedPredictiveModeling)
library(caret)
library(klaR)
library(MASS)
library(pROC)
library(randomForest)
library(parallel)

options(mc.cores = detectCores())

# ====================================================================================
# Generating Data and Models for Classification ---------------------------
# ====================================================================================
set.seed(975)
simulatedTrain <- quadBoundaryFunc(500)
simulatedTest <- quadBoundaryFunc(1000)

head(simulatedTrain)
# Random forest and quadratic discriminant models will be fit to the data:
rfModel <- randomForest(class ~ X1 + X2, 
                        data = simulatedTrain, 
                        ntree = 200)
rfModel

qdaModel <- qda(class ~ X1 + X2, 
                data = simulatedTrain)
qdaModel
# The output of the predict function for qda objects includes both the predicted classes
# (in a slot called class) and the associated probabilities are in a matrix called 
# posterior. For the QDA model, predictions will be created for the training and test sets. 
# Later in this section, the training set probabilities will be used in an additional 
# model to calibrate the class probabilities. The calibration will then be applied to the
# test set probabilities:
qdaTrainPred <- predict(qdaModel, simulatedTrain)

names(qdaTrainPred)
head(qdaTrainPred$class)
head(qdaTrainPred$posterior)

qdaTestPred <- predict(qdaModel, simulatedTest)

names(qdaTestPred)
head(qdaTestPred$class)
head(qdaTestPred$posterior)

simulatedTrain$QDAProb <- qdaTrainPred$posterior[, "Class1"]
simulatedTest$QDAProb <- qdaTestPred$posterior[, "Class1"]

# The random forest model requires two calls to the predict function to get the predicted
# classes and the class probabilities:
rfTestPred <- predict(rfModel, simulatedTest, type = "prob")
head(rfTestPred)

simulatedTest$RFprob <- rfTestPred[, "Class1"]
simulatedTest$RFclass <- predict(rfModel, simulatedTest)

# ====================================================================================
# Sensitvity and Specificty -----------------------------------------------
# ====================================================================================
# caret has functions for computing sensitivity and specificity. These functions require 
# the user to indicate the role of each of the classes.

# # Class 1 will be used as the event of interest:
sensitivity(data = simulatedTest$RFclass, 
            reference = simulatedTest$class,
            positive = "Class1")
specificity(data = simulatedTest$RFclass,
            reference = simulatedTest$class,
            negative = "Class2")

# Predictive values can also be computed either by using the prevalence found in the data
# set (46 %) or by using prior judgement:
posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1")
negPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             negative = "Class2")

# Change the prevalence manually:
posPredValue(data = simulatedTest$RFclass,
             reference = simulatedTest$class,
             positive = "Class1",
             prevalence = .9)

# ====================================================================================
# Confusion Matrix  -------------------------------------------------------
# ====================================================================================
# There are several functions in R to create the confusion matrix. The confusionMatrix 
# function in the caret package produces the table and associ- ated statistics:
confusionMatrix(data = simulatedTest$RFclass, 
                reference = simulatedTest$class,
                positive = "Class1")
# There is also an option in this function to manually set the prevalence. If there were
# more than two classes, the sensitivity, specificity, and similar statistics are 
# calculated on a “one-versus-all” basis (e.g., the first class versus a pool of classes
# two and three).

# ====================================================================================
# Receiver Operating Characteristic Curves --------------------------------
# ====================================================================================
# The pROC package can create the curve and derive various statistics. First, an R object
# must be created that contains the relevant information using the pROC function roc. The
# resulting object is then used to generate the ROC curve or calculate the area under the
# curve. For example,
rocCurve <- roc(simulatedTest$class, simulatedTest$RFprob,
                levels = rev(levels(simulatedTest$class)))
auc(rocCurve)
ci.auc(rocCurve)
# Note that R has a number of packages that can compute the ROC curve, including ROCR, 
# caTools, PresenceAbsence, and others.

# We can also use the plot function to produce the ROC curve itself:
plot.roc(rocCurve, legacy.axes = TRUE, add = TRUE)
# By default, the x-axis goes backwards, used the option legacy.axes = TRUE to get 
# 1-spec on the x-axis moving from 0 to 1.

# ====================================================================================
# Lift Charts -------------------------------------------------------------
# ====================================================================================
labs <- c(RFprob = "Random Forest",
          QDAProb = "Quadratic Discriminant Analysis")
liftCurve <- lift(class ~ RFprob + QDAProb, 
                  data = simulatedTest,
                  labels = labs)
liftCurve
# To plot two lift curves, the xyplot function is used to create a lattice plot:
xyplot(liftCurve, 
       auto.key = list(
        columns = 2,
        lines = TRUE,
        points = FALSE))

# ====================================================================================
# Calibrating Probabilities -----------------------------------------------
# ====================================================================================
# Calibration plots as described above are available in the calibration.plot function in
# the PresenceAbsence package and in the caret function calibration. The syntax for the
# calibration function is similar to the lift function:
calCurve <- calibration(class ~ RFprob + QDAProb, 
                        data = simulatedTest)
calCurve

xyplot(calCurve,
       auto.key = list(
        columns = 2))

# An entirely different approach to calibration plots that model the observed event rate
# as a function of the class probabilities can be found in the calibrate.plot function of
# the gbm package.

# To recalibrate the QDA probabilities, a post-processing model is created that models 
# the true outcome as a function of the class probability. To fit a sigmoidal function, a 
# logistic regression model is used  via the glm function in base R. This function is an 
# interface to a broad set of methods called generalized linear models which includes 
# logistic regression. To fit the model, the function requires the family argument to 
# specify the type of outcome data being modeled. Since our outcome is a discrete 
# category, the binomial distribution is selected.

# The glm() function models the probability of the second factor level, so the function 
# relevel() is used to temporarily reverse the factors levels:
sigmoidalCal <- glm(relevel(class, ref = "Class2") ~ QDAProb,
                    data = simulatedTrain, family = "binomial")
coef(summary(sigmoidalCal))
# The corrected probabilities are created by taking the original model and applying the 
# equation

# p = 1 / 1 + exp(- beta0 - beta1xphat)

# with the estimated slope and intercept. In R, the predict function can be used:
sigmoidProbs <- predict(sigmoidalCal,
                        newdata = simulatedTest[, "QDAProb", drop = FALSE],
                        type = "response")
simulatedTest$QDAsigmoid <- sigmoidProbs

# The Bayesian approach for calibration is to treat the training set class probabilities
# to estimate the probabilities Pr[X] and Pr[X|Y = Cl]. . In R, the naive Bayes model 
# function NaiveBayes in the klaR package can be used for the computations:
BayesCal <- NaiveBayes(class ~ QDAProb, data = simulatedTrain,
                       usekernel = TRUE)
# The option usekernel = TRUE allows a flexible function to model the probability 
# distribution of the class probabilities.

# Like qda(), the predict function for this model creates both the classes and the
# probabilities:
BayesProbs <- predict(BayesCal,
                      newdata = simulatedTest[, "QDAProb", drop = FALSE])
simulatedTest$QDABayes <- BayesProbs$posterior[, "Class1"]
# The probability values before and after calibration:
head(simulatedTest[, c(5, 6, 8, 9)])

# These new probabilities are evaluated using another plot:
calCurve_2 <- calibration(class ~ QDAProb + QDABayes + QDAsigmoid,
                          data = simulatedTest)
xyplot(calCurve_2)

# End file ----------------------------------------------------------------
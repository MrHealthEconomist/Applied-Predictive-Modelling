# ============================================================================================
# Libraries ----------------------------------------------------------
# ============================================================================================
# The R packages used in this section are caret, Cubist, gbm, ipred, party, partykit, 
# randomForest, rpart, RWeka.
library(caret)
library(Cubist)
library(gbm)
library(ipred)
library(party)
library(partykit)
library(randomForest)
library(rpart)
library(RWeka)
library(AppliedPredictiveModeling)

# ============================================================================================
# Single Trees ------------------------------------------------------------
# ============================================================================================
# Two widely used implementations for single regression trees in R are rpart and party. The 
# rpart package makes splits based on the CART methodology using the rpart function, whereas 
# the party makes splits based on the conditional inference framework using the ctree function. 
# Both rpart and ctree functions use the formula method:
rpartTree <- rpart(y ~ ., data = trainData)
# or, 
ctreeTree <- ctree(y ~ ., data = trainData)

data("solubility")
# The rpart function has several control parameters that can be accessed through the 
# rpart.control argument. Two that are commonly used in training and that can be accessed 
# through the train function are the complexity parameter (cp) and maximum node depth 
# (maxdepth). To tune an CART tree over the complexity parameter, the method option in the 
# train function should be set to method = "rpart". To tune over maximum depth, the method 
# option should be set to method = "rpart2":
set.seed(100)
rpartTune <- train(solTrainXtrans, solTrainY, 
                   method = "rpart2",
                   tuneLength = 10, 
                   trControl = trainControl(method = "cv"))

# Likewise, the party package has several control parameters that can be accessed through the 
# ctree_control argument. Two of these parameters are commonly used in training: mincriterion 
# and maxdepth. mincriterion defines the statistical criterion that must be met in order to 
# continue splitting; maxdepth is the maximum depth of the tree. To tune a conditional 
# inference tree over mincriterion, the method option in the train function should be set to 
# method = "ctree". To tune over maximum depth, the method option should be set to 
# method = "ctree2".

# To produce such plots for rpart trees, the partykit can be used to first convert the rpart 
# object to a party object and then use the plot function. This can also be generated when 
# using the train function as follows:
treeObject <- as.party(rpartTune$finalModel)
plot(treeObject)

# ============================================================================================
# Model Trees -------------------------------------------------------------
# ============================================================================================
# The main implementation for model trees can be found in the Weka software suite, but the 
# model can be accessed in R using the RWeka package. There are two different interfaces: M5P 
# fits the model tree, while M5Rules uses the rule-based version. In either case, the 
# functions work with formula methods:
m5tree <- M5P(y ~ ., data = trainData)
# or, for rules:
m5rules <- M5Rules(y ~ ., data = trainData)
# For the example used in the chapter discussion, the minimum number of training set points 
# required to create additional splits was raised from the default of 4–10. To do this, the 
# control argument is used:
m5tree <- M5P(y ~ ., data = trainData, control = Weka_control(M = 10))

# The control argument also has options for toggling the use of smoothing and pruning. If the 
# full model tree is used, a visualisation can be created by the plot function on the output 
# from M5P.

# To tune these models, the train function in the caret package has two options: using 
# method = "M5" evaluates model trees and the rule-based versions of the model, as well as the
# use of smoothing and pruning. Figure 8.12 shows the results of evaluating these models from
# the code:
set.seed(100)
m5Tune <- train(solTrainXtrans, solTrainY,
                method = "M5",
                trControl = trainControl(method = "cv"),
                # Use an option for M5() to specify the minimum 
                # number of samples needed to further split the 
                # data to be 10 splits 
                control = Weka_control(M = 10))
# This is then followed by plot(m5Tune). train with method = "M5Rules" evaluates only the 
# rule-based version of the model.

# Note: it seems the RWeka package is having compatibility issues with the latest R and MAC 
# updates...

# ============================================================================================
# Bagged Trees ------------------------------------------------------------
# ============================================================================================
# The ipred package contains two functions for bagged trees: bagging uses the formula 
# interface and ipredbagg has the non-formula interface:
baggedTree <- ipredbagg(solTrainY, solTrainXtrans)

# or,
baggedTree_2 <- bagging(y ~ ., data = trainData)
# The function uses the rpart function and details about the type of tree can be specified by 
# passing rpart.control to the control argument for bagging and ipredbagg. By default, the 
# largest possible tree is created.

# The caret package has a general framework for bagging many model types, including trees, 
# called bag. Conditional inference trees can also be bagged using the cforest function in 
# the party package if the argument mtry is equal to the number of predictors:
bagCtrl <- cforest_control(mtry = ncol(trainData) - 1)
# The mtry parameter should be the number of predictors (i.e. the number of columns minus
# the outcome).
baggedTree <- cforest(y ~ ., data = trainData, control = bagCtrl)

# ============================================================================================
# Random Forest -----------------------------------------------------------
# ============================================================================================
# The primary implementation for random forest comes from the package with the same name, i.e.
# the randomforest package.
rfMod <- randomForest(solTrainXtrans, solTrainY)
# or,
rfMod2 <- randomForest(y ~ ., data = trainData)

# The two main arguments are mtry for the number of predictors that are randomly sampled as 
# candidates for each split and ntree for the number of bootstrap samples. The default for 
# mtry in regression is the number of predictors divided by 3. The number of trees should be
# large enough to provide a stable, reproducible results. Although the default is 500, at 
# least 1,000 bootstrap samples should be used (and perhaps more depending on the number of 
# predictors and the values of mtry). Another important option is importance; by default, 
# variable importance scores are not computed as they are time consuming; importance = TRUE 
# will generate these values:
rfModel <- randomForest(solTrainXtrans, solTrainY,
                        importance = TRUE, 
                        ntree = 1000)
# For forests built using conditional inference trees, the cforest function in the party 
# package is available. It has similar options, but the controls argument (note the plural) 
# allows the user to pick the type of splitting algorithm to use (e.g., biased or unbiased).

# Note! Neither of these functions can be used with missing data.

# The train function contains wrappers for tuning either of these models by specifying either
# method = "rf" or method = "cforest". Optimizing the mtry parameter may result in a slight 
# increase in performance. Also, train can use standard resampling methods for estimating 
# performance (as opposed to the out-of-bag estimate).

# For randomForest models, the variable importance scores can be accessed using a function in
# that package called importance. For cforest objects, the analogous function in the party 
# package is varimp.

# Each package tends to have its own function for calculating importance scores, similar to 
# the situation for class probabilities. caret has a unifying function called varImp that is
# a wrapper for variable importance functions for the following tree-model objects: rpart, 
# classbagg (produced by the ipred package’s bagging functions) randomForest, cforest, gbm, 
# and cubist.

# ============================================================================================
# Boosted Trees -----------------------------------------------------------
# ============================================================================================
# The most widely used package for boosting regression trees via stochastic gradient boosting
# machines is gbm. Like the random forests interface, models can be built in two distinct 
# ways:
gbmModel <- gbm.fit(solTrainXtrans, solTrainY, 
                    distribution = "gaussian")
# or,
gbmModel <- gbm(y ~ ., data = trainData,
                distribution = "gaussian")
# he distribution argument defines the type of loss function that will be optimized during 
# boosting. For a continuous response, distribution should be set to “gaussian.” The number 
# of trees (n.trees), depth of trees (interaction.depth), shrinkage (shrinkage), and 
# proportion of observations to be sampled (bag.fraction) can all be directly set in the call
# to gbm.

# Like other parameters, the train function can be used to tune over these parameters. To tune
# over interaction depth, number of trees, and shrinkage, for example, we first define a
# tuning grid. Then we train over this grid as follows:
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(100, 1000, by = 50),
                       shrinkage = c(.01, .1),
                       n.minobsinnode = 10)
# Note: tune functions indicates error unless minimum observation per node is specificed...
set.seed(100)
gbmTune <- train(solTrainXtrans, solTrainY, 
                 method = "gbm",
                 tuneGrid = gbmGrid,
                 verbose = FALSE)

# ============================================================================================
# Cubist ------------------------------------------------------------------
# ============================================================================================
# As previously mentioned, the implementation for this model created by Rule-Quest was 
# recently made public using an open-source license. An R package called Cubist was created 
# using the open-source code. The function does not have a formula method since it is 
# desirable to have the Cubist code manage the creation and usage of dummy variables. To 
# create a simple rule-based model with a single committee and no instance-based adjustment, 
# we can use the simple code:
cubistMod <- cubist(solTrainXtrans, solTrainY)

# An argument, committees, fits multiple models. The familiar predict method would be used 
# for new samples:
predict(cubistMod, solTestX)
# The choice of instance-based corrections does not need to be made until samples are 
# predicted. The predict function has an argument, neighbors, that can take on a single 
# integer value (between 0 and 9) to adjust the rule-based predictions from the training set.

# Once the model is trained, the summary function generates the exact rules that were used, 
# as well as the final smoothed linear model for each rule. Also, as with most other models, 
# the train function in the caret package can tune the model over values of committees and 
# neighbors through resampling:
cubistTuned <- train(solTrainXtrans, solTrainY, 
                     method = "cubist")

plot(cubistTuned)
cubPred <- predict(cubistTuned, solTestXtrans)

plot(cubPred, solTestY)
abline(0, 1, 
       col = "red")

plot(resid(cubistTuned), predict(cubistTuned))

# End file ----------------------------------------------------------------

# I have not completed the exercises of this chapter due to the computational heavy nature of
# these regression methods. One can utilise parallel computing, however, I was not motivated to do so at the point in time. I
# have had previous tutorial experience with Tree based models, such as David Langer's Titanic tutorial. Further, it is 
# unlikley that I will frequently use such methods in my domain of expertise.

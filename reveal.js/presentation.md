% TMLE / SuperLearning
% Drew Dimmery <drewd@nyu.edu>
% March 24, 2014  

# Structure
- SuperLearning
- SuperLearning for Causal Inference
- TMLE to correct for residual biases

# SuperLearner
- Fit a bunch of different models of varying complexity
- Question: is there an art to this selection of models?
    - Rose & van der Laan seem to say just throw them all in
    - Should we think about ensuring that complexity of models is allowed to
      grow as data increases?
    - That is, some sort of sieve estimator
    - Or will this come out in the CV wash?

# Discrete SuperLearner
- Choose the model with the lowest CV risk

# Full SuperLearner
- Use a logit type model to construct a set of weights over the set of
  estimated models.
- The fitted values from this "model of models" gives us the fitted values from
  the SuperLearner.

# Application

```r
dataset <- read.csv('wdbc.data',head=FALSE)
index <- 1:nrow(dataset)
testindex <- sample(index, trunc(length(index)*30/100))
testset <- dataset[testindex,]
trainset <- dataset[-testindex,]
require(e1071,quietly=TRUE)
tuned.svm <- tune.svm(V2~., data = trainset, gamma = 10^(-6:0), cost = 10^(-1:1))
```

# Tuning

```r
summary(tuned.svm)
```

```
## 
## Parameter tuning of 'svm':
## 
## - sampling method: 10-fold cross validation 
## 
## - best parameters:
##  gamma cost
##   0.01    1
## 
## - best performance: 0.0175 
## 
## - Detailed performance results:
##    gamma cost     error dispersion
## 1  1e-06  0.1 0.3656410 0.08502459
## 2  1e-05  0.1 0.3656410 0.08502459
## 3  1e-04  0.1 0.3656410 0.08502459
## 4  1e-03  0.1 0.2828205 0.09807148
## 5  1e-02  0.1 0.0500641 0.02880648
## 6  1e-01  0.1 0.0775641 0.03612565
## 7  1e+00  0.1 0.3656410 0.08502459
## 8  1e-06  1.0 0.3656410 0.08502459
## 9  1e-05  1.0 0.3656410 0.08502459
## 10 1e-04  1.0 0.2703205 0.09506124
## 11 1e-03  1.0 0.0500641 0.02880648
## 12 1e-02  1.0 0.0175000 0.02058182
## 13 1e-01  1.0 0.0300000 0.03291403
## 14 1e+00  1.0 0.3631410 0.08750059
## 15 1e-06 10.0 0.3656410 0.08502459
## 16 1e-05 10.0 0.2678205 0.09229077
## 17 1e-04 10.0 0.0500641 0.02880648
## 18 1e-03 10.0 0.0225641 0.01845738
## 19 1e-02 10.0 0.0175000 0.01687371
## 20 1e-01 10.0 0.0325641 0.03127522
## 21 1e+00 10.0 0.3606410 0.08424615
```

# SVM

```r
svm.model  <- svm(V2~., data = trainset, kernel="radial", gamma=tuned.svm$best.parameters$gamma,cost=tuned.svm$best.parameters$cost) 
svm.pred <- predict(svm.model, testset[,-2])
# Hard to improve on this:
table(pred = svm.pred, true = testset[,2])
```

```
##     true
## pred   B   M
##    B 102   8
##    M   2  58
```

# SuperLearning for Classification

```r
require(SuperLearner,quietly=TRUE)
create.SL.knn <- function(k = c(20, 30, 40, 50)) {
  for(mm in seq(length(k))){
    eval(parse(text = paste('SL.knn.', k[mm], '<- function(..., k = ', k[mm], ') SL.knn(..., k = k)', sep = '')), envir = .GlobalEnv)
  }
  invisible(TRUE)
}
create.SL.knn(k = c(3,5,7,9,11))
SL.svm.tune <- function(Y,X,newX,family, ...) {
  SuperLearner:::.SL.require("e1071")
  if (family$family == "binomial") {
    tune.svm <- tune.svm(y = Y, x = X, kernel="radial", gamma=10^{-6:0}, cost = 10^{-1:1})
    fit.svm <- svm(y = Y, x = X, fitted = FALSE,kernel="radial",gamma=tune.svm$best.parameters$gamma,cost=tune.svm$best.parameters$cost)
    pred <- predict(fit.svm, newdata = newX)
    fit <- list(object = fit.svm)
  }
  out <- list(pred = pred, fit = fit)
  class(out$fit) <- c("SL.svm")
  return(out)
} 

SL.classmod <- SuperLearner(
  ifelse(trainset$V2=="B",1,0),
  subset(trainset,select=-V2),
  SL.library=c(paste0("SL.knn.",c(3,5)),"SL.svm","SL.svm.tune","SL.bart","SL.randomForest","SL.polymars","SL.nnet"),
  family="binomial",newX=testset[,-2]
)
```

```
## Loading required package: polspline
```

```
## Warning in library(package, lib.loc = lib.loc, character.only = TRUE,
## logical.return = TRUE, : there is no package called 'polspline'
```

```
## Error: You have selected polymars or polyclass as a library algorithm but either do not have the polspline package installed or it can not be loaded
```

```r
SL.classmod
```

```
## Error in eval(expr, envir, enclos): object 'SL.classmod' not found
```

# Confusion Matrices


```r
table(pred = svm.pred, true = testset[,2])
```

```
##     true
## pred   B   M
##    B 102   8
##    M   2  58
```

```r
table(pred = ifelse(SL.classmod$SL.predict>=.5,1,0), true = ifelse(testset[,2]=="B",1,0))
```

```
## Error in ifelse(SL.classmod$SL.predict >= 0.5, 1, 0): object 'SL.classmod' not found
```

# SuperLearning for Causal Inference
- Causal Inference is fundamentally a missing data problem.
- We can make causal inference if we know either:
    - The function which causes missingness $g$
    - The function which generates outcomes $Q$
- Superlearning for causal inference is about estimating those functions in as
  flexible a manner as is possible.
- $\bar{Q}_n(A,W)$ - This function takes a treatment $A$ and a covariate vector
  $W$ and maps them to the outcome space $Y$.
- $\Psi(Q_n)$ a function of the outcome which we want to estimate.
- Take $\Psi(Q_n) = E[Q_n(1,W) - Q_n(0,W)]$ as an additive causal effect.
- We can make a first step substitution estimator of the causal effect as
  follows: $\frac{1}{n} \sum_{i=1}^{n} \bar{Q}_n(1,W_i) - \bar{Q}_n(0,W_i)$
- Note that it's a substitution estimator as we are marginalizing over the
  empirical distribution of $W$.

# BUT!
- This estimator isn't good enough for several reasons.
    - It suffers from bias due to the fact that for a unit $i$,
      $\bar{Q}_n(A_i,W_i)$ is not necessarily equal to $Y_i$. (and we aren't
      making an explicit tradeoff calculation)
    - We aren't using information about the treatment process $g$
- We can solve this with an iterative updating step, which R&vdL call
  "targeting"
- We do this with a simple bivariate regression:
- $Y_i - \bar{Q}_n(A_i,W_i) = \epsilon_n H_n(A_i,W_i)$
- $H_n(A_i,W_i) = \left(\frac{I(A=1)}{g_n(1|W)} -
  \frac{I(A=0)}{g_n(0|W)}\right)$
- And this regression can be modified to ensure things stay in the correct
  support (ie by using logit transforms for a binary outcome)
- We update $\bar{Q}_n(1,W)$ as $\bar{Q}_n(1,W) + \epsilon_n H_n(1,W)$ (and
  similarly for $\bar{Q}_n(0,W)$.
- It can be shown that this procedure ensures consistent estimates of
  parameters when *either* our models (via SuperLearning) for $g$ or $Q$ are specified correctly.

# Application
- I'm looking at the introduction of Fox News on voting
- The authors estimate a propensity score model, but use it only to try and
  demonstrate that (conditional on demographic factors) there is no linear
  correlation between vote share and the introduction of Fox News.
- Thus, the idea is that regression will identify a causal effect after we
  partial out the demographic factors.


```r
require(foreign)
d<-read.dta("FoxNewsFinalDataQJE.dta")
ds<-subset(d,sample12000==1)
covs<-c("pop","hs","hsp","college","male","black","hisp","empl","unempl","married","income","urban")
covs<-c(paste0(covs,2000),paste0(covs,"00m90"))
cablecovs<-names(ds)[c(grep("poptot2000d",names(ds)),grep("noch2000d",names(ds)))]
ds<-subset(ds,select=c(covs,cablecovs,"countystate","foxnews2000","reppresfv2p1996","totpreslvpop1996","reppresfv2p00m96"))
# There should be a fixed effect in the first stage, but this just takes too long
form.1st<-paste0("foxnews2000~0+reppresfv2p1996+totpreslvpop1996+",paste(covs,collapse="+"),"+",paste(cablecovs,collapse="+"))
# They weight by 1996 voters, but we won't.
# wts<-ds$totpresvotes1996/sum(ds$totpresvotes1996)
wts <- rep(1,nrow(ds)) 
d<-do.call("rbind",by(ds,ds$countystate,function(x) x-matrix(colMeans(x),ncol=ncol(x),nrow=nrow(x),byrow=TRUE)))
```

# IPW


```r
lin.1st.mod<-lm(form.1st,ds,weights=wts)
summary(lin.1st.mod)$coefficients[1:2,]
```

```
##                      Estimate  Std. Error    t value     Pr(>|t|)
## reppresfv2p1996   0.128656519 0.031787427  4.0474027 0.0000522126
## totpreslvpop1996 -0.006543252 0.007851649 -0.8333602 0.4046632052
```

```r
form.out<-as.formula(paste0("reppresfv2p00m96~0+foxnews2000+",paste(covs,collapse="+"),"+",paste(cablecovs,collapse="+")))
lin.out.mod<-lm(form.out,ds,weights=wts)
summary(lin.out.mod)$coefficients[1,]
```

```
##    Estimate  Std. Error     t value    Pr(>|t|) 
## 0.002678709 0.001370674 1.954300369 0.050695913
```

```r
# This glm takes shockingly long with so many FEs
pscore <- fitted(glm(form.1st,ds,weights=wts,family="binomial"))
A<-ds$foxnews2000
ipw <- A/pscore + (1-A)/(1-pscore)
ipw.mod <- lm(form.out,ds,weights=wts*ipw)
summary(ipw.mod)$coefficients[1,]
```

```
##     Estimate   Std. Error      t value     Pr(>|t|) 
## 0.0032060149 0.0008894163 3.6046280802 0.0003142505
```

# BART


```r
# Take out a bunch of covs because it takes too long
W<-subset(ds,select=c(covs,cablecovs)) #should include statecounty
Y<-ds$reppresfv2p00m96
require(BayesTree)
bart.1st.mod <- bart(W,A) # No analytic weights
```

```
## NOTE: assumming numeric response is binary
## 
## 
## Running BART with binary y
## 
## number of trees: 200
## Prior:
## 	k: 2.000000
## 	binary offset is: 0.000000
## 	power and base for tree prior: 2.000000 0.950000
## 	use quantiles for rule cut points: 0
## data:
## 	number of training observations: 9256
## 	number of test observations: 0
## 	number of explanatory variables: 42
## 
## 
## Cutoff rules c in x<=c vs x>c
## Number of cutoffs: (var: number of possible c):
## (1: 100) (2: 100) (3: 100) (4: 100) (5: 100) 
## (6: 100) (7: 100) (8: 100) (9: 100) (10: 100) 
## (11: 100) (12: 100) (13: 100) (14: 100) (15: 100) 
## (16: 100) (17: 100) (18: 100) (19: 100) (20: 100) 
## (21: 100) (22: 100) (23: 100) (24: 100) (25: 100) 
## (26: 100) (27: 100) (28: 100) (29: 100) (30: 100) 
## (31: 100) (32: 100) (33: 100) (34: 100) (35: 100) 
## (36: 100) (37: 100) (38: 100) (39: 100) (40: 100) 
## (41: 100) (42: 100) 
## 
## 
## Running mcmc loop:
## iteration: 100 (of 1100)
## iteration: 200 (of 1100)
## iteration: 300 (of 1100)
## iteration: 400 (of 1100)
## iteration: 500 (of 1100)
## iteration: 600 (of 1100)
## iteration: 700 (of 1100)
## iteration: 800 (of 1100)
## iteration: 900 (of 1100)
## iteration: 1000 (of 1100)
## iteration: 1100 (of 1100)
## time for loop: 610
## 
## Tree sizes, last iteration:
## 3 2 3 2 2 4 2 4 2 2 3 4 2 2 2 4 2 2 3 2 
## 2 2 1 2 2 2 2 3 2 3 3 2 3 2 2 2 2 2 2 2 
## 1 3 2 2 2 1 2 2 2 2 2 2 2 2 1 5 2 2 2 3 
## 3 2 2 2 4 3 2 3 2 3 3 3 2 1 2 4 2 3 2 3 
## 2 2 2 3 2 2 2 2 2 2 2 3 1 2 2 2 4 2 3 4 
## 3 3 2 2 3 3 2 2 4 3 4 4 2 3 1 2 3 2 2 3 
## 2 3 2 2 2 2 2 3 2 3 5 2 2 2 2 3 4 2 2 3 
## 4 2 2 2 3 2 3 3 2 2 2 2 4 3 4 2 3 2 4 2 
## 2 2 2 2 2 3 2 2 2 4 1 3 3 3 2 4 2 2 2 2 
## 4 2 2 2 4 3 2 2 3 2 3 4 3 2 5 2 2 2 2 3 
## Variable Usage, last iteration (var:count):
## (1: 8) (2: 9) (3: 7) (4: 7) (5: 3) 
## (6: 2) (7: 6) (8: 10) (9: 2) (10: 3) 
## (11: 5) (12: 12) (13: 2) (14: 11) (15: 4) 
## (16: 1) (17: 9) (18: 6) (19: 4) (20: 7) 
## (21: 5) (22: 4) (23: 5) (24: 6) (25: 4) 
## (26: 4) (27: 9) (28: 5) (29: 7) (30: 5) 
## (31: 13) (32: 10) (33: 12) (34: 4) (35: 9) 
## (36: 8) (37: 4) (38: 11) (39: 8) (40: 13) 
## (41: 13) (42: 14) 
## DONE BART 11-2-2014
```

```r
# So we know this stuff is suboptimal, but sacrifices must be made in the name of speediness
pscore <- pnorm(colMeans(bart.1st.mod$yhat.train))
ipw<- A/pscore + (1-A)/(1-pscore)
bart.ipw.mod <- lm(form.out,ds,weights=wts*ipw)
summary(bart.ipw.mod)$coefficients[1,]
```

```
##     Estimate   Std. Error      t value     Pr(>|t|) 
## 0.0010661135 0.0009122361 1.1686814740 0.2425622273
```

# Super Learner

```r
require(SuperLearner,quietly=TRUE)
SL.mod <-
   SuperLearner(A,W,family="binomial",SL.library=c("SL.bart","SL.svm.tune","SL.knn.3"))
```

```
## Note: no visible global function definition for 'knn'
```

```
## Loading required package: class
```

```r
pscore <- SL.mod$SL.predict
ipw<- A/pscore + (1-A)/(1-pscore)
SL.mod$coef
```

```
##     SL.bart_All SL.svm.tune_All    SL.knn.3_All 
##       0.3392114       0.4597771       0.2010115
```

```r
SL.ipw.mod <- lm(form.out,ds,weights=wts*ipw)
```

# Do TMLE

```r
source('SL.bart.R')
require(tmle,quietly=TRUE)
```

```
## Welcome to the tmle package, version 1.2.0-2
## 
## Use tmleNews() to see details on changes and bug fixes
## 
## Attaching package: 'tmle'
## 
## The following object is masked from 'package:SuperLearner':
## 
##     SL.glm.interaction
```

```r
create.SL.loess <- function(span = c(0.25, 0.50)) {
  for(mm in seq(length(span))) {
    eval(parse(text = paste('SL.loess.', span[mm], '<- function(..., span = ', span[mm], ') SL.loess(..., span = span)', sep = '')), envir = .GlobalEnv)
  }
  invisible(TRUE)
}
create.SL.loess(span=c(0.25,.5,.75))
Q.SL.lib<-c("SL.bart","SL.loess.0.25","SL.loess.0.5","SL.randomForest")
g.SL.lib<-c("SL.bart","SL.knn.5","SL.knn.9","SL.randomForest","SL.svm.tune")
fit.tmle<-tmle(Y,A,W,g.SL.library = g.SL.lib,Q.SL.library=Q.SL.lib)
```

```
## Loading required package: randomForest
```

```
## Error in estimateQ(Y = stage1$Ystar, Z, A, W, Delta, Q = stage1$Q, Qbounds = stage1$Qbounds, : Super Learner failed when estimating Q. Exiting program
```

```r
# fit.tmle<-tmle(Y,A,W)
fit.tmle$coef
```

```
## Error in eval(expr, envir, enclos): object 'fit.tmle' not found
```

```r
fit.tmle
```

```
## Error in eval(expr, envir, enclos): object 'fit.tmle' not found
```















% Instrumental Variables
% Drew Dimmery <drewd@nyu.edu>
% February 20, 2015

# Debugging
- Rubber duck debugging:
    - Use when you can't figure out why your code doesn't work right
    - Find something inanimate to talk to
    - Explain what your code does, line by excruciating line
    - If you can't explain it, that's probably where the problem is.
    - This works ridiculously well.
    - You should also be able to tell your duck exactly what is stored in each
      variable at all times.
- Check individual elements of your code on small data such that you know what
  the right answer *should* be.

# Introduce an Example
- We'll be working with data from a paper in the most recent issue of IO.
- Helfer, L.R. and E. Voeten. (2014) "International Courts as Agents of Legal Change: Evidence from LGBT Rights in Europe"
- The treatment we are interested in is the presence of absence of a ECtHR judgment.
- The outcome is the adoption of progressive LGBT policy.
- And there's a battery of controls, of course.
- Voeten has helpfully put all [replication materials online](http://hdl.handle.net/1902.1/19324).

# Prepare example


```r
require(foreign,quietly=TRUE)
d <- read.dta("replicationdataIOLGBT.dta")
#Base specification
d$ecthrpos <- as.double(d$ecthrpos)-1
d.lm <- lm(policy~ecthrpos+pubsupport+ecthrcountry+lgbtlaws+cond+eumember0+euemploy+coemembe+lngdp+year+issue+ccode,d)
d <- d[-d.lm$na.action,]
d$issue <- as.factor(d$issue)
d$ccode <- as.factor(d$ccode)
summary(d.lm)$coefficients[1:11,]
```

```
##                   Estimate   Std. Error    t value     Pr(>|t|)
## (Intercept)  -1.588605e+00 4.956355e-01 -3.2051890 1.360035e-03
## ecthrpos      6.500937e-02 1.056423e-02  6.1537237 8.289029e-10
## pubsupport    6.549488e-03 2.742967e-03  2.3877390 1.699714e-02
## ecthrcountry  1.297322e-01 3.583626e-02  3.6201389 2.979822e-04
## lgbtlaws      2.358238e-02 6.280655e-03  3.7547646 1.758966e-04
## cond          9.277344e-02 1.795954e-02  5.1656905 2.508722e-07
## eumember0    -8.586409e-03 8.497519e-03 -1.0104607 3.123339e-01
## euemploy      3.659200e-03 1.269275e-02  0.2882905 7.731389e-01
## coemembe      2.082823e-02 7.276808e-03  2.8622754 4.227313e-03
## lngdp        -7.522448e-07 4.501392e-07 -1.6711382 9.477027e-02
## year          8.019830e-04 2.522046e-04  3.1798904 1.484223e-03
```



# Marginal Effects
- [Blattman (2009)](http://chrisblattman.com/projects/sway/) uses marginal effects "well" in the sense of causal inference.
- Use the builtin `predict` function; it will make your life easier.

. . .


```r
d.lm.interact <- lm(policy~ecthrpos*pubsupport+ecthrcountry+lgbtlaws+cond+eumember0+euemploy+coemembe+lngdp+year+issue+ccode,d)
frame0 <- frame1 <- model.frame(d.lm.interact)
frame0[,"ecthrpos"] <- 0
frame1[,"ecthrpos"] <- 1
meff <- mean(predict(d.lm.interact,newd=frame1) - predict(d.lm.interact,newd=frame0))
meff
```

```
## [1] 0.08197142
```

- Why might this be preferable to "setting things at their means/medians"?
- It's essentially integrating over the sample's distribution of observed characteristics.
- (And if the sample is a SRS from the population [or survey weights make it LOOK like it is], this will then get you the marginal effect on the population of interest)

# Delta Method
- Note 1: We know that our vector of coefficients are asymptotically multivariate normal.
- Note 2: We can approximate the distribution of many (not just linear) functions of these coefficients using the delta method.
- Delta method says that you can approximate the distribution of $h(b_n)$ with $\bigtriangledown{h}(b)'\Omega\bigtriangledown{h}(b)$ Where $\Omega$ is the asymptotic variance of $b$.
- In practice, this means that we just need to be able to derive the function whose distribution we wish to approximate.

# Trivial Example
- Maybe we're interested in the ratio of the coefficient on `ecthrpos` to that of `pubsupport`.
- Call it $b_2 \over b_3$. The gradient is $(\frac{1}{b_3}, \frac{b_2}{b_3^2})$
- Estimate this easily in R with:

. . .


```r
grad<-c(1/coef(d.lm)[3],coef(d.lm)[2]/coef(d.lm)[3]^2)
grad
```

```
## pubsupport   ecthrpos 
##   334.0251  8046.4669
```

```r
se<-sqrt(t(grad)%*%vcov(d.lm)[2:3,2:3]%*%grad)
est<-coef(d.lm)[2]/coef(d.lm)[3]
c(estimate=est,std.error=se)
```

```
## estimate.ecthrpos         std.error 
##          24.08941          35.32946
```

```r
require(car)
```

```
## Loading required package: car
```

```
## Warning in library(package, lib.loc = lib.loc, character.only = TRUE,
## logical.return = TRUE, : there is no package called 'car'
```

```r
deltaMethod(d.lm,"ecthrpos/pubsupport")
```

```
## Error in eval(expr, envir, enclos): could not find function "deltaMethod"
```

# Linear Functions
- But for most "marginal effects", you don't need to use the delta method.
- Just remember your rules for variances.
- $\text{var}(aX+bY) = a^2\text{var}(X) + b^2\text{var}(Y) + 2ab\text{cov}(X,Y)$
- If you are just looking at changes with respect to a single variable, you can just multiply standard errors.
- That is, a change in a variable of 3 units means that the standard error for the marginal effect would be 3 times the standard error of the coefficient.
- This isn't what Clarify does, though.
  
# Instrumental Variables
- $\rho = {\text{Cov}(Y_i,Z_i) \over \text{Cov}(S_i,Z_i)} = { { \text{Cov}(Y_i,Z_i) \over \text{Var}(Z_i)} \over {\text{Cov}(S_i,Z_i) \over \text{Var}(Z_i)}} = {\text{Reduced form} \over \text{First stage}}$
- If we have a perfect instrument, this will be unbiased.
- But bias is a function of both violation of exclusion restriction and of strength of first stage.
- 2SLS has finite sample bias. (Cyrus showed this, but didn't dwell on it)
- In particular, it [can be shown](http://econ.lse.ac.uk/staff/spischke/ec533/Weak%20IV.pdf) that this bias "is":  
${\sigma_{\eta \xi} \over \sigma_{\xi}^2}{1 \over F + 1}$  
where $\eta$ is the error in the structural model and $\xi$ is the error in the first stage.
- With an irrelevant instrument ($F=0$), the bias is equal to that of OLS (regression of $Y$ on $X$).
- There are some bias corrections for this, we might talk about this next week.

# Setup IV example
- For our example with IV, we will start with AJR (2001) - Colonial Origins of Comparative Development
- Treatment is average protection from expropriation
- Exogenous covariates are dummies for British/French colonial presence
- Instrument is settler mortality
- Outcome is log(GDP) in 1995

. . .


```r
require(foreign)
dat <- read.dta("maketable5.dta")
```

```
## Error in read.dta("maketable5.dta"): unable to open file: 'No such file or directory'
```

```r
dat <- subset(dat, baseco==1)
```

```
## Error in subset(dat, baseco == 1): object 'dat' not found
```

# Estimate IV via 2SLS


```r
require(AER)
```

```
## Loading required package: AER
```

```
## Warning in library(package, lib.loc = lib.loc, character.only = TRUE,
## logical.return = TRUE, : there is no package called 'AER'
```

```r
first <- lm(avexpr~logem4+f_brit+f_french,dat)
```

```
## Error in is.data.frame(data): object 'dat' not found
```

```r
iv2sls<-ivreg(logpgp95~avexpr+f_brit+f_french,~logem4+f_brit+f_french,dat)
```

```
## Error in eval(expr, envir, enclos): could not find function "ivreg"
```

```r
require(car)
```

```
## Loading required package: car
```

```
## Warning in library(package, lib.loc = lib.loc, character.only = TRUE,
## logical.return = TRUE, : there is no package called 'car'
```

```r
linearHypothesis(first,"logem4",test="F")
```

```
## Error in eval(expr, envir, enclos): could not find function "linearHypothesis"
```

# Examine Output


```r
summary(iv2sls)
```

```
## Error in summary(iv2sls): object 'iv2sls' not found
```

# Sensitivity Analysis
- Conley, Hansen and Rossi (2012)
- Suppose that the exclusion restriction does NOT hold, and there exists a direct effect from the instrument to the outcome.
- That is, the structural model is:  
$Y = X\beta + Z\gamma + \epsilon$
- If $\gamma$ is zero, the exclusion restriction holds (we're in a structural framework)
- We can assume a particular value of $\gamma$, take $\tilde{Y} = Y - Z\gamma$ and estimate our model, gaining an estimate of $\beta$.
- This defines a sensitivity analysis on the exclusion restriction.
- Subject to an assumption about the support of $\gamma$, they suggest estimating in a grid over this domain, and then taking the union of the confidence intervals for each value of $\gamma$ as the combined confidence interval (which will cover).

. . .


```r
gamma <- seq(-1,1,.25)
ExclSens <- function(g) {
  newY <- dat$logpgp95 - g*dat$logem4
  coef(ivreg(newY~avexpr+f_brit+f_french,~logem4+f_brit+f_french,cbind(dat,newY)))[2]
}
sens.coefs <- sapply(gamma,ExclSens)
```

```
## Note: no visible binding for global variable 'dat' 
## Note: no visible binding for global variable 'dat' 
## Note: no visible global function definition for 'ivreg' 
## Note: no visible binding for global variable 'dat'
```

```
## Error in FUN(c(-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1)[[1L]], ...): object 'dat' not found
```

```r
names(sens.coefs)<- round(gamma,3)
```

```
## Error in names(sens.coefs) <- round(gamma, 3): object 'sens.coefs' not found
```

```r
round(sens.coefs,3)
```

```
## Error in eval(expr, envir, enclos): object 'sens.coefs' not found
```

# More IV Stuff
- We're going to be looking at [Ananat
  (2011)](http://www.aeaweb.org/articles.php?doi=10.1257/app.3.2.34) in AEJ
- This study looks at the effect of racial segregation on economic outcomes.
- Outcome: Poverty rate & Inequality (Gini index)
- Treatment: Segregation
- Instrument: "railroad division index"
- Main covariate of note: railroad length in a town
- I'm dichotomizing treatment and instrument for simplicity.
- And my outcomes are for the Black subsample

. . .


```r
require(foreign)
d<-read.dta("aej_maindata.dta")
d$herf_b<-with(d,ifelse(herf >= quantile(herf,.5),1,0))
d$dism1990_b<-with(d,ifelse(dism1990 >= quantile(dism1990,.5),1,0))
first.stage <- lm(dism1990~herf+lenper,d)
first.stage.b <- lm(dism1990_b~herf_b+lenper,d)
require(AER)
```

```
## Loading required package: AER
```

```
## Warning in library(package, lib.loc = lib.loc, character.only = TRUE,
## logical.return = TRUE, : there is no package called 'AER'
```

```r
gini.iv <- ivreg(lngini_b~dism1990+lenper,~herf+lenper,d)
```

```
## Error in eval(expr, envir, enclos): could not find function "ivreg"
```

```r
gini.iv.b <- ivreg(lngini_b~dism1990_b+lenper,~herf_b+lenper,d)
```

```
## Error in eval(expr, envir, enclos): could not find function "ivreg"
```

```r
pov.iv <- ivreg(povrate_b~dism1990+lenper,~herf+lenper,d)
```

```
## Error in eval(expr, envir, enclos): could not find function "ivreg"
```

```r
pov.iv.b <- ivreg(povrate_b~dism1990_b+lenper,~herf_b+lenper,d)
```

```
## Error in eval(expr, envir, enclos): could not find function "ivreg"
```

# Base Results

```r
round(summary(first.stage)$coefficients[2,],3)
```

```
##   Estimate Std. Error    t value   Pr(>|t|) 
##      0.357      0.081      4.395      0.000
```

```r
round(summary(first.stage.b)$coefficients[2,],3)
```

```
##   Estimate Std. Error    t value   Pr(>|t|) 
##      0.372      0.083      4.481      0.000
```

```r
round(summary(gini.iv)$coefficients[2,],3)
```

```
## Error in summary(gini.iv): object 'gini.iv' not found
```

```r
round(summary(gini.iv.b)$coefficients[2,],3)
```

```
## Error in summary(gini.iv.b): object 'gini.iv.b' not found
```

```r
round(summary(pov.iv)$coefficients[2,],3)
```

```
## Error in summary(pov.iv): object 'pov.iv' not found
```

```r
round(summary(pov.iv.b)$coefficients[2,],3)
```

```
## Error in summary(pov.iv.b): object 'pov.iv.b' not found
```

# Abadie's $\kappa$
- Recall from the lecture that we can use a weighting scheme to calculate
  statistics on the compliant population.
- $E[g(Y,D,X)|D_1 > D_0] = {1 \over p(D_1>D_0)} E[\kappa g(Y,D,X)]$
- $\kappa = 1 - {D_i (1-Z_i) \over p(Z_i =0|X)} - {(1-D_i)Z_i \over p(Z_i =1|X)}$
- $E[\kappa|X] = E[D_1 -D_0|X] = E[D|X,Z=1] - E[D|X,Z=0]$
- Take $w_i = {\kappa_i \over E[D_{1}-D_{0}|X_i]}$
- Use this in calculating any interesting statistics (means, variance, etc)
- This let's you explore the units composing your LATE.

. . .


```r
getKappaWt<-function(D,Z) {
  pz <- mean(Z)
  pcomp <- mean(D[Z==1]) - mean(D[Z==0])
  if(pcomp < 0) stop("Assuming p(D|Z) > .5")
  kappa <- 1 - D*(1-Z)/(1-pz) - (1-D)*Z/pz
  # Note that pcomp = mean(kappa)
  kappa / pcomp
}
w <- with(d,getKappaWt(D=dism1990_b,Z=herf_b))
varlist <- c("closeness","area1910","ctyliterate1920","hsdrop_b","manshr","ctymanuf_wkrs1920","ngov62")
samp.stats<-sapply(varlist,function(v) mean(d[,v],na.rm=TRUE))
comp.stats<-sapply(varlist,function(v) weighted.mean(d[,v],w,na.rm=TRUE))
```

# Examine Complier Statistics

```r
summary(w)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  -2.511  -2.429   2.470   1.000   2.470   2.470
```

```r
rbind(sample=samp.stats,compliers=comp.stats)
```

```
##           closeness area1910 ctyliterate1920  hsdrop_b    manshr
## sample    -362.4348 14626.43       0.9585012 0.2516300 0.1891766
## compliers -299.1428 18012.56       0.9514523 0.2423754 0.2109807
##           ctymanuf_wkrs1920   ngov62
## sample            0.4618666 55.55072
## compliers         0.4266065 83.65072
```

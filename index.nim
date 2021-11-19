import nimib

nbInit
nbDoc.useLatex

nbText: md"""
# Bayesian Inference with Linear Model

At the time of this writing, Nim does not have any libraries for
conducting Bayesian inference. However, it is quite easy for us to
write all of the necessary code ourselves. This is a good exercise for learning
about Bayesian inference and the syntax and speed of Nim make it an excellent 
choice for the this. In this tutorial we will walk through 
the different parts needed to perform Bayesian inference with a linear model. 
We will assume that you have some basic understanding of Bayesian
inference already. There are many excellent introductions available in books
and online.

For a quick refresher, Bayes rule can be written as:

$$ P(\theta|\text{Data}) = \frac{P(\text{Data}|\theta)P(\theta)}{\sum_{\theta}P(\text{Data}|\theta)P(\theta)}$$

Where each of these terms are referred to as:

$$ \text{Posterior} = \frac{\text{Likelihood} \cdot \text{Prior}}{\text{Marginal Likelihood}}$$

In this tutorial we will condsider a simple linear model 
$ y_{i} = \beta_{0} + \beta_{1} x_{i} + \epsilon_{i} $ where the parameters
$\beta_0$ (y intercept), $\beta_1$ (slope), and $\epsilon$ (random error)
describe the relationship between a predictor variable $x$ and a response
variable $y$, with some unaccounted for residual ranodom error that 
is normally distributed. 

We well estimate the values of the slope ($\beta_{0}$), the y-intercept 
($\beta_{1}$), and the standard deviation of the normally distributed random
error which we will call $\tau$.

We can express this using Bayes rule:

$$ \displaystyle P(\beta_{0}, \beta_{1}, \tau | Data) =
  \frac{P(Data|\beta_{0}, \beta_{1}, \tau) P(\beta_{0}, \beta_{1}, \tau)}
  {\iiint d\beta_{0} d\beta_{1} d\tau P(Data|\beta_{0}, \beta_{1}, \tau) P(\beta_{0}, \beta_{1}, \tau)} $$


# Generate Data
We need some data to work with. Let's simulate data  
under the model: $y = 0 + 1x + \epsilon$ where $\epsilon \sim N(0, 1)$ with 
$\beta_{0}=0$ $\beta_{1}=1$ and $\tau=1$
"""
nbCode:
  import std/sequtils, std/random 
  import stats
  var 
    n = 100
    b0 = 0.0
    b1 = 1.0
    sd = 1.0
    x = newSeq[float](n)
    y = newSeq[float](n)
  for i in 0 ..< n: 
    x[i] = rand(10.0..100.0) 
    y[i] = b0 + b1 * x[i] + gauss(0.0, sd) 
nbText: md"""
We can use `ggplotnim` to see what these data look like.
"""
nbCode:
  import datamancer, ggplotnim
  var sim = seqsToDf(x, y)
  ggplot(sim, aes("x", "y")) +
    geom_point() +
    ggsave("images/simulated-data.png")
nbImage("images/simulated-data.png")


nbText: md"""
# Priors
We need to choose prior probability distributions for each of the parameters 
that we are estimating.  Let's use a normal distribution for the priors on 
$\beta_{0}$ and $\beta_{1}$ The $\tau$ parameter be a positive value greater 
than 0 so let's use the gamma distribution.

$$ \beta_{0} \sim Normal(\mu_{0}, \tau_{0})$$
$$ \beta_{1} \sim Normal(\mu_{1}, \tau_{1})$$
$$ \tau \sim Gamma(\alpha_{0}, \beta_{0})$$

Since this tutorial is for the purpose of demonstration, let's use very informed
priors so that we can get a good sample from the posterior more quickly.

$$ \beta_{0} \sim Normal(0, 1)$$
$$ \beta_{1} \sim Normal(1, 1)$$
$$ \tau \sim Gamma(1, 1)$$

To calculate the prior probability of a proposed parameter value, we will need 
the proability density functions for our priors.

#### Normal PDF
$$ p(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^{2}} $$

#### Gamma PDF
$$ p(x) = \frac{1}{\Gamma(k)\theta^{k}} x^{k-1} e^{-\frac{x}{\theta}} $$

"""

nbCode:
  import distributions  
  import std/math

  proc normalPdf(z, mu, sigma: float): float = 
    result = pow(E, (-0.5 * ((z - mu) / sigma)^2)) / (sigma * sqrt(2.0 * PI))

  proc gammaPdf(x, k, theta: float): float = 
    result = pow(x, k - 1.0) * pow(E, -(x / theta)) / (gamma(k) * pow(theta, k))

nbText: md"""

Now that we have probability density functions available, we will be able to
compute a prior probability for a given parameterization of the model.
We will actually be using the $ln$ of the probabilities to
reduce rounding error since these values can be quite small.
"""
nbCode:
  proc logPrior(b0, b1, sd: float): float = 
    let 
      b0Prior = ln(normalPdf(b0, 0.0, 1.0))
      b1Prior = ln(normalPdf(b1, 1.0, 1.0))
      sdPrior = ln(gammaPdf(1/sd, 1.0, 1.0))
    result = b0Prior + b1Prior + sdPrior


nbText: md"""
# Likelihood
We need to be able to calculate the likelihood of the observed $y_{i}$ values
given the observed $x_{i}$ values and proposed parameter values for $\beta_{0}$, 
$\beta_{1}$, and $\tau$. 

We can write the model in a slightly different way:   

$$\mu = \beta_{0} +\beta_{1} x$$
$$ y \sim N(\mu, \tau) $$

Then to compute the likelihood for a given set of $\beta_{0}$, $\beta_{1}$, $\tau$ 
parameters and data values $x_{i}$ and $y_{i}$ we use the normal probability 
density function which we wrote before. We will again work with the $ln$ of the  
likelihood as we did with the priors.

"""
nbCode:
  proc logLikelihood(x, y: seq[float], b0, b1, sd: float): float = 
    var likelihoods = newSeq[float](y.len) 
    for i in 0 ..< y.len: 
      let pred = b0 + b1 * x[i]
      likelihoods[i] = ln(normalPdf(y[i], pred, sd))
    result = sum(likelihoods) 


nbText: md"""
# Posterior
We cannot analytically solve the posterior probability distribution of our
linear model as the integration of the marginal likelihood is intractable. 
But we can approximate it with markov chain monte carlo (mcmc) thanks to this 
property of Bayes rule:

$$ \displaystyle P(\beta_{0}, \beta_{1}, \tau | Data) \propto
  P(Data|\beta_{0}, \beta_{1}, \tau) P(\beta_{0}, \beta_{1}, \tau) $$

"""
nbCode:
  proc logPosterior(x, y: seq[float], b0, b1, sd: float): float = 
    let 
      like = logLikelihood(x=x, y=y, b0=b0, b1=b1, sd=sd)
      prior = logPrior(b0=b0, b1=b1, sd=sd)
    result = like + prior


nbText: md"""
# MCMC
We will use a Metropolis-Hastings algorithm to approximate the posterior. 
The steps of this algorithm are as follows:
1) Choose starting values
2) Propose new parameter values close to the previous ones.
3) Accept the proposed parameter value with probability ...
"""
nbCode:
  proc mcmc(x, y: seq[float], nSamples: int, b0, b1, sd, pb0, pb1, psd: float): 
      (seq[float], seq[float], seq[float]) =  
    var 
      b0Samples = newSeq[float](nSamples+1) 
      b1Samples = newSeq[float](nSamples+1) 
      sdSamples = newSeq[float](nSamples+1) 
  
    b0Samples[0] = b0  
    b1Samples[0] = b1 
    sdSamples[0] = sd 
    
    for i in 1..nSamples:
      let 
        prevB0 = b0Samples[i-1]
        prevB1 = b1Samples[i-1]
        prevSd = sdSamples[i-1]
        propB0 = gauss(prevB0, pb0) 
        propB1 = gauss(prevB1, pb1)
        propSd = gauss(prevSd, psd)
      if propSd > 0.0:
        var
          prevLogPost = logPosterior(x=x, y=y, b0=prevB0, b1=prevB1, sd=prevSd) 
          propLogPost = logPosterior(x=x, y=y, b0=propB0, b1=propB1, sd=propSd) 
          ratio = exp(propLogPost - prevLogPost)  
        if rand(1.0) < ratio:
          b0Samples[i] = propB0  
          b1Samples[i] = propB1  
          sdSamples[i] = propSd  
        else: 
          b0Samples[i] = prevB0  
          b1Samples[i] = prevB1  
          sdSamples[i] = prevSd  
      else:
        b0Samples[i] = prevB0  
        b1Samples[i] = prevB1  
        sdSamples[i] = prevSd  
    result = (b0Samples, b1Samples, sdSamples)
  var
    nSamples = 100000
    (b0Samples, b1Samples, sdSamples) = mcmc(x, y, nSamples, 0.0, 1.0, 1.0, 0.1, 0.1, 0.1) 
  

nbText: md"""
# Burnin
Initially the mcmc chain may spend time exploring unlikely regions of 
parameter space. We can get a better approximation of the posterior if we 
exclude these early steps in the chain. These excluded samples are referred to   
as the burnin. A burnin of $10%$ seems to work well with our informative priors 
and starting values. 
"""
nbCode:
  let burnin = (nSamples.float * 0.1 + 1).int


nbText: md"""
# Posterior means
One way to summarize the estimates from the posterior distribution is to calculate
the mean. Let's see how close these values are to the true values of the parameters. 
"""
nbCode:
  import stats
  let
    meanB0 = mean(b0Samples[burnin..^1])
    meanB1 = mean(b1Samples[burnin..^1])
    meanSd = mean(sdSamples[burnin..^1])
  echo meanB0
  echo meanB1
  echo meanSd


nbText: md"""
# Highest density interval
The means give us a point estimate for our parameter values but they tell us 
nothing about the uncertainty of our estimate. We can get a sense for that by 
looking at the highest density interval. 
"""
nbCode:
  import algorithm
  proc hdi(samples: seq[float], credMass: float): (float, float) =  
    let 
      sortedSamples = sorted(samples, system.cmp[float])
      ciIdxInc = int(floor(credMass * float(sortedSamples.len)))
      nCIs = sortedSamples.len - ciIdxInc
    var ciWidth = newSeq[float](nCIs)
    for i in 0..<nCIs:
      ciWidth[i] = sortedSamples[i + ciIdxInc] - sortedSamples[i]
    let
      minCiWidthIx = minIndex(ciWidth)
      hdiMin = sortedSamples[minCiWidthIx]
      hdiMax = sortedSamples[minCiWidthIx + ciIdxInc]
    result = (hdiMin, hdiMax)

  let 
    (b0HdiMin, b0HdiMax) = hdi(b0Samples[burnin..^1], 0.95)
    (b1HdiMin, b1HdiMax) = hdi(b1Samples[burnin..^1], 0.95)
    (sdHdiMin, sdHdiMax) = hdi(sdSamples[burnin..^1], 0.95)
  echo b0HdiMin, " ", b0HdiMax
  echo b1HdiMin, " ", b1HdiMax
  echo sdHdiMin, " ", sdHdiMax


nbText: md"""
# Trace plots 
We can get a sense for how well our mcmc performed and therefore gain some  
sense for how good our estimates might be by looking at the trace plot which 
shows the parameter value store during each step in the mcmc chain. Either 
the accepted proposal or the previous one if the a proposal is rejected. Trace
plots can be unreliable for mcmc performance so it is a good  
idea to assess this with other methods as well. 
"""
nbCode:
  let 
    ixs = toSeq(0 ..< b0Samples.len-burnin)
    df = seqsToDf({
      "ixs":ixs, 
      "b0":b0Samples[burnin..^1],
      "b1":b1Samples[burnin..^1],
      "sd":sdSamples[burnin..^1]})
  ggplot(df, aes(x="ixs", y="b0")) + 
    geom_line() +
    ggsave("images/samples-b0.png")
  
  ggplot(df, aes(x="ixs", y="b1")) + 
    geom_line() +
    ggsave("images/samples-b1.png")
  
  ggplot(df, aes(x="ixs", y="sd")) + 
    geom_line() +
    ggsave("images/samples-sd.png")
nbImage("images/samples-b0.png")
nbImage("images/samples-b1.png")
nbImage("images/samples-sd.png")


nbText: md"""
# Histograms 
"""
nbCode:
  ggplot(df, aes("b0")) +
    geom_histogram() +
    ggsave("images/hist-b0.png")
  
  ggplot(df, aes("b1")) +
    geom_histogram() +
    ggsave("images/hist-b1.png")
  
  ggplot(df, aes("sd")) +
    geom_histogram() +
    ggsave("images/hist-sd.png")
nbImage("images/hist-b0.png")
nbImage("images/hist-b1.png")
nbImage("images/hist-sd.png")


nbText: md"""
# Standardize Data
"""
nbCode: 
  var 
    stX = newSeq[float](n)
    stY = newSeq[float](n)
    meanX = mean(x) 
    sdX = standardDeviation(x)
    meanY = mean(y) 
    sdY = standardDeviation(y)
  for i in 0 ..< n:
    stX[i] = (x[i] - meanX) / sdX 
    stY[i] = (y[i] - meanY) / sdY


nbText: md"""
We can see that these data are now centered around zero and have the same scale.
"""
nbCode:
  var standardized = seqsToDf(stX, stY)
  ggplot(standardized, aes("x", "y")) +
    geom_point() +
    ggsave("images/st-simulated-data.png")
nbImage("images/st-simulated-data.png")


nbText: md"""
We can run the MCMC as before with some slight changes. Since our data are on a  
different scale, the proposals we were making before wont work very well. So
we should make the proposed changes smaller.
"""
nbCode:
  var 
    (b0StSamples, b1StSamples, sdStSamples) = mcmc(stX, stY, nSamples, 0, 1, 1, 0.01, 0.01, 0.01) 


nbText: md""" 
### Convert back to original scale
$$ \beta_{0} = \zeta_{0} SD_{y} + M_{y} - \zeta_{1} SD_{y} M_{x} / SD_{x} $$  
$$ \beta_{1} = \zeta_{1} SD_{y} / SD_{x} $$ 
Is this right?
$$ \tau = \zeta_{\tau} SD_{y} + M_{y} - \zeta_{1} SD_{y} M_{x} / SD_{x} $$  
"""
nbCode: 
  for i in 0 ..< nSamples:
    b0StSamples[i] = b0StSamples[i] * sdY + meanY - b1StSamples[i] * sdY * meanX / sdX
    b1StSamples[i] = b1StSamples[i] * sdY / sdX 
    sdStSamples[i] = sdStSamples[i] * sdY + meanY - b1StSamples[i] * sdY * meanX / sdX


nbText: md"""
### Means and HDIs
"""
nbCode:
  let
    meanStB0 = mean(b0StSamples[burnin..^1])
    meanStB1 = mean(b1StSamples[burnin..^1])
    meanStSd = mean(sdStSamples[burnin..^1])
  echo meanStB0
  echo meanStB1
  echo meanStSd

  let 
    (b0StHdiMin, b0StHdiMax) = hdi(b0StSamples[burnin..^1], 0.95)
    (b1StHdiMin, b1StHdiMax) = hdi(b1StSamples[burnin..^1], 0.95)
    (sdStHdiMin, sdStHdiMax) = hdi(sdStSamples[burnin..^1], 0.95)
  echo b0StHdiMin, " ", b0StHdiMax
  echo b1StHdiMin, " ", b1StHdiMax
  echo sdStHdiMin, " ", sdStHdiMax
  let 
    stDf = seqsToDf({
      "ixs":ixs, 
      "b0":b0StSamples[burnin..^1],
      "b1":b1StSamples[burnin..^1],
      "sd":sdStSamples[burnin..^1]})


nbText: md"""
### Plots
"""
nbCode:
  ggplot(stDf, aes(x="ixs", y="b0")) + 
    geom_line() +
    ggsave("images/st-samples-b0.png")
  
  ggplot(stDf, aes(x="ixs", y="b1")) + 
    geom_line() +
    ggsave("images/st-samples-b1.png")
  
  ggplot(stDf, aes(x="ixs", y="sd")) + 
    geom_line() +
    ggsave("images/st-samples-sd.png")

  ggplot(stDf, aes("b0")) +
    geom_histogram() +
    ggsave("images/st-hist-b0.png")
  
  ggplot(stDf, aes("b1")) +
    geom_histogram() +
    ggsave("images/st-hist-b1.png")
  
  ggplot(stDf, aes("sd")) +
    geom_histogram() +
    ggsave("images/st-hist-sd.png")

nbImage("images/st-samples-b0.png")
nbImage("images/st-samples-b1.png")
nbImage("images/st-samples-sd.png")

nbImage("images/st-hist-b0.png")
nbImage("images/st-hist-b1.png")
nbImage("images/st-hist-sd.png")


nbText: md"""
# Final Note 
Of course we could have simply and more efficiently done this using least squares regression.
However the Bayesian approach allows us to very easily and intuitively express
uncertainty about our estimates and can be easily extended to much more complex 
models for which there are not such simple solutions.
"""

nbSave
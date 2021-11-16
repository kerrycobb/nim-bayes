import nimib

nbInit
nbDoc.useLatex
nbDoc.context["mathjax_support"] = true


nbText: md"""
# Linear Model

$ y = \beta_{0} + \beta_{1} x + \epsilon $

$ \displaystyle p(\beta_{0}, \beta_{1}, \tau | y) = 
  \frac{p(y|\beta_{0}, \beta_{1}, \tau) p(\beta_{0}, \beta_{1}, \tau)}
  {\iiint d\beta_{0} d\beta_{1} d\tau p(y|\beta_{0}, \beta_{1}, \tau) p(\beta_{0}, \beta_{1}, \tau)} $

$ p(\beta_{0}, \beta_{1}, \tau | y) \propto p(y|\beta_{0}, \beta_{1}, \tau) p(\beta_{0}, \beta_{1}, \tau) $


Simulate some data:
"""
nbCode:
  import sequtils, random 
  var 
    n = 100
    b0 = 0.0
    b1 = 1.0
    sd = 100.0
    x = newSeq[float](n)
    y = newSeq[float](n)
  for i in 0 ..< n: 
    x[i] = rand(10.0..100.0) 
    y[i] = b0 + b1 * x[i] + gauss(0.0, sd) 


# Plot data
nbText: md"""
Plot the data
"""
nbCode:
  import datamancer, ggplotnim
  var sim = seqsToDf(x, y)
  ggplot(sim, aes("x", "y")) +
    geom_point() +
    ggsave("plots/simulated-data.png")
nbImage("plots/simulated-data.png")


nbText: md"""
Rescale the data:
"""
nbCode:
  import stats
  let 
    meanX = mean(x) 
    sdX = standardDeviation(x)
    meanY = mean(y) 
    sdY = standardDeviation(y)
  for i in 0 ..< n:
    x[i] = (x[i] - meanX) / sdX 
  ggplot(sim, aes("x", "y")) +
    geom_point() +
    ggsave("plots/simulated-scaled-data.png")
nbImage("plots/simulated-scaled-data.png")



## Likelihood
nbText: md"""
The likelihood:
"""
nbCode:
  import distributions, math
  proc logLikelihood(x, y: seq[float], b0, b1, sd: float): float = 
    var likelihoods = newSeq[float](y.len) 
    for i in 0..<y.len: 
      let pred = b0 + b1 * x[i]
      likelihoods[i] = ln(initNormalDistribution(pred, sd).pdf(y[i]))
    result = sum(likelihoods) 


## Prior
nbText: md"""
The prior:
"""
nbCode:
  proc logPrior(b0, b1, sd: float): float = 
    let 
      b0Prior = ln(initNormalDistribution(0.0, 1.0).pdf(b0))
      b1Prior = ln(initNormalDistribution(1.0, 1.0).pdf(b1))
      sdPrior = ln(initGammaDistribution(1.0, 1.0).pdf(1/sd))
    result = b0Prior + b1Prior + sdPrior


nbText: md"""
The posterior:
"""
nbCode:
  proc logPosterior(x, y: seq[float], b0, b1, sd: float): float = 
    let 
      like = logLikelihood(x=x, y=y, b0=b0, b1=b1, sd=sd)
      prior = logPrior(b0=b0, b1=b1, sd=sd)
    result = like + prior


nbText: md"""
MCMC:
"""
nbCode:
  var 
    nSamples = 200000
    b0Samples = newSeq[float](nSamples+1) 
    b1Samples = newSeq[float](nSamples+1) 
    sdSamples = newSeq[float](nSamples+1) 

  b0Samples[0] = 0.0  
  b1Samples[0] = 1.0 
  sdSamples[0] = 10.0 
  
  for i in 1..nSamples:
    let 
      prevB0 = b0Samples[i-1]
      prevB1 = b1Samples[i-1]
      prevSd = sdSamples[i-1]
      propB0 = gauss(prevB0, 0.3) 
      propB1 = gauss(prevB1, 0.1)
      propSd = gauss(prevSd, 1.0)
    if propSd > 0.0:
      var
        prevLogPosterior = logPosterior(x=x, y=y, b0=prevB0, b1=prevB1, sd=prevSd) 
        propLogPosterior = logPosterior(x=x, y=y, b0=propB0, b1=propB1, sd=propSd) 
        ratio = exp(propLogPosterior - prevLogPosterior)  
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



nbText: md"""
Burnin:
"""
nbCode:
  let burnin = (nSamples.float * 0.1 + 1).int


nbText: md"""
Posterior means:
"""
nbCode:
  let
    meanB0 = mean(b0Samples[burnin..^1])
    meanB1 = mean(b1Samples[burnin..^1])
    meanSd = mean(sdSamples[burnin..^1])
  echo meanB0
  echo meanB1
  echo meanSd


nbText: md"""
Highest density interval:
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

nbText: "The intervals are:"
nbCode:
  let 
    (b0HdiMin, b0HdiMax) = hdi(b0Samples[burnin..^1], 0.95)
    (b1HdiMin, b1HdiMax) = hdi(b1Samples[burnin..^1], 0.95)
    (sdHdiMin, sdHdiMax) = hdi(sdSamples[burnin..^1], 0.95)
  echo b0HdiMin, " ", b0HdiMax
  echo b1HdiMin, " ", b1HdiMax
  echo sdHdiMin, " ", sdHdiMax


nbText: md"""
Look at the output.
Trace plots:
"""
nbCode:
  let 
    ixs = toSeq(0 ..< b0Samples.len-burnin)
    df = seqsToDf({
      "ixs":ixs, 
      "b0":b0Samples[burnin..^1],
      "b1":b1Samples[burnin..^1],
      "sd":sdSamples[burnin..^1]})
  ggplot(df, aes("b0")) +
    geom_histogram() +
    ggsave("plots/hist-b0.png")
  
  ggplot(df, aes("b1")) +
    geom_histogram() +
    ggsave("plots/hist-b1.png")
  
  ggplot(df, aes("sd")) +
    geom_histogram() +
    ggsave("plots/hist-sd.png")
nbImage("plots/hist-b0.png")
nbImage("plots/hist-b1.png")
nbImage("plots/hist-sd.png")


nbText: md"""
Histograms:
"""
nbCode:
  ggplot(df, aes(x="ixs", y="b0")) + 
    geom_line() +
    ggsave("plots/samples-b0.png")
  
  ggplot(df, aes(x="ixs", y="b1")) + 
    geom_line() +
    ggsave("plots/samples-b1.png")
  
  ggplot(df, aes(x="ixs", y="sd")) + 
    geom_line() +
    ggsave("plots/samples-sd.png")
nbImage("plots/samples-b0.png")
nbImage("plots/samples-b1.png")
nbImage("plots/samples-sd.png")


nbSave
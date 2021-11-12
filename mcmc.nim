import math
import distributions
import random 
import stats
import sequtils
import algorithm
import datamancer
import ggplotnim
import strformat

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

proc logLikelihood(x, y: seq[float], b0, b1, sd: float): float = 
  var likelihoods = newSeq[float](y.len) 
  for i in 0..<y.len: 
    let pred = b0 + b1 * x[i]
    likelihoods[i] = ln(initNormalDistribution(pred, sd).pdf(y[i]))
  result = sum(likelihoods) 

proc logPrior(b0, b1, sd: float): float = 
  let 
    b0Prior = ln(initNormalDistribution(0.0, 1.0).pdf(b0))
    b1Prior = ln(initNormalDistribution(1.0, 1.0).pdf(b1))
    sdPrior = ln(initGammaDistribution(1.0, 1.0).pdf(1/sd))
  result = b0Prior + b1Prior + sdPrior

proc logPosterior(x, y: seq[float], b0, b1, sd: float): float = 
  # if sd > 0.0:
  let 
    like = logLikelihood(x=x, y=y, b0=b0, b1=b1, sd=sd)
    prior = logPrior(b0=b0, b1=b1, sd=sd)
  result = like + prior
  # else:
    # result = 0.0

proc mcmc(x, y: seq[float], nSamples: int, b0Start, b1Start, sdStart: float): 
    (seq[float], seq[float], seq[float]) =
  var 
    b0Samples = newSeq[float](nSamples+1) 
    b1Samples = newSeq[float](nSamples+1) 
    sdSamples = newSeq[float](nSamples+1) 

  b0Samples[0] = b0Start 
  b1Samples[0] = b1Start
  sdSamples[0] = sdStart 
  
  for i in 1..nSamples:
    let 
      prevB0 = b0Samples[i-1]
      prevB1 = b1Samples[i-1]
      prevSd = sdSamples[i-1]
      propB0 = gauss(prevB0, 0.3) 
      propB1 = gauss(prevB1, 0.1)
      propSd = gauss(prevSd, 0.3)
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
  result = (b0Samples, b1Samples, sdSamples) 

randomize()

### Generate data
var
  N = 100
  b0 = 0.0
  b1 = 1.0
  sd = 10.0
  x = newSeq[float](N)
  y = newSeq[float](N)
for i in 0..<N: 
  x[i] = rand(-100.0..100.0).float 
  y[i] = b0 + b1 * x[i] + gauss(0.0, sd) 

var sim = seqsToDf(x, y)
ggplot(sim, aes("x", "y")) +
  geom_point() +
  ggsave("plots/simulated-data.png")

### Run Analysis
let 
  nSamples = 100000
  (b0Samples, b1Samples, sdSamples) = mcmc(x=x, y=y, nSamples=nSamples, 
  b0Start=b0, b1Start=b1, sdStart=sd)

### Summarize output
let  
  burnin = 1001
  meanB0 = mean(b0Samples[burnin..^1])
  meanB1 = mean(b1Samples[burnin..^1])
  meanSd = mean(sdSamples[burnin..^1])
  ixs = toSeq(0 ..< b0Samples.len-burnin)
 

# Compute HDIs
let (b0HdiMin, b0HdiMax) = hdi(b0Samples[burnin..^1], 0.95)
let (b1HdiMin, b1HdiMax) = hdi(b1Samples[burnin..^1], 0.95)
let (sdHdiMin, sdHdiMax) = hdi(sdSamples[burnin..^1], 0.95)

echo "b0: ", meanB0
echo b0HdiMin, " ", b0HdiMax
echo "b1: ", meanB1
echo b1HdiMin, " ", b1HdiMax
echo "sd: ", meansd
echo sdHdiMin, " ", sdHdiMax

# Plots
let df = seqsToDf({
    "ixs":ixs, 
    "b0":b0Samples[burnin..^1],
    "b1":b1Samples[burnin..^1],
    "sd":sdSamples[burnin..^1]})

ggplot(df, aes(x="ixs", y="b0")) + 
  geom_line() +
  ggsave("plots/samples-b0.png")

ggplot(df, aes(x="ixs", y="b1")) + 
  geom_line() +
  ggsave("plots/samples-b1.png")

ggplot(df, aes(x="ixs", y="sd")) + 
  geom_line() +
  ggsave("plots/samples-sd.png")


ggplot(df, aes("b0")) +
  geom_histogram() +
  ggsave("plots/hist-b0.png")

ggplot(df, aes("b1")) +
  geom_histogram() +
  ggsave("plots/hist-b1.png")

ggplot(df, aes("sd")) +
  geom_histogram() +
  ggsave("plots/hist-sd.png")




# Save simulated data & mcmc sample
let data = open("data.csv", fmWrite)
data.writeLine("x,y")
for i in 0..<x.len: 
  data.writeLine(&"{x[i]},{y[i]}")

let output = open("samples.csv", fmWrite) 
output.writeLine("b0,b1,sd")
for i in 0 .. nSamples:
  output.writeLine(&"{b0Samples[i]},{b1Samples[i]},{sdSamples[i]}")


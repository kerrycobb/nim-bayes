import ggplotnim
import datamancer
import sequtils 
import stats


# proc potentialScaleReductionFactor(chains: seq[seq[float]]):

# proc effectiveSampleSize(samples: seq[float]): float = 
  


# # let simData = readCsv("data.csv")
# # ggplot(simData, aes("x", "y")) +
# #   geom_point() +
# #   ggsave("simulated-data.png")

var 
  # burn = 1000
  burn = 0
  samples = readCsv("samples.csv")
samples["ix"] = toSeq(0 ..< samples.len)
let
  burnin = samples[burn..^1]

echo samples

# ggplot(burnin, aes(x="ix", y="b0")) + 
#   geom_line() +
#   ggsave("plots/samples-b0.png")

# ggplot(burnin, aes(x="ix", y="b1")) + 
#   geom_line() +
#   ggsave("plots/samples-b1.png")

# ggplot(burnin, aes(x="ix", y="sd")) + 
#   geom_line() +
#   ggsave("plots/samples-sd.png")

# ggplot(burnin, aes("b0")) +
#   geom_histogram() +
#   ggsave("plots/hist-b0.png")

# ggplot(burnin, aes("b1")) +
#   geom_histogram() +
#   ggsave("plots/hist-b1.png")

# ggplot(burnin, aes("sd")) +
#   geom_histogram() +
#   ggsave("plots/hist-sd.png")





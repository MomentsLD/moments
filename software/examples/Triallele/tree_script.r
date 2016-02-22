
library('ape')

sink("intervals_nu2.0_tau.1_ns20.txt")
for (n in 1:10000)
{
system("./ms 20 1 -eN .025 .5 -T > tree.tre", intern = FALSE, wait = TRUE, input = NULL)
MyTree <- read.tree("tree.tre")
ints = coalescent.intervals(MyTree)
line = ints[2][[1]]
cat(line)
cat("\n")
}
sink()


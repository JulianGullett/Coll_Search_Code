# null model (Simpler model)
llNull <- -9455
dfNull <- 1            # how many free (non-fixed) parameters in the model?

# alternative model (More complex model)
llAlternative <- -5291    # 
dfAlaternative <- 4       #

# Comparisons
# q = null model / alternative <= FOR raw likelihood (not logLik)
# p-value is -2 * q
# but with LL it should be chi = -2 * (llNull - llAlternative)
# difference in size of the parameter space between null and alternative models.
qNullVSAlternative <- -2 * (llNull - llAlternative)
pchisq(qNullVSAlternative, dfAlaternative-dfNull, lower.tail = FALSE)


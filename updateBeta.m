function beta = updateBeta(n, m, t, S,instanceWiseError,weights)
    targetSumWeights = m/(n+m) + (t/(S-1))*(n/(n+m));
    fun=@(x)targetSumWeights * sum(weights(1:n).*(x.^instanceWiseError(1:n,t)))...
    +(targetSumWeights - 1) * sum(weights(n+1:n+m));
    beta =Division(fun,1e-4,0,1);
end 
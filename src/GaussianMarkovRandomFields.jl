module GaussianMarkovRandomFields

using LDLFactorizations
using LinearAlgebra, SparseArrays
using Distributions
using Random

export GMRF, cholesky_ldl, logdet_ldl, prec

function cholesky_ldl(A)
    F = ldl(A)
    L = cholesky(F)
    return L, F.P
end

function LinearAlgebra.cholesky(F::LDLFactorizations.LDLFactorization)
    # https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition
    all(diag(F.D) .> 0) || PosDefException(0)
    L = ((F.L + I) * sqrt.(F.D))#[invperm(F.P), invperm(F.P)]
    return L
end

function logdet_from_chol(L)
    return 2 * sum(log(L[i]) for i in diagind(L))
end

function logdet_ldl(M::AbstractMatrix)
    L, P = cholesky_ldl(M)
    return logdet_from_chol(L)
end

struct GMRF{Tv<:AbstractVector{<:Real},
        Tm<:AbstractMatrix{<:Real},
        Tl<:AbstractMatrix{<:Real}} <: ContinuousMultivariateDistribution
    μ::Tv
    Q::Tm
    L::Tl
end

function GMRF(μ::AbstractVector, Q::AbstractMatrix)
    n = LinearAlgebra.checksquare(Q)
    length(μ) == n || DimensionMismatch("The dimensions of μ and Q are inconsistent.")
    fac = ldl(Q)
    L, P = cholesky_ldl(Q)
    return GMRF(μ, Q, L)
end

function GMRF(Q::AbstractMatrix{T}) where T
    n = LinearAlgebra.checksquare(Q)
    return GMRF(zeros(T, n), Q)
end

function Distributions._logpdf(d::GMRF, x::AbstractVector{T}) where T
    k = length(d.μ)
    ld = logdet_from_chol(d.L)
    x0 = x .- d.μ
    return -0.5 * (-ld + dot(x0, d.Q, x0) + k*log(2pi))
end

Base.length(d::GMRF) = length(mean(d))
Base.eltype(d::GMRF) = eltype(mean(d))
prec(d::GMRF) = d.Q
Distributions.mean(d::GMRF) = d.μ
Distributions.var(d::GMRF) = 1 ./ diag(prec(d))
Distributions.cov(d::GMRF) = error( "ERROR: The covariance matrix of a GMRF `g` can often be " *
    "dense and can cause the computer to run out of memory. If you are sure you have "*
    "enough memory, you can invert the precision matrix with `inv(Matrix(prec(g)))`")

function Distributions._rand!(rng::Random.AbstractRNG, d::GMRF, x::AbstractArray)
    z = randn(rng, size(x))
    x .= mean(d) .+ d.L \ z
end

end # module

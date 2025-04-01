
# using ChainRulesCore
using SparseArrays
using LinearAlgebra
# using SparseInverseSubset
using Distributions, PDMats
import Zygote, Mooncake, Enzyme, ForwardDiff, ReverseDiff, FiniteDiff
using DifferentiationInterface
using DifferentiationInterfaceTest
using DataFrames
using BenchmarkTools
import GaussianMarkovRandomFields

struct GMRF{T1,T2} <: Distributions.AbstractMvNormal
    μ::T1
    Q::T2
end

function GMRF(Q)
    GMRF(zeros(size(Q, 1)), Q)
end

function Distributions.logpdf(d::GMRF, x::AbstractVector{T}) where T
    k = length(x)
    xm = x .- d.μ
    -0.5 * (-logdet(d.Q) + dot(xm, d.Q, xm) + k*log(2π))
end


n = 1000
function f(u, p)
    Q = sparse(1:length(u), 1:length(u), exp.(u))
    logpdf(GMRF(zeros(length(u)), Q), p.x)
end

p = (x = randn(n),)
u = ones(n)
f(u, p)
@code_warntype f(u, p)

backends = [
    AutoZygote(),
    AutoMooncake(config=nothing),
    AutoForwardDiff(),
    AutoReverseDiff(),
    AutoReverseDiff(true),
    # AutoEnzyme() # broken, see below
]

#=
Enzyme compilation failed due to illegal type analysis.
This usually indicates the use of a Union type, which is not fully supported with Enzyme.API.strictAliasing set to true [the default].
Ideally, remove the union (which will also make your code faster), or try setting Enzyme.API.strictAliasing!(false) before any autodiff call.
To toggle more information for debugging (needed for bug reports), set Enzyme.Compiler.VERBOSE_ERRORS[] = true (default false)

Caused by:
Stacktrace:
[1] getproperty
  @ ~/.julia/juliaup/julia-1.11.4+0.x64.linux.gnu/share/julia/stdlib/v1.11/SparseArrays/src/solvers/umfpack.jl:781
=#


∇f(u) = gradient(f, AutoFiniteDiff(), u, Constant(p))

scenarios = [
    Scenario{:gradient, :out}(f, u; contexts=(Constant(p),), res1=∇f(u))
]

test_differentiation(backends, scenarios)
bench = benchmark_differentiation(backends, scenarios)
sort(bench, :time)

function f1(u, p)
    Q = sparse(1:length(u), 1:length(u), exp.(u))
    logpdf(GaussianMarkovRandomFields.GMRF(zeros(length(u)), Q), p.x)
end
@btime f(u, p)
@btime f1(u, p)

gradient(f, AutoMooncake(config=nothing), u, Constant(p))
# gradient(f1, AutoMooncake(config=nothing), u, Constant(p)) #error
@btime gradient(f, AutoReverseDiff(), u, Constant(p));
@btime gradient(f1, AutoReverseDiff(), u, Constant(p));

@btime gradient(f, AutoZygote(), u, Constant(p));
@btime gradient(f1, AutoZygote(), u, Constant(p));
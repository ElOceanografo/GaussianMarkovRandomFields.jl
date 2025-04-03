using GaussianMarkovRandomFields

using Random
using BenchmarkTools
using SparseArrays, LinearAlgebra
using LDLFactorizations
using Distributions, PDMats

using DifferentiationInterface
using DifferentiationInterfaceTest
import FiniteDiff, ForwardDiff, ReverseDiff, Zygote, Mooncake
using Turing

using Test

function make_precision(ρ, k)
    return spdiagm(-1 => -ρ*ones(k-1), 0 => ones(k), 1 => -ρ*ones(k-1))
end

function test_variables(k=50)
    ρ = 0.4
    μ = zeros(k)
    Q = make_precision(ρ, k)
    return k, ρ, μ, Q
end

@testset "Constructors" begin
    k = 10
    T = Float32
    Q = sprandn(T, k, k, 0.1)
    Q = Q'Q + I
    println(eltype(Q))
    μ = zeros(T, k)
    # L, P = cholesky_ldl(Q)
    F = cholesky(Q)

    g1 = GMRF(Q)
    g2 = GMRF(μ, Q)
    # g3 = GMRF(μ, Q, F)
    # g3 = GMRF(μ, Q, L)
    # g4 = GMRF(μ, F)
    # g5 = GMRF(F)
    for g in [g2]
        @test mean(g1) == mean(g)
        @test prec(g1) == prec(g)
        # @test all(sparse(F.L) .≈ sparse(g.F)) not sure why this is failing
        @test eltype(g) == T
    end
    # for gf in [g4, g5]
    #     @test all(mean(gf) .≈ mean(g1))
    #     @test all(prec(gf) .≈ prec(g1))
    # end
    #  @test all(var(g1) .≈ 1 ./ diag(Q))
    #  Tests for variance below
    @test_throws ErrorException cov(g1)
end

# @testset "Linear Algebra" begin
#     k, ρ, μ, Q = test_variables()

#     L, P = cholesky_ldl(Q)
#     L1 = sparse(cholesky(Q).L)
#     @test all(Matrix(L * L') .≈ Matrix(Q)[P, P])
#     @test prod(diag(L).^2) .≈ det(Q)
#     @test all(Matrix(L) .≈ Matrix(sparse(cholesky(Q).L)))
#     @test logdet_ldl(Q) .≈ log(det(Matrix(Q)))
#     @test logdet_ldl(Q) .≈ logdet(Matrix(Q))
# end

@testset "Likelihood" begin
    k, ρ, μ, Q = test_variables()
    Qd = Matrix(Q)

    d = GMRF(μ, Q)
    dd = MvNormalCanon(PDMat(Qd))
    dd1 = MvNormal(zeros(k), inv(Symmetric(Matrix(Q))))
    Random.seed!(1)
    x = rand(d)
    @test logpdf(d, x) ≈ logpdf(dd, x)
    @test logpdf(d, x) ≈ logpdf(dd1, x)
end

@testset "Performance" begin
    Random.seed!(1)
    k, ρ, μ, Q = test_variables(2000)
    Qd = Matrix(Q)

    # t_logdet_ldl = @benchmark logdet_ldl($Q)
    # t_logdet_ldl_d = @benchmark logdet_ldl($Qd)
    # t_logdet_d = @benchmark logdet($Qd)
    # println(" ")
    # println("logdet_ldl (sparse): $(minimum(t_logdet_ldl))")
    # println("logdet_ldl (dense): $(minimum(t_logdet_ldl_d))")
    # println("logdet (dense): $(minimum(t_logdet_d))")

    # @test minimum(t_logdet_ldl.times) .< minimum(t_logdet_ldl_d.times)
    # @test minimum(t_logdet_ldl.times) .< minimum(t_logdet_d.times)


    d = GMRF(μ, Q)
    d1 = MvNormalCanon(PDSparseMat(Q))
    dd = MvNormalCanon(PDMat(Qd))
    dd1 = MvNormal(zeros(k), inv(Symmetric(Matrix(Q))))
    x = rand(d)

    t_logpdf_s = @benchmark logpdf($d, $x)
    t_logpdf_s1 = @benchmark logpdf($d1, $x)
    t_logpdf_d = @benchmark logpdf($dd, $x)
    t_logpdf_d1 = @benchmark logpdf($dd1, $x)
    println(" ")
    println("logpdf (sparse, GMRF): $(minimum(t_logpdf_s))")
    println("logpdf (sparse, MvNormalCanon): $(minimum(t_logpdf_s1))")
    println("logpdf (dense, MvNormalCanon): $(minimum(t_logpdf_d))")
    println("logdet (dense, MvNormal): $(minimum(t_logpdf_d1))")

    # @test minimum(t_logpdf_s.times) .< minimum(t_logpdf_s1.times)
    # @test minimum(t_logpdf_s.times) .< minimum(t_logpdf_d.times)
    # @test minimum(t_logpdf_s.times) .< minimum(t_logpdf_d1.times)
end

@testset "AD" begin
    Random.seed!(1)

    function f(u, p)
        Q = sparse(1:length(u), 1:length(u), exp.(u))
        logpdf(GMRF(zeros(length(u)), Q), p.x)
    end
    
    n = 1000
    p = (x = randn(n),)
    u = ones(n)

    backends = [
        AutoZygote(),
        AutoMooncake(config=nothing),
        AutoForwardDiff(),
        AutoReverseDiff(),
        AutoReverseDiff(compile=true),
    ]

    ∇f(u) = gradient(f, AutoFiniteDiff(), u, Constant(p))

    scenarios = [
        Scenario{:gradient, :out}(f, u; contexts=(Constant(p),), res1=∇f(u))
    ]

    test_differentiation(backends, scenarios)
    bench = benchmark_differentiation(backends, scenarios)
    sort!(bench, :time)
    show(bench)
    println()

end

# @testset "Turing" begin
#     Random.seed!(1)
#     k, ρ, μ, Q = test_variables()
#     d = GMRF(μ, Q)
#     x = rand(d)

#     @model function GMRFTest(x)
#         n = length(x)
#         ρ ~ Uniform(0, 0.5)
#         Q = make_precision(ρ, n)
#         x ~ GMRF(Q)
#     end

#     m = GMRFTest(x)

#     backends = [
#         AutoZygote(),
#         AutoMooncake(config=nothing),
#         AutoForwardDiff(),
#         AutoReverseDiff(),
#         AutoReverseDiff(compile=true),
#     ]
    
#     for backend in backends
#         @test typeof(sample(m, NUTS(), 50, adtype=backend)) <: Turing.Chains
#     end
# end

@testset "Marginal variance" begin
    ### Setup AR1 example
    function ar1_precision(n::I, ρ::T) where {I<:Integer,T}
        offdiagvec = -ρ * ones(T, n - 1)
        diagvec = T[one(ρ); ones(T, n - 2) .+ ρ^2; one(ρ)]
        return  spdiagm(-1 => offdiagvec, 0 => diagvec, 1 => offdiagvec)
    end
    function ar1_precision(n, ρ, σ)
        ρ, σ = promote(ρ, σ)
        return 1 / σ^2 * ar1_precision(n, ρ)
    end
    #  function ar1_precision(n, ρ::T, σ::AbstractVector{T}) where T
        #  scalemat = Diagonal(σ)
        #  return scalemat * ar1_precision(n, ρ) * scalemat
    #  end

    k, ρ, μ, Q = test_variables()

    Q1 = ar1_precision(k, ρ)
    Q2 = ar1_precision(k, ρ, 2.0)
    # Qns = ar1_precision(k, ρ, range(0.1, 1.9, length = k))

    n = 10_000   # number of measurements in time series
    σ = 2.0      # innovation variance
    ρ = 0.4      # autocorrelation coefficient

    Qbig = ar1_precision(n, ρ, σ)
    dbig = GMRF(Qbig)

    # Compare to the analytic marginal variance for an AR(1) process and ensure
    # that we don't overflow the stack on large precision matrices.
    @test all(var(dbig) .≈ σ^2 / (1 - ρ^2))


    # Test that all elements are computed accurately
    Q_small = ar1_precision(k, ρ)
    # Convert to generic sparse matrix
    L_small = dropzeros(sparse(cholesky(Matrix(Q_small)).L))
    # Invert directly to get full covariance matrix for comparison
    Σ_small = inv(Matrix(Q_small))

    Σ_elem = similar(Σ_small)
    for j in k:-1:1, i in k:-1:1
        Σ_elem[i, j] = GaussianMarkovRandomFields.cov_element(L_small, i, j)
    end

    @test all(Σ_elem .≈ Σ_small)
end

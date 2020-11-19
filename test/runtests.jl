using GaussianMarkovRandomFields

using Random
using BenchmarkTools
using SparseArrays, LinearAlgebra
using LDLFactorizations
using Distributions, PDMats

using ForwardDiff
using ReverseDiff
# using Zygote # broken
using FiniteDiff
using Turing

using Test

function make_precision(ρ, k)
    return spdiagm(-1 => -ρ*ones(k-1), 0 => ones(k), 1 => -ρ*ones(k-1))
end

function test_variables()
    k = 50
    ρ = 0.4
    μ = zeros(k)
    Q = make_precision(ρ, k)
    return k, ρ, μ, Q
end

@testset "Constructors" begin
    k = 10
    T = Float32
    Q = sprandn(k, k, 0.1)
    Q = Q'Q + I
    μ = zeros(T, k)
    L, P = cholesky_ldl(Q)

    g1 = GMRF(Q)
    g2 = GMRF(μ, Q)
    g3 = GMRF(μ, Q, L)
    @test g1 == g2
    @test g1 == g3
    @test mean(g1) == μ
    @test prec(g1) == Q
    @test eltype(g2) == T
    @test all(var(g1) .≈ 1 ./ diag(Q))
    @test_throws ErrorException cov(g1)
end

@testset "Linear Algebra" begin
    k, ρ, μ, Q = test_variables()

    L, P = cholesky_ldl(Q)
    L1 = sparse(cholesky(Q).L)
    @test all(Matrix(L * L') .≈ Matrix(Q)[P, P])
    @test prod(diag(L).^2) .≈ det(Q)
    @test all(Matrix(L) .≈ Matrix(sparse(cholesky(Q).L)))
    @test logdet_ldl(Q) .≈ log(det(Matrix(Q)))
    @test logdet_ldl(Q) .≈ logdet(Matrix(Q))
end

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
    k, ρ, μ, Q = test_variables()
    Qd = Matrix(Q)

    t_logdet_ldl = @benchmark logdet_ldl($Q)
    t_logdet_ldl_d = @benchmark logdet_ldl($Qd)
    t_logdet_d = @benchmark logdet($Qd)
    println(" ")
    println("logdet_ldl (sparse): $(minimum(t_logdet_ldl))")
    println("logdet_ldl (dense): $(minimum(t_logdet_ldl_d))")
    println("logdet (dense): $(minimum(t_logdet_d))")

    @test minimum(t_logdet_ldl.times) .< minimum(t_logdet_ldl_d.times)
    @test minimum(t_logdet_ldl.times) .< minimum(t_logdet_d.times)


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
    println("logpdf (sparse, LDL): $(minimum(t_logpdf_s))")
    println("logpdf (sparse, MvNormalCanon): $(minimum(t_logpdf_s1))")
    println("logpdf (dense, MvNormalCanon): $(minimum(t_logpdf_d))")
    println("logdet (dense, MvNormal): $(minimum(t_logpdf_d1))")

    @test_broken minimum(t_logpdf_s.times) .< minimum(t_logpdf_s1.times)
    @test minimum(t_logpdf_s.times) .< minimum(t_logpdf_d.times)
    @test minimum(t_logpdf_s.times) .< minimum(t_logpdf_d1.times)
end

@testset "AD" begin
    Random.seed!(1)
    k, ρ, μ, Q = test_variables()
    d = GMRF(μ, Q)
    x = rand(d)

    logistic(x) = 1 / (1 + exp(-x))

    function test_loglik(θ)
        σ = exp(θ[1])
        ρ = σ/2 * logistic(θ[2])
        μ = θ[3:end]
        Q = make_precision(ρ, length(μ))
        d = GMRF(μ, Q)
        return logpdf(d, x)
    end

    θ = randn(k + 2)
    Δfinite = FiniteDiff.finite_difference_gradient(test_loglik, θ)
    @test all(ForwardDiff.gradient(test_loglik, θ) .≈ Δfinite)
    @test all(ReverseDiff.gradient(test_loglik, θ) .≈ Δfinite)
    # Zygote broken for some reason
    # @test all(Zygote.gradient(test_loglik, θ) .≈ Δfinite)
end

@testset "Turing" begin
    Random.seed!(1)
    k, ρ, μ, Q = test_variables()
    d = GMRF(μ, Q)
    x = rand(d)

    @model function GMRFTest(x)
        n = length(x)
        ρ ~ Uniform(0, 0.5)
        Q = make_precision(ρ, n)
        x ~ GMRF(Q)
    end

    m = GMRFTest(x)
    Turing.setadbackend(:forwarddiff)
    @test typeof(sample(m, NUTS(), 50)) <: Turing.Chains
    Turing.setadbackend(:reversediff)
    @test typeof(sample(m, NUTS(), 50)) <: Turing.Chains
end

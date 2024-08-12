## this code is for the project 'Testing and Identifying Substitution and Complementarity Patterns' by Rui Wang
## it is implemneting four different estimators for substitution parameters

using Distributions, LinearAlgebra, GLM, Optim, Random

Random.seed!(1234)

## Initial configuration for simulation
B = 500                # Number of simulations
N = 1000               # Sample size
T = 2                  # Two periods panel
dx = 2                 # Dimension of X
dz = 2                 # Dimension of Z

## True coefficients
beta0 = ones(dx, 1)    # True coefficient for a single good
gamma0 = ones(dz, 1)   # True coefficient for incremental utility

## Coefficients for fixed effects
eta1 = 0.5 * ones(dx, 1)
eta0 = 0

## Two-step estimator in the paper
beta_twostep = ones(dx - 1, B)  # Two-step estimator for beta
gamma_twostep = ones(dz - 1, B) # Two-step estimator for gamma
err = ones(B)                   # Percentage of having wrong signs of substitution patterns

## Parametric estimator 
beta_par = ones(dx - 1, B)
gamma_par = ones(dz - 1, B)
err_par = ones(B)

## Two estimators assuming no bundles
beta_non = ones(dx - 1, B)  # Estimator under stationarity but assuming no bundles
beta_fl = ones(dx - 1, B)   # Fixed effect logit estimator

## Helper functions
sg(y) = sign.(y)                        # Determine the sign of y
D(x) = x[:, dx + 1:end] - x[:, 1:dx]    # Difference of characteristics of two periods
dp(y) = y[:, 2] - y[:, 1]               # Difference of choice probabilities
G(x) = abs.(x)                          # Function for absolute value
G1(x) = x * (x >= 0)                    # Function for nonnegative value

## First step estimator to predict consumers' choice 
alpha = 10^(-3)                             # Learning rate
ite = 200                                   # Number of iterations
phi(v) = 1 ./ (1 .+ exp.(-v))               # Activation function
dphi(v) = exp.(-v) ./ (1 .+ exp.(-v)).^2    # Derivative of phi

## Function for first step estimator
function neu(x, y)
    """
    - input:
    x: covariates
    y: discrete dependent varaiable
    
    - output: choice probability for each option
    """
    dx = size(x, 2)               # dimension of x
    dy = size(y, 2)               # dimension of y
    n = size(y, 1)                # sample size
    Xt = [ones(n, 1) x]           # covariate
    w = ones(dx + 1, dy)          # initial weights
    for i = 1:ite
        for j = 1:n
            pre = phi(Xt[j, :]' * w)    # prediction
            err = y[j, :]' - pre        # prediction error
            w += 2 * alpha * Xt[j, :] * err * dphi(Xt[j, :]' * w)  # update weight
        end
    end
    phat = phi(Xt * w)   # predicted choice probability
    return phat
end


## Criterion function in this paper
function mom(theta, XA, XB, Z, p)
    """
    - theta: true parameter
    - XA(XB): covariate for good A(B) at all periods
    - p: estimated choice probability for Y1, Y2, Y3, Y0
    """
    beta = [1; theta[1:dx - 1]]
    gamma = [1; theta[dx:end]]
    Gamma = Z * gamma            # Incremental utility

    deltaA = D(XA) * beta        # Variation in covariate index for good A
    deltaB = D(XB) * beta        # Variation in covariate index for good B

    p1 = p[:, 1:T]               # Estimated choice probability for choosing only good A
    p2 = p[:, T + 1:2 * T]       # Probability of choosing only good B
    p0 = p[:, 2 * T + 1:3 * T]   # Neither
    p3 = p[:, 3 * T + 1:4 * T]   # Two goods together
    p13 = p1 + p3                # Demand for good A
    p23 = p2 + p3                # Demand for good B

    ## ID1--construct index for a single good
    I1 = 1 .- ((sg(dp(p1)) .* deltaA .> 0) .| (sg(dp(p1)) .* deltaB .< 0))
    I2 = 1 .- ((sg(dp(p2)) .* deltaA .< 0) .| (sg(dp(p2)) .* deltaB .> 0))
    I3 = 1 .- ((sg(dp(p3)) .* deltaA .> 0) .| (sg(dp(p3)) .* deltaB .> 0))
    I0 = 1 .- ((sg(dp(p0)) .* deltaA .< 0) .| (sg(dp(p0)) .* deltaB .< 0))

    # Construct objective function using a single choice
    Q1 = mean(G1(dp(p1)) .* I1 + G1(dp(p2)) .* I2 + G1(dp(p3)) .* I3 + G1(dp(p0)) .* I0)

    ## ID2--construct index for demand of good A and good B
    I13 = 1 .- ((sg(dp(p13)) .* deltaA .> 0) .| ((sg(dp(p13)) .* (deltaA + sg(Gamma) .* deltaB) .> 0)))
    I23 = 1 .- ((sg(dp(p23)) .* deltaB .> 0) .| ((sg(dp(p23)) .* (deltaB + sg(Gamma) .* deltaA) .> 0)))

    # Objective function using conditional demand
    Q2 = mean(G1(dp(p13)) .* I13 + G1(dp(p23)) .* I23)

    ## ID3--construct index for sum of two choice probabilities
    Ib1(t1) = 1 .- (Gamma .< min.((-1) ^ t1 .* deltaA, (-1) ^ (t1 - 1) .* deltaB)) .*
                     ((-1) ^ t1 .* (deltaA - deltaB) .> 0)
    Ib2(t1) = 1 .- (Gamma .> max.((-1) ^ (t1 - 1) .* deltaA, (-1) ^ (t1 - 1) .* deltaB)) .*
                     ((-1) ^ t1 .* (deltaA + deltaB) .> 0)

    # Objective function by sum of two choice probabilities
    L1(t1, t2) = p1[:, t1] + p2[:, t2] .- 1
    L2(t1, t2) = p3[:, t1] + p0[:, t2] .- 1
    Q3 = mean(G1(L1(2, 1)) .* Ib1(2) + G1(L1(1, 2)) .* Ib1(1) + G1(L2(2, 1)) .* Ib2(2) + G1(L2(1, 2)) .* Ib2(1))

    return Q1 + Q2 + Q3
end


## Mixed-effect logit estimator: epj~Gumbel, alphaj=eta0 + Xbar*eta1 + vj where vj~N(0,1)
# Use the simulated method of moments to calculate choice probabilities
S = 20                           # Number of simulations
ep0_s = rand(Gumbel(), N, T, S)  # Simulated shocks for baseline
epA_s = rand(Gumbel(), N, T, S)  # Simulated shocks for good A
epB_s = rand(Gumbel(), N, T, S)  # Simulated shocks for good B
vA_s = randn(N, S)               # Simulated error term in fixed effect for good A
vB_s = randn(N, S)               # Simulated error term in fixed effect for good B

muA_s = zeros(N, T, S)           # Sum of two error terms: epA + vA
muB_s = zeros(N, T, S)           # Sum of two error terms: epB + vB
for t = 1:T
    muA_s[:, t, :] = vA_s + epA_s[:, t, :] - ep0_s[:, t, :]
    muB_s[:, t, :] = vB_s + epB_s[:, t, :] - ep0_s[:, t, :]
end

## Criterion function for the parametric method
function par_sim(theta, XA, XB, Z, Y1, Y2, Y3)
    """
    - theta: true parameter
    - XA(XB): covariate for good A(B) at all periods
    - Z: covariate for substitution patters
    - Y1, Y2, Y3: dependent variables
    """
    beta = [1; theta[1:dx-1]]
    gamma = [1; theta[dx:dx+dz-2]]
    eta1 = theta[dx+dz-1:end-1]
    eta0 = theta[end]

    alphaA = eta0 .+ ((XA[:, 1:dx] + XA[:, dx+1:end]) / T) * eta1  # Mean fixed effect for good A
    alphaB = eta0 .+ ((XB[:, 1:dx] + XB[:, dx+1:end]) / T) * eta1  # Mean fixed effect for good B
    delA = [XA[:, 1:dx] * beta XA[:, dx+1:end] * beta] .+ alphaA   # Mean utility for good A
    delB = [XB[:, 1:dx] * beta XB[:, dx+1:end] * beta] .+ alphaB   # Mean utility for good B
    Gam = Z * gamma                                                # Incremental utility

    # Simulated choice probability as a function of parameters
    p1_s = mean((muA_s .+ delA .>= max.(muB_s .+ delB, 0)) .* (muB_s .<= -delB .- Gam), dims=3)[:, :]
    p2_s = mean((muB_s .+ delB .>= max.(muA_s .+ delA, 0)) .* (muA_s .<= -delA .- Gam), dims=3)[:, :]
    p3_s = mean((muA_s .+ delA .+ Gam .>= max.(-muB_s .- delB, 0)) .* (muB_s .>= -delB .- Gam), dims=3)[:, :]

    X = [XA XB Z]'

    # Moment conditions for the two periods
    Q_s1 = [X * (Y1 - p1_s)[:, 1]; X * (Y2 - p2_s)[:, 1]; X * (Y3 - p3_s)[:, 1]] / N
    Q_s2 = [X * (Y1 - p1_s)[:, 2]; X * (Y2 - p2_s)[:, 2]; X * (Y3 - p3_s)[:, 2]] / N

    Q = Q_s1' * Q_s1 + Q_s2' * Q_s2
    return Q
end

## Nonparametric estimator which does not allow bundles: gamma=-infty
function mom1(beta, XA, XB, p)
    """
    - theta: true parameter
    - XA(XB): covariate for good A(B) at all periods
    - p: estimated choice probability for Y1, Y2, Y3, Y0
    """
    b = [1; beta]

    deltaA = D(XA) * b               # Variation in covariate index for good A
    deltaB = D(XB) * b               # Variation in covariate index for good B

    p1 = p[:, 1:T]                   # Estimated choice probability for choosing only good A
    p2 = p[:, 2*T-1:2*T]             # Estimated choice probability for choosing only good B
    p0 = p[:, 3*T-1:3*T]             # Estimated choice probability for choosing neither

    ## Construct index for a single good
    I1 = 1 .- ((sg(dp(p1)) .* deltaA .> 0) .| (sg(dp(p1)) .* (deltaA - deltaB) .> 0))
    I2 = 1 .- ((sg(dp(p2)) .* deltaB .> 0) .| (sg(dp(p2)) .* (deltaB - deltaA) .> 0))
    I0 = 1 .- ((sg(dp(p0)) .* deltaA .< 0) .| (sg(dp(p0)) .* deltaB .< 0))

    # Construct objective function using a single choice
    Q = mean(G(dp(p1)) .* I1 + G(dp(p2)) .* I2 + G(dp(p0)) .* I0)
    return Q
end


## Chamberlain's fixed-effect logit model which does not allow bundles: gamma=-infty
function momlogit(beta, XA, XB, Y)
    """
    - theta: true parameter
    - XA(XB): covariate for good A(B) at all periods
    - Y: dependent variables
    """
    b = [1; beta]

    # Indicator function for choosing c1 at t=1 and c2 at t=2
    dc(c1, c2) = (Y[:, 1] .== c1) .* (Y[:, 2] .== c2)

    # Choice probability calculations
    p01 = exp.(D(XA) * b) ./ (1 .+ exp.(D(XA) * b))  # choosing (0,1) conditional on y1+y2=1
    p02 = exp.(D(XB) * b) ./ (1 .+ exp.(D(XB) * b))
    p12 = exp.((D(XB) - D(XA)) * b) ./ (1 .+ exp.((D(XB) - D(XA)) * b))

    # Calculating the log-likelihood for each pair of choices
    Q1 = mean(dc(0, 1) .* log.(p01) + dc(1, 0) .* log.(1 .- p01))
    Q2 = mean(dc(0, 2) .* log.(p02) + dc(2, 0) .* log.(1 .- p02))
    Q3 = mean(dc(1, 2) .* log.(p12) + dc(2, 1) .* log.(1 .- p12))

    # Return the negative log-likelihood
    return -(Q1 + Q2 + Q3)
end


## Simulation starts here
for b = 1:B

    XA1 = rand(MvNormal(zeros(dx), dx * I), N)'  # covariate for good A at T=1
    XA2 = rand(MvNormal(zeros(dx), dx * I), N)'  # covariate for good A at T=2
    XA = [XA1 XA2]
    XB1 = rand(MvNormal(zeros(dx), dx * I), N)'  # covariate for good B at T=1
    XB2 = rand(MvNormal(zeros(dx), dx * I), N)'  # covariate for good B at T=2
    XB = [XB1 XB2]
    Z = [rand(Normal(2, 2), N) rand(Normal(0, 1), N)]

    # error terms and fixed effects
    rho = -0.7
    ep1 = rand(MvNormal([2; -2], [1 rho; rho 1]), N)'
    ep2 = rand(MvNormal([2; -2], [1 rho; rho 1]), N)'
    epA = [ep1[:, 1] ep2[:, 1]]
    epB = [ep1[:, 2] ep2[:, 2]]
    ep0 = randn(N, T)
    alphaA = ((XA1 + XA2 - 2 * (XB1 + XB2)) * beta0 / 4) .* (1 .+ randn(N, 1))
    alphaB = ((XB1 + XB2 - 2 * (XA1 + XA2)) * beta0 / 4) .* (1 .+ randn(N, 1))

    # latent utilities
    X = [XA XB Z]
    uA = [XA[:, 1:dx] * beta0 XA[:, dx + 1:end] * beta0] .+ alphaA + epA - ep0
    uB = [XB[:, 1:dx] * beta0 XB[:, dx + 1:end] * beta0] .+ alphaB + epB - ep0
    u0 = 0
    uAB = uA + uB .+ Z * gamma0

    # dependent variables
    um = max.(uA, uB, uAB, u0)
    Y = (um .== uA) + (um .== uB) * 2 + (um .== uAB) * 3

    # two step estimator in this paper
    phat = neu(X, [Y .== 1 Y .== 2 Y .== 0 Y .== 3])  # first step estimator
    obj(theta) = mom(theta, XA, XB, Z, phat)          # objective function
    initial = [ones(dx - 1); 0.5 .* ones(dz - 1)]
    opt = optimize(obj, initial)
    beta_twostep[:, b] = opt.minimizer[1:dx - 1]
    gamma_twostep[:, b] = opt.minimizer[dx:end]
    err[b] = mean(abs.((Z * gamma0 .>= 0) - (Z * [1; gamma_twostep[:, b]] .>= 0)))

    # mixed effect parametric estimator
    objpar(theta) = par_sim(theta, XA, XB, Z, Y .== 1, Y .== 2, Y .== 3)
    initial_par = [ones(dx + dz); 0.5 .* ones(dx - 1)]
    optpar = optimize(objpar, initial_par)
    beta_par[:, b] = optpar.minimizer[1:dx - 1]
    gamma_par[:, b] = optpar.minimizer[dx:dz + dx - 2]
    err_par[b] = mean(abs.((Z * gamma0 .>= 0) - (Z * [1; gamma_par[:, b]] .>= 0)))

    # logit estimator without bundles
    objlogit(beta) = momlogit(beta, XA, XB, Y)
    optfl = optimize(objlogit, -5, 5)
    beta_fl[b] = optfl.minimizer

    # nonparametric estimator without bundles
    phat1 = phat[:, 1:3 * T]
    obj1(beta) = mom1(beta, XA, XB, phat1)
    opt1 = optimize(obj1, -5, 5)
    beta_non[b] = opt1.minimizer
end

## Evaluation of estimators
function M(theta)
    return [mean(theta) - 1, std(theta), sqrt(var(theta) + (mean(theta) - 1)^2), median(abs.(theta .- 1))] # Bias, SD, RMSE, MAD
end

## Performance of estimators
rMSEnon = M(beta_twostep)
rMSEpar = M(beta_par)
rMSEnon1 = M(beta_non)
rMSEfl = M(beta_fl)
rMSEgnon = M(gamma_twostep)
rMSEgpar = M(gamma_par)
e = [mean(err), mean(err_par)]

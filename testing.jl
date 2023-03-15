
# %%
include("utilities/Samplers.jl")
using .Samplers

# %%
t = collect(range(0.0, 10.0, length=400)) 
X = brownian_sampler(t)

# %%
using Plots
plot(t,X)

# %%
X2 = brownian_sampler(0.1,1000) 

# %%
plot(X2)

# %%
X3 = brownian_bridge_sampler(t)

plot(t,X3)

# %%
X3[length(t)],X3[1]

# %%
X4 = fractional_brownian_sampler(t,0.8)

# %%
plot(t,X4)


# %%
Covariance(s,tv) = cos(s-tv)
mean(tv) = 0.0

X5 = general_gaussian_process_sampler(t,mean,Covariance)

# %%
plot(t,X5)

# %%
l = 2.0
exponential_spectral_f(ν) = l / (π *(1+l^2 * ν^2 ))
X6 = spectral_quadrature_sampler(12.0,1000,500,exponential_spectral_f)
X6


# %%
s = collect(range(1.0,12.0,length=30))
Zs = interpolated_spectral_quadrature_sampler(s,1000,500,exponential_spectral_f)
Zs

# %% For gaussian random field sampling we use the package: https://github.com/PieterjanRobbe/GaussianRandomFields.jl
using GaussianRandomFields
cov = CovarianceFunction(2, Exponential(.5))
pts = range(0, stop=1, length=1001)
grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts, minpadding=2001)
heatmap(grf)

# %%
include("sode/sode_solvers.jl")
using .SODE_Solvers

# %% Algorithm 8.2
λ = 1.0
α = 1.0
σ = 1.0
F(u) = [u[2],-u[2]*(λ+u[1]^2) + α*u[1]-u[1]^3]
G(u) = [0, σ * u[1]]
T = 10.0
N = 1000
u0 = [0.5,0]
U,t = euler_murayama(u0,T,N,1,F,G)

plot(t,U[1,:])
plot!(t,U[2,:])
xlabel!("t")
ylabel!("u")
title!("Duffing-van der Pol λ=$λ ,α=$α , σ=$σ")

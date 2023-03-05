
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

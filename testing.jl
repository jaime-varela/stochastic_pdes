
# %%
x = [1,1,2,3,5,8,13,21]

# %%
y = diff(x)

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
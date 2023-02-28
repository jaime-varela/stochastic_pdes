
module Samplers
export brownian_sampler, brownian_bridge_sampler
using Random, Distributions

normal_unit_dist = Normal()

# Algorithm 5.1
function brownian_sampler(time::Vector)
    N = length(time)
    X = Vector{Float32}(undef,N)
    dt = diff(time)
    X[1] =0.0
    randomNums = rand(normal_unit_dist,N-1)
    dX = sqrt.(dt) .* randomNums
    X[2:N] = cumsum(dX)
    return X
end

function brownian_sampler(time::LinRange)
    return brownian_sampler(collect(time))
end

function brownian_sampler(dt::AbstractFloat,num::Integer)
    time = LinRange(0.0,num*dt,num)
    return brownian_sampler(time)    
end

# Algorithm 5.2
function brownian_bridge_sampler(time::Vector)
    W = brownian_sampler(time)
    N = length(time)
    subtraction = W[N] .* (time .- time[1])  
    return W .- (subtraction./(time[N] - time[1]))
end

function brownian_bridge_sampler(time::LinRange)
    return brownian_bridge_sampler(collect(time))
end

function brownian_bridge_sampler(dt::AbstractFloat,num::Integer)
    time = LinRange(0.0,num*dt,num)
    return brownian_bridge_sampler(time)    
end


end;
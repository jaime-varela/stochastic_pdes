
module Samplers
export brownian_sampler, brownian_bridge_sampler
export fractional_brownian_sampler
using Random, Distributions, LinearAlgebra

normal_unit_dist = Normal()

# Algorithm 5.1
function brownian_sampler(time::Vector)
    N = length(time)
    X = Vector{Float64}(undef,N)
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


# Algorithm 5.3
function fractional_brownian_sampler(time::Vector,H::Number)
    @assert H >= 0.0 && H <= 1.0
    N = length(time)
    C_N = Matrix{Float64}(undef,N,N)
    for i in 1:N
        for j in 1:N 
            ti,tj = time[i], time[j]
            C_N[i,j] = 0.5*(ti^(2*H)+tj^(2*H) - abs(ti-tj)^(2*H))
        end         
    end
    randomNums = rand(normal_unit_dist,N)
    Decomposition = eigen(C_N)
    return Decomposition.vectors * (sqrt.(Decomposition.values) .* randomNums)
end

function fractional_brownian_sampler(time::LinRange,H::Number)
    return fractional_brownian_sampler(collect(time),H)
end

function fractional_brownian_sampler(dt::AbstractFloat,num::Integer,H::Number)
    time = LinRange(0.0,num*dt,num)
    return fractional_brownian_sampler(time,H)    
end


end;
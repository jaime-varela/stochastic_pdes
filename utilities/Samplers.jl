
module Samplers
export brownian_sampler, brownian_bridge_sampler
export fractional_brownian_sampler
export general_gaussian_process_sampler
export spectral_quadrature_sampler
export interpolated_spectral_quadrature_sampler
using Random, Distributions, LinearAlgebra, FFTW
using Interpolations

normal_unit_dist = Normal()

function distributed_complex_number(dist::Distribution)
    rands = rand(dist,2)
    return rands[1] + im * rands[2]    
end

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
    Decomposition = eigen(Symmetric(C_N))
    return Decomposition.vectors * (sqrt.(Decomposition.values) .* randomNums)
end

function fractional_brownian_sampler(time::LinRange,H::Number)
    return fractional_brownian_sampler(collect(time),H)
end

function fractional_brownian_sampler(dt::AbstractFloat,num::Integer,H::Number)
    time = LinRange(0.0,num*dt,num)
    return fractional_brownian_sampler(time,H)    
end

# Equation 5.29
# X(t) = μ(t) +∑_{j-1}^{N} √v_j u_j ξ_j , ξ_j ~ N(-,1) iid
# Where v_j are the eignevalues of C(t_i,t_j) for t_i,t_j ∈ [t_1, ... t_N]
function general_gaussian_process_sampler(time::Vector,μ::Function,C::Function)
    N = length(time)
    C_N = Matrix{Float64}(undef,N,N)
    for i in 1:N
        for j in 1:N 
            ti,tj = time[i], time[j]
            C_N[i,j] = C(ti,tj)
        end         
    end
    randomNums = rand(normal_unit_dist,N)
    Decomposition = eigen(Symmetric(C_N))
    eigen_vals = Decomposition.values
    has_negative = any(x -> x < 0.0, eigen_vals)
    if has_negative
        # TODO: allow user to pass error bounds and ignore these small negative eignevalues
        @show "Negative eigenvalue encountered, returning mean only"
        return μ.(time)
    end

    return μ.(time) .+ Decomposition.vectors * (sqrt.(Decomposition.values) .* randomNums)

end


# Algorithm 6.4 with N sample times, T is the end time, M is the discretization
# parameter from 6.33, and f is the handle for f(ν) (the spectral density)
# see (6.26) on why there are various √2 factors
function spectral_quadrature_sampler(T::Number,N::Integer,M::Integer,f::Function)
    Δt = T/(N-1)
    t = Δt .* [0:1:N-1;]
    R = π / Δt
    Δν = 2*π / (N*Δt*M)
    Z = zeros(ComplexF64,N)
    coeff= zeros(ComplexF64,N)
    for m in 1:M
        for k in 1:N
            ν = -R + ((k-1)*M + (m-1))*Δν
            ξ = distributed_complex_number(normal_unit_dist)
            coeff[k] = sqrt(f(ν) * Δν)* ξ
            if (m==1 && k==1) || (m==M && k==N)
                coeff[k] = coeff[k]/√2
            end
        end
        Zi = N .* ifft(coeff)
        Z = Z + exp.((im * (-R + (m-1)*Δν)).*t) .* Zi
    end
    return Z
end 

# Algorithm 6.5
function interpolated_spectral_quadrature_sampler(S::Vector,N::Integer,M::Integer,f::Function)
    tmax = maximum(S)
    tmin = minimum(S)
    T = tmax - tmin
    t_range = collect(range(0.0, T, length=N))
    Z = spectral_quadrature_sampler(T,N,M,f)
    interpolator = LinearInterpolation(t_range .+ tmin, Z)
    return map(x -> interpolator(x),S)
end

end;
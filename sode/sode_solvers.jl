
module SODE_Solvers

export euler_murayama,euler_murayama_paths, MilsteinDiag
using Random, Distributions, LinearAlgebra, FFTW

normal_unit_dist = Normal()


"""
    Simple Euler Murayama solver in Algorithm 8.1
"""
function euler_murayama(u0::Vector,T::Number,N::Integer,m::Integer,
    F::Function,G::Function)
    Δt = T/N
    t = [0:Δt:T;]
    d = length(u0)
    u = zeros(Float64,d,N+1)
    u[:,1] = u0
    u_n = u0
    for time_ind in 1:N
        dW = sqrt(Δt) .* adjoint(rand(normal_unit_dist,m))
        u_new = u_n + Δt .* F(u_n) + (G(u_n) * dW)
        u[:,time_ind + 1] = u_new
        u_n = u_new
    end
   return (u,t) 
end

# Algorithm 8.5 
function euler_murayama_paths(u0::Vector,T::Number,N::Integer,m::Integer,
    F::Function,G::Function,κ0::Integer,M::Integer)
    d = length(u0)
    Δt_ref = T/N # small step
    Δt = κ0 * Δt_ref # large step
    NN = Int(N / κ0) 
    u = zeros(Float64,d,M,NN+1)
    t = zeros(Float64,NN+1,1)
    gdW = zeros(d,M)
    u_n = u0
    sqrt_ref = sqrt(Δt_ref)
    for n in 1:NN+1
        t[n] = (n-1)*Δt 
        u[:,:,n] .= u_n
        dW = sqrt_ref * squeeze(sum(randn(m,M,κ0),3))
        for mm in 1:M
            gdW[:,mm] = G(u_n[:,mm]) * dW[:,mm]
        end
        u_new = u_n + Δt .* F(u_n) + gdW
        u_n = u_new
    end
    return u,t
end


function MilsteinDiag(u0::Vector, T::Number, 
    N::Integer, d::Integer, m::Integer, 
    F::Function, G::Function, DG::Function)
    """
    Alg 8.3 Page 334
    """
    Dt = T / N
    u = zeros(d, N + 1)
    t = range(0, stop=T, length=N+1)
    sqrtDt = sqrt(Dt)
    u[:,1] = u0
    u_n = copy(u0)
    
    for n in 1:N
        dW = sqrtDt * randn(m)
        gu_n = G(u_n)
        F_n = F(u_n)
        dg_n = DG(u_n) 
        u_new = u_n + Dt * F_n + gu_n .* dW + 
                0.5 * (dg_n .* gu_n) .* (dW .^ 2 .- Dt)
        
        u[:,n + 1] = u_new
        u_n = u_new
    end
    
    return t, u
end


end;
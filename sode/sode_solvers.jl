
module SODE_Solvers

export euler_murayama
using Random, Distributions, LinearAlgebra, FFTW

normal_unit_dist = Normal()


# Algorithm 8.1
function euler_murayama(u0::Vector,T::Number,N::Integer,m::Integer,
    F::Callable,G::Callable)
    Δt = T/N
    t = [0;Δt;T;]
    d = length(u0)
    u = zeros(Float64,d,N+1)
    u[:,1] = u0
    u_n = u_0
    for time_ind in 1:N
        dW = sqrt(Δt) .* rand(normal_unit_dist,m)
        u_new = u_n + Δt .* F(u_n) + dW .* G(u_n)
        u[:,time_ind + 1] = u_new
        u_n = u_new
   return (u,t) 
end



end;
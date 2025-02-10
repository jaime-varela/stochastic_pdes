module SPDE_Solvers


export finite_difference_with_exponent_covariance, fNagumo
export l2_sq_mct, finite_difference_with_white_noise
using FFTW, Random, LinearAlgebra, SparseArrays


function fNagumo(u::Float64)
    return u * (1 - u) * (u + 0.5)
end

"""
A6.7 Page 246
"""
function circ_cov_sample(c)
    N = length(c)
    d = ifft(c) * N
    xi = randn(N) .+ im * randn(N)
    Z = fft(sqrt.(d) .* xi) / sqrt(N)
    return real(Z), imag(Z)
end

"""
A6.8 Page 247
"""
function circulant_embed_sample(c)
    tilde_c = vcat(c, c[end-1:-1:2])
    X, Y = circ_cov_sample(tilde_c)
    N = length(c)
    return X[1:N], Y[1:N]
end


"""
A6.9 Page 248
"""
function circulant_exp(N, dt, ell)
    t = (0:N-1) * dt
    c = exp.(-abs.(t) / ell)
    X, Y = circulant_embed_sample(c)
    return t, X, Y
end

"""
Alg 10.7 Page 456
"""
function finite_difference_with_exponent_covariance(u0, T, a, N, J, epsilon, sigma, ell, fhandle)
    Dt = T / N
    t = LinRange(0, T, N+1)
    h = a / J
    e = ones(J+1)
    
    A = spdiagm(-1 => -ones(J), 0 => 2*ones(J+1), 1 => -ones(J))
    A[1,2] = 2
    A[end,end-1] = 2
    
    EE = I + (Dt * epsilon / h^2) * A
    ut = zeros(J+1, length(t))
    ut[:,1] = u0
    u_n = u0
    flag = false
    EEinv = lu(EE) # LU factorization for solving linear systems

    dW2 = zeros(length(u_n)) # initialization
    for n in 1:N
        if length(u_n) == 1
            fu = fhandle(u_n)
        else
            fu = fhandle.(u_n)
        end

        if !flag
            x, dW, dW2 = circulant_exp(length(u_n), h, ell)
            flag = true
        else
            dW = dW2
            flag = false
        end
        u_new = EEinv \ (u_n + Dt * fu + sigma * sqrt(Dt) * dW)
        ut[:,n+1] = u_new
        u_n = u_new
    end
    return t, ut
end


"""
Alg 10.8 Page 458
"""
function finite_difference_with_white_noise(u0, T, a, N, J, epsilon, sigma, fhandle)
    Dt = T / N
    t = range(0, stop=T, length=N+1)
    h = a / J
    e = ones(J + 1)
    A = spdiagm(-1 => -ones(J), 0 => 2 * ones(J + 1), 1 => -ones(J))
    ind = 2:J  # Julia is 1-based
    A = A[ind, ind]
    EE = I + (Dt * epsilon / h^2) * A
    ut = zeros(J + 1, length(t))
    ut[:, 1] .= u0
    u_n = u0[ind]
    EEinv = factorize(EE)
    
    for n in 1:N
        if length(u_n) == 1
            fu = fhandle(u_n)
        else
            fu = fhandle.(u_n)
        end
        Wn = sqrt(Dt / h) * randn(J - 1)
        u_new = EEinv \ (u_n + Dt * fu + sigma * Wn)
        ut[ind, n + 1] .= u_new
        u_n = u_new
    end
    
    return t, ut
end

"""
Alg 10.9 Page 459
"""
function l2_sq_mct(T, a, N, J, M, epsilon, sigma)
    v = 0.0
    u0 = zeros(J + 1)
    
    for _ in 1:M
        t, ut = finite_difference_with_white_noise(u0, T, a, N, J, epsilon, sigma, x -> 0)
        v += norm(ut[1:end-1, end])^2
    end
    
    return v * a / (J * M)
end



end

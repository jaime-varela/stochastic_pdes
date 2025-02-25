module SPDE_Solvers


export finite_difference_with_exponent_covariance, fNagumo
export l2_sq_mct, finite_difference_with_white_noise
export get_twod_bj, get_twod_dW, spde_twod_Gal
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



# 2 D problems

fft2 = FFTW.fft
ifft2 = FFTW.ifft

function get_twod_bj(dtref, J, a, alpha)
    """
    Alg 4.5 Page 443
    """
    lambdax = 2 * pi * vcat(0:J[1]÷2, -J[1]÷2+1:-1) / a[1]
    lambday = 2 * pi * vcat(0:J[2]÷2, -J[2]÷2+1:-1) / a[2]
    lambdaxx, lambdayy = [repeat(lambdax, 1, length(lambday)), repeat(lambday', length(lambdax), 1)]
    root_qj = exp.(-alpha .* (lambdaxx .^ 2 .+ lambdayy .^ 2) ./ 2)
    bj = root_qj .* sqrt(dtref) .* J[1] .* J[2] / sqrt(a[1] * a[2])
    return bj
end


function get_twod_dW(bj, kappa, M)
    """
    Alg 10.6 Page 444
    """
    J = size(bj)
    if kappa == 1
        nn = randn(M, J[1], J[2], 2)
    else
        nn = sum(randn(kappa, M, J[1], J[2], 2), dims=1)[1, :, :, :, :]
    end
    # nn2 = nn ⋅ [1, 1im]  # Equivalent to dot product with complex number
    nn2 = sum(nn .* reshape([1, 1im], (1,1,1,2)), dims=4)
    a,b,c,d = size(nn2)
    nn2 = reshape(nn2, (a,b,c))
    a, b = size(bj)
    product_arr = nn2 .* reshape(bj,1,a,b)
    tmp = ifft2(product_arr)  # Element-wise multiplication and inverse FFT
    dW1 = real(tmp)
    dW2 = imag(tmp)
    
    return dW1, dW2
end


function spde_twod_Gal(u0, T, a, N, kappa, J, epsilon, fhandle, ghandle, alpha, M)
    """
    Alg 10.11 Page 471
    """
    dtref = T / N
    Dt = kappa * dtref
    t = range(0, T, length=N+1)
    
    lambdax = (2 * pi / a[1]) .* vcat(0:J[1]÷2, -J[1]÷2+1:-1)
    lambday = (2 * pi / a[2]) .* vcat(0:J[2]÷2, -J[2]÷2+1:-1)
    lambdaxx, lambdayy = [repeat(lambdax, 1, length(lambday)), repeat(lambday', length(lambdax), 1)]
    Dx, Dy = im .* lambdaxx, im .* lambdayy
    A = - (Dx .^ 2 .+ Dy .^ 2)
    MM = real.(epsilon .* A)
    EE = 1 ./ (1 .+ Dt .* MM)
    
    bj = get_twod_bj(dtref, J, a, alpha)

    u = permutedims(repeat(u0[1:end-1, 1:end-1], 1, 1, M),(3, 1, 2))
    uh = permutedims(repeat(fft2(u0[1:end-1, 1:end-1]), 1, 1, M),(3, 1, 2))
    ut = zeros(J[1]+1, J[2]+1, N÷kappa+1)
    ut[:, :, 1] .= u0
    for n in 1:N÷kappa
        fh = fft2(fhandle(u),(2,3))        
        dW, dW2 = get_twod_dW(bj, kappa, M)
        gudWh = fft2(ghandle(u) .* dW,(2,3))
        adim,bdim = size(EE)
        
        uh_new = reshape(EE,1,adim,bdim) .* (uh .+ Dt .* fh .+ gudWh)
        u = real(ifft2(uh_new,(2,3)))
        
        ut[1:end-1, 1:end-1, n+1] = u[end, :, :]
        uh = uh_new
    end
    u[:, end, :] .= u[:, 1, :]
    u[:, :,end] .= u[:, :,1]
    ut[:, end, :] .= ut[:, 1, :]
    ut[end, :,:] .= ut[1, :, :]
    
    return t, u, ut
end


end

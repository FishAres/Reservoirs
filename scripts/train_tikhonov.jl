using DrWatson
@quickactivate "Reservoirs"

using LinearAlgebra, Statistics
using Plots
using DynamicalSystems
using PredefinedDynamicalSystems: lorenz, roessler
using Flux, Zygote
using SparseArrays
using Distributions: Uniform
## ====
function norm_(x; dims=1)
    mn = mean(x, dims=dims)
    sd = std(x, dims=dims)
    return (x .- mn) ./ sd
end

function gen_ds_trajectory(; total_time=100.0, dt=0.02)
    ds = lorenz([0.0, 10.0, 0.0]; σ=14.0, ρ=34.0, β=8 / 3)
    Tr, t = trajectory(ds, total_time; Ttr=2.2, Δt=dt)
    # data as Float32
    return norm_(Matrix{Float32}(Tr), dims=1)
end

function res_forward(x, hprev; f=tanh)
    u = x * f_in
    # h = (1 - α) * hprev + f.(u + α * hprev * R)
    h = f.((1 - α) * hprev + u + α * hprev * R)
    return h, u
end

function run_model(X)
    h = zeros(Float32, 1, r_dim)
    hs, us = [], []
    for i in 1:size(X, 1)-1
        x = X[i:i, :]
        h, u = res_forward(x, h)
        push!(hs, h)
        push!(us, u)
    end
    return vcat(hs...), vcat(us...)
end

function norm_eig(x)
    λ = maximum(abs.(real.(eigvals(x))))
    return x ./ λ
end

function gen_sparse_R(r_dim; ρ=0.1)
    r_init = sprandn(Float32, r_dim, r_dim, ρ)
    r_init = Matrix{Float32}(r_init) |> norm_eig
    return sparse(r_init)
end

## ====

r_dim = 400
out_dim = 3

X = gen_ds_trajectory(total_time=150.0)

using Random

begin
    Random.seed!(42)
    f_in = randn(Float32, 3, r_dim)
    f_in = f_in ./ norm(f_in)
    R = gen_sparse_R(r_dim)
    α = 0.5f0
end

function solve_tikhonov(Y, hs; β=0.1f0)
    K = hs' * inv(hs * hs' + β * I)
    return K * Y
end

Y = X[2:end, :]
V = hcat(run_model(X)...)
Wout = solve_tikhonov(Y, V, β=0.01f0)

sg(x; d=1000.0f0) = Float32(σ(x - d))

begin
    hs, yhats, xs = [], [], []
    h = zeros(Float32, 1, r_dim)
    push!(hs, h |> cpu)
    ŷ = zeros(Float32, 1, 3)
    tlen = size(X, 1) - 1
    for i in 1:tlen
        inp = X[i:i, :]
        γ = sg(i; d=2000.0f0)
        x = (1 - γ) * inp + γ * reshape(ŷ, 1, :)
        h, u = res_forward(x, h)
        # ŷ = ([u h]*Wout)[:]
        ŷ = ([u h]*Wout)[:]
        push!(hs, h |> cpu)
        push!(yhats, ŷ |> cpu)
    end

    H = vcat(hs...)
    ys = Matrix(hcat(yhats...)')
end

begin
    dim = 2
    tspan = 1500:3000
    plot(ys[tspan, dim], label="pred")
    plot!(X[tspan.+1, dim], label="data")
end

tp = 2000
p = begin
    plot3d(ys[1:tp, 1], ys[1:tp, 2], ys[1:tp, 3])
    plot3d!(ys[tp.+1:end, 1], ys[tp.+1:end, 2], ys[tp.+1:end, 3])
end
mse(x, y; dims=2) = mean((x - y) .^ 2, dims=dims)

e = mse(ys[tp.+1:end, :], X[tp.+2:end, :])[:]
plot(e)

ys1 = ys

heatmap(H[tspan, :]')

histogram(H[1:2000, :][:])

plot3d(X[:, 1], X[:, 2], X[:, 3])


begin
    hs, yhats = [], []
    h = rand(Uniform(-1.0f0, 1.0f0), 1, r_dim) .|> Float32
    push!(hs, h |> cpu)
    for i in 1:size(X, 1)-1
        x = X[i:i, :]
        h = res_forward(x, h)
        ŷ = f_out(h[:])

        push!(hs, h |> cpu)
        push!(yhats, ŷ |> cpu)
    end
    H = vcat(hs...)
    ys = Matrix(hcat(yhats...)')
end

begin
    dim = 2
    tspan = 1:5000
    plot(ys[tspan, dim], label="pred")
    plot!(cpu(Y)[tspan, dim], label="data")
end


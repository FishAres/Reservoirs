using DrWatson
@quickactivate "Reservoirs"

using LinearAlgebra, Statistics
using Plots
using DynamicalSystems
using PredefinedDynamicalSystems: lorenz, roessler
using Flux, Zygote
using Distributions: Uniform
## ====
π32 = Float32(π)
ds = lorenz([0.0, 10.0, 0.0]; σ=14.0, ρ=34.0, β=8 / 3)
total_time = 100.0
sampling_time = 0.02
Tr, t = trajectory(ds, total_time; Ttr=2.2, Δt=sampling_time)

function norm_(x; dims=2)
    mn = mean(x, dims=dims)
    sd = std(x, dims=dims)
    return (x .- mn) ./ sd
end

# data as Float32
X = norm_(Matrix{Float32}(Tr), dims=1)
Y = circshift(X, (-1, 0))

## ====

function res_forward(x, hprev; f=tanh)
    u = x * f_in * R_in
    # h = f.((1 - γ) * u + γ * hprev * R + b)
    h = f.(u + hprev * R + b)
    return h
end

function train_model(opt, ps, X; d=Uniform(-1.0f0, 1.0f0))
    losses = []
    # h = rand(d, 1, r_dim) .|> Float32
    h = zeros(Float32, 1, r_dim)
    for i in 1:size(X, 1)-1
        x = X[i:i, :]
        y = X[i+1, :]
        h = res_forward(x, h)
        loss, grad = withgradient(ps) do
            ŷ = f_out(h[:])
            Flux.mse(ŷ, y)# + 0.01f0 * norm(h, 2)
        end
        Flux.update!(opt, ps, grad)
        push!(losses, loss)
    end
    return losses
end

function norm_eig(x)
    λ = maximum(abs.(real.(eigvals(x))))
    return x ./ λ
end

## ====

in_dim = 16
r_dim = 256
out_dim = 3

begin
    f_in = randn(Float32, 3, in_dim)
    R_in = randn(Float32, in_dim, r_dim)
    R = randn(Float32, r_dim, r_dim) |> norm_eig
    b = randn(Float32, 1, r_dim)
    f_out = Dense(r_dim, out_dim, bias=false)
    # ps = Flux.params(f_out)
    ps = Flux.params(f_out)
end

opt = ADAM(1e-5)

@time ls = vcat([train_model(opt, ps, X) for _ in 1:20]...)
plot(ls)


sg(x; d=1000.0f0) = Float32(σ(x - d))

begin
    hs, yhats, xs = [], [], []
    # h = rand(Uniform(-1.0f0, 1.0f0), 1, r_dim) .|> Float32
    h = zeros(Float32, 1, r_dim)
    push!(hs, h |> cpu)
    ŷ = zeros(Float32, 1, 3)
    tlen = size(X, 1) - 1
    for i in 1:tlen
        inp = X[i:i, :]
        γ = sg(i; d=2000.0f0)
        x = (1 - γ) * inp + γ * reshape(ŷ, 1, :)
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
    tspan = 1900:3000
    plot(ys[tspan, dim], label="pred")
    plot!(cpu(Y)[tspan, dim], label="data")
end

X
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


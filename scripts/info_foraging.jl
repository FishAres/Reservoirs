using DrWatson
@quickactivate "Reservoirs"

using LinearAlgebra, Statistics
using CairoMakie
import CairoMakie: lines, plot
using Flux, Zygote
using Flux: flatten, unsqueeze
using Flux: DataLoader
using SparseArrays
using Random
using MLDatasets

include(srcdir("interp_utils.jl"))

## ====

args = Dict(
    :bsz => 1,
    :scale_offset => -0.8f0,
    :img_size => (28, 28),
    :img_channels => 1,
    :r_f => tanh,
)

## ====
train_digits, _ = MNIST(:train)[:]
test_digits, _ = MNIST(:test)[:]

train_loader = DataLoader(train_digits, batchsize=1, shuffle=true)
test_loader = DataLoader(test_digits, batchsize=1, shuffle=true)


## ====

const sampling_grid = get_sampling_grid(args[:img_size]...)[1:2, :, :]
const ones_vec = ones(1, 1, args[:bsz])
const zeros_vec = zeros(1, 1, args[:bsz])
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec...; dims=3)
const diag_off = cat(1.0f-6 .* diag_vec...; dims=3)

const thetas0 = zeros(Float32, 6, 1)
## =====

function plot(x::AbstractMatrix; kwargs...)
    f = Figure()
    Axis(f[1, 1])
    for j in eachrow(x)
        lines!(j; kwargs...)
    end
    f
end

function res_forward(x, hprev; f=args[:r_f])
    u = x * f_in
    h = (1 - α) * hprev + f.(u + α * hprev * R)
    return h
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

function xy_thetas(b1, b2)
    theta_vec = [0.0f0; 0.0f0; 0.0f0; 0.0f0; b1; b2]
    thetas0 .+ theta_vec
end

a_to_dx(a) = [sin.(a); cos.(a)]
a_out(x) = 2 * π32 * x

## ====
r_dim = 300
begin
    f_in = randn(Float32, 784 + 2, r_dim)
    f_in = f_in ./ norm(f_in)
    R = gen_sparse_R(r_dim, ρ=0.15)
    f_a = Dense(r_dim, 1, a_out)
    f_f = Dense(r_dim, 2, sin)
    f_rec = Dense(r_dim, 784, relu, bias=false)
    ps = Flux.params(f_a, f_f, f_rec)
    α = 0.5f0
end


function model_loss(x; dt=0.5f0)
    fb = [0.0f0; 0.0f0]
    hprev = zeros(Float32, 1, r_dim)
    h = res_forward(collect([flatten(x); fb]'), hprev)
    a = f_a(h')
    fb = f_f(h')'
    thetas = xy_thetas(dt * a_to_dx(a))
    xbar = sample_patch(x, thetas, sampling_grid)
    x̂ = f_rec(h')
    L = Flux.mse(x̂[:], x[:])
    for t in 2:10
        h = res_forward([collect(flatten(xbar)') fb], h)
        a = f_a(h')
        fb = collect(f_f(h')')
        thetas += xy_thetas(dt * a_to_dx(a))
        xbar = sample_patch(x, thetas, sampling_grid)
        x̂ = f_rec(h')
        L += Flux.mse(x̂[:], x[:])
    end
    L
end

function train_RC(train_loader, opt, ps)
    ls = []
    for (i, x) in enumerate(train_loader)
        if i <= 20
            loss, grad = withgradient(ps) do
                model_loss(x)
            end
            Flux.update!(opt, ps, grad)
            push!(ls, loss)
            if i % 100 == 0
                @info "i == $i, loss=$loss"
            end
        end
    end
    return ls
end

opt = ADAM(1e-4)
ls = vcat([train_RC(train_loader, opt, ps) for _ in 1:15]...)
lines(ls)

x = first(train_loader)

function model_forward(x; dt=0.5f0)
    xbars, xhats = [], []
    fb = [0.0f0; 0.0f0]
    hprev = zeros(Float32, 1, r_dim)
    h = res_forward(collect([flatten(x); fb]'), hprev)
    a = f_a(h')
    fb = f_f(h')'
    thetas = xy_thetas(dt * a_to_dx(a))
    xbar = sample_patch(x, thetas, sampling_grid)
    x̂ = f_rec(h')
    push!(xhats, x̂)
    push!(xbars, xbar)
    L = Flux.mse(x̂[:], x[:])
    for t in 2:10
        h = res_forward([collect(flatten(xbar)') fb], h)
        a = f_a(h')
        fb = collect(f_f(h')')
        thetas += xy_thetas(dt * a_to_dx(a))
        xbar = sample_patch(x, thetas, sampling_grid)
        x̂ = f_rec(h')
        L += Flux.mse(x̂[:], x[:])
        push!(xhats, x̂)
        push!(xbars, xbar)
    end
    L, xhats, xbars
end

l, xhats, xbars = model_forward(x)


for (i, x) in enumerate(train_loader)
    if i <= 100
        model_loss(x)
    end
end



## =====

i = 0
begin
    i += 1
    p1 = heatmap(xs[i])
end

hh = vcat(hs...)

plot(hh')


th = hcat(ths...)
begin
    plot(th[1, :], th[2, :])
    xlims!(-1, 1)
    ylims!(-1, 1)
end
begin
    args[:scale_offset] = -0.8f0
    thetas = xy_thetas(randn(Float32, 2)...)
    xbar = sample_patch(x, thetas, sampling_grid)
    heatmap(xbar[:, :, 1, 1])
end






## ====


function train_model(opt, ps, X)
    losses = []
    h = zeros(Float32, 1, r_dim)
    for i in 1:size(X, 1)-1
        x = X[i:i, :]
        y = X[i+1, :]
        h = res_forward(x, h)
        loss, grad = withgradient(ps) do
            ŷ = f_out(h[:])
            Flux.mse(ŷ, y)
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

function gen_sparse_R(r_dim; ρ=0.1)
    r_init = sprandn(Float32, r_dim, r_dim, ρ)
    r_init = Matrix{Float32}(r_init) |> norm_eig
    return sparse(r_init)
end

sg(x; d=1000.0f0) = Float32(σ(x - d))

## ====

r_dim = 300
out_dim = 3

X = gen_ds_trajectory(total_time=150.0)



begin
    Random.seed!(42)
    f_in = randn(Float32, 3, r_dim)
    f_in = f_in ./ norm(f_in)
    R = gen_sparse_R(r_dim, ρ=0.15)
    f_out = Dense(r_dim, out_dim)
    ps = Flux.params(f_out)
    α = 0.5f0
end

opt = ADAM(4e-5)

@time ls = vcat([train_model(opt, ps, X) for _ in 1:10]...)
plot(ls)

begin
    tp = 2000
    hs, yhats, xs = [], [], []
    h = zeros(Float32, 1, r_dim)
    push!(hs, h |> cpu)
    ŷ = zeros(Float32, 1, 3)
    tlen = size(X, 1) - 1
    for i in 1:tlen
        inp = X[i:i, :]
        γ = sg(i; d=Float32(tp))
        x = (1 - γ) * inp + γ * reshape(ŷ, 1, :)
        h = res_forward(x, h)
        ŷ = f_out(h[:])
        push!(hs, h |> cpu)
        push!(yhats, ŷ |> cpu)
    end

    H = vcat(hs...)
    ys = Matrix(hcat(yhats...)')

    e = mse(ys[tp.+1:end, :], X[tp.+2:end, :])[:]
    plot(e)
end

begin
    dim = 3
    tspan = 1500:5000
    plot(ys[tspan, dim], label="pred")
    plot!(X[tspan.+1, dim], label="data")
end

p = begin
    # plot3d(ys[1:tp, 1], ys[1:tp, 2], ys[1:tp, 3], linestyle=:dash)
    plot3d(X[tp.+2:end, 1], X[tp.+2:end, 2], X[tp.+2:end, 3])
    plot3d!(ys[tp.+1:end, 1], ys[tp.+1:end, 2], ys[tp.+1:end, 3], linestyle=:dash)

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


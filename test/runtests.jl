using hNODEModel
using Test
using Lux, Random, DiffEqFlux, DifferentialEquations, ComponentArrays, LuxCUDA

data_dims = 16
latent_dims = 2 * data_dims
data_n = 2 * latent_dims
batches = 2 * data_n

encoder = Chain(Conv((1,), (data_dims => 1)), FlattenLayer(), Dense(data_n, latent_dims))
decoder = Dense(latent_dims, data_dims)
ode = Dense(latent_dims, latent_dims)
control = Chain(Conv((1,), (data_dims => 1)), FlattenLayer(), Dense(data_n, Lux.parameterlength(ode)))
solver = Euler()
hnode = hNODEModel.hNODE(; encoder=encoder, decoder=decoder, control=control, ode=ode, solver=solver)

function run_model(model, dev)
    ps, st = Lux.setup(Random.default_rng(), hnode) .|> dev

    data = begin
        xs = rand32(data_n, data_dims, batches) |> dev
        hNODEInput(xs, xs)
    end

    u0s, st = hNODEModel.encode(model, data, ps, st)
    controls, st = hNODEModel.control(model, data, ps, st)
    (trajectories, un), st = hNODEModel.integrate(model, (u0s, controls), ps, (st..., T=st.Δt * data_n))
    decoded, st = hNODEModel.decode(model, trajectories, ps, st)
    (decoded2, _), st = model(data, ps, (st..., T=st.Δt * data_n))

    return data, u0s, controls, trajectories, decoded, decoded2, ps, st
end

devices = Vector{Any}()
push!(devices, cpu_device())

if MLDataDevices.functional(CUDADevice)
    push!(devices, CUDADevice())
end

if MLDataDevices.functional(AMDGPUDevice)
    push!(devices, AMDGPUDevice())
end

@testset verbose = true for dev ∈ devices
    data, u0s, controls, trajectories, decoded, decoded2, ps, st = run_model(hnode, dev)
    @testset "encode" begin
        @test size(u0s) == (latent_dims, batches)
        @test u0s == encoder(data.encoder_input, ps.encoder, st.encoder)[1]
    end

    @testset "integrate" begin
        @test size(trajectories) == (data_n, latent_dims, batches)

        for (control, u0, target) ∈ zip(
            eachslice(controls; dims=ndims(controls)),
            eachslice(u0s; dims=ndims(u0s)),
            eachslice(trajectories; dims=ndims(trajectories))
        )
            tspan = (0.0f0, st.T)
            node = DiffEqFlux.NeuralODE(ode, tspan, solver, dt=st.Δt, saveat=0f0:st.Δt:st.T, save_on=true, save_start=true, save_end=true, verbose=false)

            trajectory = begin
                ode_params = ComponentArray(control, hnode.ode_axes)
                solution = node(u0, ode_params, st.ode)[1].u
                reduce(hcat, solution)'
            end
            prediction = trajectory[1:end-1, :]
            @test prediction ≈ target
        end

    end

    @testset "decode" begin
        @test size(decoded) == (data_n, data_dims, batches)
        for (x, y, y2) ∈ zip(
            eachslice(trajectories; dims=1),
            eachslice(decoded; dims=1),
            eachslice(decoded2; dims=1)
        )
            if dev == CUDADevice()
                CUDA.allowscalar() do
                    x = CuMatrix(x)
                    expected_y, _ = decoder(x, ps.decoder, st.decoder)
                    @test y ≈ y2 ≈ expected_y
                end
            else
                expected_y, _ = decoder(x, ps.decoder, st.decoder)
                @test y ≈ y2 ≈ expected_y
            end
        end
    end
end

using Lux, DiffEqFlux, ComponentArrays, Random, DifferentialEquations, LuxCUDA

struct hNODEInput
    encoder_input
    control_input
end

struct hNODE <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder, :control)}
    encoder::Lux.AbstractLuxLayer
    decoder::Lux.AbstractLuxLayer
    control::Lux.AbstractLuxLayer
    ode::Lux.AbstractLuxLayer
    solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm
    ode_axes::Tuple

    function hNODE(;
        encoder::Lux.AbstractLuxLayer,
        decoder::Lux.AbstractLuxLayer,
        control::Lux.AbstractLuxLayer,
        ode::Lux.AbstractLuxLayer,
        solver::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm=Tsit5()
    )
        ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes
        return new(encoder, decoder, control, ode, solver, ode_axes)
    end
end

Lux.initialstates(rng::AbstractRNG, m::hNODE) = (
    rng=rng,
    control=Lux.initialstates(rng, m.control),
    encoder=Lux.initialstates(rng, m.encoder),
    decoder=Lux.initialstates(rng, m.decoder),
    Δt=0.1f0,
    T=1f0,
    ode=Lux.initialstates(rng, m.ode)
)

function (m::hNODE)(xs::hNODEInput, ps, st::NamedTuple)
    u0s, st = encode(m, xs, ps, st)
    return m((xs, u0s), ps, st)
end

function (m::hNODE)((xs, u0s)::Tuple{<:hNODEInput,<:AbstractArray}, ps, st::NamedTuple)
    controls, st = control(m, xs, ps, st)
    (trajectories, un), st = integrate(m, (u0s, controls), ps, st)
    decoded, st = decode(m, trajectories, ps, st)
    return (decoded, un), st
end

# first stage - embed into the latent space
function encode(m::hNODE, xs::hNODEInput, ps, st)
    u0s, st_encoder = m.encoder(xs.encoder_input, ps.encoder, st.encoder)
    return u0s, (st..., encoder=st_encoder)
end

# second stage - infer the control parameters
function control(
    m::hNODE,
    xs::hNODEInput,
    ps,
    st::NamedTuple
)
    controls, st_control = m.control(xs.control_input, ps.control, st.control)
    return controls, (st..., control=st_control)
end

cat3(x...) = cat(x...; dims=3)
# third stage - use the feature vector to get the control for ode and integrate it using the embedded sound as u0
function integrate(
    m::hNODE,
    (u0s, controls)::Tuple{<:AbstractArray,<:AbstractArray},
    ps,
    st::NamedTuple
)
    tspan = (0.0f0, st.T)
    node = NeuralODE(m.ode, tspan, m.solver, dt=st.Δt, saveat=0f0:st.Δt:st.T, save_on=true, save_start=true, save_end=true, verbose=false)

    trajectories = (
        begin
            ode_params = ComponentArray(control, m.ode_axes)
            solution = node(u0, ode_params, st.ode)[1].u
            reduce(hcat, solution)'
        end
        for (control, u0) ∈ zip(
            eachslice(controls; dims=ndims(controls)),
            eachslice(u0s; dims=ndims(u0s))
        )
    )

    trajectories_as_array = reduce(cat3, trajectories)

    prediction = trajectories_as_array[1:end-1, :, :]
    new_u0s = trajectories_as_array[end, :, :]

    return (prediction, new_u0s), st
end

# fourth stage - use the decoder to convert latent trajectories to time series space
function decode(m::hNODE, trajectories::AbstractArray, ps, st::NamedTuple)
    decoded_trajectories = (
        begin
            y = m.decoder(x, ps.decoder, st.decoder)[1]
            reshape(y, 1, size(y)...)
        end
        for x ∈ eachslice(trajectories; dims=1)
    )

    return reduce(vcat, decoded_trajectories), st
end

function decode(m::hNODE, trajectories::CuArray, ps, st::NamedTuple)
    decoded_trajectories = (
        begin
            y = m.decoder(CuMatrix(x), ps.decoder, st.decoder)[1]
            reshape(y, 1, size(y)...)
        end
        for x ∈ eachslice(trajectories; dims=1)
    )

    return reduce(vcat, decoded_trajectories), st
end

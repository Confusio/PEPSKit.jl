"""
$(TYPEDEF)

CTMRG algorithm where all sides are grown and renormalized at the same time. In particular,
the projectors are applied to the corners from two sides simultaneously.

## Fields

$(TYPEDFIELDS)

## Constructors

    SimultaneousCTMRG(; kwargs...)

Construct a simultaneous CTMRG algorithm struct based on keyword arguments.
For a full description, see [`leading_boundary`](@ref). The supported keywords are:

* `tol::Real=$(Defaults.ctmrg_tol)`
* `maxiter::Int=$(Defaults.ctmrg_maxiter)`
* `miniter::Int=$(Defaults.ctmrg_miniter)`
* `verbosity::Int=$(Defaults.ctmrg_verbosity)`
* `trscheme::Union{TruncationScheme,NamedTuple}=(; alg::Symbol=:$(Defaults.trscheme))`
* `svd_alg::Union{<:SVDAdjoint,NamedTuple}`
* `projector_alg::Symbol=:$(Defaults.projector_alg)`
"""
struct SimultaneousCTMRG{P<:ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function SimultaneousCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg=:simultaneous, kwargs...)
end

CTMRG_SYMBOLS[:simultaneous] = SimultaneousCTMRG

function ctmrg_iteration(network, env::CTMRGEnv, alg::SimultaneousCTMRG)
    coordinates = eachcoordinate(network, 1:4)
    T_corners = Base.promote_op(
        TensorMap ∘ EnlargedCorner, typeof(network), typeof(env), eltype(coordinates)
    )
    enlarged_corners′ = similar(coordinates, T_corners)
    enlarged_corners::typeof(enlarged_corners′) =
        dtmap!!(enlarged_corners′, eachcoordinate(network, 1:4)) do idx
            corner = TensorMap(EnlargedCorner(network, env, idx))
            return corner / norm(corner)
        end  # expand environment
    projectors, info = simultaneous_projectors(enlarged_corners, env, alg.projector_alg)  # compute projectors on all coordinates
    env′ = renormalize_simultaneously(enlarged_corners, projectors, network, env)  # renormalize enlarged corners
    return env′, info
end

# Work-around to stop Zygote from choking on first execution (sometimes)
# Split up map returning projectors and info into separate arrays
function _split_proj_and_info(proj_and_info)
    P_left = map(x -> x[1][1], proj_and_info)
    P_right = map(x -> x[1][2], proj_and_info)
    truncation_error = maximum(x -> x[2].truncation_error, proj_and_info)
    condition_number = maximum(x -> x[2].condition_number, proj_and_info)
    U = map(x -> x[2].U, proj_and_info)
    S = map(x -> x[2].S, proj_and_info)
    V = map(x -> x[2].V, proj_and_info)
    U_full = map(x -> x[2].U_full, proj_and_info)
    S_full = map(x -> x[2].S_full, proj_and_info)
    V_full = map(x -> x[2].V_full, proj_and_info)
    info = (; truncation_error, condition_number, U, S, V, U_full, S_full, V_full)
    return (P_left, P_right), info
end

"""
    simultaneous_projectors(enlarged_corners::Array{E,3}, env::CTMRGEnv, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:simultaneous` scheme either for all provided
enlarged corners or on a specific `coordinate`.
"""
function simultaneous_projectors(
    enlarged_corners::Array{E,3}, env::CTMRGEnv, alg::HalfInfiniteProjector
) where {E}
    rowsize, colsize = size(enlarged_corners)[2:3]
    coordinates = eachcoordinate(env, 1:4)
    T_dst = Base.promote_op(
        compute_projector,
        AbstractTensorMap,
        AbstractTensorMap,
        SVDAdjoint,
        TruncationScheme,
        Int,
    )
    proj_and_info′ = similar(coordinates, T_dst)
    proj_and_info::typeof(proj_and_info′) =
        dtmap!!(proj_and_info′, coordinates) do coordinate
            coordinate′ = _next_coordinate(coordinate, rowsize, colsize)
            trscheme = truncation_scheme(alg, env.edges[coordinate[1], coordinate′[2:3]...])

            svd_alg = svd_algorithm(alg, coordinate)
            return compute_projector(
                enlarged_corners[coordinate...],
                enlarged_corners[coordinate′...],
                svd_alg,
                trscheme,
                alg.verbosity,
            )
        end
    return _split_proj_and_info(proj_and_info)
end
function simultaneous_projectors(
    enlarged_corners::Array{E,3}, env::CTMRGEnv, alg::FullInfiniteProjector
) where {E}
    rowsize, colsize = size(enlarged_corners)[2:3]
    coordinates = eachcoordinate(env, 1:4)

    enlarged_corners′ = similar(coordinates, E)
    enlarged_corners_full::typeof(enlarged_corners′) =
        dtmap!!(enlarged_corners′, coordinates) do coordinate
            return enlarged_corners[_prev_coordinate(coordinate, rowsize, colsize)...] ⊙
                   enlarged_corners[coordinate...]
        end

    T_dst = Base.promote_op(
        compute_projector,
        AbstractTensorMap,
        AbstractTensorMap,
        SVDAdjoint,
        TruncationScheme,
        Int,
    )
    proj_and_info′ = similar(coordinates, T_dst)
    proj_and_info::typeof(proj_and_info′) =
        dtmap!!(proj_and_info′, coordinates) do coordinate
            coordinate′ = _next_coordinate(coordinate, rowsize, colsize)
            trscheme = truncation_scheme(alg, env.edges[coordinate[1], coordinate′[2:3]...])

            coordinate3 = _next_coordinate(coordinate′, rowsize, colsize)
            svd_alg = svd_algorithm(alg, coordinate)
            return compute_projector(
                enlarged_corners_full[coordinate...],
                enlarged_corners_full[coordinate3...],
                svd_alg,
                trscheme,
                alg.verbosity,
            )
        end
    return _split_proj_and_info(proj_and_info)
end

"""
$(SIGNATURES)

Renormalize all enlarged corners and edges simultaneously.
"""
function renormalize_simultaneously(enlarged_corners, projectors, network, env)
    P_left, P_right = projectors
    coordinates = eachcoordinate(env, 1:4)
    T_CE = Tuple{cornertype(env),edgetype(env)}
    corners_edges′ = similar(coordinates, T_CE)
    corners_edges::typeof(corners_edges′) =
        dtmap!!(corners_edges′, coordinates) do (dir, r, c)
            if dir == NORTH
                corner = renormalize_northwest_corner(
                    (r, c), enlarged_corners, P_left, P_right
                )
                edge = renormalize_north_edge((r, c), env, P_left, P_right, network)
            elseif dir == EAST
                corner = renormalize_northeast_corner(
                    (r, c), enlarged_corners, P_left, P_right
                )
                edge = renormalize_east_edge((r, c), env, P_left, P_right, network)
            elseif dir == SOUTH
                corner = renormalize_southeast_corner(
                    (r, c), enlarged_corners, P_left, P_right
                )
                edge = renormalize_south_edge((r, c), env, P_left, P_right, network)
            elseif dir == WEST
                corner = renormalize_southwest_corner(
                    (r, c), enlarged_corners, P_left, P_right
                )
                edge = renormalize_west_edge((r, c), env, P_left, P_right, network)
            end
            return corner / norm(corner), edge / norm(edge)
        end

    return CTMRGEnv(map(first, corners_edges), map(last, corners_edges))
end

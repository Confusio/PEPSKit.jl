"""
    struct InfinitePEPO{T<:PEPOTensor}

Represents an infinite projected entangled-pair operator (PEPO) on a 3D cubic lattice.
"""
struct InfinitePEPO{T<:PEPOTensor} <: InfiniteSquareNetwork{T,3}
    A::Array{T,3}

    function InfinitePEPO(A::Array{T,3}) where {T<:PEPOTensor}
        # space checks
        for (d, w, h) in Tuple.(CartesianIndices(A))
            space(A[d, w, h], 1) == space(A[d, w, _next(h, end)], 2)' ||
                throw(SpaceMismatch("Physical space at site $((d, w, h)) does not match."))
            space(A[d, w, h], 3) == space(A[_prev(d, end), w, h], 5)' || throw(
                SpaceMismatch("North virtual space at site $((d, w, h)) does not match."),
            )
            space(A[d, w, h], 4) == space(A[d, _next(w, end), h], 6)' || throw(
                SpaceMismatch("East virtual space at site $((d, w, h)) does not match.")
            )
        end
        return new{T}(A)
    end
end

## Constructors
"""
    InfinitePEPO(A::AbstractArray{T, 3})

Allow users to pass in an array of tensors.
"""
function InfinitePEPO(A::AbstractArray{T,3}) where {T<:PEPOTensor}
    return InfinitePEPO(Array(deepcopy(A)))
end

"""
    InfinitePEPO(f=randn, T=ComplexF64, Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces.
"""
function InfinitePEPO(
    Pspaces::A, Nspaces::A, Espaces::A=Nspaces
) where {A<:AbstractArray{<:ElementarySpace,3}}
    return InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPO(
    f, T, Pspaces::A, Nspaces::A, Espaces::A=Nspaces
) where {A<:AbstractArray{<:ElementarySpace,3}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (1, 0, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1, 0)))
    Ppspaces = adjoint.(circshift(Pspaces, (0, 0, -1)))

    P = map(Pspaces, Ppspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, Pp, N, E, S, W
        return TensorMap(f, T, P * Pp ← N * E * S * W)
    end

    return InfinitePEPO(P)
end

function InfinitePEPO(
    Pspaces::A, Nspaces::A, Espaces::A=Nspaces
) where {A<:AbstractArray{<:ElementarySpace,2}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Pspaces = reshape(Pspaces, (size(Pspaces)..., 1))
    Nspaces = reshape(Pspaces, (size(Nspaces)..., 1))
    Espaces = reshape(Pspaces, (size(Espaces)..., 1))

    return InfinitePEPO(Pspaces, Nspaces, Espaces)
end

"""
    InfinitePEPO(A; unitcell=(1, 1, 1))

Create an InfinitePEPO by specifying a tensor and unit cell.
"""
function InfinitePEPO(A::T; unitcell::Tuple{Int,Int,Int}=(1, 1, 1)) where {T<:PEPOTensor}
    return InfinitePEPO(fill(A, unitcell))
end

"""
    InfinitePEPO(f=randn, T=ComplexF64, Pspace, Nspace, [Espace]; unitcell=(1,1,1))

Create an InfinitePEPO by specifying its spaces and unit cell.
"""
function InfinitePEPO(
    Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int,Int}=(1, 1, 1)
) where {S<:ElementarySpace}
    return InfinitePEPO(
        randn,
        ComplexF64,
        fill(Pspace, unitcell),
        fill(Nspace, unitcell),
        fill(Espace, unitcell),
    )
end
function InfinitePEPO(
    f, T, Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int,Int}=(1, 1, 1)
) where {S<:ElementarySpace}
    return InfinitePEPO(
        f, T, fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end

unitcell(T::InfinitePEPO) = T.A
TensorKit.space(T::InfinitePEPO, i, j) = space(T[i, j, end], 1)

function initializePEPS(
    T::InfinitePEPO{<:PEPOTensor{S}}, vspace::S
) where {S<:ElementarySpace}
    Pspaces = map(Iterators.product(axes(T, 1), axes(T, 2))) do (r, c)
        return space(T, r, c)
    end
    Nspaces = repeat([vspace], size(T, 1), size(T, 2))
    Espaces = repeat([vspace], size(T, 1), size(T, 2))
    return InfinitePEPS(Pspaces, Nspaces, Espaces)
end

const SAME_POINT_THRESH = 5

"""
    Camera()
    Camera{T}()

Construct `Camera` object.
"""
mutable struct Camera{T}
    # internal parameters
    params::SMatrix{3, 3, T, 9}
    # external parameters
    t::SVector{3, T}       # translation
    Q::SMatrix{3, 3, T, 9} # rotation
end

function Camera{T}() where {T}
    params = zero(SMatrix{3, 3, T})
    t = zero(SVector{3, T})
    R = zero(SMatrix{3, 3, T})
    Camera{T}(params, t, R)
end

Camera() = Camera{Float64}()

"""
    rq(A)

Compute the RQ factorization of the matrix `A`.
"""
function rq(A)
    # https://math.stackexchange.com/questions/1640695/rq-decomposition
    Q, R = qr(reverse(A, dims = 1)')
    reverse(reverse(R', dims = 1), dims = 2), reverse(Q', dims = 1)
end

"""
    compute_homogeneous_matrix(xᵢ => Xᵢ)

Compute `H` in ``\\bm{x} \\simeq \\bm{H} \\bm{X}``.
"""
function compute_homogeneous_matrix((xᵢ, Xᵢ)::Pair{<: AbstractVector{SVector{2, T}}, <: AbstractVector{SVector{DIM, U}}}) where {DIM, T <: Real, U <: Real}
    n = length(eachindex(xᵢ, Xᵢ)) # number of samples
    ElType = promote_type(T, U)
    A = Array{ElType}(undef, 2n, 2(DIM+1)+DIM)
    b = Vector{ElType}(undef, 2n)
    @assert size(A, 1) ≥ size(A, 2)
    for i in 1:n
        x = xᵢ[i]
        X = Xᵢ[i]
        I = 2i - 1
        A[I:I+1, :] .= vcat([X; SVector(1); zero(X); SVector(0); -x[1]*X]',
                            [zero(X); SVector(0); X; SVector(1); -x[2]*X]')
        b[I:I+1] .= x
    end
    SMatrix{3, DIM+1}(reshape(push!(A \ b, 1), DIM+1, 3)')
end

"""
    calibrate!(camera::Camera, xᵢ => Xᵢ)

Calibrate `camera` from the pair of coordinates of image `xᵢ` and its corresponding actual coordinates `Xᵢ`.
The elements of `xᵢ` should be vector of length `2` and those of `Xᵢ` should be vector of length `3`.
"""
function calibrate!(camera::Camera, (xᵢ, Xᵢ)::Pair{<: AbstractVector{<: SVector{2}}, <: AbstractVector{<: SVector{3}}})
    P = compute_homogeneous_matrix(xᵢ => Xᵢ)
    R, Q = rq(P[SOneTo(3), SOneTo(3)])
    M = diagm(sign.(diag(R)))
    R = R * M
    Q = M * Q

    # internal parameters
    camera.params = R
    # external parameters
    camera.Q = Q
    camera.t = inv(R) * P[:, 4]

    camera
end

function projection_matrix(camera)
    A = camera.params
    Q = camera.Q
    t = camera.t
    [A*Q A*t]
end

"""
    camera(X)

Calculate coordinates in image from actual coordinates `X`.
`camera` should be [`calibrate!`](@ref)d before using this function.
"""
function (camera::Camera)(X::AbstractVector)
    @assert length(X) == 3
    P = projection_matrix(camera)
    b = [X; 1]
    scale = P[3, :]' * b
    x = (P * b) / scale
    SVector(x[1], x[2])
end

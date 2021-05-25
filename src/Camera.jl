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
    calibrate!(camera::Camera, xᵢ => Xᵢ)

Calibrate `camera` from the pair of coordinates of image `xᵢ` and its corresponding actual coordinates `Xᵢ`.
The elements of `xᵢ` should be vector of length `2` and those of `Xᵢ` should be vector of length `3`.
"""
function calibrate!(camera::Camera{T}, (xᵢ, Xᵢ)::Pair{<: AbstractVector{<: AbstractVector}, <: AbstractVector{<: AbstractVector}}) where {T}
    n = length(eachindex(xᵢ, Xᵢ))
    @assert n > 5

    A = zeros(T, 2n, 11)
    b = Vector{T}(undef, 2n)
    for i in 1:n
        x = xᵢ[i]
        X = Xᵢ[i]
        I = 2i - 1
        # matrix
        A[I,   1:3] .= X; A[I,   4] = 1; A[I,   9:11] .= -x[1] .* X
        A[I+1, 5:7] .= X; A[I+1, 8] = 1; A[I+1, 9:11] .= -x[2] .* X
        # vector
        b[I:I+1] .= x
    end

    P = SMatrix{3, 4}(reshape(push!(A \ b, 1), 4, 3)')

    Q, R = qr(P[SOneTo(3), SOneTo(3)])
    if R[3,3] < 0
        Q *= -1
        R *= -1
    end
    if R[1,1] < 0 || R[2,2] < 0
        M11 = R[1,1] < 0 ? -1 : 1
        M22 = R[2,2] < 0 ? -1 : 1
        M = @SMatrix [M11   0 0
                      0   M22 0
                      0     0 1]
        R = R * M
        Q = M * Q
    end

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

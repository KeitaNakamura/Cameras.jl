const SAME_POINT_THRESH = 5

"""
    Camera()
    Camera{T}()

Construct `Camera` object.
"""
mutable struct Camera{T}
    # intrinsic parameters
    A::Mat{3, 3, T, 9}
    # extrinsic parameters
    t::Vec{3, T}       # translation
    Q::Mat{3, 3, T, 9} # rotation
    # distortion
    distcoefs::Vector{T}
end

function Camera{T}() where {T}
    params = zero(Mat{3, 3, T})
    t = zero(Vec{3, T})
    R = zero(Mat{3, 3, T})
    Camera{T}(params, t, R, [])
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
function compute_homogeneous_matrix((xᵢ, Xᵢ)::Pair{Vector{Vec{2, T}}, Vector{Vec{DIM, U}}}) where {DIM, T <: Real, U <: Real}
    n = length(eachindex(xᵢ, Xᵢ)) # number of samples
    ElType = promote_type(T, U)
    A = Array{ElType}(undef, 2n, 2(DIM+1)+DIM)
    b = Vector{ElType}(undef, 2n)
    @assert size(A, 1) ≥ size(A, 2)
    for i in 1:n
        x = xᵢ[i]
        X = Xᵢ[i]
        I = 2i - 1
        A[I:I+1, :] .= vcat([X; 1; zero(X); 0; -x[1]*X]',
                            [zero(X); 0; X; 1; -x[2]*X]')
        b[I:I+1] .= x
    end
    Mat{3, DIM+1}(reshape(push!(A \ b, 1), DIM+1, 3)')
end

"""
    calibrate!(camera::Camera, xᵢ => Xᵢ)

Calibrate `camera` from the pair of coordinates of image `xᵢ` and its corresponding actual coordinates `Xᵢ`.
The elements of `xᵢ` should be vector of length `2` and those of `Xᵢ` should be vector of length `3`.
"""
function calibrate!(camera::Camera, (xᵢ, Xᵢ)::Pair{Vector{Vec{2, T}}, Vector{Vec{3, U}}}) where {T, U}
    P = compute_homogeneous_matrix(xᵢ => Xᵢ)
    R, Q = rq(@Tensor P[1:3, 1:3])
    M = diagm(sign.(diag(R)))
    R = R ⋅ M
    Q = M ⋅ Q

    # intrinsic parameters
    camera.A = R
    # extrinsic parameters
    camera.Q = Q
    camera.t = inv(R) ⋅ P[:, 4]

    camera
end

function calibrate_intrinsic!(camera::Camera, planes::Vector{Pair{Vector{Vec{2, T}}, Vector{Vec{2, U}}}}) where {T, U}
    n = length(planes)
    ElType = promote_type(T, U)
    V = Array{ElType}(undef, 2n, 6)
    @assert size(V, 1) ≥ size(V, 2)
    vij(H, i, j) = Vec(H[1,i]*H[1,j],
                       H[1,i]*H[2,j] + H[2,i]*H[1,j],
                       H[2,i]*H[2,j],
                       H[3,i]*H[1,j] + H[1,i]*H[3,j],
                       H[3,i]*H[2,j] + H[2,i]*H[3,j],
                       H[3,i]*H[3,j])
    for i in 1:n
        xᵢ, Xᵢ = planes[i]
        H = compute_homogeneous_matrix(xᵢ => Xᵢ)
        v12 = vij(H, 1, 2)
        v11 = vij(H, 1, 1)
        v22 = vij(H, 2, 2)
        I = 2i - 1
        V[I:I+1, :] .= vcat(v12', (v11 - v22)')
    end
    b = eigvecs(Symmetric(V' * V))[:,1] # extract eigen values corresponding minumum eigen value

    B11, B12, B22, B13, B23, B33 = b
    y0 = (B12*B13 - B11*B23) / (B11*B22 - B12^2)
    λ = B33 - (B13^2 + y0*(B12*B13 - B11*B23)) / B11
    α = sqrt(λ / B11)
    β = sqrt(λ*B11 / (B11*B22 - B12^2))
    γ = -B12*α^2*β/λ
    x0 = γ*y0/β - B13*α^2/λ

    # ignore skew parameter γ
    camera.A = @Mat [α 0 x0
                     0 β y0
                     0 0  1]

    camera
end

function calibrate_extrinsic!(camera::Camera, (xᵢ, Xᵢ)::Pair{Vector{Vec{2, T}}, Vector{Vec{2, U}}}) where {T, U}
    H = compute_homogeneous_matrix(xᵢ => Xᵢ)
    A⁻¹ = inv(camera.A)
    λ1 = 1 / norm(A⁻¹ ⋅ H[:, 1])
    λ2 = 1 / norm(A⁻¹ ⋅ H[:, 2])
    λ = (λ1 + λ2) / 2 # λ1 = λ2 in theory
    r1 = λ * A⁻¹ ⋅ H[:, 1]
    r2 = λ * A⁻¹ ⋅ H[:, 2]
    r3 = r1 × r2
    t = λ * A⁻¹ ⋅ H[:, 3]

    F = svd(hcat(r1, r2, r3))
    camera.Q = F.U ⋅ F.V'
    camera.t = t

    camera
end

function calibrate!(camera::Camera, planes::Vector{Pair{Vector{Vec{2, T}}, Vector{Vec{2, U}}}}) where {T, U}
    calibrate_intrinsic!(camera, planes)
    calibrate_extrinsic!(camera, planes[end])
    camera
end

"""
    calibrate!(camera::Camera, chessboards::Vector{<: Chessboard})

Calibrate intrinsic parameters of `camera` from `chessboards`.
"""
function calibrate_intrinsic!(camera::Camera, boards::Vector{<: Chessboard})
    planes = map(board -> vec(imagepoints(board)) => vec(objectpoints(board)), boards)
    calibrate_intrinsic!(camera, planes)
    camera
end

"""
    calibrate!(camera::Camera, chessboard::Chessboard; [gridspace = 1])

Calibrate extrinsic parameters of `camera` from `chessboard`.
"""
function calibrate_extrinsic!(camera::Camera, board::Chessboard; gridspace::Real = 1)
    calibrate_extrinsic!(camera, vec(imagepoints(board)) => vec(objectpoints(board) * gridspace))
    camera
end

"""
    calibrate!(camera::Camera, chessboards::Vector{<: Chessboard}; [gridspace = 1])

Calibrate `camera` from `chessboards`.

See also [`calibrate_intrinsic!`](@ref) and [`calibrate_extrinsic!`](@ref).
"""
function calibrate!(camera::Camera, boards::Vector{<: Chessboard}; gridspace::Real = 1)
    calibrate_intrinsic!(camera, boards)
    calibrate_extrinsic!(camera, boards[end]; gridspace)
    camera
end

"""
    projection_matrix(camera)

Return projection matrix to convert actual coordinate to corresponding 2D image coordinates.
"""
function projection_matrix(camera)
    A = camera.A
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
    Vec(x[1], x[2])
end

"""
    Camera()
    Camera{T}()

Construct `Camera` object.
"""
mutable struct Camera{T}
    # intrinsic parameters
    A::Mat{3, 3, T, 9}
    # distortion
    distcoefs::Vector{T} # k1, k2, p1, p2, k3
end

function Camera{T}(; ndistcoefs::Int = 4) where {T}
    @assert ndistcoefs == 0 || ndistcoefs == 2 || ndistcoefs == 4 || ndistcoefs == 5
    params = zero(Mat{3, 3, T})
    distcoefs = zeros(T, ndistcoefs)
    Camera{T}(params, distcoefs)
end
Camera(; ndistcoefs::Int = 4) = Camera{Float64}(; ndistcoefs)

ndistcoefs(camera::Camera) = length(camera.distcoefs)

function camera_matrix(params::Union{Tuple, AbstractVector})
    @assert length(params) == 4
    fx, fy, cx, cy = params[1], params[2], params[3], params[4]
    @Mat [fx 0  cx
          0  fy cy
          0  0  1]
end

struct CameraExtrinsic{T}
    R::Mat{3, 3, T, 9} # rotation
    t::Vec{3, T}       # translation
end
function CameraExtrinsic{T}() where {T}
    R = zero(Mat{3, 3, T})
    t = zero(Vec{3, T})
    CameraExtrinsic{T}(R, t)
end
CameraExtrinsic() = CameraExtrinsic{Float64}()
# iteration for destructuring into components
Base.iterate(x::CameraExtrinsic) = (x.R, Val(:t))
Base.iterate(x::CameraExtrinsic, ::Val{:t}) = (x.t, Val(:done))
Base.iterate(x::CameraExtrinsic, ::Val{:done}) = nothing

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
    projection_matrix(Xᵢ, xᵢ)

Compute the projection matrix from `Xᵢ` to `xᵢ`.
The matrix ``\\bm{P}`` is defined in 3D as

```math
\\begin{Bmatrix}
x \\\\ y \\\\ 1
\\end{Bmatrix}
= \\bm{P} \\begin{Bmatrix}
X \\\\ Y \\\\ Z \\\\ 1
\\end{Bmatrix}.
```

In 2D, this corresponds to homography matrix ``\\bm{H}`` as

```math
\\begin{Bmatrix}
x \\\\ y \\\\ 1
\\end{Bmatrix}
= \\bm{H} \\begin{Bmatrix}
X \\\\ Y \\\\ 1
\\end{Bmatrix}.
```
"""
function projection_matrix(Xᵢ::AbstractArray{Vec{dim, T}}, xᵢ::AbstractArray{Vec{2, U}}) where {dim, T, U}
    n = length(eachindex(xᵢ, Xᵢ)) # number of samples
    ElType = promote_type(T, U)
    A = Array{ElType}(undef, 2n, 2(dim+1)+dim)
    b = Vector{ElType}(undef, 2n)
    @assert size(A, 1) ≥ size(A, 2)
    for i in 1:n
        x = xᵢ[i]
        X = Xᵢ[i]
        I = 2i - 1
        A[I:I+1, :] .= [[X; 1; zero(X); 0; -x[1]*X]'
                        [zero(X); 0; X; 1; -x[2]*X]']
        b[I:I+1] .= x
    end
    Mat{3, dim+1}(reshape(push!(A \ b, 1), dim+1, 3)')
end

"""
    calibrate!(camera::Camera, Xᵢ, xᵢ) -> CameraExtrinsic

Calibrate `camera` from the pair of coordinates of image `xᵢ` and its corresponding actual coordinates `Xᵢ`.
The elements of `xᵢ` should be vector of length `2` and those of `Xᵢ` should be vector of length `3`.
"""
function calibrate!(camera::Camera, Xᵢ::AbstractArray{<: Vec{3}}, xᵢ::AbstractArray{<: Vec{2}})
    P = projection_matrix(Xᵢ, xᵢ)
    A, R = rq(@Tensor P[1:3, 1:3])
    M = diagm(sign.(diag(R))) # for fixing sign

    # intrinsic parameters
    camera.A = A ⋅ M
    # extrinsic parameters
    R = M ⋅ R
    t = inv(R) ⋅ P[:, 4]

    CameraExtrinsic(R, t)
end

function calibrate_intrinsic!(camera::Camera, Xᵢ::AbstractArray{Vec{2, T}}, xᵢ_set::AbstractVector{<: AbstractArray{Vec{2, U}}}) where {T, U}
    N = length(xᵢ_set)
    V = Array{promote_type(T, U)}(undef, 2N, 6)
    @assert size(V, 1) ≥ size(V, 2)
    vij(H, i, j) = Vec(H[1,i]*H[1,j],
                       H[1,i]*H[2,j] + H[2,i]*H[1,j],
                       H[2,i]*H[2,j],
                       H[3,i]*H[1,j] + H[1,i]*H[3,j],
                       H[3,i]*H[2,j] + H[2,i]*H[3,j],
                       H[3,i]*H[3,j])
    for i in 1:N
        H = projection_matrix(Xᵢ, xᵢ_set[i])
        v12 = vij(H, 1, 2)
        v11 = vij(H, 1, 1)
        v22 = vij(H, 2, 2)
        I = 2i - 1
        V[I:I+1, :] .= [v12'
                        (v11 - v22)']
    end
    b = eigvecs(Symmetric(V' * V))[:,1] # extract eigen values corresponding minumum eigen value

    B11, B12, B22, B13, B23, B33 = b
    cy = (B12*B13 - B11*B23) / (B11*B22 - B12^2)
    λ = B33 - (B13^2 + cy*(B12*B13 - B11*B23)) / B11
    fx = sqrt(λ / B11)
    fy = sqrt(λ*B11 / (B11*B22 - B12^2))
    γ = -B12*fx^2*fy/λ
    cx = γ*cy/fy - B13*fx^2/λ

    # ignore skew parameter γ
    camera.A = camera_matrix((fx, fy, cx, cy))

    camera
end

function calibrate_extrinsic(camera::Camera, Xᵢ::AbstractArray{<: Vec{2}}, xᵢ::AbstractArray{<: Vec{2}})
    H = projection_matrix(Xᵢ, xᵢ)
    A⁻¹ = inv(camera.A)
    λ1 = 1 / norm(A⁻¹ ⋅ H[:, 1])
    λ2 = 1 / norm(A⁻¹ ⋅ H[:, 2])
    λ = (λ1 + λ2) / 2 # λ1 = λ2 in theory
    r1 = λ * A⁻¹ ⋅ H[:, 1]
    r2 = λ * A⁻¹ ⋅ H[:, 2]
    r3 = r1 × r2
    t = λ * A⁻¹ ⋅ H[:, 3]

    F = svd([r1 r2 r3])
    R = F.U ⋅ F.V'

    CameraExtrinsic(R, t)
end

function calibrate_firstguess(camera::Camera, Xᵢ::AbstractArray{Vec{2, T}}, xᵢ_set::AbstractVector{<: AbstractArray{Vec{2, U}}}) where {T, U}
    N = length(xᵢ_set)
    n_d = ndistcoefs(camera)
    ElType = promote_type(T, U)
    params0 = Vector{ElType}(undef, 4+n_d+(3+3)*N) # 4: (camera matrix), 5: distortion, 3: rotation, 3: translation

    calibrate_intrinsic!(camera, Xᵢ, xᵢ_set)
    params0[1:4] .= (camera.A[1,1], camera.A[2,2], camera.A[1,3], camera.A[2,3]) # fx, fy, cx, cy
    params0[5:5+n_d-1] .= 0 # k1, k2, p1, p2, k3

    I = 4 + n_d + 1
    for i in 1:N
        R, t = calibrate_extrinsic(camera, Xᵢ, xᵢ_set[i])
        θ, n = angleaxis(R)
        params0[I:I+2] .= θ * n
        params0[I+3:I+5] .= t
        I += 6
    end
    @assert length(params0) + 1 == I

    params0
end

# nonlinear minimization model for calibration
# this is used in `curve_fit` in LsqFit
function calibration_nonlinear_minimization_model(Xᵢ_flat, params, N, n_d)
    A = camera_matrix(params[1:4])
    distcoefs = params[5:5+n_d-1]
    I = 4 + n_d + 1
    xs = Matrix{eltype(params)}(undef, length(Xᵢ_flat), N) # estimation of flat version of xᵢ_set
    for j in 1:N
        ω = @Vec [params[I], params[I+1], params[I+2]]
        R = rotmat(norm(ω), ω)
        t = @Vec [params[I+3], params[I+4], params[I+5]]
        for i in 1:2:length(Xᵢ_flat)
            X = @Vec [Xᵢ_flat[i], Xᵢ_flat[i+1], 0]
            x = R ⋅ X + t
            x′ = @Tensor(x[1:2]) / x[3]
            x′′ = distort(x′, distcoefs)
            u = A ⋅ [x′′; 1]
            xs[i,j]   = u[1]
            xs[i+1,j] = u[2]
        end
        I += 6
    end
    vec(xs)
end

function calibrate!(camera::Camera, Xᵢ::AbstractArray{Vec{2, T}}, xᵢ_set::AbstractVector{<: AbstractArray{Vec{2, U}}}) where {T, U}
    N = length(xᵢ_set)
    n_d = ndistcoefs(camera)

    fit = curve_fit((Xᵢ_flat, params) -> calibration_nonlinear_minimization_model(Xᵢ_flat, params, N, n_d),
                    vec(reinterpret(T, Xᵢ)),
                    vcat(map(xᵢ -> vec(reinterpret(U, xᵢ)), xᵢ_set)...),
                    calibrate_firstguess(camera, Xᵢ, xᵢ_set);
                    autodiff = :forwarddiff)
    @show fit.converged

    camera.A = camera_matrix(fit.param[1:4])
    camera.distcoefs = fit.param[5:5+n_d-1]

    exts = Vector{CameraExtrinsic{promote_type(T, U)}}(undef, N)
    I = 4 + n_d + 1
    for i in 1:N
        v = Vec{3}(fit.param[I:I+2])
        R = rotmat(norm(v), v)
        t = Vec{3}(fit.param[I+3:I+5])
        exts[i] = CameraExtrinsic(R, t)
        I += 6
    end
    @assert length(fit.param) + 1 == I

    exts
end

"""
    calibrate!(camera::Camera, chessboards::Vector{<: Chessboard})

Calibrate intrinsic parameters of `camera` from `chessboards`.
"""
function calibrate_intrinsic!(camera::Camera, boards::Vector{<: Chessboard})
    @assert all(board -> board == boards[1], map(objectpoints, boards))
    objpts = objectpoints(boards[1])
    calibrate_intrinsic!(camera, objpts, map(imagepoints, boards))
end

"""
    calibrate!(camera::Camera, chessboard::Chessboard; [gridspace = 1])

Calibrate extrinsic parameters of `camera` from `chessboard`.
"""
function calibrate_extrinsic!(camera::Camera, board::Chessboard; gridspace::Real = 1)
    calibrate_extrinsic!(camera, objectpoints(board)*gridspace, imagepoints(board))
end

"""
    calibrate!(camera::Camera, chessboards::Vector{<: Chessboard}; [gridspace = 1])

Calibrate `camera` from `chessboards`.

See also [`calibrate_intrinsic!`](@ref) and [`calibrate_extrinsic!`](@ref).
"""
function calibrate!(camera::Camera, boards::Vector{<: Chessboard}; gridspace::Real = 1)
    @assert all(pts -> pts == objectpoints(boards[1]), map(objectpoints, boards))
    objpts = objectpoints(boards[1])
    calibrate!(camera, objpts, map(imagepoints, boards))
end

# `(x′, y′)` is normalized coordinate
function distort(coords::Vec{2}, params::Vector{T}) where {T}
    x′, y′ = coords
    distcoefs = zeros(T, 5)
    distcoefs[1:length(params)] .= params
    k1, k2, p1, p2, k3 = distcoefs
    r² = x′^2 + y′^2
    x′′ = x′*(1 + k1*r² + k2*r²^2 + k3*r²^3) + 2p1*x′*y′ + p2*(r²+2x′^2)
    y′′ = y′*(1 + k1*r² + k2*r²^2 + k3*r²^3) + p1*(r²+2y′^2) + 2p2*x′*y′
    Vec(x′′, y′′)
end

function undistort(camera::Camera, image::AbstractArray)
    undistorted = zero(image)
    for I in CartesianIndices(undistorted)
        u = Vec(Tuple(I))
        x′ = inv(camera.A) ⋅ [u; 1] # normalized coordinate
        x′′ = distort(@Tensor(x′[1:2]), camera.distcoefs)
        u′ = camera.A ⋅ [x′′; 1]
        i, j = round.(Int, u′) # nearest-neighbor interpolation
        if checkbounds(Bool, image, i, j)
            undistorted[I] = image[i, j]
        end
    end
    undistorted
end

#=
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
=#

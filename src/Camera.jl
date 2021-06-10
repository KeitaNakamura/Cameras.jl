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
    distcoefs::Vec{5, T} # k1, k2, k3, p1, p2
end

function Camera{T}() where {T}
    params = zero(Mat{3, 3, T})
    t = zero(Vec{3, T})
    Q = zero(Mat{3, 3, T})
    distcoefs = zero(Vec{5, T})
    Camera{T}(params, t, Q, distcoefs)
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
    calibrate!(camera::Camera, Xᵢ, xᵢ)

Calibrate `camera` from the pair of coordinates of image `xᵢ` and its corresponding actual coordinates `Xᵢ`.
The elements of `xᵢ` should be vector of length `2` and those of `Xᵢ` should be vector of length `3`.
"""
function calibrate!(camera::Camera, Xᵢ::AbstractArray{<: Vec{3}}, xᵢ::AbstractArray{<: Vec{2}})
    P = projection_matrix(Xᵢ, xᵢ)
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

function calibrate_extrinsic!(camera::Camera, Xᵢ::AbstractArray{<: Vec{2}}, xᵢ::AbstractArray{<: Vec{2}})
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
    camera.Q = F.U ⋅ F.V'
    camera.t = t

    camera
end

function calibrate!(camera::Camera, Xᵢ::AbstractArray{Vec{2, T}}, xᵢ_set::AbstractVector{<: AbstractArray{Vec{2, U}}}) where {T, U}
    N = length(xᵢ_set)
    ElType = promote_type(T, U)
    params0 = Vector{ElType}(undef, 4+5+(3+3)*N) # 4: (camera matrix), 5: distortion, 3: rotation, 3: translation

    calibrate_intrinsic!(camera, Xᵢ, xᵢ_set)
    params0[1:4] .= (camera.A[1,1], camera.A[2,2], camera.A[1,3], camera.A[2,3]) # α, β, x0, y0
    params0[5:9] .= 0 # k1, k2, k3, p1, p2

    I = 10
    for i in 1:N
        calibrate_extrinsic!(camera, Xᵢ, xᵢ_set[i])
        θ, n = angleaxis(camera.Q)
        params0[I:I+2] .= θ * n
        params0[I+3:I+5] .= camera.t
        I += 6
    end

    model(Xᵢ_flat, params) = begin
        α, β, x0, y0 = params[1:4]
        k1, k2, k3, p1, p2 = params[5:9]
        I = 10
        xs = Matrix{ElType}(undef, length(Xᵢ_flat), N) # estimation of flat version of xᵢ_set
        for j in 1:N
            v = @Vec [params[I], params[I+1], params[I+2]]
            R = rotmat(norm(v), v)
            t = @Vec [params[I+3], params[I+4], params[I+5]]
            for i in 1:2:length(Xᵢ_flat)
                X = @Vec [Xᵢ_flat[i], Xᵢ_flat[i+1], 0]
                x = R ⋅ X + t
                x′ = x[1] / x[3]
                y′ = x[2] / x[3]
                x′′, y′′ = distort(Vec(x′,y′), Vec(k1, k2, k3, p1, p2))
                u = α * x′′ + x0
                v = β * y′′ + y0
                xs[i,j] = u
                xs[i+1,j] = v
            end
            I += 6
        end
        vec(xs)
    end

    fit = curve_fit(model,
                    vec(reinterpret(T, Xᵢ)),
                    vcat(map(xᵢ -> vec(reinterpret(U, xᵢ)), xᵢ_set)...),
                    params0)

    α, β, x0, y0 = fit.param[1:4]
    k1, k2, k3, p1, p2 = fit.param[5:9]
    camera.A = @Mat [α 0 x0
                     0 β y0
                     0 0  1]
    camera.distcoefs = @Vec [k1, k2, k3, p1, p2]
    I = 10 + 6*(N-1)
    v = Vec{3}(fit.param[I:I+2])
    camera.Q = rotmat(norm(v), v)
    camera.t = Vec{3}(fit.param[I+3:I+5])

    camera
end

"""
    calibrate!(camera::Camera, chessboards::Vector{<: Chessboard})

Calibrate intrinsic parameters of `camera` from `chessboards`.
"""
function calibrate_intrinsic!(camera::Camera, boards::Vector{<: Chessboard})
    @assert all(board -> board == boards[1], map(objectpoints, boards))
    objpts = objectpoints(boards[1])
    calibrate_intrinsic!(camera, objpts, map(imagepoints, boards))
    camera
end

"""
    calibrate!(camera::Camera, chessboard::Chessboard; [gridspace = 1])

Calibrate extrinsic parameters of `camera` from `chessboard`.
"""
function calibrate_extrinsic!(camera::Camera, board::Chessboard; gridspace::Real = 1)
    calibrate_extrinsic!(camera, objectpoints(board)*gridspace, imagepoints(board))
    camera
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
    camera
end

# `x` is the normalized coordinate
function distort((x′, y′)::Vec{2}, (k1, k2, k3, p1, p2)::Vec{5})
    r² = x′^2 + y′^2
    x′′ = x′*(1 + k1*r² + k2*r²^2 + k3*r²^3) + 2p1*x′*y′ + p2*(r²+2x′^2)
    y′′ = y′*(1 + k1*r² + k2*r²^2 + k3*r²^3) + p1*(r²+2y′^2) + 2p2*x′*y′
    Vec(x′′, y′′)
end

function undistort(camera::Camera, image::AbstractArray)
    undistorted = similar(image)
    for I in CartesianIndices(undistorted)
        u, v = Tuple(I)
        x′, y′ = inv(camera.A) ⋅ Vec(u, v, 1) # normalized coordinate
        x′′, y′′ = distort(Vec(x′, y′), camera.distcoefs)
        u′, v′ = camera.A ⋅ Vec(x′′, y′′, 1)
        i, j = round(Int, u′), round(Int, v′) # nearest-neighbor interpolation
        if checkbounds(Bool, image, i, j)
            undistorted[I] = image[i,j]
        else
            undistorted[I] = zero(eltype(image))
        end
    end
    undistorted
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

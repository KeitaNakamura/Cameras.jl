function testimage(name::String)
    if splitext(name)[1] == "buffalo"
        return load(joinpath(dirname(@__FILE__), "../images/buffalo.tif"))
    end
    throw(ArgumentError("test image $name is not exist"))
end

function perpendicular_distance(x::SVector{dim}, line::Pair{<: SVector{dim}, <: SVector{dim}}) where {dim}
    start, stop = line
    v1 = stop - start
    v2 = x - start
    n = normalize(v1)
    norm((v2 ⋅ n) * n - v2)
end

function _douglas_peucker(list::AbstractVector{SVector{dim, T}}, ϵ::Real) where {dim, T}
    # Find the point with the maximum distance
    index = 0
    dmax = zero(T)
    for i in 2:length(list)-1
        d = perpendicular_distance(list[i], list[1] => list[end])
        if d > dmax
            index = i
            dmax = d
        end
    end
    # If max distance is greater than ϵ, recursively simplify
    if dmax > ϵ
        lhs = _douglas_peucker(list[1:index], ϵ)
        rhs = _douglas_peucker(list[index:end], ϵ)
        vcat(lhs[1:end-1], rhs)
    else
        [list[1], list[end]]
    end
end

function douglas_peucker(list::AbstractVector{SVector{dim, T}}; thresh::Real, isclosed::Bool) where {dim, T}
    if isclosed
        dmax = zero(T)
        lhs_start = 1
        rhs_start = 1
        for i in 1:length(list)
            for j in i+1:length(list)
                d = norm(list[i] - list[j])
                if d > dmax
                    dmax = d
                    lhs_start = i
                    rhs_start = j
                end
            end
        end
        lhs = _douglas_peucker(list[lhs_start:rhs_start], thresh)
        rhs = _douglas_peucker(vcat(list[rhs_start:end], list[1:lhs_start]), thresh)
        vcat(lhs[1:end-1], rhs[1:end-1])
    else
        _douglas_peucker(list, thresh)
    end
end

function arclength(list::AbstractVector; isclosed::Bool)
    l = sum(norm(list[i]-list[i+1]) for i in 1:length(list)-1)
    if isclosed
        l += norm(list[1] - list[end])
    end
    l
end

function contourarea(list::AbstractVector{<: AbstractVector{T}}) where {T}
    poly = [list; [list[1]]]
    A = zero(T)
    for i in 1:length(poly)-1
        @inbounds begin
            Xᵢ = poly[i]
            Xᵢ₊₁ = poly[i+1]
            xᵢ, yᵢ = Xᵢ[1], Xᵢ[2]
            xᵢ₊₁, yᵢ₊₁ = Xᵢ₊₁[1], Xᵢ₊₁[2]
        end
        a = (xᵢ * yᵢ₊₁ - xᵢ₊₁ * yᵢ)
        A += a
    end
    A /= 2
    abs(A)
end

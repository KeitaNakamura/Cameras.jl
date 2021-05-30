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

function douglas_peucker(list::AbstractVector{SVector{dim, T}}, ϵ::Real) where {dim, T}
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
        lhs = douglas_peucker(list[1:index], ϵ)
        rhs = douglas_peucker(list[index:end], ϵ)
        vcat(lhs[1:end-1], rhs)
    else
        [list[1], list[end]]
    end
end

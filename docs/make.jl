using Documenter
using Cameras

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Cameras, :DocTestSetup, recursive = true,
    quote
        using Cameras
    end
)

makedocs(;
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Cameras],
    sitename = "Cameras.jl",
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Camera.md",
            "Utilities.md",
        ],
    ],
    doctest = true, # :fix
)

deploydocs(
    repo = "github.com/KeitaNakamura/Cameras.jl.git",
    devbranch = "main",
)

# For now the best version of Julia to use seems to be 1.9.0
# One can use juliaup to get this version, namely
# juliaup add 1.9.0 
# juliaup default 1.9.0

# In case Skyrmions3d has not been added to Julia, start there
# ]add https://github.com/chrishalcrow/Skyrmions3D.jl.git

# When this has been run the first time this should only need to be run again
# when the package is updated. 

# Do some standard loading for Skyrmions3d
using Skyrmions3D
using GLMakie
GLMakie.activate!()
Makie.inline!(false)

# We also load packages used in JNR functionality
using Optim
using IntervalArithmetic, IntervalRootFinding, ForwardDiff, StaticArrays, LinearAlgebra

# Initialise the skyrmion grid to work on. 
# I am yet to play about with this. 
my_skyrmion = Skyrmion([40, 40, 40], [0.2, 0.2, 0.2])

# Load up the jnr_skyrmions file to get the make_JNR! function
include("jnr_skyrmions.jl")

# Let's start with the  case, a single Skyrmion
# We first consider the parameters from Sutcliffe's paper. 
B = 1
weights = [1.0, 1.0]
poles = [Quaternion(0.0, 0.0, 0.0, 0.0) for a in 1:B+1]
poles[1] = Quaternion(2.0, 0.0, 0.0, 0.0)
poles[2] = Quaternion(-2.0, 0.0, 0.0, 0.0)
mu = sqrt(2.0)

# Let's calculate the Baryon number, Energy, and plot the skyrmion. 
make_JNR!(my_skyrmion, poles, weights, mu)
Baryon(my_skyrmion)
Energy(my_skyrmion)
plot_baryon_density(my_skyrmion)

# We can also intialise without picking mu and let the code determine a best 
# value
make_JNR!(my_skyrmion, poles, weights)
Baryon(my_skyrmion)
Energy(my_skyrmion)
# We find that the value of mu is similar (1.829...), and that the energy is 
# mildly smaller, while the baryon number is still close to 1. 

# We see what the effect of shifting the poles in the x4 direction is, as is
# to be expected when we shift too far the approximation to the holonomy
# becomes inaccurate, as the approximation uses an "ultradiscrete" 
# approximation using just 5 discrete points centred at x4=0. 
for t in LinRange(0.0, 0.5, 6)
    poles[1] = Quaternion(2, 0.0, 0.0, t)
    poles[2] = Quaternion(-2, 0.0, 0.0, t)
    make_JNR!(my_skyrmion, poles, weights)
    B = Baryon(my_skyrmion)
    E = Energy(my_skyrmion)
    println("t=", t, ": B=", B, " : E=", E)
end


# We can check also the other example of Sutcliffe, which behaves as desired
B = 2
weights = [1.0, 1.0, 1.0]
poles = [Quaternion(0.0, 0.0, 0.0, 0.0) for a in 1:B+1]
poles[1] = Quaternion(2.0, 0.0, 0.0, 0.0)
poles[2] = Quaternion(-1.0, sqrt(3.0), 0.0, 0.0)
poles[3] = Quaternion(-1.0, -sqrt(3.0), 0.0, 0.0)
mu = 2.0
make_JNR!(my_skyrmion, poles, weights, mu)
Baryon(my_skyrmion)
Energy(my_skyrmion)
plot_baryon_density(my_skyrmion)

# We consider the next new value which is the tetrahedron 
B = 3
weights = [1.0, 1.0, 1.0, 1.0]
# For a tetrahedral JNR 3-skyrmion we have the freedom to choose the parameter
# L, which for personal reasons (I have looked at different energies) we take
# to be 1.51. 
L = 1.51
poles = [Quaternion(0.0, 0.0, 0.0, 0.0) for a in 1:B+1]
poles[1] = Quaternion(L, L, L, 0.0)
poles[2] = Quaternion(L, -L, -L, 0.0)
poles[3] = Quaternion(-L, L, -L, 0.0)
poles[4] = Quaternion(-L, -L, L, 0.0)
# We let the method choose mu, which turns out to be 3.537...
make_JNR!(my_skyrmion, poles, weights)
Baryon(my_skyrmion)
Energy(my_skyrmion)
plot_baryon_density(my_skyrmion)

# We want to look at the signed JNR positions relative to this baryon density
# They look nicely like what we expect from methods in our paper. 
plot_JNR_positions(poles, weights, multiplicities=true, with_skyrmion=my_skyrmion)

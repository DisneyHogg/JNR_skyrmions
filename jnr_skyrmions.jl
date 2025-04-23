r"""
This code uses the work of Josh Cork, Linden Disney-Hogg, Derek Harland, and Paul Sutcliffe, to provide an implementation of JNR skyrmions for the Julia package Skyrmions3D by Chris Halcrow. 

In addition, the code provides methods to consider the positions and multiplicities of the JNR skyrmions. 

ToDO:

- Reduce the amount of repeated code between `make_JNR_full_quaternionic` and `make_JNR_pure_imaginary`. 
- Remove arbitrary (hyper)parameter decisions in `make_JNR` and `JNR_positions`. 
- Allow users to determine when `make_JNR` is verbose.
- Decide whether to raise a warning if `poles` and `weights` are provided which are of different size. 
- Write tests for functions. 
- Decide if keeping inputs being quaternions is sensible. 
- Add additional package depedencies (`Optim`, `IntervalArithmetic`, `IntervalRootFinding`, `ForwardDiff`, `LinearAlgebra`) to Skyrmions3D.jl? Only the first of these is required for the core make_JNR functionality.  

AUTHORS:

- Linden Disney-Hogg (2025-04-01): initial working version

"""

# Assume in this definition mu is given
# Julia allows for optional arguments by having multiple
# methods associated to a given function name. 
function make_JNR!(skyrmion, poles, weights, mu)
"""
    make_JNR!(skyrmion, poles, weights, mu)
    
Writes an JNR skyrmion in to `skyrmion`. The JNR data is given by `poles` and `weights`, while an additional parameter of the method is `mu`. 

If no `mu` is given, the function will determine an appropriate value by optimising a loss function containing the energy of the skyrmion and a penalty term to ensure that the baryon number is close to the theoretically desired value. 

### Input

- `poles` -- array of Quaternions (from the `GLMakie` package) of size `B+1` for some integer `B`
- `weights` -- array of (positive) floats of size `B+1`
- `mu` -- positive float

### Output

Acts on `skyrmion` to populate its pion field with that defined by the Harland-Sutcliffe approximation of the corresponding JNR data. 

### Examples (of data)
```
B = 1
weights = [1.0, 1.0]
poles = [Quaternion(0.0, 0.0, 0.0, 0.0) for a in 1:B+1]
poles[1] = Quaternion(2.0, 0.0, 0.0, 0.0)
poles[2] = Quaternion(-2.0, 0.0, 0.0, 0.0)
mu = sqrt(2.0)

B = 2
weights = [1.0, 1.0, 1.0]
poles = [Quaternion(0.0, 0.0, 0.0, 0.0) for a in 1:B+1]
poles[1] = Quaternion(2.0, 0.0, 0.0, 0.0)
poles[2] = Quaternion(-1.0, sqrt(3.0), 0.0, 0.0)
poles[3] = Quaternion(-1.0, -sqrt(3.0), 0.0, 0.0)
mu = 2.0
```

### Notes

Here a JNR skyrmion refers to the approximation of a ADHM skyrmion field coming from JNR data defined by Paul Sutcliffe in JHEP12(2024)054, equation (2.5). That approximation uses the work of Sutcliffe-Harland.

In that paper Sutcliffe simplifies equation (2.5) for the Skyrme field in the case where all JNR poles are pure imaginary to equation (2.8), and this code uses this simplified formula in that case. 

At present the code quietly chooses whether to use the full quaternionic or the pure imaginary formula. It may be helpful in the future to allow one to choose this to happen out loud. 

What we call weights[i] is lambda_i^2.  

If `mu` is determined automatically, Optim.jl is used to find an appropriate value in the range [0.1, 10.0]. Future work may wish to determine is this is a suitable range to investigate over (the numerical examples from Cork-Disney-Hogg all find values in the range [1.0, 4.0]). In particular one can see if the Skyrmion does not predominantly lie close to the x4=0 plane then the Harland-Sutcliffe approximation becomes less accurate, and so the numerical Baryon number of the approximation is unlikely to be close to the true value. This means that the range for choosing `mu` is not particularly reasonable.  
"""
    # Find the value of B+1 from the number of poles
    Bp1 = size(poles)[1]

    # If any of the poles are not pure imaginary use the general equation of (2.5)
    # Note this is a comparison of floats which in general is not a good idea, 
    # but there is no harm other than speed of computation to using the full
    # quaternionic method, so it is not a fatal flaw. 
    if any(ai->ai[4]!=0.0, poles)
        # print("Initialising using (2.8)")
        make_JNR_full_quaternionic!(skyrmion, poles, weights, mu)
    else
    # If all poles are pure imaginary use the specific simplified formula of (2.8)
        # print("Initialising using (2.5)")
        make_JNR_pure_imaginary!(skyrmion, poles, weights, mu)
    end
end

# Give a version to choose an optimal mu
# Requires Optim 
function make_JNR!(skyrmion, poles, weights)
    # Currently arbitrarily set the search range to be 0.1 to 10.0

    # At present this function F to be optimised works by penalising the
    # difference between B and B_theoretical exponentially. This may not
    # be the best decision. 
    B_theoretical = size(poles)[1]-1
    function F(mu)
        make_JNR!(skyrmion, poles, weights, mu)
        B = Baryon(skyrmion)
        return Energy(skyrmion)*exp(5*abs(abs(B)-B_theoretical)/B_theoretical) 
    end
    res = optimize(F, 0.1, 10.0)
    mu = Optim.minimizer(res)
    println("Taking mu=", mu)
    make_JNR!(skyrmion, poles, weights, mu)
end


function make_JNR_full_quaternionic!(skyrmion, poles, weights, mu)
"""
    make_JNR_full_quaternionic!(skyrmion, poles, weights, mu)
    
Method used in make_JNR! when poles are full quaternionic. See the documentation of that function for further details and examples.  
"""
    # Pick out the number of lattice points and their
    # values for each of the spatial directions. 
    lp, x = skyrmion.lp, skyrmion.x

    # Find the value of B+1 from the number of poles
    Bp1 = size(poles)[1]

    # The threads introduction means that the for
    # for loop uses multiple threads. 
    # inbounds speeds up looping when giving indices
    # for as an array as it skips checking that the 
    # indices are within the bounds of the array, 
    # this skipping can be unsafe. 
    Threads.@threads for k in 1:lp[3]
        @inbounds for j in 1:lp[2], i in 1:lp[1]
            # We shall construct zeta(x), rho(x),
            # and iota(x), then package them intro
            # psi(x). We then divide and conjugate
            # appropriately. 

            # Initialise the various components to calculate
            zeta0 = 0.0
            zeta1 = 0.0
            zeta2 = 0.0
            zeta3 = 0.0
            rho_p = 0.0
            iota1_p = 0.0
            iota2_p = 0.0
            iota3_p = 0.0
            rho_m = 0.0
            iota1_m = 0.0
            iota2_m = 0.0
            iota3_m = 0.0
 
            # First loop over I and compute zeta and rho
            for I in 1:Bp1
                # When inputting quaternions on gives the components in order
                # i, j, k, 1. 
                # Makie does not allow one to take the difference of quaternions
                # so we must do this ourselves
                # make x - aI and its norm.
                ai = poles[I]
                xmai0 = -ai[4]
                xmai1 = x[1][i]-ai[1]
                xmai2 = x[2][j]-ai[2]
                xmai3 = x[3][k]-ai[3]
                xmai = Quaternion(xmai1, xmai2, xmai3, xmai0)
                norm2_xmai = xmai0^2 + xmai1^2 + xmai2^2 + xmai3^2
                norm2_xmaipm = (xmai0+mu)^2 + xmai1^2 + xmai2^2 + xmai3^2
                norm2_xmaimm = (xmai0-mu)^2 + xmai1^2 + xmai2^2 + xmai3^2

                # Calculate the contribution to zeta
                # note we are treating weights[I] and lambda_I^2.
                li = weights[I]
                zeta0 += li*xmai0/norm2_xmai
                zeta1 += li*xmai1/norm2_xmai
                zeta2 += li*xmai2/norm2_xmai
                zeta3 += li*xmai3/norm2_xmai

                # calculate the contribution to rho
                rho_p += li/norm2_xmaipm
                rho_m += li/norm2_xmaimm

                # Within this loop over J, K to calculate iota. 
                for J in 1:Bp1, K in 1:Bp1
                    # This skips out the cases I=J and I=K as we know these
                    # contribute nothing to iota.
                    ((I==J) || (I==K)) && continue

                    # First calculate the real prefactor
                    lj = weights[J]
                    lk = weights[K]
                    aj = poles[J]
                    ak = poles[K]
                    norm2_xmajpm = (mu-aj[4])^2 + (x[1][i]-aj[1])^2 + (x[2][j]-aj[2])^2 + (x[3][k]-aj[3])^2
                    norm2_xmajmm = (-mu-aj[4])^2 + (x[1][i]-aj[1])^2 + (x[2][j]-aj[2])^2 + (x[3][k]-aj[3])^2
                    norm2_xmak = ak[4]^2 + (x[1][i]-ak[1])^2 + (x[2][j]-ak[2])^2 + (x[3][k]-ak[3])^2
                    prefactor_p = mu*li*lj*lk/(norm2_xmaipm*norm2_xmajpm*norm2_xmai*norm2_xmak)
                    prefactor_m = -mu*li*lj*lk/(norm2_xmaimm*norm2_xmajmm*norm2_xmai*norm2_xmak)

                    # Next calculate the quaternionic part
                    aimaj = Quaternion(ai[1]-aj[1], ai[2]-aj[2], ai[3]-aj[3], ai[4]-aj[4])
                    aimak_conj = Quaternion(-ai[1]+ak[1], -ai[2]+ak[2], -ai[3]+ak[3], ai[4]-ak[4])
                    prod = aimaj*xmai*aimak_conj
                   
                    iota1_p += prefactor_p*prod[1]
                    iota2_p += prefactor_p*prod[2]
                    iota3_p += prefactor_p*prod[3]
                    iota1_m += prefactor_m*prod[1]
                    iota2_m += prefactor_m*prod[2]
                    iota3_m += prefactor_m*prod[3]
                end
            end                  

            psi0_p = rho_p*(zeta0^2 + zeta1^2 + zeta2^2 + zeta3^2)
            psi0_m = rho_m*(zeta0^2 + zeta1^2 + zeta2^2 + zeta3^2)
            psi_m = Quaternion(iota1_m, iota2_m, iota3_m, psi0_m)
            psi_p_conj = Quaternion(-iota1_p, -iota2_p, -iota3_p, psi0_p)

            prod = psi_m*psi_p_conj
            norm = abs(prod)

            # Note a strange thing is done where the pion 
            # fields are ordered as 
            # [π1, π2, π3, π0]. This seems to be related
            # to the order of how quaternions store their
            # elements in Makie. 
            # This step is valid only if we have imaginary poles
            skyrmion.pion_field[i,j,k,1] = prod[1]/norm
            skyrmion.pion_field[i,j,k,2] = prod[2]/norm
            skyrmion.pion_field[i,j,k,3] = prod[3]/norm
            skyrmion.pion_field[i,j,k,4] = prod[4]/norm
        end
    end
end

function make_JNR_pure_imaginary!(skyrmion, poles, weights, mu)
"""
    make_JNR_pure_imaginary!(skyrmion, poles, weights, mu)
    
Method used in make_JNR! when poles are pure imaginary. See the documentation of that function for further details and examples.  
"""
    # Pick out the number of lattice points and their
    # values for each of the spatial directions. 
    lp, x = skyrmion.lp, skyrmion.x

    # Find the value of B+1 from the number of poles
    Bp1 = size(poles)[1]

    # The threads introduction means that the for
    # for loop uses multiple threads. 
    # inbounds speeds up looping when giving indices
    # for as an array as it skips checking that the 
    # indices are within the bounds of the array, 
    # this skipping can be unsafe. 
    Threads.@threads for k in 1:lp[3]
        @inbounds for j in 1:lp[2], i in 1:lp[1]
            # We shall construct zeta(x), rho(x),
            # and iota(x), then package them intro
            # psi(x). We then divide and conjugate
            # appropriately. 

            # Initialise the various components to calculate
            zeta1 = 0.0
            zeta2 = 0.0
            zeta3 = 0.0
            rho = 0.0
            iota1 = 0.0
            iota2 = 0.0
            iota3 = 0.0
 
            # First loop over I and compute zeta and rho
            for I in 1:Bp1
                # When inputting quaternions on gives the components in order
                # i, j, k, 1. 
                # Makie does not allow one to take the difference of quaternions
                # so we must do this ourselves
                # make x - aI and its norm.
                ai = poles[I]
                xmai1 = x[1][i]-ai[1]
                xmai2 = x[2][j]-ai[2]
                xmai3 = x[3][k]-ai[3]
                xmai = Quaternion(xmai1, xmai2, xmai3, 0.0)
                norm2_xmai = xmai1^2 + xmai2^2 + xmai3^2

                # Calculate the contribution to zeta
                # note we are treating weights[I] and lambda_I^2.
                li = weights[I]
                zeta1 += li*xmai1/norm2_xmai
                zeta2 += li*xmai2/norm2_xmai
                zeta3 += li*xmai3/norm2_xmai

                # calculate the contribution to rho
                rho += li/(norm2_xmai+mu^2)

                # Within this loop over J, K to calculate iota. 
                for J in 1:Bp1, K in 1:Bp1
                    # This skips out the cases I=J and I=K as we know these
                    # contribute nothing to iota.
                    ((I==J) || (I==K)) && continue

                    # First calculate the real prefactor
                    lj = weights[J]
                    lk = weights[K]
                    aj = poles[J]
                    ak = poles[K]
                    norm2_xmajpm = mu^2 + (x[1][i]-aj[1])^2 + (x[2][j]-aj[2])^2 + (x[3][k]-aj[3])^2
                    norm2_xmak = ak[4]^2 + (x[1][i]-ak[1])^2 + (x[2][j]-ak[2])^2 + (x[3][k]-ak[3])^2
                    prefactor = -mu*li*lj*lk/((norm2_xmai+mu^2)*norm2_xmajpm*norm2_xmai*norm2_xmak)

                    # Next calculate the quaternionic part
                    aimaj = Quaternion(ai[1]-aj[1], ai[2]-aj[2], ai[3]-aj[3], ai[4]-aj[4])
                    aimak_conj = Quaternion(-ai[1]+ak[1], -ai[2]+ak[2], -ai[3]+ak[3], ai[4]-ak[4])
                    prod = aimaj*xmai*aimak_conj
                   
                    iota1 += prefactor*prod[1]
                    iota2 += prefactor*prod[2]
                    iota3 += prefactor*prod[3]
                end
            end                  

            psi0 = rho*(zeta1^2 + zeta2^2 + zeta3^2)
            psi = Quaternion(iota1, iota2, iota3, psi0)
            prod = psi*psi
            norm = abs(prod)

            # Note a strange thing is done where the pion 
            # fields are ordered as 
            # [π1, π2, π3, π0]. This seems to be related
            # to the order of how quaternions store their
            # elements in Makie. 
            # This step is valid only if we have imaginary poles
            skyrmion.pion_field[i,j,k,1] = prod[1]/norm
            skyrmion.pion_field[i,j,k,2] = prod[2]/norm
            skyrmion.pion_field[i,j,k,3] = prod[3]/norm
            skyrmion.pion_field[i,j,k,4] = prod[4]/norm
        end
    end
end

function JNR_positions(poles, weights; multiplicities=false)
"""
   JNR_positions(poles, weights)

Finds the the points where the Skyrme field of the JNR skyrmions determined by `poles` and `weights` is -1, under the assumption that all poles are pure imaginary. 

### Input

- `poles` -- array of  pure imaginary Quaternions (from the `GLMakie` package) of size `B+1` for some integer `B`
- `weights` -- array of (positive) floats of size `B+1`
- `multiplicities` -- (default: false) logical determining whether to provide the multiplicities of positions (when possible) 

### Output

- `positions` -- array of Point3f data giving the positions of the constituent skyrmions. NOT CORRECT RIGHT NOW FORMAT-WISE
- `signs` -- (only if `multiplicities==true`) array of floats taking values +-1.0 giving the sign of the corresponding constituent skyrmion. 

### Notes

Thinking of a skymrion as a collection of constituent 1-skyrmions and anti-1-skymrions we determine the position and signs of these constituents by where the skyrmion field is -1 and it's local winding number at that point. 

For JNR skyrmions with pure imaginary poles the locations of these constituent positions are found where a function zeta (defined in the Sutcliffe paper) is 0. This is what is tested to find the positions. 

IntervalRootFinding.jl is used to find all roots of zeta, and then removes any solutions which actually contain poles. Positions are then returned as midpoints of intervals. To improve performance the bounding box in which which to search shrunk by a small amount (at most 1E-8, chosen because of the default tolerance taken in IntervalRootFinding). If the positions were within 1e-8 of the poles this would cause errors which one must be alert to, but that would be true because of the root finding tolerance regardless. One might in the future all for this tolerance to be an optional argument. 

If `multiplicities` is true, then the method used to calculate signs is that form the Cork-Disney-Hogg conjecture. 

All this requires that the positions be generic, i.e. the local winding number is +-1. In the event that there are coincident positions the numerical position finding can give non-sensical results, and the method for finding the multiplicity will not work. As an example of when this can fail, consider the B=2 case of equal weights where the poles form an equilateral triangle. 

In future this method could perhaps be improved by using the convex hull of the poles as the constraints, rather than just the bounding box. 

At present this function can be inefficient when positions are close together, with profiling suggesting that the root-finding procedure is the slow part. Future work could perhaps improve the running of this. 

In the case that B=1 and B=2 exact methods for finding the positions are known, and these could be implemented as special cases in the future. 

"""
    Bp1 = size(poles)[1]
    
    # Could give specific methods for the simpler cases.
    # if Bp1 == 2
    #     return
    # end
    # if Bp1 == 3
    #     return 
    # end

    # Extract the bounding box containing all positions. 
    xmin = minimum([poles[i][1] for i in 1:Bp1])
    xmax = maximum([poles[i][1] for i in 1:Bp1])
    ymin = minimum([poles[i][2] for i in 1:Bp1])
    ymax = maximum([poles[i][2] for i in 1:Bp1])
    zmin = minimum([poles[i][3] for i in 1:Bp1])
    zmax = maximum([poles[i][3] for i in 1:Bp1])

    # This shall be an offset to the intervals to prevent NaI in evaluations. 
    dx = min(1E-8, (xmax-xmin)/10)
    dy = min(1E-8, (ymax-ymin)/10)
    dz = min(1E-8, (zmax-zmin)/10)

    X = interval(xmin+dx, xmax-dx; format = :infsup)
    Y = interval(ymin+dy, ymax-dy; format = :infsup)
    Z = interval(zmin+dz, zmax-dz; format = :infsup)

    # We define the function zeta of which we want to find the zeros
    # To try and eke out some speed we may later remove calls to weights if all
    # weights are equal. Note that though this is float comparison there is no 
    # danger as we just fall back to a less efficient method.
    function zeta((x, y, z))
        # We initialise the components and calculate them in a sum
        zeta1 = 0.0
        zeta2 = 0.0
        zeta3 = 0.0
        Threads.@threads for i in 1:Bp1
            ai = poles[i]
            xmai1 = x-ai[1]
            xmai2 = y-ai[2]
            xmai3 = z-ai[3]
            li = weights[i]
            norm2_xmai = xmai1^2 + xmai2^2 + xmai3^2
            zeta1 += li*xmai1/norm2_xmai
            zeta2 += li*xmai2/norm2_xmai
            zeta3 += li*xmai3/norm2_xmai
        end
        # Using StaticVectors is supposed to give better performance here. 
        return SVector(zeta1, zeta2, zeta3)
    end

    # Find the roots of zeta, including the poles, as certified intervals. 
    # The Krawczyk appears to work well with repeated roots. 
    rts = roots(zeta, SVector(X, Y, Z), contractor=Krawczyk)
    # Now loop to find the true positions in these
    # Here we set what type the entries of this list to be populated will be. 
    # This could break I suppose if the entries are not Float64 for some reason. 
    # positions = SVector{Float64}[]
    positions = []
    for rt in rts
        if any(in_interval(ai[1], rt.region[1]) && in_interval(ai[2], rt.region[2]) && in_interval(ai[3], rt.region[3]) for ai in poles)
            continue
        end
        px = (rt.region[1].bareinterval.lo + rt.region[1].bareinterval.hi)/2
        py = (rt.region[2].bareinterval.lo + rt.region[2].bareinterval.hi)/2
        pz = (rt.region[3].bareinterval.lo + rt.region[3].bareinterval.hi)/2
        push!(positions, [px, py, pz])
    end

    # This is an untested method which relies on mathematics not proven.
    if multiplicities
        # Can replace the use of jacobian here with the analytic expression for
        # dzeta. 
        dzeta(x) = jacobian(zeta, x)
        # For a mathematically correct version can calculate Xi and find the 
        # sign of Re(Xi) which is then included as a prefactor. 
        signs = [-sign(det(dzeta(p))) for p in positions]
        return positions, signs
    end

    return positions
end

# Allow for kwargs after testing
function plot_JNR_positions(poles, weights; multiplicities=false, with_skyrmion=nothing)
"""
    plot_JNR_positions(poles, weights)

Plot the positions of the poles and the skyrmion centres associated to JNR data of `poles` and `weights`. 

### Input

- `poles` -- array of  pure imaginary Quaternions (from the `GLMakie` package) of size `B+1` for some integer `B`
- `weights` -- array of (positive) floats of size `B+1`
- `multiplicities` -- (default: false) logical determining whether to provide the multiplicities of positions (when possible) 
- `with_skyrmion` -- (default: nothing) if a skyrmion is given, its baryon density plot is provided on the same axes as the positions. 

### Output

- `fig` -- A figure showing the JNR positions using Makie.meshscatter. If `multiplicities==true` then the constituent positions are coloured based upon their sign. 

### Notes

To compute the positions and their signs `JNR_positions` is used, and the details of that function may be seen in its own documentation. 

The skyrmion baryon density could obscure positions, and so one should be alert to this. A simple example of this is with the B=1 JNR skyrmion. 

Another method for viewing the positions could just be to plot the isosurfaces of the relevant skyrmion field component. This is essentially using the existing method `plot_field`, namely something along the lines of `plot_field(skyrmion, component=4, iso_value=-0.99)`. Unforuntately at present this does not appear to work well.  
"""
    # Would require work to make the bdp the correct size and 
    # sufficiently transparent. 
    if isnothing(with_skyrmion)
        fig = Figure()
        ax = Axis3(fig[1,1])
    else
        # May wish to modify this if the skyrmions baryon density plot is 
        # blocking the positions, ideally by mkaing it transparents, which
        # would require a modification of plotting.jl. 
        fig = plot_baryon_density(with_skyrmion)
        ax = fig.content[1]
    end
    # right now this is an arbitrary parameter hand picked, a better method 
    # for choosing this probably exists. 
    ms = 0.05

    if multiplicities
        positions, signs = JNR_positions(poles, weights, multiplicities=true)
        positions = [Point3f(p) for p in positions]
        if all(si>0 for si in signs)
            groups = [positions]
        else
            pp = [positions[i] for i in 1:size(positions)[1] if signs[i]>0]
            np = [positions[i] for i in 1:size(positions)[1] if signs[i]<0] 
            groups = [pp, np]
        end
    else
        positions = JNR_positions(poles, weights, multiplicities=false)
        groups = [[Point3f(p) for p in positions]]
    end

    # Makie.wong_colors() uses colours that are better suited for colourblind users. 
    for i in 1:size(groups)[1]
        meshscatter!(ax, groups[i], color=Makie.wong_colors()[i], markersize=ms)
    end

    return fig
end

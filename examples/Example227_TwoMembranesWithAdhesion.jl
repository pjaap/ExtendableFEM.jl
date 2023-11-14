#=

# 227 : Two membranes with adhesion
([source code](SOURCE_URL))

=#

module Example227_TwoMembranesWithAdhesion

using ExtendableFEM
using ExtendableGrids
using SparseArrays
using LinearAlgebra
using GridVisualize



# max(0,x) with smoothing at (-ϵ,ϵ)
function max0(x,ϵ=0)
	if x ≤ -ϵ
		return 0
	elseif x < ϵ
		return (x+ϵ)^2/(4ϵ)
	else
		return x
	end
end

# diff max(0,x) with smoothing at (-ϵ,ϵ)
function d_max0(x,ϵ=0)
	if x ≤ -ϵ
		return 0
	elseif x < ϵ
		return (x+ϵ)/(2ϵ)
	else
		return 1
	end
end



# heaviside function with smoothing at (-ϵ,0)
function χ(x,ϵ=0)


	if x ≤ -ϵ
		return 0
	elseif x < 0
		return -2x^3/(ϵ^3) - 3x^2/(ϵ^2) + 1
		# return 1 + x/ϵ
	else
		return 1
	end
end


# diff of heaviside function with smoothing at (-ϵ,0)
function d_χ(x,ϵ)

	if x ≤ -ϵ
		return 0
	elseif x < 0
		return -6x^2/ϵ^3 - 6x/ϵ^2
		# return 1/ϵ
	else
		return 0
	end
end


## nonlinear kernel
function make_kernel(f1, # membrane force 1 
				     f2, # membrane force 2
					 θ,  # adhesion parameter
	                 γ,  # penetration penalty
	                 ϵ)  # smoothing parameter


	function nonlinear_kernel!(result, input, qpinfo )

		u1  = input[1]
		∇u1 = view(input, 2:3)
		u2  = input[4]
		∇u2 = view(input, 5:6)
		


		# stash ... θ*( -d_χ(u2 - u1, ϵ) )

		# first membrane force incl non-penetration
		result[1]   =  γ*max(0, u1 - u2 )- f1   - θ*( d_χ(u1 - u2, ϵ) )
		result[2:3] = ∇u1												             

		# second membrane force 
		result[4]   =   -γ*max(0, u1 - u2 ) - f2 + θ*( d_χ(u1 - u2, ϵ) )     	     
		result[5:6] = ∇u2												             

		return nothing
	end

	return nonlinear_kernel!
end


## custom sparsity pattern for the jacobians of the nonlinear_kernel (Symbolcs cannot handle conditional jumps)
## note: jacobians are defined row-wise
# rows = [1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7]
# cols = [1, 4, 7, 2, 3, 1, 4, 7, 5, 6, 1, 4, 7]
# vals = ones(Bool, length(cols))
# sparsity_pattern = sparse(rows,cols,vals)



function main(;
	f1 = 0,
	f2 = 0,
	γ = 1e8,
	θ = 1,
	ϵ = 1e-3,
	boundary_offset = 1,
	N = 32,
	order = 1,
	Plotter = nothing,
	kwargs...)

	## choose mesh,
	h = 1/(N+1)
	xgrid = simplexgrid(0:h:1,0:h:1)

	## problem description
	PD = ProblemDescription()
	u1 = Unknown("u1"; name = "lower membrane")
	u2 = Unknown("u2"; name = "upper membrane")
	assign_unknown!(PD, u1)
	assign_unknown!(PD, u2)
	
	@warn "sparse_jacobians = false for prototyping"
	nonlinear_kernel! = make_kernel(f1,f2,θ,γ,ϵ)
	assign_operator!(PD, NonlinearOperator(nonlinear_kernel!, [id(u1), grad(u1), id(u2), grad(u2)]; sparse_jacobians=false, bonus_quadorder=2, kwargs...))
	assign_operator!(PD, HomogeneousBoundaryData(u1; regions = 1:4,  kwargs...))
	assign_operator!(PD, HomogeneousBoundaryData(u2; regions = 1:4, value = boundary_offset,  kwargs...))

	## create finite element space
	FES1 = FESpace{H1Pk{1, 2, order}}(xgrid)
	FESs = [FES1, FES1]
	sol = FEVector(FESs; tags = [u1,u2])

	interpolate!(sol[u1], (result,qp) -> ( result[1] =  qp.x[1]*(1-qp.x[1])+qp.x[2]*(1-qp.x[2])   ) )
	interpolate!(sol[u2], (result,qp) -> ( result[1] = boundary_offset - ( qp.x[1]*(1-qp.x[1])+qp.x[2]*(1-qp.x[2])  ) ) )

	## solve
	sol = solve(PD, FESs; init = sol, maxiterations=999, target_residual=1e-8, kwargs...)

	u_min = minimum(sol.entries)
	u_max = maximum(sol.entries)

	
    ## plot
    pl = GridVisualizer(; Plotter = Plotter, layout = (1,2), clear = true, size = (1000,500))
    scalarplot!(pl[1,1],xgrid, view(nodevalues(sol[u1]),1,:), title = "lower membrane u1", flimits = (u_min, u_max) )
    scalarplot!(pl[1,2],xgrid, view(nodevalues(sol[u2]),1,:), title = "upper membrane u2", flimits = (u_min, u_max) )

	# compute portion of contact
	dofs_in_contact = 0
	for i in 1:length(sol[u2])
		if abs( sol[u1][i] - sol[u2][i]) < 1e-4
			dofs_in_contact += 1
		end
	end

	portion_of_contact = dofs_in_contact/length(sol[u1])

	@show portion_of_contact

	@show maximum(view(sol[u1]) .- view(sol[u2]))

	return sol
end


end # module

using GLMakie

force_density = -4

@time Example227_TwoMembranesWithAdhesion.main(Plotter=GLMakie, γ=1e8, θ=10, ϵ=1e-1, N=50, f1 = force_density, f2 = -force_density)

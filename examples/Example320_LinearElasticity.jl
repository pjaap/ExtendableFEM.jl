#=

# 320 : Linear Elasticity
([source code](@__SOURCE_URL__))

This example computes the displacement field ``u`` of the linear elasticity problem

div( σ ) = f, where σ = C:ε for a given forth-order tensor C and linear strain ε = ½(∇u + ∇uᵀ)
=#

module Example230_NonlinearElasticity

using ExtendableFEM
using ExtendableGrids
using GridVisualize
using StaticArrays
using LinearAlgebra: tr, norm, I
using Test #hide


## result =  ½(∇u + ∇uᵀ) for ∇u = input
function strain!(result, input)
	ε  = tensor_view(result, 1, TDMatrix(3))
	∇u = tensor_view(input,  1, TDMatrix(3))

	@. ε = (∇u + ∇u')/2.0
	return nothing
end


struct StVenenat_Kirchhoff_tensor{T}
	# Lamé parameters
	λ::T
	μ::T
end


## compute σ = C:ε for result=σ and input=∇u in flat form
function (C::StVenenat_Kirchhoff_tensor)(result, input, qpinfo)

	σ =  tensor_view(result, 1, TDMatrix(3))
	∇u = tensor_view(input,  1, TDMatrix(3))

	# compute strain
	ε = (∇u + ∇u') / 2

	# the mapping depends on the definition of the linear map.
	# use St. Venant--Kirchhoff for testing
	@. σ = C.λ*tr(ε)*I + 2*C.μ*ε

	return nothing

end

## everything is wrapped in a main function
function main(;
	ν = 0.3,             # Poisson number for each region/material
	E = 2.1,             # Elasticity modulus for each region/material
	N = 10,              # Grid cells per axis
	order = 2,           # finite element order
	Plotter = nothing,
	kwargs...)

	## convert to Lamé parameters
	λ = E * ν / ((1 - 2 * ν) * (1 + ν))
	μ = E / (2 * (1 + ν))

	# create a material tensor from the parameters
	𝐂 = StVenenat_Kirchhoff_tensor(λ,μ)

	## generate bimetal mesh
	h = 1/N
	xgrid = simplexgrid(0:h:1, 0:h:1, 0:h:1 )

	## create finite element space and solution vector
	FES = FESpace{H1P2{3, 3}}(xgrid)

	## problem description
	PD = ProblemDescription()

	u = Unknown("u"; name = "displacement")
	assign_unknown!(PD, u)

	@warn "We want to pass a LinearOperator in the next line, but this does not work :("
	assign_operator!(PD, NonlinearOperator(𝐂, [grad(u)]; kwargs...))
	assign_operator!(PD, HomogeneousBoundaryData(u; regions = [1], kwargs...))

	## solve
	sol = solve(PD, FES; kwargs...)

	## displace mesh and plot
	plt = GridVisualizer(; Plotter = Plotter, layout = (2, 1), clear = true, size = (1000, 1500))

	grad_nodevals = nodevalues(grad(u), sol)
	strain_norms = zeros(num_nodes(xgrid))

	temp = @MVector zeros(9)

	for j in 1:num_nodes(xgrid)
		strain!(temp, view(grad_nodevals, :, j))
		strain_norms[j] = norm(temp)
	end

	scalarplot!(plt[1,1], xgrid, strain_norms, levels = 3, colorbarticks = 7, title = "||ε(u)||")
	#vectorplot!(plt[1, 1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), rasterpoints = 20, clear = false)
	#vectorplot!(plt[2, 1], xgrid, eval_func_bary(PointEvaluator([id(u)], sol)), rasterpoints = 20, clear = false)
	displace_mesh!(xgrid, sol[u])
	gridplot!(plt[2, 1], xgrid, linewidth = 1, title = "displaced mesh")
	println(stdout, unicode_gridplot(xgrid))

	return strain_nodevals, plt
end

generateplots = default_generateplots(Example230_NonlinearElasticity, "example230.png") #hide
function runtests() #hide
	strain, plt = main(;) #hide
	@test maximum(strain) ≈ 0.17318901080065996 #hide
end #hide
end

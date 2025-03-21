###########################################
### COMBINE DOFS (e.g. for periodicity) ###
###########################################

mutable struct CombineDofs{UT, CT} <: AbstractOperator
    uX::UT                  # component nr for dofsX
    uY::UT                  # component nr for dofsY
    coupling_info::CT
    FESX::Any
    FESY::Any
    assembler::Any
    parameters::Dict{Symbol, Any}
end

default_combop_kwargs() = Dict{Symbol, Tuple{Any, String}}(
    :penalty => (1.0e30, "penalty for fixed degrees of freedom"),
    :verbosity => (0, "verbosity level"),
)

# informs solver in which blocks the operator assembles to
function dependencies_when_linearized(O::CombineDofs)
    return [O.uX, O.uY]
end


"""
````
function CombineDofs(uX, uY, dofsX, dofsY, factors; kwargs...)
````

When assembled, the dofsX of the unknown uX will be coupled
with the dofsY of uY, e.g., for periodic boundary conditions.


Keyword arguments:
$(_myprint(default_combop_kwargs()))

"""
function CombineDofs(uX, uY, dofsX, dofsY, factors = ones(Int, length(dofsX)); kwargs...)
    # build a sparse matrix from dofsX, dofsY, factors
    # we have to set the size (maximum of indices: matrix may be smaller than the system matrix, this is ok)
    # for sparse matrices, define a `combine` function for duplicates: pick the latest entry
    # convention: dofsX acts as column indices!
    coupling_matrix = sparse(dofsY, dofsX, factors, maximum(dofsY), maximum(dofsX), (a, b) -> b)
    return CombineDofs(uX, uY, coupling_matrix; kwargs...)
end

"""
````
function CombineDofs(uX, uY, coupling_matrix::AbstractMatrix; kwargs...)
````
Input:
    uX: FEVectorBlock of the "from" boundary region
    uY: FEVectorBlock of the "to" boundary region (usually the same as uX)
    coupling_matrix: coupling matrix computed from `get_periodic_coupling_matrix`

Keyword arguments:
$(_myprint(default_combop_kwargs()))
"""
function CombineDofs(uX, uY, coupling_matrix::AbstractMatrix; kwargs...)
    parameters = Dict{Symbol, Any}(k => v[1] for (k, v) in default_combop_kwargs())
    _update_params!(parameters, kwargs)
    return CombineDofs(uX, uY, coupling_matrix, nothing, nothing, nothing, parameters)
end

function apply_penalties!(A, b, sol, CD::CombineDofs{UT, CT}, SC::SolverConfiguration; assemble_matrix = true, assemble_rhs = true, kwargs...) where {UT, CT}
    if UT <: Integer
        ind = [CD.ux, CD.uY]
    elseif UT <: Unknown
        ind = [get_unknown_id(SC, CD.uX), get_unknown_id(SC, CD.uY)]
    end
    build_assembler!(CD, [sol[j] for j in ind])
    return CD.assembler(A.entries, b.entries, assemble_matrix, assemble_rhs)
end


function build_assembler!(CD::CombineDofs{UT, CT}, FE::Array{<:FEVectorBlock, 1}; time = 0.0) where {UT, CT <: AbstractMatrix}
    ## check if FES is the same as last time
    FESX, FESY = FE[1].FES, FE[2].FES
    if (CD.FESX != FESX) || (CD.FESY != FESY)
        coupling_matrix = CD.coupling_info
        offsetX = FE[1].offset
        offsetY = FE[2].offset
        if CD.parameters[:verbosity] > 0
            @info ".... coupling $(length(coupling_matrix.nzval)) dofs"
        end

        function assemble(A::AbstractSparseArray{T}, b::AbstractVector{T}, assemble_matrix::Bool, assemble_rhs::Bool, kwargs...) where {T}
            if assemble_matrix

                # first transfer all entries from the constrained rows
                for dof_i in 1:size(coupling_matrix, 2)
                    # this col-view is efficient
                    coupling_i = @views coupling_matrix[:, dof_i]

                    # first we only consider dof_i ∉ B_c
                    if nnz(coupling_i) != 0
                        continue
                    end

                    for dof_j in 1:size(A, 2)

                        targetrow = dof_i + offsetX
                        targetcol = dof_j + offsetY

                        coupling_j = @views coupling_matrix[:, dof_j]
                        if nnz(coupling_j) != 0
                            # j ∈ B_c
                            A[targetrow, targetcol] = 0
                        else
                            # j ∉ B_c

                            # this is expensive ...
                            inverse_coupling_i = coupling_matrix[dof_i, :]
                            inverse_coupling_j = coupling_matrix[dof_j, :]

                            coupled_dofs_k, weights_k = findnz(inverse_coupling_i)
                            coupled_dofs_l, weights_l = findnz(inverse_coupling_j)

                            for (dof_k, weight_ki) in zip(coupled_dofs_k, weights_k)
                                sourcerow = dof_k + offsetX
                                sourcecol = dof_j + offsetY
                                val = A[sourcerow, sourcecol]
                                _addnz(A, targetrow, targetcol, val, weight_ki)
                            end

                            for (dof_l, weight_lj) in zip(coupled_dofs_l, weights_l)
                                sourcerow = dof_i + offsetX
                                sourcecol = dof_l + offsetY
                                val = A[sourcerow, sourcecol]
                                _addnz(A, targetrow, targetcol, val, weight_lj)
                            end

                            for (dof_k, weight_ki) in zip(coupled_dofs_k, weights_k)
                                for (dof_l, weight_lj) in zip(coupled_dofs_l, weights_l)
                                    sourcerow = dof_k + offsetX
                                    sourcecol = dof_l + offsetY
                                    val = A[sourcerow, sourcecol]
                                    _addnz(A, targetrow, targetcol, val, weight_lj * weight_ki)
                                end
                            end
                        end
                    end
                end

                # now eliminate the constrained rows
                for dof_i in 1:size(coupling_matrix, 2)
                    # this col-view is efficient
                    coupling_i = @views coupling_matrix[:, dof_i]

                    # now we only consider dof_i ∈ B_c
                    if nnz(coupling_i) == 0
                        continue
                    end

                    targetrow = dof_i + offsetX
                    for dof_j in 1:size(A, 2)
                        targetcol = dof_j + offsetY
                        A[targetrow, targetcol] = 0.0
                    end

                    coupled_dofs, weights = findnz(coupling_i)

                    # put the linear combination here
                    A[targetrow, targetrow] = -1.0
                    # for (dof_j, weight) in zip(coupled_dofs, weights)
                    #     targetcol = dof_j + offsetY
                    #     A[targetrow, targetcol] = weight
                    # end
                end

                flush!(A)
            end

            if assemble_rhs

                # first transfer all entries from the constrained rows
                for dof_i in 1:size(coupling_matrix, 2)
                    # this col-view is efficient
                    coupling_i = @views coupling_matrix[:, dof_i]

                    # first we only consider dof_i ∉ B_c
                    if nnz(coupling_i) != 0
                        continue
                    end

                    inverse_coupling_i = coupling_matrix[dof_i, :]
                    coupled_dofs_k, weights_k = findnz(inverse_coupling_i)

                    targetrow = dof_i + offsetX

                    for (dof_k, weight_ki) in zip(coupled_dofs_k, weights_k)
                        sourcerow = dof_k + offsetX
                        b[targetrow] += weight_ki * b[sourcerow]
                    end
                end

                # now eliminate the constrained rows
                for dof_i in 1:size(coupling_matrix, 2)
                    # this col-view is efficient
                    coupling_i = @views coupling_matrix[:, dof_i]

                    # now we only consider dof_i ∈ B_c
                    if nnz(coupling_i) == 0
                        continue
                    end

                    targetrow = dof_i + offsetX
                    b[targetrow] = 0.0
                end
            end

            return nothing
        end

        CD.assembler = assemble
        CD.FESX = FESX
        CD.FESY = FESY
    end

    return nothing
end


# function build_assembler!(CD::CombineDofs{UT, CT}, FE::Array{<:FEVectorBlock, 1}; time = 0.0) where {UT, CT <: AbstractMatrix}
#     ## check if FES is the same as last time
#     FESX, FESY = FE[1].FES, FE[2].FES
#     if (CD.FESX != FESX) || (CD.FESY != FESY)
#         coupling_matrix = CD.coupling_info
#         offsetX = FE[1].offset
#         offsetY = FE[2].offset
#         if CD.parameters[:verbosity] > 0
#             @info ".... coupling $(length(coupling_matrix.nzval)) dofs"
#         end
#         function assemble!(A::AbstractSparseArray{T}, b::AbstractVector{T}, assemble_matrix::Bool, assemble_rhs::Bool, kwargs...) where {T}
#             if assemble_matrix

#                 # put FE adjacency information into this matrix row
#                 for dof_i in 1:size(coupling_matrix, 2)
#                     # this col-view is efficient
#                     coupling_i = @views coupling_matrix[:, dof_i]
#                     # do nothing if no coupling for dof_i
#                     if nnz(coupling_i) == 0
#                         continue
#                     end

#                     targetrow = dof_i + offsetX

#                     coupled_dofs_i, weights_i = findnz(coupling_i)

#                     for dof_j in 1:size(coupling_matrix, 2)
#                         # this col-view is efficient
#                         coupling_j = @views coupling_matrix[:, dof_j]

#                         if nnz(coupling_j) == 0
#                             # not both i and j on the boundary

#                             targetcol = dof_j + offsetY
#                             for (dof_k, weight_k) in zip(coupled_dofs_i, weights_i)
#                                 sourcerow = dof_k + offsetX
#                                 sourcecol = targetcol
#                                 val = A[sourcerow, sourcecol]
#                                 _addnz(A, targetrow, targetcol, val, weight_k)
#                             end


#                         else
#                             # i and j on the same boundary

#                             coupled_dofs_j, weights_j = findnz(coupling_j)
#                             targetcol = dof_j + offsetY

#                             for (dof_k, weight_k) in zip(coupled_dofs_i, weights_i)
#                                 for (dof_l, weight_l) in zip(coupled_dofs_j, weights_j)

#                                     sourcerow = dof_k + offsetX
#                                     sourcecol = dof_l + offsetY

#                                     val = A[sourcerow, sourcecol]
#                                     # @show sourcerow, sourcecol, targetrow, targetcol, val, weight_k, weight_l
#                                     _addnz(A, targetrow, targetcol, val, weight_k * weight_l)

#                                 end
#                             end
#                         end
#                     end
#                 end

#                 # replace the geometric coupling rows
#                 for dof_i in 1:size(coupling_matrix, 2)
#                     # go through the rows of the matrix; use transposed matrix implicitly

#                     # this row-view is not efficient :(
#                     coupling_i = coupling_matrix[dof_i, :]
#                     # do nothing if no coupling for dof_i
#                     if nnz(coupling_i) == 0
#                         continue
#                     end


#                     # get the coupled dofs of dof_i and the corresponding weights
#                     coupled_dofs, weights = findnz(coupling_i)

#                     sourcerow = dof_i + offsetX

#                     # eliminate the sourcerow
#                     for col in 1:size(A, 2)
#                         A[sourcerow, col] = 0
#                     end

#                     # replace sourcerow with coupling linear combination
#                     _addnz(A, sourcerow, sourcerow, -1.0, 1)
#                     for (dof_j, weight) in zip(coupled_dofs, weights)
#                         # weights for ∑ⱼ wⱼdofⱼ - dofᵢ = 0
#                         _addnz(A, sourcerow, dof_j + offsetY, weight, 1)
#                     end

#                 end
#                 flush!(A)
#             end

#             if assemble_rhs

#                 for dof_i in 1:size(coupling_matrix, 2)
#                     # this col-view is efficient
#                     coupling_i = @views coupling_matrix[:, dof_i]
#                     # do nothing if no coupling for dof_i
#                     if nnz(coupling_i) == 0
#                         continue
#                     end

#                     # get the coupled dofs of dof_i and the corresponding weights
#                     coupled_dofs, weights = findnz(coupling_i)

#                     # transfer all assembly information to dof_i
#                     targetrow = dof_i + offsetY
#                     for (dof_j, weight) in zip(coupled_dofs, weights)
#                         sourcerow = dof_j + offsetY
#                         b[targetrow] += weight * b[sourcerow]
#                     end
#                 end


#                 for dof_i in 1:size(coupling_matrix, 2)
#                     coupling_i = coupling_matrix[dof_i, :]
#                     # do nothing if no coupling for dof_i
#                     if nnz(coupling_i) == 0
#                         continue
#                     end

#                     sourcerow = dof_i + offsetX

#                     b[sourcerow] = 0.0

#                 end
#             end

#             return nothing
#         end

#         CD.assembler = assemble!
#         CD.FESX = FESX
#         CD.FESY = FESY
#     end

#     return nothing
# end

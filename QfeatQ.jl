module QfeatQ

using ITensors: findfirst
using Base: updated_methodloc, Float64
export Tab, PauliOp, initialize, state_generation_Controller, create_data_acquisition_scheme_Controller, measurement_Controller, observable_prediction_Controller,simpleCliffordToMat,generateCompleteRandomObservables, readObservables, generateSimpleRandomClifford, naiveRandomTabClifford, generateSpinEW, generateEW_cor, reduceMultiplets

#Template for short-Implementation of individual framework components into a sinlge script
using Juqst
using LinearAlgebra
using StatsBase
using Distributions
using HDF5, JLD
using ProgressMeter


#--------Experiment Simulation Controller Framework
#Load all components:
include("state_generation.jl")
include("create_data_acquisition_scheme.jl")
include("measurement.jl")
include("prediction_shadow.jl")
include("generate_observables.jl")

#--------(Complete) Tab framework:  (The Backend)

Vdiag(x)=diag(x)
Pdiag(x)=collect(Diagonal(x))
Pupps(x)= triu(x,1)

setupTab(num_qudits, d) = vcat([transpose([if row==col 1 else 0 end for col in 1:2*num_qudits]) for row in 1:2*num_qudits]...)
setupStab(num_qudits, d) = [["+1",join([if i==j "Z" else "I" end for j in 1:num_qudits])] for i in 1:num_qudits]

mutable struct Tab  #Structure follows: DOI:10.1103/PhysRevA.71.042315
    num_qudits::Int
    d::Int
    cmds::Array{Array{Any,1},1} #Which "named" clifford commands were performed to arrive/represent this tab: e.g. [("P", [1])]
    XZencoding::Array{Int, 2} #Encodes Paulis from mapping of elementary Pauli XZ basis E_i : mod d & 2n x 2n qudits
                              #Collumns = Single pauli!
    phases::Array{Int, 1} #WARNING: Encodes power of omegaPhase mod 2d, NOT the phase of XZencoding cols themselves!
    stab::Array{Array{String, 1}, 1} #"Plaintext" stabilizers with phase, Array of tuples w/ 0:phase, 1:pauli

   
    Tab(num_qudits,d, cmds=[],XZencoding=setupTab(num_qudits, d),phases=[0 for i in 1:(2*num_qudits)], stab=setupStab(num_qudits, d))=new(num_qudits,d, cmds, XZencoding,phases, stab)
 end

 struct PauliOp
    num_qudits::Int
    d::Int
    XZencoding::Array{Int, 1}
    phase::Int #Power of omegaPhase
    PauliOp(num_qudits, d, XZencoding,phase) = new(num_qudits, d, XZencoding, phase)
 end


 function checkConditions(input::Tab,d)
    conda, condb = checkConditions(input.XZencoding, input.phases,d)
    return conda, condb
end
function checkConditions(inputXZ::Array{Int, 2}, inputPhases::Array{Int, 1}, d)
    n=Int(length(inputPhases)/2)
    omega = [zeros((n,n)) Matrix(1I, n, n); -Matrix(1I, n, n) zeros((n,n))]
    
    isSymplec = maximum(abs.(transpose(inputXZ)*(omega*inputXZ) - omega)) < 0.001
    U = [zeros((n, n)) zeros((n, n)); Matrix(1I, n, n) zeros((n, n))]

    condIsStat = maximum(abs.(mod.((d-1)*Vdiag(transpose(inputXZ)*U*inputXZ)+inputPhases, 2))) < 0.001
    return isSymplec, condIsStat
end

function pauliToMat(pauli::PauliOp, system_size, subsystem_size, outSparse=false)
    splitHalf(x) = [x[1:Int(length(x)/2)], x[Int(length(x)/2)+1:end]]
    pXZ = splitHalf(pauli.XZencoding)
    pPhase = pauli.phase
    if system_size ==1 && outSparse
        return omegaPhaseTilde^pPhase * (sparse(genPauliMat["Xd"]^(pXZ[1][1]))*sparse(genPauliMat["Zd"]^(pXZ[2][1])))
    elseif system_size == 1
        return omegaPhaseTilde^pPhase * (genPauliMat["Xd"]^(pXZ[1][1])*genPauliMat["Zd"]^(pXZ[2][1]))    
    elseif outSparse
        return omegaPhaseTilde^pPhase * kron([sparse(genPauliMat["Xd"]^(pXZ[1][q]))*sparse(genPauliMat["Zd"]^(pXZ[2][q])) for q in 1:system_size]...)
    else
        return omegaPhaseTilde^pPhase * kron([genPauliMat["Xd"]^(pXZ[1][q])*genPauliMat["Zd"]^(pXZ[2][q]) for q in 1:system_size]...)
    end
end

function simpleCliffordToMat(Cliff::Tab, system_size, subsystem_dim)
    totalMat = Matrix(1LinearAlgebra.I, subsystem_dim^system_size, subsystem_dim^system_size)
    Id = Matrix(1LinearAlgebra.I, subsystem_dim, subsystem_dim)
    for cmd in Cliff.cmds
        cmd_weight = length(cmd[2])
        function getCNQuditGateMat(Activeq, system_size,subsystem_dim)
            function eyeMat(ind,size)
                tempvec = zeros((size,size))
                tempvec[ind,ind] = 1
                return tempvec
            end
            Xd = genPauliMat["Xd"]
            c_qubit = Activeq[1] #Controlqudit
            t_qubit = Activeq[2]
            completeGate =sum([kron([if qi==c_qubit eyeMat(i,subsystem_dim) elseif qi==t_qubit Id*Xd^(i-1) else Id end for qi in 1:system_size]...) for i in 1:subsystem_dim])
            return completeGate
        
        end

        if cmd[1]=="CN" && cmd_weight==2
            totalMat = getCNQuditGateMat(cmd[2], system_size,subsystem_dim)*totalMat  
        elseif cmd_weight == 1 && system_size==1
            t_qudit = 1
            totalMat = (genCliffordMat[cmd[1]])*totalMat
        elseif cmd_weight == 1
            t_qudit = cmd[2][1]
            totalMat = kron([if qi==t_qudit genCliffordMat[cmd[1]] else Id end for qi in 1:system_size]...)*totalMat
         else
            println("What else?")
            exit()
        end
    end

    return totalMat 
end


function initialize(system_size, subsystem_dim)
    """
    Returns (optional) key components of QfeatQ framework for use in non-global scope
    Initializes QfeatQ framework by initializing key components in the global scope or on return

    Args
    - ...
    """
    println("Initializing framework...")
    #Initials general qudit (generalized) Clifford & Pauli concepts:
    omegaPhase = exp(im*2*pi / subsystem_dim) #Overall phase in e.g. commutation of generalized paulis
    omegaPhaseTilde = if iseven(subsystem_dim) sqrt(omegaPhase) else omegaPhase end #Phase for Generalized Pauli Group

    genPauliMat = Dict( "Id"=>convert(Array{ComplexF64}, round.(vcat([transpose([if row==col 1 else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10)),
                        "Xd"=>convert(Array{ComplexF64},round.(vcat([transpose([if mod(row,subsystem_dim)==mod((col+1),subsystem_dim) 1 else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10)),
                        "Zd"=>convert(Array{ComplexF64},round.(vcat([transpose([if mod(row,subsystem_dim)==mod(col,subsystem_dim) omegaPhase^(row-1) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10))) 
    genPauliTab = Dict(bP => PauliOp(system_size, subsystem_dim, digits(bP, base=subsystem_dim, pad=2), 0) for bP in 0:subsystem_dim^2-1) 
    genPauliTraces = Dict(join(["Xd",xi,"Zd",zi]) => tr((genPauliMat["Xd"]^xi) * (genPauliMat["Zd"]^zi)) for xi in 0:subsystem_dim-1 for zi in 0:subsystem_dim-1) 
    #@show genPauliTraces
    
    genCliffordMat = Dict("CN" => vcat([transpose([if (digits(row-1, base=subsystem_dim, pad=2)[2]==digits(col-1, base=subsystem_dim, pad=2)[2] && digits(row-1, base=subsystem_dim, pad=2)[1]== mod(sum(digits(col-1, base=subsystem_dim, pad=2)),subsystem_dim)) 1 else 0 end for col in 1:subsystem_dim^2]) for row in 1:subsystem_dim^2]...),
                        #"CZ" => vcat([transpose([if row==col omegaPhase^(prod(digits(row-1, base=d, pad=2))) else 0 end for col in 1:d^2]) for row in 1:d^2]...),   #2-qudit gate
                        "F"=> 1/sqrt(subsystem_dim)* vcat([transpose([omegaPhase^((col-1)*(row-1)) for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...),
                        "P"=> vcat([transpose([if row==col omegaPhaseTilde^((row-1)*((row-1) +subsystem_dim)) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), 
                        "Z"=> vcat([transpose([if mod(row,subsystem_dim)==mod(col,subsystem_dim) omegaPhase^(row-1) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...))
    C_CN, p_CN = [1 0 0 0; 1 1 0 0; 0 0 1 -1; 0 0 0 1], zeros(Int,4)
    C_F, p_F = [0 -1; 1 0], zeros(Int,2)
    C_P, p_P = [1 0; 1 1], [subsystem_dim+1,0]
    C_Z, p_Z = [1 0; 0 1], [-2,0]

    @assert all(checkConditions(C_CN, p_CN,subsystem_dim))
    @assert all(checkConditions(C_F, p_F,subsystem_dim))
    @assert all(checkConditions(C_P, p_P,subsystem_dim))
    @assert all(checkConditions(C_Z, p_Z,subsystem_dim))

    genCliffordTab = Dict("CN" => Tab(2, subsystem_dim, [["CN",[1,2]]], mod.(C_CN, subsystem_dim), mod.(p_CN, 2*subsystem_dim), [["+1","ZZ"]]),
                        #"CZ" => ??u +
                        "F"=> Tab(1, subsystem_dim, [["F",[1]]], mod.(C_F, subsystem_dim), mod.(p_F, 2*subsystem_dim), [["+1","X"]]),
                        "P"=> Tab(1, subsystem_dim, [["P",[1]]], mod.(C_P, subsystem_dim), mod.(p_P, 2*subsystem_dim), [["+1","Z"]]), 
                        "Z"=> Tab(1, subsystem_dim, [["Z",[1]]], mod.(C_Z, subsystem_dim), mod.(p_Z, 2*subsystem_dim), [["+1","Z"]])  ) #C=I mod d, h=(-2,0) mod 2d 
    genCliffordOrder = Dict("CN" => subsystem_dim,
                        "F"=> if subsystem_dim != 2 4 else 2 end,
                        "P"=> if isodd(subsystem_dim) subsystem_dim else 4*subsystem_dim end, 
                        "Z"=> subsystem_dim)  #Stores order of gates for inversiion or removing redundancy 
                        
    
    U = sparse([zeros((system_size, system_size)) zeros((system_size, system_size)); Matrix(1I, system_size, system_size) zeros((system_size,system_size))])
    P = mod.(U - transpose(U), subsystem_dim)
    #@assert all(checkConditions(genCliffordTab["CN"],subsystem_dim))
    

    #Globalize:
    global genCliffordTab
    global genCliffordMat
    global genPauliMat
    global genPauliTraces
    global genCliffordOrder
    global omegaPhaseTilde
    global omegaPhase
    global U
    global P

    genPauliRotations = nothing
    if isfile("rot-$subsystem_dim.jld")
        genPauliRotations = load("rot-$subsystem_dim.jld")["rot-$subsystem_dim"]
    else
        println("Obtaining Pauli Rotations...")
        genPauliRotations = getPauliRotationDic(subsystem_dim)
    end
    global genPauliRotations

    eigZ = eigen(genPauliMat["Zd"])
    orderedZeigVals = zeros(ComplexF64, length(eigZ.values))
    for i in 1:length(eigZ.values)
        orderedZeigVals[findfirst(x->x==1.0,eigZ.vectors[:,i])] = eigZ.values[i]
    end
    eigZ = nothing
    sortEig(x)= if findfirst(y->y==round(x, digits=8) ,round.(orderedZeigVals, digits=8)) == nothing argmin(abs.(orderedZeigVals .- x)) else findfirst(y->y==round(x, digits=8) ,round.(orderedZeigVals, digits=8)) end
    global orderedPeigVals = Dict([xp, zp] => eigvals(genPauliMat["Xd"]^xp * genPauliMat["Zd"]^zp, sortby=sortEig) for xp in 0:subsystem_dim-1 for zp in 0:subsystem_dim-1)

    return omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genPauliTraces, genCliffordMat, genCliffordTab,genCliffordOrder, U,P
end

function applyCliffLocally(LocCliff, globalInputTab, system_size, subsystem_dim)
    #Local, unentangled version! Don't use for entangled/correlated Clifford
    globalOutputTab = deepcopy(globalInputTab)
    
    for cmd in LocCliff.cmds
        push!(globalOutputTab.cmds, cmd)
    end

    CliffCopy = deepcopy(LocCliff)
    for cmd in CliffCopy.cmds
        involvedQudits = cmd[2]
        involvedNumQudits = length(involvedQudits)
        involvedSlice = vcat(involvedQudits, involvedQudits .+ system_size)

        LocCliff = deepcopy(genCliffordTab[cmd[1]])
        LocCliff.cmds[1][2] = involvedQudits

        globalOutputTab.XZencoding[involvedSlice, involvedSlice] = mod.((LocCliff.XZencoding*globalInputTab.XZencoding[involvedSlice, involvedSlice]), subsystem_dim)           
        
        U = [zeros((involvedNumQudits, involvedNumQudits)) zeros((involvedNumQudits, involvedNumQudits)); Matrix(1I, involvedNumQudits, involvedNumQudits) zeros((involvedNumQudits, involvedNumQudits))]
        Cp_TUCp = transpose(LocCliff.XZencoding)*U*LocCliff.XZencoding
        globalOutputTab.phases[involvedSlice] = mod.((globalInputTab.phases[involvedSlice] + transpose(globalInputTab.XZencoding[involvedSlice, involvedSlice])*LocCliff.phases + Vdiag(transpose(globalInputTab.XZencoding[involvedSlice, involvedSlice])*(2*Pupps(Cp_TUCp)+Pdiag(Cp_TUCp))*globalInputTab.XZencoding[involvedSlice, involvedSlice]) - transpose(globalInputTab.XZencoding[involvedSlice, involvedSlice])*Vdiag(Cp_TUCp)) , (2*subsystem_dim))
    end

    return deepcopy(globalOutputTab)
end

function applyClifftoCliff(Cliff, InputTab, system_size, subsystem_dim)
    OutputTab = deepcopy(InputTab)
    
    #Update command list:

    for cmd in Cliff.cmds
        push!(OutputTab.cmds, cmd)
    end

    if size(Cliff.XZencoding) == size(InputTab.XZencoding)
        #Update XZencoding and phases
        OutputTab.XZencoding = mod.((Cliff.XZencoding*InputTab.XZencoding), subsystem_dim)
        
        U = [zeros((system_size, system_size)) zeros((system_size, system_size)); Matrix(1I, system_size,system_size) zeros((system_size, system_size))]
        Cp_TUCp = transpose(Cliff.XZencoding)*U*Cliff.XZencoding
        OutputTab.phases = mod.((InputTab.phases + transpose(InputTab.XZencoding)*Cliff.phases + Vdiag(transpose(InputTab.XZencoding)*(2*Pupps(Cp_TUCp)+Pdiag(Cp_TUCp))*InputTab.XZencoding) - transpose(InputTab.XZencoding)*Vdiag(Cp_TUCp)) , (2*subsystem_dim))
    elseif size(Cliff.XZencoding)[1] < size(InputTab.XZencoding)[1]
        
        CliffCopy = deepcopy(Cliff)
        for cmd in CliffCopy.cmds
            vacuumTab = Tab(system_size, subsystem_dim)

            globalizedCliff = applyCliffLocally(Cliff, vacuumTab, system_size, subsystem_dim)
            
            #Update XZencoding and phases of input globally:
            OutputTab.XZencoding = mod.((globalizedCliff.XZencoding*InputTab.XZencoding), subsystem_dim)
            
            U = [zeros((system_size, system_size)) zeros((system_size, system_size)); Matrix(1I, system_size,system_size) zeros((system_size, system_size))]
            Cp_TUCp = transpose(globalizedCliff.XZencoding)*U*globalizedCliff.XZencoding
            OutputTab.phases = mod.((InputTab.phases + transpose(InputTab.XZencoding)*globalizedCliff.phases + Vdiag(transpose(InputTab.XZencoding)*(2*Pupps(Cp_TUCp)+Pdiag(Cp_TUCp))*InputTab.XZencoding) - transpose(InputTab.XZencoding)*Vdiag(Cp_TUCp)) , (2*subsystem_dim))

        end
    
    else
        println("You should check your dimensions!")
        exit()
    end
    return deepcopy(OutputTab)
end

function conjugateClifftoPauli(Cliff::Tab, InputPauli::PauliOp, system_size, subsystem_dim)
    #Update XZ and phase encoding for pauli:
    U = [zeros((system_size, system_size)) zeros((system_size, system_size));Matrix(1I, system_size, system_size) zeros((system_size, system_size))]
    C_TUC = transpose(Cliff.XZencoding)*U*Cliff.XZencoding
    updatedEncoding = mod.((Cliff.XZencoding * InputPauli.XZencoding) , subsystem_dim)
    updatedPhase = mod.((InputPauli.phase + transpose(Cliff.phases - Vdiag(C_TUC))*InputPauli.XZencoding + transpose(InputPauli.XZencoding)*((2*Pupps(C_TUC)+ Pdiag(C_TUC))*InputPauli.XZencoding)) , (2*subsystem_dim))
    
    return PauliOp(system_size, subsystem_dim, updatedEncoding, updatedPhase)
end


function invertCliffTab(Cliff::Tab,system_size, subsystem_dim)
    Cliff = deepcopy(Cliff)

    inverted_cmds = []
    for cmd in reverse(Cliff.cmds)
        if cmd[1] == "P" 
            if isodd(subsystem_dim) #See paper for difference in phase / order
                numRep = subsystem_dim -1
            elseif iseven(subsystem_dim)
                numRep = 4*subsystem_dim - 1
            end
        elseif cmd[1] == "F"
            numRep = 4-1 #Look up why order 4, seems to work numerically!
        elseif cmd[1] == "CN" || cmd[1]== "Z"
            numRep = subsystem_dim-1 #CN shifts second bit # places further accord. to first bit
        else
            println("What gate should that be?")
            exit()
        end

        for rep in 1:numRep
            push!(inverted_cmds, cmd)
        end
    end

    #tempInverse = inv(Cliff.XZencoding)
    U = [zeros((system_size, system_size)) zeros((system_size, system_size)); Matrix(1I, system_size, system_size) zeros((system_size, system_size))]
    P = mod.(U - transpose(U), subsystem_dim)
    tempInverse = mod.(-P*transpose(Cliff.XZencoding)*P, subsystem_dim)
^
    Cinv_TUCinv = transpose(tempInverse)*(U*tempInverse)
    #@show tempInverse
    invertedEncoding = mod.(tempInverse , subsystem_dim)
    invertedPhase = mod.(-transpose(tempInverse)*(Cliff.phases + Vdiag(transpose(Cliff.XZencoding)*(2*Pupps(Cinv_TUCinv) + Pdiag(Cinv_TUCinv))*Cliff.XZencoding)- transpose(Cliff.XZencoding)*Vdiag(Cinv_TUCinv)) , (2*subsystem_dim))
    #@show system_size, subsystem_dim, inverted_cmds, invertedEncoding, invertedPhase
    return Tab(system_size, subsystem_dim, inverted_cmds, invertedEncoding, invertedPhase)
end


function multiplyPauliOps(Pop1::PauliOp, Pop2::PauliOp, num_qudits, d)
    #For function g in Aaronson/Gottesman paper -> see Hostens eq. 4 = phase by Pauli Multiplication
    phaseOut = nothing
    if Pop1.num_qudits == 1
        U_loc = [0 0; 1 0]
        phaseOut = mod(Pop1.phase + Pop2.phase + 2*transpose(Pop1.XZencoding)*U_loc*Pop2.XZencoding, 2*d)
    else
        phaseOut = mod(Pop1.phase + Pop2.phase + 2*transpose(Pop1.XZencoding)*U*Pop2.XZencoding, 2*d)
    end

    XZOut = mod.(Pop1.XZencoding .+ Pop2.XZencoding, d)
    return PauliOp(num_qudits, d, XZOut, phaseOut) 
end

function multiplyPauliOpsMod(Pop1::PauliOp, Pop2::PauliOp, num_qudits, d)
    #For function g in Aaronson/Gottesman paper -> see Hostens eq. 4 = phase by Pauli Multiplication
    
    phaseOut = mod(2*Pop1.phase + 2*Pop2.phase + sum(multiplyPauliOps(PauliOp(1,d,Pop1.XZencoding[[j, j+num_qudits]],0), PauliOp(1,d,Pop2.XZencoding[[j, j+num_qudits]],0),1, d).phase for j in 1:num_qudits), 2*d) 
    XZOut = mod.(Pop1.XZencoding .+ Pop2.XZencoding, d)
    return PauliOp(num_qudits, d, XZOut, phaseOut) 
end


function getPauliRotationDic(d)
    genPauliRotations = Dict()
    for xpow in 0:d-1
        for zpow in 0:d-1
            c_pauli = PauliOp(1, d, [xpow, zpow],0)
            #@show c_pauli

            #If the operator is the identity or just any power of Z, the answer is simple and rigid!
            if xpow == 0
                totCTab = Tab(1,d)
                extraPhase = 0

                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                if conCP.XZencoding == [0,1] || zpow!=1
                    push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                    continue
                else
                    #@show conCP
                    @assert false
                end
            end

            #First we try it the most efficient brute force way we know to keep circuits and runtime as small as possible
            for cc in unique(Iterators.product(fill(["I","Z", "P", "F"],6)...))
                p = collect(cc)

                totCTab = Tab(1,d)
                for el in p
                    if el == "I"
                        continue
                    else
                        ct_temp = deepcopy(genCliffordTab[el])
                        totCTab = applyClifftoCliff(ct_temp, totCTab,1,d)
                    end
                end
  
                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                extraPhase = (conCP.phase - c_pauli.phase)

                if conCP.XZencoding == [0,1] && !haskey(genPauliRotations, join(["X", xpow,"Z",zpow]))

                    push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                elseif conCP.XZencoding == [0,1] && haskey(genPauliRotations, join(["X", xpow,"Z",zpow])) && extraPhase==0 && 
                    if genPauliRotations[join(["X", xpow,"Z",zpow])][1] != 0
                        push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                    end
                end

            end

            #If no limited brute force methods help, apply the emergency case for extensively long circuits:
            if haskey(genPauliRotations, join(["X", xpow,"Z",zpow]))
                continue
            elseif xpow < 2 && zpow < 2
                @assert false
            else
                totCTab = Tab(1,d)
                #Look for any equivalence in the same 
                found_zp = nothing
                for zp in 0:d-1
                    #Exists there an equal powered X already? If not, there is not nothing we can do ;(
                    if haskey(genPauliRotations, join(["X", xpow,"Z",zp]))
                        found_zp = zp
                    end
                end
                if found_zp == nothing
                    continue
                end
                totCTab = genPauliRotations[join(["X", xpow,"Z",found_zp])][2]
                
                #Else:
                extraPhase = (conCP.phase - c_pauli.phase) #1 = 01 = Zd
                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                #push!(genPauliRotations, join(["X", xpow,"Z",0]) => [extraPhase , totCTab])
            end
        end
    end

    save("rot-$d.jld", "rot-$d", genPauliRotations)
    #reload with: load("rot-$d.jld")["rot-$d"]
    return genPauliRotations
end



function readObservables(observableFile)
    """
    Assumes local pauli observables and weights are always assumed!
    """
    observableContent = readlines(observableFile)
    system_size = parse(Int, split(observableContent[1], " ")[1])
    subsystem_dim = parse(Int, split(observableContent[1], " ")[2])

    all_observables = []
    weightArray = []

    for line in observableContent[2:end]
        one_observable = []

        wBool = 1 #Are there weights?
        XZencoding = zeros(Int,2*system_size)

        for q_obs in split(line, " ")[4:end]
      
            pos = parse(Int, split(q_obs, "|")[1])
            xzpow = split(q_obs, "|")[2]
            xpow, zpow = xzpow[2], xzpow[4]

            XZencoding[pos] = mod(parse(Int,xpow), subsystem_dim)
            XZencoding[pos+system_size] = mod(parse(Int,zpow), subsystem_dim)
        end

        phase = mod(Int(parse(Float64, split(line, " ")[3])), 2*subsystem_dim)

        if wBool == 1
            push!(weightArray, parse(Float64, split(line, " ")[1])+im*parse(Float64, split(line, " ")[2])  )
        end

        one_observable = PauliOp(system_size, subsystem_dim,XZencoding, phase)
        push!(all_observables, one_observable)
    end
    return all_observables, weightArray, system_size
end


function observable_generation_Controller(kindOfObservable, system_size, subsystem_size, observableFile, num_of_observables=nothing)
    if kindOfObservable == "r" #Random observables
        maxWeight = 3
        generateCompleteRandomObservables(num_of_observables, system_size, subsystem_dim, observableFile, maxWeight)
    elseif kindOfObservable == "ew" #Dicke optimal entanglement witness
        generateEW
    end
end

    function state_generation_Controller(kindOfState, system_size, subsystem_dim, stateFile)
    #Currently only allowing scalar = equal subsystem dimensions = Default: qubits: 2
    #systemsizenum of qudits
    system_dim = subsystem_dim^system_size

    if kindOfState == "rp" #random pure state
        receivedState = randomPureState(system_dim)
        
        #Store the state:
        h5open(stateFile, "w") do fid
            fid[stateFile]= receivedState
        end
    elseif kindOfState == "rm" #random mixed state
        randomMixedState(subsystem_dim, system_dim)

        #Store the state:
        h5open(stateFile, "w") do fid
            fid[stateFile]= receivedState
        end
    elseif kindOfState == "ghz" #random mixed state
        receivedState = GHZ(system_size, subsystem_dim)

        #Store the state:
        h5open(stateFile, "w") do fid
            fid[stateFile]= receivedState
        end

    elseif kindOfState == "rp-stab" #random, pure stabilizer
        receivedTableauState = naiveRandomTabState(system_size, subsystem_dim)
        #Store the state:
        tableauStateMemory = receivedTableauState
        return tableauStateMemory
    
    elseif kindOfState == "ghz+_stab"
        GHZtab = generateGHZTab(system_size, subsystem_dim, "+")
        return GHZtab
    elseif kindOfState == "ghz-_stab"
        tableauImpurityMemory = generateGHZTab(system_size,subsystem_dim, "-")
        return tableauImpurityMemory

    elseif kindOfState == "dicke" #random mixed state
        numExcit = 2
        receivedState = generateSimpleDickeState(system_size, subsystem_dim,numExcit)
        @assert round(tr(receivedState),digits=4) == 1.0
        
        #Store the state:
        h5open(stateFile, "w") do fid
            fid[stateFile]= receivedState
        end
    elseif kindOfState == "dicke-gen" #random mixed state
        numExcit = 2
        receivedState = generateGenSimpleDickeState(system_size, subsystem_dim,numExcit) 
        @assert round(tr(receivedState),digits=4) == 1.0
        #Store the state:
        h5open(stateFile, "w") do fid
            fid[stateFile]= receivedState
        end
    elseif kindOfState == "pdc-oam" #random mixed state

        l_cutoff = Int((subsystem_dim-1)/2)
        receivedState =generatePDC_OAMstate(l_cutoff, system_size, subsystem_dim,[fill(1/sqrt(l_cutoff),l_cutoff),fill(1/sqrt(l_cutoff*l_cutoff), (l_cutoff, l_cutoff))])
        @assert round(tr(receivedState),digits=4) == 1.0
        #Store the state:
        h5open(stateFile, "w") do fid
            fid[stateFile]= receivedState
        end
    end
end

function create_data_acquisition_scheme_Controller(kindOfScheme, number_of_measurements, system_size, subsystem_dim ,observableFile,onlyLocalOps=false)
    if kindOfScheme == "d" #derandomized_classical_shadow
        all_observables, weightArray, system_size = readObservables(observableFile)

        if length(weightArray) == 0
            measurement_procedure = derandomized_classical_shadow(all_observables, meas_repetitions_per_observable, system_size)
        
        elseif length(weightArray) == length(all_observables)
            println("We detected weight inputs!")
            measurement_procedure = derandomized_classical_shadow(all_observables, meas_repetitions_per_observable, system_size, weightArray)
        
        else
            println(stderr,"Check your weights!")
            print_usage()
            measurement_procedure = []
        end

        #Store the resulting measurement_procedure
        open(measurementSchemeFile, "w") do outIO
            write(outIO, string(system_size))
            for single_round_measurement in measurement_procedure
                println(join(single_round_measurement, " "))
                write(outIO, string("\n", join(single_round_measurement, " ")))
            end
        end
    
    elseif kindOfScheme == "r" #Randomized classical shadow
        measurement_procedure = randomized_classical_shadow(total_number_of_measurements, system_size)
    
        #Store the resulting measurement_procedure

        open(measurementSchemeFile, "w") do outIO
            write(outIO, string(system_size))
            for single_round_measurement in measurement_procedure
                println(join(single_round_measurement, " "))
                write(outIO, string("\n", join(single_round_measurement, " ")))
            end
        end

    elseif kindOfScheme == "rc" #Random CLifford rotations/measurements
        clifford_measurement_scheme = []

        @showprogress "Sampling Measurement Scheme..." for mi in 1:number_of_measurements
            push!(clifford_measurement_scheme, generateSimpleRandomClifford(system_size, subsystem_dim))
        end

        return clifford_measurement_scheme

    elseif kindOfScheme == "rlc" #Random local Clifford rotations/measurements
        clifford_measurement_scheme = []
        @showprogress "Sampling Measurement Scheme..." for mi in 1:number_of_measurements
            push!(clifford_measurement_scheme, [naiveRandomTabClifford(1, subsystem_dim) for qi in 1:system_size])
        end
        return clifford_measurement_scheme 

    elseif kindOfScheme == "rlp" #Random local Pauli-Clifford rotations/measurements
        measurement_procedure = randomized_local_pauliMeasurements(number_of_measurements, system_size, subsystem_dim)
        return measurement_procedure
    
    elseif kindOfScheme == "dlp" #Random local Pauli-Clifford rotations/measurements
        all_observables, weightArray, system_size = readObservables(observableFile)
    
        measurement_procedure = derandomized_local_pauliMeasurements(all_observables, number_of_measurements, system_size, subsystem_dim, weightArray)
        
        return measurement_procedure

    elseif kindOfScheme == "lbs" #Locally Biased shadows
        observableArray, weightArray, system_size = readObservables(observableFile)

        lbs_measurement_scheme = []
        if length(weightArray) == length(all_observables)
            lbs_measurement_scheme = LocBiasedShadow(total_number_of_measurements, system_size, observableArray, weightArray)
        else
            lbs_measurement_scheme = LocBiasedShadow(total_number_of_measurements, system_size, observableArray)
        end
        
        #Store the resulting measurement_procedure
        open(measurementSchemeFile, "w") do outIO
            write(outIO, string(system_size))
            for single_round_measurement in measurement_procedure
                println(join(single_round_measurement, " "))
                write(outIO, string("\n", join(single_round_measurement, " ")))
            end
        end
    end 
end
function measurement_Controller(kindOfMeasurement,compOut, system_size,subsystem_dim, cliffMeasScheme, state)

    if kindOfMeasurement == "pauli"
        subsystem_dim = 2

        stateArray = h5open(stateFile, "r") do fid
            this = fid[stateFile]
            #@show this
            return read(this)
        end
        
        system_size, measurement_procedure = open(measurementSchemeFile, "r") do inIO
            measurements = readlines(inIO)
            measurement_procedure = []
            for line in measurements[2:end]
                single_measurement = []
                for pauli_XYZ in split(line, " ")
                    push!(single_measurement, pauli_XYZ)
                end
                push!(measurement_procedure, single_measurement)
            end
            
            return parse(Int, measurements[1]), measurement_procedure
        end

        @assert Int(log(size(stateArray)[1])/log(subsystem_dim)) == system_size

        open(measurementOutcomeFile, "w") do outIO #Store outcomes along the way
            write(outIO, string(system_size))
            
            num_of_measurements = length(measurement_procedure)
            for (m,measurement) in enumerate(measurement_procedure)
                outcomeString = measurePauli(measurement, stateArray)

                printString = join([val for pair in zip(measurement, outcomeString) for val in pair], " ")
                println(printString)
                write(outIO, string("\n", printString))
                @show "Finished measurement $m of $num_of_measurements"
            end
        end

    elseif kindOfMeasurement == "clifford"
        tableauStateMemory = state

        #initialize the earlier produced measurement scheme and state
        cliffordOutcomes = []
        total_num_of_measurements = length(cliffMeasScheme)
        
        #@show clifford_measurement_scheme
        @showprogress "Measuring..." for (i, cliff) in enumerate(cliffMeasScheme)
            tabState = deepcopy(tableauStateMemory)
            push!(cliffordOutcomes,measureGlobalCliffordDirect(tabState, deepcopy(cliff), system_size,subsystem_dim,true))
            
        end
        return cliffordOutcomes

    elseif kindOfMeasurement == "clifford-loc"
        tableauStateMemory = state
        #initialize the earlier produced measurement scheme and state
        cliffordOutcomes = []
        total_num_of_measurements = length(cliffMeasScheme)
     
        @showprogress "Measuring..." for (i, cliff) in enumerate(cliffMeasScheme)
            tabState = deepcopy(tableauStateMemory)
            push!(cliffordOutcomes,measureLocalCliffordDirect(tabState, deepcopy(cliff), system_size,subsystem_dim,true))
        end
        return cliffordOutcomes

    elseif kindOfMeasurement == "clifford-mat"
        stateFile = state
        #initialize the earlier produced measurement scheme and state
        cliffordOutcomes = []
        total_num_of_measurements = length(cliffMeasScheme)
        
        stateArray = h5open(stateFile, "r") do fid
            this = fid[stateFile]
            return sparse(read(this))
        end

        @showprogress "Measuring..." 1 for (i, cliff) in enumerate(cliffMeasScheme)
            push!(cliffordOutcomes, measureGlobalCliffordMat(stateArray, cliff, system_size, subsystem_dim, true))
        end 

        return cliffordOutcomes 
    elseif kindOfMeasurement == "clifford-loc-mat"
        pauliMeasScheme = cliffMeasScheme #Input is in this actually a pauli input
        stateFile = state
        #initialize the earlier produced measurement scheme and state
        cliffordOutcomes = []
        total_num_of_measurements = length(pauliMeasScheme)
        
        stateArray = h5open(stateFile, "r") do fid
            this = fid[stateFile]

            return read(this)
        end

        @showprogress "Measuring..." for (i, pauli) in enumerate(pauliMeasScheme)

            push!(cliffordOutcomes, measureLocalCliffordMat(stateArray, pauli, system_size, subsystem_dim, true))
        end 

        return cliffordOutcomes
    elseif kindOfMeasurement == "clifford-eff-global"
        tableauStateMemory = state

        #initialize the earlier produced measurement scheme and state
        cliffordOutcomes = []
        total_num_of_measurements = length(cliffMeasScheme)

        @showprogress "Measuring..." for (i, cliff) in enumerate(cliffMeasScheme)
            stateIdx = rand(Categorical(tableauStateMemory[2]))
            tabState = deepcopy(tableauStateMemory[1][stateIdx])

            push!(cliffordOutcomes,measureCliffordEff(tabState, deepcopy(cliff), system_size,subsystem_dim,"g",true))
        end
        return cliffordOutcomes
    elseif kindOfMeasurement == "clifford-eff-local"
        tableauStateMemory = state

        #initialize the earlier produced measurement scheme and state
        cliffordOutcomes = []
        total_num_of_measurements = length(cliffMeasScheme)
        pauliMeasScheme = cliffMeasScheme

        @showprogress "Measuring..." for (i, pauli) in enumerate(pauliMeasScheme)
            tabState = deepcopy(tableauStateMemory)

            push!(cliffordOutcomes,measureCliffordEff(tabState,deepcopy(pauli) , system_size,subsystem_dim,"l",true))
        end
        return cliffordOutcomes
    end
end


function observable_prediction_Controller(kindOfMethod, kindOfPrediction, system_size, subsystem_dim, observableFile, cliffordOutcomes=nothing, cliffMeasScheme=nothing, inputState=nothing)
    outcomeArray = cliffordOutcomes
    if kindOfMethod == "truthPauli" #Get from precompiled state true expectation values
        if kindOfPrediction == "o" #Local observables
            #Initialize and read in True state:
            stateArray = nothing
            stateMode = "v" #vector

            if typeof(inputState) == String 
                stateArray = h5open(inputState, "r") do fid
                    this = fid[inputState]
                    return sparse(read(this))
                end
                stateMode = "m" #matrix
            elseif typeof(inputState) == Tab #State in inputState = prefer = in tableau format
                stateArray = QfeatQ.simpleCliffordToMat(deepcopy(inputState), system_size, subsystem_dim)[:,1]
            end
            #Initialize and read in observables:
            all_observables, weightArray ,system_size = readObservables(observableFile)

            trueExpectations = []

            @showprogress "Predicting truth of observables..." for (obs_i, one_observable) in enumerate(all_observables)
                #Prepare full observable array for prediction
                full_tensor_observable = sparse(pauliToMat(one_observable, system_size, subsystem_dim))
                
                #Calculate expectation for this observable:
                if stateMode == "m"
                    push!(trueExpectations, real(sum(diag(stateArray*full_tensor_observable))))
                    
                elseif stateMode == "v"
                    push!(trueExpectations, tr(full_tensor_observable*(stateArray*adjoint(stateArray))))
                end
            end
            return trueExpectations
        elseif kindOfPrediction == "o-eff"
            #Supports only stabilizer Tableaus as input
            all_observables, weightArray ,system_size = readObservables(observableFile)

            trueExpectations = []
            @showprogress 1 "Predicting truth of observables..." for (obs_i, one_observable) in enumerate(all_observables)
        
                pauliPhase = one_observable.phase
                rotatedState = deepcopy(inputState)
                rotatedState.phases = rotatedState.phases .+ fill(pauliPhase, length(rotatedState.phases))

                @showprogress "Rotating state to computational basis..." for meas_qi in 1:system_size
                    xpow, zpow = one_observable.XZencoding[meas_qi], one_observable.XZencoding[system_size+meas_qi]  #.XZencoding[1]  .XZencoding[2]
                    
                    #extraPhase = omegaPhaseTilde^genPauliRotations[string("X", xpow, "Z", zpow)][1]
                    thisCliff = deepcopy(genPauliRotations[string("X", xpow, "Z", zpow)][2])
                    if length(thisCliff.cmds) != 0
                        for i in 1:length(thisCliff.cmds)
                            thisCliff.cmds[i][2] = [meas_qi]
                        end
                        
                        rotatedState = applyClifftoCliff(thisCliff,deepcopy(rotatedState), system_size ,subsystem_dim)
                    end
                end
                
                sample_size = 900
                outcomeAv = 0
                for mi in 1:sample_size
                    #outcomeInt = parse(Int,join(measureCliffordEff(deepcopy(inputState),  one_observable, system_size,subsystem_dim, "lg", true)), base=subsystem_dim)
                    outcomeString =  measureCliffordEff(deepcopy(rotatedState),  one_observable, system_size,subsystem_dim, "lg", true)
                    outcomeAv += prod([orderedPeigVals[one_observable.XZencoding[[oi,oi+system_size]]][o+1] for (oi,o) in enumerate(outcomeString)])
                end
                outcomeAv /= sample_size
                push!(trueExpectations, outcomeAv)
            end

            return trueExpectations

        elseif kindOfPrediction == "ew"
            trueExpArray = observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, nothing, "systemState.txt")
            all_observables, weightArray ,system_size = readObservables(observableFile)

            term1to5 = dot(weightArray[1:firstsqrtindex-1],trueExpArray[1:firstsqrtindex-1])
                
            wpe = dot(weightArray[firstsqrtindex:end],trueExpArray[firstsqrtindex:end])
            term6 = 0
            for idx in 1:Int(length(clusterlengths)/2)
                if idx == 1
                    term6 -= sqrt(sum(wpe[1:clusterlengths[1]])*sum(wpe[clusterlengths[1]:sum(clusterlengths[1:2])]))
                else 
                    term6 -= sqrt(sum(wpe[sum(clusterlengths[1:2*(idx-1)])+1:sum(clusterlengths[1:2*(idx-1) +1])])*sum(wpe[sum(clusterlengths[1:2*(idx-1) +1])+1:sum(clusterlengths[1:2*(idx)])]))
                end
            end
            ewTrueExpectation = term1to5 + term6
            return trueExpArray, ewTrueExpectation
        elseif kindOfPrediction=="f"
            @assert typeof(inputState) == Array{Tab,1} [stateCopy, pureStateMemory]
            
            inputVec = simpleCliffordToMat(inputState[1],system_size, subsystem_dim)[:,1]
            outputVec = simpleCliffordToMat(inputState[2],system_size, subsystem_dim)[:,1]
            return fidelityPrediction(inputVec*adjoint(inputVec),outputVec*adjoint(outputVec),"m")
        end
        
    elseif kindOfMethod == "cShadowPauli"
        measurementOutcomes = readlines(measurementOutcomeFile)
        system_size = parse(Int64, measurementOutcomes[1])
        
        full_measurement = []
        for line in measurementOutcomes[2:end]
            single_measurement = []
            for (pauli_XYZ, outcome) in zip(split(line, " ")[1:2:end], split(line, " ")[2:2:end])
                push!(single_measurement, (pauli_XYZ, parse(Int64, outcome)))
            end
            push!(full_measurement, single_measurement)
        end
        
        #Ask what to predict with classical shadows:
        if kindOfPrediction == "o" #Set of local observables

            all_observables = readObservables(observableFile)
            predictedExpectation = []

            @showprogress "Predicting Local Observables..." for (obs_i, one_observable) in enumerate(all_observables)
             
                sum_product, cnt_match = estimate_exp(full_measurement, one_observable)
                #@show cnt_match
                if cnt_match > 0
                    predictedExp = sum_product / cnt_match #Final expectation prediction: Sum of average outcome in each measurement -> normalize = average
                else
                    println("WARNING: Observable $obs_i not measured once, we'll expect 0 for it!")
                    predictedExp = 0
                end
                #println("For observable $obs_i : $predictedExp")
                push!(predictedExpectation, predictedExp)
            end
            return predictedExpectation
        elseif kindOfPrediction == "e"
            println("Sorry, not yet Implemented")
            exit()
        end
    elseif kindOfMethod=="cShadowLocClifford-direct"
        if kindOfPrediction == "lo"
            all_observables, weightArray, system_size = readObservables(observableFile)
            predictions = []
            @showprogress "Predicting Local Observables..." for (obs_i, obs) in enumerate(all_observables)
                push!(predictions, estimate_exp_direct(cliffMeasScheme, cliffordOutcomes, obs, system_size, subsystem_dim))
            end
            return predictions
        end
    elseif kindOfMethod == "cShadowLocClifford-mat"
        if kindOfPrediction == "lo" #local observables
            k=1
            return predictLocalCliffordLocalObservables(observableFile, cliffordMeasurementScheme, outcomeArray , system_size, 1)
        end
    elseif kindOfMethod == "cShadowClifford-stab"
        if kindOfPrediction == "lo" #Local observables
            k = 1 #Number of partitions to make for median of means
            all_observables,weightArray, system_size = readObservables(observableFile)

            predictedExpectation = []
            trueExpectation = []

            @showprogress "Predicting Local Observables..." for (obs_i, one_observable) in enumerate(all_observables)
                one_expectation = predictCliffordLocalObservables(one_observable, cliffordOutcomes, cliffMeasScheme, system_size, subsystem_dim, k, obs_i)
                #println("For observable $obs_i : $one_expectation")
                push!(predictedExpectation, one_expectation)
                #push!(trueExpectation, true_one_expectation)
            end
            return predictedExpectation

        elseif kindOfPrediction == "ew" #Entanglement Witness

            k = 1 #Number of partitions to make for median of means
            all_observables,weightArray, system_size = readObservables(observableFile)
    
            predictedExpectation = []
            trueExpectation = []
    
            @showprogress "Predicting Local Observables for Entanglement Witness..." for (obs_i, one_observable) in enumerate(all_observables)
                one_expectation = predictCliffordLocalObservables(one_observable, cliffordOutcomes, cliffMeasScheme, system_size, subsystem_dim, k, obs_i)
                #println("For observable $obs_i : $one_expectation")
                push!(predictedExpectation, one_expectation)
                #push!(trueExpectation, true_one_expectation)
            end
    
            term1to5 = dot(weightArray[1:firstsqrtindex-1],predictedExpectation[1:firstsqrtindex-1])
            
            wpe = dot(weightArray[firstsqrtindex:end],predictedExpectation[firstsqrtindex:end])
            term6 = 0
            for idx in 1:Int(length(clusterlengths)/2)
                if idx == 1
                    term6 -= sqrt(sum(wpe[1:clusterlengths[1]])*sum(wpe[clusterlengths[1]:sum(clusterlengths[1:2])]))
                else 
                    term6 -= sqrt(sum(wpe[sum(clusterlengths[1:2*(idx-1)])+1:sum(clusterlengths[1:2*(idx-1) +1])])*sum(wpe[sum(clusterlengths[1:2*(idx-1) +1])+1:sum(clusterlengths[1:2*(idx)])]))
                end
            end
            ewExpectation = term1to5 + term6
            return predictedExpectation, ewExpectation

        elseif kindOfPrediction == "f" #Fidelity to original state
            #Assumes input state simulated or given!: inputState
            @assert typeof(inputState) == Tab
            inputVec = sparse(simpleCliffordToMat(inputState, system_size, subsystem_dim))[:, 1]
            inputMat = inputVec*adjoint(inputVec)
            @assert size(inputMat) == (subsystem_dim^system_size, subsystem_dim^system_size)
            
            k=1
            reduced_shadow_fidelities = []
            partition_size = Int(length(cliffordOutcomes) / k)
            for p in 1:k
                tempSum = zeros(ComplexF64, (subsystem_dim^system_size, subsystem_dim^system_size))
                for out_i in collect((p-1)*partition_size+1:(p)*partition_size)                   
                    compBasisVec(k) = vcat(zeros(k-1),1.0, zeros(subsystem_dim-k))
                    compBasisOutcome = kron([compBasisVec(b+1) for b in cliffordOutcomes[out_i]]...)
                    @show compBasisOutcome, cliffordOutcomes[out_i]
                    #sExcitVec, sGroundVec = vcat(zeros(d-1),1), vcat(1,zeros(d-1))
                    
                    tempM = adjoint(simpleCliffordToMat(cliffMeasScheme[out_i], system_size, subsystem_dim))
                    tempVec = tempM*compBasisOutcome
                    #@show (((subsystem_dim^system_size +1) * (tempVec* adjoint(tempVec) )) - sparse(Matrix(LinearAlgebra.I, (subsystem_dim^system_size, subsystem_dim^system_size))))
                    tempSum = tempSum + (((subsystem_dim^system_size +1) * (tempVec* adjoint(tempVec) )) - sparse(Matrix(LinearAlgebra.I, (subsystem_dim^system_size, subsystem_dim^system_size))))
                end
                push!(reduced_shadow_fidelities, fidelityPrediction(collect(inputMat), collect(tempSum ./ partition_size), "m"))
            end
            return reduced_shadow_fidelities
        end
    elseif kindOfMethod == "cShadowLocClifford-stab"
        k = 1 #Number of partitions to make for median of means
        all_observables,weightArray, system_size = readObservables(observableFile)

        predictedExpectation = []
        trueExpectation = []

        @showprogress "Predicting Local Observables..." for (obs_i, one_observable) in enumerate(all_observables)
            one_expectation = predictLocCliffordLocalObservables(one_observable, cliffordOutcomes, cliffMeasScheme, system_size, subsystem_dim, k, obs_i)
            #println("For observable $obs_i : $one_expectation")
            push!(predictedExpectation, one_expectation)
            #push!(trueExpectation, true_one_expectation)
        end
        return predictedExpectation
    end
end

end

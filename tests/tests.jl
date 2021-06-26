using LinearAlgebra
using StatsBase
using SparseArrays
using Combinatorics
using HDF5, JLD
using Statistics
using Distributions


#=
function tab2Mat(inputTab, num_qudits,d)
    genPauliEncoding = Dict("Xd"=> 1, "Zd"=> 2) 
    inverseGenPauliEncoding = Dict(1=> "Xd", 2=>"Zd")
    totalMatrix = []
    for col in 1:num_qudits
        powerInd = inputTab.XZencoding[:, col]
        phaseInd = inputTab.phases[col]
        singleQuditMat = genPauliMat["Id"]
        for pauli in collect(keys(genPauliMat))
            if all(powerInd .== 0) || pauli=="Id"
                continue 
            else
                tempMat = ((omegaPhase^phaseInd) * genPauliMat[pauli]^round(powerInd[genPauliEncoding[pauli]],digits=10))^d
                @show pauli, col, tempMat
                singleQuditMat = singleQuditMat*tempMat
            end
        end
        if col == 1
            totalMatrix = singleQuditMat
        else
            kron(totalMatrix, singleQuditMat)
        end 
    end
    return totalMatrix
end
=#

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

function removeRedundancy(inputTab, localArrayInput::Bool)

    if localArrayInput
        outputTab = []
        for locCTab in inputTab
            outputLocTab = deepcopy(locCTab)
            seenCmds = []
            for cmd in outputLocTab.cmds
                if cmd in seenCmds
                    continue
                else
                    kernelSize = genCliffordOrder[cmd[1]]
                    boolMask = fill(false, length(outputLocTab.cmds))
                    indices = findall(x->x==cmd, outputLocTab.cmds)
                    if length(indices) >= kernelSize
                        numPartitions = Int(floor(length(indices)/kernelSize))-1
                        #@show indices, kernelSize, numPartitions
                        partitions = [indices[i+1:i+kernelSize] for i in 0:numPartitions]
                        deleteat!(outputLocTab.cmds, sort(unique([vcat(partitions...)...])))
                        push!(seenCmds, cmd)
                    end
                end
            end

            push!(outputTab, outputLocTab)
        end
        return outputTab
    else
        outputTab = deepcopy(inputTab)
        seenCmds = []
        for cmd in inputTab.cmds
            if cmd in seenCmds
                continue
            else
                kernelSize = genCliffordOrder[cmd[1]]
                boolMask = fill(false, length(inputTab.cmds))
                for i in 1:(length(inputTab.cmds)-kernelSize)
                    if outputTab.cmds[i:i+kernelSize]==fill(cmd, kernelSize)
                        boolMask[i:i+kernelSize] .= true 
                    end
                end
                deleteat!(outputTab.cmds, boolMask)
                push!(seenCmds, cmd)
            end
        end
        return outputTab
    end
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
    #@show isSymplec, condIsStat, maximum(abs.(transpose(inputXZ)*(omega*inputXZ) - omega)), ((mod.((d-1)*Vdiag(transpose(inputXZ)*U*inputXZ)+inputPhases, 2)))
    #@show transpose(inputXZ)*(omega*inputXZ) - omega
    return isSymplec, condIsStat
end

function pauliToMat(pauli::PauliOp, system_size, subsystem_size)
    splitHalf(x) = [x[1:Int(length(x)/2)], x[Int(length(x)/2)+1:end]]
    pXZ = splitHalf(pauli.XZencoding)
    pPhase = pauli.phase    
    return omegaPhaseTilde^pPhase * kron([genPauliMat["Xd"]^(pXZ[1][q])*genPauliMat["Zd"]^(pXZ[2][q]) for q in 1:system_size]...)
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
         
            #@show size(blockdiag(sparse(Matrix(1LinearAlgebra.I,left_id_dim,left_id_dim)), sparse(genCliffordMat[cmd[1]]) ,sparse(Matrix(1LinearAlgebra.I,right_id_dim, right_id_dim ))))
        else
            println("What else?")
            exit()
        end
    end

    return totalMat 
end

function pauliToCliff(InputP::PauliOp)
    outCliff = Tab(InputP.num_qudits, InputP.d, [], Matrix(1LinearAlgebra.I, 2*InputP.num_qudits, 2*InputP.num_qudits), zeros(2*InputP.num_qudits))
    
    U = [zeros((InputP.num_qudits, InputP.num_qudits)) zeros((InputP.num_qudits,InputP.num_qudits)); Matrix(1I, InputP.num_qudits,InputP.num_qudits) zeros((InputP.num_qudits, InputP.num_qudits))]
    P = mod.(U - transpose(U), InputP.d)
    newPhases = mod(-2*P*InputP.XZencoding, 2*InputP.d)
    outCliff.phases = newPhases
    return outCliff
end

function initialize(system_size, subsystem_dim)
    #Initials general qudit (generalized) Clifford & Pauli concepts:
    omegaPhase = exp(im*2*pi / subsystem_dim) #Overall phase in e.g. commutation of generalized paulis
    omegaPhaseTilde = if iseven(subsystem_dim) sqrt(omegaPhase) else omegaPhase end #Phase for Generalized Pauli Group

    genPauliMat = Dict( "Id"=>convert(Array{ComplexF64}, round.(vcat([transpose([if row==col 1 else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10)),
                        "Xd"=>convert(Array{ComplexF64},round.(vcat([transpose([if mod(row,subsystem_dim)==mod((col+1),subsystem_dim) 1 else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10)),
                        "Zd"=>convert(Array{ComplexF64},round.(vcat([transpose([if mod(row,subsystem_dim)==mod(col,subsystem_dim) omegaPhase^(row-1) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10))) 
    genPauliTab = Dict(bP => PauliOp(system_size, subsystem_dim, reverse(digits(bP, base=subsystem_dim, pad=2)), 0) for bP in 0:subsystem_dim^2-1) 

    genCliffordMat = Dict("CN" => vcat([transpose([if (digits(row-1, base=subsystem_dim, pad=2)[2]==digits(col-1, base=subsystem_dim, pad=2)[2] && digits(row-1, base=subsystem_dim, pad=2)[1]== mod(sum(digits(col-1, base=subsystem_dim, pad=2)),subsystem_dim)) 1 else 0 end for col in 1:subsystem_dim^2]) for row in 1:subsystem_dim^2]...),
                        #"CZ" => vcat([transpose([if row==col omegaPhase^(prod(digits(row-1, base=d, pad=2))) else 0 end for col in 1:d^2]) for row in 1:d^2]...),   #2-qudit gate
                        "F"=> 1/sqrt(subsystem_dim)* vcat([transpose([omegaPhase^((col-1)*(row-1)) for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...),
                        "P"=> vcat([transpose([ if row==col omegaPhaseTilde^((row-1)*((row-1) +d)) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), 
                        "Z"=> vcat([transpose([if mod(row,subsystem_dim)==mod(col,subsystem_dim) omegaPhase^(row-1) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...),
                        "I"=> Matrix(1LinearAlgebra.I, subsystem_dim, subsystem_dim))
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
    
    #@assert all(checkConditions(genCliffordTab["CN"],subsystem_dim))
    genCliffordOrder = Dict("CN" => subsystem_dim,
    "F"=> if subsystem_dim != 2 4 else 2 end,
    "P"=> if isodd(subsystem_dim) subsystem_dim else 4*subsystem_dim end, 
    "Z"=> subsystem_dim)  #Stores order of gates for inversiion or removing redundancy 
    
    return omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab, genCliffordOrder
end



function applyCliffLocally(LocCliff::Tab, globalInputTab::Tab, system_size, subsystem_dim)
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
        @show involvedSlice
        LocCliff = deepcopy(genCliffordTab[cmd[1]])
        LocCliff.cmds[1][2] = involvedQudits

        globalOutputTab.XZencoding[involvedSlice, involvedSlice] = mod.((LocCliff.XZencoding*globalInputTab.XZencoding[involvedSlice, involvedSlice]), subsystem_dim)           
        
        #@show globalInputTab.phases[involvedSlice]

        U = [zeros((involvedNumQudits, involvedNumQudits)) zeros((involvedNumQudits, involvedNumQudits)); Matrix(1I, involvedNumQudits, involvedNumQudits) zeros((involvedNumQudits, involvedNumQudits))]
        Cp_TUCp = transpose(LocCliff.XZencoding)*U*LocCliff.XZencoding
        globalOutputTab.phases[involvedSlice] = mod.((globalInputTab.phases[involvedSlice] + transpose(globalInputTab.XZencoding[involvedSlice, involvedSlice])*LocCliff.phases + Vdiag(transpose(globalInputTab.XZencoding[involvedSlice, involvedSlice])*(2*Pupps(Cp_TUCp)+Pdiag(Cp_TUCp))*globalInputTab.XZencoding[involvedSlice, involvedSlice]) - transpose(globalInputTab.XZencoding[involvedSlice, involvedSlice])*Vdiag(Cp_TUCp)) , (2*subsystem_dim))
    end

    return deepcopy(globalOutputTab)
end

function applyClifftoCliff(Cliff::Tab, InputTab::Tab, system_size, subsystem_dim)
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
            #involvedQudits = cmd[2]
            #involvedNumQudits = length(involvedQudits)
            #Cliff = deepcopy(genCliffordTab[cmd[1]])
            #Cliff.cmds[1][2] = involvedQudits
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

function giveCMDInverse(singleCMD)
    if singleCMD[1] == "P" 
        if isodd(subsystem_dim) #See paper for difference in phase / order
            numRep = subsystem_dim -1
        elseif iseven(subsystem_dim)
            numRep = 4*subsystem_dim - 1
        end
    elseif singleCMD[1] == "F"
        numRep = 4-1 #Look up why order 4, seems to work numerically!
    elseif singleCMD[1] == "CN" || singleCMD[1]== "Z"
        numRep = subsystem_dim-1 #CN shifts second bit # places further accord. to first bit
    else
        println("What gate should that be?")
        exit()
    end

    inverted_cmd = []
    for rep in 1:numRep
        push!(inverted_cmd, singleCMD)
    end
    return inverted_cmd
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

    Cinv_TUCinv = transpose(tempInverse)*(U*tempInverse)
    @show tempInverse
    invertedEncoding = mod.(tempInverse , subsystem_dim)
    invertedPhase = mod.(-transpose(tempInverse)*(Cliff.phases + Vdiag(transpose(Cliff.XZencoding)*(2*Pupps(Cinv_TUCinv) + Pdiag(Cinv_TUCinv))*Cliff.XZencoding)- transpose(Cliff.XZencoding)*Vdiag(Cinv_TUCinv)) , (2*subsystem_dim))
    @show system_size, subsystem_dim, inverted_cmds, invertedEncoding, invertedPhase
    return Tab(system_size, subsystem_dim, inverted_cmds, invertedEncoding, invertedPhase)
end


#@show applyClifftoCliff(Tab(3,2, [["F",[1]]]), Tab(3,2), num_qudits, d)

#GHZ:
function generateGHZTab(num_qudits, d, phase)
if phase == "+"
    ghztab = Tab(num_qudits,d)
    #@show ghztab
    Hgate = deepcopy(genCliffordTab["F"])
    Hgate.cmds[1][2] = [1]
    ghztab = applyClifftoCliff(Hgate,ghztab, num_qudits,d)
    for CNgate_i in 2:num_qudits
        CNgate = deepcopy(genCliffordTab["CN"])
        CNgate.cmds[1][2] = [1,CNgate_i]
        ghztab = applyClifftoCliff(CNgate,ghztab, num_qudits,d)
    end
    finalghztab = ghztab
    #@show finalghztab
    #@assert all(checkConditions(finalghztab, d))
    return finalghztab
elseif phase == "-"
    @assert iseven(d)
    if d==2
        num_phaseGates = 2
    else
        num_phaseGates = Int(genCliffordOrder["Z"]/2)
    end
    ghztab = Tab(num_qudits,d)
    #@show ghztab
    Hgate = deepcopy(genCliffordTab["F"])
    Hgate.cmds[1][2] = [1]
    ghztab = applyClifftoCliff(Hgate,ghztab, num_qudits,d)
    for CNgate_i in 2:num_qudits
        CNgate = deepcopy(genCliffordTab["CN"])
        CNgate.cmds[1][2] = [1,CNgate_i]
        ghztab = applyClifftoCliff(CNgate,ghztab, num_qudits,d)
    end
    for pg in 1:num_phaseGates    
        Pgate = deepcopy(genCliffordTab["P"])
        Pgate.cmds[1][2] = [num_qudits]
        ghztab = applyClifftoCliff(Pgate,ghztab, num_qudits,d)
    end
    finalghztab = ghztab
    return finalghztab
end
end

function naiveRandomTabState(num_qudits, d)
    tempStateTab = Tab(num_qudits,d)
    for i in 1:3^d
        randGateStr = rand(collect(keys(genCliffordTab)))
        randGate = deepcopy(genCliffordTab[randGateStr])
        if randGateStr == "CN"
            randGate.cmds[1][2] = sample(1:num_qudits, 2, replace = false) #Ensures no two equal
        elseif randGateStr in ["P", "F", "Z"]
            randGate.cmds[1][2] = [rand(1:num_qudits)]
        else
            @show randGateStr
            println("Check or update your gate-set!")
            exit()
        end
        tempStateTab = applyClifftoCliff(randGate,tempStateTab, num_qudits,d)
    end

    #@assert all(checkConditions(tempStateTab, d))
    return tempStateTab
end
function naiveRandomTabClifford(num_qudits,d, onlyLocalOps=false)
    cliffArray = []
    for i in 1:rand(d^num_qudits:600)
        println("Sample gate $i")
        if num_qudits == 1 || onlyLocalOps
            gateSet = collect(keys(genCliffordTab))
            filter!(x->x!="CN",gateSet)
            randGateStr = rand(gateSet)
        else
            randGateStr = rand(collect(keys(genCliffordTab)))
        end    
        #randGateStr = rand(["P", "F", "Z"])
        randGate = deepcopy(genCliffordTab[randGateStr])
        
        if randGateStr == "CN"
            #qb = rand(1:num_qudits-1)
            randGate.cmds[1][2] = StatsBase.sample(1:num_qudits, 2, replace = false) #Ensures no two equal
        elseif randGateStr in ["P", "F", "Z"]
            randGate.cmds[1][2] = [rand(1:num_qudits)]
        else
            #randGateStr = rand(["F", "Z"])
            #randGate.cmds[1][2] = [rand(1:num_qudits)]
            println("Check or update your gate-set!")
            exit()
        end

        cmdEntryIdentifier = [randGateStr, randGate.cmds[1][2]]

        if onlyLocalOps == false || length(cliffArray)<genCliffordOrder[randGateStr]-1  #Currently global circuits are not corrected!
            push!(cliffArray, [randGate,cmdEntryIdentifier])
        elseif count(x->x[2]==cmdEntryIdentifier, cliffArray[end-genCliffordOrder[randGateStr]+1+1:end]) != genCliffordOrder[randGateStr]-1 
            @show genCliffordOrder[randGateStr], length(cliffArray[end-genCliffordOrder[randGateStr]+1+1:end])
            #If local circuit: Check whether adding one more gate of this type would make trivial
            push!(cliffArray, [randGate,cmdEntryIdentifier])
        else count(x->x[2]==cmdEntryIdentifier, cliffArray[end-genCliffordOrder[randGateStr]+1+1:end]) == genCliffordOrder[randGateStr]-1 
            #If this command would make existing commands trivial -> remove 
            deleteat!(cliffArray, findall(x->x[2]==cmdEntryIdentifier, cliffArray))
        end
        @show randGateStr, cliffArray
    end
    tempCliffTab = Tab(num_qudits,d)
    for ct in cliffArray
        tempCliffTab = applyClifftoCliff(ct[1],tempCliffTab, num_qudits,d)
    end
    return tempCliffTab
end
#-----------Basis rotations--------------------

#num_qudits, d = 2,9
#omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab, genCliffordOrder = initialize(num_qudits, d)
#@show a[:,1]*adjoint(a[:,1]) -  a[:,2]*adjoint(a[:,2])
#show(stdout, "text/plain", round.(eigvecs(genPauliMat["Xd"]*genPauliMat["Zd"]),digits=8))
#show(stdout, "text/plain", round.(eigvals(genPauliMat["Xd"]*genPauliMat["Zd"]), digits=8))

#show(stdout, "text/plain", round.(genCliffordMat["F"],digits=8)*adjoint(round.(genCliffordMat["P"],digits=8))*(round.(genPauliMat["Xd"], digits=8)*round.(genPauliMat["Zd"], digits=8))*adjoint( round.(genCliffordMat["F"],digits=8)*adjoint(round.(genCliffordMat["P"],digits=8))))
#show(stdout, "text/plain", round.(genCliffordMat["F"],digits=8)*(round.(genPauliMat["Xd"], digits=8))*adjoint( round.(genCliffordMat["F"],digits=8)))
#show(stdout, "text/plain",round.(genPauliMat["Zd"],digits=8))  #*genPauliMat["Xd"]
#show(stdout, "text/plain",round.(genPauliMat["Zd"],digits=8)./(round.(genCliffordMat["F"],digits=8)*adjoint(round.(genCliffordMat["P"],digits=8))*(round.(genPauliMat["Xd"], digits=8)*round.(genPauliMat["Zd"], digits=8))*adjoint( round.(genCliffordMat["F"],digits=8)*adjoint(round.(genCliffordMat["P"],digits=8)))))  #*genPauliMat["Xd"]
#show(stdout, "text/plain",round.(genPauliMat["Zd"],digits=8)./(round.(genCliffordMat["F"],digits=8)*(round.(genPauliMat["Xd"], digits=8))*adjoint( round.(genCliffordMat["F"],digits=8))))  #*genPauliMat["Xd"]

function getPauliRotationDic(d)
    genPauliRotations = Dict()
    for xpow in 0:d-1
        for zpow in 0:d-1
            c_pauli = PauliOp(1, d, [xpow, zpow],0)
            @show c_pauli

            #If the operator is the identity or just any power of Z, the answer is simple and rigid!
            if xpow == 0
                totCTab = Tab(1,d)
                extraPhase = 0

                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                if conCP.XZencoding == [0,1] || zpow!=1
                    push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                    continue
                else
                    @show conCP
                    @assert false
                end
            #=elseif xpow == 0 && zpow != 0 #Solution !NOT! straightforward
                p = fill("Z", genCliffordOrder["Z"]-(zpow)+1)
                totCTab = Tab(1,d)
                for el in p
                    ct_temp = deepcopy(genCliffordTab[el])
                    totCTab = applyClifftoCliff(ct_temp, totCTab,1,d)
                end
                @show totCTab
                extraPhase = (length(p)*genPauliTab[1].phase - zpow*genPauliTab[1].phase) #1 = 01 = Zd
                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                if conCP.XZencoding == [0,1]
                    push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                    continue
                else
                    @show conCP
                    @assert false
                end =#
            #=elseif xpow == 0 && zpow != 0 #Solution !NOT! straightforward
                p = fill("Z", genCliffordOrder["Z"]-(zpow)+1)
                totCTab = Tab(1,d)
                for el in p
                    ct_temp = deepcopy(genCliffordTab[el])
                    totCTab = applyClifftoCliff(ct_temp, totCTab,1,d)
                end
                @show totCTab
                extraPhase = (length(p)*genPauliTab[1].phase - zpow*genPauliTab[1].phase) #1 = 01 = Zd
                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                if conCP.XZencoding == [0,1]
                    push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                    continue
                else
                    @show conCP
                    @assert false
                end =#
            end

            #First we try it the most efficient brute force way we know to keep circuits and runtime as small as possible
            #for cc in filter(v -> (!(subset2(fill("P", genCliffordOrder["P"]), collect(v)))|| !(subset2(fill("F", genCliffordOrder["F"]), collect(v)))|| !(subset2(fill("Z", genCliffordOrder["Z"]), collect(v)))), unique(Iterators.product(fill(["I","Z", "P", "F"],6)...)))
            for cc in unique(Iterators.product(fill(["I","Z", "P", "F"],6)...))
                p = collect(cc)
                #for p in unique(permutations(cc))
                
                totCTab = Tab(1,d)
                for el in p
                    if el == "I"
                        continue
                    else
                        ct_temp = deepcopy(genCliffordTab[el])
                        totCTab = applyClifftoCliff(ct_temp, totCTab,1,d)
                    end
                end
                #totC = prod([genPauliMat[p_i] for p_i in p])
                #pro = totC* ((genPauliMat["Xd"]^xpow)+(genPauliMat["Zd"]^zpow))*totC
                conCP = conjugateClifftoPauli(totCTab,c_pauli,1,d)
                extraPhase = (conCP.phase - c_pauli.phase)
                @show xpow, zpow
                if conCP.XZencoding == [0,1] && !haskey(genPauliRotations, join(["X", xpow,"Z",zpow]))
                    @show "Noq"
                    push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                elseif conCP.XZencoding == [0,1] && haskey(genPauliRotations, join(["X", xpow,"Z",zpow])) && extraPhase==0 && 
                    if genPauliRotations[join(["X", xpow,"Z",zpow])][1] != 0
                        push!(genPauliRotations, join(["X", xpow,"Z",zpow]) => [extraPhase , totCTab])
                    end
                end
                #end
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
    @show keys(genPauliRotations) 
    save("rot-$d.jld", "rot-$d", d)
    #reload with: load("rot-$d.jld")["rot-$d"]
    return genPauliRotations
end

#=
function Back2normal(inp)
    i=1
    threshold = 10^(-8)
    while (abs(real(inp[i])) < threshold && abs(imag(inp[i])) < threshold )
        i+=1
    end 
    @show i
    if real(inp[i]) > threshold
        return inp
    elseif real(inp[i]) < -threshold
        return inp .* (-1)
    elseif abs(real(inp[i])) < threshold && imag(inp[i]) > 0
        return inp .* (-im)
    elseif abs(real(inp[i])) < threshold && imag(inp[i]) < 0
        return inp .* im
    end
end

by(x) = (abs(x), mod(angle(x)+2*pi, 2*pi))
by2(x) = -findfirst(y->abs(y-x)<10^(-8), eigvals((genPauliMat["Zd"]^1)))
genPauliBasisVector = Dict("X1Z0" => [Back2normal(round.(eigvec, digits=10)) for eigvec in eachcol(eigvecs((genPauliMat["Xd"]^1)*(genPauliMat["Zd"]^0), sortby=by2))],
                           "X0Z1" => reverse([eigvec for eigvec in eachcol(eigvecs((genPauliMat["Xd"]^0)*(genPauliMat["Zd"]^1)))]))
genRotationMatrices = Dict("X1Z0" => sum([vcat(zeros(ev-1),1,zeros(d-ev))*adjoint(genPauliBasisVector["X1Z0"][ev]) for ev in 1:d]))

#@show omegaPhase, omegaPhaseTilde
@show "Out"
t1 = genRotationMatrices["X1Z0"]
show(stdout, "text/plain",t1)
println("\n")
t2 = round.(genCliffordMat["F"],digits=8)
show(stdout, "text/plain", t2)
println("\n")
show(stdout, "text/plain", t1./t2)
=#


#-----------Test general points for random states/cliffords (simple)--------------------------------

#=
num_qudits, d = 4,3
omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab = initialize(num_qudits, d)

@show a= Tab(num_qudits, d)
#@show a= naiveRandomTabState(num_qudits, d)
#@show b= naiveRandomTabClifford(num_qudits, d)
@show b= generateGHZTab(num_qudits,d)
b_mat = round.(simpleCliffordToMat(b, num_qudits, d))[:,1]
@show b_mat
invdCliffTab = invertCliffTab(deepcopy(b) ,num_qudits, d)
@show ba= applyClifftoCliff(b, a, num_qudits, d)
@show applyClifftoCliff(invdCliffTab, ba, num_qudits, d)

=#
#--------------------Test Pauli conjugation------------------------
#=
num_qudits, d = 3,2
omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab = initialize(num_qudits,d)

P = PauliOp(num_qudits,d, [1, 1, 1, 1, 0,1],2)

Hgate = deepcopy(genCliffordTab["F"])
Hgate.cmds[1][2] = [1]

#C = Tab(num_qudits, d)
Cp = naiveRandomTabClifford(num_qudits, d)
##@show C
#Cp = generateGHZTab(num_qudits,d)
#Cp = applyClifftoCliff(Hgate, C, num_qudits, d)

r1 =  round.(simpleCliffordToMat(Cp,num_qudits, d)*pauliToMat(P,num_qudits, d)*adjoint(simpleCliffordToMat(Cp,num_qudits, d)), digits=10)
r2 =  round.(simpleCliffordToMat(Cp,num_qudits, d)*pauliToMat(P,num_qudits, d)*simpleCliffordToMat(invertCliffTab(Cp,num_qudits, d),num_qudits, d), digits=10)

cctp = conjugateClifftoPauli(Cp, P, num_qudits, d)
cctp_mat = round.(pauliToMat(cctp,num_qudits, d), digits=10)
#@show cctp_mat

@show maximum(abs.(r1 .-cctp_mat))
@show maximum(abs.(r2 .-cctp_mat))
@show maximum(abs.(r2 .-r1))
#@show r1
#@show r2
#@show cctp
=#

#--------------------CNOT problem --------------------------------

#=
num_qudits, d = 3,2
omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab = initialize(num_qudits,d)

cntab = Tab(num_qudits,d)
CNgate1 = deepcopy(genCliffordTab["CN"])
CNgate1.cmds[1][2] = [1,2]
cntab = applyClifftoCliff(CNgate1,deepcopy(cntab), num_qudits,d)
show(stdout, "text/plain", cntab.XZencoding) 

CNgate2 = deepcopy(genCliffordTab["F"])
CNgate2.cmds[1][2] = [2]
cntab = applyClifftoCliff(CNgate2,deepcopy(cntab), num_qudits,d)
show(stdout, "text/plain", cntab.XZencoding) 
=#

#--------------------Test GHZ + with inversion -----------------------
#=
finalghztab = generateGHZTab(3,2)

TBI_finalghztab = deepcopy(finalghztab)
ST_finalghztab = deepcopy(finalghztab)

invdCliffTab = invertCliffTab(TBI_finalghztab ,3, 2)
res = applyClifftoCliff(invdCliffTab, ST_finalghztab, 3, 2)
@show res
@show finalghztab
=#
#@show conjugateClifftoPauli()
#Base.show(io::IO,z::Tab) = print(io,z.cmds)

#----------Test removeRedundancy-----------------
#num_qudits, d = 3,2
#omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab, genCliffordOrder = initialize(num_qudits,d)

#GLobal clifford:
#Cp = naiveRandomTabClifford(num_qudits, d)
#@show Cp.cmds, length(Cp.cmds)

#reduc_Cp = removeRedundancy(Cp, false)
#@show reduc_Cp.cmds, length(reduc_Cp.cmds)

#Local Clifford collection:
#Cp = [naiveRandomTabClifford(1, d, true) for i in 1:1000]

#reduc_Cp = removeRedundancy(Cp, true)
#@show Cp[3].cmds, length(Cp[3].cmds) #Example index=3
#@show reduc_Cp[3].cmds, length(reduc_Cp[3].cmds)

#-------------------------Exploring Clifford sampling:
#=
num_qudits, d = 3,2
omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab, genCliffordOrder = initialize(num_qudits,d)

function samplingCols(pauli_app_size,subsystem_dim)
    """
    Pauli_app_size = Defines (remaining) range of application for the sampled paulis
    """
    #Sample randomly 2*k dits:
    randPauli1 = rand(0:subsystem_dim-1, 2*pauli_app_size)
    while (randPauli1 == zeros(Int, 2*pauli_app_size))  #Not identity!
        randPauli1 = rand(0:subsystem_dim-1, 2*pauli_app_size)
    end
    
    randPauli2 = rand(0:subsystem_dim-1, 2*pauli_app_size)
    randPhases = rand(0:2*subsystem_dim -1, 2) #Phases always mod 2*d!
    
    U = sparse([zeros((pauli_app_size, pauli_app_size)) zeros((pauli_app_size, pauli_app_size)); Matrix(1I, pauli_app_size, pauli_app_size) zeros((pauli_app_size,pauli_app_size))])
    P = mod.(U - transpose(U), subsystem_dim)
    doCommute(x,y) = (omegaPhase^(mod(transpose(x)*P*y, subsystem_dim)) == 1.0 + 0.0*im) #Eq. 5 in paper 

    doColsCommute = doCommute(randPauli1, randPauli2)
    while(doColsCommute) #Paulis/cols need to anticommute!
        randPauli2 = rand(0:subsystem_dim-1, 2*pauli_app_size)
        doColsCommute = doCommute(randPauli1, randPauli2)
    end

    return hcat(randPauli1, randPauli2), randPhases
end

function sweepingSubtab(tabinput::Tab, colIndices, rowIndices, pauli_app_size,subsystem_dim)
    subtab = deepcopy(tabinput.XZencoding[rowIndices, colIndices])
    subtabphases = deepcopy(tabinput.phases[colIndices])
    cmd_output = []

    function applyGateOnArrays(gateStr,x, xp, js)
        #First create global clifford from local one:
        globalGateXZtemp = setupTab(pauli_app_size, subsystem_dim)
        globalGatePhasetemp = zeros(Int, 2*pauli_app_size)
        globalGateXZout = deepcopy(globalGateXZtemp)
        globalGatePhaseout = deepcopy(globalGatePhasetemp)

        involvedQudits = js
        involvedNumQudits = length(involvedQudits)
        involvedSlice = vcat(involvedQudits, involvedQudits .+ pauli_app_size)
        #@show involvedSlice, size(globalGatePhasetemp), size(globalGateXZtemp)
        #@show "Stop"
        #sleep(30)

        LocCliff = deepcopy(genCliffordTab[gateStr])
        LocCliff.cmds[1][2] = involvedQudits

        globalGateXZout[involvedSlice, involvedSlice] = mod.((LocCliff.XZencoding*globalGateXZtemp[involvedSlice, involvedSlice]), subsystem_dim)           

        U = [zeros((involvedNumQudits, involvedNumQudits)) zeros((involvedNumQudits, involvedNumQudits)); Matrix(1I, involvedNumQudits, involvedNumQudits) zeros((involvedNumQudits, involvedNumQudits))]
        Cp_TUCp = transpose(LocCliff.XZencoding)*U*LocCliff.XZencoding
        
        globalGatePhaseout[involvedSlice] = mod.((globalGatePhasetemp[involvedSlice] + transpose(globalGateXZtemp[involvedSlice, involvedSlice])*LocCliff.phases + Vdiag(transpose(globalGateXZtemp[involvedSlice, involvedSlice])*(2*Pupps(Cp_TUCp)+Pdiag(Cp_TUCp))*globalGateXZtemp[involvedSlice, involvedSlice]) - transpose(globalGateXZtemp[involvedSlice, involvedSlice])*Vdiag(Cp_TUCp)) , (2*subsystem_dim))
      
        #Apply created Clifford globally (to include entanglement)
        reduc_x_size = size(x)[2]
        reduc_xp_length = length(xp)
        x = hcat(x,zeros(Int,(2*pauli_app_size, 2*pauli_app_size -reduc_x_size)))
        xp = vcat(xp, zeros(Int,2*pauli_app_size- reduc_xp_length)) 


        xXZout = mod.((globalGateXZout*x), subsystem_dim)
            
        U = [zeros((pauli_app_size, pauli_app_size)) zeros((pauli_app_size, pauli_app_size)); Matrix(1I, pauli_app_size,pauli_app_size) zeros((pauli_app_size, pauli_app_size))]
        Cp_TUCp = transpose(globalGateXZout)*U*globalGateXZout
        xPhaseout = mod.((xp + transpose(x)*xp + Vdiag(transpose(x)*(2*Pupps(Cp_TUCp)+Pdiag(Cp_TUCp))*x) - transpose(x)*Vdiag(Cp_TUCp)) , (2*subsystem_dim))

        return xXZout[1:end, 1:reduc_x_size], xPhaseout[1:reduc_xp_length]
    end

    #Step 1: Clear Z-block elements in first col:
    @assert pauli_app_size == Int(size(subtab)[1]/2)
    first_zblock = subtab[pauli_app_size+1:end,1]
    found_j = findall(x -> x!=0, first_zblock) #Find all non-zero indices in Z
    for j in found_j
        first_xblock = subtab[1:pauli_app_size,1]
        if first_xblock[j] == 0 #If zero, just swap entries X,Z
            subtab, subtabphases = applyGateOnArrays("F",subtab,subtabphases , [j])
            push!(cmd_output, ["F",[rowIndices[1]-1+j]])
        else
            for it in 1:(genCliffordOrder["P"]-first_zblock[j]) #Apply P so often that first Z block is cleared!
                subtab, subtabphases = applyGateOnArrays("P",subtab,subtabphases , [j])
                push!(cmd_output, ["P",[rowIndices[1]-1+j]])
            end
        end
    end

    #Step 2: Determine the (sorted) list of indices J and Apply CN to J_i, J_i+1
    first_xblock = subtab[1:pauli_app_size,1]
    J = sort(findall(x -> x!=0, first_xblock)) #Needs to be non-empty!

    while(length(J) != 1)
        for i in collect(1:2:length(J))
            if i < length(J)
                subtab, subtabphases = applyGateOnArrays("CN",subtab,subtabphases , [J[i], J[i+1]])
                push!(cmd_output, ["CN",[rowIndices[1]-1+J[i], rowIndices[1]-1+J[i+1]]])
            end
        end
        J = J[collect(1:2:length(J))] #Update J to its entries upon only odd indices
    end
    @assert length(J)==1

    #Step 3:
    if J[1] != 1 #If first place non-zero - ?CHECK CORRECTNESS OF CONDITION!!!?
        #Swap J[1] with qubit 1: i.e. swap gate = 3 CNOT gates:
        subtab, subtabphases = applyGateOnArrays("CN",subtab,subtabphases , [J[1], 1]) #Order correct || irrelevant?
        subtab, subtabphases = applyGateOnArrays("CN",subtab,subtabphases , [1, J[1]])
        subtab, subtabphases = applyGateOnArrays("CN",subtab,subtabphases , [J[1], 1])
        push!(cmd_output, ["CN",[rowIndices[1]-1+J[1], rowIndices[1]-1+1]])
        push!(cmd_output, ["CN",[rowIndices[1]-1+1, rowIndices[1]-1+J[1]]])
        push!(cmd_output, ["CN",[rowIndices[1]-1+J[1], rowIndices[1]-1+1]])
    end 
    @assert subtab[pauli_app_size+1,2] != 0

    #Step 4:
    if subtab[1:end,1] != vcat(zeros(pauli_app_size), [1], zeros(pauli_app_size-1)) #Skip step if second puali = +- Z_1
        #First apply H(1):
        subtab, subtabphases = applyGateOnArrays("F",subtab,subtabphases , [1])
        push!(cmd_output, ["F",[rowIndices[1]-1+1]])
        #Repeat steps 1 and 2 for second x and z pauli blocks
        #Rep. Step 1: 
        second_zblock = subtab[pauli_app_size+1:end,2]
        found_j = findall(x -> x!=0, second_zblock) #Find all non-zero indices in Z
        for j in found_j
            second_xblock = subtab[1:pauli_app_size,2]
            if second_xblock[j] == 0 #If zero, just swap entries X,Z
                subtab, subtabphases = applyGateOnArrays("F",subtab,subtabphases , [j])
                push!(cmd_output, ["F",[rowIndices[1]-1+j]])
            else
                for it in 1:(genCliffordOrder["P"]-second_zblock[j]) #Apply P so often that first Z block is cleared!
                    subtab, subtabphases = applyGateOnArrays("P",subtab,subtabphases , [j])
                    push!(cmd_output, ["P",[rowIndices[1]-1+j]])
                end
            end
        end

        #Step 2: Determine the (sorted) list of indices J and Apply CN to J_i, J_i+1
        second_xblock = subtab[1:pauli_app_size,2]
        J2 = sort(findall(x -> x!=0, second_xblock)) #Needs to be non-empty!

        while(length(J2) != 1)
            for i in collect(1:2:length(J2))
                if i < length(J2)
                    subtab, subtabphases = applyGateOnArrays("CN",subtab,subtabphases , [J2[i], J2[i+1]])
                    push!(cmd_output, ["CN",[rowIndices[1]-1+J2[i], rowIndices[1]-1+J2[i+1]]])
                end
            end
            J2 = J2[collect(1:2:length(J2))] #Update J2 to its entries upon only odd indices
        end
        @assert length(J2)==1

        #Again apply H(1) to obtain Paulis X1 and Z1
        subtab, subtabphases = applyGateOnArrays("F",subtab,subtabphases , [1])
        push!(cmd_output, ["F",[rowIndices[1]-1+1]])
    end

    #Step 5: Clear out any signs : Check whether these conditions apply also to qudits!!
    #=
    if subtabphases[1]==0 && subtabphases[2]==1
        #Apply X_1
    elseif subtabphases[1]==1 && subtabphases[2]==1
        #Apply Y_1
    else
        #Apply Z_1
        subtab, subtabphases = applyGateOnArrays("Z",subtab,subtabphases , [1])
    end
    =#
    #We don't have a valid operator X;Y in Clifford group, let's randomize the sign by applying Z-gates by chance
    if rand([0,1])==1   
        for repit in rand(1:genCliffordOrder["Z"]-1)
            subtab, subtabphases = applyGateOnArrays("Z",subtab,subtabphases , [1])
            push!(cmd_output, ["Z",[rowIndices[1]-1+1]])
            #Noramally we would need to apply Z (or X;Y) only on qubit 1 to keep a correct XZencoding
            #However, since we redraw the circuit after sampling this doesn't matter.
        end
    end

    #Close:

    taboutput = deepcopy(tabinput)
    taboutput.XZencoding[rowIndices, colIndices] = subtab
    taboutput.phases[colIndices] = subtabphases

    push!(taboutput.cmds, cmd_output...)
    return taboutput, cmd_output
end

function generateSimpleRandomClifford(system_size, subsystem_dim)
    #Create empty tableau:
    RC = Tab(system_size, subsystem_dim, [], zeros((2*system_size, 2*system_size)), zeros(2*system_size))

    cmd_log = []
    #Do iterations over each 2 dim subtableau:
    for l in 1:system_size
        #Step 1: Select subtableau mask indices:
        colIndices = [2*l-1, 2*l] # which Paulis
        rowIndices = vcat(collect(l:system_size), collect(system_size+l:2*system_size)) #which Qudits
        pauli_app_size = system_size-l+1

        #Step 2: Sample random anticommuting Paulis on n+1−l qudits w/ random signs for current cols w.r.t to previous sweeping
        
        rand_cols_XZs, rand_cols_phases = samplingCols(pauli_app_size, subsystem_dim)
        RC.XZencoding[rowIndices, colIndices] = rand_cols_XZs
        RC.phases[colIndices] = rand_cols_phases

        #Step 3: Sweep subtableau to Pauli basis locally X_1^1 and Z_1^1
        #@show colIndices, rowIndices, size(RC.XZencoding)
        #sleep(40)
        RC ,sweep_cmd = sweepingSubtab(RC, colIndices, rowIndices, pauli_app_size, subsystem_dim)
        push!(cmd_log, sweep_cmd...)
    end
    
    #Construct the (Adjoint) clifford tableau as the complete randomized clifford tableau
    #Normally collect all random sampled paulis during above process for efficiency and use that -> but mind the phase in step 5!
    outCliff = Tab(num_qudits, subsystem_dim)

    for s_cmd in cmd_log
        @show s_cmd
        randGate = deepcopy(genCliffordTab[s_cmd[1]])
        randGate.cmds[1][2] = s_cmd[2]
        applyClifftoCliff(randGate,outCliff, num_qudits, subsystem_dim)
    end

    return outCliff #,RC, cmd_log
end

outC = generateSimpleRandomClifford(num_qudits, d)

@show outC

=#

#-----------------Tableau measurements on qudits:
function getOrderedBases(system_size, subsystem_dim)
    eigZ = eigen(genPauliMat["Zd"])
    orderedZeigVals = zeros(ComplexF64, length(eigZ.values))
    for i in 1:length(eigZ.values)
        orderedZeigVals[findfirst(x->x==1.0,eigZ.vectors[:,i])] = eigZ.values[i]
    end
    sortEig(x)= if findfirst(y->y==round(x, digits=8) ,round.(orderedZeigVals, digits=8)) == nothing argmin(abs.(orderedZeigVals .- x)) else findfirst(y->y==round(x, digits=8) ,round.(orderedZeigVals, digits=8)) end
    @assert convert(Array{Int,2},eigvecs(genPauliMat["Zd"], sortby=sortEig)) == Matrix(LinearAlgebra.I, (subsystem_dim,subsystem_dim))
    orderedZeigVecs = eigvecs(genPauliMat["Zd"], sortby=sortEig)
    orderedPeigVecs = Dict([xp, zp] => eigvecs(genPauliMat["Xd"]^xp * genPauliMat["Zd"]^zp, sortby=sortEig) for xp in 0:subsystem_dim-1 for zp in 0:subsystem_dim-1)

    return orderedPeigVecs, orderedZeigVals, orderedZeigVecs
end
#orderedPeigVecs, orderedZeigVals, orderedZeigVecs = getOrderedBases(system, subsystem_dim)

#------------------Inner product of tableaus: ----------------

U = [zeros((3, 3)) zeros((3, 3)); Matrix(1I,3,3) zeros((3, 3))]

function tab_innerProduct(leftTab, rightTab, system_size, subsystem_dim)
    """Matthijs S.C. RIJLAARSDAM - Delft Master Thesis """
    leftUTab = invertCliffTab(deepcopy(leftTab), system_size, subsystem_dim)
    updatedRightTab = applyClifftoCliff(leftUTab, rightTab, system_size, subsystem_dim)
    overlap = 1

    #Measure overlap:

    U_loc = [0 0; 1 0]
    P_loc = mod.(U_loc - transpose(U_loc), subsystem_dim)
    doCommute(x,y) = Int(mod(transpose(x.XZencoding)*P_loc*y.XZencoding, subsystem_dim)) == 0 #Eq. 5 in paper 
    commutePhase(x,y) =  Int(mod(transpose(x.XZencoding)*P_loc*y.XZencoding, 2*subsystem_dim))
    extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
    
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
        #@show 2*Pop1.phase + 2*Pop2.phase + sum(multiplyPauliOps(PauliOp(1,d,Pop1.XZencoding[[j, j+snum_qudits]],0), PauliOp(1,d,Pop2.XZencoding[[j, j+num_qudits]],0),1, d).phase for j in 1:num_qudits)
        #sleep(10)
        XZOut = mod.(Pop1.XZencoding .+ Pop2.XZencoding, d)
        return PauliOp(num_qudits, d, XZOut, phaseOut) 
    end
    
    function rowsum(t, h, i)
        """Modified from Aar / Got to represent Pauli Operator Multiplication """
        resultPop = multiplyPauliOps(PauliOp(system_size,subsystem_dim, t.XZencoding[:,h] ,t.phases[h]), PauliOp(system_size, subsystem_dim,t.XZencoding[:,i],t.phases[i]), 1, subsystem_dim)
        t.XZencoding[:,h] = resultPop.XZencoding
        t.phases[h] = resultPop.phase
        return t
    end
    for meas_qi in 1:system_size
        #Measure qubit i:
        #println("Measuring qubit $meas_qi ...")
        measPauliOp = PauliOp(system_size, subsystem_dim, vcat(zeros(system_size+meas_qi-1),1,zeros(system_size-meas_qi)),0)
        locMeasPauliOp = PauliOp(1, subsystem_dim, [0,1],0)
      
        foundNone = true
        for col in 1+system_size:2*system_size #Go over all STABILIZERS
            #tempStab = PauliOp(system_size, subsystem_dim, rotatedState.XZencoding[:,col],0)
            tempStab = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],col],0)
            #@show "Check $col, $locMeasPauliOp"
            #if doCommute(measPauliOp, tempStab)
            if doCommute(locMeasPauliOp, tempStab)
                continue
            else
                foundNone = false
                #-----------Get measurement outcome (GOTTESMAN STYLE): -------------
                #Obtain probabilities for each of the d possible outcomes:
                locTempStab = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],col],0) 
                p=[0.5, 0.5] #Qubit
                #Qudits: Question is whether probabilities for d>2 really equal
                p=fill(1/subsystem_dim, subsystem_dim) 
                
               # @show p
                s_outcome = rand(Categorical(p)) - 1 #mod d digit of outcome 0:d-1
                #overlap *= 1/sqrt(2)
^
                comPhase = commutePhase( locMeasPauliOp, tempStab)
                @show comPhase, tempStab, meas_qi
                overlap *= cos(comPhase*(pi/2)/subsystem_dim)

                #-------Post-select state-tableau for next measurement: -----------------------
                #Call rowsum on all indices i \neq p, p = rowsum(i,p)
                for iterCol in 1:2*system_size #GO OVER (DE)STABILIZERS
                    #Nelement = PauliOp(system_size, subsystem_dim, rotatedState.XZencoding[:,iterCol],0)
                    Nelement = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],iterCol],0)
                    #@show Nelement
                    #if doCommute(measPauliOp, Nelement)
                    if doCommute(locMeasPauliOp, Nelement)
                        #@show Nelement
                        continue
                    elseif iterCol != col
                        updatedRightTab = rowsum(updatedRightTab, iterCol, col)
                    end
                end

                #Copy non-commuting operator to destabilizer on same position:
                updatedRightTab.phases[col-system_size] = updatedRightTab.phases[col]
                updatedRightTab.XZencoding[:,col-system_size] = updatedRightTab.XZencoding[:,col]
                
                #Replace non-commuting operator by measurement operator:
                updatedRightTab.phases[col] = measPauliOp.phase
                updatedRightTab.XZencoding[:,col] = measPauliOp.XZencoding

                updatedRightTab.phases[col] = 2*0 #Post-select as if measurement outcome = +1 = ket{0}
                break #Only handle all of this for the first appearing p
            end
        end
        if foundNone == true
            @show "DETERMINISTIC",updatedRightTab
            #Measurement deterministic
            #Add some scratch space:
            #@show "Here, $meas_qi for "
            updatedRightTab.XZencoding = hcat(updatedRightTab.XZencoding, zeros(Int, 2*system_size))
            updatedRightTab.phases = vcat(updatedRightTab.phases, 0)

            #Call rowsum for i in 1:n : rowsum(2n +1  i+n)
            for col in 1:system_size
                #tempStab = PauliOp(system_size, subsystem_dim, rotatedState.XZencoding[:,col],0)
                tempStab = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],col],0)
                #if doCommute(measPauliOp, tempStab)
                if doCommute(locMeasPauliOp, tempStab)
                    continue
                else    
                    updatedRightTab = rowsum(updatedRightTab, 2*system_size+1, col+system_size) #MIND ORDER ROW <-> COL INTERCHANGED
                end
            end

           
            if isodd(subsystem_dim)
                s_outcome = mod(Int(updatedRightTab.phases[2*system_size+1]), subsystem_dim)      
            else
                try 
                    s_outcome = Int(updatedRightTab.phases[2*system_size+1] / 2) #Convert 0/d -> 1/2 -> hope that other don't exist
                catch v                                                                                                                                                             
                    s_outcome = Int(mod(2*mod(updatedRightTab.phases[2*system_size+1],2*subsystem_dim) / 2, subsystem_dim))
                end
            end
            
            
            @show s_outcome, updatedRightTab.phases

            if s_outcome != 0 
                overlap = 0
            end
            #Else keep whatever it is currrently
            #Remove Scratch space again:
            updatedRightTab.XZencoding = updatedRightTab.XZencoding[:, 1:end-1]
            updatedRightTab.phases = updatedRightTab.phases[1:end-1]
            #Deterministic = no further postselection required
        end
    end
    return overlap
end

num_qudits,d = 3,3
omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab, genCliffordOrder = initialize(num_qudits,d)

leftTab = deepcopy(naiveRandomTabState(num_qudits,d))
rightTab = deepcopy(naiveRandomTabState(num_qudits,d))
#leftTab = deepcopy(generateGHZTab(num_qudits,d,"+"))
#rightTab = deepcopy(generateGHZTab(num_qudits,d,"-"))
show(stdout, "text/plain", simpleCliffordToMat(deepcopy(leftTab), num_qudits,d)[:,1])
println("\n")
show(stdout, "text/plain", simpleCliffordToMat(deepcopy(rightTab),num_qudits,d)[:,1])
println("\n DOT=")
show(stdout, "text/plain", adjoint(simpleCliffordToMat(deepcopy(leftTab), num_qudits,d)[:,1])*simpleCliffordToMat(deepcopy(rightTab),num_qudits,d)[:,1])
println("\n")
@show tab_innerProduct(leftTab, rightTab, num_qudits, d)

#-------------misc----
#=
num_qudits,d = 1,2
omegaPhase, omegaPhaseTilde, genPauliMat, genPauliTab, genCliffordMat, genCliffordTab, genCliffordOrder = initialize(num_qudits,d)

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
xz1,xz2 = [1,1], [0,1]
@show multiplyPauliOps(PauliOp(num_qudits, d,xz1,0),PauliOp(num_qudits, d,xz2,0), num_qudits,d).phase

g= nothing
if xz1[1]== xz1[2] && xz1[1] == 0 
    g= 0
elseif xz1[1]== xz1[2] && xz1[1] == 1
    g = xz2[2]-xz2[1]
elseif xz1[1]== 1 && xz1[2] == 0 
    g = xz2[2]*(2*xz2[1]-1); 
elseif xz1[1]== 0 && xz1[2] == 1
    g= xz2[1]*(1- 2*xz2[2])
end
@show g
=#
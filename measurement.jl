using Pkg
using LinearAlgebra
using Distributions
using HDF5
using QuantumInformation
#using QuantumClifford
using Combinatorics
using SparseArrays
using DelimitedFiles


function measureCliffordEff(stateTab, measInput, system_size,subsystem_dim, mode, compOut=false)
    """
    Returns a Array{Array{Int,1}, 1} of measurement outcomes of measInput on stateTab

    Args
    - mode::String = "g": global or "l": local
    """
    U_loc = [0 0; 1 0]
    P_loc = mod.(U_loc - transpose(U_loc), subsystem_dim)
    doCommute(x,y) = Int(mod(transpose(x.XZencoding)*P_loc*y.XZencoding, subsystem_dim)) == 0 #Eq. 5 in paper 
    extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
    
    function rowsum(t, h, i)
        resultPop = multiplyPauliOps(PauliOp(system_size,subsystem_dim, t.XZencoding[:,h] ,t.phases[h]), PauliOp(system_size, subsystem_dim,t.XZencoding[:,i],t.phases[i]), 1, subsystem_dim)
        t.XZencoding[:,h] = resultPop.XZencoding
        t.phases[h] = resultPop.phase
        return t
    end
    
    rotatedState = nothing

    if mode == "l"
        #println("Chosen local measurement mode")
        pauliOperatorArray = measInput
        @assert typeof(pauliOperatorArray) == Array{Any,1}
        rotatedState = deepcopy(stateTab)

        @showprogress "Rotating state to computational basis..." for meas_qi in 1:system_size

            xpow, zpow = extractXZpow(pauliOperatorArray[meas_qi])  #.XZencoding[1]  .XZencoding[2]
            extraPhase = omegaPhaseTilde^genPauliRotations[pauliOperatorArray[meas_qi]][1]
            thisCliff = deepcopy(genPauliRotations[pauliOperatorArray[meas_qi]][2])
            if length(thisCliff.cmds) != 0
                for i in 1:length(thisCliff.cmds)
                    thisCliff.cmds[i][2] = [meas_qi]
                end
                
                rotatedState = applyClifftoCliff(thisCliff,deepcopy(rotatedState), system_size ,subsystem_dim)
            end
        end
    elseif mode == "g"
        #println("Chosen global measurement mode")
        @assert typeof(measInput) == Tab
        thisCliff = measInput
        rotatedState = applyClifftoCliff(thisCliff,deepcopy(stateTab), system_size ,subsystem_dim)
    elseif mode == "lg" #Apply locally, but input given as global PauliOp
        #println("Chosen local measurement mode")
        gPauliOperator = measInput
        @assert typeof(gPauliOperator) == PauliOp
        
        pauliPhase = gPauliOperator.phase
        rotatedState = deepcopy(stateTab)
    end

    #-----------Make the measurements---------------
    outcomeDigits=[]

    @showprogress "Measuring single register ..." for meas_qi in 1:system_size
        #Measure qubit i:
         measPauliOp = PauliOp(system_size, subsystem_dim, vcat(zeros(system_size+meas_qi-1),1,zeros(system_size-meas_qi)),0)
        locMeasPauliOp = PauliOp(1, subsystem_dim, [0,1],0)
        s_outcome = nothing

        foundNone = true
        for col in 1+system_size:2*system_size #Go over all STABILIZERS
            tempStab = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],col],0)
            if doCommute(locMeasPauliOp, tempStab)
                continue
            else
                foundNone = false
                locTempStab = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],col],0) 

                #-----------Get measurement outcome (GOTTESMAN STYLE): -------------
                #= Obtain probabilities for each of the d possible outcomes:
                p = [abs(dot(eigvecs(pauliToMat(tempStab, system_size, subsystem_dim))[:,postOuti],eigvecs(pauliToMat(measPauliOp, system_size, subsystem_dim))[:,posOuti]))^2 for postOuti in 1:subsystem_dim]
                Sample outcome:
                s_outcome = rand(Categorical(p)) - 1 #mod d digit of outcome 0:d-1
                push!(outcomeDigits, s_outcome)  
                =#

                #-----------Get measurement outcome (AARONSON STYLE): -------------
                # -> -> -> -> Manual calculation of probabilities:
                #tempStab = PauliOp(system_size, subsystem_dim, rotatedState.XZencoding[:,col],0)
                #p = [abs(dot(eigvecs(pauliToMat(tempStab, system_size, subsystem_dim))[:,postOuti],  eigvecs(pauliToMat(measPauliOp, system_size, subsystem_dim))[:,postOuti]))^2 for postOuti in 1:subsystem_dim]
                
                #sortEig(x)= findfirst(y->y==round(x, digits=8) ,round.(orderedZeigVals, digits=8))
                #eigLocTempStab = eigvecs(pauliToMat(locTempStab, 1, subsystem_dim), sortby=sortEig)
                #convert(Array{Int,2},eigvecs(genPauliMat["Zd"], sortby=sortEig))
                
                #pp = real.([abs(dot(orderedPeigVecs[locTempStab.XZencoding][:,postOuti], orderedZeigVecs[:,postOuti]))^2 for postOuti in 1:subsystem_dim])
                #if sum(pp) == 0.0
                #    pp = real.([abs(dot(orderedPeigVecs[locTempStab.XZencoding][:,subsystem_dim-postOuti+1], orderedZeigVecs[:,postOuti]))^2 for postOuti in 1:subsystem_dim])
                #end
                #pp = pp ./ sum(pp)   
                #@assert p == pp
                #p = pp

                #->->->->-> The lazy/simple way:
                p=[0.5, 0.5] #Qubit
                #Qudits: Question is whether probabilities for d>2 always equal
                p=fill(1/subsystem_dim, subsystem_dim) 
                
                #Sample outcome:
                s_outcome = rand(Categorical(p)) - 1 #mod d digit of outcome 0:d-1
                push!(outcomeDigits, s_outcome)


                #-------Post-select state-tableau for next measurement: -----------------------
                #Call rowsum on all indices i \neq p, p = rowsum(i,p)
                for iterCol in 1:2*system_size #GO OVER (DE)STABILIZERS
                    Nelement = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],iterCol],0)
                    if doCommute(locMeasPauliOp, Nelement)
                        continue
                    elseif iterCol != col
                        rotatedState = rowsum(rotatedState, iterCol, col)
                    end
                end

                #Copy non-commuting operator to destabilizer on same position:
                rotatedState.phases[col-system_size] = rotatedState.phases[col]
                rotatedState.XZencoding[:,col-system_size] = rotatedState.XZencoding[:,col]
                
                #Replace non-commuting operator by measurement operator:
                rotatedState.phases[col] = measPauliOp.phase
                rotatedState.XZencoding[:,col] = measPauliOp.XZencoding

                rotatedState.phases[col] = 2*s_outcome #Convert 0/1 -> 0/2, Encode measurement outcome in state
                break #Only handle all of this for the first appearing p
            end
        end
        if foundNone == true
            #Measurement deterministic

            #Add some scratch space:
            rotatedState.XZencoding = hcat(rotatedState.XZencoding, zeros(Int, 2*system_size))
            rotatedState.phases = vcat(rotatedState.phases, 0)

            #Call rowsum for i in 1:n : rowsum(2n +1  i+n)
            for col in 1:system_size
                tempStab = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],col],0)
                if doCommute(locMeasPauliOp, tempStab)
                    continue
                else    
                    rotatedState = rowsum(rotatedState, 2*system_size+1, col+system_size) #MIND ORDER ROW <-> COL INTERCHANGED
                end
            end

            s_outcome = nothing
            if isodd(subsystem_dim)
                s_outcome = mod(Int(rotatedState.phases[2*system_size+1]), subsystem_dim)      
            else
                try 
                    s_outcome = Int(rotatedState.phases[2*system_size+1] / 2) #Convert 0/d -> 1/2 -> hope that other don't exist
                catch v
                    open("./singularPoints.txt", "a") do io
                        writedlm(io, [system_size subsystem_dim;"-" "-"])
                    end                                                                                                                                                               
                    s_outcome = Int(mod(2*mod(rotatedState.phases[2*system_size+1],2*subsystem_dim) / 2, subsystem_dim))
                end
            end
            push!(outcomeDigits, s_outcome)
            
            #Remove Scratch space again:
            rotatedState.XZencoding = rotatedState.XZencoding[:, 1:end-1]
            rotatedState.phases = rotatedState.phases[1:end-1]
            #Deterministic = no further postselection required
        end
    end

    @assert length(outcomeDigits) == system_size
    #NOTE: "reverse" is correct since in kronecker product, qubit = largest structure in total state
    
    #@show outcomeDigits
    #@show measInput

    return outcomeDigits
end

function measureCliffordEff_Cor(stateTab, measInput, system_size,subsystem_dim, mode, compOut=false)
    """
    Returns a Array{Array{Int,1}, 1} of measurement outcomes of measInput on stateTab
    Corrected version which includes terms with higher powers of Z

    Args
    - mode::String = "g": global or "l": local
    """
    U_loc = [0 0; 1 0]
    P_loc = mod.(U_loc - transpose(U_loc), subsystem_dim)
    doCommute(x,y) = Int(mod(transpose(x.XZencoding)*P_loc*y.XZencoding, subsystem_dim)) == 0 #Eq. 5 in paper 
    extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
    
    genKetExpansPhases = Dict(b => mod.([(-k*b) for k in 0:subsystem_dim-1], subsystem_dim) for b in 0:subsystem_dim-1) #Powers of omegaPhase of measurement operator : Inverse fourier tranform (ket{k}}]) sum_k omegaPhase^{-b*k}
    genKetExpansPhasesInverse = Dict(mod.([(-k*b) for k in 0:subsystem_dim-1], subsystem_dim) => b for b in 0:subsystem_dim-1) #Powers of omegaPhase of measurement operator : Inverse fourier tranform (ket{k}}]) sum_k omegaPhase^{-b*k}


    function rowsum(t, h, i)
        resultPop = multiplyPauliOps(PauliOp(system_size,subsystem_dim, t.XZencoding[:,h] ,t.phases[h]), PauliOp(system_size, subsystem_dim,t.XZencoding[:,i],t.phases[i]), 1, subsystem_dim)
        t.XZencoding[:,h] = resultPop.XZencoding
        t.phases[h] = resultPop.phase
        return t
    end
    symp_innerprod(x,y) =  Int(mod(transpose(x.XZencoding)*P*y.XZencoding, 2*subsystem_dim)) #TODO: check

    function weightedrowsum(t, h, i, w_i ,measPauliOp)   
        """ w_i = Weight index i (mainly destabilizers)"""              

        resultPop = multiplyPauliOps(PauliOp(system_size,subsystem_dim, t.XZencoding[:,h] ,t.phases[h]), PauliOp(system_size, subsystem_dim,t.XZencoding[:,i],t.phases[i]),  1, subsystem_dim)
        C_kh = symp_innerprod(PauliOp(system_size, subsystem_dim,t.XZencoding[:,w_i],t.phases[w_i]),measPauliOp) 
        
        t.XZencoding[:,h] = mod.(C_kh .* resultPop.XZencoding, subsystem_dim)
        t.phases[h] = resultPop.phase

        return t
    end

    rotatedState = nothing

    if mode == "l"
        #println("Chosen local measurement mode")
        pauliOperatorArray = measInput
        @assert typeof(pauliOperatorArray) == Array{Any,1}
        rotatedState = deepcopy(stateTab)

        @showprogress "Rotating state to computational basis..." for meas_qi in 1:system_size

            xpow, zpow = extractXZpow(pauliOperatorArray[meas_qi])  #.XZencoding[1]  .XZencoding[2]
            extraPhase = omegaPhaseTilde^genPauliRotations[pauliOperatorArray[meas_qi]][1]
            thisCliff = deepcopy(genPauliRotations[pauliOperatorArray[meas_qi]][2])
            if length(thisCliff.cmds) != 0
                for i in 1:length(thisCliff.cmds)
                    thisCliff.cmds[i][2] = [meas_qi]
                end
                
                rotatedState = applyClifftoCliff(thisCliff,deepcopy(rotatedState), system_size ,subsystem_dim)
            end
        end
    elseif mode == "g"
        #println("Chosen global measurement mode")
        @assert typeof(measInput) == Tab
        thisCliff = measInput
        rotatedState = applyClifftoCliff(thisCliff,deepcopy(stateTab), system_size ,subsystem_dim)
    elseif mode == "lg" #Apply locally, but input given as global PauliOp
        #println("Chosen local measurement mode")
        gPauliOperator = measInput
        @assert typeof(gPauliOperator) == PauliOp
        
        pauliPhase = gPauliOperator.phase
        rotatedState = deepcopy(stateTab)

    end

    #-----------Make the measurements---------------
    outcomeDigits=[]
    @showprogress "Measuring single register ..." for meas_qi in 1:system_size
        expansion_phases = [0] #Identity needs not be checked
        for z_op_pow in 1:subsystem_dim-1 #Loop over fourier expansion of basis states -> need statisfy commutation relations individually
            
            #Measure qubit i:
            measPauliOp = PauliOp(system_size, subsystem_dim, vcat(zeros(system_size+meas_qi-1),z_op_pow,zeros(system_size-meas_qi)),0)
            locMeasPauliOp = PauliOp(1, subsystem_dim, [0,z_op_pow],0)

            s_outcome = nothing

            foundNone = true
            for col in 1+system_size:2*system_size #Go over all STABILIZERS
                tempStab = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],col],0)
                if doCommute(locMeasPauliOp, tempStab)
                    continue
                else
                    foundNone = false
                    locTempStab = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],col],0) 

                    #-----------Get measurement outcome (GOTTESMAN STYLE): -------------
                    
                    #Obtain probabilities for each of the d possible outcomes:
                    #p = [abs(dot(eigvecs(pauliToMat(tempStab, system_size, subsystem_dim))[:,postOuti],eigvecs(pauliToMat(measPauliOp, system_size, subsystem_dim))[:,posOuti]))^2 for postOuti in 1:subsystem_dim]
                    #Sample outcome:
                    #s_outcome = rand(Categorical(p)) - 1 #mod d digit of outcome 0:d-1
                    #push!(outcomeDigits, s_outcome)

                    #-----------Get measurement outcome (AARONSON STYLE): -------------
                    # -> -> -> -> Manual calculation of probabilities:
                    #tempStab = PauliOp(system_size, subsystem_dim, rotatedState.XZencoding[:,col],0)
                    #p = [abs(dot(eigvecs(pauliToMat(tempStab, system_size, subsystem_dim))[:,postOuti],  eigvecs(pauliToMat(measPauliOp, system_size, subsystem_dim))[:,postOuti]))^2 for postOuti in 1:subsystem_dim]
                    #sortEig(x)= findfirst(y->y==round(x, digits=8) ,round.(orderedZeigVals, digits=8))
                    #eigLocTempStab = eigvecs(pauliToMat(locTempStab, 1, subsystem_dim), sortby=sortEig)
                    #convert(Array{Int,2},eigvecs(genPauliMat["Zd"], sortby=sortEig))
                    
                    #pp = real.([abs(dot(orderedPeigVecs[locTempStab.XZencoding][:,postOuti], orderedZeigVecs[:,postOuti]))^2 for postOuti in 1:subsystem_dim])
                    #if sum(pp) == 0.0
                    #    pp = real.([abs(dot(orderedPeigVecs[locTempStab.XZencoding][:,subsystem_dim-postOuti+1], orderedZeigVecs[:,postOuti]))^2 for postOuti in 1:subsystem_dim])
                    #end
                    #pp = pp ./ sum(pp)   
                    #@assert p == pp
                    #p = pp

                    #->->->-> The simple/lazy way:
                    p=[0.5, 0.5] #Qubit
                    #Qudits: Question is whether probabilities for d>2 really equal
                    p=fill(1/subsystem_dim, subsystem_dim)

                    #Sample outcome:
                    s_outcome = rand(Categorical(p)) - 1 #mod d digit of outcome 0:d-1
                    push!(expansion_phases, s_outcome) 


                    #-------Post-select state-tableau for next measurement: -----------------------
                    #Call rowsum on all indices i \neq p, p = rowsum(i,p)
                    for iterCol in 1:2*system_size #GO OVER (DE)STABILIZERS
                        Nelement = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],iterCol],0)
                        if doCommute(locMeasPauliOp, Nelement)
                            #@show Nelement
                            continue
                        elseif iterCol != col
                            rotatedState = rowsum(rotatedState, iterCol, col)
                        end
                    end

                    #Copy non-commuting operator to destabilizer on same position:
                    rotatedState.phases[col-system_size] = rotatedState.phases[col]
                    rotatedState.XZencoding[:,col-system_size] = rotatedState.XZencoding[:,col]
                    
                    #Replace non-commuting operator by measurement operator:
                    rotatedState.phases[col] = measPauliOp.phase
                    rotatedState.XZencoding[:,col] = measPauliOp.XZencoding

                    rotatedState.phases[col] = mod(2*s_outcome, 2*subsystem_dim)  #TODO: Check #Convert 0/1 -> 0/2, Encode measurement outcome in state
                    break #Only handle all of this for the first appearing p
                end
            end
            if foundNone == true
                #Measurement deterministic

                #Add some scratch space:
                rotatedState.XZencoding = hcat(rotatedState.XZencoding, zeros(Int, 2*system_size))
                rotatedState.phases = vcat(rotatedState.phases, 0)

                #Call rowsum for i in 1:n : rowsum(2n +1  i+n)
                for col in 1:system_size
                    tempStab = PauliOp(1, subsystem_dim, rotatedState.XZencoding[[meas_qi, system_size+meas_qi],col],0)
                    if doCommute(locMeasPauliOp, tempStab)
                        continue
                    else
                        rotatedState = weightedrowsum(rotatedState, 2*system_size+1, col+system_size, col, measPauliOp) #MIND ORDER ROW <-> COL INTERCHANGED
                    end
                end

                s_outcome = nothing
                
                if isodd(subsystem_dim)
                    s_outcome = mod(Int(rotatedState.phases[2*system_size+1]), subsystem_dim)      
                else
                    try 
                        s_outcome = Int(rotatedState.phases[2*system_size+1] / 2) #Convert 0/d -> 1/2 -> hope that other don't exist
                    catch v                                                                                                                                                         
                        s_outcome = Int(mod(2*mod(rotatedState.phases[2*system_size+1],2*subsystem_dim) / 2, subsystem_dim))
                    end
                end
                
                push!(expansion_phases,s_outcome )
                
                #Remove Scratch space again:
                rotatedState.XZencoding = rotatedState.XZencoding[:, 1:end-1]
                rotatedState.phases = rotatedState.phases[1:end-1]
                #Deterministic = no further postselection required
            end 
        end
        try
            @show collect(keys(genKetExpansPhasesInverse)), expansion_phases
            push!(outcomeDigits, genKetExpansPhasesInverse[expansion_phases])
        catch v
            @show "Failed"
            minDev = argmin([mean(abs.(collect(keys(genKetExpansPhasesInverse))[i] .- expansion_phases)) for i in 1:system_size])
            push!(outcomeDigits, genKetExpansPhasesInverse[collect(keys(genKetExpansPhasesInverse))[minDev]])
        end
    end

    @assert length(outcomeDigits) == system_size

    return outcomeDigits
end

function measureGlobalCliffordDirect(stateTab, cliffordTab, system_size,subsystem_dim, compOut=false)
    """
    Returns a Array{Array{Int,1}, 1} of measurement outcomes of global measurement cliffordTab on stateTab
    
    Args
    - stateTab::Tab = input state on which to apply  the measurement; tableau format
    """
    rotatedState = applyClifftoCliff(cliffordTab, stateTab, system_size,subsystem_dim)
    #Realize the state in naive matrix multiplication:-
    totalMat = Matrix(1LinearAlgebra.I, subsystem_dim^system_size, subsystem_dim^system_size)
    Id = Matrix(1LinearAlgebra.I, subsystem_dim, subsystem_dim)
    for cmd in rotatedState.cmds
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
            completeGate = sum([kron([if qi==c_qubit eyeMat(i,subsystem_dim) elseif qi==t_qubit Id*Xd^(i-1) else Id end for qi in 1:system_size]...) for i in 1:subsystem_dim])
            return completeGate
        
        end
        
        #Left multiply!:-
        if cmd[1]=="CN" && cmd_weight==2
            totalMat = getCNQuditGateMat(cmd[2], system_size,subsystem_dim)*totalMat  
        elseif cmd_weight == 1
            t_qudit = cmd[2][1]
            totalMat = kron([if qi==t_qudit genCliffordMat[cmd[1]] else Id end for qi in 1:system_size]...)*totalMat
        else
            println("What else?")
            exit()
        end
    end

    #nth measurement prob. = |<b_n| C C' | vac > |^2 -> first colom (nth row) as probArray:

    probArray = abs.(round.(totalMat[:,1], digits=10)) .^2

    outcomeIndex = rand(Categorical(real.(probArray)))
    outcomeDigits = reverse(digits(outcomeIndex-1, base=subsystem_dim, pad=system_size))
    #NOTE: "reverse" is correct since in kronecker product, qubit = largest structure in total state
    if compOut
        return outcomeDigits
    end
end

function measureLocalCliffordDirect(stateTab, cliffordTabArray, system_size,subsystem_dim, compOut=false)
    """
    Returns a Array{Array{Int,1}, 1} of measurement outcomes of local measurement cliffordTab on stateTab
    
    Args
    - stateTab::Tab = input state on which to apply  the measurement; tableau format
    """

    rotatedState = applyClifftoCliff(cliffordTabArray[1], deepcopy(stateTab), system_size,subsystem_dim)

    for meas_qi in 2:system_size
        rotatedState = applyClifftoCliff(cliffordTabArray[meas_qi], deepcopy(rotatedState), system_size,subsystem_dim)
    end
    
    totalMat = Matrix(1LinearAlgebra.I, subsystem_dim^system_size, subsystem_dim^system_size)
    Id = Matrix(1LinearAlgebra.I, subsystem_dim, subsystem_dim)

    for cmd in rotatedState.cmds
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
            completeGate = sum([kron([if qi==c_qubit eyeMat(i,subsystem_dim) elseif qi==t_qubit Id*Xd^(i-1) else Id end for qi in 1:system_size]...) for i in 1:subsystem_dim])
            return completeGate
        end
        

        #Left multiply!:-
        if cmd[1]=="CN" && cmd_weight==2
            totalMat = getCNQuditGateMat(cmd[2], system_size,subsystem_dim)*totalMat  
        elseif cmd_weight == 1
            t_qudit = cmd[2][1]
            totalMat = kron([if qi==t_qudit genCliffordMat[cmd[1]] else Id end for qi in 1:system_size]...)*totalMat
        else
            println("What else?")
            exit()
        end
    end

    probArray = abs.(round.(totalMat[:,1], digits=10)) .^2

    outcomeIndex = rand(Categorical(real.(probArray)))
    outcomeDigits = reverse(digits(outcomeIndex-1, base=subsystem_dim, pad=system_size))
    
    if compOut
        return outcomeDigits
    end
end


function measureGlobalCliffordMat(state_mat, cliffordTableau, system_size,subsystem_dim, compOut=false)
    """
        Measure matrix/array based on global Cliffords
    """
    if typeof(state_mat) == Array{ComplexF64,2}
        state_mat = sparse(state_mat)
    end
    #Realize the state in naive matrix multiplication:-
    Id = Matrix(1LinearAlgebra.I, subsystem_dim, subsystem_dim)
    for cmd in cliffordTableau.cmds
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
            completeGate = sum([kron([if qi==c_qubit eyeMat(i,subsystem_dim) elseif qi==t_qubit Id*Xd^(i-1) else Id end for qi in 1:system_size]...) for i in 1:subsystem_dim])
            return completeGate
        end
        
        #Left multiply!:-
        if cmd[1]=="CN" && cmd_weight==2
            totalCliff = sparse(getCNQuditGateMat(cmd[2], system_size,subsystem_dim))
            state_mat = totalCliff*state_mat*adjoint(totalCliff)
        elseif cmd_weight == 1
         
            t_qudit = cmd[2][1]
            totalCliff =  sparse(kron([if qi==t_qudit genCliffordMat[cmd[1]] else Id end for qi in 1:system_size]...))
            state_mat = totalCliff*state_mat*adjoint(totalCliff)
        else
            println("What else?")
            exit()
        end
    end

    probArray = diag(state_mat)
    probArray = collect(probArray)
    outcomeIndex = rand(Categorical(real.(probArray)))
    outcomeDigits = reverse(digits(outcomeIndex-1, base=subsystem_dim, pad=system_size))
    #NOTE: "reverse" is correct since in kronecker product, qubit = largest structure in total state
      
    if compOut
        return outcomeDigits
    end
end


function measureLocalCliffordMat(state_mat, pauliOperatorArray::Array{Any,1}, system_size,subsystem_dim, compOut=false)
    """
        Measure matrix-based local Cliffords 
    """
    #state_mat = sparse(state_mat)
    
    Id = sparse(convert(Array{ComplexF64,2},Matrix(LinearAlgebra.I, subsystem_dim, subsystem_dim)))

    for meas_qi in 1:system_size
        extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
        
        xpow, zpow = extractXZpow(pauliOperatorArray[meas_qi])  #.XZencoding[1]  .XZencoding[2]
        extraPhase = omegaPhaseTilde^genPauliRotations[pauliOperatorArray[meas_qi]][1]
        if length(genPauliRotations[pauliOperatorArray[meas_qi]][2].cmds) != 0
            for cmd in genPauliRotations[pauliOperatorArray[meas_qi]][2].cmds
                
                totalCliff = sparse(kron([if qi==meas_qi sparse(convert(Array{ComplexF64,2},genCliffordMat[cmd[1]])) else Id end for qi in 1:system_size]...))
                state_mat = convert(Array{ComplexF64,2}, (totalCliff*(state_mat*adjoint(totalCliff)))) #extraPhase* ??

            end
        end
    end
    
    #= Realize the state in naive matrix multiplication:
    ### IGNORE - ONLY FOR DEBUGGING - INEFFICIENT
    
    for cmd in rotatedState.cmds
        cmd_weight = length(cmd[2])

        function getCNQuditGateMat(Activeq, system_size,subsystem_dim)
            function eyeMat(ind,size)
                tempvec = zeros((size,size))
                tempvec[ind,ind] = 1
                return tempvec
            end
            #completeGate = []
            #ffq = findfirst(x->x==1, Activeq)
            #if ffq == nothing
            #    completeGate = sparse(Matrix(1LinearAlgebra.I,subsystem_dim, subsystem_dim))
            #else
            #Id = Matrix(1LinearAlgebra.I, subsystem_dim, subsystem_dim)
            Xd = genPauliMat["Xd"]
            c_qubit = Activeq[1] #Controlqudit
            t_qubit = Activeq[2]
            completeGate = sum([kron([if qi==c_qubit eyeMat(i,subsystem_dim) elseif qi==t_qubit Id*Xd^(i-1) else Id end for qi in 1:system_size]...) for i in 1:subsystem_dim])
            return completeGate

            #subsystem_dim^(q-1)
            #vcat([transpose([if (digits(row-1, base=subsystem_dim, pad=2)[2]==digits(col-1, base=subsystem_dim, pad=2)[2] && digits(row-1, base=subsystem_dim, pad=2)[1]== mod(sum(digits(col-1, base=subsystem_dim, pad=2)),subsystem_dim)) 1 else 0 end for col in 1:subsystem_dim^2]) for row in 1:subsystem_dim^2]...)
        end
        
        #Left multiply!:-
        if cmd[1]=="CN" && cmd_weight==2
            #totalMat = getCNQuditGateMat(cmd[2], system_size,subsystem_dim)*totalMat  
            totalCliff = sparse(getCNQuditGateMat(cmd[2], system_size,subsystem_dim))
            state_mat = totalCliff*state_mat*adjoint(totalCliff)
        elseif cmd_weight == 1
            #totalMat = kron([if qi==t_qudit genCliffordMat[cmd[1]] else Id end for qi in 1:system_size]...)*totalMat
         
            t_qudit = cmd[2][1]
            totalCliff =  sparse(kron([if qi==t_qudit genCliffordMat[cmd[1]] else Id end for qi in 1:system_size]...))
            state_mat = totalCliff*state_mat*adjoint(totalCliff)
 
            #@show size(blockdiag(sparse(Matrix(1LinearAlgebra.I,left_id_dim,left_id_dim)), sparse(genCliffordMat[cmd[1]]) ,sparse(Matrix(1LinearAlgebra.I,right_id_dim, right_id_dim ))))
        else
            println("What else?")
            exit()
        end
    end
    =#

    probArray = round.(real.(diag(state_mat)), digits=8)
    #Normalize:
    probArray = probArray ./ sum(probArray)
  
    probArray = collect(probArray)
    outcomeIndex = rand(Categorical(probArray))
    outcomeDigits = reverse(digits(outcomeIndex-1, base=subsystem_dim, pad=system_size))
    #NOTE: "reverse" is correct since in kronecker product, qubit = largest structure in total state
    
    if compOut
        return outcomeDigits
    end
end


function compBasisMeasurement(stateCompBasis)
    """
    Measure any input array in the computational basis!
    Output comp. basis index or vector.
    """
    #@show tr(stateCompBasis)
    dim = size(stateCompBasis)[1]
    num_qubits = Int(log(dim)/log(2))

    probArray = []

    probArray = round.(LinearAlgebra.diag(stateCompBasis), digits=8)

    #@show sum(probArray)
    
    outcomeIndex = rand(Categorical(real.(probArray)))
    outcomeDigits = reverse(digits(outcomeIndex-1, base=2, pad=num_qubits))
    #NOTE: "reverse" is correct since in kronecker product, qubit = largest structure in total state
    
    #@show outcomeDigits, outcomeIndex 
    
    outcomeVector = zeros(dim)
    outcomeVector[outcomeIndex]=1

    return outcomeDigits, outcomeVector
end

#=
function measureGlobalClifford(stateTab, cliffordTab, system_size,subsystem_dim, compOut=false)

    rotatedState = applyClifftoCliff(cliffordTab, stateTab, system_size,subsystem_dim)
    
    thresholdProb = rand()
    #=
    outcomeIndex = nothing
    cumsumProb = 0
    for bv in 1:subsystem_dim^system_size #go over any basis vector for |<b_n| C C' | vac > |^2
        m = length(cliffordTab.cmds)
        for e_i in Iterators.product([1:subsystem_dim^system_size +1] for i in 1:m-1]...)
            println(e_i)
            if cmd[1]=="CN" && cmd_weight==2
                exp *= getCNQuditGateMat(cmd[2], system_size,subsystem_dim)
            elseif cmd_weight == 1
                t_qudit = cmd[2][1]
                exp *= sparse(kron([if qi==t_qudit genPauliMat[cmd[1]] else Id end for qi in 1:system_size]...))[]
             
                #@show size(blockdiag(sparse(Matrix(1LinearAlgebra.I,left_id_dim,left_id_dim)), sparse(genCliffordMat[cmd[1]]) ,sparse(Matrix(1LinearAlgebra.I,right_id_dim, right_id_dim ))))
            else
                println("What else?")
                exit()
            end
     
        end
        bv_prob = abs()^2

        cumsumProb += bv_prob
        if cumsumProb >= thresholdProb
            outcomeIndex = bv
            break          
        end
    end
    =#
    if compOut
        return compBasisMeasurement(rotatedState) 
    else
        return adjoint(clifford_mat)*compBasisMeasurement(rotatedState)*clifford_mat
    end
end
=#

#= ---------------------- Some basic initializing from the qubit era:
pauliEncoding = Dict("X"=>0, "Y"=>1, "Z"=>2)
pauliMapping = Dict("X"=> [0.0 1.0; 1.0 0.0], "Y"=> [0.0 -im; im 0.0], "Z"=>[1.0 0.0;0.0 -1.0], "I"=>[1.0 0.0; 0.0 1.0])
pauliTraces = Dict(keystring => LinearAlgebra.tr(pauliMapping[keystring]) for keystring in ["X","Y","Z","I"])
pauliBasisVector = Dict("X" => [[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]], "Y" => [[1/sqrt(2), im/sqrt(2)],[1/sqrt(2), -im/sqrt(2)]], "Z"=> [[1,0], [0,1]])

rotationMatrices = Dict("X" => (pauliBasisVector["Z"][1]*adjoint(pauliBasisVector["X"][1]) + pauliBasisVector["Z"][2]*adjoint(pauliBasisVector["X"][2])),
                        "Y" => (pauliBasisVector["Z"][1]*adjoint(pauliBasisVector["Y"][1]) + pauliBasisVector["Z"][2]*adjoint(pauliBasisVector["Y"][2])),
                        "Z" => (pauliBasisVector["Z"][1]*adjoint(pauliBasisVector["Z"][1]) + pauliBasisVector["Z"][2]*adjoint(pauliBasisVector["Z"][2])))

cliffordTransformations = Dict("I" => [1.0 0.0; 0.0 1.0],
                            "H"=> 1/sqrt(2) * [1.0 1.0; 1.0 -1.0],
                            "P"=> [1.0 0.0; 0.0 im],
                            "CN"=> [1 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0])
allequal(x) = all(y->real(y)==real(x[1])&&imag(y)==imag(x[1]),x)

#function checkForCommonFactor(a1, a2)
#    a1vec = vec(a1)
#    a2vec = vec(a2)
#    
#    collectedDivision = []
#    for idx in 1:length(a1vec)
#        if a2vec[idx] != 0.0
#            push!(collectedDivision, a1vec[idx]/a2vec[idx])
#            @show a1vec[idx]/a2vec[idx]
#        elseif a2vec[idx] == 0.0 && a1vec[idx] != 0.0
#            #@show a1vec, a2vec
#            return false, nothing
#        elseif a2vec[idx] == 0.0 && a1vec[idx] == 0.0
#            continue
#        end     
#    end
#    @show collectedDivision
#    if allequal(collectedDivision)
#        return true, collectedDivision[1]
#    else
#        return false, nothing
#    end
#end

function findTransformPhase(P,C)
    """
    For multiqubit gates C, provide P as array
    """
    @show P,C

    if C != "CN"
        transformed = cliffordTransformations[C]*(pauliMapping[P]*adjoint(cliffordTransformations[C]))

        for cand in ["X", "Y", "Z", "I"]
            haveCF, foundPhase = checkForCommonFactor(transformed, pauliMapping[cand]) 
            if haveCF
                return foundPhase
            end
        end
    elseif C == "CN"
        transformed = cliffordTransformations[C]*(kron(pauliMapping[P[1]], pauliMapping[P[2]])*adjoint(cliffordTransformations[C]))
        for cand1 in ["X", "Y", "Z", "I"]
            for cand2 in ["X", "Y", "Z", "I"]
                haveCF, foundPhase = checkForCommonFactor(transformed, kron(pauliMapping[cand1], pauliMapping[cand2])) 
                @show cand1, cand2, haveCF, foundPhase
                if haveCF
                    return foundPhase
                end
            end
        end
    end
    println("No common factor/phase found!")
end

pauliCliffordTransformPhase = Dict(join([C,P,C]) => round(findTransformPhase(P,C), digits=10) for P in ["I", "X","Y", "Z"] for C in ["I", "H", "P"])
#Need to handle multi/2-qubit gates independently:
for P1 in ["X", "Y", "Z", "I"]
    for P2 in ["X", "Y", "Z", "I"]
        pauliCliffordTransformPhase[join(["CN",P1,P2,"CN"])]=  round(findTransformPhase([P1, P2],"CN"), digits=10)
    end
end

@show pauliCliffordTransformPhase


#-----------------------The following stems from Juqst/QuantumClifford dependent 
#-----------------------Routines from the qubit era = don't use!

function randomClifford(num_qubits)
    cliffTableau = cliffordToTableau(num_qubits, rand(1:getNumberOfSymplecticCliffords(num_qubits)),rand(1:2^num_qubits)) # #NOTE: see paper that indeed the last bitstrings r,s are meant to be abitrary. alpha, beta, gamma, delta = constrained & r,s (phase) = free
    #randomGates = [rand(["CN", "H", "P"]) for i in 1:2^num_qubits]
    #cliffTableau = []
    #for gate in randomGates
    #   targetQubit = rand(1:num_qubits)
    #   if gate == "CN"  
    #       controlQubit = rand(1:num_qubits)
    #       while controlQubit == targetQubit
    #           controlQubit = rand(1:num_qubits)
    #       end
    #       push!(cliffTableau, [gate, [targetQubit, controlQubit]])
    #   else 
    #       push!(cliffTableau, [gate, [targetQubit]])
    #   end
    #end
    #@show num_qubits, *adjoint(makeFromCommand(cliffTableau))
    return cliffTableau    #round.(makeFromCommand(cliffTableau), digits=10)
end

function alt_randomClifford(N)
    """
    Alterantive random Clifford generation based on more modern (2020) algorithm 
    ONLY WITHIN QuantumCLifford.jl framework!
    """
    return random_clifford(N) #bravyi2020hadamard.
end

function tableau_innerProduct(leftTableau, rightTableau)
    
    leftString = split(string(leftTableau), "\n")
    rightString = split(string(rightTableau), "\n")

    leftStabArray = leftString[Int((length(leftString)+1)/2)+1:end-1]#Slice = cut out empty strings at start and end + destabilizers
    rightStabArray = rightString[Int((length(rightString)+1)/2)+1:end-1]
    
    @assert length(leftStabArray) == length(rightStabArray)
    system_size = length(leftStabArray)

    #Convert for non-trivial part to QuantumClifford library
    IdentityOperator = PauliOperator(0, fill(false, system_size), fill(false, system_size))
    phaseDict = Dict("+" => 0, "-"=> 2,"+i" => 1, "-i" => 3)
    
    leftStabArray_c = [PauliOperator(phaseDict[join(split(s,"")[1:end-system_size])], collect.([split(s,"")[end-system_size+1:end] .== "X"])[1], collect.([split(s,"")[end-system_size+1:end] .== "Z"])[1]) for s in leftStabArray]
    filter!(x->x!=IdentityOperator,leftStabArray_c)
    
    rightStabArray_c = [PauliOperator(phaseDict[join(split(s,"")[1:end-system_size])], collect.([split(s,"")[end-system_size+1:end] .== "X"])[1], collect.([split(s,"")[end-system_size+1:end] .== "Z"])[1]) for s in rightStabArray]
    
    filter!(x->x!=IdentityOperator,rightStabArray_c)
    
    @show leftStabArray_c

    pushfirst!(leftStabArray_c, IdentityOperator) #*im factor neutralizes phase
    pushfirst!(rightStabArray_c, IdentityOperator) #*im factor neutralizes phase
    

    function condMult(a1, a2)
        a1split = split(string(a1)[end-system_size+1:end], "")
        a2split = split(string(a2)[end-system_size+1:end], "")
        output = a1*a2
        outputsplit = split(string(output)[end-system_size+1:end], "")
        if count(a1split .!= a2split) <= min(count(outputsplit .!= a1split), count(outputsplit.!= a2split))
            return IdentityOperator
        else
            return output
        end
    end
    
    all_left_generators = unique([condMult(g1,g2) for g2 in leftStabArray_c[2:end] for g1 in leftStabArray_c])
    filter!(x->x!=IdentityOperator,all_left_generators)
    
    all_right_generators = unique([condMult(g1,g2) for g2 in rightStabArray_c[2:end] for g1 in rightStabArray_c])
    filter!(x->x!=IdentityOperator,all_right_generators)
    
    @show length(all_left_generators), length(all_right_generators)
    function getOppositePhase(iPhase)
        if iPhase == "+"
            return "-"
        elseif iPhase == "-"
            return "+"
        elseif iPhase == "+i"
            return "-i"
        elseif iPhase == "-i"
            return "+i"
        end
    end
    
    orthogonal = true
    for lStab in leftStabArray
        for rStab in rightStabArray
            @show lStab, rStab
            if lStab[1:end-system_size] == getOppositePhase(rStab[1:end-system_size]) && lStab[end-system_size+1:end] == rStab[end-system_size+1:end]
               continue 
            end
            orthogonal = false
        end
    end
    if orthogonal
        return 0 #Inner product zero if same stabilizer with opposite sign
    end

    @show all_left_generators
    @show all_right_generators
    #If not zero then 2^-s/2

    minDeviation = system_size*2 # =s   : initial= something bigger then system_size= max.
    
    for gL in powerset(all_left_generators, system_size, system_size)
        for gR in powerset(all_right_generators, system_size, system_size)
            @show minDeviation
            gL_array = vcat([split(string(gLi)[[1:end-system_size-1; end-system_size+1:end]], "") for gLi in gL]...) #Split of phase and split apart chars
            gR_array = vcat([split(string(gRi)[[1:end-system_size-1; end-system_size+1:end]], "") for gRi in gR]...) #Split of phase and split apart chars
            @show gL_array, gR_array
            tempCnt = count(gL_array .!= gR_array)
            if tempCnt < minDeviation
                minDeviation = tempCnt
            elseif tempCnt == 0
                return 1.0 #Cant get lower -> skip rest of loop
            end
        end
    end
    
    @show minDeviation
    return 2^(-minDeviation/2)
end

function getGateCounts(inputTab, tqubits)

    cliffordCommandArray = inputTab.commands[2:end] #Need to exclude first = initalize command

    counts = Dict("H"=> 0, "P"=>0, "CN"=> 0)

    for single_cmd in cliffordCommandArray
        splitted = split(single_cmd, "(")
        splitted[2] = splitted[2][1:end-1] #Remove ")" symbol
        
        if splitted[1] == "hadamard"
            if tqubits[1] == parse(Int, splitted[2])
                counts["H"] += 1
            end
        elseif splitted[1] == "phase"
            if tqubits[1] == parse(Int, splitted[2])
                counts["P"] += 1
            end
        elseif splitted[1] == "cnot"
            involvedQubits = split(splitted[2],",") #Divide qubit definition 
            if [parse(Int, involvedQubits[1]), parse(Int, involvedQubits[2])] == tqubits
                counts["CN"] += 1
            end
        elseif splitted[1] == "initialize"
            println("Oh no, the initialization command has entered the command-list!")
            exit()
        end
    end
    @show counts
    return counts
end

function randomStabilizerState(num_qubits)
    #Input system size num_qubits
    #Output a stabilizer tableau t that encodes the state
    
    t = Tableau(num_qubits)
    for i in 1:rand(1:2^num_qubits)
        randGate = rand(1:3) 
        randQubit = rand(1:num_qubits)
        if randGate == 1
            Juqst.hadamard(t, randQubit)    
        elseif randGate == 2
            Juqst.phase(t, randQubit)         
        elseif randGate == 3
            randControl = rand(1:num_qubits)
            Juqst.cnot(t,randControl,randQubit)
        end
    end
    return t
end

function reverse_clifford(clifford, cmdMode=false)
    """ 
    Reverses the Clifford command list given a Tableau to apply an inverse CLifford Unitary
    Warning: Initliaze command is excluded
    cmdMode::Bool -> clifford input CommandArray / Tableau ?
    """
    CliffCommands = clifford
    if cmdMode == false
        CliffCommands = clifford.commands[2:end]
    end

    reverseCliffCommands = reverse(CliffCommands)

    #Phase-gate = not its own inverse -> invert by 3 x repetition:
    phaseIndices = findall( x -> occursin("phase", x), reverseCliffCommands)
    repetitions = ones(Int, length(reverseCliffCommands))
    for phIx in phaseIndices
        repetitions[phIx] = 3
    end
    finalReverseCliffCommands = vcat(fill.(reverseCliffCommands, repetitions)...) #Inserts the phase repetitions
    return finalReverseCliffCommands
end

function applyCliffordLocal(stateTableau, cliffordCommandArray,targetQubits, tabMode=false)
    """
    Applies Clifford (or any other set of gate commands) to stateTableau
    Always Provide (single) targetQubit indices in array: i=1: Real target, i=2: Control qubit (optional)
    """

    if tabMode
        cliffordCommandArray = cliffordCommandArray.commands[2:end] #Need to exclude first = initalize command
    end
    
    #Check whether the tableau size of given target qubits and clifford correspond!
    @assert parse(Int,cliffordCommandArray[1][end-1]) == length(targetQubits)

    for single_cmd in cliffordCommandArray
        splitted = split(single_cmd, "(")
        splitted[2] = splitted[2][1:end-1] #Remove ")" symbol
        
        if splitted[1] == "hadamard"
            Juqst.hadamard(stateTableau, targetQubits[1])
        elseif splitted[1] == "phase"
            Juqst.phase(stateTableau, targetQubits[1])
        elseif splitted[1] == "cnot"
             Juqst.cnot(stateTableau, targetQubits[2], targetQubits[1])
        end
    end
    return deepcopy(stateTableau)
end

function applyClifford(stateTableau, cliffordCommandArray, tabMode=false)
    """
    Takes Tableau commands and applies them to Tableau::stateTableau
    Warning: Initlize needs to be excluded!
    """

    if tabMode
        cliffordCommandArray = cliffordCommandArray.commands[2:end] #Need to exclude first = initalize command
    end
    
    for single_cmd in cliffordCommandArray
        splitted = split(single_cmd, "(")
        splitted[2] = splitted[2][1:end-1] #Remove ")" symbol
        
        if splitted[1] == "hadamard"
            targetQubit = parse(Int, splitted[2])
            Juqst.hadamard(stateTableau, targetQubit)
        elseif splitted[1] == "phase"
            targetQubit = parse(Int, splitted[2])
            Juqst.phase(stateTableau, targetQubit)
        elseif splitted[1] == "cnot"
            involvedQubits = split(splitted[2],",") #Divide qubit definition 
            controlQubit, targetQubit = parse(Int, involvedQubits[1]), parse(Int, involvedQubits[2])
            Juqst.cnot(stateTableau, controlQubit, targetQubit)
        elseif splitted[1] == "initialize"
            println("Oh no, the initialization command has entered the command-list!")
            exit()
        end
    end
    return deepcopy(stateTableau)

end

function measureLocalClifford(state_mat, cliffordTableauArray, system_size, compOut=false,subsystem_dim=2)
    d = subsystem_dim
    
    clifford_mat = cliffordTableauArray[1]
    #@show adjoint(clifford_mat)*clifford_mat
    
    for i in 2:system_size
        clifford_mat = kron(clifford_mat, cliffordTableauArray[i])
    end
    clifford_mat = round.(clifford_mat,digits=10)
    rotatedState = clifford_mat*state_mat*adjoint(clifford_mat)
    if compOut
        return compBasisMeasurement(rotatedState) 
    else
        return adjoint(clifford_mat)*compBasisMeasurement(rotatedState)*clifford_mat
    end
end

function measureRandomClifford(stateTableau, cliffordTableau, system_size, compOut=false)
    """
    Measure a single clifford on a stateTableau
    """

    d = 2 #Default: 2, qu
    #Load the required gates to be applied on the stateTableau
    cliffordCommandArray = cliffordTableau.commands[2:end] #Need to exclude first = initalize command
    
    stateTableau = applyClifford(deepcopy(stateTableau), cliffordCommandArray)

    #Measure out all qubits:
    completeOutcomeString = []
    for qubit_i in 1:system_size
        single_qubit_outcome = measure(stateTableau, qubit_i)
        push!(completeOutcomeString, single_qubit_outcome)
    end

    #= WANT MORE EFFICIENT METHOD, OTHERWISE ACTIVATE THIS PART
    #Rotate back the state with the clifford unitary to obtain the stabilizer measurement outcome
    outcomeBaseVector = zeros(d^system_size)
    outcomeBaseVector[parse(Int,join(completeOutcomeString),base=d)+1] = 1
    #@show size(outcomeBaseVector)
    stabilizerOutcomeString = adjoint(round.(makeFromCommand(cliffordTableau), digits=10))*outcomeBaseVector
    return stabilizerOutcomeString
    =# 

    #outcomeInteger = parse(Int,join(completeOutcomeString),base=d)+1 #Encodes index of outcome > 1
    outcomeTableau = Tableau(system_size)
    @show completeOutcomeString, stateTableau, cliffordTableau
    for (pos,digit) in enumerate(completeOutcomeString)
        if digit == 1
            @show pos
            Juqst.X(outcomeTableau, pos) # digits(outcomeInteger, base=2)
            #Don't need to reverse since not converting between coord. vector and ket!
        end
    end

    if compOut
        return deepcopy(outcomeTableau)
    else
        rotatedOutcomeTableau = applyClifford(deepcopy(outcomeTableau), reverse_clifford(cliffordCommandArray, true))
        return rotatedOutcomeTableau
    end
end

function invert_clifford_tableau(CliffordOperator, system_size)
    """Alternative: Clifford tableau inversion USING QuantumClifford.jl"""
    #Take picture in X and Z basis of operator as stabilizer state:
    Zimage = apply!(one(Stabilizer, system_size), Cliffy)
    Ximage = apply!(one(Stabilizer, system_size, basis=:X), Cliffy)
    ImageOperators = []
    for Xim in 1:length(Ximage) 
        push!(ImageOperators, Ximage[Xim])
    end
    for Zim in 1:length(Zimage)
        push!(ImageOperators, Zimage[Zim])
    end
    ImageOperators = convert(AbstractArray{typeof(ImageOperators[1])}, ImageOperators)
    totalStab = Stabilizer(ImageOperators)
    @show totalStab
    canonicalize!(totalStab)
    varb = generate!(P"+ZIII", totalStab)
    @show varb
    current = varb[1]
    for op in ImageOperators[varb[2]]
        current *= op
    end
end

function alt_measureRandomClifford(destabState, CliffordOperator, system_size)
    d = subsystem_dim #Default: 2, qubits

    apply!(destabState, CliffordOperator)

    #Computational basis measurement:
    projectors = [single_z(system_size, i) for i in 1:system_size]
    resultArray = []
    for projector in projectors
        destabState, anticomindex, result = project!(destabState, projector) 
        if isnothing(result)
            destabState.phases[anticomindex] = rand([0x0,0x2])
            destabState, anticomindex, result = project!(destabState, projector) 
        end
        push!(resultArray, result)
    end

    apply!(destabState, invert_clifford_tableau())
end

=#


#=
function measurePauli(measurement, state, incomplete=false, subsystem_dim=2)
    if incomplete
        dim = size(state)[1]
        num_subsystems = Int(log(dim)/log(2))
        traceOutIndices = findall(x -> x=="I", measurement)
        @show measurement, traceOutIndices
        state = QuantumInformation.ptrace(state, [subsystem_dim for i in 1:num_subsystems],traceOutIndices)
        measurement = measurement[findall(x -> x!="I", measurement)]
    end
    @show "Success"
    totalRotation = rotationMatrices[measurement[1]]  #Identity matrix
    for (i, single_qubit_measurement) in enumerate(measurement[2:end])
        #@show i
        totalRotation = kron(totalRotation, rotationMatrices[single_qubit_measurement])
        #@show single_qubit_measurement, rotationMatrices["Z"]
    end
    stateCompBasis = totalRotation*state*adjoint(totalRotation)
    outcomeString  = compBasisMeasurement(stateCompBasis)  # #0/1 -> -1/1 
    #@show outcomeString, outcomeVector_compBasis
    #outcomeString = ((outcomeString.*2 .- 1) .* (-1))
    #outcomeVector = adjoint(totalRotation)*outcomeVector_compBasis
    #@show norm(outcomeVector)
    return outcomeString
end




#
# The following code is only used when we run this code through the command line interface
# WARNING: as the below raises errors when not calling it from the commandline, its deactivated for REPL use
#


#=
if abspath(PROGRAM_FILE) == @__FILE__
    function print_usage()
        println(stderr, "Usage: Unknown\n")
    end
    
   # if length(ARGS) != 3
        #print_usage()
    #end

    if ARGS[1] == "-pauli"

        subsystem_dim = 2

        thisRead = []
        stateArray = h5open("pureState.txt", "r") do fid
            this = fid["pureState"]
            @show this
            return read(this)
        end
        
        system_size, measurement_procedure = open("measurementscheme.txt", "r") do inIO
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

        @assert Int(log(size(stateArray))/log(subsystem_dim)) == system_size

        open("measurementOutcome.txt", "w") do outIO #Store outcomes along the way
            write(outIO, string(system_size))
            
            for measurement in measurement_procedure
                outcomeString = measurePauli(measurement, stateArray)

                printString = join([val for pair in zip(measurement, outcomeString) for val in pair], " ")
                println(printString)
                write(outIO, string("\n", printString))
            end
        end
    end
end

=#
=#
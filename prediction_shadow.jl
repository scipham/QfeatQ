
#using Juqst

pauliOpDict = Dict("I" => ITableau, "X" => XTableau, "Y" => YTableau, "Z" => ZTableau)
pauliOpStringDict = Dict(join(split(string(ITableau),"\n")[[2,end-1]]) => "I", join(split(string(XTableau),"\n")[[2,end-1]]) => "X", join(split(string(YTableau),"\n")[[2,end-1]]) => "Y", join(split(string(ZTableau),"\n")[[2,end-1]]) => "Z")

#Main program:
function predictCliffordLocalObservables(one_observable,cliffordOutcomes, cliffMeasScheme, system_size, subsystem_dim, k, tempi)
    """
    Returns a ComplexF64 of the prediction of one_observables from comp. basis measurement outcomes cliffordOutcomes after rotation by cliffMeasScheme

    Args
    - ....
    - mode::String = "g": global or "l": local
    """
    #@show "Now $one_observable"
    function cliff_pauli_exp(outcomeDigits, cliffTab, one_observable, system_size, subsystem_dim,tempi)

        #Can live conjugate SINGLE cliffords and paulis!
        #""..."" = applyClifftoCliff(full_left_tab, pauliOpDict[sqo[1]], [sqo[2]+1], true)
 
        updatedPauli = one_observable #Assuming to be of type PauliOperator

        #GLobal version:
        updatedPauli = deepcopy(conjugateClifftoPauli(deepcopy(cliffTab), deepcopy(updatedPauli), system_size, subsystem_dim))

        #Measure the expectation based on the measurement outcome of this measurement
        outcomeInteger = parse(Int,join(outcomeDigits), base=subsystem_dim)
        
        measPredict = pauliToMat(updatedPauli, system_size, subsystem_dim, true)[outcomeInteger+1, outcomeInteger+1]
        
        #= MORE EFFICIENT VERSION:
        splitHalf(x) = [x[1:Int(length(x)/2)], x[Int(length(x)/2)+1:end]]
        rightExpVector = mod.((outcomeDigits .+ splitHalf(updatedPauli.XZencoding)[1]), subsystem_dim) #Paper eq. 2
        leftExpVector = outcomeDigits
        #measPredict = dot(leftExpVector, rightExpVector)
        areEqual = all(rightExpVector .== leftExpVector)
        if areEqual
            measPredict = omegaPhase^(mod(transpose(splitHalf(updatedPauli.XZencoding)[2])*outcomeDigits,2*subsystem_dim)) * 1
        else 
            measPredict = nothing
        end
        =#

        return measPredict
    end

    cliffordExp = []
    for c_i in 1:length(cliffMeasScheme)
        tempExp = cliff_pauli_exp(cliffordOutcomes[c_i], cliffMeasScheme[c_i], one_observable,system_size, subsystem_dim, tempi)  # e.g. H X/Y/Z H = phase* X/Y/Z
        if tempExp != nothing
            push!(cliffordExp, tempExp)
        end
    end

    partition_size = Int(length(cliffordExp) / k)
    
    cliffordPrediction = median(real.((subsystem_dim^system_size +1)*[mean(cliffordExp[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k] .- prod([genPauliTraces[join(["Xd",one_observable.XZencoding[col], "Zd",one_observable.XZencoding[col+system_size]])] for col in 1:system_size])))   
    return cliffordPrediction
end

function predictLocCliffordLocalObservables(one_observable,cliffordOutcomes, cliffMeasScheme, system_size, subsystem_dim, k, tempi)
    function cliff_pauli_exp(outcomeDigits, pauliArray, one_observable, system_size, subsystem_dim,tempi)
        #Can live conjugate SINGLE cliffords and paulis!
        #""..."" = applyClifftoCliff(full_left_tab, pauliOpDict[sqo[1]], [sqo[2]+1], true)

        #= Local version
        for cmd in cliffTab.cmds 
            #phaseDict[join([cmd[1], sqo[1]])]
            tempCliffGate = deepcopy(genCliffordTab[cmd[1]])
            tempCliffGate.cmds[1][2] = cmd[2]
            tempGlobCliff = Tab(system_size, subsystem_dim)
            tempGlobCliff = applyClifftoCliff(tempCliffGate,deepcopy(tempGlobCliff), system_size, subsystem_dim)
            
            cctp = conjugateClifftoPauli(tempGlobCliff, updatedPauli, system_size, subsystem_dim)
            updatedPauli = deepcopy(cctp)
        end
        =#

        #= Old global version:
        #updatedPauli = deepcopy(conjugateClifftoPauli(invertCliffTab(deepcopy(cliffTab),system_size,subsystem_dim), updatedPauli, system_size, subsystem_dim))

        extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
        
        measPredict = omegaPhaseTilde^one_observable.phase
        for qi in 1:system_size
            singleQubitPauli = PauliOp(1, subsystem_dim, one_observable.XZencoding[[qi, system_size+qi]], 0)
            xpow, zpow = extractXZpow(pauliArray[qi])
            extraPhase = omegaPhaseTilde^genPauliRotations[pauliArray[qi]][1]

            updatedPauli = deepcopy(conjugateClifftoPauli(deepcopy(genPauliRotations[pauliArray[qi]][2]), singleQubitPauli, 1, subsystem_dim))
            measPredict *= pauliToMat(updatedPauli, 1, subsystem_dim)[outcomeDigits[qi]+1, outcomeDigits[qi]+1] #extraPhase*
            #=
            if tempi>27
                @show " "
                @show one_observable, updatedPauli, pauliArray[qi], genPauliRotations[pauliArray[qi]][2]
                show(stdout, "text/plain", pauliToMat(updatedPauli, 1, subsystem_dim))
                sleep(2)
            end
            =#
        end

        =#
        ###
        
        #Global version:
        extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
        
        measPredict = omegaPhaseTilde^one_observable.phase
        for qi in 1:system_size
            singleQubitPauli = PauliOp(1, subsystem_dim, one_observable.XZencoding[[qi, system_size+qi]], 0)
            xpow, zpow = extractXZpow(pauliArray[qi])
            extraPhase = omegaPhaseTilde^genPauliRotations[pauliArray[qi]][1]

            updatedPauli = deepcopy(conjugateClifftoPauli(deepcopy(genPauliRotations[pauliArray[qi]][2]), singleQubitPauli, 1, subsystem_dim))
            measPredict *= (subsystem_dim+1)*pauliToMat(updatedPauli, 1, subsystem_dim)[outcomeDigits[qi]+1, outcomeDigits[qi]+1] -  genPauliTraces[join(["Xd",one_observable.XZencoding[qi], "Zd",one_observable.XZencoding[qi+system_size]])] #extraPhase*

        end

        #Measure the expectation based on the measurement outcome of this measurement
        #outcomeInteger =parse(Int,join(outcomeDigits), base=subsystem_dim)

        return measPredict
    end

    partition_size = Int(length(cliffordOutcomes) / k)
    
    cliffordExp = [cliff_pauli_exp(cliffordOutcomes[c_i], cliffMeasScheme[c_i], one_observable,system_size, subsystem_dim, tempi) for c_i in 1:length(cliffMeasScheme)]  # e.g. H X/Y/Z H = phase* X/Y/Z

    cliffordPrediction = median(real.([mean(cliffordExp[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k]))
    
    return cliffordPrediction
end

function estimate_exp_direct(fullMeasurementScheme, fullMeasurementOutcomes, single_observable, system_size,subsystem_dim)
    """
    Pauli Measurements & (De)randomized classical shadows:
    Takes full measurement scheme with a single observable to predict its expectation
    """

    extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
    
    sum_product, cnt_match = 0, 0 #Initialize 
    for (mi, single_measurement) in enumerate(fullMeasurementScheme)
        not_match = false
        product = 1

        for position in 1:system_size #Iterate over each observable qubit-part 
            mxp, mzp = extractXZpow(single_measurement[position])

            if single_observable.XZencoding[[position, system_size+position]] == [0,0]
                continue
            elseif single_observable.XZencoding[[position, system_size+position]] != [mxp, mzp]  #Compatible? 
                not_match = true
                break
            elseif single_observable.XZencoding[[position, system_size+position]] == [mxp, mzp] 
                outDigit = fullMeasurementOutcomes[mi][position]
                product *= orderedPeigVals[[mxp,mzp]][outDigit+1]  #If compatible, "add" result to total expectation product
            end
        end

        if not_match
            continue
        end
        #If observable and measurement matched:
        sum_product += product  #Add single observable expectation for each comptible measurement together (later taken average with)
        cnt_match += 1  #Count how many measurements were compatible for taking expectation average (/normalize sum) over all compatible measurements
    end

    #Wipe memory to use less RAM:


    if cnt_match == 0
        return 0
    else
        #@show single_observable.phase
        return (omegaPhaseTilde^single_observable.phase)*sum_product/cnt_match #Expectation for observable
    end
end


function estimate_exp(full_measurement, one_observable)
    """
    Pauli Measurements & (De)randomized classical shadows:
    Takes full measurement scheme with a single observable to predict its expectation
    """
    sum_product, cnt_match = 0, 0 #Initialize 
    for single_measurement in full_measurement
        not_match = 0
        product = 1

        for (pauli_XYZ, position) in one_observable #Iterate over each observable qubit-part 
            if pauli_XYZ != single_measurement[position+1][1] #Check if compatible (identities, again, not included!)
                not_match = 1 
                break
            end
            product *= single_measurement[position+1][2]  #If compatible, "add" result to total expectation product
        end

        if not_match == 1
            continue
        end

        sum_product += product  #Add single observable expectation for each comptible measurement together (later taken average with)
        cnt_match += 1  #Count how many measurements were compatible for taking expectation average (/normalize sum) over all compatible measurements
    end
    return sum_product, cnt_match
end

function fidelityPrediction(input, output, mode) 
    """Predict fidelity between input and output state in array format"""
    F = nothing
    if mode == "m"
        @assert size(input) == size(output)
        F = tr(sqrt(sqrt(input) * output * adjoint(sqrt(input))))^2
        #@show F
        #return F^2
    elseif mode =="v"
        F = abs(adjoint(input)*output)^2
    elseif mode == "stab"
        println("Sorry, not yet implemented")
    end
    return F
end

function tab_innerProduct(leftTab, rightTab, system_size, subsystem_dim)
    """
    Returns a Float64 of the inner-product between leftTab and rightTab on measurement based method
    Source: Matthijs S.C. RIJLAARSDAM - Delft Master Thesis
    
    Args
    - ...
    - mode::String = "g": global or "l": local
    """
    
    leftUTab = invertCliffTab(deepcopy(leftTab), system_size, subsystem_dim)
    updatedRightTab = applyClifftoCliff(leftUTab, rightTab, system_size, subsystem_dim)
    overlap = 1

    #Measure overlap:

    U_loc = [0 0; 1 0]
    P_loc = mod.(U_loc - transpose(U_loc), subsystem_dim)
    doCommute(x,y) = Int(mod(transpose(x.XZencoding)*P_loc*y.XZencoding, subsystem_dim)) == 0 #Eq. 5 in paper 
    extractXZpow(x) = (parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2]) )
    
    function rowsum(t, h, i)
        resultPop = multiplyPauliOpsMod(PauliOp(system_size,subsystem_dim, t.XZencoding[:,h] ,t.phases[h]), PauliOp(system_size, subsystem_dim,t.XZencoding[:,i],t.phases[i]), 1, subsystem_dim)
        t.XZencoding[:,h] = resultPop.XZencoding
        t.phases[h] = resultPop.phase
        return t
    end
    for meas_qi in 1:system_size
        #Measure qubit i:
        measPauliOp = PauliOp(system_size, subsystem_dim, vcat(zeros(system_size+meas_qi-1),1,zeros(system_size-meas_qi)),0)
        locMeasPauliOp = PauliOp(1, subsystem_dim, [0,1],0)
      
        foundNone = true
        for col in 1+system_size:2*system_size #Go over all STABILIZERS
            tempStab = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],col],0)
            if doCommute(locMeasPauliOp, tempStab)
                continue
            else
                foundNone = false
                #-----------Get measurement outcome (GOTTESMAN STYLE): -------------
                locTempStab = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],col],0) 
                p=[0.5, 0.5] #Qubit
                #Qudits: Question is whether probabilities for d>2 really equal
                p=fill(1/subsystem_dim, subsystem_dim) 
                
                s_outcome = rand(Categorical(p)) - 1 #mod d digit of outcome 0:d-1
                overlap *= 1/sqrt(2)

                #-------Post-select state-tableau for next measurement: -----------------------
                #Call rowsum on all indices i \neq p, p = rowsum(i,p)
                for iterCol in 1:2*system_size #GO OVER (DE)STABILIZERS
                    Nelement = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],iterCol],0)
                    if doCommute(locMeasPauliOp, Nelement)
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
            #Measurement deterministic
            #Add some scratch space:
            #@show "Here, $meas_qi for "
            updatedRightTab.XZencoding = hcat(updatedRightTab.XZencoding, zeros(Int, 2*system_size))
            updatedRightTab.phases = vcat(updatedRightTab.phases, 0)

            #Call rowsum for i in 1:n : rowsum(2n +1  i+n)
            for col in 1:system_size
                tempStab = PauliOp(1, subsystem_dim, updatedRightTab.XZencoding[[meas_qi, system_size+meas_qi],col],0)
                if doCommute(locMeasPauliOp, tempStab)
                    continue
                else    
                    updatedRightTab = rowsum(updatedRightTab, 2*system_size+1, col+system_size) #MIND ORDER ROW <-> COL INTERCHANGED
                end
            end

            s_outcome = Int(updatedRightTab.phases[2*system_size+1] / 2) #Convert 0/d -> 1/2 -> hope that other don't exist
            if s_outcome != 0 
                overlap = 0
            end
            #Else keep whatever it is currrently
            #Remove Scratch space again:
            rotatedState.XZencoding = rotatedState.XZencoding[:, 1:end-1]
            rotatedState.phases = rotatedState.phases[1:end-1]
            #Deterministic = no further postselection required
        end
    end
    return overlap
end


# OLD QUBIT/ Juqst based:
#=
function predictCliffordLocalObservablesEff(one_observable, cliffordOutcomes, clifford_measurement_scheme, system_size, k)
    """
    Predicts Clifford of single Local Observables (*More* efficient!)
    """
    function cliff_pauli_exp(outcomeTab, cliffTab, one_observable, system_size)
        #@show "Now $one_observable"
        YCnt = 0 #For -i/2 factor correction
        #Sample the X/Y/Z qubits for each non-trivial in cliffTab 
        
        #Build observable tableau:
        #We need apply the cliffords anyway all at once for subsystem overlapping gates
        cliff_adjoint_cmds = reverse_clifford(deepcopy(cliffTab))
        full_right_tab = applyClifford(Tableau(system_size), cliff_adjoint_cmds ,false)  # = full Clifford from the right hand side! in C*P*C^dag = e^i*phi * P
        full_left_tab = deepcopy(cliffTab)

        prefactorCorrection = 1 #(-im/2)^YCnt

        #@show one_observable
        for sqo in one_observable
            #Per single non-trivial site on observable, apply:
            full_left_tab = applyCliffordLocal(full_left_tab, pauliOpDict[sqo[1]], [sqo[2]+1], true)
            
            if sqo[1] == "Y"
                #Juqst.X(full_right_tab, sqo[2]+1)
                #Juqst.Z(full_right_tab, sqo[2]+1)
                YCnt += 1
            elseif sqo[1] == "Z"
                #Juqst.Z(full_right_tab, sqo[2]+1)
            elseif sqo[1] == "X"
                #Juqst.X(full_right_tab, sqo[2]+1)
            end

            for gate in ["H", "P"]
                prefactorCorrection *= ((pauliCliffordTransformPhase[join([gate,sqo[1],gate])])^(getGateCounts(cliffTab, [sqo[2]+1])[gate]))
            end
            #CNOT handling seperate for multi-qubit:

            
            #=
            cqbit = findfirst(x-> x[2]==sqo[2], one_observable)
            if cqbit != nothing
                prefactorCorrection *= ((pauliCliffordTransformPhase[join(["CN",one_observable[cqbit][1],sqo[1],"CN"])])^( getGateCounts(cliffTab,[sqo[2],sqo[2]+1])["CN"]))
            elseif sqo[2] != 0
                prefactorCorrection *= ((pauliCliffordTransformPhase[join(["CN","I",sqo[2],"CN"])])^(getGateCounts(cliffTab, [sqo[2],sqo[2]+1])["CN"]))
            end
            =#
            for cqbit in 1:system_size
                CN_cnt = getGateCounts(cliffTab,[cqbit,sqo[2]+1])["CN"]
                if CN_cnt != 0 && CN_cnt != nothing
                    cqbit_idx = findfirst(x-> x[2]==cqbit-1, one_observable)
                    if cqbit_idx != nothing && cqbit != sqo[2]+1
                        prefactorCorrection *= ((pauliCliffordTransformPhase[join(["CN",one_observable[cqbit_idx][1],sqo[1],"CN"])])^( CN_cnt))
                    elseif cqbit != sqo[2]+1
                        prefactorCorrection *= ((pauliCliffordTransformPhase[join(["CN","I",sqo[1],"CN"])])^(CN_cnt))
                    end
                end
            end

        end

        #Correct Y-phase:
        prefactorCorrection *= (-im/2)^YCnt

        #@show "adjoint stab"
        #display(round.(makeFromCommand(full_right_tab), digits=10))
        #@show "adjoint end"
   
        #full_left_tab = deepcopy(cliffTab)
        full_transformed_tab = applyClifford(full_left_tab,full_right_tab, true)

        display(round.(makeFromCommand(full_transformed_tab)))

        #Optimal solution:
        #measureState = applyClifford(full_transformed_tab, deepcopy(outcomeTab),true)
        #prob = tableau_innerProduct(outcomeTab, measureState)
        
        #Brute force method:
        #Building up statistics:
        
        #result_array = zeros(2^system_size)
        #for stat_rep in 1:100*2^system_size
        #    temp_idx = sum(measure(deepcopy(full_transformed_tab),i)*2^(i-1) for i in Iterators.reverse(1:system_size))
        #    result_array[temp_idx+1] += 1
        #end
        #@show result_array/(100*2^system_size)

        #Experimental Alternative
        #@show split(string(outcomeTab)[2:end-1],"\n")
        #@show split(string(full_transformed_tab)[2:end-1],"\n")
        outcomeTabStr = split(string(outcomeTab)[2:end-1],"\n")
        full_transformed_tabStr = split(string(full_transformed_tab)[2:end-1],"\n")
        
        #@show split(kets(outcomeTab)[end-3-system_size+1:end-system_size], "")
        #parse(Int,split(kets(outcomeTab)[end-3-system_size+1:end-system_size], ""), base=2)
        outcomeKet = parse.(Int,split(kets(outcomeTab)[end-3-system_size+1:end-system_size], ""))
    

        totalExpectation = prefactorCorrection
        for qi in 1:system_size
            #outcome_destab = outcomeTabStr[1:Int(floor(length(outcomeTabStr)/2))][qi][[1; qi+1]]
            #outcome_stab = outcomeTabStr[Int(floor(length(outcomeTabStr)/2)+2):end][qi][[1; qi+1]]
            full_transformed_stab = full_transformed_tabStr[Int(floor(length(full_transformed_tabStr)/2)+2):end][qi][[1; qi+1]]
            full_transformed_destab = full_transformed_tabStr[1:Int(floor(length(full_transformed_tabStr)/2))][qi][[1; qi+1]]
            
            #pauliOp = PauliOperator(phaseDict[string(outcome_stab[1])], [if (string(outcome_stab[2])=="X") true else false end], [if (string(outcome_stab[2])=="Z") true else false end])
            
            part_state_vec = zeros(2)
            part_state_vec[outcomeKet[qi]+1] = 1
            #@show full_transformed_destab, full_transformed_stab,join([full_transformed_destab, full_transformed_stab])
            measurementOperator = pauliMapping[pauliOpStringDict[join([full_transformed_destab, full_transformed_stab])]] #As matrix
            single_qubit_expectation = adjoint(part_state_vec)*measurementOperator*part_state_vec
            #@show single_qubit_expectation
            totalExpectation *= single_qubit_expectation
        end

        #@show totalExpectation
        #cliffordTrue = round.(trueCliffordLocalObservables(outcomeTab, cliffTab, one_observable, system_size),digits=10)  # e.g. H X/Y/Z H = phase* X/Y/Z
        #@show cliffordTrue
        #display(prob)
        #applyClifford(outcomeTab, cliffTab, true)
        return totalExpectation #prob*(-im/2)^YCnt * 
    end

    #TESTBLOCK:
    #system_size = 3
    #Htab = randomClifford(system_size)
    #Htab = Tableau(system_size)
    #Juqst.hadamard(Htab, 1)
    #Juqst.cnot(Htab, [1,2]...)
    #@show Htab.commands

    #partial_observable = [("Z",0), ("Z",1), ("X",2)]
    #temp_state = randomStabilizerState(system_size)
    #cliff_pauli_exp(temp_state, deepcopy(Htab), partial_observable,system_size) 
    #partial_clifford = [("H", "0"), ("H", "1")]

    partition_size = Int(length(cliffordOutcomes) / k)
    
    full_one_observable = fill("I", system_size)
    for obs in one_observable
        full_one_observable[obs[2]+1] = obs[1]
    end

    cliffordTrue = [trueCliffordLocalObservables(cliffordOutcomes[c_i], clifford_measurement_scheme[c_i],one_observable,system_size) for c_i in 1:length(clifford_measurement_scheme)]  # e.g. H X/Y/Z H = phase* X/Y/Z
    truePrediction = median(real.((2^system_size +1)*[mean(cliffordTrue[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k] .- prod([pauliTraces[sqo] for sqo in full_one_observable])))
    #@show truePrediction 
    
    
    cliffordExp = [cliff_pauli_exp(cliffordOutcomes[c_i], clifford_measurement_scheme[c_i], one_observable,system_size) for c_i in 1:length(clifford_measurement_scheme)]  # e.g. H X/Y/Z H = phase* X/Y/Z
    cliffordPrediction = median(real.((2^system_size +1)*[mean(cliffordExp[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k] .- prod([pauliTraces[sqo] for sqo in full_one_observable])))
    return truePrediction, cliffordPrediction
end

function predictLocalCliffordLocalObservablesEff(one_observable, cliffordOutcomes, clifford_measurement_scheme, system_size, k)
    """
    Predicts Local Clifford of single Local Observables (*More* efficient!)
    """
    partition_size = Int(length(cliffordOutcomes) / k)
    


    full_one_observable = fill("I", system_size)
    for obs in one_observable
        full_one_observable[obs[2]+1] = obs[1]
    end

    cliffordExp = [localcliff_pauli_exp(cliffordOutcomes[c_i], clifford_measurement_scheme[c_i], one_observable,system_size) for c_i in 1:length(clifford_measurement_scheme)]  # e.g. H X/Y/Z H = phase* X/Y/Z
    cliffordPrediction = median(real.((2^system_size +1)*[sum(cliffordExp[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k] .- prod([pauliTraces[sqo] for sqo in full_one_observable])))
    return cliffordPrediction
end

function trueCliffordLocalObservables(temp_state, Htab, partial_observable,system_size)
    """
    temp_state = Clifford outcome state in tableau
    Htab = clifford unitary in tableau
    """

    temp_state_vec = makeFromCommand(temp_state)[:,1]

    full_one_observable = fill("I", system_size)
    for (pauli_XYZ, position) in partial_observable
        full_one_observable[convert(Int, position)+1] = pauli_XYZ
    end

    totalObservable = pauliMapping[full_one_observable[1]]
    for obs in full_one_observable[2:end]
        totalObservable = kron(totalObservable, pauliMapping[obs])
    end

    #full_one_clifford = fill("I", system_size)
    #for (clifford_HPCN, position) in partial_clifford
    #    full_one_clifford[parse(Int, position)+1] = clifford_HPCN
    #end

    #totalClifford = cliffordTransformations[full_one_observable[1]]
    #for cliff in full_one_clifford[2:end]
    #    totalObservable = kron(totalObservable, cliffordTransformations[cliff])
    #end
    totalClifford = round.(makeFromCommand(Htab), digits=10)

    compareM = totalClifford*totalObservable*adjoint(totalClifford)
    
    #@show "adjoint direct"
    #display(totalClifford*totalObservable)
    #@show "adjoint end"

    display(round.(compareM, digits=10))
    trueExp = (adjoint(temp_state_vec)*(compareM*temp_state_vec))
    #TESTBLOCK End
    return trueExp
end

function predictLocalCliffordLocalObservables(observableFile, clifford_measurement_scheme,cliffordOutcomes, system_size, k)
    d=2
    all_observables,weightArray,system_size = readObservables(observableFile)
    partition_size = k

    #@show cliffordOutcomes[2][1], cliffordOutcomes
    predictedExpectation = []

    @showprogress "Predicting Local Observables..." for (obs_i, one_observable) in enumerate(all_observables)

        one_full_observable = fill("I", system_size)
        for obs in one_observable 
            one_full_observable[obs[2]+1]=string(obs[1])
        end

        #And use Born's rule to evaluate the expectation values
        function compBVec(x)
            a = zeros(2)
            a[x+1]=1
            return a
        end

        expectationArray = []
        for (i,cl) in enumerate(clifford_measurement_scheme)
            cl_expectation = 1
            for qi in 1:system_size
                localRotatedVector = adjoint(cl[qi])*compBVec(cliffordOutcomes[i][qi])
                #@show pauliMapping[string(one_full_observable[qi][1])],localRotatedVector, tr((localRotatedVector*adjoint(localRotatedVector)))
                cl_expectation *= (adjoint(localRotatedVector)*pauliMapping[string(one_full_observable[qi][1])]*localRotatedVector)
            end
            push!(expectationArray, cl_expectation)
        end
        #@show expectationArray
        predictedExp = median(real.((2^system_size +1)*[mean(expectationArray[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k] .- prod([pauliTraces[string(sqo[1])] for sqo in one_full_observable])))            
        #println("For observable $obs_i : $predictedExp")
        push!(predictedExpectation, predictedExp)
    end
    return predictedExpectation
end

function predictCliffordLocalObservables(observableFile, cliffs, outcomes, system_size)
    """
    Predicts Clifford Local Observables (Inefficient!) from a reduced (sampled mean) set of classical shadows via meedians of means
    """
    observablesContent = readlines(observableFile)
    @assert system_size == parse(Int64, observablesContent[1])

    weightArray = []
    
    predictedExpectation = []

    for (obs_i, line) in enumerate(observablesContent[2:end])

        wBool = 0 #Does the provided file with observables include weights?
        if (length(split(line, " "))-1) % parse(Int64, split(line, " ")[1]) != 0
            wBool = 1
        end
        
        full_one_observable = fill("I", system_size)
        for (pauli_XYZ, position) in zip(split(line," ")[2:2:end-wBool], split(line," ")[3:2:end-wBool])
            full_one_observable[parse(Int, position)+1] = pauli_XYZ
        end
        #@show full_one_observable

        full_tensor_observable = pauliMapping[full_one_observable[1]]
        for obs in full_one_observable[2:end]
            full_tensor_observable = kron(full_tensor_observable, pauliMapping[obs])
        end
        
        if wBool == 1
            push!(weightArray, parse(Float64, split(line, " ")[end]))
        end
        
        #And use Born's rule to evaluate the expectation values
        
        
        expectationArray= [((adjoint(C)*outcomes[i])*(full_one_observable*(adjoint(outcomes[i])*C))) for (i,C) in enumerate(cliffs)]
        predictedExp = median(real.((2^system_size +1)*[mean(expectationArray[(p-1)*partition_size+1:(p)*partition_size]) for p in 1:k] .- prod([pauliTraces[sqo[1]] for sqo in full_observable])))            
        println("For observable $obs_i : $predictedExp")
        push!(predictedExpectation, predictedExp)
    end
    return predictedExpectation
end





function hammingDistance(n1, n2)
    x = n1 ⊻ n2 
    setBits = 0
  
    while x > 0
        setBits += x & 1
        x >>= 1
    end
    return setBits
end

#Need "observables working on qubit i"-array (indices)! observables_acting_on_ith_qubit
#Nested? i.e. position -> which of the three paulis -> individual count 
#observables_acting_on_ith_qubit


#------------ OLD --- NOT WORKING ----- ALPHA ---- Don't use

#=
#Initialize some helpfull shortcuts:
ITableau = Tableau(1)
XTableau = Tableau(1)
Juqst.X(XTableau, 1)
YTableau = Tableau(1)
Juqst.X(YTableau, 1)
Juqst.Z(YTableau, 1) #Need still to transfer factor -i/2!
ZTableau = Tableau(1)
Juqst.Z(ZTableau,1)

function alt_predict_renyi_2_entropy(full_measurement, all_subsystems)

    for (s,subsystem) in enumerate(all_subsystems)
        subsystem_size = length(subsystem)

        measurements_covered = []
        measurements_covered_outcomes = []
        unzip(a) = [getindex.(a, i) for i in 1:length(a[1])]
    
        for (m,single_measurement) in enumerate(full_measurement)
            #cnt = count(x-> x==single_measurement, full_measurement)
            sm_operator, sm_outcome = unzip(single_measurement)
            push!(measurements_covered, sm_operator)
            push!(measurements_covered_outcomes, sm_outcome)
        end

        unique_measurements = unique(measurements_covered)
        unique_m_bitstring_prob = []
 

        for (m, single_unique_measurement) in enumerate(unique_measurements)
            #@show m
            unique_idxs = findall(x-> x==single_unique_measurement, measurements_covered)
            
            N_M = length(unique_idxs)

            unique_m_bitstrings = []
            for i in unique_idxs
                ittobit = convert(Array{Int64},(measurements_covered_outcomes[i].+1)./2) #-/+1 -> 0,1
                push!(unique_m_bitstrings, [ittobit[subsystem[si]+1] for si in 1:subsystem_size])
            end
            
            bitstring_sum_m = zeros(2^subsystem_size)

            for umb in unique_m_bitstrings
                found = 0
                umb = string.(umb)
                pushfirst!(umb, "0b") #prepare binary-to-int conversion
                for b in 0:(2^(subsystem_size)-1)
                    if b == parse(Int64, join(umb))
                        bitstring_sum_m[b+1] += 1
                        found = 1
                    end
                end
                if found == 0
                    println("Oops, something goes wrong")
                end
            end

            bitstring_prob_m = bitstring_sum_m ./ length(unique_m_bitstrings)
            push!(unique_m_bitstring_prob, bitstring_prob_m)
        end

        totalProb = 0
        sum_terms_ij = []
        for i in 0:(2^(subsystem_size) -1)
            for j in 0:(2^(subsystem_size) -1)
                P_ij_mn_sum = 0
                for (m, m_prob_m) in enumerate(unique_m_bitstring_prob)
                    for (n, m_prob_n) in enumerate(unique_m_bitstring_prob)
                        #@show m,n
                        if m != n
                            P_ij_mn_sum += m_prob_m[i+1] * m_prob_n[j+1]
                        end
                    end
                end
                P_ij_mean = P_ij_mn_sum/(length(unique_m_bitstring_prob)*(length(unique_m_bitstring_prob)-1))
                totalProb += P_ij_mean
                push!(sum_terms_ij, (-2)^(-convert(Float64,hammingDistance(i,j))) * P_ij_mean)
                #@show i,j, P_ij_mn_sum/(length(unique_m_bitstring_prob)*(length(unique_m_bitstring_prob)-1))
            end
        end
        #@show totalProb
        trace_of_square = 2^subsystem_size * sum(sum_terms_ij)
        println(-1.0 * log2(min(max(trace_of_square, 1.0 / (2.0^subsystem_size)), 1.0 - 1e-9)))
    end
end

function alt2_predict_renyi_2_entropy(full_measurement, all_subsystems)

    for (s,subsystem) in enumerate(all_subsystems)
        subsystem_size = length(subsystem)

        measurements_covered = []
        measurements_covered_outcomes = []
        unzip(a) = [getindex.(a, i) for i in 1:length(a[1])]
    
        for (m,single_measurement) in enumerate(full_measurement)
            #cnt = count(x-> x==single_measurement, full_measurement)
            sm_operator, sm_outcome = unzip(single_measurement)
            push!(measurements_covered, [sm_operator[si+1] for si in 1:subsystem_size])
            push!(measurements_covered_outcomes, [sm_outcome[si+1] for si in 1:subsystem_size])
        end

        unique_measurements = unique(measurements_covered)
        unique_m_bitstring_prob = []
 

        for (m, single_unique_measurement) in enumerate(unique_measurements)
            unique_idxs = findall(x-> x==single_unique_measurement, measurements_covered)
            
            N_M = length(unique_idxs)

            unique_m_bitstrings = []
            for i in unique_idxs
                push!(unique_m_bitstrings, convert(Array{Int64},(measurements_covered_outcomes[i].+1)./2))
            end
            
            bitstring_sum_m = zeros(2^subsystem_size)

            for umb in unique_m_bitstrings
                found = 0
                umb = string.(umb)
                pushfirst!(umb, "0b") #prepare binary-to-int conversion
                for b in 0:(2^(subsystem_size)-1)
                    if b == parse(Int64, join(umb))
                        bitstring_sum_m[b+1] += 1
                        found = 1
                    end
                end
                if found == 0
                    println("Oops, something goes wrong")
                end
            end

            bitstring_prob_m = bitstring_sum_m ./ length(unique_m_bitstrings)
            push!(unique_m_bitstring_prob, bitstring_prob_m)
        end

        totalProb = 0
        sum_terms_ij = []
        for i in 0:(2^(subsystem_size) -1)
            for j in 0:(2^(subsystem_size) -1)
                P_ij_mn_sum = 0
                for (m, m_prob_m) in enumerate(unique_m_bitstring_prob)
                    for (n, m_prob_n) in enumerate(unique_m_bitstring_prob)
                        if m != n
                            P_ij_mn_sum += m_prob_m[i+1] * m_prob_n[j+1]
                        end
                    end
                end
                P_ij_mean = P_ij_mn_sum/(length(unique_m_bitstring_prob)*(length(unique_m_bitstring_prob)-1))
                totalProb += P_ij_mean
                push!(sum_terms_ij, (-2)^(-convert(Float64,hammingDistance(i,j))) * P_ij_mean)
                #@show i,j, P_ij_mn_sum/(length(unique_m_bitstring_prob)*(length(unique_m_bitstring_prob)-1))
            end
        end
        #@show totalProb
        trace_of_square = 2^subsystem_size * sum(sum_terms_ij)
        println(-1.0 * log2(min(max(trace_of_square, 1.0 / (2.0^subsystem_size)), 1.0 - 1e-9)))
    end
end

#Predicting Entanglement entropy: Experimental!
mapPauli = Dict("X" => 0, "Y"=> 1, "Z"=> 2)

function predict_renyi_2_entropy(full_measurement, all_subsystems)
    renyi_sum_of_binary_outcome = []
    renyi_number_of_outcomes = []

    for (s,subsystem) in enumerate(all_subsystems)
        subsystem_size = length(subsystem)

        renyi_sum_of_binary_outcome = zeros(2^(2*subsystem_size))
        renyi_number_of_outcomes = zeros(2^(2*subsystem_size))

        for (m, measurement) in enumerate(full_measurement)
            encoding = 0+1
            cumulative_outcome = 1
            
            renyi_sum_of_binary_outcome[1] += 1
            renyi_number_of_outcomes[1] += 1

            #Using gray code iteration over all 2^n possible outcomes
            for b in 1:(2^subsystem_size - 1) #Iterate over binary bit strings
                change_i = trailing_zeros(b)+1  #Look for next gray code flip bit from binary string
                index_in_original_system = subsystem[change_i]+1  #In the larger system, look up what this corresponds to
                cumulative_outcome *= measurement[index_in_original_system][2]  #[2] = outcome, record system qubit outcome for total
                #@show change_i
                #@show index_in_original_system
                #@show mapPauli[measurement[index_in_original_system][1]]
                #@show encoding
                encoding ⊻= (mapPauli[measurement[index_in_original_system][1]]+1) << (2*change_i)  #[1] = pauli X/Y/Z
                #Shifts Pauli operator number bitwise according to change_i occurence -> XOR with current bitstring = only change a single bit if operator AND shift equal to before = same encoding
                #@show encoding
                renyi_sum_of_binary_outcome[encoding] += cumulative_outcome #Record total expectation (sample) sum for operator on shift encoding
                renyi_number_of_outcomes[encoding] += 1 #For probability normalization, count encoding coincidences
            end
        end 

        level_cnt=zeros(2 * subsystem_size) #level_count
        level_ttl=zeros(2 * subsystem_size) #level_total

        for c in 0:(2^(2*subsystem_size) -1) #Iterate over binary bit-strings
            nonId = 1
            for i in 1:subsystem_size
                nonId += convert(Int64, ((c>>(2*i)) & 3) != 0)  #bool -> 0 or 1 
                #? Undo encoding for subsystem qubit (=change_i) and bitstring; check whether 3=Z is involved in this bitstring
                #-> if it is add -> nonId = count for how many subsystem qubits this holds for this bit string c
            end
            if renyi_number_of_outcomes[c+1] >= 2
                level_cnt[nonId] += 1 #Record for this bitstring if number of measurements is significant (>=2) and record for which number of qubit Z-pauli condition applied to
            end
            level_ttl[nonId] += 1 #Record how many bitstrings considered for this for (probability) normalization later
        end

        trace_of_square = 0 #Trace of square of subsystem density matrix in prediction
        for c in 0:(2^(2*subsystem_size) -1)
            if renyi_number_of_outcomes[c+1] <= 1
                continue  #Insignificant/insufficient amount of measurement recordings for this bitstring
            end
            nonId = 1
            for i in 1:subsystem_size
                nonId += convert(Int64, ((c>>(2*i)) & 3) != 0)  #bool -> 0 or 1 
                #see comment above
            end
           trace_of_square += 1.0/(renyi_number_of_outcomes[c+1] * (renyi_number_of_outcomes[c+1] - 1)) * (renyi_sum_of_binary_outcome[c+1] * renyi_sum_of_binary_outcome[c+1] - renyi_number_of_outcomes[c+1])/(2^subsystem_size) * level_ttl[nonId] / level_cnt[nonId]
           #@show renyi_sum_of_binary_outcome[c+1] 
           #@show renyi_number_of_outcomes[c+1] 
           #@show level_ttl[nonId] 
           #@show level_cnt[nonId]
        end

        println(-1.0 * log2(min(max(trace_of_square, 1.0 / (2.0^subsystem_size)), 1.0 - 1e-9)))
    end
end
=#


#----------------------------------------------------------------------------
#
# The following code is only used when we run this code through the command line interface
# WARNING: as the below raises errors when not calling it from the commandline, its deactivated for REPL use
#

#=
if abspath(PROGRAM_FILE) == @__FILE__
    function print_usage()
        print(stderr, "Usage:\n")
        print(stderr, "./prediction_shadow -o [measurement.txt] [observable.txt]")
        print(stderr, "    This option predicts the expectation of local observables.")
        print(stderr, "    We would output the predicted value for each local observable given in [observable.txt]")
    end
    @show ARGS
    
    if length(ARGS) != 3
        print_usage()
        print("\n")
        exit()
    end

    measurements = readlines(ARGS[2])
    system_size = parse(Int64, measurements[1])
    
    full_measurement = []
    for line in measurements[2:end]
        single_measurement = []
        for (pauli_XYZ, outcome) in zip(split(line, " ")[1:2:end], split(line, " ")[2:2:end])
            push!(single_measurement, (pauli_XYZ, parse(Int64, outcome)))
        end
        push!(full_measurement, single_measurement)
    end

    if ARGS[1]=="-o"
        observablesContent = readlines(ARGS[3])
        @assert system_size == parse(Int64, observablesContent[1])

        weightArray = []
        
        for (obs_i, line) in enumerate(observablesContent[2:end])
            one_observable = []

            wBool = 0 #Does the provided file include weights?
            if (length(split(line, " "))-1) % parse(Int64, split(line, " ")[1]) != 0
                wBool = 1
            end

            for (pauli_XYZ, position) in zip(split(line," ")[2:2:end-wBool], split(line," ")[3:2:end-wBool])
                push!(one_observable, (pauli_XYZ, parse(Int64, position)))
            end

            if wBool == 1
                push!(weightArray, parse(Float64, split(line, " ")[end]))
            end

            sum_product, cnt_match = estimate_exp(full_measurement, one_observable)
            if cnt_match > 0
                predictedExp = sum_product / cnt_match #Final expectation prediction: Sum of average outcome in each measurement -> normalize = average
            else
                println("WARNING: Observable $obs_i not measured once, we'll expect 0 for it!")
                predictedExp = 0
            end
            println("For observable $obs_i : $predictedExp")
        end
    elseif ARGS[1]=="-e"
        subsystemsContent = readlines(ARGS[3]) #Read lines of subsystems file
        
        @assert system_size == parse(Int64, subsystemsContent[1])
        
        all_subsystems = []
        subsystem_sizes = []

        for line in subsystemsContent[2:end]
            one_subsystem = []
            subsystem_size = parse(Int64, split(line, " ")[1])
            for (i, qubit_i) in enumerate(split(line," ")[2:end])
                push!(one_subsystem, parse(Int64, qubit_i))
            end
            push!(subsystem_sizes, subsystem_size)
            push!(all_subsystems, one_subsystem)
        end
        alt_predict_renyi_2_entropy(full_measurement, all_subsystems)
    else
        println(stderr, "Check you arguments! \n")
        print_usage()
        println(stderr, "\n")
        exit()
    end
end

=#
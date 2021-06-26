using Combinatorics
using LinearAlgebra
using IterTools
using ProgressMeter


function generateDW(observableFile,system_size, subsystem_dim) 
    """
    Generate and store an Pauli Expansion of a Dimensionality Witness
    """
    all_observables = []

    #Store the observables with weights with to file:
    open(observableFile, "w") do outIO
        write(outIO, join([string(system_size), " ", string(subsystem_dim+1)]))
        for single_observable in all_observables
            #println(single_observable)
            #Single line write:
            write(outIO, string("\n", single_observable))
        end 
    end
    @assert sum(clusterlengths) == length(all_observables) - firstsqrtindex + 1
    return firstsqrtindex, clusterlengths
end

function generateSpinEW(observableFile,system_size, subsystem_dim)
    """
    Generate and store an Pauli Expansion of a Simple Spin Entanglement Witness
    -> see paper ....
    """
    #>5
    all_observables = []
    YtoXZphase = im #Need Phase correction in X,Z generation!

    tempPauliDic = Dict("It"=>[[0,0],0],"Xt"=>[[1,0], 0], "Yt"=> [[1,1], 0], "Zt" => [[0,1], 0])
    
    Jx = unique(permutations(vcat(["Xt"], ["It" for i in 1:system_size -1])))
    Jy = unique(permutations(vcat(["Yt"], ["It" for i in 1:system_size -1])))
    for J_set in [Jx, Jy]
        for (i, permi1) in enumerate(J_set)
            for (j,permi2) in enumerate(J_set)
                
                single_q_obs = []
                cumPhase = 0 #As power of omegaPhaseTilde!!
                cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase
                
                permi = nothing
                doubleCnt = 1
                if i==j && J_set == Jy
                    cumPhase = 2
                    doubleCnt = 2
                    permi = permi1
                elseif i==j
                    doubleCnt = 2
                    permi = permi1
                else 
                    doubleCnt = 1
                    permi = permi1
                    permi[j] = permi2[j]
                end

                for (qi, qi_obsStr) in enumerate(permi)
                    xpow, zpow = doubleCnt*tempPauliDic[qi_obsStr][1][1], doubleCnt*tempPauliDic[qi_obsStr][1][2]
                    push!(single_q_obs, join([qi,"|","X",xpow,"Z", zpow]))
                    cumPhase += tempPauliDic[qi_obsStr][2]
                end
                
                #Take Y phases into account:
                Ycnt = count(x-> x== "Yt", permi)    
                cumWeight_complex *= (YtoXZphase^(Ycnt*2)) #If there is a Y, there are 2 of them


                totalTermWeightFactor = 1.0/4 #Formal definition of the total phase of this term
                cumWeight_complex *= totalTermWeightFactor

                phaseInfoStr = join([real(cumWeight_complex), imag(cumWeight_complex) ,cumPhase], " ")

                one_line = join([phaseInfoStr, single_q_obs...]," ")
                push!(all_observables, one_line)
            end
        end
    end
    #Store the observables with weights with to file:
    open(observableFile, "w") do outIO
        write(outIO, join([string(system_size), " ", string(subsystem_dim+1)]))
        for single_observable in all_observables
            #println(single_observable)
            #Single line write:
            write(outIO, string("\n", single_observable))
        end 
    end
end


function generateEW(observableFile, excitations=2,system_size=4, subsystem_dim=2) 
    """
    Generate and store an Pauli Expansion of .... Entanglement Witness
    e.g. see paper Huber et al (2016)
    Fixed to qubits i.e. subsystem_dim = 2
    """
    m,n = excitations,system_size

    YtoXZphase = im #Need Phase correction in X,Z generation!

    tempPauliDic = Dict("It"=>[[0,0],0],"Xt"=>[[1,0], 0], "Yt"=> [[1,1], 1], "Zt" => [[0,1], 0])

    all_observables = []

    #Term 1:
    for permi in permutations(["It", "It", "ph_i", "ph_i"]) #Should I include non-unique permutations?
        for i in ["Xt", "Yt"]
            permiCopy = deepcopy(permi)
            replace!(permiCopy, "ph_i"=>i)

            single_q_obs = []
            cumPhase = 0 #As power of omegaPhaseTilde!!
            cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase

            for (qi, qi_obsStr) in enumerate(permiCopy)
                xpow, zpow = tempPauliDic[qi_obsStr][1][1], tempPauliDic[qi_obsStr][1][2]
                push!(single_q_obs, join([qi,"|","X",xpow,"Z", zpow]))
                cumPhase += tempPauliDic[qi_obsStr][2]
            end

            #Take Y phases into account:
            #if i == "Yt"
            #    cumWeight_complex *= (YtoXZphase^2) #If there is a Y, there are 2 of them
            #end

            totalTermWeightFactor = 1.0 #Formal definition of the total phase of this term
            cumWeight_complex *= totalTermWeightFactor

            phaseInfoStr = join([real(cumWeight_complex), imag(cumWeight_complex) ,cumPhase], " ")

            one_line = join([phaseInfoStr, single_q_obs...]," ")
            push!(all_observables, one_line)
        end
    end

    #Term 2:
    for permi in permutations(["ph_i", "ph_i", "Zt", "Zt"]) ### #Should I include non-unique permutations?
        for i in ["Xt", "Yt"]
            permiCopy = deepcopy(permi)
            replace!(permiCopy, "ph_i"=>i)

            single_q_obs = []
            cumPhase = 0 #As power of omegaPhaseTilde!!
            cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase

            for (qi, qi_obsStr) in enumerate(permiCopy)
                xpow, zpow = tempPauliDic[qi_obsStr][1][1], tempPauliDic[qi_obsStr][1][2]
                push!(single_q_obs, join([qi,"|","X",xpow,"Z", zpow]))
                cumPhase += tempPauliDic[qi_obsStr][2]
            end

            #Take Y phases into account:
            #if i == "Yt"
            #    cumWeight_complex *= (YtoXZphase^2) #If there is a Y, there are 2 of them
            #end

            totalTermWeightFactor = -1.0 #Formal definition of the total phase of this term
            cumWeight_complex *= totalTermWeightFactor

            phaseInfoStr = join([real(cumWeight_complex), imag(cumWeight_complex) ,cumPhase], " ")

            one_line = join([phaseInfoStr, single_q_obs...]," ")
            push!(all_observables, one_line)
        end
    end

    #Term 3: (Constant)
    push!(all_observables, join([join([real(-3.0), imag(0.0) ,0], " "), join([1,"|","X",0,"Z", 0])]," "))
    
    #Term 4: 
    for iteri in 1:1
        single_q_obs = []
        for (qi, qi_obsStr) in enumerate(fill("Zt",4))
            xpow, zpow = tempPauliDic[qi_obsStr][1][1], tempPauliDic[qi_obsStr][1][2]
            push!(single_q_obs, join([qi,"|","X",xpow,"Z", zpow]))
        end
        phaseInfoStr = join([real(-3.0), imag(0.0) ,0], " ")

        one_line = join([phaseInfoStr, single_q_obs...]," ")
        push!(all_observables, one_line)
    end

    #Term 5:
    for permi in permutations(["It", "It", "Zt", "Zt"]) ### #Should I include non-unique permutations?

        single_q_obs = []
        cumPhase = 0 #As power of omegaPhaseTilde!!
        cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase

        for (qi, qi_obsStr) in enumerate(permi)
            xpow, zpow = tempPauliDic[qi_obsStr][1][1], tempPauliDic[qi_obsStr][1][2]
            push!(single_q_obs, join([qi,"|","X",xpow,"Z", zpow]))
            cumPhase += tempPauliDic[qi_obsStr][2]
            #Take Y phases into account:
            #if qi_obsStr == "Yt"
            #    cumWeight_complex *= YtoXZphase
            #end
        end

        totalTermWeightFactor = 1.0 #Formal definition of the total phase of this term
        cumWeight_complex *= totalTermWeightFactor

        phaseInfoStr = join([real(cumWeight_complex), imag(cumWeight_complex) ,cumPhase], " ")

        one_line = join([phaseInfoStr, single_q_obs...]," ")
        push!(all_observables, one_line)
    end

    #Protocol point from where sum of pairswise sqrt must be taken during Post-Processing
    firstsqrtindex = length(all_observables)+1

    #Term 6:
    clusterlengths = []
    omegaPhase = exp(im*2*pi / subsystem_dim) #Overall phase in e.g. commutation of generalized paulis
    genPauliMat = Dict( "It"=>convert(Array{ComplexF64}, round.(vcat([transpose([if row==col 1 else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10)),
                        "Xt"=>convert(Array{ComplexF64},round.(vcat([transpose([if mod(row,subsystem_dim)==mod((col+1),subsystem_dim) 1 else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10)),
                        "Zt"=>convert(Array{ComplexF64},round.(vcat([transpose([if mod(row,subsystem_dim)==mod(col,subsystem_dim) omegaPhase^(row-1) else 0 end for col in 1:subsystem_dim]) for row in 1:subsystem_dim]...), digits=10))) 
    
    ZIcombinations = unique(collect(Iterators.flatten([collect(permutations(combi)) for combi in with_replacement_combinations(["It", "Zt"],4)])))
    #show(stdout, "text/plain", ZIcombinations)
    ZIcombinationsMat = [kron([genPauliMat[sqp] for sqp in combi]...) for combi in ZIcombinations]
    
    pauliTupleDic = Dict("It" => (0,0), "Zt" => (0,1), "Xt" => (1,0), "Yt" =>(1,1))
    matElement2PauliDic = Dict((0,0) => [["It", 1/2],["Zt", 1/2]] , (0,1)=>[["Xt", 1/2],["Yt", 1/2*im]] , (1,0) =>[["Xt", 1/2],["Yt", -1/2*im]], (1,1) => [["It", 1/2],["Zt", -1/2]])
    #matElement2PauliDic = Dict((0,0) => [["It", 1/d],["Zt", 1/d]] , (0,1)=>[["Xt", 1/2],["Yt", 1/2*im]] , (1,0) =>[["Xt", 1/2],["Yt", -1/2*im]], (1,1) => [["It", 1/2],["Zt", -1/2]])

    function matElement2paulisEff(matElementLeft, matElementRight)
        matElements = collect(zip(matElementLeft, matElementRight))
        combiProductIndices = vec(collect(IterTools.product([0:1 for i in 1:4]...)))  #Indices for working out all terms of (X+Y)(Z+Q)(V+W)
        #@show matElement2PauliDic[matElements[1]]
        combiCoeff = [prod([matElement2PauliDic[matElements[i]][idx+1][2] for (i, idx) in enumerate(idx_str)]) for idx_str in combiProductIndices]
        combiStrings = [[matElement2PauliDic[matElements[i]][idx+1][1] for (i, idx) in enumerate(idx_str)] for idx_str in combiProductIndices]
-       
        return zip(combiStrings, combiCoeff)
    end

    function matElement2paulis(matElementProjector)
        combiCoeff = [tr(combiMat*matElementProjector) for combiMat in ZIcombinationsMat]
        nonzeroCombis = findall(x->round(x,digits=6)!=0.0,combiCoeff)
        @show combiCoeff[nonzeroCombis]
        return zip(ZIcombinations[nonzeroCombis], combiCoeff[nonzeroCombis])
    end


    alphaSets = combinations(1:4, 2) #Create all 2(=excit.)-sets of indices in range 1:system_size=4
    gammaPairs = reduce(vcat, collect(IterTools.product(collect(alphaSets),collect(alphaSets))))
    filter!(x-> count(y -> y in x[2] ,x[1]) ==1,gammaPairs) #gamma = get all pairs of indexsets with overlap only in a single element

    for gammaPair in gammaPairs
        leftIndices, rightIndices = zeros(Int, system_size), zeros(Int, system_size)
        leftIndices[gammaPair[1]] .= 1
        rightIndices[gammaPair[2]] .= 1
        #@show gammaPair, vcat(gammaPair[1], gammaPair[2]), collect(2:2:2*(system_size))
        
        #TODO: Finish Implementation:
        combiPairs = hcat(leftIndices, rightIndices)
        shiftedPairs = deepcopy(combiPairs)
        shiftedPairs[Bool.(leftIndices),:] = circshift(combiPairs[Bool.(leftIndices),:],(0,1))
        leftPermKet, rightPermKet = shiftedPairs[:,1], shiftedPairs[:,2]
        #=
        allPairs = [[vcat(leftIndices, rightIndices)[i-1],vcat(leftIndices,rightIndices)[i]] for i in 2:2:2*(system_size)]
        shiftedPairs = circshift(allPairs[Bool.(leftIndices)],1)
        allPairs[Bool.(leftIndices)] = shiftedPairs
        leftPermKet, rightPermKet = vcat(allPairs...)[1:length(allPairs)], vcat(allPairs...)[length(allPairs)+1:end] 
        =#

        sExcitVec, sGroundVec = [0,1], [1,0]
        leftCompBasisVec, rightCompBasisVec = normalize(kron([b*sExcitVec + abs(b-1)*sGroundVec for b in leftPermKet]...)), normalize(kron([b*sExcitVec + abs(b-1)*sGroundVec for b in rightPermKet]...))
        #@show gammaPair, leftCompBasisVec
        #sleep(2)
        
        #leftDecomp = matElement2paulis(leftCompBasisVec *adjoint(leftCompBasisVec))
        #rightDecomp = matElement2paulis(rightCompBasisVec *adjoint(rightCompBasisVec))
        
        leftDecomp = matElement2paulisEff(leftPermKet, leftPermKet)
        rightDecomp = matElement2paulisEff(rightPermKet, rightPermKet)


        push!(clusterlengths, length(rightDecomp))
        
        push!(clusterlengths, length(leftDecomp))

        for term in rightDecomp
            single_q_right_obs = []
            right_cumPhase = 0
            right_cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase
            totalRightTermWeightFactor = 1.0


            for (qi_r, qi_r_obsStr) in enumerate(term[1])
                xpow, zpow = tempPauliDic[qi_r_obsStr][1][1], tempPauliDic[qi_r_obsStr][1][2]
                push!(single_q_right_obs, join([qi_r,"|","X",xpow,"Z", zpow]))
            end
            totalRightTermWeightFactor *= term[2]

            right_cumWeight_complex *= totalRightTermWeightFactor

            rightPhaseInfoStr = join([real(right_cumWeight_complex), imag(right_cumWeight_complex) ,Int(right_cumPhase)], " ")
            one_right_line = join([rightPhaseInfoStr, single_q_right_obs...]," ")
            push!(all_observables, one_right_line)
        end
        
        for term in leftDecomp
            #@show (term)
            #Save decomposition to lines:
            single_q_left_obs = []

            left_cumPhase = 0 #As power of omegaPhaseTilde!!

            left_cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase

            totalLeftTermWeightFactor = 1.0 #Formal definition of the total phase of this term

            for (qi_l, qi_l_obsStr) in enumerate(term[1]) 
                xpow, zpow = tempPauliDic[qi_l_obsStr][1][1], tempPauliDic[qi_l_obsStr][1][2]
                push!(single_q_left_obs, join([qi_l,"|","X",xpow,"Z", zpow]))
            end
            totalLeftTermWeightFactor *= term[2]

            left_cumWeight_complex *= totalLeftTermWeightFactor

            leftPhaseInfoStr = join([real(left_cumWeight_complex), imag(left_cumWeight_complex) ,Int(left_cumPhase)], " ")

            one_left_line = join([leftPhaseInfoStr, single_q_left_obs...]," ")
            push!(all_observables, one_left_line)

            #NOTE TO TAKE *PAIRWISE* SQUARE ROOT LATER IN PREDICTION PHASE!! -> TODO: keep track which!!
        end
    end
    #Store the observables with weights with to file:
    open(observableFile, "w") do outIO
        write(outIO, join([string(system_size), " ", string(subsystem_dim+1)]))
        for single_observable in all_observables
            #println(single_observable)
            #Single line write:
            write(outIO, string("\n", single_observable))
        end 
    end
    @assert sum(clusterlengths) == length(all_observables) - firstsqrtindex + 1
    return firstsqrtindex, clusterlengths #Specify from which index on we need to take the pairwise sqaure root to obtain our result.
end

function reduceMultiplets(inputArray,inverse=false, multipletsDic=nothing, sortArray=nothing)
    if inverse==false
        multipletsDic = Dict()
        reducedArray = []
        for (el_i, el) in enumerate(inputArray)
            if el in collect(keys(multipletsDic))
                push!(multipletsDic[el], el_i)
            else
                push!(reducedArray, el)
                multipletsDic[el] = [el_i]
            end
        end
        return reducedArray, multipletsDic
    elseif inverse
        reconstructedArray = fill(0.0+0.0*im, sum(length.(collect(values(multipletsDic)))))
        for (key, value) in multipletsDic
            if sortArray != nothing
                #@show typeof(reconstructedArray), typeof(inputArray), value, key, typeof(sortArray), findfirst(x->x==key, sortArray)
                reconstructedArray[value] .= inputArray[findfirst(x->x==key, sortArray)]
            else
                reconstructedArray[value] .= key
            end
        end
        @assert !(nothing in reconstructedArray)  
        return reconstructedArray
    end
end

function generateEW_cor(observableFile, excitations,system_size, subsystem_dim, extObservableFile) 
    """
    Generate and store an Pauli Expansion of .... Entanglement Witness
    e.g. see paper Huber et al (2016)
    """
    m,n = excitations,system_size

    all_observables = []

    genKetExpansPhasesInverse = Dict((b,c) => [PauliOp(1, subsystem_dim, mod.([b-c, k], subsystem_dim), mod(2*(-k*c), 2*subsystem_dim)) for k in 0:subsystem_dim-1] for b in 0:subsystem_dim-1 for c in 0:subsystem_dim -1) #Powers of omegaPhase of measurement operator : Inverse fourier tranform (ket{k}}]) sum_k omegaPhase^{-b*k}
    #\ket{b} \bra{c} = ...q > ADD NORMALIZING 1/d MANUALLY

    function matElement2paulisEff(matElementLeft, matElementRight)
        matElements = collect(zip(matElementLeft, matElementRight))
        combiProductIndices = vec(collect(IterTools.product([0:(subsystem_dim-1) for i in 1:system_size]...)))  #Indices for working out all terms of (X+Y)(Z+Q)(V+W)
        #@show matElement2PauliDic[matElements[1]]
        combiPhases = [sum([genKetExpansPhasesInverse[matElements[i]][idx+1].phase for (i, idx) in enumerate(idx_str)]) for idx_str in combiProductIndices]
        combiStrings = [[genKetExpansPhasesInverse[matElements[i]][idx+1].XZencoding for (i, idx) in enumerate(idx_str)] for idx_str in combiProductIndices]
    
        return zip(combiStrings, combiPhases)
    end

    alphaSets = combinations(1:system_size, m) #Create all 2(=excit.)-sets of indices in range 1:system_size=4
    
    # -------- Alpha sum -- /--- D ------ terms
    N_D =  m*(system_size-m-1) #m*binomial(system_size-m-1, m-1)
    for alphaSet in alphaSets
        alphaBitArray = zeros(Int, system_size)
        alphaBitArray[alphaSet] .= 1
        matElementAlpha_decomp = matElement2paulisEff(alphaBitArray, alphaBitArray)
        for term in matElementAlpha_decomp
            single_q_c_obs = []
            c_cumPhase = 0
            c_cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase
            totalcTermWeightFactor = -1.0*N_D/subsystem_dim


            for (qi_c, qi_c_xz) in enumerate(term[1])
                xpow, zpow = qi_c_xz
                push!(single_q_c_obs, join([qi_c,"|","X",xpow,"Z", zpow]))
            end
            c_cumPhase += term[2]

            c_cumWeight_complex *= totalcTermWeightFactor

            cPhaseInfoStr = join([real(c_cumWeight_complex), imag(c_cumWeight_complex) ,Int(c_cumPhase)], " ")
            one_c_line = join([cPhaseInfoStr, single_q_c_obs...]," ")
            push!(all_observables, one_c_line)
        end
    end
    
    #-------- Gamma sum ------------------
    gammaPairs = reduce(vcat, collect(IterTools.product(collect(alphaSets),collect(alphaSets))))
    filter!(x-> count(y -> y in x[2] ,x[1]) ==m-1,gammaPairs) #gamma = get all pairs of indexsets with overlap only in a single element

    
    #------------ O ------ terms    
    for gammaPair in gammaPairs
        leftIndices, rightIndices = zeros(Int, system_size), zeros(Int, system_size)
        leftIndices[gammaPair[1]] .= 1 #alpha
        rightIndices[gammaPair[2]] .= 1 #beta

        matElementAlphaBeta_decomp = matElement2paulisEff(leftIndices, rightIndices)
        
        for term in matElementAlphaBeta_decomp #AlphaBeta / cross-terms
            single_q_c_obs = []
            c_cumPhase = 0
            c_cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase
            totalcTermWeightFactor = 1.0/subsystem_dim


            for (qi_c, qi_c_xz) in enumerate(term[1])
                xpow, zpow = qi_c_xz
                push!(single_q_c_obs, join([qi_c,"|","X",xpow,"Z", zpow]))
            end

            c_cumPhase += term[2]

            c_cumWeight_complex *= totalcTermWeightFactor

            cPhaseInfoStr = join([real(c_cumWeight_complex), imag(c_cumWeight_complex) ,Int(c_cumPhase)], " ")
            one_c_line = join([cPhaseInfoStr, single_q_c_obs...]," ")
            push!(all_observables, one_c_line)
        end
    end

    #Protocol point from where sum of pairswise sqrt must be taken during Post-Processing
    firstsqrtindex = length(all_observables)+1

    #Term 6:
    clusterlengths = []

     #------------ sqrt(P) ----- terms
    for gammaPair in gammaPairs
        leftIndices, rightIndices = zeros(Int, system_size), zeros(Int, system_size)
        leftIndices[gammaPair[1]] .= 1 #alpha
        rightIndices[gammaPair[2]] .= 1 #beta

       
        combiPairs = hcat(leftIndices, rightIndices)
        shiftedPairs = deepcopy(combiPairs)
        shiftedPairs[Bool.(leftIndices),:] = circshift(combiPairs[Bool.(leftIndices),:],(0,1))
        leftPermKet, rightPermKet = shiftedPairs[:,1], shiftedPairs[:,2]
        #=
        allPairs = [[vcat(leftIndices, rightIndices)[i-1],vcat(leftIndices,rightIndices)[i]] for i in 2:2:2*(system_size)]
        shiftedPairs = circshift(allPairs[Bool.(leftIndices)],1)
        allPairs[Bool.(leftIndices)] = shiftedPairs
        leftPermKet, rightPermKet = vcat(allPairs...)[1:length(allPairs)], vcat(allPairs...)[length(allPairs)+1:end] 
        =#

        sExcitVec, sGroundVec = [0,1], [1,0]
        leftCompBasisVec, rightCompBasisVec = normalize(kron([b*sExcitVec + abs(b-1)*sGroundVec for b in leftPermKet]...)), normalize(kron([b*sExcitVec + abs(b-1)*sGroundVec for b in rightPermKet]...))
        #@show gammaPair, leftCompBasisVec
        #sleep(2)
        
        #leftDecomp = matElement2paulis(leftCompBasisVec *adjoint(leftCompBasisVec))
        #rightDecomp = matElement2paulis(rightCompBasisVec *adjoint(rightCompBasisVec))
        
        leftDecomp = matElement2paulisEff(leftPermKet, leftPermKet)
        rightDecomp = matElement2paulisEff(rightPermKet, rightPermKet)


        push!(clusterlengths, length(rightDecomp))
        
        push!(clusterlengths, length(leftDecomp))

        for term in rightDecomp
            single_q_right_obs = []
            right_cumPhase = 0
            right_cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase
            totalRightTermWeightFactor = 1.0/subsystem_dim


            for (qi_r, qi_r_xz) in enumerate(term[1])
                xpow, zpow = qi_r_xz
                push!(single_q_right_obs, join([qi_r,"|","X",xpow,"Z", zpow]))
            end
            right_cumPhase += term[2]

            right_cumWeight_complex *= totalRightTermWeightFactor

            rightPhaseInfoStr = join([real(right_cumWeight_complex), imag(right_cumWeight_complex) ,Int(right_cumPhase)], " ")
            one_right_line = join([rightPhaseInfoStr, single_q_right_obs...]," ")
            push!(all_observables, one_right_line)
        end
        
        for term in leftDecomp
            #@show (term)
            #Save decomposition to lines:
            single_q_left_obs = []

            left_cumPhase = 0 #As power of omegaPhaseTilde!!

            left_cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase

            totalLeftTermWeightFactor = 1.0/subsystem_dim #Formal definition of the total phase of this term

            for (qi_l, qi_l_xz) in enumerate(term[1]) 
                xpow, zpow = qi_l_xz
                push!(single_q_left_obs, join([qi_l,"|","X",xpow,"Z", zpow]))
            end
            left_cumPhase += term[2]

            left_cumWeight_complex *= totalLeftTermWeightFactor

            leftPhaseInfoStr = join([real(left_cumWeight_complex), imag(left_cumWeight_complex) ,Int(left_cumPhase)], " ")

            one_left_line = join([leftPhaseInfoStr, single_q_left_obs...]," ")
            push!(all_observables, one_left_line)

            #NOTE TO TAKE *PAIRWISE* SQUARE ROOT LATER IN PREDICTION PHASE!! -> TODO: keep track which!!
        end
    end

    @assert sum(clusterlengths) == length(all_observables) - firstsqrtindex + 1
    #=
    if extObservableFile != nothing
        open(extObservableFile, "w") do outIO
            write(outIO, join([string(system_size), " ", string(subsystem_dim)]))
            for single_observable in all_observables
                #println(single_observable)
                #Single line write:
                write(outIO, string("\n", single_observable))
            end 
        end
        reducedObservables, mDic = reduceMultiplets(all_observables)
        all_observables = reducedObservables
    end
    =#

    #Store the observables with weights with to file:
    open(observableFile, "w") do outIO
        write(outIO, join([string(system_size), " ", string(subsystem_dim)]))
        for single_observable in all_observables
            #println(single_observable)
            #Single line write:
            write(outIO, string("\n", single_observable))
        end 
    end
    return firstsqrtindex, clusterlengths, nothing, nothing#, reducedObservables, mDic #Specify from which index on we need to take the pairwise sqaure root to obtain our result.
end

#@show generateEW("./observables_entanglementWitness3d.txt", 2,4)

function generateCompleteRandomObservables(num_observables, system_size, subsystem_dim, observableFile, maxWeight)
    tempPauliDic = Dict("It"=>[[0,0],0],"Xt"=>[[1,0], 0], "Yt"=> [[1,1], 0], "Zt" => [[0,1], 0])

    all_observables = []

    @showprogress "Sampling random observables..." for obsi in 1:num_observables
        single_q_obs = []
        cumPhase = 0 #As power of omegaPhaseTilde!!
        cumWeight_complex = 1.0+0.0*im #Any phase/prefactor, not covered by cumPhase
        cnt_weight = 0

        for qi in 1:system_size
            #rand_obsStr = rand(collect(keys(tempPauliDic)))
            rand_obsStr = nothing
            if cnt_weight < maxWeight
                rand_obsStr = rand(["It", "It", "It", "Xt", "Yt", "Zt"])
            else
                rand_obsStr = "It"
            end
            xpow, zpow = tempPauliDic[rand_obsStr][1][1], tempPauliDic[rand_obsStr][1][2]
            push!(single_q_obs, join([qi,"|","X",xpow,"Z", zpow]))
            cumPhase += tempPauliDic[rand_obsStr][2]
            if rand_obsStr != "It"
                cnt_weight += 1
            end
        end


        totalTermWeightFactor = 1.0 #Formal definition of the total phase of this term
        cumWeight_complex *= totalTermWeightFactor
        cumWeight_complex *= rand([1,1,1,rand(0.0:0.1:1.0)])*rand([1,1,im])

        phaseInfoStr = join([real(cumWeight_complex), imag(cumWeight_complex) ,cumPhase], " ")

        one_line = join([phaseInfoStr, single_q_obs...]," ")
        push!(all_observables, one_line)
    end


    #Store the observables with weights with to file:
    open(observableFile, "w") do outIO
        write(outIO, join([string(system_size), " ", string(subsystem_dim)]))
        for single_observable in all_observables
            #println(single_observable)
            #Single line write:
            write(outIO, string("\n", single_observable))
        end
    end
end

#=
function fun(gammaPair)
    system_size=4
    allPairs = [[vcat(gammaPair[1], gammaPair[2])[i-1],vcat(gammaPair[1], gammaPair[2])[i]] for i in 2:2:2*(system_size)]
    shiftedPairs = circshift(allPairs[Bool.(gammaPair[1])],1)
    @show shiftedPairs
    allPairs[Bool.(gammaPair[1])] = shiftedPairs
    @show allPairs
    leftPermKet, rightPermKet = vcat(allPairs...)[1:length(allPairs)], vcat(allPairs...)[length(allPairs)+1:end] 
    #matElement2paulis(leftPermKet *adjoint(leftPermKet))
    return leftPermKet, rightPermKet
end
@show fun(([0,1,0,1], [0,0,1,1]))
=#

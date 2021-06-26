using Pkg
#Pkg.add("StatsBase")
using StatsBase
using LinearAlgebra


function naiveRandomTabClifford(num_qudits,d, onlyLocalOps=false, mi= nothing)
    """
    Returns (pseudo-)random Clifford tableau

    Args:
    - num_qudits::Int = System size; number of qudits
    - d::Int = Subsystem dimension
    - (opt) onlyLocalOps::Bool = Allow only sampling of local (single-qudit) clifford
    - (opt) mi::Int = Meta parameter; communicates value of current measurement counter if used for sampling measurements
    """
    
    cliffArray = []
    for i in 1:rand(d*num_qudits:150)
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
            randGate.cmds[1][2] = StatsBase.sample(1:num_qudits, 2, replace = false) #Ensures no two equal
        elseif randGateStr in ["P", "F", "Z"]
            randGate.cmds[1][2] = [rand(1:num_qudits)]
        else
            println("Check or update your gate-set!")
            exit()
        end

        cmdEntryIdentifier = [randGateStr, randGate.cmds[1][2]]

        if onlyLocalOps == false || length(cliffArray)<genCliffordOrder[randGateStr]-1  #Currently global circuits are not corrected!
            push!(cliffArray, [randGate,cmdEntryIdentifier])
        elseif count(x->x[2]==cmdEntryIdentifier, cliffArray[end-genCliffordOrder[randGateStr]+1+1:end]) != genCliffordOrder[randGateStr]-1 
            #If local circuit: Check whether adding one more gate of this type would make trivial
            push!(cliffArray, [randGate,cmdEntryIdentifier])
        else count(x->x[2]==cmdEntryIdentifier, cliffArray[end-genCliffordOrder[randGateStr]+1+1:end]) == genCliffordOrder[randGateStr]-1 
            #If this command would make existing commands trivial -> remove 
            deleteat!(cliffArray, findall(x->x[2]==cmdEntryIdentifier, cliffArray))
        end
    end
    tempCliffTab = Tab(num_qudits,d)
    for ct in cliffArray
        tempCliffTab = applyClifftoCliff(ct[1],tempCliffTab, num_qudits,d)
    end
    return tempCliffTab
end

#-------Less naive method:

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

function sweepingSubtab(tabinput, colIndices, rowIndices, pauli_app_size,subsystem_dim)
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
    #@assert subtab[pauli_app_size+1,2] != 0

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
    """
    Returns uniformly sampled (global) Clifford circuit 

    Args:
    - system_size::Int = System size; number of qudits
    - subsystem_dim::Int = Subsystem dimension
    """

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
    outCliff = Tab(system_size, subsystem_dim)
    #@show cmd_log
    for s_cmd in cmd_log
        #@show s_cmd
        randGate = deepcopy(genCliffordTab[s_cmd[1]])
        randGate.cmds[1][2] = s_cmd[2]
        outCliff = applyClifftoCliff(randGate,outCliff, system_size, subsystem_dim)
        #@show randGate, outCliff.cmds
    end

    return outCliff #,RC, cmd_log
end

function randomized_local_pauliMeasurements(num_total_measurements::Int,system_size, subsystem_dim)
    measurement_procedure = [] #Initializes which measurements are used
    max_meas_dim = 2 #Remove when Pauli rotation problem resolved
    @showprogress "Sampling Measurement Scheme..." for t in 1:num_total_measurements
        single_round_measurement = [] 
        for i=1:system_size
            r1 = rand(0:max_meas_dim-1)
            r2 = rand(0:max_meas_dim-1)
            while(r1==0 && r2==0)
                r2 = rand(0:max_meas_dim-1)
            end
            push!(single_round_measurement,join(["X",r1, "Z", r2]))
        end
        push!(measurement_procedure, single_round_measurement)
    end
    return measurement_procedure
end

function derandomized_local_pauliMeasurements(all_observables, num_of_measurements_per_observable, system_size, subsystem_dim, weights)
    max_meas_dim = 2 #Remove if pauli rotation problem resovled!
    @assert ((weights != nothing) && (length(weights)==length(all_observables)))
    
    #For purpose & meaning of weights here, also any phase needs to be taken into account:
    weights = abs.([w * (omegaPhaseTilde .^ all_observables[wi].phase) for (wi,w) in enumerate(weights)])
    pauliCollection = filter(x-> x.XZencoding!=[0,0],[PauliOp(1,subsystem_dim,[xp,zp],0) for xp in 0:max_meas_dim-1 for zp in 0:max_meas_dim-1])
    #@show weights
    function cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round)
        """
        #Modified cost function in Huang2021 eq. C7/C8
        # num_of_measurements_so_far == number of measurements of "measurements per observable" covered in completely deterministic measurements
        # num_of_matches_needed_in_this_round == Per observable, how much would it need to match in the remaining qubits in this measurement i.e. weight of observable for qubit k to n
        """

        eta = 0.9 # a hyperparameter subject to change
        nu = 1 - exp(-eta/2)

        cost = 0
        for (i, zipitem) in enumerate(zip(num_of_measurements_so_far, num_of_matches_needed_in_this_round))

            measurement_so_far, matches_needed = zipitem
            if num_of_measurements_so_far[i] >= floor(abs(weights[i]) * num_of_measurements_per_observable)
                #@show "skip", num_of_measurements_so_far, num_of_measurements_so_far[i], matches_needed
                continue #? Hard limit: wouldn't we want to maximize the cost for sufficiently measured observables?
            end

            if system_size < matches_needed
                V = eta / 2 * measurement_so_far
            else #Second term non-zero if all 1 to k qubit elements compatible 
                V = eta / 2 * measurement_so_far - log(1 - nu / (3^matches_needed))
            end
            cost += exp(-V/weights[i]) #mind the negative!
        end
        #@show cost
        return cost
    end

    function match_up(qubit_i, dice_roll_pauli, single_observable)
        """Checks whether single qubit measurement at qubit i compatible with a single observable at qubit i
         """

        if single_observable.XZencoding[[qubit_i, system_size+qubit_i]] == [0,0]
            return 0
        elseif single_observable.XZencoding[[qubit_i, system_size+qubit_i]] != dice_roll_pauli.XZencoding  #Compatible? NOTE: identities don't appear in single_observable *explicitely*!
            return -1 #Incompatible (also completely)
        elseif  single_observable.XZencoding[[qubit_i, system_size+qubit_i]] == dice_roll_pauli.XZencoding
            return 1 #Compatible on this qubit
        end
    end

    num_of_measurements_so_far = zeros(length(all_observables))
    measurement_procedure = []
    
    #No need to fix measurement number M: use modified cost function (M independent) -> also NO need to first randomize everything = Simulate dice rolls! See below.
    totReps = (num_of_measurements_per_observable * length(all_observables))
    println("Derandomization with $totReps repetitions")
    @showprogress "Sampling Measurement Scheme..." for repetition in 1:(num_of_measurements_per_observable * length(all_observables))
        #Iteration over measurements (if fixed = 1 to M)
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_needed_in_this_round = [count([P.XZencoding[[col,col+system_size]]!=[0,0] for col in 1:system_size]) for P in all_observables] # Observable weight!
        single_round_measurement = []

        for qubit_i in 1:system_size #Iteration over qubits within measurement m=repetition and gain cost
            #XZstring2PauliOp(x) = PauliOp(1,subsystem_dim,[parse(Int,split(split(x, "Z")[1], "X")[2]),parse(Int,split(x, "Z")[2])],0) 
            #cost_of_outcomes = Dict{String,Float64}(filter(x-> x!="X0Z0",[join(["X",xp, "Z", zp]) => 0 for xp in 0:max_meas_dim-1 for zp in 0:max_meas_dim-1])...)

            #cost_of_outcomes = Dict{PauliOp,Float64}(filter(x-> x.first.XZencoding != [0,0],[PauliOp(1,subsystem_dim,[xp,zp],0) => 0 for xp in 0:max_meas_dim-1 for zp in 0:max_meas_dim-1])...)
            cost_of_outcomes = Dict()
            #@show cost_of_outcomes
            #sleep(10)
            #cost_of_outcomes = Dict{String,Float64}([[xp,zp] => 0 for xp in 0:max_meas_dim-1 for zp in 0:max_meas_dim-1]...)
 

            for dice_roll_pauli in pauliCollection #Evaluate cost per Pauli option
                # Assume the dice rollout to be "dice_roll_pauli"
                
                for (i,single_observable) in enumerate(all_observables)
                    #Adjust number of matches needed: 
                    #if Pauli incompatible for a single observable -> let it blow up (so big that it also blows up for any proceeding iteration on the same measurement P )
                    #if compatible -> need to take reduction of LEFT accountable weight into sum of cost function of 2nd term in cost function
                    #
                    
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    #@show result, num_of_matches_needed_in_this_round
                    if result == -1
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                        #? Why this number: 100*(system_size+10)??, yes needs to be big enough, but why such specific? why not system_size+1?
                    end
                    if result == 1
                        num_of_matches_needed_in_this_round[i] -= 1 # m/atch up one Pauli X/YZ
                    end
                end
                
                cost_of_outcomes[dice_roll_pauli] = cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round)

                # Revert the dice roll for next dice roll trail from same config
                for (i, single_observable) in enumerate(all_observables)
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1
                        num_of_matches_needed_in_this_round[i] -= 100 * (system_size+10) # impossible to measure
                    end
                    if result == 1
                        num_of_matches_needed_in_this_round[i] += 1 # match up one Pauli X/Y/Z
                    end
                end
            end

            #@show cost_of_outcomes
            
            for dice_roll_pauli in pauliCollection #Award minimal cost Pauli, store it and bookkeep number of measurements left within this measurement iteration m for all compatible observables 
                if minimum(collect(values(cost_of_outcomes))) < cost_of_outcomes[dice_roll_pauli] #Check next dice role if cost is not reduced by this dice role iteration
                    #Mind: if-expression checks if it isn't the minimal cost!
                    continue
                end
                #@show "ANY", dice_roll_pauli, cost_of_outcomes, cost_of_outcomes[dice_roll_pauli]
                # The best dice roll outcome will come to this line
                push!(single_round_measurement, join(["X", dice_roll_pauli.XZencoding[1], "Z", dice_roll_pauli.XZencoding[2]])) #store
                
                for (i, single_observable) in enumerate(all_observables)
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                    end
                    if result == 1
                        num_of_matches_needed_in_this_round[i] -= 1 # match up one Pauli X/Y/Z
                    end
                end
                break #First best Pauli that has smaller cost is chosen
            end
        end

        #sleep(6)
        push!(measurement_procedure, single_round_measurement)

        for (i, single_observable) in enumerate(all_observables)
            if num_of_matches_needed_in_this_round[i] == 0 # finished measuring all qubits + observable i covered completely in this measurement?
                num_of_measurements_so_far[i] += 1
            end
        end

        success = 0
        for (i, single_observable) in enumerate(all_observables) #Check for how many observables the targeted goal of number of measurements has been fullfilled yet
            if num_of_measurements_so_far[i] >= floor(abs(weights[i]) * num_of_measurements_per_observable)
                success += 1
            end
        end

        status = convert(Int64,floor(num_of_measurements_per_observable*success/length(all_observables)))
        println("Measurement $repetition: reached $status measurements for all observables ")

        if success == length(all_observables)  #If we already reached our goal don't generate any new measurements i.e. enlarge M further
            break
        end
    end

    return measurement_procedure  #End and return measurement procedure, even if measurement number goal is not reached for all observables
end

#-------------------------------------FOLLOWING NOT TABLEAU COMPATIBLE:
function randomized_classical_shadow(num_total_measurements::Int, system_size)
    """ Implementation of the randomized classical shadow
    #
    #    num_total_measurements: int for the total number of measurement rounds
    #    system_size: int for how many qubits in the quantum system
    """
    measurement_procedure = [] #Initializes which measurements are used
    for t in 1:num_total_measurements
        single_round_measurement = [StatsBase.sample(["X", "Y", "Z"]) for i=1:system_size]
        push!(measurement_procedure, single_round_measurement)
    end
    return measurement_procedure
end

function derandomized_classical_shadow(all_observables, num_of_measurements_per_observable, system_size, weights=nothing)
   """
    # Implementation of the derandomized classical shadow
    #
    #     all_observables: a list of Pauli observables, each Pauli observable is a list of tuple
    #                      of the form ("X", position) or ("Y", position) or ("Z", position)
    #     num_of_measurements_per_observable: int for the number of measurement for each observable
    #     system_size: int for how many qubits in the quantum system
    #     weight: None or a list of coefficients for each observable
    #             None -- neglect this parameter
    #             a list -- modify the number of measurements for each observable by the corresponding weight
    """
    #if weight == nothing
    if weight == nothing 
        weight = ones(length(all_observables))
    end

    @assert length(weight)==length(all_observables)

    function cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round)
        """
        #Modified cost function in Huang2021 eq. C7/C8
        # num_of_measurements_so_far == number of measurements of "measurements per observable" covered in completely deterministic measurements
        # num_of_matches_needed_in_this_round == Per observable, how much would it need to match in the remaining qubits in this measurement i.e. weight of observable for qubit k to n
        """

        eta = 0.9 # a hyperparameter subject to change
        nu = 1 - exp(-eta/2)

        cost = 0
        for (i, zipitem) in enumerate(zip(num_of_measurements_so_far, num_of_matches_needed_in_this_round))
            measurement_so_far, matches_needed = zipitem
            if num_of_measurements_so_far[i] >= floor(weight[i] * num_of_measurements_per_observable)
                continue #? Hard limit: wouldn't we want to maximize the cost for sufficiently measured observables?
            end

            if system_size < matches_needed
                V = eta / 2 * measurement_so_far
            else #Second term non-zero if all 1 to k qubit elements compatible 
                V = eta / 2 * measurement_so_far - log(1 - nu / (3^matches_needed))
            end
            cost += exp(-V/weight[i]) #mind the negative!
        end
        return cost
    end

    function match_up(qubit_i, dice_roll_pauli, single_observable)
        """Checks whether single qubit measurement at qubit i compatible with a single observable at qubit i
        #qubit_i: measurement qubit: int
        #dice_roll_pauli: Pauli guess: String
        #single_observable: single (full qubit) obsersable to check compatibility: tuple(str, int)
        """

        for (pauli, pos) in single_observable #Searches through qubits for qubit i
            if pos != qubit_i
                continue
            else
                if pauli != dice_roll_pauli  #Compatible? NOTE: identities don't appear in single_observable *explicitely*!
                    return -1 #Incompatible (also completely)
                else
                    return 1 #Compatible on this qubit
                end
            end 
        end
        return 0
    end

    num_of_measurements_so_far = zeros(length(all_observables))
    measurement_procedure = []
    
    #No need to fix measurement number M: use modified cost function (M independent) -> also NO need to first randomize everything = Simulate dice rolls! See below.

    for repetition in 1:(num_of_measurements_per_observable * length(all_observables))
        #Iteration over measurements (if fixed = 1 to M)
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_needed_in_this_round = [length(P) for P in all_observables] # Recall, we have list of tuples = list of observables i.e. len(P) = observable weight
        single_round_measurement = []

        for qubit_i in 0:system_size-1 #Iteration over qubits within measurement m=repetition and gain cost
            cost_of_outcomes = Dict{String,Float64}("X" => 0, "Y" => 0, "Z" => 0)

            for dice_roll_pauli in ["X", "Y", "Z"] #Evaluate cost per Pauli option
                # Assume the dice rollout to be "dice_roll_pauli"
                
                for (i, single_observable) in enumerate(all_observables)
                    #Adjust number of matches needed: 
                    #if Pauli incompatible for a single observable -> let it blow up (so big that it also blows up for any proceeding iteration on the same measurement P )
                    #if compatible -> need to take reduction of LEFT accountable weight into sum of cost function of 2nd term in cost function
                    #
                    
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                        #? Why this number: 100*(system_size+10)??, yes needs to be big enough, but why such specific? why not system_size+1?
                    end
                    if result == 1
                        num_of_matches_needed_in_this_round[i] -= 1 # m/atch up one Pauli X/YZ
                    end
                end
                
                cost_of_outcomes[dice_roll_pauli] = cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round)

                # Revert the dice roll for next dice roll trail from same config
                for (i, single_observable) in enumerate(all_observables)
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1
                        num_of_matches_needed_in_this_round[i] -= 100 * (system_size+10) # impossible to measure
                    end
                    if result == 1
                        num_of_matches_needed_in_this_round[i] += 1 # match up one Pauli X/Y/Z
                    end
                end
            end

            for dice_roll_pauli in ["X", "Y", "Z"] #Award minimal cost Pauli, store it and bookkeep number of measurements left within this measurement iteration m for all compatible observables 
                if minimum(collect(values(cost_of_outcomes))) < cost_of_outcomes[dice_roll_pauli] #Check next dice role if cost is not reduced by this dice role iteration
                    #Mind: if-expression checks if it isn't the minimal cost!
                    continue
                end
                
                # The best dice roll outcome will come to this line
                push!(single_round_measurement, dice_roll_pauli) #store
                for (i, single_observable) in enumerate(all_observables)
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                    end
                    if result == 1
                        num_of_matches_needed_in_this_round[i] -= 1 # match up one Pauli X/Y/Z
                    end
                end
                break #First best Pauli that has smaller cost is chosen
            end
        end
        push!(measurement_procedure, single_round_measurement)

        for (i, single_observable) in enumerate(all_observables)
            if num_of_matches_needed_in_this_round[i] == 0 # finished measuring all qubits + observable i covered completely in this measurement?
                num_of_measurements_so_far[i] += 1
            end
        end

        success = 0
        for (i, single_observable) in enumerate(all_observables) #Check for how many observables the targeted goal of number of measurements has been fullfilled yet
            if num_of_measurements_so_far[i] >= floor(weight[i] * num_of_measurements_per_observable)
                success += 1
            end
        end

        status = convert(Int64,floor(num_of_measurements_per_observable*success/length(all_observables)))
        println("Measurement $repetition: reached $status measurements for all observables ")

        if success == length(all_observables)  #If we already reached our goal don't generate any new measurements i.e. enlarge M further
            break
        end
    end

    return measurement_procedure  #End and return measurement procedure, even if measurement number goal is not reached for all observables
end



function LocBiasedShadow(num_total_measurements::Int, system_size, observableArray, weightArray=nothing)
    """
    Implements Locally Biased Shadow as DAQ Scheme creation procedure
    ONLY PURE states
    """ 


    pauliEncoding = Dict("X"=>0, "Y"=>1, "Z"=>2)

    if weightArray == nothing 
        weightArray = ones(length(observableArray))
    end

    @assert length(weightArray)==length(observableArray)

    function diagonalCost(beta, observables, preferredP=nothing, preferredQubit=nothing) 
        """
        Eq. 16 in HadField et al 2020
        -> State independent cost 
        Observable input for resue in closed form optimization 
        """
        sumResult = 0
        sawPreferredPOnce = false

        for (alphaQ, Q) in zip(weightArray, observables)
            prod = 1
            preferredQubitNonTrivial = false

            for (single_qubit_pauli, i) in Q #Go qubit-wise
                q = pauliEncoding[single_qubit_pauli]+1 #Index = 1 correction

                #Use only for optimization: --------------------
                if preferredQubit == i && preferredP in ["X", "Y", "Z"] && single_qubit_pauli != preferredP
                    prod = 0.0
                end

                if preferredQubit == i
                    preferredQubitNonTrivial = true
                end

                #--------------------------------

                if beta[i+1][q] != 0.0 #DOES NOT APPEAR ANYWAY: single_qubit_pauli[] != "I" 
                    prod *= (1/beta[i+1][q])
                elseif beta[i+1][q] == 0.0
                    prod *= 0.0
                end
            end
            if preferredP == "I" && preferredQubitNonTrivial == false
                println("No observable with qubit $preferredQubit")
                continue #Then don't add sumterm!
            elseif preferredP in ["X", "Y", "Z"] && preferredQubitNonTrivial == false
                println("No observable with qubit $preferredQubit")
                continue #Then don't add sumterm!
            end
            sumResult += (alphaQ^2 * prod)
        end

        if preferredP == "I" && sumResult==0.0
            #@show "Here"
            return 3
        end
        return sumResult
    end

    function optimizeBeta(observables, system_size)
        tempBeta = [[1/3, 1/3, 1/3] for k in 0:system_size-1]  # Start with the uniform distribution = random shadows
        stepSize = 0.05 #Need to choose abitrarily in 0 to 1tempBeta = 1/3
                   
        iter_optimize = 1
        threshold = 0.01
        difference = 1.0

        while difference > threshold
            #@show iter_optimize #Make sure to notice when ending up in infinite loop_

           # @show  [[[diagonalCost(tempBeta, observables, measureP,qubit_i), diagonalCost(tempBeta, observables, "I", qubit_i)] for measureP in ["X", "Y", "Z"] ] for qubit_i in 0:system_size-1]
            #@show (1-stepSize)^(1/system_size)
            updatedBeta = (((1-stepSize) *tempBeta)) + ((stepSize) * [[(diagonalCost(tempBeta, observables, measureP,qubit_i)/diagonalCost(tempBeta, observables, "I", qubit_i)) for measureP in ["X", "Y", "Z"] ] for qubit_i in 0:system_size-1]) #Eq 19
            #multiplication + selection of P_i per qubit i can occur later by invariance
            #But normalization has to occur differently by product!! beta(P) = prod_i beta_i(P_i)
            #That is ^1/system_size for this case

            updatedBeta = [LinearAlgebra.normalize(ub,1) for ub in updatedBeta] #Did only work in size not norm!

            difference = maximum([maximum(abs.((tempBeta-updatedBeta)[i])) for i in 1:length(tempBeta)])
            #@show tempBeta,updatedBeta
            tempBeta = updatedBeta
            iter_optimize += 1
        end

        return tempBeta
    end

    #Find optimal distribution:
    optimalBeta = optimizeBeta(observableArray, system_size)

    #@show optimalBeta

    measurement_procedure = [] #Initializes which measurements are used
    for t in 1:num_total_measurements
        single_round_measurement = [StatsBase.sample(["X", "Y", "Z"], ProbabilityWeights(optimalBeta[i])) for i=1:system_size]
        #@show single_round_measurement
        push!(measurement_procedure, single_round_measurement)
    end

    return measurement_procedure
end

function sortedInsertion(paulisArray, weights)
    """
    Implements Collecting Algorithm: Crawford et al 2021
    Expect full paulisArray of pauli strings i.e. identities included
    """

    if weights == nothing 
        weights = ones(length(paulisArray))
    end
    @assert length(weightArray)==length(observableArray)

    function docommute(pauliString1, pauliString2)
        """
        Checks whether two pauliStrings, provided as array of strings, commute or not
        """
        parity_cnt = 0
        for (pauli_i, pauli_j) in zip(pauliString1, pauliString2)
            if pauli_i != pauli_j && pauli_i != "I" && pauli_j != "I"
                parity_cnt = ((parity_cnt+1) % 2)
            end
        end
        if parity_cnt == 0
            return true
        elseif parity_cnt == 1
            return false
        end
    end
    weightPauliSet = sort(zip(weights, paulisArray), by = first, rev=true) #Sort based on weight backwards
    
    commCollections = [[]]
    for wp in weightPauliSet
        for (c,col) in enumerate(commCollections)
            commute_cnt = 0
            for el in col
                if docommute(el, wp[2])
                    commute_cnt += 1
                end
            end
            if commute_cnt == length(col)
                push!(commCollections[c], wp[2])
            else
                push!(commCollections, [wp[2]])
            end
        end
    end

end

#
# The following code is only used when we run this code through the command line interface
# WARNING: as the below raises errors when not calling it from the commandline, its deactivated for REPL use
#

#=
if length(ARGS) != 0 && abspath(PROGRAM_FILE) == @__FILE__
    function print_usage()
        println(stderr, "Usage:\n")
        println(stderr,"./shadow_data_acquisition -d [number of measurements per observable] [observable.txt]")
        println(stderr,"    This is the derandomized version of classical shadow.")
        println(stderr,"    We would output a list of Pauli measurements to measure all observables")
        println(stderr,"    in [observable.txt] for at least [number of measurements per observable] times.")
        println(stderr,"<or>\n")
        println(stderr,"./shadow_data_acquisition -r [number of total measurements] [system size]")
        println(stderr,"    This is the randomized version of classical shadow.")
        println(stderr,"    We would output a list of Pauli measurements for the given [system size]")
        println(stderr,"    with a total of [number of total measurements] repetitions.")
    end
    
    if length(ARGS) != 3
        print_usage()
    end

    if ARGS[1] == "-d"
        #open(ARGS[3], "r") do f
        #    content = f.readlines()
        #end
        content = readlines(ARGS[3])
        system_size = parse(Int64, content[1])

        all_observables = []
        weightArray = []

        for line in content[2:end]
            one_observable = []
            wBool = 0 #Does the provided file include weights?
            if (length(split(line, " "))-1) % parse(Int64, split(line, " ")[1]) != 0
                wBool = 1
            end
        end

            for (pauli_XYZ, position) in zip(split(line, " ")[2:2:end-wBool], split(line," ")[3:2:end-wBool])
                push!(one_observable, (pauli_XYZ, parse(Int64, position)))
            end
            push!(all_observables, one_observable)

            if wBool == 1
                push!(weightArray, parse(Float64, split(line, " ")[end]))
            end
            
        end
        meas_repetitions_per_observable = parse(Int64, ARGS[2])
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
    elseif ARGS[1] == "-r"
        measurement_procedure = randomized_classical_shadow(parse(Int64, ARGS[2]), parse(Int64, ARGS[3]))
    else
        print_usage()
    end
end

=#

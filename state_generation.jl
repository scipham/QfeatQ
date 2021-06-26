using Pkg
#Pkg.add("LinearAlgebra")
using LinearAlgebra
#Pkg.add("ITensors")
using ITensors
using HDF5
#Pkg.add("RandomQuantum")
using RandomQuantum
#using QuantumClifford
using StatsBase
using Combinatorics


mutable struct BFock    
    num_qudits::Int 
    qn_label::Array{String, 1} #Labels for follow-up qns
    qns::Array{Array{Number,1},1} #Which qns on each position
    fockKet::Array{Int, 1}
    #@assert sum(fockKet) == num_qudits
    BFock(num_qudits, qn_label, qns, fockKet=zeros(num_qudits))=new(num_qudits, qn_label, qns, fockKet)
 end

function Bfock2compstate(fockInput::BFock, num_qudits, max_d)
    """
    Return a computational basis vector corresponding to fockInput in the fock basis

    Args:
    - num_qubits::Int = system size
    """
    @assert fockInput.num_qudits == num_qudits

    compBasisVec(k) = vcat(zeros(k-1),k, zeros(max_d-k))
    basicConfigArray = vcat([fill(qnAtPos, fockInput.fockKet[i]) for (i,fockPos) in enumerate(fockInput.qns)]...)
    allConfigPermutations = unique(permutations(basicConfigArray))
    #@show allConfigPermutations, basicConfigArray
    
    compstateOutput = sparse(1/sqrt(length(allConfigPermutations))*sum([kron([compBasisVec(li+max_d) for li in permi]...) for permi in allConfigPermutations]))
    return compstateOutput *adjoint(compstateOutput)
end

function generatePDC_OAMstate(l_cutoff, system_size, subsystem_dim, coeffArray=[1/sqrt(2) *fill(1/sqrt(l_cutoff),l_cutoff),1/sqrt(2) *fill(1/sqrt(l_cutoff*l_cutoff), (l_cutoff, l_cutoff))])
    system_size, max_subsystem_dim = system_size, 2*l_cutoff+1
    @assert subsystem_dim == max_subsystem_dim
    totalState = zeros(max_subsystem_dim^system_size)

    #First term:
    for l1 in 1:l_cutoff #Or 0:...?
        for l2 in 1:l_cutoff
            if l1 != l2
                l1l2coeff = coeffArray[1]
                @assert size(l1l2coeff)==(l_cutoff, l_cutoff)
                tempFock = BFock(system_size, ["l"], [[l1],[l2], [-l1], [-l2]], [1,1,1,1])
                totalState += lcoeff[l1, l2]*Bfock2compstate(tempFock,system_size, max_subsystem_dim)
            end
        end
    end
    totalState = totalState*adjoint(totalState)

    #Second term:
    numExcit = Int(system_size/2) #system_size/2
    lcoeff = coeffArray[2]
    
    @assert length(lcoeff)==l_cutoff
    @show system_size^max_subsystem_dim, size(totalState)
    for l in 1:l_cutoff
        @show size(generateSimpleDickeState(system_size, max_subsystem_dim, numExcit))
        totalState += lcoeff[l]*generateSimpleDickeState(system_size, max_subsystem_dim, numExcit)
    end
    @show LinearAlgebra.tr(totalState)
    @assert round(LinearAlgebra.tr(totalState), digits=3) == 1.0
    return totalState
end

#Dicke State: Eq.3/4 PHYSICAL REVIEW A 83, 040301(R) (2011
function generateSimpleDickeState(num_qudits, d, m) 
    #m := Number of excitations
    # All combis of | l lbar lbar l ... >
    sExcitVec, sGroundVec = vcat(zeros(d-1),1), vcat(1,zeros(d-1))
    basisBitArray = [if i<=m 1 else 0 end for i in 1:num_qudits] #Create bitarray with m excitations
    DickeOutput = 1/(sqrt(binomial(num_qudits, m)))* sum([kron([b*sExcitVec + abs(b-1)*sGroundVec for b in permi]...) for permi in unique(permutations(basisBitArray))])
    @assert round(norm(DickeOutput), digits=3) ==1
    return DickeOutput*adjoint(DickeOutput)
end

function generateGenSimpleDickeState(num_qudits, d, m) 
    #m := Number of excitations
    # All combis of | l lbar lbar l ... >
    sExcitVec, sGroundVec = vcat(zeros(d-1),1), vcat(1,zeros(d-1))
    sBasisVec(x) = vcat(zeros(x-1),1,zeros(d-x))
    basisBitArray = [if i<=m 1 else 0 end for i in 1:num_qudits] #Create bitarray with m excitations
    DickeOutput = zeros(Float64, d^num_qudits)
    for yit in 1:d-1
        for xit in 0:yit-1
            DickeOutput = DickeOutput .+ sum([kron([b*sBasisVec(yit+1) + abs(b-1)*sBasisVec(xit+1) for b in permi]...) for permi in unique(permutations(basisBitArray))])
        end
    end
    DickeOutput = normalize(DickeOutput .* (1/(sqrt(binomial(num_qudits, m))*(2/(d*(d-1))))))
    @show round(norm(DickeOutput), digits=3)
    @assert round(norm(DickeOutput), digits=3) ==1
    return DickeOutput*adjoint(DickeOutput)
end


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
        for Prep in 1:num_phaseGates
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
            randGate.cmds[1][2] = StatsBase.sample(1:num_qudits, 2, replace = false) #Ensures no two equal
        elseif randGateStr in ["P", "F", "Z"]
            randGate.cmds[1][2] = [rand(1:num_qudits)]
        else
            #@show randGateStr
            println("Check or update your gate-set!")
            exit()
        end
        tempStateTab = applyClifftoCliff(randGate,tempStateTab, num_qudits,d)
    end

    #@assert all(checkConditions(tempStateTab, d))
    return tempStateTab
end


function randomUnitary(n)  #Mezzadri2007 Routine 
    #A Random matrix (n x n) distributed with Haar measure
    
    z = randn(ComplexF64, (n,n))
    @show "Before QR"
    qr = LinearAlgebra.qr(z)
    @show "After QR"
    q = qr.Q
    d = LinearAlgebra.diag(qr.R)
    ph = LinearAlgebra.Diagonal(d ./ abs.(d))
    q = q*ph*q
    return q
end

function randomPureState(n)
    """
        Return a Array{ComplexF64,2}/Density Matrix of a random Pure state

        Args:
        - n::Int = total state space dimension
    """
    randCol = rand(1:n)
    randStateVector = randomUnitary(n)[:,randCol]
    println("Created a randomUnitary")
    #@show norm(randStateVector)
    #randStateVector = rand(FubiniStudyPureState(n))
    return randStateVector* adjoint(randStateVector)
end

function randomMixedState(d,n)
    """
        Return a Array{ComplexF64,2}/Density Matrix of a random Mixed state

        Args:
        - d::Int = subsystem dimension
        - n::Int = total state space dimension
    """
    #backreference to pure-state function
    num_subsys = Int(log(n)/log(d))
    PartialTrace(randomPureState(n^2), collect((num_subsys):(num_subsys*2-1)), fill(d, n*2))
    #randomStateVector = rand(FubiniStudyMixedState(n, rand(1:num_subsys)))
    return randomStateVector*adjoint(randomStateVector)
end


function GHZ(N,d)
    #N = number of qubits = system_size
    #outputAsTableau = Output GHZ for stabilizer measurements in tableau format: [nothing, "+", "-"]
    
    #Alternative, shorter loop way:
    system_size = N
    altStateVector = zeros(d^N)

    temp = Int((d^N-1)/(d-1))
    for t in 0:(d-1)
        altStateVector[temp*t+1] = 1
    end
    return altStateVector*adjoint(altStateVector)
end

function encoderv2(number, dims)
    """
    Return modulo-dims digits of number

    Args:
    - number::Int = Integer to be converted
    - dims::Int = subsystem dimenison
    """
    #Encoding stuff: WARNING: Asymmetric dimension arrays do not work! Use homogenous dimension d
    
    total_sys_dim = prod(dims)
    @assert number < total_sys_dim

    largest_divider = Int(total_sys_dim/dims[end])
    digits = zeros(Int, length(dims))
    remainder = number
    current_divider = largest_divider
    for (dim_i, dim_val) in enumerate(dims)

        current_divider = Int(current_divider)

        #@show current_divider, remainder, remainder/current_divider

        if Int(floor(remainder/current_divider)) < dims[dim_i]
            digits[dim_i] = Int(floor(remainder/current_divider))
            remainder %= current_divider
        elseif Int(floor(remainder/current_divider)) >= dims[dim_i]
            digits[dim_i] = dims[dim_i]-1
            remainder -= (dims[dim_i]-1)
        end
        current_divider /= dims[dim_i]
    end
    return digits
end


function PartialTrace(X, S, dim::Int64) #DIY method
    """
    Return a Array{ComplexF64,2}/ reduced Density Matrix from PartialTrace of X w.r.t S

    Args:
    - dim::Int = subsystem dimension
    - X::Array{ComplexF64,2} =
    - S::Array{Int,1} =
    """
    #Expect density matrix, subsystem-indices to keep and an  scalar of subsystem dimensions
    #Numbering subsystems from 0!!!

    total_sys_dim = Int(sqrt(prod(size(X))))
    if total_sys_dim % dim != 0
        println(stderr, "The subsystem dimensionality must divide the total system size")
    end

    num_of_subsys = Int(floor(log(total_sys_dim)/log(dim)))
    @assert length(S) < num_of_subsys

    total_sys_partition = collect(0:num_of_subsys-1)
    
    S_c = setdiff(total_sys_partition, S) #Complement indices of S for traced out subsystems
    
    S_total_dim = prod(dim^(length(S))) #Total S dimension after consdieration of each index
    S_c_total_dim = prod(dim^(length(S_c))) 

    reduced_dm = zeros(S_total_dim,S_total_dim)
    #Collect reduced density matrix entries
    for i in 0:S_total_dim-1
        for j in 0:S_total_dim-1
            println("Begin with $i and $j")
            i_digits = digits(i, base=dim, pad=length(S))
            j_digits = digits(j, base=dim, pad=length(S))
            contractionSum = 0
            for sci in 0:S_c_total_dim-1
                sci_digits = digits(sci, base=dim, pad=length(S_c))
                iencoding = []
                jencoding = []
                for idx in 0:length(total_sys_partition)-1
                    if idx in S
                        #@show i_digits
                        push!(iencoding, i_digits[S .== idx][1])
                        push!(jencoding, j_digits[S .== idx][1])
                    elseif idx in S_c
                        #@show S_c .== idx, S_c
                        push!(iencoding, sci_digits[S_c .== idx][1])
                        push!(jencoding, sci_digits[S_c .== idx][1])
                    else
                        println(stderr, "Problem!")
                        exit()
                    end
                end
                #@show iencoding
                new_i_index = sum(iencoding.*dim .^(0:(length(iencoding)-1))) +1
                new_j_index = sum(jencoding.*dim .^(0:(length(jencoding)-1))) +1
                contractionSum += X[new_i_index,new_j_index]
            end
            reduced_dm[i+1, j+1] = contractionSum
        end
    end
    return reduced_dm
end


function sampleCliffordSchemeFromJuqst(num_qubits)
    ct = cliffordToTableau(num_qubits, rand(1:getNumberOfSymplecticCliffords(num_qubits)),rand(1:2^num_qubits)) # #NOTE: see paper that indeed the last bitstrings r,s are meant to be abitrary. alpha, beta, gamma, delta = constrained & r,s (phase) = free
    newCmdArray = []
    newXZencoding = transpose(ct.state)[1:end-1,:]
    newPhases = transpose(ct.state)[end,:]
    for ctcmd in ct.commands[2:end]
        splitted = split(ctcmd, "(")
        splitted[2] = splitted[2][1:end-1] #Remove ")" symbol
        
        if splitted[1] == "hadamard"
            targetQubit = parse(Int, splitted[2])
            push!(newCmdArray, ["F", [targetQubit]])
        elseif splitted[1] == "phase"
            targetQubit = parse(Int, splitted[2])
            realGate = rand(["Z","P"])
            push!(newCmdArray, [realGate, [targetQubit]])
        elseif splitted[1] == "cnot"
            involvedQubits = split(splitted[2],",") #Divide qubit definition 
            controlQubit, targetQubit = parse(Int, involvedQubits[1]), parse(Int, involvedQubits[2])
            push!(newCmdArray, ["CN", [controlQubit,targetQubit]])
        end
    end
    cliffTab = Tab(num_qubits,2,newCmdArray, newXZencoding, newPhases)
    return cliffTab
end


#------------------OLD Qubit / Juqst / QuantumClifford version
#=
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
            while randControl == randQubit
                randControl = rand(1:num_qubits)
            end
            Juqst.cnot(t,randControl,randQubit)
        end
    end
    return t
end

function fullGate(gate, localQubits)
    #Always provide (sinlge) local qubits in arrays!
    if gate == Hadamard && length(localQubits) == 1
        if system_size - localQubits != 0 && localQubits[1]-1 != 0
            return tensor_pow(CliffordId, localQubits[1]-1) ⊗ Hadamard ⊗ tensor_pow(CliffordId, systemsize-localQubits[1])
        elseif system_size - localQubits == 0 && localQubits[1]-1 != 0
            return Hadamard ⊗ tensor_pow(CliffordId, systemsize-localQubits[1])
        elseif system_size - localQubits != 0 && localQubits[1]-1 == 0
            return tensor_pow(CliffordId, localQubits[1]-1) ⊗ Hadamard
        end
    elseif gate == Phase && length(localQubits) == 1
        if system_size - localQubits != 0 && localQubits[1]-1 != 0
            return tensor_pow(CliffordId, localQubits[1]-1) ⊗ Phase ⊗ tensor_pow(CliffordId, systemsize-localQubits[1])
        elseif system_size - localQubits == 0 && localQubits[1]-1 != 0
            return Phase ⊗ tensor_pow(CliffordId, systemsize-localQubits[1])
        elseif system_size - localQubits != 0 && localQubits[1]-1 == 0
            return tensor_pow(CliffordId, localQubits[1]-1) ⊗ Phase
        end
    elseif gate == CNOT && length(localQubits) == 1
        if system_size - localQubits != 0 && localQubits[1]-1 != 0
            return tensor_pow(CliffordId, localQubits[1]-1) ⊗ CNOT ⊗ tensor_pow(CliffordId, systemsize-localQubits[1]-1)
        elseif system_size - localQubits == 0 && localQubits[1]-1 != 0
            return CNOT ⊗ tensor_pow(CliffordId, systemsize-localQubits[1]-1)
        elseif system_size - localQubits != 0 && localQubits[1]-1 == 0
            return tensor_pow(CliffordId, localQubits[1]-1) ⊗ CNOT
        end
    end
end

function alt_ghzState(system_size)

    phaseSet = [0x0, 0x1, 0x2, 0x3]
    pauliEncod = Dict("X" => [1; 0],"Y" =>[1; 1], "Z" => [0;1]) #X,Y,Z

    #Initialize the vacuumState
    shortDecideZ = ((x,y) -> if x==y return 1 else return 0 end)
    PauliArray = [PauliOperator(phaseSet[1], Bool.(zeros(Int, system_size)), Bool.([shortDecideZ(i,j) for j in 1:system_size])) for i in 1:system_size]
    vacuumState = Stabilizer([PauliArray]...)
    
    outState = deepcopy(vacuumState)
    #h1Gate = fullGate(Hadamard,[1])
    apply!(outState, Hadamard, [1])

    for i in 2:system_size
        #cnoteiGate = fullGate(CNOT,[i, i+1])
        apply!(outState, CNOT, [1, i])
    end

    return vacuumState
end


function alt_randomStabilizerState(system_size)

    randomStab =  random_stabilizer(system_size)
    #=
    for i in 1:rand(1:2^system_size)
        randOperation = rand([Hadamard, Phase, CNOT])
        rand()
        if randOperation == CNOT
            c, t = rand(1:system_size), rand(1:system_size)
            while c == t
                c, t = rand(1:system_size), rand(1:system_size)
            end
            apply!(vacuumState, randOperation, [c, t])
        else
            totalOperation = tensor_pow(CliffordId, system_size-)
            apply!(vacuumState, totalOperation)
        end
    end
    =#
    return randomStab
end




function PartialTrace(X, S, dim::Array) #Using Tensor contraction
    #Expect density matrix, subsystem-indices to trace out and an array of subsystem dimensions
    #Numbering subsystems from 0 like qubits!!!

    total_sys_dim = Int(prod(dim))
    Subsystem_indices = [ITensors.Index(dim_i) for dim_i in dim] #Subsystems along x
    Primed_Subsystem_indices = ITensors.prime.(Subsystem_indices) #Subsystems along y

    total_subsys_indices = vcat(Subsystem_indices, Primed_Subsystem_indices)

    complete_dm = ITensors.ITensor(total_subsys_indices...)
    #To prevent errors we need to initialize the Itensor as complex thing: how to do this more beautifully?
    initIndices = ones(length(total_subsys_indices))
    initIndices[end]=2
    setindex!(complete_dm ,XtempVal, tempTensorInd...) = 1.02 + 1.24*im

    for x in 0:total_sys_dim-1
        for y in 0:total_sys_dim-1
            println("Assigning the ITensor values at x=$x and y=$y")
            XtempVal = X[x+1,y+1]
            tempTensorInd = vcat((encoderv2(x, dim) .+ 1), (encoderv2(y, dim) .+ 1))
            @show XtempVal, tempTensorInd
            setindex!(complete_dm ,ComplexF64(XtempVal), tempTensorInd...)
        end
    end

    num_of_subsys = length(dim)
    @assert length(S) < num_of_subsys

    total_sys_partition = collect(0:num_of_subsys-1)
    
    S_c = setdiff(total_sys_partition, S) #Complement indices of S for subsystems to keep
    
    for subsys in S
        #noprime!(complete_dm, total_subsys_indices[num_of_subsys+subsys+1])
        #@show reduced_dm[total_subsys_indices[num_of_subsys+subsys+1] => (subsys+1)]
        #@show complete_dm *= delta() 
        complete_dm *= delta(total_subsys_indices[subsys+1], total_subsys_indices[num_of_subsys+subsys+1])
    end
    return array(complete_dm)
end


function destab_GHZ(N, negativePhase=false)
    initial_state = one(Stabilizer, N)
    apply!(initial_state, Hadamard ⊗ tensor_pow(CliffordId, N-1))
    apply!(initial_state, CNOT ⊗ tensor_pow(CliffordId, N-2))
    for i in 2:N-1
        apply!(initial_state,tensor_pow(CliffordId, i-1) ⊗ CNOT ⊗ tensor_pow(CliffordId, N-i))
    end

    if negativePhase
        apply!(initial_state, CliffordId ⊗ Phase ⊗ tensor_pow(CliffordId, N-2)) #phase i
        apply!(initial_state, CliffordId ⊗ Phase ⊗ tensor_pow(CliffordId, N-2)) #phase -1
    end 
    return initial_state
end


function GHZ(N,d=2, outputAsTableau=nothing)
    #N = number of qubits = system_size
    #outputAsTableau = Output GHZ for stabilizer measurements in tableau format: [nothing, "+", "-"]
    
    #stateVector = zeros(d^N)
    #for j in 0:(d^N-1)
    #    digits = encoderv2(j, [d for i in 1:N])
    #    if all(x-> x==digits[1],digits) #Check if base-d expansion has equal values
    #        stateVector[j+1] += 1
    #    end
    #end
    
    #Alternative, shorter loop way:
    system_size = N
    if outputAsTableau =="+" && d==2    #Only qubits supported by used library Juqst
        tempTab = Tableau(system_size) #Start with vacuum state tableau
        Juqst.hadamard(tempTab, 1)
        for i in 2:system_size
            Juqst.cnot(tempTab, 1, i)
        end
        return tempTab
    elseif outputAsTableau == "-" && d==2
        tempTab = Tableau(system_size)
        Juqst.hadamard(tempTab, 1)
        for i in 2:system_size
            Juqst.cnot(tempTab, 1, i)
        end
        for i in 1:2 #Correct phase to a relative phase -1
            Juqst.phase(tempTab, 2)
        end
        return tempTab
    else
        altStateVector = zeros(d^N)

        temp = Int((d^N-1)/(d-1))
        for t in 0:(d-1)
            altStateVector[temp*t+1] = 1
        end
        return altStateVector*adjoint(altStateVector)
    end
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

    if length(ARGS) != 3 && true==false
        print_usage()
    end
    
    d=2 #state base
    system_size=10 #num of qudits
    system_dim = d^system_size

    if ARGS[1] == "-rp" #random pure state
        receivedState = randomPureState(system_dim)
        #@show receivedState, size(receivedState)
        #@show receivedState*LinearAlgebra.adjoint(receivedState)
        h5open("pureState.txt", "w") do fid
        fid["pureState"]= receivedState
        end

    elseif ARGS[1] == "-rm"
        @show randomMixedState(d, system_dim)
    elseif ARGS[1] == "-rdm"
        @show PartialTrace(reshape(1:16, 4,4), [1], [2,2])
    else
        print_usage()
    end
end

=#
=#
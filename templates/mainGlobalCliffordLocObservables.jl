using LinearAlgebra
#include("controller.jl")
using Statistics
using HDF5
using JLD

#Import framework:
include("QfeatQ.jl")

function mainExperimentController(system_size, subsystem_dim, number_of_measurements,  observableFile="observables.txt")

    QfeatQ.initialize(system_size, subsystem_dim)
    
    #Generate states to simulate on:
    #tableauStateMemory = state_generation_Controller("ghz+_stab", system_size, subsystem_dim, genCliffordTab)
    #tableauStateMemory = Tab(system_size, subsystem_dim) 
    #tableauStateMemory = naiveRandomTabState(system_size, subsystem_dim)
    tableauStateMemory = QfeatQ.state_generation_Controller("rp-stab", system_size, subsystem_dim, "systemState.txt")
    
    #state_generation_Controller("rp", "systemState.txt", system_size, subsystem_dim)

    #@show tableauStateMemory
    #Create a measurement_procedure: Randomized : GLOBAL
    cliffMeasScheme = QfeatQ.create_data_acquisition_scheme_Controller("rc", number_of_measurements, system_size,subsystem_dim, observableFile, false)
    #cliffMeasScheme = [sampleCliffordSchemeFromJuqst(system_size) for i in 1:number_of_measurements]
    #t=Tab(system_size, subsystem_dim)
    #t.cmds = [["F", [2]], ["Z", [1]], ["F", [1]], ["CN", [1, 3]], ["CN", [2, 3]], ["Z", [3]], ["P", [2]], ["Z", [1]]]
    #cliffMeasScheme = [deepcopy(t) for i in 1:number_of_measurements]


    #@show cliffMeasScheme

    #sleep(10)
    #Measure the cliffords on the given state - GLOBAL
    #cliffordOutcomes = measurement_Controller("clifford", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)
    cliffordOutcomes = QfeatQ.measurement_Controller("clifford-eff-global", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)

    #cliffordOutcomes = measurement_Controller("clifford-mat", true, system_size, "systemState.txt", "cliffordMeasurementscheme.txt", "cliffordOutcomes.txt", cliffMeasScheme)

    #=
    alt_out = []
    for c_op in cliffMeasScheme
        vSt = simpleCliffordToMat(tableauStateMemory, system_size, subsystem_dim)[:,1]
        push!(alt_out,compBasisMeasurement(simpleCliffordToMat(c_op, system_size, subsystem_dim)*(vSt*adjoint(vSt))*adjoint(simpleCliffordToMat(c_op, system_size, subsystem_dim))))
    end
    =#
    #println(stdout, "text/plain", collect(zip(alt_out, cliffordOutcomes)))

    #=
    abzip = zip([cs.cmds for cs in cliffMeasScheme],cliffordOutcomes)
    for (a,b) in abzip
        @show a, b
    end
    show(stdout, "text/plain", round.(simpleCliffordToMat(cliffMeasScheme[1], system_size, subsystem_dim),digits=10))
    =#


    #Post-process (classical shadows) the measurement outcomes & Predict observables

    stateCopy = deepcopy(tableauStateMemory)
    #stabVec = round.(makeFromCommand(stateCopy))[1,:]

    #Observable Predictions
    
    #stateCopy = zeros(subsystem_dim^system_size)
    #stateCopy[1] = 1
    #stateCopy[end] = 1/sqrt(2)

    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowClifford-stab", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    
    trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o-eff", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)
    #=
    mostTrueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)
    #@show "It can't get better than this:"
    #show(stdout,"text/plain", mostTrueExpectations)
    #@show "Truth-error:"
    truthError =  abs.(mostTrueExpectations .- trueExpectations)
    show(stdout,"text/plain", truthError)
    =#

    #predictions, truePredictions = observable_prediction_Controller("cShadowLocClifford-mat", "lo", system_size, "cliffordOutcomes.txt", observableFile, "systemState.txt", cliffordOutcomes, cliffMeasScheme)
    #trueExpectations = observable_prediction_Controller("truthPauli", "o", system_size, "cliffordOutcomes.txt", observableFile, "systemState.txt", cliffordOutcomes, cliffMeasScheme, nothing)

    #Output/plot stuff:
    println("Let's predict:")
    show(stdout, "text/plain",predictions)
    println("The truth is:")
    show(stdout, "text/plain", trueExpectations)
    
   
    error = abs.(predictions .- trueExpectations)
    @show error, mean(error)

    return convert(Array{ComplexF64, 1},predictions), convert(Array{ComplexF64, 1},trueExpectations), convert(Array{ComplexF64, 1}, error)
end


mainExperimentController(3, 4, 400, string("observables_",3,"qubit_identity.txt"))

#=
qubit_number, runi = 10, 1

h5open("3to12benchmark.dat", "w") do fid

p, t, e = mainExperimentController(qubit_number, 2, 400, string("observables_",qubit_number,"qubit.txt"))
@show "Done with $qubit_number iteration $i"

fid[string("errorArray",qubit_number,"q",runi)] = e
fid[string("predictionArray",qubit_number,"q", runi)] = p
fid[string("truePredictionArray", qubit_number, "q", runi)] = t

end
=#

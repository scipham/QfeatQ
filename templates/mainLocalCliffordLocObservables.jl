using LinearAlgebra
#include("controller.jl")
using Statistics
using HDF5
using JLD
using Juqst

#Import framework:
include("QfeatQ.jl")

function mainExperimentController(system_size, subsystem_dim, number_of_measurements,  observableFile="observables.txt")

    QfeatQ.initialize(system_size, subsystem_dim)
    
    #Generate states to simulate on:
    #-tableauStateMemory = QfeatQ.state_generation_Controller("ghz+_stab", system_size, subsystem_dim, genCliffordTab)
    tableauStateMemory = QfeatQ.state_generation_Controller("rp-stab", system_size, subsystem_dim, nothing)
 
    #tableauStateMemory = Tab(system_size, subsystem_dim) 

    #@show tableauStateMemory
    #Create a measurement_procedure: Randomized : GLOBAL
    #cliffMeasScheme = [sampleCliffordSchemeFromJuqst(system_size) for i in 1:number_of_measurements]
    #t=Tab(system_size, subsystem_dim)
    #t.cmds = [["F", [2]], ["Z", [1]], ["F", [1]], ["CN", [1, 3]], ["CN", [2, 3]], ["Z", [3]], ["P", [2]], ["Z", [1]]]
    #cliffMeasScheme = [deepcopy(t) for i in 1:number_of_measurements]

    #Create a measurement_procedure: Randomized : LOCAL
    cliffMeasScheme = QfeatQ.create_data_acquisition_scheme_Controller("rlp", number_of_measurements, system_size,subsystem_dim, observableFile)


    #Measure the cliffords on the given state - LOCAL
    cliffordOutcomes = QfeatQ.measurement_Controller("clifford-eff-local", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)


    #Post-process (classical shadows) the measurement outcomes & Predict observables

    stateCopy = deepcopy(tableauStateMemory)
    #stabVec = round.(makeFromCommand(stateCopy))[1,:]

    #Observable Predictions

    #predictions = round.(QfeatQ.observable_prediction_Controller("cShadowLocClifford-stab", "o", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowLocClifford-direct", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    #predictionsCor = round.(QfeatQ.observable_prediction_Controller("cShadowLocClifford-direct", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomesCor, cliffMeasScheme) ,digits=10)

    trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)
    
    #trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o-eff", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)


    #predictions, truePredictions = observable_prediction_Controller("cShadowLocClifford-mat", "lo", system_size, "cliffordOutcomes.txt", observableFile, "systemState.txt", cliffordOutcomes, cliffMeasScheme)
    #trueExpectations = observable_prediction_Controller("truthPauli", "o", system_size, "cliffordOutcomes.txt", observableFile, "systemState.txt", cliffordOutcomes, cliffMeasScheme, nothing)

    #Output/plot stuff:
    println("Let's predict:")
    show(stdout, "text/plain",predictions)
    println("The truth is:")
    show(stdout, "text/plain", trueExpectations)
    
   
    error = abs.(predictions .- trueExpectations)
    @show error, mean(error), maximum(error)
    #predError = abs.(predictions .- predictionsCor)
    #@show predError, mean(predError), maximum(predError)

    return convert(Array{ComplexF64, 1},predictions), convert(Array{ComplexF64, 1},trueExpectations), convert(Array{ComplexF64, 1}, error)
end


num_qudits, d = 3,4
QfeatQ.generateCompleteRandomObservables(400, num_qudits, d, "randObs.txt", 3)

#mainExperimentController(num_qudits, d, 10000,string("observables_",3,"qubit_identity.txt"))

mainExperimentController(num_qudits, d, 10000,string("randObs.txt"))
@show "End"
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

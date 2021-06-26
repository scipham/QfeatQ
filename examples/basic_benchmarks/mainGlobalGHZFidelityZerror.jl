
using LinearAlgebra
#include("controller.jl")
using Statistics
using HDF5
using JLD
using ProgressMeter
using Distributions

#Import framework:
include("../QfeatQ.jl")

function mainExperimentController(system_size, subsystem_dim, number_of_measurements,  observableFile, zerrorProb)

    QfeatQ.initialize(system_size, subsystem_dim)
    
    #Generate states to simulate on:
    pureStateMemory = QfeatQ.state_generation_Controller("ghz+_stab", system_size, subsystem_dim, nothing)
    impurityStateMemory = QfeatQ.state_generation_Controller("ghz-_stab", system_size, subsystem_dim, nothing)
    stateDistribution = [[pureStateMemory, impurityStateMemory],[1-zerrorProb, zerrorProb]]

    #tableauStateMemory = Tab(system_size, subsystem_dim) 
    #tableauStateMemory = naiveRandomTabState(system_size, subsystem_dim)
    #tableauStateMemory = QfeatQ.state_generation_Controller("rp-stab", system_size, subsystem_dim, "systemState.txt")
    
    #state_generation_Controller("rp", "systemState.txt", system_size, subsystem_dim)

    #@show tableauStateMemory
    #Create a measurement_procedure: Randomized : GLOBAL
    cliffMeasScheme = QfeatQ.create_data_acquisition_scheme_Controller("rc", number_of_measurements, system_size,subsystem_dim, observableFile, false)

    #Measure the cliffords on the given state - GLOBAL
    #cliffordOutcomes = measurement_Controller("clifford", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)
    
    cliffordOutcomes = QfeatQ.measurement_Controller("clifford-eff-global", true, system_size,subsystem_dim, cliffMeasScheme, stateDistribution)

    #cliffordOutcomes = measurement_Controller("clifford-mat", true, system_size, "systemState.txt", "cliffordMeasurementscheme.txt", "cliffordOutcomes.txt", cliffMeasScheme)

    #Post-process (classical shadows) the measurement outcomes & Predict observables

    #Observable Predictions
    
    #stateCopy = zeros(subsystem_dim^system_size)
    #stateCopy[1] = 1
    #stateCopy[end] = 1/sqrt(2)

    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowClifford-stab", "f", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme, deepcopy(pureStateMemory)) ,digits=10)
    
    trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "f", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, [stateDistribution, pureStateMemory])
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

    #return convert(ComplexF64,predictions), convert(ComplexF64,trueExpectations), convert(ComplexF64, error)
    return predictions, trueExpectations, error
end

h5open("./ghzbenchmarkCor.dat", "w") do fid

runRange = collect(1:1)
zerrorRange = range(0.0,stop=1.0,length=10)
allFids = []
@showprogress 1 "Running over Z-errors..." for ze in zerrorRange #phase flip error
    @showprogress 1 "Running zerror=$ze mutliple times..." for runi in runRange
        p,t,e = mainExperimentController(3, 2, 60000, nothing,ze)
        @show "Done with zerror=$ze, iteration $runi"

        fid[string("errorArray",ze ,"ze",runi)] = e
        fid[string("predictionArray",ze ,"ze",runi)] = p
        fid[string("truthArray",ze ,"ze",runi)] = t

        cpFids = []
        averageFid = 0
        if typeof(p) == Array{ComplexF64,1}
            averageFid += p[1]
            push!(cpFids, p[1])
        else
            averageFid += p
            push!(cpFids, p)
        end
        push!(allFids, cpFids)
    end
    averageFid = averageFid/length(runRange)
    println("The mean for zerror= $ze is F=$averageFid")
end
end


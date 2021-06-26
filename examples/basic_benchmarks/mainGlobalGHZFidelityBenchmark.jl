
using LinearAlgebra
#include("controller.jl")
using Statistics
using HDF5
using JLD
using ProgressMeter
using Distributions
using StatsBase

#Import framework:
include("../QfeatQ.jl")

function mainExperimentController(system_size, subsystem_dim, number_of_measurements,  observableFile)

    QfeatQ.initialize(system_size, subsystem_dim)


    #Generate states to simulate on:
    tableauStateMemory = QfeatQ.state_generation_Controller("ghz+_stab", system_size, subsystem_dim, nothing)
 
    #tableauStateMemory = Tab(system_size, subsystem_dim) 
    #tableauStateMemory = naiveRandomTabState(system_size, subsystem_dim)
    #tableauStateMemory = QfeatQ.state_generation_Controller("rp-stab", system_size, subsystem_dim, "systemState.txt")
    
    #state_generation_Controller("rp", "systemState.txt", system_size, subsystem_dim)

    #@show tableauStateMemory
    #Create a measurement_procedure: Randomized : GLOBAL
    cliffMeasScheme = QfeatQ.create_data_acquisition_scheme_Controller("rc", number_of_measurements, system_size,subsystem_dim, observableFile, false)

    #Measure the cliffords on the given state - GLOBAL
    #cliffordOutcomes = measurement_Controller("clifford", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)
    
    cliffordOutcomes = QfeatQ.measurement_Controller("clifford-eff-global", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)

    #cliffordOutcomes = measurement_Controller("clifford-mat", true, system_size, "systemState.txt", "cliffordMeasurementscheme.txt", "cliffordOutcomes.txt", cliffMeasScheme)

    #Post-process (classical shadows) the measurement outcomes & Predict observables

    #stabVec = round.(makeFromCommand(stateCopy))[1,:]

    #Observable Predictions
    
    #stateCopy = zeros(subsystem_dim^system_size)
    #stateCopy[1] = 1
    #stateCopy[end] = 1/sqrt(2)

    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowClifford-stab", "f", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme, deepcopy(tableauStateMemory)) ,digits=10)
 
    #trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "f", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, [stateDistribution, pureStateMemory])
    #trueExpectations would be overkill as long as Inner product i.e. correct measurement scheme on stabilizers is not fully implemented
    #But we know that it should be 1 in any case for this experiment!
    trueExpectations = fill(1.0, length(predictions))
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
    return real(mean(predictions))
end

h5open("./ghzbenchmarkNummeasCor.dat", "w") do fid

runRange = collect(1:15)
syssizeRange = reverse(collect(3:7)) 
subsystem_dim = 5
    
max_nummeas = 6000
F_threshold = 0.99 

tempNumMeas = []
@showprogress 1 "Running over system sizes..." for system_size in syssizeRange #phase flip error
    @showprogress 1 "Running over number of measurements..." for nm in (10 .^(range(1.1,stop=3.8,length=150)))
        println("Now starting with $nm for q=$system_size")
        nm = Int(round(nm))
        c_fidelity_array = []
        @showprogress 1 "Running prediction mutliple times..." for runi in runRange
            c_fidelity = mainExperimentController(system_size, subsystem_dim, nm, nothing)
            push!(c_fidelity_array, c_fidelity)
            @show "Done with system_size=$system_size, iteration $runi"
        end
        
            prob = cdf(Normal(F_threshold, 1/(k*nm)^2), mean(c_fidelity_array))
            @show prob
        #if mean(c_fidelity_array) - 1/sqrt(nm) >= F_threshold
        #    push!(tempNumMeas, nm)
        #    break
        #else
        #    continue
        #end
    end
end

fid[string("nummeasArray")] =  convert(Array{Int,1},tempNumMeas)
show(stdout, "text/plain", tempNumMeas)
end

using LinearAlgebra
#include("controller.jl")
using Statistics
using HDF5
using JLD
using ProgressMeter

#Import framework:
include("../QfeatQ.jl")

function mainExperimentController(system_size, subsystem_dim, number_of_measurements,  observableFile="observables.txt")

    QfeatQ.initialize(system_size, subsystem_dim)
    
    tableauStateMemory = QfeatQ.state_generation_Controller("rp-stab", system_size, subsystem_dim, "systemState.txt")
    
    #state_generation_Controller("rp", "systemState.txt", system_size, subsystem_dim)
    
    #Create a measurement_procedure: Randomized : LOCAL
    cliffMeasScheme = QfeatQ.create_data_acquisition_scheme_Controller("rlp", number_of_measurements, system_size,subsystem_dim, observableFile)

    #Measure the cliffords on the given state - LOCAL
    
    cliffordOutcomes = QfeatQ.measurement_Controller("clifford-eff-local", true, system_size,subsystem_dim, cliffMeasScheme, tableauStateMemory)
    
    #Post-process (classical shadows) the measurement outcomes & Predict observables

    stateCopy = deepcopy(tableauStateMemory)
    
    #predictions = round.(QfeatQ.observable_prediction_Controller("cShadowClifford-stab", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    
    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowLocClifford-direct", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    
    trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o-eff", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)
    #trueExpectations =  QfeatQ.observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)

    #=
    mostTrueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, cliffMeasScheme, stateCopy)
    #@show "It can't get better than this:"
    #show(stdout,"text/plain", mostTrueExpectations)
    #@show "Truth-error:"
    truthError =  abs.(mostTrueExpectations .- trueExpectations)
    show(stdout,"text/plain", truthError)
    =#

    #Output/plot stuff:
    #println("Let's predict:")
    #show(stdout, "text/plain",predictions)
    #println("The truth is:")
    #show(stdout, "text/plain", trueExpectations)
    
   
    error = abs.(predictions .- trueExpectations)
    @show error, mean(error)

    return convert(Array{ComplexF64, 1},predictions), convert(Array{ComplexF64, 1},trueExpectations), convert(Array{ComplexF64, 1}, error)
end



#mainExperimentController(3, 2, 100,string("observables_",3,"qubit.txt"))


h5open("basicLoctotalbenchmarkAltMatTruth.dat", "w") do fid
#syssizeRange = collect(3:15)
subsysdimRange = reverse(collect(2:7))
syssizeRange = reverse(collect(2:7))
errArray = []
numRuns = 6

@showprogress 1 "Running..." for runi in 1:numRuns
    @showprogress 1 "Running Through Subsystem dimensions..." for subsystem_dim in subsysdimRange
        @showprogress 1 "Running Through System Sizes..." for system_size in syssizeRange
            QfeatQ.generateCompleteRandomObservables(80, system_size, subsystem_dim, "randObs.txt", 3)

            qudit_number,subsystem_dim = (system_size,subsystem_dim)

            p,t, e = mainExperimentController(system_size, subsystem_dim, 40000, string("./randObs.txt"))

            #push!()
            #push!(errArray, mean(e))

            @show "Done with q=$system_size,d=$subsystem_dim iteration $runi"

            fid[string("errorArray",system_size ,"q",subsystem_dim,"d",runi)] = e
            fid[string("predictionArray",system_size ,"q",subsystem_dim,"d", runi)] = p
            fid[string("truthArray",system_size ,"q",subsystem_dim,"d", runi)] = t
       	    GC.gc()
	 end
    end
end

end

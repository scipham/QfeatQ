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
    QfeatQ.state_generation_Controller("rp", system_size, subsystem_dim, "systemState.txt")

    #Create a measurement_procedure: Randomized : GL=OBAL

    cliffMeasScheme = QfeatQ.create_data_acquisition_scheme_Controller("rc", number_of_measurements, system_size,subsystem_dim, observableFile, false)

    #cliffMeasScheme = [sampleCliffordSchemeFromJuqst(1) for i in 1:number_of_measurements]
    #t=Tab(system_size, subsystem_dim)
    #t.cmds = [["F", [2]], ["Z", [1]], ["F", [1]], ["CN", [1, 3]], ["CN", [2, 3]], ["Z", [3]], ["P", [2]], ["Z", [1]]]
    #cliffMeasScheme = [deepcopy(t) for i in 1:number_of_measurements]

    #@show cliffMeasScheme

    #sleep(10)
    #Measure the cliffords on the given state - GLOBAL
    cliffordOutcomes = QfeatQ.measurement_Controller("clifford-mat", true, system_size,subsystem_dim, cliffMeasScheme,"systemState.txt")

    #cliffordOutcomes = measurement_Controller("clifford-mat", true, system_size, "systemState.txt", "cliffordMeasurementscheme.txt", "cliffordOutcomes.txt", cliffMeasScheme)

    #=
    abzip = zip([cs.cmds for cs in cliffMeasScheme],cliffordOutcomes)
    for (a,b) in abzip
        @show a, b
    end
    show(stdout, "text/plain", round.(simpleCliffordToMat(cliffMeasScheme[1], system_size, subsystem_dim),digits=10))
    =#


    #Post-process (classical shadows) the measurement outcomes & Predict observables
    #stabVec = round.(makeFromCommand(stateCopy))[1,:]

    #Observable Predictions
    
    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowClifford-stab", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    
    #stateCopy = zeros(subsystem_dim^system_size)
    #stateCopy[1] = 1
    #stateCopy[end] = 1/sqrt(2)
    #stateCopy = simpleCliffordToMat(tableauStateMemory, system_size, subsystem_dim)[:,1]

    trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, nothing, "systemState.txt")
    

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

mainExperimentController(3, 3, 3000, string("observables_",3,"qubit_identity.txt"))

#=
subsysdimRange = collect(2:5)
errArray = []


for subsystem_dim in subsysdimRange
qudit_number,subsystem_dim, runi = 3,subsystem_dim, 1

h5open("quditbenchmark.dat", "w") do fid

p, t, e = mainExperimentController(3, subsystem_dim, 2000, string("observables_",3,"qubit",subsystem_dim,".txt"))

push!(errArray, mean(e))

@show "Done with d=$subsystem_dim iteration $i"

#fid[string("errorArray",qudit_number,"q",runi)] = e
#fid[string("predictionArray",qudit_number,"q", runi)] = p
#fid[string("truePredictionArray", qudit_number, "q", runi)] = t

end
end

plt = plot(subsysdimRange, errArray, title="Error - qudits")
xlabel!("Subsystem size")
ylabel!("Mean error")
display(plt)

=#
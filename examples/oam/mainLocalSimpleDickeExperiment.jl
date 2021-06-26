using Base: input_color
using LinearAlgebra
#include("controller.jl")
using Statistics
using HDF5
using JLD

#Import framework:
include("../QfeatQ.jl")

function mainExperimentController(system_size, subsystem_dim, number_of_measurements,  observableFile="observables.txt")
    QfeatQ.initialize(system_size,subsystem_dim)
    
    #Generate states to simulate on:
    QfeatQ.state_generation_Controller("dicke", system_size, subsystem_dim, "systemState.txt")
    
    @show "Finished state generation"
    #Create a measurement_procedure: Randomized : GL=OBAL
    #cliffMeasScheme = create_data_acquisition_scheme_Controller("rlp", number_of_measurements, system_size,subsystem_dim, observableFile, false)

    cliffMeasScheme =  QfeatQ.create_data_acquisition_scheme_Controller("rlp", number_of_measurements, system_size,subsystem_dim, observableFile, nothing)

    #cliffMeasScheme = [sampleCliffordSchemeFromJuqst(1) for i in 1:number_of_measurements]
    #t=Tab(system_size, subsystem_dim)
    #t.cmds = [["F", [2]], ["Z", [1]], ["F", [1]], ["CN", [1, 3]], ["CN", [2, 3]], ["Z", [3]], ["P", [2]], ["Z", [1]]]
    #cliffMeasScheme = [deepcopy(t) for i in 1:number_of_measurements]


    #@show cliffMeasScheme

    #sleep(10)
    #Measure the cliffords on the given state - GLOBAL
    #cliffordOutcomes = measurement_Controller( "clifford-loc-mat", true, system_size,subsystem_dim, cliffMeasScheme,"systemState.txt")
    #cliffordOutcomes = measurement_Controller( "clifford-loc-mat", true, system_size,subsystem_dim, cliffMeasScheme,"systemState.txt")

    cliffordOutcomes =  QfeatQ.measurement_Controller( "clifford-loc-mat", true, system_size,subsystem_dim, cliffMeasScheme,"systemState.txt")

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


    predictions = round.(QfeatQ.observable_prediction_Controller("cShadowLocClifford-direct", "lo", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    #@show predictions
    #Now the direct ones:
    #predictions2 = round.(QfeatQ.observable_prediction_Controller("cShadowLocClifford-stab", "o", system_size, subsystem_dim, observableFile, cliffordOutcomes, cliffMeasScheme) ,digits=10)
    
    #stateCopy = zeros(subsystem_dim^system_size)
    #stateCopy[1] = 1
    #stateCopy[end] = 1/sqrt(2)
    #stateCopy = simpleCliffordToMat(tableauStateMemory, system_size, subsystem_dim)[:,1]

    trueExpectations = QfeatQ.observable_prediction_Controller("truthPauli", "o", system_size,subsystem_dim, observableFile, nothing, nothing, "systemState.txt")
    

    #predictions, truePredictions = observable_prediction_Controller("cShadowLocClifford-mat", "lo", system_size, "cliffordOutcomes.txt", observableFile, "systemState.txt", cliffordOutcomes, cliffMeasScheme)
    #trueExpectations = observable_prediction_Controller("truthPauli", "o", system_size, "cliffordOutcomes.txt", observableFile, "systemState.txt", cliffordOutcomes, cliffMeasScheme, nothing)

   
    error = abs.(predictions .- trueExpectations)
    @show error, mean(error)
    #predictionDiff =  abs.(predictions .- predictions2)
   # @show predictionDiff, mean(predictionDiff)
    return convert(Array{ComplexF64, 1},predictions),convert(Array{ComplexF64, 1},trueExpectations),  convert(Array{ComplexF64, 1}, error)
end

h5open("dicke2basic_simpleEW.dat", "w") do fid
system_size, subsystem_dim = 4,2

QfeatQ.generateSpinEW(string("./observables_spinEW_ext",subsystem_dim,"d.txt"),system_size,subsystem_dim)
#QfeatQ.generateSpinEW(string("./observables_SpinEntanglementWitness",subsystem_dim,"d.txt"),system_size,subsystem_dim)
#firstsqrtindex, clusterlengths,reducedObservables, mDic = QfeatQ.generateEW_cor(string("./observables_entanglementWitness",subsystem_dim,"d.txt"), 2,system_size,subsystem_dim, string("./observables_entanglementWitness",subsystem_dim,"d.txt"))

global firstsqrtindex
global clusterlengths

for nm in Int.(round.((10 .^(range(1.1,stop=4.8,length=25)))))
    for runi in 1:5
        GC.gc()

        p,t,e = mainExperimentController(system_size, subsystem_dim,nm, string("./observables_spinEW_ext",subsystem_dim,"d.txt")) #"../observables_entanglementWitness",3,"d.txt"
        #p,t,e = mainExperimentController(system_size, subsystem_dim,10000, string("./observables_SpinEntanglementWitness",subsystem_dim,"d.txt")) #"../observables_entanglementWitness",3,"d.txt"

        #p = QfeatQ.reduceMultiplets(p,true,mDic, reducedObservables)
        #t = QfeatQ.reduceMultiplets(p,true,mDic, reducedObservables)
        #e = QfeatQ.reduceMultiplets(p,true,mDic, reducedObservables)


        fid[string("errorArray",system_size,"q",subsystem_dim,"d",nm,"m", runi)] = e
        fid[string("predictionArray",system_size,"q",subsystem_dim,"d",nm,"m", runi)] = p
        fid[string("truePredictionArray", system_size,"q",subsystem_dim,"d",nm,"m", runi)] = t
        println("Finished iteration $runi and m= $nm")
    end
end
end
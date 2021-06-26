function readObservables(observableFile)
    """
    Assumes local pauli observables and weights are always assumed!
    """
    observableContent = readlines(observableFile)
    system_size = parse(Int, split(observableContent[1], " ")[1])
    subsystem_dim = parse(Int, split(observableContent[1], " ")[2])

    all_observables = []
    weightArray = []

    for line in observableContent[2:end]
        one_observable = []

        wBool = 1 #Are there weights?
        XZencoding = zeros((2*system_size, 2*system_size))

        for q_obs in split(line, " ")[4:end]
            #@show pos, xzpow, split.(string.(split(line, " ")[3:end]), "|")
            pos = parse(Int, split(q_obs, "|")[1])
            xzpow = split(q_obs, "|")[2]
            xpow, zpow = xzpow[2], xzpow[4]
            #@show pos, xpow, zpow
            XZencoding[pos] = mod(parse(Int,xpow), subsystem_dim)
            XZencoding[pos+system_size] = mod(parse(Int,zpow), subsystem_dim)
        end

        phase = mod(parse(Int, split(line, " ")[3]), 2*subsystem_dim)

        if wBool == 1
            push!(weightArray, parse(Float64, split(line, " ")[1])+im*parse(Float64, split(line, " ")[2])  )
        end

        one_observable = PauliOp(system_size, subsystem_dim,XZencoding, phase)
        push!(all_observables, one_observable)


        
    end
    return all_observables, weightArray, system_size
end

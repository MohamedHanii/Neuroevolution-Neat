package de.uni_passau.fim.se2.sbse.neat.chromosomes;

/**
 * Represents a neuron gene that is part of every NEAT chromosome.
 */
public class NeuronGene {

    // TODO: It's your job to implement this class.
    //  Please do not change the signature of the given constructor and methods and ensure to implement them.
    //  You can add additional methods, fields, and constructors if needed.

    private final int id;
    private final ActivationFunction activationFunction;
    private final NeuronType neuronType;

    /**
     * Creates a new neuron with the given ID and activation function.
     *
     * @param id                 The ID of the neuron.
     * @param activationFunction The activation function of the neuron.
     */
    public NeuronGene(int id, ActivationFunction activationFunction, NeuronType neuronType) {
        this.id = id;
        this.activationFunction = activationFunction;
        this.neuronType = neuronType;
    }

    public int getId() {
        return id;
    }

    public NeuronType getNeuronType() {
        return neuronType;
    }

    public double applyActivation(double input) {
        return switch (activationFunction) {
            case SIGMOID -> 1 / (1 + Math.exp(-input));
            case TANH -> Math.tanh(input);
            default -> input;
        };
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}

package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.ConnectionInnovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;

import java.util.*;

import static java.util.Objects.requireNonNull;

/**
 * Creates fully connected feed-forward neural networks consisting of one input and one output layer.
 */
public class NetworkGenerator {

    /**
     * The number of desired input neurons.
     */
    private final int inputSize;

    /**
     * The number of desired output neurons.
     */
    private final int outputSize;

    /**
     * The random number generator.
     */
    private final Random random;

    /**
     * The set of innovations that occurred so far in the search.
     * Novel innovations created during the generation of the network must be added to this set.
     */
    private final Set<Innovation> innovations;

    /**
     * Creates a new network generator.
     *
     * @param innovations The set of innovations that occurred so far in the search.
     * @param inputSize   The number of desired input neurons.
     * @param outputSize  The number of desired output neurons.
     * @param random      The random number generator.
     * @throws NullPointerException if the random number generator is {@code null}.
     */
    public NetworkGenerator(Set<Innovation> innovations, int inputSize, int outputSize, Random random) {
        this.innovations = requireNonNull(innovations);
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.random = requireNonNull(random);
    }

    /**
     * Generates a new fully connected feed-forward network chromosome.
     *
     * @return a new network chromosome.
     */
    public NetworkChromosome generate() {
        Map<Double, List<NeuronGene>> layerMap = new HashMap<>();
        List<NeuronGene> inputLayer = new ArrayList<>();
        List<NeuronGene> outputLayer = new ArrayList<>();

        int neuronIdCounter = 1;

        for (int i = 0; i < inputSize; i++) {
            inputLayer.add(new NeuronGene(neuronIdCounter++, ActivationFunction.NONE, NeuronType.INPUT));
        }

        inputLayer.add(new NeuronGene(neuronIdCounter++, ActivationFunction.NONE, NeuronType.BIAS));

        for (int j = 0; j < outputSize; j++) {
            outputLayer.add(new NeuronGene(neuronIdCounter++, ActivationFunction.TANH, NeuronType.OUTPUT));
        }

        layerMap.put(0.0, inputLayer);
        layerMap.put(1.0, outputLayer);

        List<ConnectionGene> connections = new ArrayList<>();

        for (NeuronGene inNeuron : inputLayer) {
            for (NeuronGene outNeuron : outputLayer) {
                int sourceId = inNeuron.getId();
                int targetId = outNeuron.getId();

                ConnectionInnovation innovation = findInnovation(sourceId, targetId);
                if (innovation == null) {
                    int innovationNumber = innovations.size() + 1;
                    innovation = new ConnectionInnovation(sourceId, targetId, innovationNumber);
                    innovations.add(innovation);
                }

                double weight = random.nextDouble() * 2 - 1;
                connections.add(new ConnectionGene(inNeuron, outNeuron, weight, true, innovation.getInnovationNumber()));
            }
        }

        return new NetworkChromosome(layerMap, connections);
    }

    public ConnectionInnovation findInnovation(int sourceId, int targetId) {
        ConnectionInnovation temp = new ConnectionInnovation(sourceId, targetId, 0);
        for (Innovation inv : innovations) {
            if (inv.equals(temp)) {
                return (ConnectionInnovation) inv;
            }
        }
        return null;
    }
}

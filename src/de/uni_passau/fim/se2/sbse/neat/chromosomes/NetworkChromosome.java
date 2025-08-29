package de.uni_passau.fim.se2.sbse.neat.chromosomes;


import java.util.*;

import static java.util.Objects.requireNonNull;

/**
 * Represents a network chromosome in the NEAT algorithm.
 */
public class NetworkChromosome implements Agent {

    // TODO: It's your job to implement this class.
    //  Please do not change the signature of the given constructor and methods and ensure to implement them.
    //  You can add additional methods, fields, and constructors if needed.

    public static final double INPUT_LAYER = 0;
    public static final double OUTPUT_LAYER = 1;

    /**
     * Maps the layer number to a list of neurons in that layer, with zero representing the input layer and one the output layer.
     * All hidden layers between the input and output layer are represented by values between zero and one.
     * For instance, if a new neuron gets added between the input and output layer, it might get the layer number 0.5.
     */
    private final Map<Double, List<NeuronGene>> layers;

    /**
     * Hosts all connections of the network.
     */
    private final List<ConnectionGene> connections;

    private double fitness;

    /**
     * Creates a new network chromosome with the given layers and connections.
     *
     * @param layers      The layers of the network.
     * @param connections The connections of the network.
     */
    public NetworkChromosome(Map<Double, List<NeuronGene>> layers, List<ConnectionGene> connections) {
        this.layers = requireNonNull(layers);
        this.connections = requireNonNull(connections);
    }

    public Map<Double, List<NeuronGene>> getLayers() {
        return layers;
    }

    public List<ConnectionGene> getConnections() {
        return connections;
    }

    @Override
    public List<Double> getOutput(List<Double> state) {
        Map<Integer, Double> neuronOutputs = new HashMap<>();

        for (int i = 0; i < state.size(); i++) {
            neuronOutputs.put(layers.get(INPUT_LAYER).get(i).getId(), state.get(i));
        }

        neuronOutputs.put(layers.get(INPUT_LAYER).get(state.size()).getId(), 1.0);

        List<Double> sortedLayers = new ArrayList<>(layers.keySet());
        Collections.sort(sortedLayers);

        for (double layer : sortedLayers) {
            if (layer == INPUT_LAYER) {
                continue;
            }

            for (NeuronGene neuron : layers.get(layer)) {
                double sum = 0.0;

                for (ConnectionGene conn : connections) {
                    if (conn.getTargetNeuron().getId() == neuron.getId() && Boolean.TRUE.equals(conn.getEnabled())) {
                        double sourceOutput = neuronOutputs.getOrDefault(conn.getSourceNeuron().getId(), 0.0);
                        sum += sourceOutput * conn.getWeight();
                    }
                }

                double activatedOutput = neuron.applyActivation(sum);
                neuronOutputs.put(neuron.getId(), activatedOutput);
            }
        }

        List<Double> outputs = new ArrayList<>();
        for (NeuronGene outputNeuron : layers.get(OUTPUT_LAYER)) {
            outputs.add(neuronOutputs.getOrDefault(outputNeuron.getId(), 0.0));
        }

        return outputs;
    }

    @Override
    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    @Override
    public double getFitness() {
        return fitness;
    }

    public NetworkChromosome copy() {
        Map<Double, List<NeuronGene>> newLayerMap = new HashMap<>();
        for (Map.Entry<Double, List<NeuronGene>> entry : layers.entrySet()) {
            newLayerMap.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }

        List<ConnectionGene> newConnections = new ArrayList<>(connections);
        return new NetworkChromosome(newLayerMap, newConnections);
    }

    public int getMaxNeuronId() {
        int max = 0;
        for (List<NeuronGene> neuronList : layers.values()) {
            for (NeuronGene neuron : neuronList) {
                if (neuron.getId() > max) {
                    max = neuron.getId();
                }
            }
        }
        return max;
    }

    public double getLayerForNeuron(NeuronGene neuron) {

        for (Map.Entry<Double, List<NeuronGene>> entry : layers.entrySet()) {
            List<NeuronGene> neuronGenes = entry.getValue();

            for (NeuronGene n : neuronGenes) {
                if (n.getId() == neuron.getId()) {
                    return entry.getKey();
                }
            }
        }
        throw new IllegalArgumentException("Neuron does not exist");
    }

    public List<NeuronGene> getAllNeurons() {
        List<NeuronGene> allNeurons = new ArrayList<>();
        for (List<NeuronGene> neurons : layers.values()) {
            allNeurons.addAll(neurons);
        }
        return allNeurons;
    }

    public void addNeuronToLevel(NeuronGene neuron, double level) {
        layers.computeIfAbsent(level, k -> new ArrayList<>()).add(neuron);
    }


    public Map<Integer, ConnectionGene> getConnectionMap() {
        Map<Integer, ConnectionGene> connectionMap = new HashMap<>();
        for (ConnectionGene connection : connections) {
            connectionMap.put(connection.getInnovationNumber(), connection);
        }
        return connectionMap;
    }
}

package de.uni_passau.fim.se2.sbse.neat.mutation;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.ConnectionInnovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static java.util.Objects.requireNonNull;

/**
 * Implements the mutation operator for the Neat algorithm, which applies four types of mutations based on probabilities:
 * 1. Add a new neuron to the network.
 * 2. Add a new connection to the network.
 * 3. Mutate the weights of the connections in the network.
 * 4. Toggle the enabled status of a connection in the network.
 */
public class NeatMutation implements Mutation<NetworkChromosome> {


    private static final double PROB_ADD_NEURON = 0.03;
    private static final double PROB_ADD_CONNECTION = 0.05;
    private static final double PROB_MUTATE_WEIGHTS = 0.8;
    private static final double PROB_TOGGLE_CONNECTION = 0.01;

    /**
     * The random number generator to use.
     */
    private final Random random;

    /**
     * The list of innovations that occurred so far in the search.
     * Since Neat applies mutations that change the structure of the network,
     * the set of innovations must be updated appropriately.
     */
    private final Set<Innovation> innovations;

    /**
     * Constructs a new NeatMutation with the given random number generator and the list of innovations that occurred so far in the search.
     *
     * @param innovations The list of innovations that occurred so far in the search.
     * @param random      The random number generator.
     */
    public NeatMutation(Set<Innovation> innovations, Random random) {
        this.innovations = requireNonNull(innovations);
        this.random = requireNonNull(random);
    }


    /**
     * Applies mutation to the given network chromosome.
     * If a structural mutation is applied, no further non-structural mutations are applied.
     * Otherwise, the weights of the connections are mutated and/or the enabled status of a connection is toggled.
     *
     * @param parent The parent chromosome to mutate.
     * @return The mutated parent chromosome.
     */
    @Override
    public NetworkChromosome apply(NetworkChromosome parent) {
        NetworkChromosome offspring = parent.copy();
        double chance = random.nextDouble();

        if (chance < PROB_ADD_NEURON) {
            offspring = addNeuron(offspring);
        }

        if (chance < PROB_ADD_CONNECTION) {
            offspring = addConnection(offspring);
        }

        if (chance < PROB_TOGGLE_CONNECTION) {
            offspring = toggleConnection(offspring);
        }

        if (chance < PROB_MUTATE_WEIGHTS) {
            offspring = mutateWeights(offspring);
        }

        return offspring;
    }


    /**
     * Adds a hidden neuron to the given network chromosome by splitting an existing connection.
     * The connection to be split is chosen randomly from the list of connections in the network chromosome.
     * The connection is disabled and two new connections are added to the network chromosome:
     * One connection with a weight of 1.0 from the source neuron of the split connection to the new hidden neuron,
     * and one connection with the weight of the split connection from the new hidden neuron to the target neuron of the split connection.
     * <p>
     * Since this mutation changes the structure of the network,
     * novel innovations for the new connections must be created if the same mutation has not occurred before.
     * If the same innovation has occurred before, the corresponding innovation numbers must be reused.
     *
     * @param parent The network chromosome to which the new neuron and connections will be added.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome addNeuron(NetworkChromosome parent) {
        NetworkChromosome offspring = parent.copy();
        Map<Double, List<NeuronGene>> layers = offspring.getLayers();
        List<ConnectionGene> connections = offspring.getConnections();


        // Disable the selected connection
        int randomIndex = random.nextInt(connections.size());
        ConnectionGene selectedConnection = connections.get(randomIndex);
        ConnectionGene disabledConnection = new ConnectionGene(
                selectedConnection.getSourceNeuron(),
                selectedConnection.getTargetNeuron(),
                selectedConnection.getWeight(),
                false,
                selectedConnection.getInnovationNumber()
        );


        // Create a new neuron
        NeuronGene newNeuron = new NeuronGene(
                parent.getMaxNeuronId() + 1,
                ActivationFunction.TANH,
                NeuronType.HIDDEN
        );


        int firstInnovationNumber = getInnovationNumberForConnection(
                selectedConnection.getSourceNeuron().getId(),
                newNeuron.getId()
        );
        int secondInnovationNumber = getInnovationNumberForConnection(
                newNeuron.getId(),
                selectedConnection.getTargetNeuron().getId()
        );

        // Create new connections (split the original connection)
        ConnectionGene firstConnection = new ConnectionGene(
                selectedConnection.getSourceNeuron(),
                newNeuron,
                1.0,
                true,
                firstInnovationNumber
        );
        ConnectionGene secondConnection = new ConnectionGene(
                newNeuron,
                selectedConnection.getTargetNeuron(),
                selectedConnection.getWeight(),
                true,
                secondInnovationNumber
        );

        // Determine the layer for the new neuron
        Double newLayerNumber = getRandomValueBetweenInputAndOutput(
                layers,
                selectedConnection.getSourceNeuron().getId(),
                selectedConnection.getTargetNeuron().getId()
        );

        offspring.addNeuronToLevel(newNeuron, newLayerNumber);

        connections.set(randomIndex, disabledConnection);
        connections.add(firstConnection);
        connections.add(secondConnection);
        return offspring;
    }

    /**
     * Adds a connection to the given network chromosome.
     * The source neuron of the connection is chosen randomly from the list of neurons in the network chromosome,
     * excluding output neurons.
     * The target neuron of the connection is chosen randomly from the list of neurons in the network chromosome,
     * excluding input and bias neurons.
     * The connection is added to the network chromosome with a random weight between -1.0 and 1.0.
     * The connection must not be recurrent.
     * <p>
     * Since this mutation changes the structure of the network,
     * novel innovations for the new connection must be created if the same mutation has not occurred before.
     * If the same innovation has occurred before, the corresponding innovation number must be reused.
     *
     * @param parent The network chromosome to which the new connection will be added.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome addConnection(NetworkChromosome parent) {
        NetworkChromosome offspring = parent.copy();

        List<NeuronGene> allNeurons = offspring.getAllNeurons();

        List<NeuronGene> sourceCandidates = allNeurons.stream()
                .filter(n -> n.getNeuronType() != NeuronType.OUTPUT)
                .toList();

        List<NeuronGene> targetCandidates = allNeurons.stream()
                .filter(n -> n.getNeuronType() != NeuronType.INPUT && n.getNeuronType() != NeuronType.BIAS)
                .toList();


        for (int attempts = 0; attempts < 100; attempts++) {
            NeuronGene source = sourceCandidates.get(random.nextInt(sourceCandidates.size()));
            NeuronGene target = targetCandidates.get(random.nextInt(targetCandidates.size()));


            if (offspring.getLayerForNeuron(source) >= offspring.getLayerForNeuron(target)) {
                continue;
            }


            boolean exists = offspring.getConnections().stream()
                    .anyMatch(c -> c.getSourceNeuron().getId() == source.getId()
                            && c.getTargetNeuron().getId() == target.getId());
            if (exists) {
                continue;
            }

            int innovNumber = getInnovationNumberForConnection(source.getId(), target.getId());

            double weight = random.nextDouble() * 2 - 1;
            ConnectionGene newConn = new ConnectionGene(source, target, weight, true, innovNumber);
            offspring.getConnections().add(newConn);
            return offspring;
        }

        return offspring;
    }

    /**
     * Mutates the weights of the connections in the given network chromosome.
     * The weight is mutated by adding gaussian noise to every weight in the network chromosome.
     *
     * @param parent The network chromosome to mutate.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome mutateWeights(NetworkChromosome parent) {
        NetworkChromosome offspring = parent.copy();

        offspring.getConnections().replaceAll(connectionGene -> new ConnectionGene(
                connectionGene.getSourceNeuron(),
                connectionGene.getTargetNeuron(),
                connectionGene.getWeight() + random.nextGaussian() * 0.1,
                connectionGene.getEnabled(),
                connectionGene.getInnovationNumber()
        ));
        return offspring;
    }


    /**
     * Toggles the enabled status of a random connection in the given network chromosome.
     *
     * @param parent The network chromosome to mutate.
     * @return The mutated network chromosome.
     */
    public NetworkChromosome toggleConnection(NetworkChromosome parent) {
        NetworkChromosome offspring = parent.copy();
        int index = random.nextInt(offspring.getConnections().size());
        ConnectionGene conn = offspring.getConnections().get(index);

        ConnectionGene toggled = new ConnectionGene(
                conn.getSourceNeuron(),
                conn.getTargetNeuron(),
                conn.getWeight(),
                !conn.getEnabled(),
                conn.getInnovationNumber()
        );

        offspring.getConnections().set(index, toggled);
        return offspring;
    }

    private int getInnovationNumberForConnection(int sourceId, int targetId) {
        for (Innovation innovation : innovations) {
            if (((ConnectionInnovation) innovation).getSource() == sourceId && ((ConnectionInnovation) innovation).getTarget() == targetId) {
                return ((ConnectionInnovation) innovation).getInnovationNumber();
            }
        }

        int newInnovationNumber = innovations.size() + 1;
        innovations.add(new ConnectionInnovation(sourceId, targetId, newInnovationNumber));

        return newInnovationNumber;
    }

    public Double getRandomValueBetweenInputAndOutput(Map<Double, List<NeuronGene>> layers, double inputId, double outputId) {
        Double inputValue = null;
        Double outputValue = null;

        for (Map.Entry<Double, List<NeuronGene>> entry : layers.entrySet()) {
            List<NeuronGene> neuronGenes = entry.getValue();

            // Check if the list contains inputId
            for (NeuronGene neuron : neuronGenes) {
                if (neuron.getId() == inputId) {
                    inputValue = entry.getKey();
                }
                if (neuron.getId() == outputId) {
                    outputValue = entry.getKey();
                }
            }
        }

        // If either inputId or outputId is not found, log an error or throw an exception
        if (inputValue == null || outputValue == null) {
            throw new IllegalArgumentException("Either inputId or outputId not found in the layers map.");
        }

        // Generate a random value between the two values
        double min = Math.min(inputValue, outputValue);
        double max = Math.max(inputValue, outputValue);
        return min + (max - min) * random.nextDouble();
    }


}

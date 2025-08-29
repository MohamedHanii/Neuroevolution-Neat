package de.uni_passau.fim.se2.sbse.neat.crossover;


import de.uni_passau.fim.se2.sbse.neat.chromosomes.ConnectionGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronGene;

import java.util.*;

import static java.util.Objects.requireNonNull;

/**
 * A NEAT crossover operation that is used by the NEAT algorithm to combine two parent chromosomes.
 */
public class NeatCrossover implements Crossover<NetworkChromosome> {

    /**
     * The random number generator to use.
     */
    private final Random random;

    /**
     * Creates a new NEAT crossover operator with the given random number generator.
     *
     * @param random The random number generator to use.
     */
    public NeatCrossover(Random random) {
        this.random = requireNonNull(random);
    }

    /**
     * Applies a crossover operation to the given parent chromosomes by combining their genes.
     * During the crossover operation, we determine for each gene whether it is a matching gene or a disjoint/excess gene.
     * Matching genes are inherited with a 50% chance from either parent,
     * while disjoint/excess genes are only inherited from the fitter parent.
     *
     * @param parent1 The first crossover parent.
     * @param parent2 The second crossover parent.
     * @return A new chromosome resulting from the crossover operation.
     */
    @Override
    public NetworkChromosome apply(NetworkChromosome parent1, NetworkChromosome parent2) {
        NetworkChromosome fitter, other;
        if (parent1.getFitness() >= parent2.getFitness()) {
            fitter = parent1;
            other = parent2;
        } else {
            fitter = parent2;
            other = parent1;
        }

        Map<Integer, ConnectionGene> fitterGenes = fitter.getConnectionMap();
        Map<Integer, ConnectionGene> otherGenes = other.getConnectionMap();

        Set<Integer> allInnovations = new HashSet<>();
        allInnovations.addAll(fitterGenes.keySet());
        allInnovations.addAll(otherGenes.keySet());

        List<ConnectionGene> offspringConnections = new ArrayList<>();

        for (Integer innov : allInnovations) {
            ConnectionGene geneFromFitter = fitterGenes.get(innov);
            ConnectionGene geneFromOther = otherGenes.get(innov);

            if (geneFromFitter != null && geneFromOther != null) {
                ConnectionGene chosen = random.nextBoolean() ? geneFromFitter.clone() : geneFromOther.clone();
                offspringConnections.add(chosen);
            } else if (geneFromFitter != null) {
                offspringConnections.add(geneFromFitter.clone());
            }
        }

        Map<Double, List<NeuronGene>> offspringLayerMap = fitter.getLayers();

        for (ConnectionGene connectionGene : offspringConnections) {
            NeuronGene source = connectionGene.getSourceNeuron();
            NeuronGene target = connectionGene.getTargetNeuron();
            double sourceLayerNumber = fitter.getLayerForNeuron(source);
            double targetLayerNumber = fitter.getLayerForNeuron(target);
            if (sourceLayerNumber != 0.0)
                addNeuronToMap(offspringLayerMap, sourceLayerNumber, source);

            if (targetLayerNumber != 1.0)
                addNeuronToMap(offspringLayerMap, targetLayerNumber, target);
        }

        return new NetworkChromosome(offspringLayerMap, offspringConnections);
    }

    public void addNeuronToMap(Map<Double, List<NeuronGene>> offspringLayerMap, Double key, NeuronGene givenNeuron) {
        if (offspringLayerMap.containsKey(key)) {
            List<NeuronGene> neuronGenesList = offspringLayerMap.get(key);
            boolean idExists = false;
            for (NeuronGene neuron : neuronGenesList) {
                if (neuron.getId() == givenNeuron.getId()) {
                    idExists = true;
                    break;
                }
            }

            if (!idExists) {
                neuronGenesList.add(givenNeuron);
            }
        } else {
            List<NeuronGene> newList = new ArrayList<>();
            newList.add(givenNeuron);
            offspringLayerMap.put(key, newList);
        }
    }

}

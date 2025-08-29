package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class NeatAlgorithm implements Neuroevolution {
    private final int populationSize;
    private final int maxGenerations;
    private final Random random;
    private final NeatMutation mutation;
    private final NeatCrossover crossover;
    private List<NetworkChromosome> population;
    private int currentGeneration;
    private double deltaThreshold;
    private final int desiredSpeciesCount;
    private final Set<Innovation> innovations;

    public NeatAlgorithm(int populationSize, int maxGenerations, Random random) {
        this.populationSize = populationSize;
        this.maxGenerations = maxGenerations;
        this.random = random;

        this.innovations = new HashSet<>();
        this.mutation = new NeatMutation(innovations, random);
        this.crossover = new NeatCrossover(random);
        this.population = new ArrayList<>();

        currentGeneration = 0;
        desiredSpeciesCount = 10;
        deltaThreshold = 2.5;

    }

    @Override
    public Agent solve(Environment environment) {
        NetworkChromosome bestAgent = null;
        NetworkGenerator generator = new NetworkGenerator(innovations, environment.getState().size(), environment.actionInputSize(), random);

        for (int i = 0; i < populationSize; i++) {
            NetworkChromosome chromosome = generator.generate();
            population.add(chromosome);
        }


        while (currentGeneration < maxGenerations) {
            // Evaluation
            for (NetworkChromosome chromosome : population) {
                double fitness = environment.evaluate(chromosome);
                chromosome.setFitness(fitness);
                if (bestAgent == null || fitness > bestAgent.getFitness()) {
                    bestAgent = chromosome;
                }
                if (environment.solved(bestAgent)){
                    return bestAgent;
                }
            }

            // Convert to Species
            List<Species> speciesList = assignSpecies(population);

            // Fitness Sharing
            double totalSharedFitness = 0.0;
            for (Species species : speciesList) {
                totalSharedFitness += species.getSharedFitness();
            }

            for (Species species : speciesList) {
                int offspringCount = (int) Math.round((species.getSharedFitness() / totalSharedFitness) * populationSize);
                species.setOffspringCount(offspringCount);
            }

            // Adjust Threshold
            adjustThreshold(speciesList.size());

            // Cross-over and mutation
            List<NetworkChromosome> nextGeneration = new ArrayList<>(populationSize);


            for (Species species : speciesList) {
                int offspringCount = species.getOffspringCount();

                if (offspringCount > 0) {
                    List<NetworkChromosome> sortedMembers = species.getMembers().stream()
                            .sorted((a, b) -> Double.compare(b.getFitness(), a.getFitness()))
                            .toList();

                    NetworkChromosome elite = sortedMembers.get(0).copy();
                    nextGeneration.add(elite);
                    offspringCount--;
                }

                while (offspringCount > 0 && nextGeneration.size() < populationSize) {
                    NetworkChromosome parent1 = species.selectParent(random);
                    NetworkChromosome parent2 = species.selectParent(random);
                    NetworkChromosome child;

                    // Apply crossover or retain the best of two parents
                    if (random.nextDouble() < 0.75) {
                        child = crossover.apply(parent1, parent2);
                    } else {
                        child = (parent1.getFitness() >= parent2.getFitness() ? parent1.copy() : parent2.copy());
                    }

                    // Apply mutation to the child
                    child = mutation.apply(child);
                    nextGeneration.add(child);
                    offspringCount--;

                    if (nextGeneration.size() >= populationSize) {
                        break;
                    }
                }

                // Ensure that the population size doesn't exceed the specified size and break early if necessary
                if (nextGeneration.size() >= populationSize) {
                    break;
                }
            }

            while (nextGeneration.size() < populationSize) {
                Species randomSpecies = speciesList.get(random.nextInt(speciesList.size()));
                NetworkChromosome parent = randomSpecies.selectParent(random);
                NetworkChromosome child = mutation.apply(parent.copy());
                nextGeneration.add(child);
            }

            population = nextGeneration;
            currentGeneration++;
        }

        return bestAgent;
    }

    List<Species> assignSpecies(List<NetworkChromosome> agents) {
        List<Species> speciesList = new ArrayList<>();

        for (NetworkChromosome agent : agents) {
            boolean assigned = false;

            for (Species species : speciesList) {
                double distance = computeCompatibilityDistance(agent, species.getRepresentative());
                if (distance < deltaThreshold) {
                    species.addMember(agent);
                    assigned = true;
                    break;
                }
            }

            if (!assigned) {
                Species newSpecies = new Species(random);
                newSpecies.addMember(agent);
                speciesList.add(newSpecies);
            }
        }

        return speciesList;
    }

    void adjustThreshold(int speciesCount) {
        if (speciesCount < desiredSpeciesCount) {
            deltaThreshold -= 0.3;
        } else if (speciesCount > desiredSpeciesCount) {
            deltaThreshold += 0.3;
        }
    }

    double computeCompatibilityDistance(NetworkChromosome a, NetworkChromosome b) {
        Map<Integer, ConnectionGene> genesA = buildGeneMap(a);
        Map<Integer, ConnectionGene> genesB = buildGeneMap(b);

        Set<Integer> innovationNumbers = Stream.concat(genesA.keySet().stream(), genesB.keySet().stream())
                .collect(Collectors.toSet());


        int matching = 0, disjoint = 0, excess = 0;
        double weightDiffSum = 0.0;

        int maxInnovA = genesA.keySet().stream().max(Integer::compareTo).orElse(0);
        int maxInnovB = genesB.keySet().stream().max(Integer::compareTo).orElse(0);
        int lowerMaxInnovation = Math.min(maxInnovA, maxInnovB);


        for (int innov : innovationNumbers) {
            boolean inA = genesA.containsKey(innov);
            boolean inB = genesB.containsKey(innov);

            if (inA && inB) {
                matching++;
                double diff = Math.abs(genesA.get(innov).getWeight() - genesB.get(innov).getWeight());
                weightDiffSum += diff;
            } else {
                if (innov <= lowerMaxInnovation) {
                    disjoint++;
                } else {
                    excess++;
                }
            }
        }

        double avgWeightDiff = matching > 0 ? weightDiffSum / matching : 0.0;

        int normalizationFactor = Math.max(a.getConnections().size(), b.getConnections().size());
        if (normalizationFactor < 20) {
            normalizationFactor = 1;
        }

        double c1 = 1.0;
        double c2 = 1.0;
        double c3 = 0.4;

        return (c1 * disjoint / normalizationFactor) +
                (c2 * excess / normalizationFactor) +
                (c3 * avgWeightDiff);
    }

    private Map<Integer, ConnectionGene> buildGeneMap(NetworkChromosome chromosome) {
        Map<Integer, ConnectionGene> geneMap = new HashMap<>();
        for (ConnectionGene gene : chromosome.getConnections()) {
            int innovationNumber = gene.getInnovationNumber();
            if (!geneMap.containsKey(innovationNumber)) {
                geneMap.put(innovationNumber, gene);
            }
        }
        return geneMap;
    }

    @Override
    public int getGeneration() {
        return currentGeneration;
    }



    public double getDeltaThreshold() {
        return deltaThreshold;
    }

    public void setDeltaThreshold(double deltaThreshold) {
        this.deltaThreshold = deltaThreshold;
    }

    public List<NetworkChromosome> getPopulation() {
        return population;
    }

}

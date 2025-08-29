package de.uni_passau.fim.se2.sbse.neat.algorithms;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import de.uni_passau.fim.se2.sbse.neat.crossover.NeatCrossover;
import de.uni_passau.fim.se2.sbse.neat.environments.Environment;
import de.uni_passau.fim.se2.sbse.neat.mutation.NeatMutation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class NeatAlgorithmTest {

    private Random random;
    private Environment environment;
    private NetworkGenerator generator;
    private NeatCrossover crossover;
    private NeatMutation mutation;
    private NeuronGene inputNeuron, outputNeuron;
    private ConnectionGene connection;

    @BeforeEach
    public void setUp() {
        random = new Random(42);
        environment = mock(Environment.class);
        generator = mock(NetworkGenerator.class);
        crossover = mock(NeatCrossover.class);
        mutation = mock(NeatMutation.class);

        inputNeuron = new NeuronGene(1, ActivationFunction.TANH, NeuronType.INPUT);
        outputNeuron = new NeuronGene(2, ActivationFunction.TANH, NeuronType.OUTPUT);
        connection = new ConnectionGene(inputNeuron, outputNeuron, 1.0, true, 1);
    }


    @Test
    public void testAdjustThreshold() {
        NeatAlgorithm algo = new NeatAlgorithm(10, 1, random);
        algo.setDeltaThreshold(2.5);

        algo.adjustThreshold(8); // Below 10
        assertEquals(2.2, algo.getDeltaThreshold(), 0.001);

        algo.setDeltaThreshold(2.5);
        algo.adjustThreshold(12); // Above 10
        assertEquals(2.8, algo.getDeltaThreshold(), 0.001);

        algo.setDeltaThreshold(2.5);
        algo.adjustThreshold(10); // Equal to 10
        assertEquals(2.5, algo.getDeltaThreshold(), 0.001);
    }

    @Test
    public void testComputeCompatibilityDistance() {
        NeatAlgorithm algo = new NeatAlgorithm(10, 1, random);
        NetworkChromosome chrom1 = createChromosomeWithConnections(
                Arrays.asList(new ConnectionGene(inputNeuron, outputNeuron, 1.0, true, 1),
                        new ConnectionGene(inputNeuron, outputNeuron, 2.0, true, 2))
        );
        NetworkChromosome chrom2 = createChromosomeWithConnections(
                Arrays.asList(new ConnectionGene(inputNeuron, outputNeuron, 1.5, true, 1),
                        new ConnectionGene(inputNeuron, outputNeuron, 3.0, true, 3))
        );

        double distance = algo.computeCompatibilityDistance(chrom1, chrom2);
        // Matching: innov 1 (diff = 0.5)
        // Disjoint: innov 2, 3
        // Excess: none
        // N = 2 < 20, so normalizationFactor = 1
        // Distance = (1.0 * 2 / 1) + (1.0 * 0 / 1) + (0.4 * 0.5) = 2.2
        assertEquals(2.2, distance, 0.001);
    }

    @Test
    public void testSolveFullGenerations() {
        Random mockRandom = mock(Random.class);
        when(mockRandom.nextDouble()).thenReturn(0.8); // Avoid crossover (0.8 >= 0.75)
        when(mockRandom.nextInt(anyInt())).thenReturn(0); // Consistent species/parent selection
        NeatAlgorithm algo = new NeatAlgorithm(3, 2, mockRandom);
        when(environment.getState()).thenReturn(Arrays.asList(1.0));
        when(environment.actionInputSize()).thenReturn(1);
        NetworkChromosome chrom1 = createChromosome(10.0, 1);
        NetworkChromosome chrom2 = createChromosome(20.0, 2);
        NetworkChromosome chrom3 = createChromosome(30.0, 3);
        when(generator.generate()).thenReturn(chrom1, chrom2, chrom3);
        when(environment.evaluate(any(NetworkChromosome.class))).thenAnswer(inv -> {
            NetworkChromosome chrom = inv.getArgument(0);
            int innov = chrom.getConnections().get(0).getInnovationNumber();
            return switch (innov) {
                case 1 -> 10.0;
                case 2 -> 20.0;
                case 3 -> 30.0;
                default -> 0.0;
            };
        });
        when(environment.solved(any())).thenReturn(false);
        when(crossover.apply(any(), any())).thenAnswer(inv -> chrom2.copy());
        when(mutation.apply(any())).thenAnswer(inv -> chrom3.copy());

        Agent bestAgent = algo.solve(environment);
        assertEquals(10.0, bestAgent.getFitness(), 0.001);
        assertEquals(2, algo.getGeneration());
        assertEquals(3, algo.getPopulation().size());
    }

    @Test
    public void testGettersAndSetters() {
        NeatAlgorithm algo = new NeatAlgorithm(5, 3, random);
        algo.setDeltaThreshold(3.0);
        assertEquals(3.0, algo.getDeltaThreshold(), 0.001);

        List<NetworkChromosome> population = algo.getPopulation();
        assertNotNull(population);
        assertTrue(population.isEmpty());
    }

    // Helper methods
    private NetworkChromosome createChromosome(double fitness, int innov) {
        NetworkChromosome chrom = new NetworkChromosome(
                Map.of(0.0, Collections.singletonList(inputNeuron),
                        1.0, Collections.singletonList(outputNeuron)),
                Collections.singletonList(new ConnectionGene(inputNeuron, outputNeuron, 1.0, true, innov))
        );
        chrom.setFitness(fitness);
        return chrom;
    }

    private NetworkChromosome createChromosomeWithConnections(List<ConnectionGene> connections) {
        return new NetworkChromosome(
                Map.of(0.0, Collections.singletonList(inputNeuron),
                        1.0, Collections.singletonList(outputNeuron)),
                connections
        );
    }
}

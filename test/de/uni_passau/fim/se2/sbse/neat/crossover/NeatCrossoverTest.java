package de.uni_passau.fim.se2.sbse.neat.crossover;

import de.uni_passau.fim.se2.sbse.neat.chromosomes.ConnectionGene;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NetworkChromosome;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.NeuronGene;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class NeatCrossoverTest {

    private Random random;

    @BeforeEach
    public void setUp() {
        random = new Random(42); // Fixed seed for deterministic behavior
    }

    @Test
    public void testCrossoverParent1Fitter() {
        // Create neurons
        NeuronGene neuron1 = new NeuronGene(1, null, null);
        NeuronGene neuron2 = new NeuronGene(2, null, null);
        NeuronGene neuron3 = new NeuronGene(3, null, null);
        NeuronGene neuron4 = new NeuronGene(4, null, null);
        NeuronGene neuron5 = new NeuronGene(5, null, null);

        // Parent 1 (fitter)
        Map<Double, List<NeuronGene>> layers1 = new HashMap<>();
        layers1.put(0.0, Arrays.asList(neuron1, neuron2));
        layers1.put(0.5, Arrays.asList(neuron4));
        layers1.put(1.0, Arrays.asList(neuron3));
        List<ConnectionGene> connections1 = Arrays.asList(
                new ConnectionGene(neuron1, neuron3, 1.0, true, 1),
                new ConnectionGene(neuron2, neuron3, 1.0, true, 2),
                new ConnectionGene(neuron1, neuron4, 1.0, true, 3),
                new ConnectionGene(neuron4, neuron3, 1.0, true, 4)
        );
        NetworkChromosome parent1 = new NetworkChromosome(layers1, connections1);
        parent1.setFitness(10.0);

        // Parent 2
        Map<Double, List<NeuronGene>> layers2 = new HashMap<>();
        layers2.put(0.0, Arrays.asList(neuron1, neuron2));
        layers2.put(0.5, Arrays.asList(neuron5));
        layers2.put(1.0, Arrays.asList(neuron3));
        List<ConnectionGene> connections2 = Arrays.asList(
                new ConnectionGene(neuron1, neuron3, 2.0, true, 1),
                new ConnectionGene(neuron2, neuron3, 2.0, true, 2),
                new ConnectionGene(neuron2, neuron5, 2.0, true, 5),
                new ConnectionGene(neuron5, neuron3, 2.0, true, 6)
        );
        NetworkChromosome parent2 = new NetworkChromosome(layers2, connections2);
        parent2.setFitness(5.0);

        // Apply crossover
        NeatCrossover crossover = new NeatCrossover(random);
        NetworkChromosome offspring = crossover.apply(parent1, parent2);

        // Assertions
        assertOffspringConnections(offspring, parent1, parent2, true,neuron1);
        assertOffspringLayers(offspring, neuron4, neuron5);
    }

    @Test
    public void testCrossoverParent2Fitter() {
        // Create neurons
        NeuronGene neuron1 = new NeuronGene(1, null, null);
        NeuronGene neuron2 = new NeuronGene(2, null, null);
        NeuronGene neuron3 = new NeuronGene(3, null, null);
        NeuronGene neuron4 = new NeuronGene(4, null, null);
        NeuronGene neuron5 = new NeuronGene(5, null, null);

        // Parent 1
        Map<Double, List<NeuronGene>> layers1 = new HashMap<>();
        layers1.put(0.0, Arrays.asList(neuron1, neuron2));
        layers1.put(0.5, Arrays.asList(neuron4));
        layers1.put(1.0, Arrays.asList(neuron3));
        List<ConnectionGene> connections1 = Arrays.asList(
                new ConnectionGene(neuron1, neuron3, 1.0, true, 1),
                new ConnectionGene(neuron2, neuron3, 1.0, true, 2),
                new ConnectionGene(neuron1, neuron4, 1.0, true, 3),
                new ConnectionGene(neuron4, neuron3, 1.0, true, 4)
        );
        NetworkChromosome parent1 = new NetworkChromosome(layers1, connections1);
        parent1.setFitness(5.0);

        // Parent 2 (fitter)
        Map<Double, List<NeuronGene>> layers2 = new HashMap<>();
        layers2.put(0.0, Arrays.asList(neuron1, neuron2));
        layers2.put(0.5, Arrays.asList(neuron5));
        layers2.put(1.0, Arrays.asList(neuron3));
        List<ConnectionGene> connections2 = Arrays.asList(
                new ConnectionGene(neuron1, neuron3, 2.0, true, 1),
                new ConnectionGene(neuron2, neuron3, 2.0, true, 2),
                new ConnectionGene(neuron2, neuron5, 2.0, true, 5),
                new ConnectionGene(neuron5, neuron3, 2.0, true, 6)
        );
        NetworkChromosome parent2 = new NetworkChromosome(layers2, connections2);
        parent2.setFitness(10.0);

        // Apply crossover
        NeatCrossover crossover = new NeatCrossover(random);
        NetworkChromosome offspring = crossover.apply(parent1, parent2);

        // Assertions
        assertOffspringConnections(offspring, parent1, parent2, false,neuron1);
        assertOffspringLayers(offspring, neuron5, neuron4);
    }

    private void assertOffspringConnections(NetworkChromosome offspring, NetworkChromosome parent1, NetworkChromosome parent2, boolean parent1Fitter, NeuronGene neuron1) {
        List<ConnectionGene> offspringConnections = offspring.getConnections();
        assertEquals(4, offspringConnections.size());

        Map<Integer, ConnectionGene> offspringConnMap = new HashMap<>();
        for (ConnectionGene conn : offspringConnections) {
            offspringConnMap.put(conn.getInnovationNumber(), conn);
        }

        // Check innovation numbers
        assertTrue(offspringConnMap.containsKey(1));
        assertTrue(offspringConnMap.containsKey(2));
        if (parent1Fitter) {
            assertTrue(offspringConnMap.containsKey(3));
            assertTrue(offspringConnMap.containsKey(4));
            assertFalse(offspringConnMap.containsKey(5));
            assertFalse(offspringConnMap.containsKey(6));

            assertEquals(1.0, offspringConnMap.get(3).getWeight(), 0.001);
            assertEquals(1.0, offspringConnMap.get(4).getWeight(), 0.001);
        } else {
            assertTrue(offspringConnMap.containsKey(5));
            assertTrue(offspringConnMap.containsKey(6));
            assertFalse(offspringConnMap.containsKey(3));
            assertFalse(offspringConnMap.containsKey(4));

            assertEquals(2.0, offspringConnMap.get(5).getWeight(), 0.001);
            assertEquals(2.0, offspringConnMap.get(6).getWeight(), 0.001);
        }

        // Check weights for matching genes
        double weight1 = offspringConnMap.get(1).getWeight();
        assertTrue(weight1 == 1.0 || weight1 == 2.0);
        double weight2 = offspringConnMap.get(2).getWeight();
        assertTrue(weight2 == 1.0 || weight2 == 2.0);

        // Check cloning
        List<ConnectionGene> fitterConnections = parent1Fitter ? parent1.getConnections() : parent2.getConnections();
        assertNotSame(fitterConnections.get(0), offspringConnMap.get(1));
        assertSame(offspringConnMap.get(1).getSourceNeuron(), neuron1);
    }

    private void assertOffspringLayers(NetworkChromosome offspring, NeuronGene expectedHidden, NeuronGene unexpectedHidden) {
        Map<Double, List<NeuronGene>> layers = offspring.getLayers();
        assertEquals(3, layers.size());
        assertTrue(layers.containsKey(0.0));
        assertTrue(layers.containsKey(0.5));
        assertTrue(layers.containsKey(1.0));

        List<NeuronGene> inputLayer = layers.get(0.0);
        assertEquals(2, inputLayer.size());
        assertTrue(inputLayer.stream().anyMatch(n -> n.getId() == 1));
        assertTrue(inputLayer.stream().anyMatch(n -> n.getId() == 2));

        List<NeuronGene> hiddenLayer = layers.get(0.5);
        assertEquals(1, hiddenLayer.size());
        assertTrue(hiddenLayer.contains(expectedHidden));

        List<NeuronGene> outputLayer = layers.get(1.0);
        assertEquals(1, outputLayer.size());
        assertTrue(outputLayer.stream().anyMatch(n -> n.getId() == 3));

        // Check that unexpected neuron is not present
        for (List<NeuronGene> neuronList : layers.values()) {
            assertFalse(neuronList.contains(unexpectedHidden));
        }
    }
}
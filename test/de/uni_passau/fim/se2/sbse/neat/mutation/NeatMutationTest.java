package de.uni_passau.fim.se2.sbse.neat.mutation;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import de.uni_passau.fim.se2.sbse.neat.chromosomes.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class NeatMutationTest {

    private Set<Innovation> innovations;
    private Random random;
    private NetworkChromosome parent;
    private NeuronGene inputNeuron, inputNeuron2, inputNeuron3, outputNeuron;
    private ConnectionGene connection;

    @BeforeEach
    public void setUp() {
        innovations = new HashSet<>();
        random = new Random(42);

        inputNeuron = new NeuronGene(1, ActivationFunction.NONE, NeuronType.INPUT);
        inputNeuron2 = new NeuronGene(2, ActivationFunction.NONE, NeuronType.INPUT);
        inputNeuron3 = new NeuronGene(3, ActivationFunction.NONE, NeuronType.INPUT);
        outputNeuron = new NeuronGene(4, ActivationFunction.TANH, NeuronType.OUTPUT);
        Map<Double, List<NeuronGene>> layers = new HashMap<>();
        List<NeuronGene> inputLayer = new ArrayList<>();
        inputLayer.add(inputNeuron);
        inputLayer.add(inputNeuron2);
        inputLayer.add(inputNeuron3);
        layers.put(0.0, inputLayer);
        layers.put(1.0, Collections.singletonList(outputNeuron));
        connection = new ConnectionGene(inputNeuron, outputNeuron, 1.0, true, 1);
        parent = new NetworkChromosome(layers, new ArrayList<>(Collections.singletonList(connection)));
    }

    @Test
    public void testAddNeuron() {
        NeatMutation mutation = new NeatMutation(innovations, random);
        NetworkChromosome offspring = mutation.addNeuron(parent);

        List<ConnectionGene> connections = offspring.getConnections();
        assertEquals(3, connections.size());
        ConnectionGene disabled = connections.stream()
                .filter(c -> !c.getEnabled())
                .findFirst()
                .orElseThrow();
        assertEquals(1, disabled.getInnovationNumber());

        ConnectionGene conn1 = connections.stream()
                .filter(c -> c.getTargetNeuron().getId() != outputNeuron.getId() && c.getEnabled())
                .findFirst()
                .orElseThrow();
        assertEquals(1.0, conn1.getWeight(), 0.001);
        assertEquals(inputNeuron, conn1.getSourceNeuron());

        ConnectionGene conn2 = connections.stream()
                .filter(c -> c.getTargetNeuron().getId() == outputNeuron.getId() && c.getEnabled())
                .findFirst()
                .orElseThrow();
        assertEquals(1.0, conn2.getWeight(), 0.001);
        assertEquals(outputNeuron, conn2.getTargetNeuron());

        Map<Double, List<NeuronGene>> layers = offspring.getLayers();
        assertEquals(3, layers.size());
        double hiddenLayer = layers.keySet().stream()
                .filter(k -> k != 0.0 && k != 1.0)
                .findFirst()
                .orElseThrow();
        NeuronGene newNeuron = layers.get(hiddenLayer).get(0);
        assertEquals(5, newNeuron.getId());
        assertEquals(NeuronType.HIDDEN, newNeuron.getNeuronType());

        assertEquals(2, innovations.size());
    }

    @Test
    public void testAddConnection() {
        Random mockRandom = mock(Random.class);
        NeatMutation mutation = new NeatMutation(innovations, mockRandom);
        when(mockRandom.nextInt(anyInt())).thenReturn(1, 0);
        NetworkChromosome offspring = mutation.addConnection(parent);

        List<ConnectionGene> connections = offspring.getConnections();
        assertEquals(2, connections.size());

        ConnectionGene newConn = connections.get(1);
        assertTrue(newConn.getEnabled());
        assertEquals(inputNeuron2, newConn.getSourceNeuron());
        assertEquals(outputNeuron, newConn.getTargetNeuron());
        assertTrue(newConn.getWeight() >= -1.0 && newConn.getWeight() <= 1.0);

        // Verify innovation
        assertEquals(1, innovations.size());
        assertEquals(1, newConn.getInnovationNumber());
    }

    @Test
    public void testMutateWeights() {
        NeatMutation mutation = new NeatMutation(innovations, random);
        NetworkChromosome offspring = mutation.mutateWeights(parent);

        List<ConnectionGene> connections = offspring.getConnections();
        assertEquals(1, connections.size());
        ConnectionGene mutated = connections.get(0);
        assertNotEquals(1.0, mutated.getWeight());
        assertTrue(mutated.getEnabled());
        assertEquals(1, mutated.getInnovationNumber());

        // Verify no structural change
        assertEquals(parent.getLayers(), offspring.getLayers());
        assertEquals(0, innovations.size());
    }

    @Test
    public void testToggleConnection() {
        NeatMutation mutation = new NeatMutation(innovations, random);
        NetworkChromosome offspring = mutation.toggleConnection(parent);

        List<ConnectionGene> connections = offspring.getConnections();
        assertEquals(1, connections.size());
        ConnectionGene toggled = connections.get(0);
        assertFalse(toggled.getEnabled());
        assertEquals(1.0, toggled.getWeight(), 0.001);
        assertEquals(1, toggled.getInnovationNumber());

        // Verify no structural change
        assertEquals(parent.getLayers(), offspring.getLayers());
        assertEquals(0, innovations.size());
    }

    @Test
    public void testApplyAddNeuron() {
        Random mockRandom = mock(Random.class);
        when(mockRandom.nextDouble()).thenReturn(0.02);
        NeatMutation mutation = new NeatMutation(innovations, mockRandom);
        NetworkChromosome offspring = mutation.apply(parent);

        assertEquals(3, offspring.getConnections().size());
        assertEquals(3, offspring.getLayers().size());
        assertEquals(2, innovations.size());
    }

    @Test
    public void testApplyAddConnection() {
        Random mockRandom = mock(Random.class);
        when(mockRandom.nextDouble()).thenReturn(0.04);
        when(mockRandom.nextInt(anyInt())).thenReturn(1, 0);
        NeatMutation mutation = new NeatMutation(innovations, mockRandom);
        NetworkChromosome offspring = mutation.apply(parent);

        assertEquals(2, offspring.getConnections().size());
        assertEquals(1, innovations.size());
    }
}

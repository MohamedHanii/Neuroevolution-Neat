package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class NetworkChromosomeTest {

    private Map<Double, List<NeuronGene>> layers;
    private List<ConnectionGene> connections;
    private NeuronGene input1, input2, bias, output;


    @BeforeEach
    public void setUp() {
        input1 = new NeuronGene(1,ActivationFunction.NONE,NeuronType.INPUT);
        input2 = new NeuronGene(2,ActivationFunction.NONE,NeuronType.INPUT);
        bias = new NeuronGene(3,ActivationFunction.NONE,NeuronType.BIAS);
        output = new NeuronGene(4,ActivationFunction.NONE,NeuronType.OUTPUT);

        layers = new HashMap<>();
        layers.put(NetworkChromosome.INPUT_LAYER, Arrays.asList(input1, input2, bias));
        layers.put(NetworkChromosome.OUTPUT_LAYER, Collections.singletonList(output));

        connections = new ArrayList<>();
        connections.add(new ConnectionGene(input1, output, 2.0, true, 1));
        connections.add(new ConnectionGene(input2, output, 1.5, true, 2));
        connections.add(new ConnectionGene(bias, output, 0.5, true, 3));
    }

    @Test
    public void testConstructorAndGetters() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        assertSame(layers, chromosome.getLayers());
        assertSame(connections, chromosome.getConnections());
    }

    @Test
    public void testConstructorNullLayersThrowsException() {
        assertThrows(NullPointerException.class, () -> new NetworkChromosome(null, connections));
    }

    @Test
    public void testConstructorNullConnectionsThrowsException() {
        assertThrows(NullPointerException.class, () -> new NetworkChromosome(layers, null));
    }

    @Test
    public void testGetOutputSimple() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        List<Double> state = Arrays.asList(0.5, 1.0);
        List<Double> outputList = chromosome.getOutput(state);
        assertEquals(1, outputList.size());
        assertEquals(3.0, outputList.get(0), 0.001);
    }

    @Test
    public void testGetOutputWithDisabledConnection() {
        List<ConnectionGene> modifiedConnections = new ArrayList<>();
        modifiedConnections.add(new ConnectionGene(input1, output, 2.0, true, 1));
        modifiedConnections.add(new ConnectionGene(input2, output, 1.5, false, 2)); // Disabled
        modifiedConnections.add(new ConnectionGene(bias, output, 0.5, true, 3));
        NetworkChromosome chromosome = new NetworkChromosome(layers, modifiedConnections);
        List<Double> state = Arrays.asList(0.5, 1.0);
        List<Double> outputList = chromosome.getOutput(state);
        assertEquals(1, outputList.size());

        assertEquals(1.5, outputList.get(0));
    }

    // **Fitness Tests**

    @Test
    public void testSetAndGetFitness() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        chromosome.setFitness(10.0);
        assertEquals(10.0, chromosome.getFitness(), 0.001);
    }

    // **Copy Test**

    @Test
    public void testCopy() {
        NetworkChromosome original = new NetworkChromosome(layers, connections);
        NetworkChromosome copy = original.copy();
        assertNotSame(original.getLayers(), copy.getLayers());
        assertNotSame(original.getConnections(), copy.getConnections());
        assertNotSame(original.getLayers().get(NetworkChromosome.INPUT_LAYER),
                copy.getLayers().get(NetworkChromosome.INPUT_LAYER));
        assertSame(original.getLayers().get(NetworkChromosome.INPUT_LAYER).get(0),
                copy.getLayers().get(NetworkChromosome.INPUT_LAYER).get(0));
        assertSame(original.getConnections().get(0), copy.getConnections().get(0));
        assertEquals(original.getLayers().size(), copy.getLayers().size());
        assertEquals(original.getConnections().size(), copy.getConnections().size());
    }

    // **Utility Method Tests**

    @Test
    public void testGetMaxNeuronId() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        assertEquals(4, chromosome.getMaxNeuronId());
    }

    @Test
    public void testGetMaxNeuronIdEmpty() {
        NetworkChromosome chromosome = new NetworkChromosome(new HashMap<>(), new ArrayList<>());
        assertEquals(0, chromosome.getMaxNeuronId());
    }

    @Test
    public void testGetLayerForNeuron() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        assertEquals(NetworkChromosome.INPUT_LAYER, chromosome.getLayerForNeuron(input1));
        assertEquals(NetworkChromosome.OUTPUT_LAYER, chromosome.getLayerForNeuron(output));
    }

    @Test
    public void testGetLayerForNeuronNotFound() {
        NeuronGene notPresent = new NeuronGene(5,ActivationFunction.NONE,NeuronType.INPUT);
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        assertThrows(IllegalArgumentException.class, () -> chromosome.getLayerForNeuron(notPresent));
    }

    @Test
    public void testGetAllNeurons() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        List<NeuronGene> allNeurons = chromosome.getAllNeurons();
        assertEquals(4, allNeurons.size());
        assertTrue(allNeurons.containsAll(Arrays.asList(input1, input2, bias, output)));
    }

    @Test
    public void testAddNeuronToLevel() {
        NetworkChromosome chromosome = new NetworkChromosome(new HashMap<>(), new ArrayList<>());
        NeuronGene newNeuron = new NeuronGene(1,ActivationFunction.NONE,NeuronType.INPUT);
        chromosome.addNeuronToLevel(newNeuron, 0.5);
        assertTrue(chromosome.getLayers().containsKey(0.5));
        assertEquals(1, chromosome.getLayers().get(0.5).size());
        assertSame(newNeuron, chromosome.getLayers().get(0.5).get(0));

        NeuronGene anotherNeuron = new NeuronGene(2,ActivationFunction.NONE,NeuronType.INPUT);
        chromosome.addNeuronToLevel(anotherNeuron, 0.5);
        assertEquals(2, chromosome.getLayers().get(0.5).size());
        assertTrue(chromosome.getLayers().get(0.5).contains(anotherNeuron));
    }

    @Test
    public void testGetConnectionMap() {
        NetworkChromosome chromosome = new NetworkChromosome(layers, connections);
        Map<Integer, ConnectionGene> connMap = chromosome.getConnectionMap();
        assertEquals(3, connMap.size());
        assertSame(connections.get(0), connMap.get(1));
        assertSame(connections.get(1), connMap.get(2));
        assertSame(connections.get(2), connMap.get(3));
    }
}

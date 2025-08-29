package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ConnectionGeneTest {
    @Test
    public void testConstructorAndGetters() {

        NeuronGene source = new NeuronGene(1,ActivationFunction.NONE,NeuronType.INPUT);
        NeuronGene target = new NeuronGene(2,ActivationFunction.TANH,NeuronType.OUTPUT);
        double weight = 0.5;
        boolean enabled = true;
        int innovationNumber = 1;


        ConnectionGene gene = new ConnectionGene(source, target, weight, enabled, innovationNumber);


        assertEquals(source, gene.getSourceNeuron());
        assertEquals(target, gene.getTargetNeuron());
        assertEquals(weight, gene.getWeight(), 0.001);
        assertTrue(gene.getEnabled());
        assertEquals(innovationNumber, gene.getInnovationNumber());
    }

    @Test
    public void testClone() {
        NeuronGene source = new NeuronGene(1,ActivationFunction.NONE,NeuronType.INPUT);
        NeuronGene target = new NeuronGene(2,ActivationFunction.TANH,NeuronType.OUTPUT);
        double weight = 0.5;
        boolean enabled = true;
        int innovationNumber = 1;
        ConnectionGene gene = new ConnectionGene(source, target, weight, enabled, innovationNumber);

        ConnectionGene clone = gene.clone();

        assertNotSame(gene, clone);
        assertSame(source, clone.getSourceNeuron());
        assertSame(target, clone.getTargetNeuron());
        assertEquals(weight, clone.getWeight(), 0.001);
        assertEquals(enabled, clone.getEnabled());
        assertEquals(innovationNumber, clone.getInnovationNumber());
    }

    @Test
    public void testCloneWithDisabled() {
        NeuronGene source = new NeuronGene(3,ActivationFunction.NONE,NeuronType.INPUT);
        NeuronGene target = new NeuronGene(4,ActivationFunction.TANH,NeuronType.OUTPUT);

        double weight = 0.3;
        boolean enabled = false;
        int innovationNumber = 2;
        ConnectionGene gene = new ConnectionGene(source, target, weight, enabled, innovationNumber);

        ConnectionGene clone = gene.clone();

        assertNotSame(gene, clone);
        assertSame(source, clone.getSourceNeuron());
        assertSame(target, clone.getTargetNeuron());
        assertEquals(weight, clone.getWeight(), 0.001);
        assertFalse(clone.getEnabled());
        assertEquals(innovationNumber, clone.getInnovationNumber());
    }
}

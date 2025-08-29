package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.ConnectionInnovation;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.Innovation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

    public class NetworkGeneratorTest {

        private Random random;
        private Set<Innovation> innovations;

        @BeforeEach
        public void setUp() {
            random = mock(Random.class);
            innovations = new HashSet<>();
        }


        @Test
        public void testConstructorNullInnovations() {
            assertThrows(NullPointerException.class, () -> new NetworkGenerator(null, 2, 1, random),
                    "Should throw NullPointerException for null innovations");
        }

        @Test
        public void testConstructorNullRandom() {
            assertThrows(NullPointerException.class, () -> new NetworkGenerator(innovations, 2, 1, null),
                    "Should throw NullPointerException for null random");
        }

        @Test
        public void testGenerateStructure() {
            when(random.nextDouble()).thenReturn(0.5); // Weight = 0.5 * 2 - 1 = 0.0
            NetworkGenerator generator = new NetworkGenerator(innovations, 2, 1, random);
            NetworkChromosome chromosome = generator.generate();

            // Verify layers
            Map<Double, List<NeuronGene>> layers = chromosome.getLayers();
            assertEquals(2, layers.size());
            assertTrue(layers.containsKey(0.0));
            assertTrue(layers.containsKey(1.0));

            // Input layer: 2 inputs + 1 bias
            List<NeuronGene> inputLayer = layers.get(0.0);
            assertEquals(3, inputLayer.size());
            assertEquals(NeuronType.INPUT, inputLayer.get(0).getNeuronType());
            assertEquals(NeuronType.INPUT, inputLayer.get(1).getNeuronType());
            assertEquals(NeuronType.BIAS, inputLayer.get(2).getNeuronType());
            assertEquals(ActivationFunction.NONE, inputLayer.get(0).getActivationFunction());

            // Output layer: 1 output
            List<NeuronGene> outputLayer = layers.get(1.0);
            assertEquals(1, outputLayer.size());
            assertEquals(NeuronType.OUTPUT, outputLayer.get(0).getNeuronType());
            assertEquals(ActivationFunction.TANH, outputLayer.get(0).getActivationFunction());

            // Verify IDs
            assertEquals(1, inputLayer.get(0).getId());
            assertEquals(2, inputLayer.get(1).getId());
            assertEquals(3, inputLayer.get(2).getId());
            assertEquals(4, outputLayer.get(0).getId());
        }

        @Test
        public void testGenerateConnections() {
            when(random.nextDouble()).thenReturn(0.25, 0.5, 0.75); // Weights: -0.5, 0.0, 0.5
            NetworkGenerator generator = new NetworkGenerator(innovations, 2, 1, random);
            NetworkChromosome chromosome = generator.generate();


            List<ConnectionGene> connections = chromosome.getConnections();
            assertEquals(3, connections.size());

            // Check connection details
            assertConnection(connections.get(0), 1, 4, -0.5, 1);
            assertConnection(connections.get(1), 2, 4, 0.0, 2);
            assertConnection(connections.get(2), 3, 4, 0.5, 3);

            // Verify innovations
            assertEquals(3, innovations.size());
        }

        @Test
        public void testGenerateReusesInnovations() {
            innovations.add(new ConnectionInnovation(1, 4, 1));
            innovations.add(new ConnectionInnovation(2, 4, 2));
            when(random.nextDouble()).thenReturn(0.5);
            NetworkGenerator generator = new NetworkGenerator(innovations, 2, 1, random);
            NetworkChromosome chromosome = generator.generate();

            List<ConnectionGene> connections = chromosome.getConnections();
            assertEquals(3, connections.size());
            assertConnection(connections.get(0), 1, 4, 0.0, 1);
            assertConnection(connections.get(1), 2, 4, 0.0, 2);
            assertConnection(connections.get(2), 3, 4, 0.0, 3);

            assertEquals(3, innovations.size());
        }

        @Test
        public void testFindInnovation() {
            NetworkGenerator generator = new NetworkGenerator(innovations, 1, 1, random);
            innovations.add(new ConnectionInnovation(1, 2, 1));
            innovations.add(new ConnectionInnovation(2, 3, 2));

            ConnectionInnovation found = generator.findInnovation(1, 2);
            assertNotNull(found);
            assertEquals(1, found.getSource());
            assertEquals(2, found.getTarget());
            assertEquals(1, found.getInnovationNumber());

            ConnectionInnovation notFound = generator.findInnovation(3, 4);
            assertNull(notFound);
        }


        private void assertConnection(ConnectionGene conn, int sourceId, int targetId, double weight, int innov) {
            assertEquals(sourceId, conn.getSourceNeuron().getId());
            assertEquals(targetId, conn.getTargetNeuron().getId());
            assertEquals(weight, conn.getWeight(), 0.001);
            assertTrue(conn.getEnabled());
            assertEquals(innov, conn.getInnovationNumber());
        }
}

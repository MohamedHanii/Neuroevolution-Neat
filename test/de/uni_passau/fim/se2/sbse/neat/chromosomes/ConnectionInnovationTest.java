package de.uni_passau.fim.se2.sbse.neat.chromosomes;
import de.uni_passau.fim.se2.sbse.neat.algorithms.innovations.ConnectionInnovation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ConnectionInnovationTest {

    @Test
    public void testConstructorAndGetters() {
        ConnectionInnovation innovation = new ConnectionInnovation(1, 2, 3);
        assertEquals(1, innovation.getSource());
        assertEquals(2, innovation.getTarget());
        assertEquals(3, innovation.getInnovationNumber());
    }

    @Test
    public void testEqualsSameValues() {
        ConnectionInnovation innov1 = new ConnectionInnovation(1, 2, 3);
        ConnectionInnovation innov2 = new ConnectionInnovation(1, 2, 4);
        assertEquals(innov1, innov2);
        assertEquals(innov1.hashCode(), innov2.hashCode());
    }

    @Test
    public void testEqualsDifferentSource() {
        ConnectionInnovation innov1 = new ConnectionInnovation(1, 2, 3);
        ConnectionInnovation innov2 = new ConnectionInnovation(3, 2, 3);
        assertNotEquals(innov1, innov2);
    }

    @Test
    public void testEqualsDifferentTarget() {
        ConnectionInnovation innov1 = new ConnectionInnovation(1, 2, 3);
        ConnectionInnovation innov2 = new ConnectionInnovation(1, 3, 3);
        assertNotEquals(innov1, innov2);
    }

    @Test
    public void testEqualsSameObject() {
        ConnectionInnovation innov = new ConnectionInnovation(1, 2, 3);
        assertEquals(innov, innov);
    }

    @Test
    public void testEqualsNull() {
        ConnectionInnovation innov = new ConnectionInnovation(1, 2, 3);
        assertNotEquals(innov, null);
    }

    @Test
    public void testEqualsDifferentClass() {
        ConnectionInnovation innov = new ConnectionInnovation(1, 2, 3);
        Object other = new Object();
        assertNotEquals(innov, other);
    }

    @Test
    public void testHashCodeConsistency() {
        ConnectionInnovation innov1 = new ConnectionInnovation(1, 2, 3);
        ConnectionInnovation innov2 = new ConnectionInnovation(1, 2, 4);
        assertEquals(innov1.hashCode(), innov2.hashCode());
    }

    @Test
    public void testHashCodeDifferent() {
        ConnectionInnovation innov1 = new ConnectionInnovation(1, 2, 3);
        ConnectionInnovation innov2 = new ConnectionInnovation(2, 3, 3);
        assertNotEquals(innov1.hashCode(), innov2.hashCode());
    }
}
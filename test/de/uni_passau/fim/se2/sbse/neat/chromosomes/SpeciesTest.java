package de.uni_passau.fim.se2.sbse.neat.chromosomes;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

public class SpeciesTest {
    private Random random;
    private NetworkChromosome chrom1, chrom2, chrom3;

    @BeforeEach
    public void setUp() {
        random = mock(Random.class);
        chrom1 = mock(NetworkChromosome.class);
        chrom2 = mock(NetworkChromosome.class);
        chrom3 = mock(NetworkChromosome.class);
        when(chrom1.getFitness()).thenReturn(10.0);
        when(chrom2.getFitness()).thenReturn(20.0);
        when(chrom3.getFitness()).thenReturn(30.0);
    }

    @Test
    public void testConstructor() {
        Species species = new Species(random);
        assertNotNull(species.getMembers());
        assertTrue(species.getMembers().isEmpty());
        assertEquals(0, species.getOffspringCount());
    }

    @Test
    public void testAddMember() {
        Species species = new Species(random);
        species.addMember(chrom1);
        species.addMember(chrom2);

        List<NetworkChromosome> members = species.getMembers();
        assertEquals(2, members.size());
        assertTrue(members.contains(chrom1));
        assertTrue(members.contains(chrom2));
    }

    @Test
    public void testGetRepresentative() {
        Species species = new Species(random);
        species.addMember(chrom1);
        species.addMember(chrom2);
        species.addMember(chrom3);

        when(random.nextInt(3)).thenReturn(1);
        NetworkChromosome rep = species.getRepresentative();
        assertEquals(chrom2, rep);

        when(random.nextInt(3)).thenReturn(0);
        rep = species.getRepresentative();
        assertEquals(chrom1, rep);
    }

    @Test
    public void testGetAndSetOffspringCount() {
        Species species = new Species(random);
        species.setOffspringCount(5);
        assertEquals(5, species.getOffspringCount());

        species.setOffspringCount(0);
        assertEquals(0, species.getOffspringCount());
    }

    @Test
    public void testSelectParentSingleMember() {
        Species species = new Species(random);
        species.addMember(chrom1);

        Random selectionRandom = mock(Random.class);
        when(selectionRandom.nextInt(1)).thenReturn(0);
        NetworkChromosome parent = species.selectParent(selectionRandom);
        assertEquals(chrom1, parent);
    }

    @Test
    public void testSelectParentTournamentSizeTwo() {
        Species species = new Species(random);
        species.addMember(chrom1);
        species.addMember(chrom2);

        Random selectionRandom = mock(Random.class);
        when(selectionRandom.nextInt(2)).thenReturn(0, 1);
        NetworkChromosome parent = species.selectParent(selectionRandom);
        assertEquals(chrom2, parent);
    }

    @Test
    public void testSelectParentTournamentSizeThree() {
        Species species = new Species(random);
        species.addMember(chrom1);
        species.addMember(chrom2);
        species.addMember(chrom3);

        Random selectionRandom = mock(Random.class);
        when(selectionRandom.nextInt(3)).thenReturn(2, 0, 1);
        NetworkChromosome parent = species.selectParent(selectionRandom);
        assertEquals(chrom3, parent);
    }

    @Test
    public void testGetSharedFitness() {
        Species species = new Species(random);
        species.addMember(chrom1);
        species.addMember(chrom2);
        species.addMember(chrom3);

        double sharedFitness = species.getSharedFitness();

        assertEquals(20.0, sharedFitness, 0.001);
    }

    @Test
    public void testGetSharedFitnessEmpty() {
        Species species = new Species(random);
        double sharedFitness = species.getSharedFitness();
        assertEquals(0.0, sharedFitness, 0.001);
    }
}

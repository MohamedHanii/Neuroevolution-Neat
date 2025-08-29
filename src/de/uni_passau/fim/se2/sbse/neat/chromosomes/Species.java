package de.uni_passau.fim.se2.sbse.neat.chromosomes;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Species {

    private final List<NetworkChromosome> members;
    private Random random;
    private int offspringCount;

    public Species(Random random) {
        this.members = new ArrayList<>();
        this.random = random;
    }

    public void addMember(NetworkChromosome agent) {
        members.add(agent);
    }

    public NetworkChromosome getRepresentative() {
        return members.get(random.nextInt(members.size()));
    }

    public List<NetworkChromosome> getMembers() {
        return members;
    }


    public int getOffspringCount() {
        return offspringCount;
    }

    public void setOffspringCount(int offspringCount) {
        this.offspringCount = offspringCount;
    }

    public NetworkChromosome selectParent(Random random) {
        int tournamentSize = Math.min(3, members.size());
        NetworkChromosome best = null;
        for (int i = 0; i < tournamentSize; i++) {
            int index = random.nextInt(members.size());
            NetworkChromosome candidate = members.get(index);
            if (best == null || candidate.getFitness() > best.getFitness()) {
                best = candidate;
            }
        }
        return best;
    }

    public double getSharedFitness(){
        double fitness = 0;

        for (NetworkChromosome member : members) {
            fitness+= member.getFitness() / members.size();
        }

        return fitness;
    }
}

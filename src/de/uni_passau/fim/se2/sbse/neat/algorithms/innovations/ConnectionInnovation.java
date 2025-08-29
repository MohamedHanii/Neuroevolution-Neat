package de.uni_passau.fim.se2.sbse.neat.algorithms.innovations;

import java.util.Objects;

public class ConnectionInnovation implements Innovation {
    private final int source;
    private final int target;
    private final int innovationNumber;

    public ConnectionInnovation(int source, int target, int innovationNumber) {
        this.source = source;
        this.target = target;
        this.innovationNumber = innovationNumber;
    }

    public int getSource() {
        return source;
    }

    public int getTarget() {
        return target;
    }

    public int getInnovationNumber() {
        return innovationNumber;
    }

    @Override
    public int hashCode() {
        return Objects.hash(source, target);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof ConnectionInnovation)) return false;
        ConnectionInnovation other = (ConnectionInnovation) obj;
        return this.source == other.source && this.target == other.target;
    }
}
# NEAT (NeuroEvolution of Augmenting Topologies)

A Java implementation of the NEAT algorithm for neuroevolution, designed to solve reinforcement learning tasks through evolutionary computation of neural network architectures.

## 🚀 Overview

This project implements the NEAT algorithm, which is a genetic algorithm that evolves both the topology and weights of neural networks. NEAT is particularly effective for solving complex problems where the optimal network structure is not known in advance.

## ✨ Features

- **NEAT Algorithm Implementation**: Complete implementation of the NeuroEvolution of Augmenting Topologies algorithm
- **Multiple Task Environments**: Support for XOR, CartPole, and CartPole Random tasks
- **Innovation Tracking**: Proper handling of structural mutations and innovation numbers
- **Species Management**: Automatic speciation to protect topological innovations
- **Fitness Sharing**: Balanced selection pressure across species
- **Comprehensive Testing**: Extensive test coverage with multiple testing frameworks
- **Visualization**: Built-in visualization for trained agents
- **Command Line Interface**: Easy-to-use CLI with configurable parameters

## 🏗️ Architecture

The project follows a modular architecture with clear separation of concerns:

```
src/de/uni_passau/fim/se2/sbse/neat/
├── algorithms/          # Core NEAT algorithm implementation
├── chromosomes/         # Neural network representation
├── crossover/          # Genetic crossover operations
├── environments/       # Task environments and fitness functions
├── mutation/           # Genetic mutation operations
└── utils/              # Utility classes and helpers
```

### Key Components

- **NeatAlgorithm**: Main algorithm orchestrator implementing the Neuroevolution interface
- **NetworkChromosome**: Represents neural networks as chromosomes
- **Species**: Manages population diversity through speciation
- **Innovation**: Tracks structural changes for proper crossover
- **Environment**: Abstract interface for different task domains

## 🛠️ Prerequisites

- **Java 23** or higher
- **Maven 3.6+** for build management
- **Git** for version control

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Neuroevolution-Neat
   ```

2. **Build the project**:
   ```bash
   mvn clean compile
   ```

3. **Run tests** (optional but recommended):
   ```bash
   mvn test
   ```

## 🚀 Usage

### Command Line Interface

The project provides a comprehensive CLI with the following options:

```bash
java -jar target/Neuroevolution-Neat.jar [OPTIONS]
```

#### Available Options

| Option | Long Form | Description | Default |
|--------|-----------|-------------|---------|
| `-t` | `--task` | Task to solve: XOR, CART, CART_RANDOM | Required |
| `-p` | `--population-size` | Population size | 50 |
| `-g` | `--max-generations` | Maximum generations | 50 |
| `-r` | `--repetitions` | Number of task repetitions | 30 |
| `-v` | `--visualise` | Enable visualization | false |
| `-s` | `--seed` | Random seed for reproducibility | Random |

### Example Usage

1. **Solve XOR problem**:
   ```bash
   java -jar target/Neuroevolution-Neat.jar -t XOR -p 100 -g 100
   ```

2. **Solve CartPole with visualization**:
   ```bash
   java -jar target/Neuroevolution-Neat.jar -t CART -p 75 -g 75 -v
   ```

3. **Reproducible results with fixed seed**:
   ```bash
   java -jar target/Neuroevolution-Neat.jar -t XOR -s 42 -p 50 -g 50
   ```

## 🎯 Supported Tasks

### 1. XOR Problem
- **Description**: Learn the XOR (exclusive or) logical function
- **Input**: 2 binary inputs
- **Output**: 1 binary output
- **Difficulty**: Classic benchmark for neural network learning

### 2. CartPole Balancing
- **Description**: Balance a pole on a moving cart
- **Input**: Cart position, velocity, pole angle, pole angular velocity
- **Output**: Left/right force to apply to cart
- **Difficulty**: Continuous control problem

### 3. CartPole Random
- **Description**: CartPole with randomized initial conditions
- **Input**: Same as CartPole
- **Output**: Same as CartPole
- **Difficulty**: More challenging due to varied starting states

## 🔬 Algorithm Details

### NEAT Core Principles

1. **Topological Innovation**: Networks can add/remove nodes and connections
2. **Innovation Tracking**: Each structural change gets a unique innovation number
3. **Speciation**: Networks are grouped by topological similarity
4. **Fitness Sharing**: Selection pressure is balanced across species
5. **Elitism**: Best members of each species are preserved

### Evolution Process

1. **Initialization**: Create minimal networks for each input/output
2. **Evaluation**: Test networks on the target task
3. **Speciation**: Group similar networks into species
4. **Selection**: Select parents based on shared fitness
5. **Reproduction**: Apply crossover and mutation
6. **Replacement**: Create new generation while preserving elites

## 🧪 Testing

The project includes comprehensive testing with multiple frameworks:

- **JUnit 5**: Core testing framework
- **AssertJ**: Fluent assertions
- **Mockito**: Mocking framework
- **JQwik**: Property-based testing
- **JaCoCo**: Code coverage analysis
- **PIT**: Mutation testing

### Running Tests

```bash
# Run all tests
mvn test

# Run with coverage report
mvn clean test jacoco:report

# Run mutation testing
mvn pitest:mutationCoverage
```

## 📊 Expected Results

The `expectedResults/` directory contains sample outputs for different tasks:

- `XOR.txt`: Expected results for XOR problem
- `CART.txt`: Expected results for CartPole
- `CART_RANDOM.txt`: Expected results for CartPole Random

## 🔧 Development

### Project Structure

```
├── src/                    # Source code
│   └── de/uni_passau/fim/se2/sbse/neat/
│       ├── algorithms/     # Algorithm implementations
│       ├── chromosomes/    # Neural network representation
│       ├── crossover/      # Genetic operations
│       ├── environments/   # Task definitions
│       ├── mutation/       # Mutation operations
│       └── utils/          # Utility classes
├── test/                   # Test code
├── target/                 # Build output
├── pom.xml                 # Maven configuration
└── README.md              # This file
```

### Building from Source

```bash
# Clean and compile
mvn clean compile

# Package into JAR
mvn package

# Install to local repository
mvn install
```

### Code Quality

The project uses several tools to maintain code quality:

- **Error Prone**: Static analysis during compilation
- **JaCoCo**: Code coverage reporting
- **PIT**: Mutation testing for test quality
- **Maven**: Automated build and dependency management

## 📚 Dependencies

### Core Dependencies
- **Picocli**: Command-line interface framework
- **Java 23**: Modern Java features and performance

### Test Dependencies
- **JUnit 5**: Testing framework
- **AssertJ**: Assertion library
- **Mockito**: Mocking framework
- **JQwik**: Property-based testing
- **Hamcrest**: Matcher library
- **Truth**: Google's assertion library


### Development Guidelines

- Follow Java coding conventions
- Write comprehensive tests for new features
- Ensure all tests pass before submitting
- Update documentation as needed
- Use meaningful commit messages

## 📄 License

This project is part of the Search-Based Software Engineering course at the University of Passau.

## 🙏 Acknowledgments

- **University of Passau**: Academic institution supporting this research
- **NEAT Algorithm**: Original algorithm by Stanley and Miikkulainen
- **Open Source Community**: Various testing and utility libraries


**Note**: This is an academic implementation of the NEAT algorithm designed for educational and research purposes. For production use, consider additional optimizations and testing. 

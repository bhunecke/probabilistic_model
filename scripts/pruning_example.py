import numpy as np
import matplotlib.pyplot as plt
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
from probabilistic_model.distributions import *
from random_events.product_algebra import Continuous

def create_simple_circuit():
    # Create variables
    x1 = Continuous("x1")
    x2 = Continuous("x2")
    
    # Create circuit
    circuit = ProbabilisticCircuit()
    
    # Create root sum unit
    root = SumUnit(probabilistic_circuit=circuit)
    circuit.add_node(root)

    sum1 = SumUnit(probabilistic_circuit=circuit)
    sum2 = SumUnit(probabilistic_circuit=circuit)
    sum3 = SumUnit(probabilistic_circuit=circuit)
    circuit.add_nodes_from([sum1, sum2, sum3])
    
    # Create product units
    prod1 = ProductUnit(probabilistic_circuit=circuit)
    prod2 = ProductUnit(probabilistic_circuit=circuit)
    prod3 = ProductUnit(probabilistic_circuit=circuit)
    circuit.add_nodes_from([prod1, prod2, prod3])
    
    # Create leaf distributions
    leaf1 = leaf(GaussianDistribution(x1, 0.0, 1.0), circuit)
    leaf2 = leaf(GaussianDistribution(x2, 0.0, 1.0), circuit)
    leaf3 = leaf(GaussianDistribution(x1, 2.0, 1.0), circuit)
    leaf4 = leaf(GaussianDistribution(x2, 2.0, 1.0), circuit)
    leaf5 = leaf(GaussianDistribution(x1, 4.0, 1.0), circuit)
    leaf6 = leaf(GaussianDistribution(x2, 4.0, 1.0), circuit)

    # Connect the circuit
    root.add_subcircuit(sum1, np.log(0.5))  # Weight for first sum unit
    root.add_subcircuit(sum2, np.log(0.5))  # Weight for second sum unit
    root.add_subcircuit(sum3, np.log(0.5))  # Weight for third sum unit
    sum1.add_subcircuit(prod1, np.log(0.7))  # High weight
    sum1.add_subcircuit(prod2, np.log(0.3))  # Low weight
    sum1.add_subcircuit(prod3, np.log(0.1))  # Very low weight
    sum2.add_subcircuit(prod1, np.log(0.4))  # Medium weight
    sum2.add_subcircuit(prod2, np.log(0.4))  # Medium weight
    sum2.add_subcircuit(prod3, np.log(0.4))  # Medium weight
    sum3.add_subcircuit(prod1, np.log(0.4))  # Medium weight
    sum3.add_subcircuit(prod2, np.log(0.4))  # Medium weight
    sum3.add_subcircuit(prod3, np.log(0.4))  # Medium weight

    prod1.add_subcircuit(leaf1)
    prod1.add_subcircuit(leaf2)
    prod2.add_subcircuit(leaf3)
    prod2.add_subcircuit(leaf4)
    prod3.add_subcircuit(leaf5)
    prod3.add_subcircuit(leaf6)
    
    return circuit

def main():
    print("\nCreating a simple probabilistic circuit...")

    circuit = create_simple_circuit()
    print(f" - {circuit}")

    circuit.plot_structure()
    plt_name = "circuit_structure.png"
    plt.savefig(plt_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Circuit structure saved as '{plt_name}'")

    pruning_percentage = 0.4
    print(f"\nPruning {pruning_percentage}% of the edges...")
    np.random.seed(42)
    dataset = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)

    pruned_circuit = circuit.prune(dataset=dataset, pruning_percentage=pruning_percentage)
    print(f" - {pruned_circuit}")

    pruned_circuit.plot_structure()
    plt_pruned_name = "circuit_structure_pruned.png"
    plt.savefig(plt_pruned_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Pruned circuit structure saved as '{plt_pruned_name}'")

if __name__ == "__main__":
    main()

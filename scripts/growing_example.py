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
    circuit.add_nodes_from([prod1, prod2])
    
    # Create leaf distributions
    leaf1 = leaf(GaussianDistribution(x1, 0.0, 1.0), circuit)
    leaf2 = leaf(GaussianDistribution(x2, 0.0, 1.0), circuit)
    leaf3 = leaf(GaussianDistribution(x1, 2.0, 1.0), circuit)
    leaf4 = leaf(GaussianDistribution(x2, 2.0, 1.0), circuit)

    # Connect the circuit
    root.add_subcircuit(sum1, np.log(0.5))  # Weight for first sum unit
    root.add_subcircuit(sum2, np.log(0.5))  # Weight for second sum unit
    root.add_subcircuit(sum3, np.log(0.5))  # Weight for third sum unit
    sum1.add_subcircuit(prod1, np.log(0.7))  # High weight
    sum1.add_subcircuit(prod2, np.log(0.3))  # Low weight
    sum2.add_subcircuit(prod2, np.log(0.4))  # Medium weight
    sum3.add_subcircuit(prod1, np.log(0.4))  # Medium weight

    prod1.add_subcircuit(leaf1)
    prod1.add_subcircuit(leaf2)
    prod2.add_subcircuit(leaf3)
    prod2.add_subcircuit(leaf4)
    
    return circuit

def create_tiny_circuit():
    # Create variables
    x1 = Continuous("x1")
    x2 = Continuous("x2")

    # Create circuit
    circuit = ProbabilisticCircuit()

    # Create root sum unit
    root = SumUnit(probabilistic_circuit=circuit)
    circuit.add_node(root)

    # Create leaf distributions
    leaf1 = leaf(GaussianDistribution(x1, 0.0, 1.0), circuit)
    leaf2 = leaf(GaussianDistribution(x2, 0.0, 1.0), circuit)

    # Connect the circuit
    root.add_subcircuit(leaf1, np.log(0.5))  # Weight for first leaf
    root.add_subcircuit(leaf2, np.log(0.5))  # Weight for second leaf

    return circuit

def main():
    circuit = create_simple_circuit()
    circuit.plot_structure()
    plt.title("Original Circuit Structure")
    plt.savefig("original_circuit_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Original circuit structure saved as 'original_circuit_structure.png'")
    grown_circuit = circuit.grow(0.4)
    grown_circuit.plot_structure()
    plt.title("Grown Circuit Structure")
    plt.savefig("grown_circuit_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Grown circuit structure saved as 'grown_circuit_structure.png'")

if __name__ == "__main__":
    main()

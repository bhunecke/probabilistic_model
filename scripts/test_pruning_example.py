import numpy as np
import matplotlib.pyplot as plt
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, leaf
from probabilistic_model.distributions.gaussian import GaussianDistribution
from random_events.variable import Continuous

def create_simple_circuit():
    # Create variables
    x1 = Continuous("x1")
    x2 = Continuous("x2")
    
    # Create circuit
    circuit = ProbabilisticCircuit()
    
    # Create root sum unit
    root = SumUnit(probabilistic_circuit=circuit)
    circuit.add_node(root)
    
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
    root.add_subcircuit(prod1, np.log(0.7))  # High weight
    root.add_subcircuit(prod2, np.log(0.3))  # Low weight
    
    prod1.add_subcircuit(leaf1)
    prod1.add_subcircuit(leaf2)
    prod2.add_subcircuit(leaf3)
    prod2.add_subcircuit(leaf4)
    
    return circuit

def main():
    print("Creating a simple probabilistic circuit...")
    circuit = create_simple_circuit()

    # Plot and save the circuit structure
    circuit.plot_structure()
    plt.savefig("circuit_structure.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Circuit structure saved as 'circuit_structure.png'")

    print(f"Circuit has {len(circuit)} nodes")
    print(f"Circuit has {len(circuit.edges())} edges")
    
    # Generate sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    dataset = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
    
    print(f"Dataset shape: {dataset.shape}")
    
    # Test edge flow computation on a single sample
    print("\nTesting edge flow computation...")
    try:
        sample = dataset[0]
        edge_flows = circuit.compute_edge_flows(sample)
        print(f"Computed edge flows for {len(edge_flows)} edges")
        
        # Print edge flows for sum unit edges
        sum_edges = [(parent, child) for (parent, child) in edge_flows.keys() 
                     if isinstance(parent, SumUnit)]
        print(f"Sum unit edges: {len(sum_edges)}")
        for edge in sum_edges:
            flow = edge_flows[edge]
            print(f"  Edge flow: {flow:.4f}")
            
    except Exception as e:
        print(f"Error computing edge flows: {e}")
        return
    
    # Test pruning
    print("\nTesting pruning functionality...")
    try:
        original_edges = len(circuit.edges())
        circuit.prune(dataset, pruning_percentage=0.5)
        new_edges = len(circuit.edges())
        
        print(f"Edges before pruning: {original_edges}")
        print(f"Edges after pruning: {new_edges}")
        print(f"Edges removed: {original_edges - new_edges}")
        
        # Plot and save the pruned circuit structure
        circuit.plot_structure()
        plt.savefig("circuit_structure_pruned.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Pruned circuit structure saved as 'circuit_structure_pruned.png'")
        
    except Exception as e:
        print(f"Error during pruning: {e}")
        return
    
    print("\nPruning test completed successfully!")

if __name__ == "__main__":
    main()

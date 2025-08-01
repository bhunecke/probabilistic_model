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
    root.add_subcircuit(prod1, np.log(0.6))  # Weight for first component
    root.add_subcircuit(prod2, np.log(0.4))  # Weight for second component
    
    prod1.add_subcircuit(leaf1)
    prod1.add_subcircuit(leaf2)
    prod2.add_subcircuit(leaf3)
    prod2.add_subcircuit(leaf4)
    
    return circuit

def print_circuit_stats(circuit, name="Circuit"):
    print(f"\n{name} Statistics:")
    print(f"  Nodes: {len(circuit)}")
    print(f"  Edges: {len(circuit.edges())}")
    print(f"  Layers: {len(circuit.layers)}")
    
    sum_units = sum(1 for node in circuit.nodes() if isinstance(node, SumUnit))
    product_units = sum(1 for node in circuit.nodes() if isinstance(node, ProductUnit))
    leaf_units = sum(1 for node in circuit.nodes() if node.is_leaf)
    
    print(f"  Sum units: {sum_units}")
    print(f"  Product units: {product_units}")
    print(f"  Leaf units: {leaf_units}")

def test_circuit_functionality(circuit, name="Circuit"):
    print(f"\nTesting {name} functionality...")
    
    try:
        # Test likelihood computation
        sample = np.array([[0.0, 0.0]])
        log_likelihood = circuit.log_likelihood(sample)
        print(f"  Log-likelihood at [0,0]: {log_likelihood[0]:.4f}")
        
        # Test sampling
        samples = circuit.sample(5)
        print(f"  Generated 5 samples: shape {samples.shape}")
        print(f"  Sample means: {np.mean(samples, axis=0)}")
        
        # Test support
        support = circuit.support
        print(f"  Support computed successfully")
        
        return True
    except Exception as e:
        print(f"  Error during functionality test: {e}")
        return False

def main():
    print("Creating a simple probabilistic circuit for growing test...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create original circuit
    original_circuit = create_simple_circuit()
    print_circuit_stats(original_circuit, "Original Circuit")
    
    # Test original circuit functionality
    if not test_circuit_functionality(original_circuit, "Original Circuit"):
        print("Original circuit failed functionality test. Aborting.")
        return
    
    # Plot and save the original circuit structure
    try:
        original_circuit.plot_structure()
        plt.title("Original Circuit Structure")
        plt.savefig("original_circuit_structure.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Original circuit structure saved as 'original_circuit_structure.png'")
    except Exception as e:
        print(f"Could not save original circuit plot: {e}")
    
    # Test growing
    print("\n\nTESTING GROWING FUNCTIONALITY")

    try:
        # Test with default noise variance
        print("\nTesting growth with default noise variance (0.1)...")
        grown_circuit = original_circuit.__deepcopy__()
        grown_circuit.grow(noise_variance=0.1)
        
        print_circuit_stats(grown_circuit, "Grown Circuit (noise=0.1)")
        
        # Test grown circuit functionality
        if test_circuit_functionality(grown_circuit, "Grown Circuit"):
            print("Grown circuit passed functionality test!")
        else:
            print("Grown circuit failed functionality test!")
            
    except Exception as e:
        print(f"Error during growing with default parameters: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Also test with different noise variance
    try:
        print("\nTesting growth with higher noise variance (0.5)...")
        grown_circuit_high_noise = original_circuit.__deepcopy__()
        grown_circuit_high_noise.grow(noise_variance=0.5)
        
        print_circuit_stats(grown_circuit_high_noise, "Grown Circuit (noise=0.5)")
        
        # Test functionality
        if test_circuit_functionality(grown_circuit_high_noise, "Grown Circuit (high noise)"):
            print("High noise grown circuit passed functionality test!")
        else:
            print("High noise grown circuit failed functionality test!")
            
    except Exception as e:
        print(f"Error during growing with high noise: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Also test with very low noise variance
    try:
        print("\nTesting growth with low noise variance (0.01)...")
        grown_circuit_low_noise = original_circuit.__deepcopy__()
        grown_circuit_low_noise.grow(noise_variance=0.01)
        
        print_circuit_stats(grown_circuit_low_noise, "Grown Circuit (noise=0.01)")
        
        # Test functionality
        if test_circuit_functionality(grown_circuit_low_noise, "Grown Circuit (low noise)"):
            print("Low noise grown circuit passed functionality test!")
        else:
            print("Low noise grown circuit failed functionality test!")
            
    except Exception as e:
        print(f"Error during growing with low noise: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Plot and save the grown circuit structure
    try:
        grown_circuit.plot_structure()
        plt.title("Grown Circuit Structure")
        plt.savefig("grown_circuit_structure.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("\nGrown circuit structure saved as 'grown_circuit_structure.png'")
    except Exception as e:
        print(f"Could not save grown circuit plot: {e}")
    
    # Compare performance
    print("\n\nCOMPARING ORIGINAL vs GROWN CIRCUIT PERFORMANCE")

    try:
        # Generate test data
        test_data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)
        
        # Compute likelihoods
        original_ll = original_circuit.log_likelihood(test_data)
        grown_ll = grown_circuit.log_likelihood(test_data)
        
        print(f"\nOriginal circuit mean log-likelihood: {np.mean(original_ll):.4f}")
        print(f"Grown circuit mean log-likelihood: {np.mean(grown_ll):.4f}")
        print(f"Difference: {np.mean(grown_ll - original_ll):.4f}")
        
        # Compute correlation
        correlation = np.corrcoef(original_ll, grown_ll)[0, 1]
        print(f"Correlation between likelihoods: {correlation:.4f}")
        
    except Exception as e:
        print(f"Error during performance comparison: {e}")
    
    # Test multiple grow operations
    print("\n\nTESTING MULTIPLE GROW OPERATIONS")
    
    try:
        multi_grown_circuit = original_circuit.__deepcopy__()
        
        for i in range(3):
            print(f"\nGrow operation {i+1}:")
            before_nodes = len(multi_grown_circuit)
            multi_grown_circuit.grow(noise_variance=0.1)
            after_nodes = len(multi_grown_circuit)
            print(f"  Nodes: {before_nodes} → {after_nodes} (+{after_nodes - before_nodes})")
            
            if test_circuit_functionality(multi_grown_circuit, f"Multi-grown Circuit (step {i+1})"):
                print(f"  Functionality test passed")
            else:
                print(f"  Functionality test failed")
                break

        multi_grown_circuit.plot_structure()
        plt.title("Multi Grown Circuit Structure")
        plt.savefig("multi_grown_circuit_structure.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("\nMulti grown circuit structure saved as 'multi_grown_circuit_structure.png'")

    except Exception as e:
        print(f"Error during multiple grow operations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll tests completed")

if __name__ == "__main__":
    main()

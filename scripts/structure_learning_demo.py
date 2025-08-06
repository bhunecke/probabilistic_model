import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.datasets
from random_events.variable import Continuous
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit, SumUnit, ProductUnit, leaf
from probabilistic_model.distributions import GaussianDistribution, UniformDistribution
from random_events.interval import SimpleInterval

class StructureLearner:
    """
    A class to handle the iterative structure learning of a ProbabilisticCircuit
    by applying pruning and growing operations.
    """

    def __init__(self, circuit: ProbabilisticCircuit, dataset: np.ndarray, validation_set: np.ndarray = None):
        """
        Initializes the StructureLearner.

        :circuit: The initial ProbabilisticCircuit instance.
        :dataset: The dataset used for pruning and fine-tuning.
        :validation_set: Optional validation dataset for evaluation.
        """
        self.circuit = circuit
        self.dataset = dataset
        self.validation_set = validation_set if validation_set is not None else dataset
        self.initial_log_likelihood = circuit.log_likelihood(self.validation_set).sum()
        print(f"Initial model log-likelihood on validation set: {self.initial_log_likelihood:.4f}")

    def learn(self, iterations: int = 3, pruning_percentage: float = 0.75, noise_variance: float = 0.1):
        """
        Runs the iterative pruning, growing, and fine-tuning process.

        :iterations: The number of learning cycles to perform.
        :pruning_percentage: The percentage of edges to prune in each cycle.
        :noise_variance: The variance for the noise injected during growing.
        """
        print("\n--- Starting Iterative Structure Learning ---")
        for i in range(iterations):
            print(f"\n--- Iteration {i + 1}/{iterations} ---")

            # 1. Prune the circuit using the training data
            print(f"Step 1: Pruning {pruning_percentage * 100}% of sum edges...")
            self.circuit.prune(self.dataset, pruning_percentage)
            pruning_ll = self.circuit.log_likelihood(self.validation_set).sum()
            print(f"Validation log-likelihood after pruning: {pruning_ll:.4f}")

            # 2. Grow the circuit
            print("Step 2: Growing the circuit...")
            self.circuit.grow(noise_variance)
            growing_ll = self.circuit.log_likelihood(self.validation_set).sum()
            print(f"Validation log-likelihood after growing: {growing_ll:.4f}")

            # 3. Fine-tune the parameters on the training data (normalize weights)
            print("Step 3: Fine-tuning parameters...")
            self.circuit.normalize()
            current_ll = self.circuit.log_likelihood(self.validation_set).sum()
            print(f"Validation log-likelihood after fine-tuning: {current_ll:.4f}")

        print("\n--- Structure Learning Complete ---")
        final_log_likelihood = self.circuit.log_likelihood(self.validation_set).sum()
        print(f"Initial validation log-likelihood: {self.initial_log_likelihood:.4f}")
        print(f"Final validation log-likelihood:   {final_log_likelihood:.4f}")


def load_mnist_data(num_samples=1000):
    """Downloads and loads a subset of the MNIST digits dataset."""
    print("Loading MNIST digits dataset...")
    
    # Load the digits dataset (8x8 images) - this is similar to MNIST but smaller for demo purposes
    mnist = sklearn.datasets.load_digits(as_frame=False)
    images = mnist.data  # Shape: (n_samples, 64)
    labels = mnist.target
    
    # Use a subset for faster demonstration
    if num_samples < len(images):
        indices = np.random.choice(len(images), size=num_samples, replace=False)
        images = images[indices]
        labels = labels[indices]
    
    # Binarize the data as is common for PC experiments on MNIST
    # Normalize first to [0, 1] range
    images = images / 16.0  # Max pixel value in digits dataset is 16
    
    # Then binarize
    images[images >= 0.5] = 1.0
    images[images < 0.5] = 0.0
    
    # Split into training and validation
    split_idx = int(len(images) * 0.8)
    train_data = images[:split_idx]
    val_data = images[split_idx:]
    train_labels = labels[:split_idx]
    val_labels = labels[split_idx:]
    
    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")
    print(f"Data shape: {train_data.shape}, value range: [{train_data.min():.3f}, {train_data.max():.3f}]")
    return train_data, val_data, train_labels, val_labels


def create_initial_circuit_for_mnist(num_features, num_latents=16):
    """Creates a mixture model initial circuit structure for MNIST digits data.
    
    This creates a Sum-of-Products structure similar to Hidden Chow-Liu Trees (HCLTs)
    which provides multiple mixture components that can be pruned during learning.
    """
    print(f"Creating initial mixture circuit for {num_features} features with {num_latents} latent components...")
    
    # Create a probabilistic circuit
    pc = ProbabilisticCircuit()
    
    # Create variables for each pixel (treating them as continuous variables)
    variables = [Continuous(f"pixel_{i}") for i in range(num_features)]
    
    # Create the root as a sum unit (mixture model)
    root = SumUnit(probabilistic_circuit=pc)
    
    # Create multiple mixture components (product units) with different parameter settings
    # This gives the pruning algorithm multiple components to potentially remove
    for component_idx in range(num_latents):
        # Create a product unit for this mixture component
        product_unit = ProductUnit(probabilistic_circuit=pc)
        
        # Create leaf distributions for each variable with slightly different parameters
        # This creates diversity between mixture components
        for var_idx, var in enumerate(variables):
            # Vary the parameters slightly across components to create diversity
            mean_offset = (component_idx / num_latents - 0.5) * 0.4  # Range: -0.2 to 0.2
            mean = 0.5 + mean_offset
            # Clamp mean to reasonable range
            mean = max(0.1, min(0.9, mean))
            
            # Also vary variance slightly
            variance = 0.25 + (component_idx % 3 - 1) * 0.05  # Small variance around 0.25
            variance = max(0.1, min(0.5, variance))
            
            leaf_node = leaf(GaussianDistribution(var, mean, variance), pc)
            product_unit.add_subcircuit(leaf_node)
        
        # Add this product unit as a component of the mixture with equal initial weights
        # Use log weights that sum to 0 (uniform distribution)
        log_weight = np.log(1.0 / num_latents)
        root.add_subcircuit(product_unit, log_weight)
    
    # Normalize the circuit
    pc.normalize()
    
    print(f"Created mixture circuit with {len(list(pc.nodes()))} nodes and {len(list(pc.edges()))} edges.")
    print(f"Root is a SumUnit with {num_latents} mixture components.")
    return pc


if __name__ == '__main__':
    # Ensure the output directory exists
    output_dir = "structure_learning_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load MNIST digits data
    train_dataset, val_dataset, train_labels, val_labels = load_mnist_data(num_samples=200)  # Reduced for testing
    num_features = train_dataset.shape[1]

    # 2. Create an initial probabilistic circuit
    circuit = create_initial_circuit_for_mnist(num_features, num_latents=8)  # Reduced from 32
    print(f"\nInitial circuit created with {len(list(circuit.nodes()))} nodes and {len(list(circuit.edges()))} edges.")

    # Visualize the initial circuit
    initial_path = os.path.join(output_dir, "initial_mnist_circuit.png")
    try:
        circuit.plot_structure()
        plt.savefig(initial_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Initial circuit structure saved to '{initial_path}'")
    except Exception as e:
        print(f"Warning: Could not save circuit plot due to: {e}")
        print("This is normal for large circuits - skipping visualization")

    # 3. Run the structure learning process
    learner = StructureLearner(circuit, train_dataset, val_dataset)
    learner.learn(iterations=2, pruning_percentage=0.5, noise_variance=0.1)

    # 4. Visualize the final, learned circuit
    final_path = os.path.join(output_dir, "final_mnist_circuit.png")
    try:
        circuit.plot_structure()
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFinal circuit has {len(list(circuit.nodes()))} nodes and {len(list(circuit.edges()))} edges.")
        print(f"Final learned circuit structure saved to '{final_path}'")
    except Exception as e:
        print(f"\nFinal circuit has {len(list(circuit.nodes()))} nodes and {len(list(circuit.edges()))} edges.")
        print(f"Warning: Could not save final circuit plot due to: {e}")
        print("This is normal for large circuits - plotting disabled for performance reasons")

    # 5. Test the final model on some sample data
    print("\n--- Final Model Evaluation ---")
    final_train_ll = circuit.log_likelihood(train_dataset[:10]).sum()  # Test on small subset
    final_val_ll = circuit.log_likelihood(val_dataset[:10]).sum()
    print(f"Final training log-likelihood (10 samples): {final_train_ll:.4f}")
    print(f"Final validation log-likelihood (10 samples): {final_val_ll:.4f}")
    
    print(f"\nStructure learning complete. Results saved to '{output_dir}/' directory.")

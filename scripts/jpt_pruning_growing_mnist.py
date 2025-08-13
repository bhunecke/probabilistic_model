import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.jpt.jpt import JPT

def mean_log_likelihood(ll_array: np.ndarray) -> float:
    ll_array = np.asarray(ll_array)
    finite_lls = ll_array[np.isfinite(ll_array)]
    
    if len(finite_lls) == 0:
        mean_finite_ll = -np.inf
    else:
        mean_finite_ll = np.mean(finite_lls)

    return mean_finite_ll

def plot_image_grid(images, titles, n_rows=4, n_cols=8, cmap=plt.cm.gray_r, feature_subset_shape=(4, 5)):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = axes.flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        if i >= len(axes):
            break
        ax = axes[i]
        if len(img) == 20:
            ax.imshow(img.reshape(feature_subset_shape), cmap=cmap)
        else:
            ax.imshow(img.reshape(8, 8), cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
    plt.tight_layout()
    return fig

def main():
    from sklearn.datasets import load_digits
    print("Loading sklearn digits dataset...")
    digits = load_digits()

    n_features = digits.data.shape[1]
    column_names = [f'pixel_{i}' for i in range(n_features)]

    dataset = pd.DataFrame(digits.data, columns=column_names)
    dataset = dataset.astype(int)

    # Use a simpler subset of the dataset
    dataset = dataset.head(500)
    # Use first 20 features, so 4x5 pixel image
    # Everything above 20 pixels will result in -inf log-likelihood :/
    dataset = dataset.iloc[:, :20]
    print(dataset.head())
    
    print(f"Loaded {len(dataset)} samples with {len(dataset.columns)} features from the digits dataset.")
    
    # Visualize Training Data
    print("\nPlotting a few training samples...")
    example_images = dataset.values[:32]
    example_titles = [f"Sample #{i}" for i in range(len(example_images))]
    fig = plot_image_grid(example_images, example_titles)
    plt.savefig("digits_training_samples.png", dpi=150)
    plt.close(fig)

    variables = infer_variables_from_dataframe(dataset, scale_continuous_types=False)
    print(f"Inferred {len(variables)} variables: {[type(v).__name__ for v in variables[:5]]}...")
    model = JPT(variables, min_impurity_improvement=0.1, min_samples_leaf=20)
    
    print("\nFitting the JPT model...")
    pc = model.fit(dataset)
    
    pc.plot_structure()
    plt.savefig("digits_circuit_structure.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate log-likelihood on the training data
    log_likelihoods = pc.log_likelihood(dataset.values)
    mean_ll = mean_log_likelihood(log_likelihoods)
    print(f"\nOriginal PC has {len(list(pc.nodes()))} nodes and {len(list(pc.edges()))} edges. Mean log-likelihood {mean_ll:.3f}.")

    # Visualize Generated Samples
    print("\nGenerating new samples from the original PC...")
    generated_samples = pc.sample(32)
    # Use actual generated samples without padding
    fig = plot_image_grid(generated_samples, [f"Gen. #{i}" for i in range(len(generated_samples))])
    plt.savefig("digits_generated_samples.png", dpi=150)
    plt.close(fig)

    # Pruning
    print("\nPruning the circuit...")
    pruned_pc = pc.prune(dataset.values, pruning_percentage=0.7)
    pruned_log_likelihoods = pruned_pc.log_likelihood(dataset.values)
    pruned_mean_ll = mean_log_likelihood(pruned_log_likelihoods)
    print(f"Pruned PC has {len(list(pruned_pc.nodes()))} nodes and {len(list(pruned_pc.edges()))} edges. Mean log-likelihood {pruned_mean_ll:.3f}.")
    
    pruned_pc.plot_structure()
    plt.savefig("circuit_structure_pruned.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Growing
    print("\nGrowing the circuit...")
    grown_pc = pruned_pc.grow(noise_variance=0.1)
    grown_log_likelihoods = grown_pc.log_likelihood(dataset.values)
    grown_mean_ll = mean_log_likelihood(grown_log_likelihoods)
    print(f"Grown PC has {len(list(grown_pc.nodes()))} nodes and {len(list(grown_pc.edges()))} edges. Mean log-likelihood {grown_mean_ll:.3f}.")

    grown_pc.plot_structure()
    plt.savefig("circuit_structure_grown.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

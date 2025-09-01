from collections import defaultdict
from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np
import tqdm
if TYPE_CHECKING:
    from .probabilistic_circuit import ProbabilisticCircuit, Unit


class CircuitFlowAnalyzer:
    """
    A class for analyzing information flow through probabilistic circuits.
    This class computes edge flows by performing forward and backward passes through the circuit.
    """
    
    def __init__(self, circuit: 'ProbabilisticCircuit'):
        """
        Initialize the flow analyzer with a probabilistic circuit.
        
        :param circuit: The probabilistic circuit to analyze.
        """
        self.circuit = circuit
    
    def compute_flows(self, dataset: np.ndarray) -> Dict[Tuple['Unit', 'Unit'], float]:
        """
        Compute the flow of information through the circuit for a given dataset.
        
        :param dataset: The input dataset.
        :return: Dictionary mapping edge tuples to their flow values.
        """
        edge_flows = defaultdict(float)

        for x in tqdm.tqdm(dataset, desc="Computing circuit flows"):
            # Ensure x is 2D (reshape single sample to batch of size 1)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            self._forward_pass(x)
            node_flows = self._backward_pass()
            self._accumulate_edge_flows(edge_flows, node_flows)
            self._cleanup_temporary_attributes()

        return edge_flows
    
    def _forward_pass(self, x: np.ndarray):
        """
        Perform forward pass through the circuit.
        
        :param x: Input sample(s).
        """
        self.circuit.log_likelihood(x)
        
        # Convert log-likelihoods to likelihoods for flow computation
        for layer in reversed(self.circuit.layers):
            for node in layer:
                if hasattr(node, 'result_of_current_query') and node.result_of_current_query is not None:
                    if isinstance(node.result_of_current_query, np.ndarray):
                        node._flow_likelihood = np.exp(node.result_of_current_query[0])
                    else:
                        node._flow_likelihood = np.exp(node.result_of_current_query)
    
    def _backward_pass(self) -> Dict['Unit', float]:
        """
        Perform backward pass to compute node flows.
        
        :return: Dictionary mapping nodes to their flow values.
        """
        from .probabilistic_circuit import SumUnit, ProductUnit
        
        node_flows = {self.circuit.root: 1.0}
        
        for layer in self.circuit.layers:
            for node in layer:
                if node.index == self.circuit.root.index:
                    continue
                    
                node_flows[node] = 0.0
                parents = self.circuit.predecessors(node)
                
                for parent in parents:
                    if isinstance(parent, SumUnit):
                        node_flows[node] += node_flows[parent]
                    elif isinstance(parent, ProductUnit):
                        if hasattr(parent, '_flow_likelihood') and parent._flow_likelihood > 0:
                            if hasattr(node, '_flow_likelihood'):
                                contribution = node._flow_likelihood / parent._flow_likelihood
                                node_flows[node] += contribution * node_flows[parent]
        
        return node_flows
    
    def _accumulate_edge_flows(self, edge_flows: defaultdict, node_flows: Dict['Unit', float]):
        """
        Accumulate edge flows based on node flows and circuit structure.
        
        :param edge_flows: Dictionary to accumulate edge flows.
        :param node_flows: Node flows from backward pass.
        """
        from .probabilistic_circuit import SumUnit
        
        for layer in self.circuit.layers:
            for node in layer:
                if isinstance(node, SumUnit):
                    for child in node.subcircuits:
                        if hasattr(node, '_flow_likelihood') and node._flow_likelihood > 0:
                            # Get weight from the edge between parent and child
                            weight = self.circuit.graph.get_edge_data(node.index, child.index)
                            if weight is None:
                                weight = 0.0
                            
                            if hasattr(child, '_flow_likelihood'):
                                edge_flow = (np.exp(weight) * child._flow_likelihood / node._flow_likelihood * node_flows[node])
                                edge_flows[(node, child)] += edge_flow
    
    def _cleanup_temporary_attributes(self):
        """
        Clean up temporary attributes added during flow computation.
        """
        for node in self.circuit.nodes():
            if hasattr(node, '_flow_likelihood'):
                delattr(node, '_flow_likelihood')

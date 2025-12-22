"""
Model Graph Executor for DAG-based strategy execution.

Responsible for:
- Parsing model DAG configuration
- Topological sorting to determine execution order
- Executing models in dependency order
- Handling input fusion (concat, stack, attention)
- Managing special inputs (market_data, position_context)

This enables arbitrary strategy architectures like:
- Single LeJEPA → Single Policy
- Multiple LeJEPAs (different timeframes) → Single Policy with fusion
- Shared LeJEPA → Multiple Policies → Ensemble combiner
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.workers.model_registry import (
    ModelRegistry,
    LoadedModel,
    ModelType,
    combine_actions,
    combine_exit_actions,
)
from src.model.policy import EntryAction, ExitAction

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """
    A node in the model graph DAG.

    Attributes:
        node: Model ID from the registry (or inline type for combiners)
        inputs: List of input names (special: market_data, position_context)
        outputs: List of output names for downstream nodes
        fusion: How to combine multiple inputs (concat, stack, attention)
        type: Optional override for node type (for inline combiners)
        config: Extra configuration for the node
    """
    node: str
    inputs: List[str] = field(default_factory=lambda: ["market_data"])
    outputs: List[str] = field(default_factory=lambda: ["output"])
    fusion: Optional[str] = None  # concat, stack, attention
    type: Optional[str] = None  # Override model type
    config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphNode":
        """Create GraphNode from dictionary configuration."""
        return cls(
            node=d["node"],
            inputs=d.get("inputs", ["market_data"]),
            outputs=d.get("outputs", ["output"]),
            fusion=d.get("fusion"),
            type=d.get("type"),
            config=d.get("config"),
        )


class ModelGraphExecutor:
    """
    Executes a DAG of models in topological order.

    The executor handles:
    1. LeJEPA models that take market data and produce embeddings
    2. Policy models that take embeddings and produce actions
    3. Exit policy models that take embeddings + position context
    4. Combiner nodes that aggregate multiple outputs

    Model types:
    - lejepa: Takes market_data, produces embedding tensor
    - entry_policy: Takes embeddings, produces action + confidence
    - exit_policy: Takes embedding + position_context, produces exit decision
    - combiner: Combines multiple outputs (voting, averaging)
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        graph: List[Union[GraphNode, Dict[str, Any]]],
        patch_length: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the graph executor.

        Args:
            model_registry: Registry containing loaded models
            graph: List of graph nodes defining the DAG
            patch_length: Default patch length for LeJEPA models
            device: Device for tensor operations
        """
        self.model_registry = model_registry
        self.patch_length = patch_length
        self.device = device

        # Parse graph nodes
        self.nodes: List[GraphNode] = [
            node if isinstance(node, GraphNode) else GraphNode.from_dict(node)
            for node in graph
        ]

        # Build node lookup
        self.node_lookup: Dict[str, GraphNode] = {}
        for node in self.nodes:
            for output_name in node.outputs:
                self.node_lookup[output_name] = node

        # Compute execution order
        self.execution_order = self._topological_sort()
        logger.debug(f"Execution order: {[n.node for n in self.execution_order]}")

    def _topological_sort(self) -> List[GraphNode]:
        """
        Sort nodes by dependencies to determine execution order.

        Uses Kahn's algorithm for topological sorting.
        """
        # Build dependency graph
        in_degree: Dict[str, int] = {node.node: 0 for node in self.nodes}
        dependents: Dict[str, List[str]] = {node.node: [] for node in self.nodes}

        # Map outputs to their producing nodes
        output_to_node: Dict[str, str] = {}
        for node in self.nodes:
            for output in node.outputs:
                output_to_node[output] = node.node

        # Count dependencies
        for node in self.nodes:
            for input_name in node.inputs:
                # Skip special inputs
                if input_name in ("market_data", "position_context"):
                    continue
                # Find which node produces this input
                if input_name in output_to_node:
                    producer = output_to_node[input_name]
                    if producer != node.node:
                        in_degree[node.node] += 1
                        dependents[producer].append(node.node)

        # Kahn's algorithm
        queue = [node for node in self.nodes if in_degree[node.node] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for dep_name in dependents[current.node]:
                in_degree[dep_name] -= 1
                if in_degree[dep_name] == 0:
                    dep_node = next(n for n in self.nodes if n.node == dep_name)
                    queue.append(dep_node)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def execute(
        self,
        market_data: pd.DataFrame,
        idx: int,
        position_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the model graph.

        Args:
            market_data: DataFrame with market features (normalized)
            idx: Current index in the DataFrame
            position_context: Optional position state if in a trade

        Returns:
            Dictionary with all outputs from the graph, including:
            - entry_action: EntryAction if an entry policy was executed
            - exit_action: ExitAction if an exit policy was executed
            - confidence: Confidence score if available
            - embeddings: Dict of node_id -> embedding tensor
        """
        # Initialize outputs with special inputs
        outputs: Dict[str, Any] = {
            "market_data": (market_data, idx),
            "position_context": position_context,
        }

        # Cache for embeddings (for logging/analysis)
        embeddings: Dict[str, torch.Tensor] = {}

        # Execute nodes in topological order
        for node in self.execution_order:
            try:
                result = self._execute_node(node, outputs)

                # Store results for downstream nodes
                for i, output_name in enumerate(node.outputs):
                    if isinstance(result, tuple) and len(result) > i:
                        outputs[output_name] = result[i]
                    elif i == 0:
                        outputs[output_name] = result
                    else:
                        outputs[output_name] = None

                # Track embeddings
                if isinstance(result, torch.Tensor):
                    embeddings[node.node] = result

            except Exception as e:
                logger.error(f"Error executing node {node.node}: {e}")
                raise

        # Extract final outputs
        final_outputs = {
            "embeddings": embeddings,
            "raw_outputs": outputs,
        }

        # Look for entry_action in outputs
        if "entry_action" in outputs:
            action_result = outputs["entry_action"]
            if isinstance(action_result, tuple):
                final_outputs["entry_action"] = action_result[0]
                final_outputs["confidence"] = action_result[1]
            elif isinstance(action_result, EntryAction):
                final_outputs["entry_action"] = action_result
            elif isinstance(action_result, (int, np.integer)):
                final_outputs["entry_action"] = EntryAction(action_result)

        # Look for exit_action in outputs
        if "exit_action" in outputs:
            action_result = outputs["exit_action"]
            if isinstance(action_result, tuple):
                final_outputs["exit_action"] = action_result[0]
            elif isinstance(action_result, ExitAction):
                final_outputs["exit_action"] = action_result

        return final_outputs

    def _execute_node(
        self,
        node: GraphNode,
        outputs: Dict[str, Any],
    ) -> Any:
        """
        Execute a single node.

        Args:
            node: The graph node to execute
            outputs: Current outputs dictionary

        Returns:
            Node output (tensor, action, or tuple)
        """
        # Gather inputs
        inputs = {}
        for input_name in node.inputs:
            if input_name in outputs:
                inputs[input_name] = outputs[input_name]
            else:
                logger.warning(f"Input {input_name} not found for node {node.node}")
                inputs[input_name] = None

        # Determine node type
        node_type = node.type
        if node_type is None:
            # Infer from model registry
            if self.model_registry.has(node.node):
                loaded = self.model_registry.get(node.node)
                node_type = loaded.type.value

        # Execute based on type
        if node_type == "lejepa":
            return self._execute_lejepa(node, inputs)

        elif node_type in ("entry_policy", "regression_policy"):
            return self._execute_entry_policy(node, inputs)

        elif node_type == "exit_policy":
            return self._execute_exit_policy(node, inputs)

        elif node_type in ("rule_based_exit", "continuous_exit"):
            return self._execute_rule_based_exit(node, inputs, outputs)

        elif node_type == "combiner":
            return self._execute_combiner(node, inputs)

        else:
            raise ValueError(f"Unknown node type: {node_type} for node {node.node}")

    def _execute_lejepa(
        self,
        node: GraphNode,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Execute a LeJEPA encoder node.

        Produces an embedding tensor from market data.
        """
        loaded = self.model_registry.get(node.node)
        model = loaded.model

        # Get market data
        market_data, idx = inputs.get("market_data", (None, None))
        if market_data is None:
            raise ValueError(f"market_data required for LeJEPA node {node.node}")

        # Resample if needed for this model's timeframe
        timeframe = loaded.timeframe
        if timeframe != "1m":
            market_data = self.model_registry.get_resampled_data(
                market_data,
                timeframe,
                cache_key=f"exec_{idx}",
            )
            # Adjust idx for resampled data
            if idx >= len(market_data):
                idx = len(market_data) - 1

        # Get patch length from config
        patch_length = loaded.config.get("patch_length", self.patch_length)

        # Create patch from data
        patch = self._create_patch(market_data, idx, patch_length)
        if patch is None:
            # Return zero embedding if not enough history
            embedding_dim = loaded.config.get("embedding_dim", 512)
            return torch.zeros(1, embedding_dim, device=self.device)

        # Forward through encoder
        with torch.no_grad():
            embedding = model.encode_context(patch)

        return embedding

    def _create_patch(
        self,
        df: pd.DataFrame,
        idx: int,
        patch_length: int,
    ) -> Optional[torch.Tensor]:
        """
        Create a patch tensor from DataFrame.

        Args:
            df: Market data DataFrame
            idx: Current index (end of patch)
            patch_length: Number of timesteps in patch

        Returns:
            Tensor of shape (1, patch_length, num_features) or None if not enough data
        """
        # Get feature columns (exclude timestamp if present)
        feature_cols = [c for c in df.columns if c != "timestamp"]

        # Calculate start index
        start_idx = idx - patch_length + 1
        if start_idx < 0:
            # Not enough history
            return None

        # Extract patch data
        patch_data = df.iloc[start_idx:idx + 1][feature_cols].values

        if len(patch_data) < patch_length:
            # Pad with zeros if needed
            pad_size = patch_length - len(patch_data)
            padding = np.zeros((pad_size, patch_data.shape[1]))
            patch_data = np.vstack([padding, patch_data])

        # Convert to tensor
        patch = torch.tensor(
            patch_data,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # Add batch dimension

        return patch

    def _fuse_inputs(
        self,
        inputs: Dict[str, Any],
        fusion: Optional[str],
    ) -> torch.Tensor:
        """
        Fuse multiple input tensors based on fusion method.

        Args:
            inputs: Dictionary of input tensors
            fusion: Fusion method (concat, stack, attention)

        Returns:
            Fused tensor
        """
        # Filter to tensor inputs only
        tensors = [v for v in inputs.values() if isinstance(v, torch.Tensor)]

        if not tensors:
            raise ValueError("No tensor inputs for fusion")

        if len(tensors) == 1:
            return tensors[0]

        fusion = fusion or "concat"

        if fusion == "concat":
            # Concatenate along feature dimension
            return torch.cat(tensors, dim=-1)

        elif fusion == "stack":
            # Stack as separate inputs (for attention-based fusion later)
            return torch.stack(tensors, dim=1)

        elif fusion == "attention":
            # Simple attention-based fusion (learnable would require separate model)
            # Use mean pooling as simple attention
            stacked = torch.stack(tensors, dim=1)  # (B, N, D)
            return stacked.mean(dim=1)  # (B, D)

        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    def _execute_entry_policy(
        self,
        node: GraphNode,
        inputs: Dict[str, Any],
    ) -> Tuple[EntryAction, float]:
        """
        Execute an entry policy node.

        Returns (action, confidence) tuple.
        """
        loaded = self.model_registry.get(node.node)
        model = loaded.model

        # Fuse inputs if multiple
        input_keys = [k for k in inputs if k not in ("market_data", "position_context")]
        if not input_keys:
            raise ValueError(f"No embedding inputs for entry policy {node.node}")

        input_tensors = {k: inputs[k] for k in input_keys if isinstance(inputs[k], torch.Tensor)}
        embedding = self._fuse_inputs(input_tensors, node.fusion)

        # Forward through policy
        with torch.no_grad():
            result = model(embedding)

        # Handle different output formats
        if loaded.type == ModelType.REGRESSION_POLICY:
            # Regression policy outputs continuous score
            score = result.item()
            # Convert to action based on thresholds
            if abs(score) < 0.1:
                action = EntryAction.HOLD
            elif score > 0:
                action = EntryAction.BUY_CALL_ATM if score < 0.5 else EntryAction.BUY_CALL_OTM
            else:
                action = EntryAction.BUY_PUT_ATM if score > -0.5 else EntryAction.BUY_PUT_OTM
            confidence = abs(score)
        else:
            # Classification policy outputs probabilities
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            probs = torch.softmax(logits, dim=-1)
            action_idx = probs.argmax(dim=-1).item()
            confidence = probs[0, action_idx].item()
            action = EntryAction(action_idx)

        return (action, confidence)

    def _execute_exit_policy(
        self,
        node: GraphNode,
        inputs: Dict[str, Any],
    ) -> Tuple[ExitAction, Optional[str]]:
        """
        Execute an exit policy node.

        Returns (exit_action, reason) tuple.
        """
        loaded = self.model_registry.get(node.node)
        model = loaded.model

        # Get embedding input
        input_keys = [k for k in inputs if k not in ("market_data", "position_context")]
        input_tensors = {k: inputs[k] for k in input_keys if isinstance(inputs[k], torch.Tensor)}
        embedding = self._fuse_inputs(input_tensors, node.fusion)

        # Get position context
        position_context = inputs.get("position_context")
        if position_context is None:
            # No position = no exit decision needed
            return (ExitAction.HOLD_POSITION, None)

        # Build context tensor
        context = torch.tensor([
            position_context.get("position_type", 0),
            position_context.get("unrealized_pnl", 0),
            position_context.get("bars_held", 0) / 100.0,  # Normalize
            position_context.get("time_to_close", 0) / 6.5,  # Normalize to [0,1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward through policy
        with torch.no_grad():
            result = model(embedding, context)

        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result

        probs = torch.softmax(logits, dim=-1)
        action_idx = probs.argmax(dim=-1).item()
        action = ExitAction(action_idx)

        reason = "neural_exit" if action == ExitAction.CLOSE else None
        return (action, reason)

    def _execute_rule_based_exit(
        self,
        node: GraphNode,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Tuple[ExitAction, Optional[str]]:
        """
        Execute a rule-based exit policy.
        """
        loaded = self.model_registry.get(node.node)
        policy = loaded.model  # RuleBasedExitPolicy or ContinuousSignalExitPolicy

        position_context = inputs.get("position_context")
        if position_context is None:
            return (ExitAction.HOLD_POSITION, None)

        if loaded.type == ModelType.RULE_BASED_EXIT:
            # RuleBasedExitPolicy uses evaluate method
            result = policy.evaluate(
                unrealized_pnl_pct=position_context.get("unrealized_pnl", 0) * 100,
                bars_held=position_context.get("bars_held", 0),
                minutes_to_close=position_context.get("time_to_close", 0) * 60,
            )
            return result

        elif loaded.type == ModelType.CONTINUOUS_EXIT:
            # ContinuousSignalExitPolicy needs current action signal
            current_action = outputs.get("entry_action")
            if isinstance(current_action, tuple):
                current_action = current_action[0]

            result = policy.evaluate(
                current_action=current_action,
                position_type=position_context.get("position_type", 1),
                unrealized_pnl_pct=position_context.get("unrealized_pnl", 0) * 100,
                minutes_to_close=position_context.get("time_to_close", 0) * 60,
            )
            return result

        return (ExitAction.HOLD_POSITION, None)

    def _execute_combiner(
        self,
        node: GraphNode,
        inputs: Dict[str, Any],
    ) -> Any:
        """
        Execute a combiner node.

        Combines multiple action outputs using the specified method.
        """
        config = node.config or {}
        method = config.get("method", "majority")
        weights = config.get("weights")

        # Collect actions to combine
        actions = []
        for input_name in node.inputs:
            value = inputs.get(input_name)
            if value is None:
                continue

            if isinstance(value, tuple):
                value = value[0]

            if isinstance(value, EntryAction):
                actions.append(value)
            elif isinstance(value, ExitAction):
                # Handle exit actions separately
                result = combine_exit_actions(
                    [v[0] if isinstance(v, tuple) else v for v in inputs.values() if isinstance(v, (ExitAction, tuple))],
                    method=method,
                )
                return (result, None)

        if not actions:
            return EntryAction.HOLD

        # Combine entry actions
        combined = combine_actions(actions, method=method, weights=weights)

        # Calculate combined confidence (average)
        confidences = []
        for input_name in node.inputs:
            value = inputs.get(input_name)
            if isinstance(value, tuple) and len(value) > 1:
                confidences.append(value[1])

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        return (combined, avg_confidence)


def create_simple_graph(
    lejepa_id: str,
    policy_id: str,
    exit_mode: str = "rule_based",
    exit_config: Optional[Dict[str, Any]] = None,
) -> List[GraphNode]:
    """
    Create a simple graph with one LeJEPA and one policy.

    This is a convenience function for common strategy patterns.

    Args:
        lejepa_id: Model ID for the LeJEPA encoder
        policy_id: Model ID for the entry policy
        exit_mode: Exit policy type (rule_based, neural, continuous_signal)
        exit_config: Configuration for rule-based exit

    Returns:
        List of GraphNode objects
    """
    nodes = [
        GraphNode(
            node=lejepa_id,
            inputs=["market_data"],
            outputs=["embedding"],
        ),
        GraphNode(
            node=policy_id,
            inputs=["embedding"],
            outputs=["entry_action"],
        ),
    ]

    if exit_mode == "rule_based":
        nodes.append(
            GraphNode(
                node="rule_exit",
                type="rule_based_exit",
                inputs=["position_context"],
                outputs=["exit_action"],
                config=exit_config or {"take_profit_pct": 50.0, "risk_reward_ratio": 2.0},
            )
        )
    elif exit_mode == "continuous_signal":
        nodes.append(
            GraphNode(
                node="continuous_exit",
                type="continuous_exit",
                inputs=["entry_action", "position_context"],
                outputs=["exit_action"],
                config=exit_config or {},
            )
        )

    return nodes


def create_multi_timeframe_graph(
    lejepa_ids: List[str],
    policy_id: str,
    fusion: str = "concat",
) -> List[GraphNode]:
    """
    Create a multi-timeframe graph with multiple LeJEPAs feeding into one policy.

    Args:
        lejepa_ids: List of LeJEPA model IDs (different timeframes)
        policy_id: Model ID for the entry policy
        fusion: How to combine embeddings (concat, stack, attention)

    Returns:
        List of GraphNode objects
    """
    nodes = []

    # Add LeJEPA nodes
    embedding_names = []
    for i, lejepa_id in enumerate(lejepa_ids):
        embedding_name = f"embedding_{i}"
        embedding_names.append(embedding_name)
        nodes.append(
            GraphNode(
                node=lejepa_id,
                inputs=["market_data"],
                outputs=[embedding_name],
            )
        )

    # Add policy node with fused inputs
    nodes.append(
        GraphNode(
            node=policy_id,
            inputs=embedding_names,
            outputs=["entry_action"],
            fusion=fusion,
        )
    )

    return nodes

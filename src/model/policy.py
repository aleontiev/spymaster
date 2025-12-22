"""
Dual Policy Networks for 0DTE Options Trading.

Implements a two-policy architecture:
1. EntryPolicy: Supervised classifier for trade direction (HOLD/CALL/PUT)
2. ExitPolicy: RL-trained policy for position management (HOLD/CLOSE)

The entry policy predicts directional moves based on market state.
The exit policy learns optimal exit timing using realized P&L as reward.
"""
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntryAction(IntEnum):
    """Entry action space (5-class: direction + moneyness)."""
    HOLD = 0           # No entry (noise/choppy conditions)
    BUY_CALL_ATM = 1   # Buy ATM call (~50 delta) - high probability, lower leverage
    BUY_CALL_OTM = 2   # Buy OTM call (~20-30 delta) - lower probability, massive leverage
    BUY_PUT_ATM = 3    # Buy ATM put (~-50 delta) - high probability, lower leverage
    BUY_PUT_OTM = 4    # Buy OTM put (~-20-30 delta) - lower probability, massive leverage


# Legacy 3-class mapping for backward compatibility
class EntryActionLegacy(IntEnum):
    """Legacy 3-class entry action space (for backward compatibility)."""
    HOLD = 0       # No entry
    BUY_CALL = 1   # Enter long (call)
    BUY_PUT = 2    # Enter short (put)


class ExitAction(IntEnum):
    """Exit action space."""
    HOLD_POSITION = 0  # Keep position open
    CLOSE = 1          # Close position


# Legacy compatibility
class TradingAction(IntEnum):
    """Combined trading action space (legacy, for backtest compatibility)."""
    HOLD = 0
    BUY_CALL = 1
    BUY_PUT = 2
    CLOSE_POSITION = 3


class EntryPolicy(nn.Module):
    """
    Supervised entry policy for trade direction and moneyness prediction.

    Takes LeJEPA embeddings and outputs probabilities for:
    - HOLD (no trade - noise/choppy conditions)
    - BUY_CALL_ATM (bullish, ATM ~50 delta - high probability, lower leverage)
    - BUY_CALL_OTM (bullish, OTM ~20-30 delta - lower probability, massive leverage)
    - BUY_PUT_ATM (bearish, ATM ~-50 delta - high probability, lower leverage)
    - BUY_PUT_OTM (bearish, OTM ~-20-30 delta - lower probability, massive leverage)

    The model learns to predict not just direction but also convexity/moneyness.
    OTM is preferred when the IV/Gamma setup suggests an explosive move.
    ATM is preferred for grinding days with steady directional moves.

    Trained with cross-entropy/focal loss using option ROI comparison as labels.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_actions: int = 5,  # 5-class: HOLD, CALL_ATM, CALL_OTM, PUT_ATM, PUT_OTM
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions

        # Feature extractor
        layers = []
        in_dim = embedding_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Action head
        self.action_head = nn.Linear(hidden_dim, self.num_actions)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]

        Returns:
            Dict with action_logits and action_probs
        """
        features = self.feature_extractor(embeddings)
        logits = self.action_head(features)
        probs = F.softmax(logits, dim=-1)

        return {
            "action_logits": logits,
            "action_probs": probs,
        }

    def get_action(
        self,
        embeddings: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from embeddings.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]
            deterministic: Use argmax instead of sampling

        Returns:
            Tuple of (actions, probabilities)
        """
        output = self.forward(embeddings)
        probs = output["action_probs"]

        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

        return actions, probs


class RegressionEntryPolicy(nn.Module):
    """
    Continuous regression policy for trade direction and conviction.

    Instead of 5-class classification, predicts a continuous score in [-1, 1]:
    - Score > 0: Bullish (buy call), magnitude = conviction
    - Score < 0: Bearish (buy put), magnitude = conviction
    - Score ≈ 0: No signal (hold)

    This aligns better with the low-rank latent space (~3-4 factors) discovered
    by LeJEPA. Financial markets have continuous conviction, not discrete buckets.

    Advantages over classification:
    1. Smoother moneyness decisions (high conviction → OTM, low → ATM)
    2. Natural confidence calibration via magnitude
    3. Simpler loss (MSE vs cross-entropy with class imbalance)
    4. Maps directly to the continuous latent factors

    Target: Y = clip(FutureReturn * 100, -1, 1)
    - A 0.5% move becomes score 0.5
    - A 1%+ move saturates at 1.0 (max conviction)

    Architecture: MLP with Tanh output for bounded predictions.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Feature extractor
        layers = []
        in_dim = embedding_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # GELU tends to work better for regression
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Single scalar output with Tanh for [-1, 1] bounded output
        self.score_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]

        Returns:
            Continuous score in [-1, 1] of shape [B, 1]
        """
        features = self.feature_extractor(embeddings)
        score = torch.tanh(self.score_head(features))  # Bounded to [-1, 1]
        return score

    def predict_trade(
        self,
        embeddings: torch.Tensor,
        signal_threshold: float = 0.3,
        high_conviction_threshold: float = 0.6,
        iv_regime: str = "normal",
    ) -> Tuple[str, float, Optional[str]]:
        """
        Execution logic: Maps score -> Trade decision.

        Args:
            embeddings: LeJEPA embeddings [1, embedding_dim] (single sample)
            signal_threshold: Minimum |score| to enter trade (default 0.3)
            high_conviction_threshold: |score| above which to use OTM (default 0.6)
            iv_regime: 'low', 'normal', 'high' (from GEX engine)

        Returns:
            Tuple of (action, conviction, moneyness)
            - action: "HOLD", "BUY_CALL", or "BUY_PUT"
            - conviction: Absolute value of score (0 to 1)
            - moneyness: "ATM", "OTM", or None (if HOLD)
        """
        with torch.no_grad():
            score = self.forward(embeddings).item()

        conviction = abs(score)

        # 1. Check threshold - no trade if conviction too low
        if conviction < signal_threshold:
            return "HOLD", conviction, None

        # 2. Direction from sign
        action = "BUY_CALL" if score > 0 else "BUY_PUT"

        # 3. Moneyness logic: conviction + IV regime
        # High conviction AND IV not expensive → Go OTM for leverage
        # Otherwise → ATM for higher probability
        if conviction > high_conviction_threshold and iv_regime != "high":
            moneyness = "OTM"
        else:
            moneyness = "ATM"

        return action, conviction, moneyness

    def predict_batch(
        self,
        embeddings: torch.Tensor,
        signal_threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch prediction for efficient inference.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]
            signal_threshold: Minimum |score| to enter trade

        Returns:
            Tuple of (actions, convictions, scores)
            - actions: Integer tensor [B] with 0=HOLD, 1=BUY_CALL, 2=BUY_PUT
            - convictions: Float tensor [B] with |score|
            - scores: Raw scores [B] in [-1, 1]
        """
        with torch.no_grad():
            scores = self.forward(embeddings).squeeze(-1)  # [B]

        convictions = scores.abs()

        # Map to actions: 0=HOLD, 1=CALL, 2=PUT
        actions = torch.zeros_like(scores, dtype=torch.long)
        actions[scores > signal_threshold] = 1   # BUY_CALL
        actions[scores < -signal_threshold] = 2  # BUY_PUT

        return actions, convictions, scores


class ExitPolicy(nn.Module):
    """
    RL-trained exit policy for position management.

    Takes LeJEPA embeddings PLUS position context:
    - position_type: +1 (call) or -1 (put)
    - unrealized_pnl: Current P&L (normalized)
    - bars_held: How long position has been held (normalized)
    - time_to_close: Hours until market close

    Outputs probabilities for:
    - HOLD_POSITION: Keep the trade open
    - CLOSE: Exit the position

    Trained with RL using realized P&L as reward signal.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        context_dim: int = 4,  # position_type, pnl, bars_held, time_to_close
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_actions = 2  # HOLD, CLOSE

        # Embedding + context projection
        self.input_projection = nn.Linear(embedding_dim + context_dim, hidden_dim)

        # Feature extractor
        layers = []
        in_dim = hidden_dim

        for i in range(num_layers - 1):  # -1 because input_projection is first layer
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()

        # Action head (policy)
        self.action_head = nn.Linear(hidden_dim, self.num_actions)

        # Value head (for advantage estimation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        position_context: torch.Tensor,
        return_value: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]
            position_context: Position context [B, context_dim]
                - position_context[:, 0]: position_type (+1 call, -1 put)
                - position_context[:, 1]: unrealized_pnl (normalized)
                - position_context[:, 2]: bars_held (normalized, 0-1)
                - position_context[:, 3]: time_to_close (hours, normalized)
            return_value: Whether to compute value estimate

        Returns:
            Dict with action_logits, action_probs, and optionally value
        """
        # Concatenate embedding with position context
        combined = torch.cat([embeddings, position_context], dim=-1)

        # Project and extract features
        x = self.input_projection(combined)
        x = F.relu(x)
        features = self.feature_extractor(x)

        # Action head
        logits = self.action_head(features)
        probs = F.softmax(logits, dim=-1)

        output = {
            "action_logits": logits,
            "action_probs": probs,
        }

        if return_value:
            output["value"] = self.value_head(features)

        return output

    def get_action(
        self,
        embeddings: torch.Tensor,
        position_context: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from embeddings and position context.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]
            position_context: Position context [B, context_dim]
            deterministic: Use argmax instead of sampling

        Returns:
            Tuple of (actions, log_probs, values)
        """
        output = self.forward(embeddings, position_context, return_value=True)
        probs = output["action_probs"]

        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()

        # Log probabilities
        log_probs = F.log_softmax(output["action_logits"], dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        values = output["value"].squeeze(-1)

        return actions, selected_log_probs, values

    def evaluate_actions(
        self,
        embeddings: torch.Tensor,
        position_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Args:
            embeddings: LeJEPA embeddings [B, embedding_dim]
            position_context: Position context [B, context_dim]
            actions: Action indices [B]

        Returns:
            Tuple of (log_probs, entropy, values)
        """
        output = self.forward(embeddings, position_context, return_value=True)

        log_probs = F.log_softmax(output["action_logits"], dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        probs = output["action_probs"]
        entropy = -(probs * log_probs).sum(dim=-1)

        values = output["value"].squeeze(-1)

        return selected_log_probs, entropy, values


class RuleBasedExitPolicy:
    """
    Rule-based exit policy using configurable take-profit and stop-loss thresholds.

    This policy does not use neural networks. Instead, it exits positions based on:
    - Take Profit (TP): Close when unrealized P&L >= take_profit_pct
    - Stop Loss (SL): Close when unrealized P&L <= -stop_loss_pct

    The stop loss is automatically calculated from take_profit and risk_reward_ratio:
        stop_loss_pct = take_profit_pct / risk_reward_ratio

    Example:
        If take_profit_pct=50 and risk_reward_ratio=2.0:
        - Take Profit triggers at +50% P&L
        - Stop Loss triggers at -25% P&L (50 / 2 = 25)
        - This gives a 1:2 risk:reward ratio

    Args:
        take_profit_pct: Take profit threshold in percent (e.g., 50.0 = +50%)
        risk_reward_ratio: Ratio of reward to risk (e.g., 2.0 = 1:2 R:R)
        stop_loss_pct: Optional explicit stop loss percent. If provided, overrides
                       the calculated value from risk_reward_ratio.
        time_stop_hours: Optional time-based stop. Close position if held longer
                         than this many hours. None = disabled.
        eod_exit_minutes: Minutes before market close to force exit. Default 5.

    Usage:
        policy = RuleBasedExitPolicy(take_profit_pct=50.0, risk_reward_ratio=2.0)
        # This sets TP=+50%, SL=-25%

        # In trading loop:
        action = policy.get_action(unrealized_pnl_pct, time_to_close_hours)
        if action == ExitAction.CLOSE:
            close_position()
    """

    def __init__(
        self,
        take_profit_pct: float = 50.0,
        risk_reward_ratio: float = 2.0,
        stop_loss_pct: Optional[float] = None,
        time_stop_hours: Optional[float] = None,
        eod_exit_minutes: float = 5.0,
    ) -> None:
        self.take_profit_pct = take_profit_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.time_stop_hours = time_stop_hours
        self.eod_exit_minutes = eod_exit_minutes

        # Calculate stop loss from R:R ratio if not explicitly provided
        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct
        else:
            self.stop_loss_pct = take_profit_pct / risk_reward_ratio

    def get_action(
        self,
        unrealized_pnl_pct: float,
        time_to_close_hours: float,
        hours_held: Optional[float] = None,
    ) -> ExitAction:
        """
        Determine exit action based on P&L and time thresholds.

        Args:
            unrealized_pnl_pct: Current unrealized P&L as a percentage
                                (e.g., 25.0 = +25%, -10.0 = -10%)
            time_to_close_hours: Hours until market close
            hours_held: Optional hours the position has been held

        Returns:
            ExitAction.CLOSE if any exit condition is met, else ExitAction.HOLD_POSITION
        """
        # Check take profit
        if unrealized_pnl_pct >= self.take_profit_pct:
            return ExitAction.CLOSE

        # Check stop loss
        if unrealized_pnl_pct <= -self.stop_loss_pct:
            return ExitAction.CLOSE

        # Check time stop (if enabled)
        if self.time_stop_hours is not None and hours_held is not None:
            if hours_held >= self.time_stop_hours:
                return ExitAction.CLOSE

        # Check end-of-day exit
        if time_to_close_hours <= (self.eod_exit_minutes / 60.0):
            return ExitAction.CLOSE

        return ExitAction.HOLD_POSITION

    def get_action_batch(
        self,
        unrealized_pnl_pct: torch.Tensor,
        time_to_close_hours: torch.Tensor,
        hours_held: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch version of get_action for vectorized operations.

        Args:
            unrealized_pnl_pct: Tensor of unrealized P&L percentages [B]
            time_to_close_hours: Tensor of hours until close [B]
            hours_held: Optional tensor of hours held [B]

        Returns:
            Tensor of actions [B] (0 = HOLD, 1 = CLOSE)
        """
        # Initialize all as HOLD
        actions = torch.zeros_like(unrealized_pnl_pct, dtype=torch.long)

        # Take profit condition
        tp_condition = unrealized_pnl_pct >= self.take_profit_pct
        actions[tp_condition] = ExitAction.CLOSE

        # Stop loss condition
        sl_condition = unrealized_pnl_pct <= -self.stop_loss_pct
        actions[sl_condition] = ExitAction.CLOSE

        # Time stop condition
        if self.time_stop_hours is not None and hours_held is not None:
            time_condition = hours_held >= self.time_stop_hours
            actions[time_condition] = ExitAction.CLOSE

        # End-of-day condition
        eod_condition = time_to_close_hours <= (self.eod_exit_minutes / 60.0)
        actions[eod_condition] = ExitAction.CLOSE

        return actions

    def __repr__(self) -> str:
        return (
            f"RuleBasedExitPolicy("
            f"TP={self.take_profit_pct:+.1f}%, "
            f"SL={-self.stop_loss_pct:.1f}%, "
            f"R:R=1:{self.risk_reward_ratio:.1f}"
            f")"
        )


class ContinuousSignalExitPolicy:
    """
    Continuous Signal Exit Policy - uses entry policy signals for exit decisions.

    Instead of a separate exit policy, this uses the entry policy continuously while
    in a position. Exit decisions are based on:

    1. **Counter-signal confirmation**: If the entry policy emits a counter-signal
       (e.g., PUT while holding CALL) for N consecutive minutes, exit and optionally
       flip to the new direction.

    2. **Momentum plateau**: If HOLD signals dominate over the current direction
       signal in the last M minutes, exit (the move has plateaued).

    3. **Stop loss / Take profit**: Optional P&L-based thresholds as safety net.

    4. **End-of-day**: Force exit before market close.

    This approach treats the entry policy as a continuous market regime indicator
    rather than a one-shot entry decision.

    Args:
        counter_signal_confirmation: Consecutive minutes of counter-signal to trigger exit (default: 2)
        plateau_window: Rolling window in minutes to check for momentum plateau (default: 15)
        plateau_ratio: If HOLD/counter ratio exceeds this in the window, exit (default: 0.5)
        stop_loss_pct: Optional stop loss percentage (default: 25.0)
        take_profit_pct: Optional take profit percentage (default: None = disabled)
        eod_exit_minutes: Minutes before close to force exit (default: 5.0)
        flip_on_counter: Whether to flip position on counter-signal (default: True)

    Usage:
        policy = ContinuousSignalExitPolicy(
            counter_signal_confirmation=2,
            plateau_window=15,
            plateau_ratio=0.5,
        )

        # In trading loop:
        exit_decision = policy.check_exit(
            current_position_type=PositionType.CALL,
            latest_entry_action=EntryAction.BUY_PUT,  # Counter-signal!
            signal_history=[...],  # Last N entry actions
            unrealized_pnl_pct=15.0,
            time_to_close_hours=2.5,
        )
    """

    def __init__(
        self,
        counter_signal_confirmation: int = 2,
        plateau_window: int = 15,
        plateau_ratio: float = 0.5,
        stop_loss_pct: float = 25.0,
        take_profit_pct: Optional[float] = None,
        eod_exit_minutes: float = 5.0,
        flip_on_counter: bool = True,
    ) -> None:
        self.counter_signal_confirmation = counter_signal_confirmation
        self.plateau_window = plateau_window
        self.plateau_ratio = plateau_ratio
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.eod_exit_minutes = eod_exit_minutes
        self.flip_on_counter = flip_on_counter

        # Track recent signals for each position
        self.signal_history: list = []
        self.consecutive_counter_signals: int = 0

    def reset(self) -> None:
        """Reset state when entering a new position."""
        self.signal_history = []
        self.consecutive_counter_signals = 0

    def _is_call_action(self, action: EntryAction) -> bool:
        """Check if action is a CALL signal (ATM or OTM)."""
        return action in (EntryAction.BUY_CALL_ATM, EntryAction.BUY_CALL_OTM)

    def _is_put_action(self, action: EntryAction) -> bool:
        """Check if action is a PUT signal (ATM or OTM)."""
        return action in (EntryAction.BUY_PUT_ATM, EntryAction.BUY_PUT_OTM)

    def _is_counter_signal(
        self,
        position_type: "PositionType",
        entry_action: EntryAction,
    ) -> bool:
        """Check if entry action is a counter-signal to current position."""
        if position_type.value == "call":
            return self._is_put_action(entry_action)
        else:  # put
            return self._is_call_action(entry_action)

    def _is_same_direction(
        self,
        position_type: "PositionType",
        entry_action: EntryAction,
    ) -> bool:
        """Check if entry action matches current position direction."""
        if position_type.value == "call":
            return self._is_call_action(entry_action)
        else:  # put
            return self._is_put_action(entry_action)

    def check_exit(
        self,
        position_type: "PositionType",
        latest_entry_action: EntryAction,
        unrealized_pnl_pct: float,
        time_to_close_hours: float,
    ) -> Tuple[ExitAction, str, Optional[EntryAction]]:
        """
        Check if position should be exited based on continuous signal analysis.

        Args:
            position_type: Current position type (CALL or PUT)
            latest_entry_action: Most recent entry policy prediction
            unrealized_pnl_pct: Current unrealized P&L percentage
            time_to_close_hours: Hours until market close

        Returns:
            Tuple of:
            - ExitAction (HOLD_POSITION or CLOSE)
            - Reason string (empty if holding)
            - Optional new position to enter (if flip_on_counter is True)
        """
        # Add to signal history
        self.signal_history.append(latest_entry_action)

        # Trim history to plateau window
        if len(self.signal_history) > self.plateau_window:
            self.signal_history = self.signal_history[-self.plateau_window:]

        # Check end-of-day
        if time_to_close_hours <= (self.eod_exit_minutes / 60.0):
            return ExitAction.CLOSE, "EOD", None

        # Check stop loss
        if unrealized_pnl_pct <= -self.stop_loss_pct:
            return ExitAction.CLOSE, f"SL ({unrealized_pnl_pct:.1f}%)", None

        # Check take profit (if enabled)
        if self.take_profit_pct is not None and unrealized_pnl_pct >= self.take_profit_pct:
            return ExitAction.CLOSE, f"TP ({unrealized_pnl_pct:.1f}%)", None

        # Check counter-signal confirmation
        if self._is_counter_signal(position_type, latest_entry_action):
            self.consecutive_counter_signals += 1
            if self.consecutive_counter_signals >= self.counter_signal_confirmation:
                new_position = latest_entry_action if self.flip_on_counter else None
                return ExitAction.CLOSE, f"Counter ({self.consecutive_counter_signals}x)", new_position
        else:
            self.consecutive_counter_signals = 0

        # Check momentum plateau (only if we have enough history)
        if len(self.signal_history) >= self.plateau_window:
            same_direction_count = sum(
                1 for a in self.signal_history if self._is_same_direction(position_type, a)
            )
            hold_count = sum(1 for a in self.signal_history if a == EntryAction.HOLD)
            counter_count = sum(
                1 for a in self.signal_history if self._is_counter_signal(position_type, a)
            )

            # Plateau: HOLD + counter signals exceed same-direction signals
            non_same_direction = hold_count + counter_count
            same_direction_ratio = same_direction_count / len(self.signal_history)

            if same_direction_ratio < (1 - self.plateau_ratio):
                return ExitAction.CLOSE, f"Plateau ({same_direction_ratio:.0%} same)", None

        return ExitAction.HOLD_POSITION, "", None

    def get_exit_reason_stats(self) -> Dict[str, int]:
        """Get signal history statistics."""
        if not self.signal_history:
            return {}

        stats = {
            "HOLD": sum(1 for a in self.signal_history if a == EntryAction.HOLD),
            "CALL": sum(1 for a in self.signal_history if self._is_call_action(a)),
            "PUT": sum(1 for a in self.signal_history if self._is_put_action(a)),
        }
        return stats

    def __repr__(self) -> str:
        return (
            f"ContinuousSignalExitPolicy("
            f"counter_confirm={self.counter_signal_confirmation}, "
            f"plateau_window={self.plateau_window}, "
            f"plateau_ratio={self.plateau_ratio}, "
            f"SL={self.stop_loss_pct}%, "
            f"flip={self.flip_on_counter}"
            f")"
        )


# Keep PolicyNetwork for backward compatibility
class PolicyNetwork(nn.Module):
    """
    Legacy unified policy network (for backward compatibility).

    Use EntryPolicy and ExitPolicy for new implementations.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_actions: int = 4,
        dropout: float = 0.1,
        use_value_head: bool = True,
        use_position_sizing: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        self.use_value_head = use_value_head
        self.use_position_sizing = use_position_sizing

        # Shared feature extractor
        layers = []
        in_dim = embedding_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Action head (policy)
        self.action_head = nn.Linear(hidden_dim, num_actions)

        # Value head (for actor-critic RL)
        if use_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        # Position sizing head (outputs 0-1 fraction)
        if use_position_sizing:
            self.position_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        return_value: bool = True,
        return_position_size: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = self.shared_layers(embeddings)
        action_logits = self.action_head(features)
        action_probs = F.softmax(action_logits, dim=-1)

        output = {
            "action_logits": action_logits,
            "action_probs": action_probs,
        }

        if return_value and self.use_value_head:
            output["value"] = self.value_head(features)

        if return_position_size and self.use_position_sizing:
            output["position_size"] = self.position_head(features)

        return output


def test_policies() -> None:
    """Test entry and exit policies."""
    print("=" * 60)
    print("Testing Dual Policy Architecture (5-Class)")
    print("=" * 60)

    batch_size = 32
    embedding_dim = 512

    embeddings = torch.randn(batch_size, embedding_dim)

    # Test EntryPolicy with 5 classes
    print("\n" + "-" * 60)
    print("Testing EntryPolicy (5-class: HOLD, CALL_ATM, CALL_OTM, PUT_ATM, PUT_OTM)")
    print("-" * 60)

    entry_policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        num_layers=2,
        num_actions=5,  # 5-class output
    )

    entry_params = sum(p.numel() for p in entry_policy.parameters())
    print(f"Entry policy parameters: {entry_params:,}")
    print(f"Number of actions: {entry_policy.num_actions}")

    output = entry_policy(embeddings)
    print(f"Action probs shape: {output['action_probs'].shape}")
    print(f"Action probs sum: {output['action_probs'].sum(dim=-1).mean():.4f}")

    actions, probs = entry_policy.get_action(embeddings)
    print(f"Actions unique: {actions.unique().tolist()}")
    print(f"Action names: {[EntryAction(a.item()).name for a in actions.unique()]}")

    # Test ExitPolicy
    print("\n" + "-" * 60)
    print("Testing ExitPolicy")
    print("-" * 60)

    exit_policy = ExitPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        num_layers=2,
    )

    exit_params = sum(p.numel() for p in exit_policy.parameters())
    print(f"Exit policy parameters: {exit_params:,}")

    # Create dummy position context
    position_context = torch.zeros(batch_size, 4)
    position_context[:, 0] = torch.randint(0, 2, (batch_size,)) * 2 - 1  # +1 or -1
    position_context[:, 1] = torch.randn(batch_size) * 0.1  # unrealized PnL
    position_context[:, 2] = torch.rand(batch_size)  # bars held (0-1)
    position_context[:, 3] = torch.rand(batch_size) * 6.5  # time to close (hours)

    output = exit_policy(embeddings, position_context)
    print(f"Action probs shape: {output['action_probs'].shape}")
    print(f"Value shape: {output['value'].shape}")

    actions, log_probs, values = exit_policy.get_action(embeddings, position_context)
    print(f"Actions unique: {actions.unique().tolist()}")
    print(f"Log probs mean: {log_probs.mean():.4f}")
    print(f"Values mean: {values.mean():.4f}")

    # Test gradient flow
    print("\n" + "-" * 60)
    print("Testing Gradient Flow")
    print("-" * 60)

    exit_policy.zero_grad()
    log_probs_eval, entropy, values = exit_policy.evaluate_actions(
        embeddings, position_context, actions
    )
    loss = -log_probs_eval.mean() - 0.01 * entropy.mean()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in exit_policy.parameters())
    print(f"Exit policy has gradients: {has_grad}")

    print("\n" + "=" * 60)
    print("Dual policy tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_policies()

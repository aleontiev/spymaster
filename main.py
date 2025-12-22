#!/usr/bin/env python3
"""
SpyMaster Real-Time Trading System.

Two modes of operation:

1. **Worker Pool Mode** (recommended for new strategies):
   Uses the new modular architecture with event-driven workers:
   - Syncer: Real-time data sync from ThetaData/Polygon
   - Loader: Cache management and feature computation
   - StrategyRunner: DAG-based model execution

   Usage:
       # Run with config file
       uv run python main.py --worker-pool --config config/worker_pool.yaml

       # Create default config
       uv run python main.py --create-config

2. **Legacy Orchestrator Mode** (original implementation):
   Direct model inference with Alpaca execution.

   Usage:
       # Paper trading (default)
       uv run python main.py --lejepa checkpoints/lejepa_best.pt --policy checkpoints/policy_best.pt

       # Live trading (use with caution!)
       uv run python main.py --lejepa checkpoints/lejepa_best.pt --policy checkpoints/policy_best.pt --live
"""
import argparse
import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import pandas as pd
import torch
from collections import deque

from src.data.processing import MarketPatch
from src.execution.alpaca_client import (
    AlpacaClient,
    OrderSide,
    RiskConfig,
    RiskManager,
    TimeInForce,
    format_option_symbol,
)
from src.model.lejepa import LeJEPA
from src.model.policy import PolicyNetwork, TradingAction
from src.utils.logger import get_logger, get_trade_logger, setup_logging

# Module loggers
logger = get_logger(__name__)
trade_logger = get_trade_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SpyMaster Trading Bot")

    # Mode selection
    parser.add_argument(
        "--worker-pool",
        action="store_true",
        help="Run in worker pool mode (recommended)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/worker_pool.yaml",
        help="Path to worker pool config (for --worker-pool mode)",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default worker pool config file",
    )

    # Model paths (for legacy mode)
    parser.add_argument(
        "--lejepa",
        type=str,
        help="Path to LeJEPA checkpoint (legacy mode)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Path to policy checkpoint (legacy mode)",
    )

    # Trading mode
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading (default: paper)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run mode - no orders submitted",
    )

    # Risk parameters
    parser.add_argument(
        "--max_position_pct",
        type=float,
        default=0.05,
        help="Max position as %% of portfolio",
    )
    parser.add_argument(
        "--max_daily_loss_pct",
        type=float,
        default=0.02,
        help="Max daily loss as %% of portfolio",
    )
    parser.add_argument(
        "--stop_loss_pct",
        type=float,
        default=0.50,
        help="Stop loss percentage",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile models with torch.compile",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    # Timing
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=1.0,
        help="Market data poll interval (seconds)",
    )
    parser.add_argument(
        "--action_cooldown",
        type=int,
        default=5,
        help="Minutes between actions",
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class TradingOrchestrator:
    """
    Real-time trading orchestrator.

    Manages:
    - Market data buffering
    - Model inference
    - Order execution
    - Risk management
    - Position tracking
    """

    def __init__(
        self,
        lejepa: LeJEPA,
        policy: PolicyNetwork,
        client: AlpacaClient,
        risk_manager: RiskManager,
        device: torch.device,
        patch_length: int = 32,
        action_cooldown: int = 5,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize orchestrator.

        Args:
            lejepa: Pre-trained LeJEPA model
            policy: Trained policy network
            client: Alpaca trading client
            risk_manager: Risk manager
            device: Torch device
            patch_length: Input patch length
            action_cooldown: Minutes between actions
            dry_run: If True, don't submit real orders
            verbose: Verbose logging
        """
        self.lejepa = lejepa
        self.policy = policy
        self.client = client
        self.risk_manager = risk_manager
        self.device = device
        self.patch_length = patch_length
        self.action_cooldown = action_cooldown
        self.dry_run = dry_run
        self.verbose = verbose

        self.patcher = MarketPatch(patch_length=patch_length)

        # Data buffer (stores recent bars)
        self.bar_buffer: Deque[dict] = deque(maxlen=patch_length + 10)

        # State tracking
        self.last_action_time: Optional[datetime] = None
        self.current_position: Optional[dict] = None
        self.running = False

        # Kill switch
        self._killed = False

    def kill(self) -> None:
        """Emergency kill switch - close all positions."""
        trade_logger.kill_switch_activated("Manual trigger")
        self._killed = True
        self.running = False

        if not self.dry_run:
            try:
                self.client.cancel_all_orders()
                self.client.close_all_positions()
                logger.info("All orders cancelled and positions closed")
            except Exception as e:
                logger.error("Error during kill", extra={"error": str(e)})

    def add_bar(self, bar: dict) -> None:
        """
        Add a new bar to the buffer.

        Args:
            bar: Bar dict with keys: timestamp, open, high, low, close, volume
        """
        self.bar_buffer.append(bar)

    def buffer_to_dataframe(self) -> Optional[pd.DataFrame]:
        """Convert bar buffer to DataFrame for patch creation."""
        if len(self.bar_buffer) < self.patch_length:
            return None

        df = pd.DataFrame(list(self.bar_buffer))
        df.index = pd.DatetimeIndex(df["timestamp"])
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    @torch.no_grad()
    def get_action(self) -> tuple[TradingAction, float]:
        """
        Get trading action from current market state.

        Returns:
            Tuple of (action, confidence)
        """
        df = self.buffer_to_dataframe()
        if df is None or len(df) < self.patch_length:
            return TradingAction.HOLD, 0.0

        try:
            # Create patch from latest data
            start_idx = len(df) - self.patch_length
            patch = self.patcher.create_patch(df, start_idx)
        except Exception as e:
            logger.debug("Error creating patch", extra={"error": str(e)})
            return TradingAction.HOLD, 0.0

        # Get embedding
        patch_tensor = patch.unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            embedding = self.lejepa.context_encoder(patch_tensor, return_all_tokens=False)

        # Get action
        output = self.policy(embedding.float(), return_value=False)
        probs = output["action_probs"][0]

        action_idx = probs.argmax().item()
        confidence = probs[action_idx].item()

        return TradingAction(action_idx), confidence

    def should_take_action(self) -> bool:
        """Check if enough time has passed since last action."""
        if self.last_action_time is None:
            return True

        elapsed = (datetime.now() - self.last_action_time).total_seconds() / 60
        return elapsed >= self.action_cooldown

    async def execute_action(
        self,
        action: TradingAction,
        underlying_price: float,
    ) -> None:
        """
        Execute a trading action.

        Args:
            action: Trading action
            underlying_price: Current underlying price
        """
        now = datetime.now()

        # Check risk limits
        if not self.dry_run:
            try:
                account = self.client.get_account()
                positions = self.client.get_positions()
            except Exception as e:
                logger.error("Error getting account info", extra={"error": str(e)})
                return

            account_value = account["equity"]
            can_trade, reason = self.risk_manager.can_open_position(
                account_value, len(positions), now
            )

            if not can_trade:
                trade_logger.risk_limit_hit(
                    "position_check", len(positions), self.risk_manager.config.max_positions,
                    action.name
                )
                logger.debug("Trade blocked", extra={"reason": reason})
                return
        else:
            account_value = 100_000  # Dummy for dry run

        # Determine strike (ATM)
        strike = round(underlying_price)

        # Format option symbol (0DTE = expires today)
        expiry = datetime.combine(now.date(), time(16, 0))

        if action == TradingAction.BUY_CALL:
            option_symbol = format_option_symbol("SPY", expiry, strike, "C")
            option_type = "CALL"
        elif action == TradingAction.BUY_PUT:
            option_symbol = format_option_symbol("SPY", expiry, strike, "P")
            option_type = "PUT"
        elif action == TradingAction.CLOSE_POSITION:
            if self.current_position and not self.dry_run:
                try:
                    self.client.close_position(self.current_position["symbol"])
                    logger.info("Position closed", extra={"symbol": self.current_position["symbol"]})
                    self.current_position = None
                except Exception as e:
                    logger.error("Error closing position", extra={"error": str(e)})
            return
        else:  # HOLD
            return

        # Estimate option price (rough approximation)
        option_price_estimate = underlying_price * 0.005  # ~0.5% of underlying

        # Calculate position size
        qty = self.risk_manager.calculate_position_size(
            account_value, option_price_estimate
        )

        # Calculate limit price (slightly above ask estimate)
        limit_price = round(option_price_estimate * 1.02, 2)

        if self.dry_run:
            logger.info(
                f"DRY RUN: BUY {qty} {option_type} @ strike {strike}",
                extra={
                    "event": "dry_run_order",
                    "symbol": option_symbol,
                    "qty": qty,
                    "option_type": option_type,
                    "strike": strike,
                    "limit_price": limit_price,
                }
            )
        else:
            try:
                order = self.client.submit_option_order(
                    symbol=option_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    limit_price=limit_price,
                    time_in_force=TimeInForce.DAY,
                )
                trade_logger.order_submitted(
                    option_symbol, "buy", qty, limit_price, order.get("order_id")
                )
                self.current_position = {
                    "symbol": option_symbol,
                    "type": option_type,
                    "strike": strike,
                    "qty": qty,
                    "entry_price": limit_price,
                }
            except Exception as e:
                logger.error("Error submitting order", extra={"error": str(e)})

        self.last_action_time = now

    async def check_stop_loss(self) -> None:
        """Check if current position should be stopped out."""
        if self.current_position is None:
            return

        if self.dry_run:
            return

        try:
            positions = self.client.get_positions()
            for pos in positions:
                if pos["symbol"] == self.current_position["symbol"]:
                    entry = self.current_position["entry_price"]
                    current = pos["current_price"]

                    if self.risk_manager.should_stop_loss(entry, current):
                        loss_pct = (entry - current) / entry * 100
                        trade_logger.stop_loss_triggered(
                            self.current_position["symbol"], entry, current, loss_pct
                        )
                        await self.execute_action(TradingAction.CLOSE_POSITION, 0)
                    elif self.risk_manager.should_take_profit(entry, current):
                        profit_pct = (current - entry) / entry * 100
                        trade_logger.take_profit_triggered(
                            self.current_position["symbol"], entry, current, profit_pct
                        )
                        await self.execute_action(TradingAction.CLOSE_POSITION, 0)
                    break
        except Exception as e:
            logger.debug("Error checking stop loss", extra={"error": str(e)})

    async def run(self, poll_interval: float = 1.0) -> None:
        """
        Main trading loop.

        Args:
            poll_interval: Seconds between market data polls
        """
        logger.info(
            "SpyMaster Trading Bot Started",
            extra={
                "event": "bot_started",
                "mode": "dry_run" if self.dry_run else "live",
                "device": str(self.device),
                "poll_interval": poll_interval,
                "action_cooldown": self.action_cooldown,
            }
        )

        self.running = True

        while self.running and not self._killed:
            try:
                now = datetime.now()

                # Check market hours (9:30 AM - 4:00 PM ET)
                market_open = time(9, 30)
                market_close = time(16, 0)

                if not (market_open <= now.time() <= market_close):
                    logger.debug("Market closed, waiting")
                    await asyncio.sleep(60)
                    continue

                # Check if too close to close (last 30 minutes)
                minutes_to_close = (
                    datetime.combine(now.date(), market_close) - now
                ).total_seconds() / 60

                if minutes_to_close < 30:
                    logger.info(
                        "Near market close, closing positions",
                        extra={"minutes_to_close": minutes_to_close}
                    )

                    if self.current_position and not self.dry_run:
                        await self.execute_action(TradingAction.CLOSE_POSITION, 0)

                    await asyncio.sleep(60)
                    continue

                # Get current market data (in production, use WebSocket)
                # For now, we simulate with synthetic data
                if self.dry_run:
                    # Simulate a bar
                    if len(self.bar_buffer) == 0:
                        price = 470.0
                    else:
                        last_price = self.bar_buffer[-1]["close"]
                        price = last_price * (1 + np.random.randn() * 0.0003)

                    bar = {
                        "timestamp": now,
                        "open": price,
                        "high": price * 1.001,
                        "low": price * 0.999,
                        "close": price,
                        "volume": np.random.randint(100000, 500000),
                    }
                    self.add_bar(bar)
                else:
                    # In production: fetch real bars from Polygon/Alpaca
                    pass

                # Check stop loss
                await self.check_stop_loss()

                # Get action if cooldown elapsed
                if self.should_take_action() and len(self.bar_buffer) >= self.patch_length:
                    action, confidence = self.get_action()

                    logger.debug(
                        f"Policy action: {action.name}",
                        extra={"action": action.name, "confidence": confidence}
                    )

                    if action != TradingAction.HOLD and confidence > 0.25:
                        underlying_price = self.bar_buffer[-1]["close"]
                        await self.execute_action(action, underlying_price)

                await asyncio.sleep(poll_interval)

            except KeyboardInterrupt:
                logger.info("Shutdown requested via keyboard interrupt")
                break
            except Exception as e:
                logger.error("Error in main loop", extra={"error": str(e)})
                await asyncio.sleep(poll_interval)

        # Cleanup
        if not self._killed and self.current_position and not self.dry_run:
            logger.info("Closing all positions before shutdown")
            await self.execute_action(TradingAction.CLOSE_POSITION, 0)

        logger.info("Trading bot stopped", extra={"event": "bot_stopped"})


def load_models(
    lejepa_path: str,
    policy_path: str,
    device: torch.device,
    compile_models: bool = False,
) -> tuple[LeJEPA, PolicyNetwork]:
    """
    Load pre-trained models.

    Args:
        lejepa_path: Path to LeJEPA checkpoint
        policy_path: Path to policy checkpoint
        device: Torch device
        compile_models: Whether to compile with torch.compile

    Returns:
        Tuple of (lejepa, policy)
    """
    # Load LeJEPA
    logger.info("Loading LeJEPA model", extra={"path": lejepa_path})
    lejepa, _ = LeJEPA.load_checkpoint(lejepa_path, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()

    # Freeze
    for param in lejepa.parameters():
        param.requires_grad = False

    # Load policy
    logger.info("Loading policy model", extra={"path": policy_path})
    policy_ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    policy_config = policy_ckpt.get("config", {})

    policy = PolicyNetwork(
        embedding_dim=policy_config.get("embedding_dim", lejepa.embedding_dim),
        hidden_dim=policy_config.get("hidden_dim", 256),
        num_actions=policy_config.get("num_actions", 4),
        use_value_head=False,
    )
    policy.load_state_dict(policy_ckpt["policy_state_dict"])
    policy = policy.to(device)
    policy.eval()

    # Compile if requested
    if compile_models:
        logger.info("Compiling models with torch.compile")
        lejepa.compile_model(mode="reduce-overhead")
        policy = torch.compile(policy, mode="reduce-overhead")

    return lejepa, policy


async def main_worker_pool(args: argparse.Namespace) -> None:
    """Run in worker pool mode."""
    from src.workers import WorkerPool, load_config, create_default_config

    logger.info("SpyMaster Worker Pool Mode")
    logger.info(f"Loading config from: {args.config}")

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Config file not found: {args.config}")
        logger.info("Run with --create-config to create a default config file")
        return

    pool = WorkerPool(config)

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal", extra={"signal": sig})
        pool.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await pool.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        pool.stop()


def main_legacy(args: argparse.Namespace) -> None:
    """Run in legacy orchestrator mode."""
    if not args.lejepa or not args.policy:
        logger.error("Legacy mode requires --lejepa and --policy arguments")
        return

    logger.info(
        "SpyMaster - 0DTE Options Trading Bot (Legacy Mode)",
        extra={
            "event": "startup",
            "time": datetime.now().isoformat(),
            "live_mode": args.live,
        }
    )

    if not args.live:
        logger.info("Running in PAPER TRADING mode")
    else:
        logger.warning("LIVE TRADING MODE - REAL MONEY")
        response = input("Are you sure? Type 'YES' to continue: ")
        if response != "YES":
            logger.info("Startup aborted by user")
            return

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load models
    lejepa, policy = load_models(
        args.lejepa,
        args.policy,
        device,
        compile_models=args.compile,
    )

    # Initialize client
    client = AlpacaClient(paper=not args.live)

    # Initialize risk manager
    risk_config = RiskConfig(
        max_position_size_pct=args.max_position_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
        stop_loss_pct=args.stop_loss_pct,
    )
    risk_manager = RiskManager(risk_config)

    # Create orchestrator
    orchestrator = TradingOrchestrator(
        lejepa=lejepa,
        policy=policy,
        client=client,
        risk_manager=risk_manager,
        device=device,
        action_cooldown=args.action_cooldown,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal", extra={"signal": sig})
        orchestrator.running = False

    def kill_handler(sig, frame):
        orchestrator.kill()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # SIGUSR1 for kill switch (Linux only)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, kill_handler)

    # Run
    asyncio.run(orchestrator.run(poll_interval=args.poll_interval))


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging first
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(
        log_dir=args.log_dir,
        level=log_level,
        console_output=True,
        json_output=True,
    )

    # Handle --create-config
    if args.create_config:
        from src.workers import create_default_config
        create_default_config(args.config)
        logger.info(f"Created default config at: {args.config}")
        return

    # Choose mode
    if args.worker_pool:
        asyncio.run(main_worker_pool(args))
    else:
        main_legacy(args)


if __name__ == "__main__":
    main()

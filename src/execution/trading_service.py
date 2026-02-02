"""
Trading service singleton for managing the paper trader lifecycle.

Runs the LiveTrader in a background thread with its own asyncio event loop,
providing thread-safe state access for the Flask web UI.
"""
import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import torch

from src.data.providers.alpaca import AlpacaProvider
from src.db.database import get_db
from src.execution.alpaca_client import AlpacaClient
from src.execution.heuristic import HeuristicConfig
from src.execution.live_trader import LiveTrader, LiveTraderConfig
from src.execution.order_manager import OrderConfig
from src.execution.signal_logger import register_signals_dataset
from src.strategy.exit_rules import ExitConfig
from src.strategy.fusion_config import DEFAULT_MODEL_CONFIGS, FusionConfig
from src.strategy.multi_percentile_executor import MultiPercentileExecutor
from src.strategy.position_sizing import PositionSizeConfig


logger = logging.getLogger(__name__)


class TradingService:
    """
    Singleton managing the paper trader lifecycle within the Flask process.

    Provides start/stop control and thread-safe snapshot reads for the
    live dashboard. The trader runs in a background thread with its own
    asyncio event loop.
    """

    _instance: Optional["TradingService"] = None

    def __init__(self) -> None:
        self._trader: Optional[LiveTrader] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._running: bool = False
        self._error: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._config: Dict[str, Any] = {}
        self.et_tz = ZoneInfo("America/New_York")

    @classmethod
    def instance(cls) -> "TradingService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_running(self) -> bool:
        """Whether the trader is currently running."""
        return self._running

    @property
    def error(self) -> Optional[str]:
        """Last error message, if any."""
        return self._error

    @property
    def started_at(self) -> Optional[datetime]:
        """When the trader was started."""
        return self._started_at

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration used to start the trader."""
        return self._config

    def start(
        self,
        capital: float = 25000.0,
        session_id: str = "paper_trading",
        use_heuristic: bool = True,
        dry_run: bool = False,
    ) -> None:
        """
        Initialize components and launch the trader in a background thread.

        Args:
            capital: Initial trading capital
            session_id: Session ID for state persistence
            use_heuristic: Whether to enable heuristic filtering
            dry_run: If True, log signals but don't submit orders

        Raises:
            RuntimeError: If trader is already running
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Trader is already running")

            self._error = None
            self._config = {
                "capital": capital,
                "session_id": session_id,
                "use_heuristic": use_heuristic,
                "dry_run": dry_run,
            }

            logger.info("=" * 60)
            logger.info("Starting Paper Trading Service")
            logger.info("=" * 60)
            logger.info(f"Capital: ${capital:,.2f}")
            logger.info(f"Session ID: {session_id}")
            logger.info(f"Heuristic: {'Enabled' if use_heuristic else 'Disabled'}")
            logger.info(f"Dry run: {dry_run}")

            try:
                self._init_trader(capital, session_id, use_heuristic, dry_run)
            except Exception as e:
                self._error = f"Failed to initialize trader: {e}"
                logger.error(self._error, exc_info=True)
                raise RuntimeError(self._error) from e

            # Launch background thread
            self._thread = threading.Thread(
                target=self._run_loop,
                name="TradingService",
                daemon=True,
            )
            self._thread.start()
            self._running = True
            self._started_at = datetime.now(self.et_tz)

            logger.info("Trading service started in background thread")

    def _init_trader(
        self,
        capital: float,
        session_id: str,
        use_heuristic: bool,
        dry_run: bool,
    ) -> None:
        """Initialize all trader components."""
        # Register signals dataset
        register_signals_dataset()

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load models
        logger.info("Loading model checkpoints...")
        executor = MultiPercentileExecutor.from_configs(
            DEFAULT_MODEL_CONFIGS,
            FusionConfig(),
            device=device,
        )
        logger.info("Models loaded successfully")

        # Create Alpaca client and provider (always paper mode)
        alpaca_client = AlpacaClient(paper=True)
        alpaca_provider = AlpacaProvider()

        # Check account status
        try:
            account = alpaca_client.get_account()
            logger.info(f"Account equity: ${float(account['equity']):,.2f}")
            logger.info(f"Account status: {account['status']}")

            if account.get("trading_blocked"):
                raise RuntimeError("Trading is blocked on this account")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Alpaca: {e}") from e

        # Create configs
        trader_config = LiveTraderConfig(
            initial_capital=capital,
            session_id=session_id,
            use_heuristic=use_heuristic,
        )

        exit_config = ExitConfig(
            stop_loss_pct=10.0,
            breakeven_activation_pct=7.5,
            breach_breakeven_activation_pct=10.0,
            trailing_stop_distance_pct=5.0,
            runner_trailing_stop_pct=10.0,
        )

        position_size_config = PositionSizeConfig(
            base_position_pct=0.03,
            breach_position_pct=0.03,
            max_position_pct=0.05,
        )

        order_config = OrderConfig(
            max_slippage_pct=1.0,
            order_timeout_seconds=60.0,
            use_ioc=True,
        )

        heuristic_config = HeuristicConfig() if use_heuristic else None

        # Create the trader
        self._trader = LiveTrader(
            executor=executor,
            alpaca_client=alpaca_client,
            alpaca_provider=alpaca_provider,
            config=trader_config,
            exit_config=exit_config,
            position_size_config=position_size_config,
            order_config=order_config,
            heuristic_config=heuristic_config,
            device=device,
        )

    def _run_loop(self) -> None:
        """Run the asyncio event loop in the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._task = self._loop.create_task(self._trader.run())
            self._loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            logger.info("Trading task cancelled")
        except Exception as e:
            self._error = f"Trading loop error: {e}"
            logger.error(self._error, exc_info=True)
        finally:
            # Cleanup
            try:
                # Cancel remaining tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()
            self._loop = None
            self._task = None
            self._running = False
            logger.info("Trading service background thread exited")

    def stop(self) -> None:
        """
        Stop the trader gracefully.

        Signals the trader to stop, cancels the asyncio task, and joins
        the background thread.
        """
        with self._lock:
            if not self._running:
                return

            logger.info("Stopping trading service...")

            # Signal the trader to stop
            if self._trader is not None:
                self._trader.request_stop()

            # Cancel the asyncio task
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)

            # Wait for thread to finish
            if self._thread is not None:
                self._thread.join(timeout=10.0)
                if self._thread.is_alive():
                    logger.warning("Trading thread did not stop cleanly")
                self._thread = None

            # Stop price monitor
            if self._trader is not None:
                self._trader.position_manager.stop_price_monitor()

            self._running = False
            self._started_at = None
            logger.info("Trading service stopped")

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Build a snapshot dict from current trader state.

        Thread-safe: reads simple scalar fields (atomic on CPython) and
        uses position_manager's lock for position data.

        Returns:
            Dict with trader state for the live dashboard.
        """
        now = datetime.now(self.et_tz)

        result: Dict[str, Any] = {
            "is_running": self._running,
            "error": self._error,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "config": self._config,
            "timestamp": now.isoformat(),
        }

        if self._trader is None or not self._running:
            return result

        trader = self._trader

        # Market state
        minutes_elapsed = trader._get_minutes_elapsed(now)
        is_market_hours = trader._is_market_hours(now)

        # Price data
        latest_price = trader.data_buffer.get_latest_price()
        latest_bar = trader.data_buffer.get_latest_bar()
        bar_dict = None
        if latest_bar is not None:
            bar_dict = {
                "timestamp": str(latest_bar.timestamp),
                "open": latest_bar.open,
                "high": latest_bar.high,
                "low": latest_bar.low,
                "close": latest_bar.close,
                "volume": latest_bar.volume,
            }

        # VWAP bands
        vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = trader.data_buffer.get_vwap_bands()

        # Opening range
        or_high, or_low = trader.data_buffer.get_opening_range()

        # Signal state
        signal = {
            "action": trader._last_signal_action,
            "source": trader._last_signal_source,
            "confidence": trader._last_signal_confidence,
            "agreeing_models": list(trader._last_signal_agreeing_models),
        }

        # Pending state
        pending = {
            "has_pending_entry": trader.pending_entry is not None,
            "pending_entry": None,
            "has_pending_breach": trader.pending_breach is not None,
            "pending_breach_type": None,
        }
        if trader.pending_entry is not None:
            pe = trader.pending_entry
            pending["pending_entry"] = {
                "position_type": pe.position_type.value,
                "dominant_model": pe.dominant_model,
                "confidence": pe.confidence,
                "is_breach": pe.is_breach,
            }
        if trader.pending_breach is not None:
            pending["pending_breach_type"] = str(trader.pending_breach.breach_type.value)

        # Position data (thread-safe via position_manager lock)
        position = None
        active_pos = trader.position_manager.get_active_position()
        if active_pos is not None:
            hold_minutes = int((now - active_pos.entry_time).total_seconds() / 60)
            position = {
                **active_pos.to_dict(),
                "hold_minutes": hold_minutes,
            }

        # Statistics
        stats = {
            "breach_immediate_entries": trader.breach_immediate_entries,
            "breach_continuation_entries": trader.breach_continuation_entries,
            "vwap_direction_rejections": trader.vwap_direction_rejections,
            "timeframe_rejections": trader.timeframe_rejections,
            "bars_in_buffer": len(trader.data_buffer),
        }

        result.update({
            "minutes_elapsed": minutes_elapsed,
            "is_market_hours": is_market_hours,
            "current_price": latest_price,
            "latest_bar": bar_dict,
            "vwap": vwap,
            "vwap_upper_1sd": upper_1sd,
            "vwap_lower_1sd": lower_1sd,
            "vwap_upper_2sd": upper_2sd,
            "vwap_lower_2sd": lower_2sd,
            "or_high": or_high,
            "or_low": or_low,
            "signal": signal,
            "pending": pending,
            "position": position,
            "stats": stats,
            "session_id": self._config.get("session_id", "paper_trading"),
        })

        return result

    def get_candles(self) -> List[Dict]:
        """Get candle data for the chart from the data buffer."""
        if self._trader is None or not self._running:
            return []
        return self._trader.data_buffer.get_all_candles_for_chart()

    def get_chart_data(self) -> Dict[str, Any]:
        """Get full chart data for TrainingChartView from the data buffer."""
        if self._trader is None or not self._running:
            return {}
        return self._trader.data_buffer.get_full_chart_data()

    def get_trades(self) -> Dict[str, Any]:
        """
        Get trade data from the database.

        Returns:
            Dict with trades, daily stats, and capital.
        """
        session_id = self._config.get("session_id", "paper_trading")
        db = get_db()
        state = db.get_position_state(session_id=session_id)

        if state is None:
            return {
                "trades": [],
                "daily_trades": 0,
                "daily_pnl": 0.0,
                "capital": self._config.get("capital", 25000.0),
            }

        trades = state.completed_trades_json or []

        # Calculate win rate
        winning = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
        win_rate = (winning / len(trades) * 100) if trades else 0.0

        return {
            "trades": trades,
            "daily_trades": state.daily_trades or 0,
            "daily_pnl": state.daily_pnl or 0.0,
            "capital": state.capital or 25000.0,
            "win_rate": win_rate,
        }

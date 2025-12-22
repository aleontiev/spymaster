"""
Structured JSON Logging for SpyMaster.

Provides:
- JSON-formatted log output for machine parsing
- Log rotation by size and time
- Separate handlers for different log levels
- Trade-specific logging with structured fields
- Console output with human-readable format

Usage:
    from src.utils.logger import get_logger, setup_logging

    # Setup logging (call once at startup)
    setup_logging(log_dir="logs", level="INFO")

    # Get logger for a module
    logger = get_logger(__name__)
    logger.info("Starting bot", extra={"mode": "paper", "device": "cuda"})

    # Trade logging with structured fields
    logger.info("Order submitted", extra={
        "event": "order_submitted",
        "symbol": "SPY240115C00470000",
        "side": "buy",
        "qty": 5,
        "limit_price": 2.50,
    })
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:45.123456Z",
        "level": "INFO",
        "logger": "src.execution.alpaca_client",
        "message": "Order submitted",
        "event": "order_submitted",
        "symbol": "SPY240115C00470000",
        ...
    }
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_traceback: bool = True,
    ) -> None:
        """
        Initialize JSON formatter.

        Args:
            include_timestamp: Include ISO timestamp
            include_level: Include log level
            include_logger: Include logger name
            include_traceback: Include exception traceback
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {}

        # Core fields
        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        # Message
        log_data["message"] = record.getMessage()

        # Extra fields from record
        # Skip standard LogRecord attributes
        skip_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }

        for key, value in record.__dict__.items():
            if key not in skip_attrs and not key.startswith("_"):
                # Handle non-serializable values
                try:
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        # Exception info
        if record.exc_info and self.include_traceback:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter with colors.

    Format: [LEVEL] timestamp - logger - message (extra_key=extra_value, ...)
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """
        Initialize console formatter.

        Args:
            use_colors: Use ANSI colors in output
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console."""
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Level with optional color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"

        # Logger name (shortened)
        logger_name = record.name
        if logger_name.startswith("src."):
            logger_name = logger_name[4:]  # Remove 'src.' prefix

        # Message
        message = record.getMessage()

        # Extra fields
        skip_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }

        extras = []
        for key, value in record.__dict__.items():
            if key not in skip_attrs and not key.startswith("_"):
                extras.append(f"{key}={value}")

        # Build output
        output = f"[{level}] {timestamp} - {logger_name} - {message}"
        if extras:
            output += f" ({', '.join(extras)})"

        # Exception info
        if record.exc_info:
            output += "\n" + self.formatException(record.exc_info)

        return output


class TradeLogger:
    """
    Specialized logger for trade events.

    Provides structured logging methods for common trade events:
    - Order submission
    - Order fills
    - Position changes
    - P&L updates
    - Risk events
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize trade logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger

    def order_submitted(
        self,
        symbol: str,
        side: str,
        qty: int,
        limit_price: float,
        order_id: Optional[str] = None,
    ) -> None:
        """Log order submission."""
        self.logger.info(
            f"Order submitted: {side.upper()} {qty} {symbol} @ ${limit_price:.2f}",
            extra={
                "event": "order_submitted",
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "limit_price": limit_price,
                "order_id": order_id,
            },
        )

    def order_filled(
        self,
        symbol: str,
        side: str,
        qty: int,
        fill_price: float,
        order_id: Optional[str] = None,
    ) -> None:
        """Log order fill."""
        self.logger.info(
            f"Order filled: {side.upper()} {qty} {symbol} @ ${fill_price:.2f}",
            extra={
                "event": "order_filled",
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "fill_price": fill_price,
                "order_id": order_id,
            },
        )

    def order_cancelled(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log order cancellation."""
        self.logger.info(
            f"Order cancelled: {symbol}" + (f" - {reason}" if reason else ""),
            extra={
                "event": "order_cancelled",
                "symbol": symbol,
                "order_id": order_id,
                "reason": reason,
            },
        )

    def position_opened(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        position_type: str,
    ) -> None:
        """Log position opening."""
        self.logger.info(
            f"Position opened: {qty} {symbol} ({position_type}) @ ${entry_price:.2f}",
            extra={
                "event": "position_opened",
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "position_type": position_type,
            },
        )

    def position_closed(
        self,
        symbol: str,
        qty: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
    ) -> None:
        """Log position closing."""
        self.logger.info(
            f"Position closed: {symbol} - P&L: ${pnl:.2f} ({pnl_pct:.1f}%)",
            extra={
                "event": "position_closed",
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            },
        )

    def stop_loss_triggered(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        loss_pct: float,
    ) -> None:
        """Log stop loss trigger."""
        self.logger.warning(
            f"Stop loss triggered: {symbol} - Loss: {loss_pct:.1f}%",
            extra={
                "event": "stop_loss_triggered",
                "symbol": symbol,
                "entry_price": entry_price,
                "current_price": current_price,
                "loss_pct": loss_pct,
            },
        )

    def take_profit_triggered(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        profit_pct: float,
    ) -> None:
        """Log take profit trigger."""
        self.logger.info(
            f"Take profit triggered: {symbol} - Profit: {profit_pct:.1f}%",
            extra={
                "event": "take_profit_triggered",
                "symbol": symbol,
                "entry_price": entry_price,
                "current_price": current_price,
                "profit_pct": profit_pct,
            },
        )

    def risk_limit_hit(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        action_blocked: str,
    ) -> None:
        """Log risk limit breach."""
        self.logger.warning(
            f"Risk limit hit: {limit_type} - {current_value:.2f} vs limit {limit_value:.2f}",
            extra={
                "event": "risk_limit_hit",
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value,
                "action_blocked": action_blocked,
            },
        )

    def kill_switch_activated(self, reason: str) -> None:
        """Log kill switch activation."""
        self.logger.critical(
            f"KILL SWITCH ACTIVATED: {reason}",
            extra={
                "event": "kill_switch_activated",
                "reason": reason,
            },
        )

    def daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        max_drawdown: float,
    ) -> None:
        """Log daily trading summary."""
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        self.logger.info(
            f"Daily summary: {total_trades} trades, {win_rate:.1f}% win rate, "
            f"P&L: ${total_pnl:.2f}, Max DD: ${max_drawdown:.2f}",
            extra={
                "event": "daily_summary",
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "max_drawdown": max_drawdown,
            },
        )


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 10,
    console_output: bool = True,
    json_output: bool = True,
    use_colors: bool = True,
) -> None:
    """
    Setup logging configuration.

    Creates:
    - Console handler (human-readable format)
    - JSON file handler (structured logs with rotation)
    - Error file handler (errors only, for quick debugging)

    Args:
        log_dir: Directory for log files
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        console_output: Enable console logging
        json_output: Enable JSON file logging
        use_colors: Use colors in console output
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(ConsoleFormatter(use_colors=use_colors))
        root_logger.addHandler(console_handler)

    # JSON file handler (all logs)
    if json_output:
        json_handler = RotatingFileHandler(
            log_path / "spymaster.json.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)

        # Error file handler (errors only)
        error_handler = RotatingFileHandler(
            log_path / "spymaster.error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)

        # Trade-specific handler (INFO and above, trade events only)
        trade_handler = RotatingFileHandler(
            log_path / "spymaster.trades.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(JSONFormatter())
        trade_handler.addFilter(TradeEventFilter())
        root_logger.addHandler(trade_handler)


class TradeEventFilter(logging.Filter):
    """Filter that only allows trade-related events."""

    TRADE_EVENTS = {
        "order_submitted",
        "order_filled",
        "order_cancelled",
        "position_opened",
        "position_closed",
        "stop_loss_triggered",
        "take_profit_triggered",
        "risk_limit_hit",
        "kill_switch_activated",
        "daily_summary",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        """Only allow trade events."""
        event = getattr(record, "event", None)
        return event in self.TRADE_EVENTS


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_trade_logger(name: str) -> TradeLogger:
    """
    Get a trade logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        TradeLogger instance
    """
    return TradeLogger(logging.getLogger(name))


# Convenience function for quick setup
def configure_simple_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """
    Simple logging configuration for scripts and testing.

    Args:
        level: Log level
        log_file: Optional log file path
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def test_logging() -> None:
    """Test logging configuration."""
    import tempfile

    print("=" * 60)
    print("Testing Logging Configuration")
    print("=" * 60)

    # Setup logging to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        setup_logging(log_dir=tmpdir, level="DEBUG")

        # Get loggers
        logger = get_logger("test.module")
        trade_logger = get_trade_logger("test.trading")

        # Test basic logging
        print("\n--- Basic Logging ---")
        logger.debug("Debug message", extra={"debug_key": "debug_value"})
        logger.info("Info message", extra={"mode": "paper"})
        logger.warning("Warning message")
        logger.error("Error message", extra={"error_code": 500})

        # Test trade logging
        print("\n--- Trade Logging ---")
        trade_logger.order_submitted("SPY240115C00470000", "buy", 5, 2.50, "order123")
        trade_logger.order_filled("SPY240115C00470000", "buy", 5, 2.48, "order123")
        trade_logger.position_opened("SPY240115C00470000", 5, 2.48, "CALL")
        trade_logger.stop_loss_triggered("SPY240115C00470000", 2.48, 1.24, 50.0)
        trade_logger.position_closed("SPY240115C00470000", 5, 2.48, 1.24, -620.0, -50.0)
        trade_logger.daily_summary(10, 6, 450.0, 300.0)

        # Test exception logging
        print("\n--- Exception Logging ---")
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Caught an exception")

        # Show log files created
        print(f"\n--- Log Files in {tmpdir} ---")
        for f in Path(tmpdir).iterdir():
            print(f"  {f.name}: {f.stat().st_size} bytes")

        # Show sample JSON log content
        json_log = Path(tmpdir) / "spymaster.json.log"
        if json_log.exists():
            print("\n--- Sample JSON Log Content ---")
            with open(json_log) as f:
                lines = f.readlines()
                for line in lines[:3]:  # Show first 3 lines
                    parsed = json.loads(line)
                    print(json.dumps(parsed, indent=2))

    print("\n" + "=" * 60)
    print("Logging tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_logging()

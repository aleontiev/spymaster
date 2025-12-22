"""Tests for the structured logging module."""
import json
import logging
import tempfile
from pathlib import Path

import pytest

from src.utils.logger import (
    ConsoleFormatter,
    JSONFormatter,
    TradeEventFilter,
    TradeLogger,
    get_logger,
    get_trade_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self) -> None:
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.module"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_extra_fields(self) -> None:
        """Test that extra fields are included."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.symbol = "SPY"
        record.qty = 5

        result = formatter.format(record)
        data = json.loads(result)

        assert data["symbol"] == "SPY"
        assert data["qty"] == 5

    def test_exception_formatting(self) -> None:
        """Test exception info is included."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.module",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_basic_format(self) -> None:
        """Test basic console formatting."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test.module" in result
        assert "Test message" in result

    def test_extra_fields_in_console(self) -> None:
        """Test extra fields appear in console output."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.symbol = "SPY"

        result = formatter.format(record)

        assert "symbol=SPY" in result


class TestTradeEventFilter:
    """Tests for TradeEventFilter."""

    def test_allows_trade_events(self) -> None:
        """Test that trade events pass through."""
        filter_obj = TradeEventFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Order submitted",
            args=(),
            exc_info=None,
        )
        record.event = "order_submitted"

        assert filter_obj.filter(record) is True

    def test_blocks_non_trade_events(self) -> None:
        """Test that non-trade events are blocked."""
        filter_obj = TradeEventFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Starting bot",
            args=(),
            exc_info=None,
        )
        # No event attribute or wrong event
        record.event = "bot_started"

        assert filter_obj.filter(record) is False

    def test_all_trade_events_allowed(self) -> None:
        """Test all defined trade events are allowed."""
        filter_obj = TradeEventFilter()

        for event in TradeEventFilter.TRADE_EVENTS:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg="Test",
                args=(),
                exc_info=None,
            )
            record.event = event
            assert filter_obj.filter(record) is True


class TestTradeLogger:
    """Tests for TradeLogger."""

    def test_order_submitted(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test order_submitted logging."""
        with caplog.at_level(logging.INFO):
            base_logger = logging.getLogger("test.trade")
            trade_logger = TradeLogger(base_logger)

            trade_logger.order_submitted(
                symbol="SPY240115C00470000",
                side="buy",
                qty=5,
                limit_price=2.50,
                order_id="order123",
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.event == "order_submitted"
        assert record.symbol == "SPY240115C00470000"
        assert record.side == "buy"
        assert record.qty == 5
        assert record.limit_price == 2.50

    def test_stop_loss_triggered(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test stop_loss_triggered logging."""
        with caplog.at_level(logging.WARNING):
            base_logger = logging.getLogger("test.trade")
            trade_logger = TradeLogger(base_logger)

            trade_logger.stop_loss_triggered(
                symbol="SPY240115C00470000",
                entry_price=2.50,
                current_price=1.25,
                loss_pct=50.0,
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "WARNING"
        assert record.event == "stop_loss_triggered"
        assert record.loss_pct == 50.0

    def test_kill_switch_activated(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test kill_switch_activated logging."""
        with caplog.at_level(logging.CRITICAL):
            base_logger = logging.getLogger("test.trade")
            trade_logger = TradeLogger(base_logger)

            trade_logger.kill_switch_activated("Manual trigger")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "CRITICAL"
        assert record.event == "kill_switch_activated"
        assert record.reason == "Manual trigger"


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_log_directory(self) -> None:
        """Test that log directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "logs"
            setup_logging(log_dir=str(log_dir), console_output=False)

            assert log_dir.exists()

    def test_creates_log_files(self) -> None:
        """Test that log files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, console_output=False)

            logger = get_logger("test")
            logger.info("Test message")
            logger.error("Error message")

            # Force handlers to flush
            for handler in logging.getLogger().handlers:
                handler.flush()

            log_path = Path(tmpdir)
            assert (log_path / "spymaster.json.log").exists()
            assert (log_path / "spymaster.error.log").exists()

    def test_json_log_format(self) -> None:
        """Test that JSON logs are valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir, console_output=False)

            logger = get_logger("test.json")
            logger.info("Test message", extra={"key": "value"})

            # Force flush
            for handler in logging.getLogger().handlers:
                handler.flush()

            json_log = Path(tmpdir) / "spymaster.json.log"
            with open(json_log) as f:
                for line in f:
                    data = json.loads(line)
                    assert "timestamp" in data
                    assert "level" in data
                    assert "message" in data


class TestGetLogger:
    """Tests for get_logger and get_trade_logger functions."""

    def test_get_logger_returns_logger(self) -> None:
        """Test get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_trade_logger_returns_trade_logger(self) -> None:
        """Test get_trade_logger returns a TradeLogger instance."""
        trade_logger = get_trade_logger("test.trading")
        assert isinstance(trade_logger, TradeLogger)

"""Tests for the EventBus."""
import asyncio
import pytest
from datetime import datetime
from src.workers.event_bus import EventBus, Event, get_event_bus, reset_event_bus


@pytest.fixture
def event_bus():
    """Create a fresh event bus for each test."""
    return EventBus()


@pytest.fixture(autouse=True)
def reset_global_bus():
    """Reset the global event bus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.mark.asyncio
async def test_subscribe_and_emit(event_bus):
    """Test basic subscribe and emit functionality."""
    received_events = []

    async def handler(event: Event):
        received_events.append(event)

    event_bus.subscribe("test_event", handler)

    # Emit sync to wait for processing
    await event_bus.emit_sync("test_event", {"key": "value"})

    assert len(received_events) == 1
    assert received_events[0].name == "test_event"
    assert received_events[0].data == {"key": "value"}
    assert isinstance(received_events[0].timestamp, datetime)


@pytest.mark.asyncio
async def test_multiple_subscribers(event_bus):
    """Test multiple subscribers receive the same event."""
    received_1 = []
    received_2 = []

    async def handler1(event: Event):
        received_1.append(event)

    async def handler2(event: Event):
        received_2.append(event)

    event_bus.subscribe("test_event", handler1)
    event_bus.subscribe("test_event", handler2)

    await event_bus.emit_sync("test_event", "data")

    assert len(received_1) == 1
    assert len(received_2) == 1
    assert received_1[0].data == "data"
    assert received_2[0].data == "data"


@pytest.mark.asyncio
async def test_unsubscribe(event_bus):
    """Test unsubscribing from events."""
    received = []

    async def handler(event: Event):
        received.append(event)

    event_bus.subscribe("test_event", handler)
    await event_bus.emit_sync("test_event", "first")
    assert len(received) == 1

    event_bus.unsubscribe("test_event", handler)
    await event_bus.emit_sync("test_event", "second")
    assert len(received) == 1  # Should not receive second event


@pytest.mark.asyncio
async def test_no_subscribers(event_bus):
    """Test emitting event with no subscribers doesn't error."""
    await event_bus.emit_sync("no_subscribers", "data")


@pytest.mark.asyncio
async def test_event_bus_run_and_stop():
    """Test the run loop and stop functionality."""
    bus = EventBus()
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe("test", handler)

    # Start the bus
    task = asyncio.create_task(bus.run())

    # Give it time to start
    await asyncio.sleep(0.1)
    assert bus.is_running

    # Emit via queue
    await bus.emit("test", "data1")
    await asyncio.sleep(0.1)
    assert len(received) == 1

    # Stop the bus
    bus.stop()
    await task

    assert not bus.is_running


@pytest.mark.asyncio
async def test_handler_error_doesnt_break_bus(event_bus):
    """Test that handler errors don't break other handlers."""
    received = []

    async def bad_handler(event: Event):
        raise ValueError("Test error")

    async def good_handler(event: Event):
        received.append(event)

    event_bus.subscribe("test", bad_handler)
    event_bus.subscribe("test", good_handler)

    # Should not raise, and good_handler should still receive event
    await event_bus.emit_sync("test", "data")
    assert len(received) == 1


@pytest.mark.asyncio
async def test_subscriber_count(event_bus):
    """Test subscriber counting."""
    async def handler(event: Event):
        pass

    assert event_bus.get_subscriber_count("test") == 0

    event_bus.subscribe("test", handler)
    assert event_bus.get_subscriber_count("test") == 1

    event_bus.subscribe("test", handler)  # Same handler again
    assert event_bus.get_subscriber_count("test") == 2

    event_bus.unsubscribe("test", handler)
    assert event_bus.get_subscriber_count("test") == 1


def test_global_event_bus():
    """Test global event bus singleton."""
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2

    reset_event_bus()
    bus3 = get_event_bus()
    assert bus3 is not bus1


@pytest.mark.asyncio
async def test_emit_different_events(event_bus):
    """Test that different events go to correct subscribers."""
    event_a_received = []
    event_b_received = []

    async def handler_a(event: Event):
        event_a_received.append(event)

    async def handler_b(event: Event):
        event_b_received.append(event)

    event_bus.subscribe("event_a", handler_a)
    event_bus.subscribe("event_b", handler_b)

    await event_bus.emit_sync("event_a", "data_a")
    await event_bus.emit_sync("event_b", "data_b")

    assert len(event_a_received) == 1
    assert len(event_b_received) == 1
    assert event_a_received[0].data == "data_a"
    assert event_b_received[0].data == "data_b"

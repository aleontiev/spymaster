"""
Simple async event bus for worker coordination.

Events:
- "minute_synced": Emitted by Syncer when minute data is ready
- "cache_updated": Emitted by Loader when cache is updated
- "action_decided": Emitted by StrategyRunner with decisions
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Container for an event."""

    name: str
    data: Any
    timestamp: datetime


class EventBus:
    """
    Simple async event bus for worker coordination.

    Supports both sync and async handlers. Events are processed in order
    via an internal queue.

    Usage:
        bus = EventBus()

        # Subscribe to events
        async def on_minute_synced(event: Event):
            print(f"Minute synced: {event.data}")

        bus.subscribe("minute_synced", on_minute_synced)

        # Start the event loop
        asyncio.create_task(bus.run())

        # Emit events
        await bus.emit("minute_synced", {"minute": datetime.now()})
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = {}
        self._queue: asyncio.Queue[Optional[Event]] = asyncio.Queue()
        self._running = False

    def subscribe(
        self,
        event_name: str,
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        """
        Subscribe to an event.

        Args:
            event_name: Name of the event to subscribe to
            handler: Async function to call when event is emitted
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)
        logger.debug(f"Subscribed to '{event_name}': {handler.__name__}")

    def unsubscribe(
        self,
        event_name: str,
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        """
        Unsubscribe from an event.

        Args:
            event_name: Name of the event
            handler: Handler to remove
        """
        if event_name in self._subscribers:
            try:
                self._subscribers[event_name].remove(handler)
                logger.debug(f"Unsubscribed from '{event_name}': {handler.__name__}")
            except ValueError:
                pass

    async def emit(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event.

        The event is added to the queue and processed asynchronously.

        Args:
            event_name: Name of the event to emit
            data: Event data to pass to handlers
        """
        event = Event(
            name=event_name,
            data=data,
            timestamp=datetime.utcnow(),
        )
        await self._queue.put(event)
        logger.debug(f"Emitted '{event_name}'")

    async def emit_sync(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event and wait for all handlers to complete.

        Unlike emit(), this waits for handlers to finish before returning.

        Args:
            event_name: Name of the event to emit
            data: Event data to pass to handlers
        """
        event = Event(
            name=event_name,
            data=data,
            timestamp=datetime.utcnow(),
        )
        await self._process_event(event)

    async def _process_event(self, event: Event) -> None:
        """Process a single event by calling all subscribers."""
        handlers = self._subscribers.get(event.name, [])
        if not handlers:
            logger.debug(f"No handlers for '{event.name}'")
            return

        # Run all handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                handler_name = handlers[i].__name__
                logger.error(
                    f"Error in handler '{handler_name}' for '{event.name}': {result}"
                )

    async def run(self) -> None:
        """
        Run the event bus, processing events from the queue.

        This should be started as a task and will run until stop() is called.
        """
        logger.info("Event bus started")
        self._running = True

        while self._running:
            try:
                # Wait for next event with timeout to allow checking _running
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if event is None:
                    # Poison pill to stop
                    break

                await self._process_event(event)
                self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("Event bus cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event bus: {e}")

        logger.info("Event bus stopped")

    def stop(self) -> None:
        """Stop the event bus."""
        self._running = False
        # Put poison pill to unblock the queue
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    @property
    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._running

    def get_subscriber_count(self, event_name: str) -> int:
        """Get the number of subscribers for an event."""
        return len(self._subscribers.get(event_name, []))


# Convenience function to create a shared event bus
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (useful for testing)."""
    global _global_bus
    if _global_bus is not None:
        _global_bus.stop()
    _global_bus = None

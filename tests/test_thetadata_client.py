"""
Unit tests for ThetaData client for historical open interest data.

These tests use mocking to avoid requiring an actual ThetaData terminal.
For integration tests with a live terminal, run with --run-integration flag.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd
import json

from src.data.thetadata_client import (
    ThetaDataClient,
    ThetaDataError,
    ThetaDataConnectionError,
    ThetaDataAPIError,
    save_historical_oi_to_parquet,
)


class TestThetaDataClientInit:
    """Test ThetaDataClient initialization."""

    def test_default_initialization(self) -> None:
        """Test client initializes with default values."""
        client = ThetaDataClient()
        assert client.base_url == "http://localhost:25503/v3"
        assert client.timeout == 60.0

    def test_custom_initialization(self) -> None:
        """Test client with custom parameters."""
        client = ThetaDataClient(
            base_url="http://localhost:8080/v3",
            timeout=120.0,
        )
        assert client.base_url == "http://localhost:8080/v3"
        assert client.timeout == 120.0

    def test_env_var_override(self, monkeypatch) -> None:
        """Test environment variable configuration."""
        monkeypatch.setenv("THETADATA_BASE_URL", "http://custom:9999/v3")
        monkeypatch.setenv("THETADATA_TIMEOUT", "90")

        client = ThetaDataClient()
        assert client.base_url == "http://custom:9999/v3"
        assert client.timeout == 90.0


class TestThetaDataClientRequests:
    """Test ThetaDataClient request handling."""

    @pytest.fixture
    def client(self) -> ThetaDataClient:
        """Create a test client."""
        return ThetaDataClient()

    @pytest.fixture
    def mock_csv_response(self) -> str:
        """Sample CSV response from ThetaData API."""
        return """date,root,expiration,strike,right,open_interest
2024-01-02,SPY,2024-01-05,470000,C,15234
2024-01-02,SPY,2024-01-05,470000,P,12456
2024-01-02,SPY,2024-01-05,475000,C,18923
2024-01-02,SPY,2024-01-05,475000,P,14567
2024-01-03,SPY,2024-01-05,470000,C,16000
2024-01-03,SPY,2024-01-05,470000,P,13000"""

    @pytest.mark.asyncio
    async def test_check_terminal_running_success(self, client: ThetaDataClient) -> None:
        """Test terminal check when running."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await client._check_terminal_running()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_terminal_not_running(self, client: ThetaDataClient) -> None:
        """Test terminal check when not running."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            result = await client._check_terminal_running()
            assert result is False

    @pytest.mark.asyncio
    async def test_request_csv_parsing(
        self, client: ThetaDataClient, mock_csv_response: str
    ) -> None:
        """Test CSV response parsing."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = mock_csv_response

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            df = await client._request("option/history/open_interest", {"root": "SPY"})

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 6
            assert "date" in df.columns
            assert "open_interest" in df.columns

    @pytest.mark.asyncio
    async def test_request_connection_error(self, client: ThetaDataClient) -> None:
        """Test connection error handling."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(ThetaDataConnectionError) as exc_info:
                await client._request("option/history/open_interest", {"root": "SPY"})

            assert "Cannot connect to ThetaData terminal" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_timeout_error(self, client: ThetaDataClient) -> None:
        """Test timeout error handling."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(ThetaDataAPIError) as exc_info:
                await client._request("option/history/open_interest", {"root": "SPY"})

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_api_error(self, client: ThetaDataClient) -> None:
        """Test API error response handling."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request: invalid parameters"

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(ThetaDataAPIError) as exc_info:
                await client._request("option/history/open_interest", {"root": "SPY"})

            assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_503_terminal_not_ready(self, client: ThetaDataClient) -> None:
        """Test 503 response when terminal is authenticating."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.text = "Terminal not ready"

            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_instance

            with pytest.raises(ThetaDataConnectionError) as exc_info:
                await client._request("option/history/open_interest", {"root": "SPY"})

            assert "not ready" in str(exc_info.value)


class TestFetchHistoricalOpenInterest:
    """Test fetch_historical_open_interest method."""

    @pytest.fixture
    def client(self) -> ThetaDataClient:
        """Create a test client."""
        return ThetaDataClient()

    @pytest.fixture
    def mock_oi_response(self) -> str:
        """Sample open interest CSV response."""
        return """date,root,expiration,strike,right,open_interest
2024-01-02,SPY,2024-01-05,470000,C,15234
2024-01-02,SPY,2024-01-05,470000,P,12456
2024-01-02,SPY,2024-01-05,475000,C,18923"""

    @pytest.mark.asyncio
    async def test_fetch_with_default_dates(
        self, client: ThetaDataClient, mock_oi_response: str
    ) -> None:
        """Test fetching with default date range (4 years)."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_oi_response)
            )

            df = await client.fetch_historical_open_interest(root="SPY")

            # Check request was made with correct parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            params = call_args[0][1]

            assert params["root"] == "SPY"
            assert "start_date" in params
            assert "end_date" in params
            assert params["expiration"] == "*"

    @pytest.mark.asyncio
    async def test_fetch_with_date_objects(
        self, client: ThetaDataClient, mock_oi_response: str
    ) -> None:
        """Test fetching with date objects."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_oi_response)
            )

            start = date(2024, 1, 1)
            end = date(2024, 1, 31)

            df = await client.fetch_historical_open_interest(
                root="SPY",
                start_date=start,
                end_date=end,
            )

            params = mock_request.call_args[0][1]
            assert params["start_date"] == "2024-01-01"
            assert params["end_date"] == "2024-01-31"

    @pytest.mark.asyncio
    async def test_fetch_with_right_filter(
        self, client: ThetaDataClient, mock_oi_response: str
    ) -> None:
        """Test fetching with call/put filter."""
        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_oi_response)
            )

            # Test with 'call'
            await client.fetch_historical_open_interest(root="SPY", right="call")
            params = mock_request.call_args[0][1]
            assert params["right"] == "C"

            # Test with 'P'
            await client.fetch_historical_open_interest(root="SPY", right="P")
            params = mock_request.call_args[0][1]
            assert params["right"] == "P"

    @pytest.mark.asyncio
    async def test_fetch_strike_conversion(self, client: ThetaDataClient) -> None:
        """Test that strikes are converted from cents to dollars."""
        # Response with strikes in cents (470000 = $470)
        mock_response = """date,root,expiration,strike,right,open_interest
2024-01-02,SPY,2024-01-05,470000,C,15234
2024-01-02,SPY,2024-01-05,475000,P,12456"""

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_response)
            )

            df = await client.fetch_historical_open_interest(root="SPY")

            # Strikes should be converted to dollars
            assert df["strike"].iloc[0] == 470.0
            assert df["strike"].iloc[1] == 475.0

    @pytest.mark.asyncio
    async def test_fetch_date_parsing(self, client: ThetaDataClient) -> None:
        """Test that dates are parsed correctly."""
        mock_response = """date,root,expiration,strike,right,open_interest
2024-01-02,SPY,2024-01-05,470,C,15234"""

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_response)
            )

            df = await client.fetch_historical_open_interest(root="SPY")

            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert pd.api.types.is_datetime64_any_dtype(df["expiration"])


class TestFetchOpenInterestByDateRange:
    """Test batched fetching for large date ranges."""

    @pytest.fixture
    def client(self) -> ThetaDataClient:
        """Create a test client."""
        return ThetaDataClient()

    @pytest.mark.asyncio
    async def test_batch_fetching(self, client: ThetaDataClient) -> None:
        """Test that large date ranges are batched."""
        mock_response = """date,root,expiration,strike,right,open_interest
2024-01-02,SPY,2024-01-05,470,C,15234"""

        batch_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal batch_count
            batch_count += 1
            return pd.read_csv(__import__("io").StringIO(mock_response))

        with patch.object(
            client, "fetch_historical_open_interest", side_effect=mock_fetch
        ):
            df = await client.fetch_open_interest_by_date_range(
                root="SPY",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 3, 30),  # 90 days (Jan 1 - Mar 30)
                batch_days=30,
            )

            # Should make 3 batches for 90 days with 30-day batches
            assert batch_count == 3

    @pytest.mark.asyncio
    async def test_batch_deduplication(self, client: ThetaDataClient) -> None:
        """Test that duplicate rows at batch boundaries are removed."""
        # Simulate overlapping data at boundary
        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2024-01-15")],
                        "root": ["SPY"],
                        "expiration": [pd.Timestamp("2024-01-19")],
                        "strike": [470.0],
                        "right": ["C"],
                        "open_interest": [1000],
                    }
                )
            else:
                # Duplicate row
                return pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2024-01-15")],
                        "root": ["SPY"],
                        "expiration": [pd.Timestamp("2024-01-19")],
                        "strike": [470.0],
                        "right": ["C"],
                        "open_interest": [1000],
                    }
                )

        with patch.object(
            client, "fetch_historical_open_interest", side_effect=mock_fetch
        ):
            df = await client.fetch_open_interest_by_date_range(
                root="SPY",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 31),
                batch_days=15,
            )

            # Should have removed duplicates
            assert len(df) == 1

    @pytest.mark.asyncio
    async def test_batch_error_handling(self, client: ThetaDataClient) -> None:
        """Test that batch errors don't stop the entire operation."""
        call_count = 0

        async def mock_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ThetaDataAPIError("Temporary error")
            return pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-15")],
                    "root": ["SPY"],
                    "expiration": [pd.Timestamp("2024-01-19")],
                    "strike": [470.0],
                    "right": ["C"],
                    "open_interest": [1000 * call_count],
                }
            )

        with patch.object(
            client, "fetch_historical_open_interest", side_effect=mock_fetch
        ):
            df = await client.fetch_open_interest_by_date_range(
                root="SPY",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 2, 29),
                batch_days=20,
            )

            # Should still have data from successful batches
            assert len(df) == 2  # First and third batches succeeded


class TestFetchExpirations:
    """Test expiration listing."""

    @pytest.fixture
    def client(self) -> ThetaDataClient:
        """Create a test client."""
        return ThetaDataClient()

    @pytest.mark.asyncio
    async def test_fetch_expirations(self, client: ThetaDataClient) -> None:
        """Test fetching available expirations."""
        mock_response = """expiration
2024-01-05
2024-01-12
2024-01-19
2024-01-26"""

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_response)
            )

            expirations = await client.fetch_expirations(root="SPY")

            assert len(expirations) == 4
            assert all(isinstance(exp, date) for exp in expirations)
            assert expirations[0] == date(2024, 1, 5)


class TestFetchStrikes:
    """Test strike listing."""

    @pytest.fixture
    def client(self) -> ThetaDataClient:
        """Create a test client."""
        return ThetaDataClient()

    @pytest.mark.asyncio
    async def test_fetch_strikes(self, client: ThetaDataClient) -> None:
        """Test fetching available strikes."""
        mock_response = """strike
470000
471000
472000
473000
474000
475000"""

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_response)
            )

            strikes = await client.fetch_strikes(root="SPY", expiration="2024-01-05")

            assert len(strikes) == 6
            # Should convert from cents
            assert strikes[0] == 470.0
            assert strikes[-1] == 475.0

    @pytest.mark.asyncio
    async def test_fetch_strikes_requires_expiration(
        self, client: ThetaDataClient
    ) -> None:
        """Test that expiration is required."""
        with pytest.raises(ValueError, match="expiration is required"):
            await client.fetch_strikes(root="SPY")


class TestSaveHistoricalOIToParquet:
    """Test parquet saving utility."""

    @pytest.mark.asyncio
    async def test_save_to_parquet(self, tmp_path) -> None:
        """Test saving data to parquet file."""
        mock_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "root": ["SPY", "SPY"],
                "expiration": pd.to_datetime(["2024-01-05", "2024-01-05"]),
                "strike": [470.0, 470.0],
                "right": ["C", "C"],
                "open_interest": [15234, 16000],
            }
        )

        with patch(
            "src.data.thetadata_client.ThetaDataClient._check_terminal_running",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = True

            with patch(
                "src.data.thetadata_client.ThetaDataClient.fetch_open_interest_by_date_range",
                new_callable=AsyncMock,
            ) as mock_fetch:
                mock_fetch.return_value = mock_df

                output_path = await save_historical_oi_to_parquet(
                    output_dir=tmp_path,
                    root="SPY",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                )

                assert output_path.exists()
                assert output_path.suffix == ".parquet"

                # Verify contents
                saved_df = pd.read_parquet(output_path)
                assert len(saved_df) == 2
                assert "open_interest" in saved_df.columns

    @pytest.mark.asyncio
    async def test_save_raises_on_terminal_not_running(self, tmp_path) -> None:
        """Test error when terminal is not running."""
        with patch(
            "src.data.thetadata_client.ThetaDataClient._check_terminal_running",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = False

            with pytest.raises(ThetaDataConnectionError):
                await save_historical_oi_to_parquet(
                    output_dir=tmp_path,
                    root="SPY",
                )


class TestDataValidation:
    """Test data validation and edge cases."""

    @pytest.fixture
    def client(self) -> ThetaDataClient:
        """Create a test client."""
        return ThetaDataClient()

    @pytest.mark.asyncio
    async def test_empty_response_handling(self, client: ThetaDataClient) -> None:
        """Test handling of empty response."""
        mock_response = """date,root,expiration,strike,right,open_interest"""

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_response)
            )

            df = await client.fetch_historical_open_interest(root="SPY")

            assert df.empty

    @pytest.mark.asyncio
    async def test_column_name_normalization(self, client: ThetaDataClient) -> None:
        """Test that column names are normalized to lowercase."""
        # Response with mixed case columns
        mock_response = """Date,Root,Expiration,Strike,Right,Open_Interest
2024-01-02,SPY,2024-01-05,470,C,15234"""

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = pd.read_csv(
                __import__("io").StringIO(mock_response)
            )

            df = await client.fetch_historical_open_interest(root="SPY")

            # All columns should be lowercase
            assert all(col == col.lower() for col in df.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

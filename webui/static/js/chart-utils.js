/**
 * Shared chart utilities for Spymaster WebUI
 * Used by both /data and /charts pages
 */

const ChartUtils = {
    // Default chart options matching the data page
    getDefaultOptions(showTimeScale = true) {
        return {
            layout: {
                background: { type: 'solid', color: '#0a0a0a' },
                textColor: '#9ca3af',
            },
            grid: {
                vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
                horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: 'rgba(255, 255, 255, 0.2)', width: 1, style: 0 },
                horzLine: { color: 'rgba(255, 255, 255, 0.2)', width: 1, style: 0 },
            },
            leftPriceScale: {
                visible: true,
                borderColor: '#333',
                scaleMargins: { top: 0.1, bottom: 0.1 },
            },
            rightPriceScale: {
                visible: false,
                borderColor: '#333',
            },
            timeScale: {
                visible: showTimeScale,
                borderColor: '#333',
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 5,
                minBarSpacing: 3,
            },
            handleScroll: { vertTouchDrag: false },
        };
    },

    // Calculate chart heights to fill viewport
    calculateChartHeights(headerHeight, visibleChartCount, isZoomed = false) {
        const viewportHeight = window.innerHeight;
        const availableHeight = viewportHeight - headerHeight;

        if (isZoomed) {
            return { main: availableHeight - 20, sub: 0 };
        }

        // Reserve 30px per subchart for labels, plus some padding
        const labelHeight = 24;
        const chartPadding = 4;
        const subchartCount = visibleChartCount - 1; // Exclude main chart
        const totalLabelSpace = subchartCount * labelHeight;
        const totalPadding = visibleChartCount * chartPadding;

        const chartSpace = availableHeight - totalLabelSpace - totalPadding;

        // Main chart gets ~55% of space, rest divided among subcharts
        const mainRatio = 0.55;
        const mainHeight = Math.floor(chartSpace * mainRatio);
        const subHeight = subchartCount > 0 ? Math.floor((chartSpace * (1 - mainRatio)) / subchartCount) : 0;

        return { main: Math.max(mainHeight, 150), sub: Math.max(subHeight, 50) };
    },

    // Update time scale visibility on a chart
    setTimeScaleVisible(chart, visible) {
        if (chart && chart.applyOptions) {
            chart.applyOptions({
                timeScale: { visible }
            });
        }
    },

    // Format large numbers (billions/millions)
    formatLargeNumber(val) {
        const absVal = Math.abs(val);
        if (absVal >= 1e9) return (val / 1e9).toFixed(1) + 'B';
        if (absVal >= 1e6) return (val / 1e6).toFixed(1) + 'M';
        if (absVal >= 1e3) return (val / 1e3).toFixed(0) + 'K';
        return val.toFixed(0);
    },

    // Format with styled span for tooltip
    formatLargeNumberStyled(val) {
        const formatted = this.formatLargeNumber(val);
        const color = val >= 0 ? 'text-green-400' : 'text-red-400';
        return `<span class="${color}">${formatted}</span>`;
    },

    // Create main OHLCV chart
    createMainChart(container, options = {}) {
        const chartOptions = { ...this.getDefaultOptions(), ...options };
        chartOptions.height = options.height || 400;
        chartOptions.width = container.clientWidth;

        const chart = LightweightCharts.createChart(container, chartOptions);

        const candleSeries = chart.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
        });

        return { chart, candleSeries };
    },

    // Create volume chart (histogram)
    createVolumeChart(container, options = {}) {
        const self = this;
        const chartOptions = { ...this.getDefaultOptions(), ...options };
        chartOptions.height = options.height || 100;
        chartOptions.width = container.clientWidth;
        chartOptions.timeScale.visible = false;
        chartOptions.grid = { vertLines: { visible: false }, horzLines: { color: 'rgba(255, 255, 255, 0.05)' } };
        chartOptions.localization = { priceFormatter: (val) => self.formatLargeNumber(val) };

        const chart = LightweightCharts.createChart(container, chartOptions);

        const volumeSeries = chart.addHistogramSeries({
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            priceFormat: { type: 'volume' },
        });

        return { chart, volumeSeries };
    },

    // Create flow chart (call/put volume)
    createFlowChart(container, options = {}) {
        const self = this;
        const chartOptions = { ...this.getDefaultOptions(), ...options };
        chartOptions.height = options.height || 100;
        chartOptions.width = container.clientWidth;
        chartOptions.timeScale.visible = false;
        chartOptions.localization = { priceFormatter: (val) => self.formatLargeNumber(Math.abs(val)) };

        const chart = LightweightCharts.createChart(container, chartOptions);

        const callSeries = chart.addHistogramSeries({
            color: '#26a69a',
            priceScaleId: 'left',
            priceFormat: { type: 'volume' },
            lastValueVisible: false,
            priceLineVisible: false,
        });

        const putSeries = chart.addHistogramSeries({
            color: '#ef5350',
            priceScaleId: 'left',
            priceFormat: { type: 'volume' },
            lastValueVisible: false,
            priceLineVisible: false,
        });

        return { chart, callSeries, putSeries };
    },

    // Create GEX chart (histogram with threshold coloring)
    createGexChart(container, options = {}) {
        const self = this;
        const chartOptions = { ...this.getDefaultOptions(), ...options };
        chartOptions.height = options.height || 80;
        chartOptions.width = container.clientWidth;
        chartOptions.timeScale.visible = false;
        chartOptions.localization = { priceFormatter: (val) => self.formatLargeNumber(val) };

        const chart = LightweightCharts.createChart(container, chartOptions);

        const gexSeries = chart.addLineSeries({
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            lineWidth: 2,
        });

        return { chart, gexSeries };
    },

    // Create IV chart (line with threshold coloring)
    createIvChart(container, options = {}) {
        const chartOptions = { ...this.getDefaultOptions(), ...options };
        chartOptions.height = options.height || 80;
        chartOptions.width = container.clientWidth;
        chartOptions.timeScale.visible = false;
        chartOptions.localization = { priceFormatter: (val) => val.toFixed(1) + '%' };

        const chart = LightweightCharts.createChart(container, chartOptions);

        const ivSeries = chart.addLineSeries({
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            lineWidth: 2,
        });

        return { chart, ivSeries };
    },

    // Create imbalance/pressure chart (histogram)
    createImbalanceChart(container, options = {}) {
        const self = this;
        const chartOptions = { ...this.getDefaultOptions(), ...options };
        chartOptions.height = options.height || 80;
        chartOptions.width = container.clientWidth;
        chartOptions.timeScale.visible = false;
        chartOptions.localization = { priceFormatter: (val) => self.formatLargeNumber(val) };

        const chart = LightweightCharts.createChart(container, chartOptions);

        const imbalanceSeries = chart.addHistogramSeries({
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
        });

        return { chart, imbalanceSeries };
    },

    // Colorize GEX data based on $1B threshold
    colorizeGexData(data) {
        const threshold = 1e9;
        return data.map(d => ({
            time: d.time,
            value: d.value,
            color: d.value >= threshold ? '#22c55e' :
                   d.value >= 0 ? '#facc15' :
                   d.value >= -threshold ? '#f97316' : '#ef4444'
        }));
    },

    // Colorize IV data based on thresholds
    colorizeIvData(data) {
        return data.map(d => ({
            time: d.time,
            value: d.value,
            color: d.value < 20 ? '#22c55e' :
                   d.value < 25 ? '#facc15' :
                   d.value < 30 ? '#f97316' : '#ef4444'
        }));
    },

    // Sync time scales across multiple charts
    syncTimeScales(charts) {
        if (!charts || charts.length < 2) return;

        let isSyncing = false;
        const validCharts = charts.filter(c => c && c.timeScale);

        validCharts.forEach(chart => {
            chart.timeScale().subscribeVisibleTimeRangeChange(range => {
                if (isSyncing || !range) return;
                isSyncing = true;
                validCharts.forEach(otherChart => {
                    if (otherChart !== chart) {
                        try {
                            otherChart.timeScale().setVisibleRange(range);
                        } catch (e) { /* ignore */ }
                    }
                });
                isSyncing = false;
            });
        });
    },

    // Setup crosshair sync across charts
    syncCrosshairs(charts, callback) {
        if (!charts || charts.length === 0) return;

        const validCharts = charts.filter(c => c && c.subscribeCrosshairMove);

        validCharts.forEach(chart => {
            chart.subscribeCrosshairMove(param => {
                if (callback) callback(param);

                // Sync crosshair position to other charts
                validCharts.forEach(otherChart => {
                    if (otherChart !== chart && param.time) {
                        // Note: Lightweight Charts doesn't have a direct way to set crosshair position
                        // The visual sync happens through time scale sync
                    }
                });
            });
        });
    },

    // Format Unix timestamp to ET time string
    formatTimeET(unixTimestamp) {
        const date = new Date(unixTimestamp * 1000);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            timeZone: 'America/New_York',
            hour12: true
        });
    },

    // Build tooltip HTML for training_raw data
    buildTooltipContent(data, time, overlayVisible = {}) {
        let html = '';

        // Time header - display in ET to match x-axis
        const timeStr = this.formatTimeET(time);
        html += `<div class="text-gray-400 text-[10px] mb-2">${timeStr} ET</div>`;

        // OHLCV section - check both 'ohlcv' and 'candles' keys
        const ohlcvData = data.ohlcv || data.candles;
        const ohlcvPoint = ohlcvData?.find(d => d.time === time);
        if (ohlcvPoint) {
            const change = ohlcvPoint.close - ohlcvPoint.open;
            const changeColor = change >= 0 ? 'text-green-400' : 'text-red-400';
            const changePct = ((change / ohlcvPoint.open) * 100).toFixed(2);
            html += `<div class="mb-2 pb-2 border-b border-neutral-600">`;
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">O</span><span class="text-white font-mono">${ohlcvPoint.open.toFixed(2)}</span></div>`;
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">H</span><span class="text-white font-mono">${ohlcvPoint.high.toFixed(2)}</span></div>`;
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">L</span><span class="text-white font-mono">${ohlcvPoint.low.toFixed(2)}</span></div>`;
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">C</span><span class="${changeColor} font-mono">${ohlcvPoint.close.toFixed(2)} (${change >= 0 ? '+' : ''}${changePct}%)</span></div>`;
            html += `</div>`;
        }

        // Volume
        const volPoint = data.volume?.find(d => d.time === time);
        if (volPoint) {
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">Volume</span><span class="text-white font-mono">${this.formatLargeNumber(volPoint.value)}</span></div>`;
        }

        // Imbalance (Stock OFI)
        const ofiPoint = data.stock_ofi?.find(d => d.time === time);
        if (ofiPoint) {
            const ofiColor = ofiPoint.value >= 0 ? 'text-green-400' : 'text-red-400';
            const ofiSign = ofiPoint.value >= 0 ? '+' : '';
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">Imbalance</span><span class="${ofiColor} font-mono">${ofiSign}${this.formatLargeNumber(ofiPoint.value)}</span></div>`;
        }

        // GEX
        const gexPoint = data.net_gex?.find(d => d.time === time);
        if (gexPoint) {
            const gexColor = gexPoint.value >= 1e9 ? 'text-green-400' : gexPoint.value >= 0 ? 'text-yellow-400' : 'text-red-400';
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">GEX</span><span class="${gexColor} font-mono">${this.formatLargeNumber(gexPoint.value)}</span></div>`;
        }

        // IV
        const ivPoint = data.iv?.find(d => d.time === time);
        if (ivPoint) {
            const ivColor = ivPoint.value < 20 ? 'text-green-400' : ivPoint.value < 30 ? 'text-yellow-400' : 'text-red-400';
            html += `<div class="flex justify-between gap-4"><span class="text-gray-400">IV</span><span class="${ivColor} font-mono">${ivPoint.value.toFixed(1)}%</span></div>`;
        }

        // Options flow
        const callVol = data.call_volume?.find(d => d.time === time);
        const putVol = data.put_volume?.find(d => d.time === time);
        if (callVol || putVol) {
            html += `<div class="mt-2 pt-2 border-t border-neutral-600">`;
            if (callVol) html += `<div class="flex justify-between gap-4"><span class="text-green-400">Calls</span><span class="font-mono">${this.formatLargeNumber(callVol.value)}</span></div>`;
            if (putVol) html += `<div class="flex justify-between gap-4"><span class="text-red-400">Puts</span><span class="font-mono">${this.formatLargeNumber(Math.abs(putVol.value))}</span></div>`;
            html += `</div>`;
        }

        return html;
    },

    // Position tooltip near crosshair
    positionTooltip(tooltip, param, containerRect) {
        if (!param.point || !tooltip) return;

        const x = param.point.x;
        const y = param.point.y;

        // Position tooltip to the right of cursor, or left if near edge
        let left = containerRect.left + x + 20;
        let top = containerRect.top + y - tooltip.offsetHeight / 2;

        // Keep tooltip within viewport
        if (left + tooltip.offsetWidth > window.innerWidth - 20) {
            left = containerRect.left + x - tooltip.offsetWidth - 20;
        }
        if (top < 10) top = 10;
        if (top + tooltip.offsetHeight > window.innerHeight - 10) {
            top = window.innerHeight - tooltip.offsetHeight - 10;
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    },

    // Add VWAP overlay to main chart
    addVwapOverlay(chart, data, visible = true) {
        if (!data?.length) return null;

        const series = chart.addLineSeries({
            color: '#eab308',
            lineWidth: 1,
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            visible: visible,
        });
        series.setData(data);
        return series;
    },

    // Add rolling VWAP overlay
    addRollingVwapOverlay(chart, data, visible = true) {
        if (!data?.length) return null;

        const series = chart.addLineSeries({
            color: '#06b6d4',
            lineWidth: 1,
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            visible: visible,
        });
        series.setData(data);
        return series;
    },

    // Add GEX level overlays (zero GEX, walls, etc.)
    addGexLevelOverlay(chart, data, color, lineStyle = 0, visible = true) {
        if (!data?.length) return null;

        const series = chart.addLineSeries({
            color: color,
            lineWidth: 1,
            lineStyle: lineStyle, // 0=solid, 1=dotted, 2=dashed
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            visible: visible,
        });
        series.setData(data);
        return series;
    },

    // Add horizontal line (for premarket/3-day levels)
    addHorizontalLine(chart, value, times, color, lineStyle = 2, visible = true) {
        if (!value || !times?.length) return null;

        const series = chart.addLineSeries({
            color: color,
            lineWidth: 1,
            lineStyle: lineStyle,
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            visible: visible,
        });
        series.setData([
            { time: times[0], value: value },
            { time: times[times.length - 1], value: value }
        ]);
        return series;
    },

    // Cleanup charts
    destroyCharts(charts) {
        if (!charts) return;
        Object.values(charts).forEach(chart => {
            if (chart && chart.remove) {
                try {
                    chart.remove();
                } catch (e) { /* ignore */ }
            }
        });
    }
};

// Export for use in templates
if (typeof window !== 'undefined') {
    window.ChartUtils = ChartUtils;
}

/**
 * Shared Training Chart View Component
 * Used by both /data extended view and /charts page
 */

const TrainingChartView = {
    // Chart instances
    charts: {},
    chartSeries: {},
    chartData: null,
    chartZoomed: false,

    // Overlay visibility state - default to 0V, +VW, -VW only
    overlayVisible: {
        vwap: false,
        rollingVwap: false,
        vwapUpper1: false,
        vwapLower1: false,
        vwapUpper2: false,
        vwapLower2: false,
        zeroGex: false,
        zeroDex: false,
        positiveGws: false,
        negativeGws: false,
        deltaSupport: false,
        deltaResistance: false,
        gammaCallWall: false,
        gammaPutWall: false,
        dexCallWall: false,
        dexPutWall: false,
        mostActiveStrike: true,
        callWeightedStrike: true,
        putWeightedStrike: true,
        maxCallStrike: false,
        maxPutStrike: false,
        pointOfControl: false,
        premarketHigh: false,
        premarketLow: false,
        threeDayHigh: false,
        threeDayLow: false,
    },

    // Legend groups for batch toggling
    legendGroups: {
        price: ['vwap', 'rollingVwap', 'vwapUpper1', 'vwapLower1', 'vwapUpper2', 'vwapLower2', 'pointOfControl'],
        gamma: ['zeroGex', 'positiveGws', 'negativeGws', 'gammaCallWall', 'gammaPutWall'],
        delta: ['zeroDex', 'deltaSupport', 'deltaResistance', 'dexCallWall', 'dexPutWall'],
        volume: ['mostActiveStrike', 'callWeightedStrike', 'putWeightedStrike', 'maxCallStrike', 'maxPutStrike'],
        prior: ['premarketHigh', 'premarketLow', 'threeDayHigh', 'threeDayLow'],
    },

    // Configuration
    config: {
        containerIds: {
            main: 'main-chart',
            volume: 'volume-chart',
            imbalance: 'market-pressure-chart',
            flow: 'flow-chart',
            velocity: 'velocity-chart',
            netGex: 'net-gex-chart',
            iv: 'iv-chart',
            totalGex: 'total-gex-chart',
        },
        // Order of charts from bottom to top (for x-axis visibility)
        chartOrder: ['totalGex', 'iv', 'velocity', 'netGex', 'imbalance', 'flow', 'volume', 'main'],
    },

    // Base chart options
    getChartOptions(showTimeScale = false) {
        return {
            layout: {
                background: { type: 'solid', color: '#0a0a0a' },
                textColor: '#9ca3af',
            },
            grid: {
                vertLines: { color: '#1f1f1f' },
                horzLines: { color: '#1f1f1f' },
            },
            crosshair: {
                mode: 1,  // Magnet mode
                vertLine: {
                    color: '#6366f1',
                    width: 1,
                    style: 0,
                    visible: true,
                    labelVisible: showTimeScale,  // Only show label on bottom chart
                },
                horzLine: {
                    color: '#6366f180',
                    width: 1,
                    style: 2,
                    visible: true,
                    labelVisible: true,
                },
            },
            timeScale: {
                visible: showTimeScale,
                borderColor: '#333',
                timeVisible: true,
                secondsVisible: false,
                fixLeftEdge: true,
                fixRightEdge: true,
            },
            leftPriceScale: {
                visible: true,
                borderColor: '#333',
                minimumWidth: 70,
            },
            rightPriceScale: {
                visible: false,
            },
        };
    },

    // Initialize the chart view
    init(customConfig = {}) {
        // Merge custom config
        if (customConfig.containerIds) {
            this.config.containerIds = { ...this.config.containerIds, ...customConfig.containerIds };
        }
        if (customConfig.overlayVisible) {
            this.overlayVisible = { ...this.overlayVisible, ...customConfig.overlayVisible };
        }
    },

    // Destroy all charts
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.remove) {
                try { chart.remove(); } catch (e) { /* ignore */ }
            }
        });
        this.charts = {};
        this.chartSeries = {};
        this.chartData = null;
        this.chartZoomed = false;

        // Clear all container innerHTML to allow flex layout to recalculate
        Object.values(this.config.containerIds).forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = '';
            }
        });
    },

    // Get container height (for flex layouts)
    getContainerHeight(container) {
        return container?.clientHeight || 100;
    },

    // Render all charts with data
    render(data) {
        if (!data || !window.LightweightCharts) return;

        this.destroy();
        this.chartData = data;

        const chartOptions = this.getChartOptions(false);
        const self = this;

        // Helper to get container
        const getContainer = (key) => document.getElementById(this.config.containerIds[key]);

        // 1. Main OHLCV Chart
        const mainContainer = getContainer('main');
        if (mainContainer && (data.ohlcv?.length > 0 || data.price?.length > 0)) {
            mainContainer.innerHTML = '';
            const mainChart = LightweightCharts.createChart(mainContainer, {
                ...chartOptions,
                height: this.getContainerHeight(mainContainer),
                width: mainContainer.clientWidth,
            });
            this.charts.main = mainChart;

            if (data.ohlcv?.length > 0) {
                const candleSeries = mainChart.addCandlestickSeries({
                    upColor: '#26a69a',
                    downColor: '#ef5350',
                    borderVisible: false,
                    wickUpColor: '#26a69a',
                    wickDownColor: '#ef5350',
                    priceScaleId: 'left',
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
                candleSeries.setData(data.ohlcv);
                this.chartSeries.ohlcv = candleSeries;

                // Add overlays
                this.addMainChartOverlays(mainChart, data);
            }

            mainChart.timeScale().fitContent();
        }

        // Get visible range for syncing subcharts
        const mainVisibleRange = this.charts.main?.timeScale().getVisibleRange();

        // 2. Volume Chart
        const volumeContainer = getContainer('volume');
        if (volumeContainer && data.volume?.length > 0) {
            volumeContainer.innerHTML = '';
            const volumeChart = LightweightCharts.createChart(volumeContainer, {
                ...chartOptions,
                height: this.getContainerHeight(volumeContainer),
                width: volumeContainer.clientWidth,
            });
            this.charts.volume = volumeChart;

            const volumeSeries = volumeChart.addHistogramSeries({
                priceFormat: { type: 'volume' },
                priceScaleId: 'left',
                lastValueVisible: false,
                priceLineVisible: false,
            });
            volumeSeries.setData(data.volume);
            this.chartSeries.volume = volumeSeries;

            if (mainVisibleRange) volumeChart.timeScale().setVisibleRange(mainVisibleRange);
            else volumeChart.timeScale().fitContent();
        }

        // 3. Imbalance Chart (Stock OFI)
        const imbalanceContainer = getContainer('imbalance');
        if (imbalanceContainer && data.stock_ofi?.length > 0) {
            imbalanceContainer.innerHTML = '';
            const imbalanceChart = LightweightCharts.createChart(imbalanceContainer, {
                ...chartOptions,
                height: this.getContainerHeight(imbalanceContainer),
                width: imbalanceContainer.clientWidth,
                localization: { priceFormatter: (val) => ChartUtils.formatLargeNumber(val) },
            });
            this.charts.imbalance = imbalanceChart;

            const imbalanceSeries = imbalanceChart.addHistogramSeries({
                priceScaleId: 'left',
                lastValueVisible: false,
                priceLineVisible: false,
            });
            imbalanceSeries.setData(data.stock_ofi.map(d => ({
                time: d.time,
                value: d.value,
                color: d.value >= 0 ? '#22c55e' : '#ef4444'
            })));
            this.chartSeries.imbalance = imbalanceSeries;

            if (mainVisibleRange) imbalanceChart.timeScale().setVisibleRange(mainVisibleRange);
            else imbalanceChart.timeScale().fitContent();
        }

        // 4. Options Flow Chart
        const flowContainer = getContainer('flow');
        if (flowContainer && (data.call_volume?.length > 0 || data.net_gamma_flow?.length > 0)) {
            flowContainer.innerHTML = '';
            const flowChart = LightweightCharts.createChart(flowContainer, {
                ...chartOptions,
                height: this.getContainerHeight(flowContainer),
                width: flowContainer.clientWidth,
                localization: { priceFormatter: (val) => ChartUtils.formatLargeNumber(Math.abs(val)) },
            });
            this.charts.flow = flowChart;

            if (data.call_volume?.length > 0) {
                const callSeries = flowChart.addHistogramSeries({
                    color: '#26a69a',
                    priceScaleId: 'left',
                    priceFormat: { type: 'volume' },
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                callSeries.setData(data.call_volume);
                this.chartSeries.flowCall = callSeries;

                const putSeries = flowChart.addHistogramSeries({
                    color: '#ef5350',
                    priceScaleId: 'left',
                    priceFormat: { type: 'volume' },
                    lastValueVisible: false,
                    priceLineVisible: false,
                });
                if (data.put_volume?.length > 0) putSeries.setData(data.put_volume);
                this.chartSeries.flowPut = putSeries;
            }

            if (mainVisibleRange) flowChart.timeScale().setVisibleRange(mainVisibleRange);
            else flowChart.timeScale().fitContent();
        }

        // 5. Velocity/Pressure Chart
        const velocityContainer = getContainer('velocity');
        if (velocityContainer && data.market_velocity?.length > 0) {
            velocityContainer.innerHTML = '';
            const velocityChart = LightweightCharts.createChart(velocityContainer, {
                ...chartOptions,
                height: this.getContainerHeight(velocityContainer),
                width: velocityContainer.clientWidth,
            });
            this.charts.velocity = velocityChart;

            const velocitySeries = velocityChart.addHistogramSeries({
                priceFormat: { type: 'volume' },
                priceScaleId: 'left',
                priceLineVisible: false,
                lastValueVisible: false,
            });

            // Build color lookup for discrepancy detection
            const callVolByTime = new Map();
            const putVolByTime = new Map();
            const ohlcByTime = new Map();
            if (data.call_volume) data.call_volume.forEach(d => callVolByTime.set(d.time, Math.abs(d.value)));
            if (data.put_volume) data.put_volume.forEach(d => putVolByTime.set(d.time, Math.abs(d.value)));
            if (data.ohlcv) data.ohlcv.forEach(d => ohlcByTime.set(d.time, d));

            const coloredData = data.market_velocity.map(d => {
                const callVol = callVolByTime.get(d.time) || 0;
                const putVol = putVolByTime.get(d.time) || 0;
                const ohlc = ohlcByTime.get(d.time);
                const priceDown = ohlc ? ohlc.close < ohlc.open : false;
                const priceUp = ohlc ? ohlc.close > ohlc.open : false;

                let color;
                if (d.value >= 0) {
                    color = (priceDown && putVol > callVol) ? '#fbbf24' : '#22c55e';
                } else {
                    color = (priceUp && callVol > putVol) ? '#a855f7' : '#ef4444';
                }
                return { time: d.time, value: d.value, color };
            });
            velocitySeries.setData(coloredData);
            this.chartSeries.velocity = velocitySeries;

            if (mainVisibleRange) velocityChart.timeScale().setVisibleRange(mainVisibleRange);
            else velocityChart.timeScale().fitContent();
        }

        // 6. Net GEX Chart
        const netGexContainer = getContainer('netGex');
        if (netGexContainer && data.net_gex?.length > 0) {
            netGexContainer.innerHTML = '';
            const netGexChart = LightweightCharts.createChart(netGexContainer, {
                ...chartOptions,
                height: this.getContainerHeight(netGexContainer),
                width: netGexContainer.clientWidth,
                localization: { priceFormatter: (val) => ChartUtils.formatLargeNumber(val) },
            });
            this.charts.netGex = netGexChart;

            const netGexSeries = netGexChart.addLineSeries({
                priceScaleId: 'left',
                lastValueVisible: false,
                priceLineVisible: false,
                lineWidth: 2,
            });
            netGexSeries.setData(ChartUtils.colorizeGexData(data.net_gex));
            this.chartSeries.netGex = netGexSeries;

            if (mainVisibleRange) netGexChart.timeScale().setVisibleRange(mainVisibleRange);
            else netGexChart.timeScale().fitContent();
        }

        // 7. IV Chart
        const ivContainer = getContainer('iv');
        if (ivContainer && data.iv?.length > 0) {
            ivContainer.innerHTML = '';
            const ivChart = LightweightCharts.createChart(ivContainer, {
                ...chartOptions,
                height: this.getContainerHeight(ivContainer),
                width: ivContainer.clientWidth,
            });
            this.charts.iv = ivChart;

            const ivSeries = ivChart.addLineSeries({
                priceScaleId: 'left',
                lastValueVisible: false,
                priceLineVisible: false,
                lineWidth: 2,
            });
            ivSeries.setData(ChartUtils.colorizeIvData(data.iv));
            this.chartSeries.iv = ivSeries;

            if (mainVisibleRange) ivChart.timeScale().setVisibleRange(mainVisibleRange);
            else ivChart.timeScale().fitContent();
        }

        // 8. Total GEX / Regime Strength Chart
        const totalGexContainer = getContainer('totalGex');
        const gexData = data.gex_regime_strength || data.total_gex;
        if (totalGexContainer && gexData?.length > 0) {
            totalGexContainer.innerHTML = '';
            const totalGexChart = LightweightCharts.createChart(totalGexContainer, {
                ...chartOptions,
                height: this.getContainerHeight(totalGexContainer),
                width: totalGexContainer.clientWidth,
                localization: { priceFormatter: (val) => ChartUtils.formatLargeNumber(val) },
            });
            this.charts.totalGex = totalGexChart;

            const totalGexSeries = totalGexChart.addHistogramSeries({
                priceScaleId: 'left',
                base: 0,
                lastValueVisible: false,
                priceLineVisible: false,
            });
            totalGexSeries.setData(gexData);
            this.chartSeries.totalGex = totalGexSeries;

            if (mainVisibleRange) totalGexChart.timeScale().setVisibleRange(mainVisibleRange);
            else totalGexChart.timeScale().fitContent();
        }

        // Sync all time scales
        const allCharts = Object.values(this.charts).filter(c => c);
        ChartUtils.syncTimeScales(allCharts);

        // Enable x-axis only on the bottom-most visible chart
        this.updateXAxisVisibility();

        // Setup crosshair sync and tooltip
        this.setupCrosshairSync();
    },

    // Add overlays to main chart
    addMainChartOverlays(chart, data) {
        // VWAP
        if (data.vwap?.length > 0) {
            this.chartSeries.vwap = this.addLineSeries(chart, data.vwap, '#eab308', 1, 0, this.overlayVisible.vwap);
        }

        // Rolling VWAP
        if (data.rolling_vwap?.length > 0) {
            this.chartSeries.rollingVwap = this.addLineSeries(chart, data.rolling_vwap, '#06b6d4', 1, 0, this.overlayVisible.rollingVwap);
        }

        // VWAP bands
        if (data.vwap_upper_1?.length > 0) {
            this.chartSeries.vwapUpper1 = this.addLineSeries(chart, data.vwap_upper_1, '#a3e635', 1, 1, this.overlayVisible.vwapUpper1);
        }
        if (data.vwap_lower_1?.length > 0) {
            this.chartSeries.vwapLower1 = this.addLineSeries(chart, data.vwap_lower_1, '#a3e635', 1, 1, this.overlayVisible.vwapLower1);
        }
        if (data.vwap_upper_2?.length > 0) {
            this.chartSeries.vwapUpper2 = this.addLineSeries(chart, data.vwap_upper_2, '#fb923c', 1, 1, this.overlayVisible.vwapUpper2);
        }
        if (data.vwap_lower_2?.length > 0) {
            this.chartSeries.vwapLower2 = this.addLineSeries(chart, data.vwap_lower_2, '#fb923c', 1, 1, this.overlayVisible.vwapLower2);
        }

        // GEX levels
        if (data.zero_gex_level?.length > 0) {
            this.chartSeries.zeroGex = this.addLineSeries(chart, data.zero_gex_level, '#9ca3af', 1, 2, this.overlayVisible.zeroGex);
        }
        if (data.zero_dex_level?.length > 0) {
            this.chartSeries.zeroDex = this.addLineSeries(chart, data.zero_dex_level, '#ffffff', 1, 2, this.overlayVisible.zeroDex);
        }
        if (data.positive_gws?.length > 0) {
            this.chartSeries.positiveGws = this.addLineSeries(chart, data.positive_gws, '#4ade80', 2, 0, this.overlayVisible.positiveGws);
        }
        if (data.negative_gws?.length > 0) {
            this.chartSeries.negativeGws = this.addLineSeries(chart, data.negative_gws, '#f87171', 2, 0, this.overlayVisible.negativeGws);
        }

        // Delta levels
        if (data.negative_dws?.length > 0) {
            this.chartSeries.deltaSupport = this.addLineSeries(chart, data.negative_dws, '#a3e635', 2, 0, this.overlayVisible.deltaSupport);
        }
        if (data.positive_dws?.length > 0) {
            this.chartSeries.deltaResistance = this.addLineSeries(chart, data.positive_dws, '#f472b6', 2, 0, this.overlayVisible.deltaResistance);
        }

        // Wall levels
        if (data.gamma_call_wall?.length > 0) {
            this.chartSeries.gammaCallWall = this.addLineSeries(chart, data.gamma_call_wall, '#86efac', 2, 1, this.overlayVisible.gammaCallWall);
        }
        if (data.gamma_put_wall?.length > 0) {
            this.chartSeries.gammaPutWall = this.addLineSeries(chart, data.gamma_put_wall, '#fca5a5', 2, 1, this.overlayVisible.gammaPutWall);
        }
        if (data.dex_call_wall?.length > 0) {
            this.chartSeries.dexCallWall = this.addLineSeries(chart, data.dex_call_wall, '#bef264', 1, 1, this.overlayVisible.dexCallWall);
        }
        if (data.dex_put_wall?.length > 0) {
            this.chartSeries.dexPutWall = this.addLineSeries(chart, data.dex_put_wall, '#f9a8d4', 1, 1, this.overlayVisible.dexPutWall);
        }

        // Volume levels
        if (data.most_active_strike?.length > 0) {
            this.chartSeries.mostActiveStrike = this.addLineSeries(chart, data.most_active_strike, '#3b82f6', 2, 0, this.overlayVisible.mostActiveStrike);
        }
        if (data.call_weighted_strike?.length > 0) {
            this.chartSeries.callWeightedStrike = this.addLineSeries(chart, data.call_weighted_strike, '#22c55e', 2, 2, this.overlayVisible.callWeightedStrike);
        }
        if (data.put_weighted_strike?.length > 0) {
            this.chartSeries.putWeightedStrike = this.addLineSeries(chart, data.put_weighted_strike, '#ef4444', 2, 2, this.overlayVisible.putWeightedStrike);
        }
        if (data.max_call_strike?.length > 0) {
            this.chartSeries.maxCallStrike = this.addLineSeries(chart, data.max_call_strike, '#22c55e', 1, 2, this.overlayVisible.maxCallStrike);
        }
        if (data.max_put_strike?.length > 0) {
            this.chartSeries.maxPutStrike = this.addLineSeries(chart, data.max_put_strike, '#ef4444', 1, 2, this.overlayVisible.maxPutStrike);
        }

        // POC
        if (data.point_of_control?.length > 0) {
            this.chartSeries.pointOfControl = this.addLineSeries(chart, data.point_of_control, '#a855f7', 2, 0, this.overlayVisible.pointOfControl);
        }

        // Horizontal lines (premarket, 3-day)
        const times = data.ohlcv?.map(d => d.time);
        if (times?.length > 0) {
            if (data.premarket_high) {
                this.chartSeries.premarketHigh = this.addHorizontalLine(chart, data.premarket_high, times, '#34d399', this.overlayVisible.premarketHigh);
            }
            if (data.premarket_low) {
                this.chartSeries.premarketLow = this.addHorizontalLine(chart, data.premarket_low, times, '#fb7185', this.overlayVisible.premarketLow);
            }
            if (data.three_day_high) {
                this.chartSeries.threeDayHigh = this.addHorizontalLine(chart, data.three_day_high, times, '#10b981', this.overlayVisible.threeDayHigh);
            }
            if (data.three_day_low) {
                this.chartSeries.threeDayLow = this.addHorizontalLine(chart, data.three_day_low, times, '#f43f5e', this.overlayVisible.threeDayLow);
            }
        }
    },

    // Helper to add a line series
    addLineSeries(chart, data, color, lineWidth = 1, lineStyle = 0, visible = true) {
        const series = chart.addLineSeries({
            color,
            lineWidth,
            lineStyle,
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            visible,
        });
        series.setData(data);
        return series;
    },

    // Helper to add horizontal line
    addHorizontalLine(chart, value, times, color, visible = true) {
        const series = chart.addLineSeries({
            color,
            lineWidth: 1,
            lineStyle: 2,
            priceScaleId: 'left',
            lastValueVisible: false,
            priceLineVisible: false,
            visible,
        });
        series.setData([
            { time: times[0], value },
            { time: times[times.length - 1], value }
        ]);
        return series;
    },

    // Update x-axis visibility (only show on bottom chart, or main when zoomed)
    updateXAxisVisibility() {
        // First, hide x-axis on all charts
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.applyOptions({
                    timeScale: { visible: false },
                    crosshair: { vertLine: { labelVisible: false } }
                });
            }
        });

        // When zoomed, only main chart is visible - show x-axis on it
        if (this.chartZoomed && this.charts.main) {
            this.charts.main.applyOptions({
                timeScale: { visible: true },
                crosshair: { vertLine: { labelVisible: true } }
            });
            return;
        }

        // Find the bottom-most visible chart
        for (const chartName of this.config.chartOrder) {
            if (this.charts[chartName]) {
                this.charts[chartName].applyOptions({
                    timeScale: { visible: true },
                    crosshair: { vertLine: { labelVisible: true } }
                });
                break;
            }
        }
    },

    // Setup crosshair sync across all charts
    setupCrosshairSync() {
        const allCharts = Object.values(this.charts).filter(c => c);
        if (allCharts.length === 0) return;

        const tooltip = document.getElementById('chart-tooltip');
        const self = this;

        // Build chart -> series mapping for crosshair position
        const chartToSeries = new Map();
        const seriesMapping = {
            main: 'ohlcv',
            volume: 'volume',
            imbalance: 'imbalance',
            flow: 'flowCall',
            velocity: 'velocity',
            netGex: 'netGex',
            iv: 'iv',
            totalGex: 'totalGex',
        };

        Object.entries(this.charts).forEach(([key, chart]) => {
            const seriesKey = seriesMapping[key];
            if (seriesKey && this.chartSeries[seriesKey]) {
                chartToSeries.set(chart, this.chartSeries[seriesKey]);
            }
        });

        let isSyncing = false;

        allCharts.forEach(chart => {
            chart.subscribeCrosshairMove(param => {
                // Sync crosshairs
                if (!isSyncing) {
                    isSyncing = true;
                    allCharts.forEach(otherChart => {
                        if (otherChart !== chart) {
                            const series = chartToSeries.get(otherChart);
                            if (param.time && series) {
                                try { otherChart.setCrosshairPosition(0, param.time, series); } catch (e) {}
                            } else if (!param.time) {
                                try { otherChart.clearCrosshairPosition(); } catch (e) {}
                            }
                        }
                    });
                    isSyncing = false;
                }

                // Update tooltip
                if (tooltip) {
                    if (!param.time || param.point === undefined) {
                        tooltip.style.opacity = '0';
                        return;
                    }

                    const html = ChartUtils.buildTooltipContent(self.chartData, param.time, self.overlayVisible);
                    tooltip.innerHTML = html;
                    tooltip.style.opacity = '1';

                    // Position tooltip
                    const mainContainer = document.getElementById(self.config.containerIds.main);
                    if (mainContainer && param.point) {
                        const rect = mainContainer.getBoundingClientRect();
                        const chartMidpoint = rect.left + rect.width / 2;
                        const cursorX = rect.left + param.point.x;
                        const margin = 16;

                        let left = cursorX < chartMidpoint ? rect.right - 260 : rect.left + margin;
                        tooltip.style.left = left + 'px';
                        tooltip.style.top = (rect.top + margin) + 'px';
                    }
                }
            });
        });
    },

    // Toggle overlay visibility
    toggleOverlay(name) {
        this.overlayVisible[name] = !this.overlayVisible[name];
        const series = this.chartSeries[name];
        if (series) {
            series.applyOptions({ visible: this.overlayVisible[name] });
        }
    },

    // Toggle overlay group
    toggleGroup(groupName) {
        const items = this.legendGroups[groupName] || [];
        const anyVisible = items.some(name => this.overlayVisible[name]);
        const newState = !anyVisible;

        items.forEach(name => {
            this.overlayVisible[name] = newState;
            const series = this.chartSeries[name];
            if (series) {
                series.applyOptions({ visible: newState });
            }
        });
    },

    // Check if any item in group is visible
    isGroupVisible(groupName) {
        const items = this.legendGroups[groupName] || [];
        return items.some(name => this.overlayVisible[name]);
    },

    // Handle resize
    handleResize() {
        Object.entries(this.charts).forEach(([name, chart]) => {
            if (chart) {
                const containerId = this.config.containerIds[name];
                const container = document.getElementById(containerId);
                if (container) {
                    chart.applyOptions({
                        height: container.clientHeight,
                        width: container.clientWidth
                    });
                }
            }
        });
    },

    // Fit all charts to show full data range
    fitAllCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                try {
                    chart.timeScale().fitContent();
                } catch (e) { /* ignore */ }
            }
        });
    },
};

// Export for use in templates
if (typeof window !== 'undefined') {
    window.TrainingChartView = TrainingChartView;
}

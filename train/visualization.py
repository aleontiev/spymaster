"""
Training visualization utilities.

Generates HTML files with interactive charts showing training progress.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def generate_training_charts(
    history: List[Dict],
    output_path: Path,
    title: str = "Training Progress",
    model_type: str = "jepa",
) -> None:
    """Generate an HTML file with training progress charts.

    Args:
        history: List of dicts with epoch metrics, e.g.:
            [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, ...}, ...]
        output_path: Path to write the HTML file
        title: Chart title
        model_type: Type of model (jepa, entry, etc.) to customize metrics shown
    """
    if not history:
        return

    # Prepare data series
    epochs = [h["epoch"] for h in history]

    # Common metrics
    train_loss = [h.get("train_loss") for h in history]
    val_loss = [h.get("val_loss") for h in history]

    # JEPA-specific metrics
    pred_loss = [h.get("pred_loss") for h in history]
    sigreg_loss = [h.get("sigreg_loss") for h in history]
    embedding_std = [h.get("embedding_std") for h in history]
    effective_rank = [h.get("effective_rank") for h in history]
    embedding_norm = [h.get("embedding_mean_norm") for h in history]
    learning_rate = [h.get("lr") for h in history]

    # Entry-specific metrics
    accuracy = [h.get("accuracy") for h in history]
    val_accuracy = [h.get("val_accuracy") for h in history]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e5e5e5;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #fff;
        }}
        .meta {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: #141414;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #333;
        }}
        .chart-container h3 {{
            margin-bottom: 15px;
            color: #ccc;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .chart-wrapper {{
            position: relative;
            height: 250px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            max-width: 1600px;
            margin: 0 auto 30px;
        }}
        .stat-card {{
            background: #141414;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #333;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #22c55e;
        }}
        .stat-card .value.warning {{
            color: #eab308;
        }}
        .stat-card .value.danger {{
            color: #ef4444;
        }}
        .stat-card .label {{
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }}
        .repr-warmup {{
            background: rgba(234, 179, 8, 0.1);
            border-left: 3px solid #eab308;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="meta">
        Epochs: {len(history)} | Last updated: <span id="timestamp"></span>
    </div>

    <div class="summary">
        <div class="stat-card">
            <div class="value" id="best-val-loss">--</div>
            <div class="label">Best Val Loss</div>
        </div>
        <div class="stat-card">
            <div class="value" id="current-loss">--</div>
            <div class="label">Current Loss</div>
        </div>
        <div class="stat-card">
            <div class="value" id="eff-rank">--</div>
            <div class="label">Effective Rank</div>
        </div>
        <div class="stat-card">
            <div class="value" id="emb-std">--</div>
            <div class="label">Embedding STD</div>
        </div>
        <div class="stat-card">
            <div class="value" id="current-lr">--</div>
            <div class="label">Learning Rate</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <h3>Loss</h3>
            <div class="chart-wrapper">
                <canvas id="lossChart"></canvas>
            </div>
        </div>

        <div class="chart-container">
            <h3>Loss Components (Pred vs SIGReg)</h3>
            <div class="chart-wrapper">
                <canvas id="componentChart"></canvas>
            </div>
        </div>

        <div class="chart-container">
            <h3>Embedding Health</h3>
            <div class="chart-wrapper">
                <canvas id="embeddingChart"></canvas>
            </div>
        </div>

        <div class="chart-container">
            <h3>Effective Rank</h3>
            <div class="chart-wrapper">
                <canvas id="rankChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const epochs = {json.dumps(epochs)};
        const trainLoss = {json.dumps(train_loss)};
        const valLoss = {json.dumps(val_loss)};
        const predLoss = {json.dumps(pred_loss)};
        const sigregLoss = {json.dumps(sigreg_loss)};
        const embeddingStd = {json.dumps(embedding_std)};
        const effectiveRank = {json.dumps(effective_rank)};
        const embeddingNorm = {json.dumps(embedding_norm)};
        const learningRate = {json.dumps(learning_rate)};

        // Update timestamp
        document.getElementById('timestamp').textContent = new Date().toLocaleString();

        // Update summary stats
        const lastIdx = epochs.length - 1;
        const bestValLoss = Math.min(...valLoss.filter(v => v !== null));

        document.getElementById('best-val-loss').textContent = bestValLoss.toFixed(4);
        document.getElementById('current-loss').textContent = (trainLoss[lastIdx] || 0).toFixed(4);

        const effRankVal = effectiveRank[lastIdx] || 0;
        const effRankEl = document.getElementById('eff-rank');
        effRankEl.textContent = effRankVal.toFixed(0);
        if (effRankVal < 10) effRankEl.classList.add('danger');
        else if (effRankVal < 20) effRankEl.classList.add('warning');

        const embStdVal = embeddingStd[lastIdx] || 0;
        const embStdEl = document.getElementById('emb-std');
        embStdEl.textContent = embStdVal.toFixed(3);
        if (embStdVal < 0.5 || embStdVal > 2.0) embStdEl.classList.add('danger');
        else if (embStdVal < 0.6 || embStdVal > 1.5) embStdEl.classList.add('warning');

        const lrVal = learningRate[lastIdx] || 0;
        document.getElementById('current-lr').textContent = lrVal.toExponential(2);

        // Chart defaults
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = '#333';

        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{
                intersect: false,
                mode: 'index',
            }},
            plugins: {{
                legend: {{
                    position: 'top',
                    labels: {{ usePointStyle: true, padding: 15 }}
                }}
            }},
            scales: {{
                x: {{
                    grid: {{ color: '#222' }},
                    title: {{ display: true, text: 'Epoch' }}
                }},
                y: {{
                    grid: {{ color: '#222' }},
                }}
            }}
        }};

        // Loss chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Train Loss',
                        data: trainLoss,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        fill: true,
                    }},
                    {{
                        label: 'Val Loss',
                        data: valLoss,
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        tension: 0.3,
                        fill: true,
                    }}
                ]
            }},
            options: chartOptions
        }});

        // Component chart
        new Chart(document.getElementById('componentChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Pred Loss',
                        data: predLoss,
                        borderColor: '#a855f7',
                        tension: 0.3,
                    }},
                    {{
                        label: 'SIGReg Loss',
                        data: sigregLoss,
                        borderColor: '#f97316',
                        tension: 0.3,
                    }}
                ]
            }},
            options: chartOptions
        }});

        // Embedding chart
        new Chart(document.getElementById('embeddingChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Embedding STD',
                        data: embeddingStd,
                        borderColor: '#06b6d4',
                        tension: 0.3,
                        yAxisID: 'y',
                    }},
                    {{
                        label: 'Mean Norm',
                        data: embeddingNorm,
                        borderColor: '#ec4899',
                        tension: 0.3,
                        yAxisID: 'y1',
                    }}
                ]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: '#222' }},
                        title: {{ display: true, text: 'STD' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        title: {{ display: true, text: 'Norm' }}
                    }},
                }}
            }}
        }});

        // Rank chart
        new Chart(document.getElementById('rankChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Effective Rank',
                        data: effectiveRank,
                        borderColor: '#eab308',
                        backgroundColor: 'rgba(234, 179, 8, 0.1)',
                        tension: 0.3,
                        fill: true,
                    }}
                ]
            }},
            options: {{
                ...chartOptions,
                plugins: {{
                    ...chartOptions.plugins,
                    annotation: {{
                        annotations: {{
                            collapseThreshold: {{
                                type: 'line',
                                yMin: 5,
                                yMax: 5,
                                borderColor: '#ef4444',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {{
                                    display: true,
                                    content: 'Collapse Threshold',
                                    position: 'start'
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def save_training_history(history: List[Dict], output_path: Path) -> None:
    """Save training history to JSON for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)


def load_training_history(input_path: Path) -> Optional[List[Dict]]:
    """Load training history from JSON."""
    if not input_path.exists():
        return None
    with open(input_path) as f:
        return json.load(f)

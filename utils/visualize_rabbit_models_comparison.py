#!/usr/bin/env python3
"""
Visualize and compare rabbit model test results from 2026-06-24 and 2026-07-21
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path(r"E:\ActionRecognition\behavionRecognition-src\outputs\rabbit")
DATE_DIRS = ["2026-06-24", "2026-07-21"]
MODEL_PATTERNS = ["repvit_m1_0", "repvit_m1_1", "repvit_m1_5", "repvit_m2_3"]
OUTPUT_HTML = Path(r"E:\ActionRecognition\behavionRecognition-src\rabbit_models_comparison.html")

# Training log files
TRAIN_LOGS = {
    "2026-06-24": Path(r"E:\ActionRecognition\behavionRecognition-src\train_rabbit_0624_series.log"),
    "2026-07-21": Path(r"E:\ActionRecognition\behavionRecognition-src\train_rabbit_0721_series.log")
}


def find_evaluation_summaries():
    """Find all evaluation_summary.json files"""
    summaries = defaultdict(dict)

    for date in DATE_DIRS:
        date_path = BASE_DIR / date
        if not date_path.exists():
            print(f"Warning: {date_path} does not exist")
            continue

        # Check for root-level test_results (e.g., 0624_repvit_best_distilled_ema.pth)
        root_test_dirs = list(date_path.glob("test_results_*"))
        if root_test_dirs:
            for test_dir in root_test_dirs:
                summary_file = test_dir / "evaluation_summary.json"
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        checkpoint = data.get('checkpoint', '')
                        # Extract model name from checkpoint path
                        if '0624_repvit_best_distilled_ema.pth' in checkpoint:
                            model_name = 'repvit_0624_ema'
                        elif '0624_repvit_best_distilled.pth' in checkpoint:
                            model_name = 'repvit_0624'
                        else:
                            model_name = f'root_model_{test_dir.name}'

                        summaries[date][model_name] = {
                            'data': data,
                            'path': str(summary_file)
                        }
                        print(f"Found: {date}/{model_name} (root level)")

        # Check model-specific directories
        for model in MODEL_PATTERNS:
            model_path = date_path / model
            if not model_path.exists():
                continue

            # Find all test_results directories
            test_dirs = list(model_path.glob("test_results_*"))
            if test_dirs:
                # Use the latest test results
                latest_test = sorted(test_dirs)[-1]
                summary_file = latest_test / "evaluation_summary.json"

                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summaries[date][model] = {
                            'data': data,
                            'path': str(summary_file)
                        }
                        print(f"Found: {date}/{model}")

    return summaries


def parse_training_log(log_file):
    """Parse training log to extract loss and accuracy"""
    if not log_file.exists():
        print(f"Warning: Training log {log_file} not found")
        return None

    print(f"Parsing training log: {log_file}")

    training_data = defaultdict(lambda: {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    })

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        current_model = None

        for line in f:
            # Detect model name
            model_match = re.search(r'(repvit_m[12]_[0-9])', line)
            if model_match:
                current_model = model_match.group(1)

            if not current_model:
                continue

            # Parse epoch information
            # Pattern: Epoch [X/Y]
            epoch_match = re.search(r'Epoch\s+\[(\d+)/\d+\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))

            # Parse training metrics
            # Pattern: Loss: X.XXXX, Acc: XX.XX%
            if 'Loss:' in line and 'Acc:' in line:
                loss_match = re.search(r'Loss:\s+([\d.]+)', line)
                acc_match = re.search(r'Acc:\s+([\d.]+)%', line)

                if loss_match and acc_match:
                    loss = float(loss_match.group(1))
                    acc = float(acc_match.group(1))

                    if 'Val' in line or 'Validation' in line:
                        if epoch not in training_data[current_model]['epochs']:
                            training_data[current_model]['epochs'].append(epoch)
                        training_data[current_model]['val_loss'].append(loss)
                        training_data[current_model]['val_acc'].append(acc)
                    else:
                        training_data[current_model]['train_loss'].append(loss)
                        training_data[current_model]['train_acc'].append(acc)

    return dict(training_data)


def generate_html_report(summaries, training_logs):
    """Generate comprehensive HTML report"""

    html_parts = []

    # HTML Header
    html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rabbit Model Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }
        h3 {
            color: #666;
            margin-top: 20px;
        }
        .plot-container {
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-card {
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            min-width: 150px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        .metric-label {
            font-size: 12px;
            opacity: 0.9;
        }
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐰 Rabbit Behavior Recognition Model Comparison</h1>
        <div class="info-box">
            <strong>Report Generated:</strong> 2026-09-18<br>
            <strong>Comparing:</strong> 2026-06-24 vs 2026-07-21 model series<br>
            <strong>Models:</strong> RepVit-M1-0, RepVit-M1-1, RepVit-M1-5, RepVit-M2-3
        </div>
""")

    # Summary comparison table
    html_parts.append("<h2>📊 Test Results Summary</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Date</th><th>Model</th><th>Time IoU (Macro)</th><th>Event F1 @0.3</th><th>Event F1 @0.5</th><th>Event F1 @0.7</th><th>Transition F1</th></tr>")

    all_data_for_plots = []

    for date in DATE_DIRS:
        if date not in summaries:
            continue
        for model in MODEL_PATTERNS:
            if model not in summaries[date]:
                continue

            data = summaries[date][model]['data']
            agg = data['aggregate']

            all_data_for_plots.append({
                'date': date,
                'model': model,
                'time_iou': agg['time_iou_macro'],
                'f1_03': agg['event_f1_tiou_0.3'],
                'f1_05': agg['event_f1_tiou_0.5'],
                'f1_07': agg['event_f1_tiou_0.7'],
                'trans_f1': agg['transition_f1']
            })

            html_parts.append(f"<tr>")
            html_parts.append(f"<td>{date}</td>")
            html_parts.append(f"<td><strong>{model}</strong></td>")
            html_parts.append(f"<td>{agg['time_iou_macro']:.4f}</td>")
            html_parts.append(f"<td>{agg['event_f1_tiou_0.3']:.4f}</td>")
            html_parts.append(f"<td>{agg['event_f1_tiou_0.5']:.4f}</td>")
            html_parts.append(f"<td>{agg['event_f1_tiou_0.7']:.4f}</td>")
            html_parts.append(f"<td>{agg['transition_f1']:.4f}</td>")
            html_parts.append(f"</tr>")

    html_parts.append("</table>")

    # Generate Plotly comparison charts
    html_parts.append("<h2>📈 Performance Comparison Charts</h2>")

    # Chart 1: Time IoU comparison
    html_parts.append('<div class="plot-container"><div id="chart_time_iou"></div></div>')

    # Chart 2: Event F1 scores comparison
    html_parts.append('<div class="plot-container"><div id="chart_event_f1"></div></div>')

    # Chart 3: Model comparison by date
    html_parts.append('<div class="plot-container"><div id="chart_by_date"></div></div>')

    # Training curves section
    html_parts.append("<h2>📉 Training Curves</h2>")

    for date, log_data in training_logs.items():
        if not log_data:
            html_parts.append(f'<div class="warning-box">No training log data found for {date}</div>')
            continue

        html_parts.append(f"<h3>Training Series: {date}</h3>")

        for model, train_data in log_data.items():
            if not train_data['epochs']:
                continue

            html_parts.append(f'<div class="plot-container"><div id="train_{date}_{model}"></div></div>')

    # JavaScript for Plotly charts
    html_parts.append("<script>")

    # Prepare data for charts
    chart_data = defaultdict(dict)
    for item in all_data_for_plots:
        chart_data[item['date']][item['model']] = item

    # Chart 1: Time IoU Comparison
    models_0624 = [m for m in MODEL_PATTERNS if m in chart_data['2026-06-24'] and 'time_iou' in chart_data['2026-06-24'][m]]
    models_0721 = [m for m in MODEL_PATTERNS if m in chart_data['2026-07-21'] and 'time_iou' in chart_data['2026-07-21'][m]]

    html_parts.append("""
    var trace_tiou_0624 = {
        x: %s,
        y: %s,
        name: '2026-06-24',
        type: 'bar',
        marker: {color: '#4CAF50'}
    };
    var trace_tiou_0721 = {
        x: %s,
        y: %s,
        name: '2026-07-21',
        type: 'bar',
        marker: {color: '#2196F3'}
    };
    var layout_tiou = {
        title: 'Time IoU (Macro) Comparison',
        xaxis: {title: 'Model'},
        yaxis: {title: 'Time IoU', range: [0, 1]},
        barmode: 'group'
    };
    Plotly.newPlot('chart_time_iou', [trace_tiou_0624, trace_tiou_0721], layout_tiou);
    """ % (
        json.dumps(models_0624),
        json.dumps([chart_data['2026-06-24'][m]['time_iou'] for m in models_0624]),
        json.dumps(models_0721),
        json.dumps([chart_data['2026-07-21'][m]['time_iou'] for m in models_0721])
    ))

    # Chart 2: Event F1 Scores
    traces_f1 = []
    for date, color in [('2026-06-24', '#4CAF50'), ('2026-07-21', '#2196F3')]:
        models_list = [m for m in MODEL_PATTERNS if m in chart_data[date]]
        if models_list:
            # Calculate means
            mean_f1_03 = sum([chart_data[date][m]['f1_03'] for m in models_list]) / len(models_list)
            mean_f1_05 = sum([chart_data[date][m]['f1_05'] for m in models_list]) / len(models_list)
            mean_f1_07 = sum([chart_data[date][m]['f1_07'] for m in models_list]) / len(models_list)
            mean_trans = sum([chart_data[date][m]['trans_f1'] for m in models_list]) / len(models_list)

            html_parts.append("""
    var trace_f1_%s = {
        x: ['F1@0.3', 'F1@0.5', 'F1@0.7', 'Trans F1'],
        y: %s,
        name: '%s',
        type: 'scatter',
        mode: 'lines+markers',
        marker: {color: '%s', size: 10}
    };
            """ % (
                date.replace('-', ''),
                json.dumps([mean_f1_03, mean_f1_05, mean_f1_07, mean_trans]),
                date,
                color
            ))

    html_parts.append("""
    var layout_f1 = {
        title: 'Average Event F1 Scores Across Models',
        xaxis: {title: 'Metric'},
        yaxis: {title: 'Score', range: [0, 0.5]}
    };
    Plotly.newPlot('chart_event_f1', [trace_f1_20260624, trace_f1_20260721], layout_f1);
    """)

    # Chart 3: Grouped comparison
    html_parts.append("""
    var data_by_date = [];
    """)

    for model in MODEL_PATTERNS:
        dates_avail = [d for d in DATE_DIRS if model in chart_data[d]]
        if dates_avail:
            html_parts.append("""
    data_by_date.push({
        x: %s,
        y: %s,
        name: '%s',
        type: 'bar'
    });
            """ % (
                json.dumps(dates_avail),
                json.dumps([chart_data[d][model]['time_iou'] for d in dates_avail]),
                model
            ))

    html_parts.append("""
    var layout_by_date = {
        title: 'Time IoU by Model and Date',
        xaxis: {title: 'Date'},
        yaxis: {title: 'Time IoU', range: [0, 1]},
        barmode: 'group'
    };
    Plotly.newPlot('chart_by_date', data_by_date, layout_by_date);
    """)

    # Training curves
    for date, log_data in training_logs.items():
        if not log_data:
            continue

        for model, train_data in log_data.items():
            if not train_data['epochs']:
                continue

            epochs = train_data['epochs']
            train_loss = train_data['train_loss'][:len(epochs)]
            val_loss = train_data['val_loss']
            train_acc = train_data['train_acc'][:len(epochs)]
            val_acc = train_data['val_acc']

            html_parts.append("""
    var trace_train_loss = {
        x: %s,
        y: %s,
        name: 'Train Loss',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y'
    };
    var trace_val_loss = {
        x: %s,
        y: %s,
        name: 'Val Loss',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y'
    };
    var trace_train_acc = {
        x: %s,
        y: %s,
        name: 'Train Acc',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y2'
    };
    var trace_val_acc = {
        x: %s,
        y: %s,
        name: 'Val Acc',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y2'
    };
    var layout_train = {
        title: 'Training Curves: %s - %s',
        xaxis: {title: 'Epoch'},
        yaxis: {title: 'Loss', side: 'left'},
        yaxis2: {
            title: 'Accuracy (%%)',
            overlaying: 'y',
            side: 'right'
        }
    };
    Plotly.newPlot('train_%s_%s', [trace_train_loss, trace_val_loss, trace_train_acc, trace_val_acc], layout_train);
            """ % (
                json.dumps(epochs), json.dumps(train_loss),
                json.dumps(epochs), json.dumps(val_loss),
                json.dumps(epochs), json.dumps(train_acc),
                json.dumps(epochs), json.dumps(val_acc),
                date, model,
                date, model
            ))

    html_parts.append("</script>")

    # Footer
    html_parts.append("""
        <div class="info-box" style="margin-top: 40px;">
            <strong>Notes:</strong><br>
            - Time IoU (Macro): Average temporal Intersection over Union across all videos<br>
            - Event F1: F1 score for detecting behavioral events at different IoU thresholds<br>
            - Transition F1: F1 score for detecting behavior transitions<br>
            - Higher values indicate better performance
        </div>
    </div>
</body>
</html>
""")

    return ''.join(html_parts)


def main():
    print("Starting rabbit model comparison analysis...")

    # Find all test results
    print("\n" + "="*60)
    print("Step 1: Finding evaluation summaries...")
    print("="*60)
    summaries = find_evaluation_summaries()

    # Parse training logs
    print("\n" + "="*60)
    print("Step 2: Parsing training logs...")
    print("="*60)
    training_logs = {}
    for date, log_file in TRAIN_LOGS.items():
        training_logs[date] = parse_training_log(log_file)

    # Generate HTML report
    print("\n" + "="*60)
    print("Step 3: Generating HTML report...")
    print("="*60)
    html_content = generate_html_report(summaries, training_logs)

    # Write to file
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nReport generated successfully: {OUTPUT_HTML}")
    print(f"   Open this file in a web browser to view the results")

    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for date in DATE_DIRS:
        if date in summaries:
            print(f"\n{date}:")
            for model in summaries[date]:
                agg = summaries[date][model]['data']['aggregate']
                print(f"  {model}: Time IoU={agg['time_iou_macro']:.4f}, F1@0.5={agg['event_f1_tiou_0.5']:.4f}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()

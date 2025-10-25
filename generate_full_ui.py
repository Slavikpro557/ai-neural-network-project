"""
Генератор полного UI с инструкциями и графиками
"""

html_content = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AZR Model Trainer - Полное руководство</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        
        .tabs {
            display: flex;
            background: #f5f5f5;
            border-bottom: 2px solid #ddd;
            flex-wrap: wrap;
        }
        
        .tab {
            flex: 1;
            min-width: 120px;
            padding: 15px 10px;
            text-align: center;
            cursor: pointer;
            background: #f5f5f5;
            border: none;
            font-size: 0.85em;
            font-weight: 600;
            color: #666;
            transition: all 0.3s;
        }
        
        .tab:hover { background: #e0e0e0; }
        .tab.active {
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }
        
        .tab-content {
            display: none;
            padding: 30px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .tab-content.active { display: block; }
        
        .form-group {
            margin-bottom: 20px;
            position: relative;
        }
        
        .form-group label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        .help-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #667eea;
            color: white;
            font-size: 11px;
            cursor: help;
            position: relative;
        }
        
        .tooltip {
            display: none;
            position: absolute;
            background: #1f2937;
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: normal;
            width: 280px;
            z-index: 1000;
            left: 100%;
            top: -10px;
            margin-left: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            line-height: 1.4;
        }
        
        .tooltip::before {
            content: '';
            position: absolute;
            left: -6px;
            top: 15px;
            width: 0;
            height: 0;
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            border-right: 6px solid #1f2937;
        }
        
        .help-icon:hover .tooltip { display: block; }
        
        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .recommended {
            font-size: 0.85em;
            color: #10b981;
            margin-top: 4px;
            font-weight: normal;
        }
        
        .warning {
            font-size: 0.85em;
            color: #f59e0b;
            margin-top: 4px;
            font-weight: normal;
        }
        
        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-success {
            background: #10b981;
            color: white;
        }
        
        .btn-success:hover { background: #059669; }
        
        .btn-danger {
            background: #ef4444;
            color: white;
        }
        
        .btn-danger:hover { background: #dc2626; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .chart-container {
            position: relative;
            height: 350px;
            margin: 20px 0;
            background: #f9fafb;
            border-radius: 12px;
            padding: 20px;
            border: 2px solid #e5e7eb;
        }
        
        .status-box {
            background: #f9fafb;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .status-box h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .status-item:last-child { border-bottom: none; }
        
        .status-label {
            font-weight: 600;
            color: #666;
        }
        
        .status-value {
            color: #333;
            font-weight: 600;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .help-section {
            background: #f0f9ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
        }
        
        .help-section h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .help-section p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 8px;
        }
        
        .help-section ul {
            margin-left: 20px;
            color: #666;
            line-height: 1.8;
        }
        
        .info-card {
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            transition: all 0.3s;
        }
        
        .info-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        }
        
        .info-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        .info-card ol,
        .info-card ul {
            line-height: 2;
            color: #444;
        }
        
        .info-card strong {
            color: #667eea;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid;
        }
        
        .alert-success {
            background: #d1fae5;
            color: #065f46;
            border-color: #10b981;
        }
        
        .alert-error {
            background: #fee2e2;
            color: #991b1b;
            border-color: #ef4444;
        }
        
        .alert-info {
            background: #dbeafe;
            color: #1e40af;
            border-color: #3b82f6;
        }
        
        .dataset-list {
            border: 2px dashed #ddd;
            border-radius: 12px;
            padding: 20px;
            min-height: 100px;
            margin: 10px 0;
        }
        
        .dataset-item {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 10px 15px;
            margin: 8px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .dataset-item:hover {
            background: #f3f4f6;
        }
        
        .model-card {
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
        }
        
        .model-card h4 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .output-box {
            background: #1f2937;
            color: #10b981;
            padding: 20px;
            border-radius: 12px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .file-upload {
            border: 3px dashed #ddd;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .file-upload:hover {
            border-color: #667eea;
            background: #f9fafb;
        }
        
        .file-upload input {
            display: none;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        table th,
        table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        
        table th {
            background: #f9fafb;
            color: #667eea;
            font-weight: 600;
        }
        
        table tr:hover {
            background: #f9fafb;
        }
        
        code {
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        pre {
            background: #1f2937;
            color: #10b981;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 AZR Model Trainer</h1>
            <p>Absolute Zero Reasoner - Самообучающаяся AI с полными инструкциями</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('help')">📚 Помощь</button>
            <button class="tab" onclick="showTab('create')">🏗️ Создать модель</button>
            <button class="tab" onclick="showTab('datasets')">📁 Датасеты</button>
            <button class="tab" onclick="showTab('train')">🚀 Обучение</button>
            <button class="tab" onclick="showTab('generate')">✨ Генерация</button>
            <button class="tab" onclick="showTab('models')">📊 Мои модели</button>
        </div>
'''

# Сохраняем первую часть
with open(r'C:\Users\clavi\Desktop\для ии\templates\index_full.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("OK: Created first part of HTML")
print("Now adding tab content...")

# Продолжение будет добавлено

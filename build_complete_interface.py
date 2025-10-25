#!/usr/bin/env python3
"""
Строит ПОЛНЫЙ интерфейс со всеми фичами
- Инструкции встроены
- Tooltips на всех элементах
- Живой график
- Рекомендации
- Всё работает
"""

import os

# Полный HTML со всеми фичами
html = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AZR Model Trainer - Полный интерфейс</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 20px 20px 0 0; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .tabs { display: flex; background: #f5f5f5; flex-wrap: wrap; border-bottom: 2px solid #ddd; }
        .tab { flex: 1; min-width: 130px; padding: 15px 10px; text-align: center; cursor: pointer; background: #f5f5f5; border: none; font-size: 0.9em; font-weight: 600; color: #666; transition: all 0.3s; }
        .tab:hover { background: #e0e0e0; }
        .tab.active { background: white; color: #667eea; border-bottom: 3px solid #667eea; }
        .tab-content { display: none; padding: 30px; max-height: 80vh; overflow-y: auto; }
        .tab-content.active { display: block; }
        .form-group { margin-bottom: 20px; position: relative; }
        .form-group label { display: flex; align-items: center; gap: 8px; font-weight: 600; margin-bottom: 8px; color: #333; }
        .help-icon { display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; border-radius: 50%; background: #667eea; color: white; font-size: 12px; font-weight: bold; cursor: help; position: relative; }
        .tooltip { display: none; position: absolute; background: #1f2937; color: white; padding: 12px; border-radius: 8px; font-size: 12px; width: 300px; z-index: 1000; left: 100%; top: -10px; margin-left: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); line-height: 1.5; font-weight: normal; }
        .tooltip::before { content: ''; position: absolute; left: -6px; top: 15px; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-right: 6px solid #1f2937; }
        .help-icon:hover .tooltip { display: block; }
        .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; font-size: 1em; transition: border-color 0.3s; }
        .form-group input:focus, .form-group select:focus { outline: none; border-color: #667eea; }
        .recommended { font-size: 0.85em; color: #10b981; margin-top: 4px; }
        .warning { font-size: 0.85em; color: #f59e0b; margin-top: 4px; }
        .btn { padding: 12px 30px; border: none; border-radius: 8px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s; margin-right: 10px; margin-bottom: 10px; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        .btn-success { background: #10b981; color: white; }
        .btn-danger { background: #ef4444; color: white; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .chart-container { position: relative; height: 350px; margin: 20px 0; background: #f9fafb; border-radius: 12px; padding: 20px; border: 2px solid #e5e7eb; }
        .status-box { background: #f9fafb; border: 2px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-top: 20px; }
        .status-box h3 { color: #667eea; margin-bottom: 15px; }
        .status-item { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e5e7eb; }
        .status-label { font-weight: 600; color: #666; }
        .status-value { color: #333; font-weight: 600; }
        .progress-bar { width: 100%; height: 30px; background: #e5e7eb; border-radius: 15px; overflow: hidden; margin: 15px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; }
        .info-card { background: white; border: 2px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-bottom: 15px; }
        .info-card h3 { color: #667eea; margin-bottom: 10px; }
        .info-card ul, .info-card ol { margin-left: 20px; line-height: 1.8; color: #555; }
        .alert { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .alert-success { background: #d1fae5; color: #065f46; border: 2px solid #10b981; }
        .alert-error { background: #fee2e2; color: #991b1b; border: 2px solid #ef4444; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        table th, table td { padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }
        table th { background: #f9fafb; color: #667eea; font-weight: 600; }
        code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
        .dataset-item { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 15px; margin: 8px 0; display: flex; justify-content: space-between; align-items: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 AZR Model Trainer</h1>
            <p>Полный интерфейс с инструкциями, графиками и всеми возможностями</p>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('help')">📚 Помощь</button>
            <button class="tab" onclick="showTab('create')">🏗️ Создать</button>
            <button class="tab" onclick="showTab('datasets')">📁 Датасеты</button>
            <button class="tab" onclick="showTab('train')">🚀 Обучение</button>
            <button class="tab" onclick="showTab('generate')">✨ Генерация</button>
            <button class="tab" onclick="showTab('models')">📊 Модели</button>
        </div>
        
        <!-- ПОМОЩЬ -->
        <div id="help" class="tab-content active">
            <h2>📚 Полное руководство</h2>
            
            <div class="info-card">
                <h3>🎯 Быстрый старт за 3 шага</h3>
                <ol>
                    <li><strong>Создайте модель</strong> → Вкладка "Создать" → Заполните параметры → Нажмите кнопку</li>
                    <li><strong>Загрузите данные</strong> → Вкладка "Датасеты" → Выберите файлы → Прикрепите к модели</li>
                    <li><strong>Запустите обучение</strong> → Вкладка "Обучение" → Выберите модель → Начать</li>
                </ol>
            </div>
            
            <div class="info-card">
                <h3>📖 Подробные инструкции</h3>
                <h4>1. Создание модели</h4>
                <ul>
                    <li><strong>Название</strong> - любое имя (латиница, без пробелов)</li>
                    <li><strong>Vocab Size</strong> - размер словаря (5000-15000 оптимально)</li>
                    <li><strong>D Model</strong> - размерность модели:
                        <ul>
                            <li>64-128: быстро, базовое качество</li>
                            <li>256-384: оптимально (рекомендуется)</li>
                            <li>512-768: высокое качество, нужен GPU</li>
                        </ul>
                    </li>
                    <li><strong>Num Layers</strong> - количество слоёв (4-12 оптимально)</li>
                    <li><strong>Num Heads</strong> - attention heads (4-16, делитель d_model)</li>
                </ul>
                
                <h4>2. Загрузка датасетов</h4>
                <ul>
                    <li>Поддерживаются только <code>.txt</code> файлы в UTF-8</li>
                    <li>Можно загрузить несколько книг/текстов</li>
                    <li>Прикрепите нужные к модели</li>
                    <li>При обучении все прикреплённые используются автоматически</li>
                </ul>
                
                <h4>3. Обучение</h4>
                <ul>
                    <li><strong>Max Iterations</strong> - количество шагов:
                        <ul>
                            <li>1,000 = быстрый тест (10 минут)</li>
                            <li>10,000 = нормальное обучение (1-2 часа)</li>
                            <li>100,000+ = серьёзная модель (дни)</li>
                        </ul>
                    </li>
                    <li><strong>Batch Size</strong> - чем больше, тем быстрее (но нужна память)</li>
                    <li><strong>Learning Rate</strong> - не трогайте без понимания (0.0003 ОК)</li>
                    <li>Следите за графиком: Loss должен падать, Reward расти</li>
                    <li>Можно остановить (Stop) и продолжить позже (Resume)</li>
                </ul>
            </div>
            
            <div class="info-card">
                <h3>⚙️ Рекомендуемые конфигурации</h3>
                <table>
                    <tr>
                        <th>Цель</th>
                        <th>D Model</th>
                        <th>Layers</th>
                        <th>Iterations</th>
                        <th>Время</th>
                    </tr>
                    <tr>
                        <td>Быстрый тест</td>
                        <td>128</td>
                        <td>4</td>
                        <td>1,000</td>
                        <td>~10 мин</td>
                    </tr>
                    <tr>
                        <td>Прототип</td>
                        <td>256</td>
                        <td>6</td>
                        <td>10,000</td>
                        <td>~2 часа</td>
                    </tr>
                    <tr>
                        <td>Продакшн</td>
                        <td>512</td>
                        <td>12</td>
                        <td>100,000</td>
                        <td>~2 дня</td>
                    </tr>
                </table>
            </div>
            
            <div class="info-card">
                <h3>❓ Частые вопросы</h3>
                <p><strong>Q: Почему Loss не падает?</strong><br>
                A: Уменьшите learning rate, проверьте данные, увеличьте размер модели.</p>
                
                <p><strong>Q: Модель повторяет один текст?</strong><br>
                A: Overfitting - нужно больше данных или меньше итераций.</p>
                
                <p><strong>Q: Out of memory?</strong><br>
                A: Уменьшите batch_size, d_model или используйте CPU.</p>
                
                <p><strong>Q: Как остановить обучение?</strong><br>
                A: Нажмите Stop на вкладке "Обучение" или Ctrl+C в терминале.</p>
            </div>
        </div>
'''

# Сохраняем
output_path = r'C:\Users\clavi\Desktop\для ии\templates\index_complete.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html)
    f.write('''
        <!-- СОЗДАТЬ МОДЕЛЬ -->
        <div id="create" class="tab-content">
            <h2>🏗️ Создать новую модель</h2>
            
            <div class="form-group">
                <label>
                    Название модели:
                    <span class="help-icon">?
                        <span class="tooltip">Имя вашей модели. Используйте латиницу без пробелов. Например: my_model, tolstoy_style, code_generator</span>
                    </span>
                </label>
                <input type="text" id="model_name" placeholder="my_awesome_model">
            </div>
            
            <div class="grid">
                <div class="form-group">
                    <label>
                        Vocab Size:
                        <span class="help-icon">?
                            <span class="tooltip">Размер словаря модели. Больше = знает больше слов, но медленнее. 5000-10000 для начала, 15000+ для серьёзных моделей.</span>
                        </span>
                    </label>
                    <input type="number" id="vocab_size" value="8000">
                    <div class="recommended">Рекомендуется: 8000</div>
                </div>
                
                <div class="form-group">
                    <label>
                        D Model:
                        <span class="help-icon">?
                            <span class="tooltip">Размерность внутреннего представления. Больше = умнее модель, но медленнее обучение и генерация. 256-384 оптимально для большинства задач.</span>
                        </span>
                    </label>
                    <input type="number" id="d_model" value="256">
                    <div class="recommended">Рекомендуется: 256 (быстро) или 384 (лучше)</div>
                </div>
                
                <div class="form-group">
                    <label>
                        Num Layers:
                        <span class="help-icon">?
                            <span class="tooltip">Количество трансформер-блоков. Больше = глубже понимание, но тяжелее. 4-8 оптимально.</span>
                        </span>
                    </label>
                    <input type="number" id="num_layers" value="6">
                    <div class="recommended">Рекомендуется: 6</div>
                </div>
                
                <div class="form-group">
                    <label>
                        Num Heads:
                        <span class="help-icon">?
                            <span class="tooltip">Количество attention heads. Должно делить d_model нацело. 8 головок = универсально.</span>
                        </span>
                    </label>
                    <input type="number" id="num_heads" value="8">
                    <div class="recommended">Рекомендуется: 8</div>
                </div>
                
                <div class="form-group">
                    <label>
                        D FF:
                        <span class="help-icon">?
                            <span class="tooltip">Размер feed-forward сети. Обычно в 4 раза больше d_model. Не трогайте если не уверены.</span>
                        </span>
                    </label>
                    <input type="number" id="d_ff" value="1024">
                    <div class="recommended">Рекомендуется: d_model × 4</div>
                </div>
                
                <div class="form-group">
                    <label>
                        Max Seq Len:
                        <span class="help-icon">?
                            <span class="tooltip">Максимальная длина последовательности в токенах. Больше = понимает больший контекст, но медленнее.</span>
                        </span>
                    </label>
                    <input type="number" id="max_seq_len" value="256">
                    <div class="recommended">Рекомендуется: 256 или 512</div>
                </div>
            </div>
            
            <button class="btn btn-primary" onclick="createModel()">🚀 Создать модель</button>
            <div id="create_status"></div>
        </div>
        
        <!-- ДАТАСЕТЫ -->
        <div id="datasets" class="tab-content">
            <h2>📁 Управление датасетами</h2>
            
            <div class="info-card">
                <h3>📤 Загрузить файлы</h3>
                <div class="file-upload" onclick="document.getElementById('book_file').click()">
                    <p style="font-size:3em">📚</p>
                    <p>Нажмите или перетащите .txt файлы</p>
                    <input type="file" id="book_file" accept=".txt" onchange="uploadBook()">
                </div>
                <div id="upload_status"></div>
            </div>
            
            <div class="info-card">
                <h3>📋 Управление</h3>
                <div class="form-group">
                    <label>Выберите модель:</label>
                    <select id="dataset_model_name"></select>
                </div>
                
                <h4>Доступные датасеты:</h4>
                <div id="available_datasets"></div>
                
                <h4>Прикреплённые к модели:</h4>
                <div id="attached_datasets"></div>
            </div>
        </div>
        
        <!-- ОБУЧЕНИЕ -->
        <div id="train" class="tab-content">
            <h2>🚀 Обучение модели</h2>
            
            <div class="form-group">
                <label>Выбрать модель:</label>
                <select id="train_model_name"></select>
            </div>
            
            <div class="grid">
                <div class="form-group">
                    <label>
                        Max Iterations:
                        <span class="help-icon">?
                            <span class="tooltip">Сколько шагов обучения. 1,000 = 10мин, 10,000 = 2ч, 100,000 = 2 дня. Можно остановить и продолжить позже с другим значением.</span>
                        </span>
                    </label>
                    <input type="number" id="max_iterations" value="10000">
                    <div class="recommended">Начните с 1000 для теста</div>
                </div>
                
                <div class="form-group">
                    <label>
                        Batch Size:
                        <span class="help-icon">?
                            <span class="tooltip">Количество примеров за раз. Больше = быстрее, но нужна память. 8-16 для CPU, 32+ для GPU.</span>
                        </span>
                    </label>
                    <input type="number" id="batch_size" value="16">
                    <div class="recommended">16 оптимально</div>
                </div>
                
                <div class="form-group">
                    <label>
                        Learning Rate:
                        <span class="help-icon">?
                            <span class="tooltip">Скорость обучения. Меньше = медленнее но стабильнее. Не трогайте без понимания.</span>
                        </span>
                    </label>
                    <input type="number" id="learning_rate" value="0.0003" step="0.0001">
                    <div class="recommended">0.0003 по умолчанию</div>
                </div>
                
                <div class="form-group">
                    <label>Save Every:</label>
                    <input type="number" id="save_every" value="1000">
                    <div class="recommended">Сохранять каждые 1000 итераций</div>
                </div>
            </div>
            
            <button class="btn btn-primary" onclick="startTraining()">▶️ Начать обучение</button>
            <button class="btn btn-danger" onclick="stopTraining()">⏸️ Остановить</button>
            <button class="btn btn-success" onclick="updateTrainingStatus()">🔄 Обновить статус</button>
            
            <div class="chart-container">
                <canvas id="trainingChart"></canvas>
            </div>
            
            <div class="status-box">
                <h3>📊 Статус обучения</h3>
                <div class="status-item">
                    <span class="status-label">Статус:</span>
                    <span class="status-value" id="is_training">Не запущено</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress" style="width: 0%">0%</div>
                </div>
                <div class="status-item">
                    <span class="status-label">Итерация:</span>
                    <span class="status-value" id="current_iteration">0 / 0</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Loss:</span>
                    <span class="status-value" id="current_loss">0.0000</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Reward:</span>
                    <span class="status-value" id="current_reward">0.0000</span>
                </div>
            </div>
        </div>
        
        <!-- ГЕНЕРАЦИЯ -->
        <div id="generate" class="tab-content">
            <h2>✨ Генерация текста</h2>
            
            <div class="form-group">
                <label>Выбрать модель:</label>
                <select id="gen_model_name"></select>
            </div>
            
            <div class="form-group">
                <label>Промпт (начало текста):</label>
                <textarea id="prompt" rows="3" placeholder="Искусственный интеллект это..."></textarea>
            </div>
            
            <div class="grid">
                <div class="form-group">
                    <label>
                        Max Length:
                        <span class="help-icon">?
                            <span class="tooltip">Сколько токенов сгенерировать. ~50 = пара предложений, ~200 = параграф.</span>
                        </span>
                    </label>
                    <input type="number" id="max_length" value="100">
                </div>
                
                <div class="form-group">
                    <label>
                        Temperature:
                        <span class="help-icon">?
                            <span class="tooltip">Креативность. 0.5 = консервативно, 1.0 = норма, 1.5+ = очень креативно.</span>
                        </span>
                    </label>
                    <input type="number" id="temperature" value="0.8" step="0.1">
                </div>
                
                <div class="form-group">
                    <label>Top K:</label>
                    <input type="number" id="top_k" value="40">
                </div>
            </div>
            
            <button class="btn btn-primary" onclick="generateText()">✨ Сгенерировать</button>
            
            <div class="info-card">
                <h3>📝 Результат:</h3>
                <div id="generated_output" style="min-height:100px;padding:15px;background:#f9fafb;border-radius:8px;">
                    Сгенерированный текст появится здесь...
                </div>
            </div>
        </div>
        
        <!-- МОИ МОДЕЛИ -->
        <div id="models" class="tab-content">
            <h2>📊 Мои модели</h2>
            <button class="btn btn-success" onclick="loadModels()">🔄 Обновить список</button>
            <div id="models_list" style="margin-top:20px;"></div>
        </div>
    </div>
''')

    # JavaScript
    f.write('''
    <script>
        let trainingChart = null;
        let chartData = { labels: [], loss: [], reward: [] };
        
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            if (tabName === 'train') {
                loadModels();
                initChart();
                startStatusUpdates();
            } else if (tabName === 'generate') {
                loadModels();
            } else if (tabName === 'datasets') {
                loadModels();
                loadDatasets();
            } else if (tabName === 'models') {
                loadModels();
            }
        }
        
        function initChart() {
            if (trainingChart) return;
            
            const ctx = document.getElementById('trainingChart');
            if (!ctx) return;
            
            trainingChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: 'Loss',
                        data: chartData.loss,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Reward',
                        data: chartData.reward,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: false }
                    },
                    plugins: {
                        legend: { display: true, position: 'top' }
                    }
                }
            });
        }
        
        async function createModel() {
            const config = {
                name: document.getElementById('model_name').value,
                vocab_size: parseInt(document.getElementById('vocab_size').value),
                d_model: parseInt(document.getElementById('d_model').value),
                num_layers: parseInt(document.getElementById('num_layers').value),
                num_heads: parseInt(document.getElementById('num_heads').value),
                d_ff: parseInt(document.getElementById('d_ff').value),
                max_seq_len: parseInt(document.getElementById('max_seq_len').value)
            };
            
            try {
                const response = await fetch('/create_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('create_status').innerHTML = 
                        `<div class="alert alert-success">✅ Модель создана! Параметров: ${data.parameters.toLocaleString()}</div>`;
                } else {
                    document.getElementById('create_status').innerHTML = 
                        `<div class="alert alert-error">❌ ${data.detail}</div>`;
                }
            } catch (error) {
                document.getElementById('create_status').innerHTML = 
                    `<div class="alert alert-error">❌ ${error.message}</div>`;
            }
        }
        
        async function uploadBook() {
            const fileInput = document.getElementById('book_file');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/upload_book', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('upload_status').innerHTML = 
                        `<div class="alert alert-success">✅ Загружено: ${data.filename} (${(data.size/1024).toFixed(1)} KB)</div>`;
                    loadDatasets();
                }
            } catch (error) {
                document.getElementById('upload_status').innerHTML = 
                    `<div class="alert alert-error">❌ ${error.message}</div>`;
            }
        }
        
        async function loadModels() {
            const response = await fetch('/models');
            const data = await response.json();
            
            const selects = ['train_model_name', 'gen_model_name', 'dataset_model_name'];
            selects.forEach(id => {
                const select = document.getElementById(id);
                if (select) {
                    select.innerHTML = data.models.map(m => 
                        `<option value="${m.name}">${m.name}</option>`
                    ).join('');
                }
            });
            
            const modelsList = document.getElementById('models_list');
            if (modelsList) {
                modelsList.innerHTML = data.models.map(m => `
                    <div class="info-card">
                        <h3>🤖 ${m.name}</h3>
                        <p><strong>Параметры:</strong> d_model=${m.config.d_model}, layers=${m.config.num_layers}</p>
                        <p><strong>Датасетов:</strong> ${m.total_datasets || 0}</p>
                        <button class="btn btn-success" onclick="downloadModel('${m.name}')">📥 Скачать</button>
                    </div>
                `).join('');
            }
        }
        
        async function loadDatasets() {
            const response = await fetch('/books');
            const data = await response.json();
            
            const availableDiv = document.getElementById('available_datasets');
            if (availableDiv) {
                availableDiv.innerHTML = data.books.map(b => `
                    <div class="dataset-item">
                        <span>${b.name || b} (${((b.size||0)/1024).toFixed(1)} KB)</span>
                        <button class="btn btn-primary" onclick="attachDataset('${b.name || b}')">+ Прикрепить</button>
                    </div>
                `).join('');
            }
        }
        
        async function startTraining() {
            const config = {
                model_name: document.getElementById('train_model_name').value,
                max_iterations: parseInt(document.getElementById('max_iterations').value),
                batch_size: parseInt(document.getElementById('batch_size').value),
                learning_rate: parseFloat(document.getElementById('learning_rate').value),
                save_every: parseInt(document.getElementById('save_every').value)
            };
            
            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                alert('Обучение запущено! Смотрите статус ниже.');
                startStatusUpdates();
            } catch (error) {
                alert('Ошибка: ' + error.message);
            }
        }
        
        async function stopTraining() {
            await fetch('/stop_training', {method: 'POST'});
            alert('Обучение остановится после текущего батча...');
        }
        
        let statusInterval = null;
        function startStatusUpdates() {
            if (statusInterval) clearInterval(statusInterval);
            updateTrainingStatus();
            statusInterval = setInterval(updateTrainingStatus, 2000);
        }
        
        async function updateTrainingStatus() {
            try {
                const response = await fetch('/training_status');
                const status = await response.json();
                
                document.getElementById('is_training').textContent = 
                    status.is_training ? '🟢 Обучается...' : '🔴 Не активно';
                
                const progress = status.max_iterations > 0 ? 
                    (status.current_iteration / status.max_iterations * 100).toFixed(2) : 0;
                
                document.getElementById('progress').style.width = progress + '%';
                document.getElementById('progress').textContent = progress + '%';
                
                document.getElementById('current_iteration').textContent = 
                    `${status.current_iteration} / ${status.max_iterations}`;
                document.getElementById('current_loss').textContent = 
                    status.current_loss.toFixed(4);
                document.getElementById('current_reward').textContent = 
                    status.current_reward.toFixed(4);
                
                // Update chart
                if (status.current_iteration > 0) {
                    chartData.labels.push(status.current_iteration);
                    chartData.loss.push(status.current_loss);
                    chartData.reward.push(status.current_reward);
                    
                    if (chartData.labels.length > 50) {
                        chartData.labels.shift();
                        chartData.loss.shift();
                        chartData.reward.shift();
                    }
                    
                    if (trainingChart) {
                        trainingChart.update();
                    }
                }
            } catch (error) {
                console.error('Status update error:', error);
            }
        }
        
        async function generateText() {
            const config = {
                model_name: document.getElementById('gen_model_name').value,
                prompt: document.getElementById('prompt').value,
                max_length: parseInt(document.getElementById('max_length').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                top_k: parseInt(document.getElementById('top_k').value)
            };
            
            document.getElementById('generated_output').textContent = 'Генерация...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const data = await response.json();
                document.getElementById('generated_output').textContent = data.generated_text;
            } catch (error) {
                document.getElementById('generated_output').textContent = 'Ошибка: ' + error.message;
            }
        }
        
        function downloadModel(modelName) {
            window.location.href = `/download_model/${modelName}`;
        }
        
        async function attachDataset(datasetName) {
            const modelName = document.getElementById('dataset_model_name').value;
            await fetch('/attach_dataset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model_name: modelName, dataset_name: datasetName})
            });
            loadDatasets();
        }
        
        // Auto-load on page load
        window.addEventListener('load', () => {
            loadModels();
            initChart();
            startStatusUpdates();
        });
    </script>
</body>
</html>
''')

print("SUCCESS: Complete interface created!")
print(f"File: {output_path}")
print("Now update server.py to use index_complete.html")

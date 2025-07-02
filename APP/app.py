from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
import tempfile
from clasification import procesar_archivo_csv
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-csv', methods=['POST'])
def process_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    # Aquí obtengo el parámetro "anotado" del formulario
    es_anotado = request.form.get('anotado', 'false').lower() == 'true'
    
    # Guardo el archivo temporalmente para procesarlo
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_input:
        file.save(tmp_input.name)
        
        try:
            # Proceso el CSV con mi sistema de clasificación
            print(f"Procesando archivo: {tmp_input.name}, Anotado: {es_anotado}")
            resultado = procesar_archivo_csv(tmp_input.name, es_anotado)
            
            # Limpio el archivo temporal de entrada
            os.unlink(tmp_input.name)
            
            # Manejo los diferentes tipos de respuesta que puedo devolver
            if isinstance(resultado, dict):  # Archivo anotado con métricas
                file_path = resultado['file_path']
                metrics = resultado['metrics']
                return jsonify({
                    'success': True,
                    'type': 'file_with_metrics',
                    'filename': os.path.basename(file_path),
                    'download_url': f'/download/{os.path.basename(file_path)}',
                    'metrics': metrics
                })
            elif isinstance(resultado, str):  # Solo la ruta del archivo
                return jsonify({
                    'success': True,
                    'type': 'file',
                    'filename': os.path.basename(resultado),
                    'download_url': f'/download/{os.path.basename(resultado)}'
                })
            else:  # Datos devueltos directamente
                return jsonify({
                    'success': True,
                    'type': 'data',
                    'data': resultado
                })
                
        except Exception as e:
            # Limpio el archivo temporal si hay error
            if os.path.exists(tmp_input.name):
                os.unlink(tmp_input.name)
            
            error_msg = str(e)
            print(f"Error processing CSV: {error_msg}")
            print(traceback.format_exc())
            return jsonify({'error': error_msg}), 500

@app.route('/download/<filename>')
def download_file(filename):
    # Por seguridad, solo permito descargas del directorio temporal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(tempfile.gettempdir(), safe_filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    # Creo las carpetas necesarias si no existen
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

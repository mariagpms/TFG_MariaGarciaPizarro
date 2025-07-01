document.addEventListener('DOMContentLoaded', function() {
    const areaArrastre = document.getElementById('area-arrastre');
    const entradaArchivo = document.getElementById('entradaArchivo');
    const botonSeleccionarArchivo = document.getElementById('botonSeleccionarArchivo');
    const nombreArchivo = document.getElementById('nombre-archivo');
    const resultado = document.getElementById('resultado');
    const checkboxAnotado = document.getElementById('checkboxAnotado');
    const descargaArea = document.getElementById('descarga-area');
    const botonDescarga = document.getElementById('boton-descarga');
    const botonClasificar = document.getElementById('botonClasificar');
    
    let archivoSeleccionado = null;

    // Aquí evito el comportamiento por defecto del navegador cuando arrastra archivos
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(nombreEvento => {
        areaArrastre.addEventListener(nombreEvento, prevenirDefecto, false);
    });

    function prevenirDefecto(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Con esto resalto el área cuando el usuario arrastra un archivo encima
    ['dragenter', 'dragover'].forEach(nombreEvento => {
        areaArrastre.addEventListener(nombreEvento, resaltar, false);
    });

    ['dragleave', 'drop'].forEach(nombreEvento => {
        areaArrastre.addEventListener(nombreEvento, quitarResaltado, false);
    });

    function resaltar() {
        areaArrastre.classList.add('resaltado');
    }

    function quitarResaltado() {
        areaArrastre.classList.remove('resaltado');
    }

    areaArrastre.addEventListener('drop', manejarSoltar, false);

    function manejarSoltar(e) {
        const dt = e.dataTransfer;
        const archivos = dt.files;
        manejarArchivos(archivos);
    }

    // Aquí manejo el clic del botón para seleccionar archivos
    botonSeleccionarArchivo.addEventListener('click', () => {
        entradaArchivo.click();
    });

    entradaArchivo.addEventListener('change', function() {
        manejarArchivos(this.files);
    });

    function manejarArchivos(archivos) {
        if (archivos.length > 0) {
            const archivo = archivos[0];
            if (archivo.type === 'text/csv' || archivo.name.endsWith('.csv')) {
                nombreArchivo.textContent = `Archivo seleccionado: ${archivo.name}`;
                archivoSeleccionado = archivo;
                botonClasificar.disabled = false;
                botonClasificar.classList.add('habilitado');
                
                // Limpio los resultados de clasificaciones anteriores
                resultado.innerHTML = '<p>Archivo listo para clasificar. Haz clic en "Clasificar" para procesar.</p>';
                descargaArea.style.display = 'none';
            } else {
                nombreArchivo.textContent = 'Por favor, selecciona un archivo CSV.';
                archivoSeleccionado = null;
                botonClasificar.disabled = true;
                botonClasificar.classList.remove('habilitado');
            }
        }
    }

    // Aquí manejo el clic del botón clasificar
    botonClasificar.addEventListener('click', function() {
        if (archivoSeleccionado) {
            subirArchivo(archivoSeleccionado);
        }
    });

    function subirArchivo(archivo) {
        const datosFormulario = new FormData();
        datosFormulario.append('file', archivo);
        datosFormulario.append('anotado', checkboxAnotado.checked);

        resultado.innerHTML = '<p>Procesando clasificación... Esto puede tomar unos momentos.</p>';
        descargaArea.style.display = 'none';
        botonClasificar.disabled = true;
        botonClasificar.textContent = 'Procesando...';

        fetch('/process-csv', {
            method: 'POST',
            body: datosFormulario
        })
        .then(respuesta => respuesta.json())
        .then(datos => {
            if (datos.error) {
                resultado.innerHTML = `<p class="error">Error: ${datos.error}</p>`;
                descargaArea.style.display = 'none';
            } else if (datos.success) {
                if (datos.type === 'file') {
                    mostrarResultadoArchivo(datos);
                } else if (datos.type === 'file_with_metrics') {
                    mostrarResultadoArchivoConMetricas(datos);
                } else if (datos.type === 'data') {
                    mostrarResultadoDatos(datos.data);
                }
            }
        })
        .catch(error => {
            resultado.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            descargaArea.style.display = 'none';
        })
        .finally(() => {
            botonClasificar.disabled = false;
            botonClasificar.textContent = 'Clasificar';
        });
    }

    function mostrarResultadoArchivo(datos) {
        resultado.innerHTML = `
            <div class="resultado-exitoso">
                <p><strong>✅ Clasificación completada exitosamente</strong></p>
                <p>Archivo generado: <strong>${datos.filename}</strong></p>
                <p>El archivo contiene los datos originales con una nueva columna "Clases_modelo" con las clasificaciones.</p>
            </div>
        `;
        
        botonDescarga.onclick = () => {
            window.location.href = datos.download_url;
        };
        descargaArea.style.display = 'block';
    }

    function mostrarResultadoArchivoConMetricas(datos) {
        const metricas = datos.metrics;
        resultado.innerHTML = `
            <div class="resultado-exitoso">
                <p><strong>✅ Clasificación completada exitosamente</strong></p>
                <p>Archivo generado: <strong>${datos.filename}</strong></p>
                <p>Total de muestras procesadas: <strong>${metricas.total_samples}</strong></p>
                
                <h3>Métricas de Rendimiento:</h3>
                <table class="tabla-metricas">
                    <tr>
                        <th>Nivel</th>
                        <th>Accuracy</th>
                        <th>F-Score</th>
                    </tr>
                    <tr>
                        <td>Nivel 1</td>
                        <td>${(metricas.accuracy_nivel1 * 100).toFixed(2)}%</td>
                        <td>${(metricas.fscore_nivel1 * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Nivel 2</td>
                        <td>${(metricas.accuracy_nivel2 * 100).toFixed(2)}%</td>
                        <td>${(metricas.fscore_nivel2 * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td>Nivel 3 (Final)</td>
                        <td>${(metricas.accuracy_nivel3 * 100).toFixed(2)}%</td>
                        <td>${(metricas.fscore_nivel3 * 100).toFixed(2)}%</td>
                    </tr>
                </table>
                
                <p style="margin-top: 15px;">El archivo contiene los datos originales con una nueva columna "Clases_modelo" con las clasificaciones predichas.</p>
            </div>
        `;
        
        botonDescarga.onclick = () => {
            window.location.href = datos.download_url;
        };
        descargaArea.style.display = 'block';
    }

    function mostrarResultadoDatos(datos) {
        let html = '<div class="resultado-exitoso">';
        html += '<p><strong>✅ Clasificación completada</strong></p>';
        
        if (typeof datos === 'object' && datos !== null) {
            html += '<table>';
            html += '<tr><th>Campo</th><th>Valor</th></tr>';
            
            for (const [clave, valor] of Object.entries(datos)) {
                html += `<tr><td>${clave}</td><td>${valor}</td></tr>`;
            }
            html += '</table>';
        } else {
            html += `<pre>${JSON.stringify(datos, null, 2)}</pre>`;
        }
        
        html += '</div>';
        resultado.innerHTML = html;
        descargaArea.style.display = 'none';
    }
});

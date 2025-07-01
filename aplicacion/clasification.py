from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import torch
import os

#Clase MLP una capa oculta
class MLP(nn.Module):
    def __init__(self, a,b, n_clases):
        super().__init__()
        self.Linear1=nn.Linear(39,a)
        self.Linear2=nn.Linear(a, b)
        self.output=nn.Linear(b, n_clases)
        self.relu=nn.LeakyReLU()
    def forward(self, x):
        xx=self.relu(self.Linear1(x))
        xx=self.relu(self.Linear2(xx))
        return self.output(xx)
        
#Clase MLP dos capas ocultas
class MLP2(nn.Module):
    def __init__(self, a,b, c,n_clases):
        super().__init__()
        self.Linear1=nn.Linear(39,a)
        self.Linear2=nn.Linear(a, b)
        self.Linear3=nn.Linear(b,c)
        self.output=nn.Linear(c, n_clases)
        self.relu=nn.LeakyReLU()
    def forward(self, x):
        xx=self.relu(self.Linear1(x))
        xx=self.relu(self.Linear2(xx))
        xx=self.relu(self.Linear3(xx))
        return self.output(xx)

#Clase modelo completo
class ModeloCompleto:
    def __init__(self, mod_nivel0, mod_nivel1, mod_nivel2, rutas_validas, equivalencias):
        self.modelo_nivel0=mod_nivel0
        self.modelo_nivel1=mod_nivel1
        self.modelo_nivel2=mod_nivel2
        self.rutas=rutas_validas
        self.traduccion=equivalencias
    def predict(self,X):
        with torch.no_grad():
            #nivel 0
            pred_n0=self.modelo_nivel0(X).argmax(dim=1)
    
            #Nivel 1
            pred_n1=np.full(pred_n0.shape, -1)
            clases_n0=np.unique(pred_n0)
            for clase in clases_n0:
                idx_clase=np.where(pred_n0.numpy()==clase)[0]
                X_sub=X[idx_clase]
    
                modelo_n1=self.modelo_nivel1.get(clase)
                if modelo_n1 is None:
                    raise ValueError(f"No hay modelo para clase nivel 0 '{clase}'")
                pred_clase=modelo_n1(X_sub).argmax(dim=1)
                pred_n1[idx_clase]=pred_clase
    
            #Nivel 2
            pred_n2=np.full(pred_n0.shape, -1)
            
            for ruta in self.rutas:
                clase_n0, clase_n1 = ruta
                idxs=np.where((pred_n0.numpy()==clase_n0)&(pred_n1==clase_n1))[0]
                X_sub=X[idxs]
    
                modelo_n2=self.modelo_nivel2.get(ruta)
                if modelo_n2 is None:
                    raise ValueError(f"No hay modelo de nivel 2 para la ruta '{ruta}'")
                pred_clase=modelo_n2(X_sub).argmax(dim=1)
                pred_n2[idxs]=pred_clase
            
        return [pred_n0, pred_n1, pred_n2] 
    def predict_celulas(self, X):
        pred1, pred2, pred3=self.predict(X)
        pred=np.char.add(pred1.numpy().astype(str), pred2.astype(str))
        pred=np.char.add(pred.astype(str), pred3.astype(str))
        print(f"Predicciones combinadas: {np.unique(pred)}")
        pred_cells=np.array([self.traduccion.get(clave) for clave in pred])
        return pred_cells
    def accuracy_niveles(self, X, y1, y2, y3):
        y1_pred, y2_pred, y3_pred = self.predict(X)
        
        # Convertir a numpy si necesario
        y1_pred_np = y1_pred.numpy()
        y1_np = y1.numpy() if isinstance(y1, torch.Tensor) else y1
        y2_np = y2.numpy() if isinstance(y2, torch.Tensor) else y2
        y3_np = y3.numpy() if isinstance(y3, torch.Tensor) else y3
        
        acc1 = np.mean(y1_pred_np == y1_np)
        acc2 = np.mean((y1_pred_np == y1_np) & (y2_pred == y2_np))
        acc3 = np.mean((y1_pred_np == y1_np) & (y2_pred == y2_np) & (y3_pred == y3_np))
        return [acc1, acc2, acc3]
    
    def fscore_niveles(self, X, y1, y2, y3):
        y1_pred, y2_pred, y3_pred = self.predict(X)
        
        # pasar de tensor a numpy para comarar
        y1_pred_np = y1_pred.numpy()
        y1_np = y1.numpy() if isinstance(y1, torch.Tensor) else y1
        y2_np = y2.numpy() if isinstance(y2, torch.Tensor) else y2
        y3_np = y3.numpy() if isinstance(y3, torch.Tensor) else y3
        
        fscore1 = f1_score(y1_np, y1_pred_np, average="weighted")
        
        y12_pred = np.char.add(y1_pred_np.astype(str), y2_pred.astype(str))
        y12_true = np.char.add(y1_np.astype(str), y2_np.astype(str))       
        fscore2 = f1_score(y12_true, y12_pred, average="weighted")
        
        y123_pred = np.char.add(y12_pred.astype(str), y3_pred.astype(str))
        y123_true = np.char.add(y12_true.astype(str), y3_np.astype(str))       
        fscore3 = f1_score(y123_true, y123_pred, average="weighted")
        
        return [fscore1, fscore2, fscore3]

def crear_modelo():
    #General
    model_general=MLP(256,64,3)
    state_dict_general=torch.load("models/general_256_64_3.pth", map_location=torch.device("cpu"))
    model_general.load_state_dict(state_dict_general)
    model_general.eval()
    #b cells
    model_bcells=MLP(128,32,3)
    state_dict_bcells=torch.load("models/bcells_128_32_3.pth", map_location=torch.device("cpu"))
    model_bcells.load_state_dict(state_dict_bcells)
    model_bcells.eval()
    #t cells
    model_tcells=MLP(64,16,3)
    state_dict_tcells=torch.load("models/tcells_64_16_3.pth", map_location=torch.device("cpu"))
    model_tcells.load_state_dict(state_dict_tcells)
    model_tcells.eval()
    #No T & No B
    model_nono=MLP(512,128, 4)
    state_dict_nono=torch.load("models/nono_512_128_4.pth", map_location=torch.device("cpu"))
    model_nono.load_state_dict(state_dict_nono)
    model_nono.eval()
    #IgD-
    model_igd=MLP(128,32,2)
    state_dict_igd=torch.load("models/igd_128_32_2.pth", map_location=torch.device("cpu"))
    model_igd.load_state_dict(state_dict_igd)
    model_igd.eval()
    #NKT
    model_nkt=MLP(512, 256, 5)
    state_dict_nkt=torch.load("models/nkt_512_256_5.pth", map_location=torch.device("cpu"))
    model_nkt.load_state_dict(state_dict_nkt)
    model_nkt.eval()
    #Tgd
    model_tgd=MLP2(512, 256, 64, 6)
    state_dict_tgd=torch.load("models/tgd_512_256_64_6.pth", map_location=torch.device("cpu"))
    model_tgd.load_state_dict(state_dict_tgd)
    model_tgd.eval()
    #real T cells
    model_real=MLP(64, 8, 4)
    state_dict_real=torch.load("models/real_64_8_4.pth", map_location=torch.device("cpu"))
    model_real.load_state_dict(state_dict_real)
    model_real.eval()
    #NK
    model_nk=MLP(32, 8, 4)
    state_dict_nk=torch.load("models/nk_32_8_4.pth", map_location=torch.device("cpu"))
    model_nk.load_state_dict(state_dict_nk)
    model_nk.eval()
    #APC
    model_apc=MLP(128, 32, 4)
    state_dict_apc=torch.load("models/apc_128_32_4.pth", map_location=torch.device("cpu"))
    model_apc.load_state_dict(state_dict_apc)
    model_apc.eval()
    
    modelo_completo=ModeloCompleto(
        mod_nivel0=model_general,
        mod_nivel1={
            0: model_bcells,
            1: model_tcells,
            2: model_nono
        },
        mod_nivel2={
            (0, 2): model_igd,
            (1, 0): model_nkt,
            (1, 1): model_tgd,
            (1, 2): model_real,
            (2, 0): model_nk,
            (2, 1): model_apc
        },
        rutas_validas=[(0,2), (1,0), (1,1), (1,2), (2,0), (2,1)],
        equivalencias={
            '00-1': "B cells",
            '01-1': "IgD+",
            '020': "Plasmablasts",
            '021': "Memory B cells",
            '100': "NKT",
            '101': "CD2-",
            '102': "CD2+ CD8high",
            '103': "CD2+CD8dim",
            '104': "CD2+ CD8-",
            '110': "Tgd",
            '111': "Tgd/Trm",
            '112': "Tgd/Memory",
            '113': "Tgd/CD45RA+/naive",
            '114': "Tgd/CD45RA+/TEMRA/rTEMRA",
            '115': "Tgd/CD45RA+/TEMRA/nrTEMRA",
            '120': "Real T cells",
            '121': "CD4",
            '122': "CD8",
            '123': "DP",
            '200': "DR-",
            '201': "early NK",
            '202': "mature NK",
            '203': "terminal NK",
            '210': "APC",
            '211': "macrophages",
            '212': "monocytes",
            '213': "cDC",
            '22-1': "ILC",
            '23-1': "basofilos"
        }
    )

    return modelo_completo

def asignar_niveles(etiqueta):
    bcells=['Memory B cells/IgG','Memory B cells/IgA','Memory B cells','Plasmablasts','Plasmablasts/IgA','Plasmablasts/IgG','IgD+','Memory B cells/IgG- IgA-/IgM', 'Plasmablasts/IgG- IgA-/IgM', 'B cells']
    igd=['Memory B cells/IgG','Memory B cells/IgA','Memory B cells','Plasmablasts','Plasmablasts/IgA','Plasmablasts/IgG','Memory B cells/IgG- IgA-/IgM', 'Plasmablasts/IgG- IgA-/IgM']
    plasmablasts=['Plasmablasts','Plasmablasts/IgA','Plasmablasts/IgG', 'Plasmablasts/IgG- IgA-/IgM']
    memoryb=['Memory B cells/IgG','Memory B cells/IgA','Memory B cells','Memory B cells/IgG- IgA-/IgM']
    tcells=['CD8/Trm', 'Trm/Th17', 'Memory/Th1','Trm/Th1/Th17-like','Trm/Th1','CD8/CD45RA+/TEMRA/rTEMRA','Memory/Th1/Th17-like', 'CD8/Memory','Real T cells','CD2-','DP','CD4/CD45RA+/naive','Trm/Th2', 'Tgd/Memory', 
        'CD4/CD45RA+/TEMRA/nrTEMRA','CD8/CD45RA+/TEMRA/nrTEMRA', 'CD8/CD45RA+/naive','NKT', 'Memory/Th2', 'CD8','Memory/Th17','Tgd/CD45RA+/naive', 'CD4/CD45RA+/TEMRA/rTEMRA','Tgd/CD45RA+/TEMRA/rTEMRA', 'Tgd',
        'CD2+ CD8high','Tgd/CD45RA+/TEMRA/nrTEMRA', 'Tgd/Trm', 'CD2+CD8dim', 'CD2+ CD8-','CD4']
    nkt=['CD2-','NKT', 'CD2+ CD8high','CD2+CD8dim', 'CD2+ CD8-']
    tgd=['Tgd/Memory','Tgd/CD45RA+/naive','Tgd/CD45RA+/TEMRA/rTEMRA', 'Tgd','Tgd/CD45RA+/TEMRA/nrTEMRA', 'Tgd/Trm']
    realtcells=['CD8/Trm', 'Trm/Th17', 'Memory/Th1','Trm/Th1/Th17-like','Trm/Th1','CD8/CD45RA+/TEMRA/rTEMRA','Memory/Th1/Th17-like', 'CD8/Memory','Real T cells','Trm/Th2','DP','CD4/CD45RA+/naive',
                'CD4/CD45RA+/TEMRA/nrTEMRA','CD8/CD45RA+/TEMRA/nrTEMRA', 'CD8/CD45RA+/naive', 'Memory/Th2', 'CD8','Memory/Th17','CD4/CD45RA+/TEMRA/rTEMRA','CD4']
    cd4=['Trm/Th17', 'Memory/Th1','Trm/Th1/Th17-like','Trm/Th1','Memory/Th1/Th17-like', 'Trm/Th2','CD4/CD45RA+/naive',
            'CD4/CD45RA+/TEMRA/nrTEMRA','Memory/Th2','Memory/Th17','CD4/CD45RA+/TEMRA/rTEMRA','CD4']
    cd8=['CD8/Trm', 'CD8/CD45RA+/TEMRA/rTEMRA','CD8/Memory','CD8/CD45RA+/TEMRA/nrTEMRA', 'CD8/CD45RA+/naive','CD8']
    noTnoB=['DR-','monocytes','basofilos','cDC','terminal NK','mature NK','early NK','APC', 'macrophages','ILC']
    nkcells=['DR-','terminal NK','mature NK','early NK']
    apc=['monocytes','cDC','APC', 'macrophages']
    
    #Nivel 1
    if etiqueta in bcells:
        n1=0
    elif etiqueta in tcells:
        n1=1
    elif etiqueta in noTnoB:
        n1=2
    else:
        n1=-1

    #Nivel 2
    if etiqueta=="B cells":
        n2=0
    elif etiqueta =="IgD+":
        n2=1
    elif etiqueta in igd:
        n2=2
    elif etiqueta in nkt:
        n2=0
    elif etiqueta in tgd:
        n2=1
    elif etiqueta in realtcells:
        n2=2
    elif etiqueta in nkcells:
        n2=0
    elif etiqueta in apc:
        n2=1
    elif etiqueta=="ILC":
        n2=2
    elif etiqueta=="basofilos":
        n2=3
    else:
        n2=-1

    #Nivel 3
    if etiqueta in plasmablasts:
        n3=0
    elif etiqueta in memoryb:
        n3=1
    elif etiqueta=="NKT":
        n3=0
    elif etiqueta=='CD2-':
        n3=1
    elif etiqueta=='CD2+ CD8high':
        n3=2
    elif etiqueta=='CD2+CD8dim':
        n3=3
    elif etiqueta=='CD2+ CD8-':
        n3=4
    elif etiqueta=="Tgd":
        n3=0
    elif etiqueta=='Tgd/Trm':
        n3=1  
    elif etiqueta=='Tgd/Memory':
        n3=2
    elif etiqueta=='Tgd/CD45RA+/naive':
        n3=3
    elif etiqueta=='Tgd/CD45RA+/TEMRA/rTEMRA':
        n3=4
    elif etiqueta=='Tgd/CD45RA+/TEMRA/nrTEMRA':
        n3=5
    elif etiqueta=="Real T cells":
        n3=0
    elif etiqueta in cd4:
        n3=1
    elif etiqueta in cd8:
        n3=2
    elif etiqueta=="DP":
        n3=3
    elif etiqueta=="DR-":
        n3=0  
    elif etiqueta=='early NK':
        n3=1
    elif etiqueta=='mature NK':
        n3=2
    elif etiqueta=='terminal NK':
        n3=3
    elif etiqueta=="APC":
        n3=0
    elif etiqueta=='macrophages':
        n3=1
    elif etiqueta=='monocytes':
        n3=2
    elif etiqueta=='cDC':
        n3=3
    else:
        n3=-1    

    return [n1,n2,n3]

def comprobar_datos(data, anotado):
    esperado= 40 if anotado else 39
    ncols=data.shape[1]
    if ncols !=esperado:
        raise ValueError(
            f"Se esperaban {esperado} columnas (anotado={anotado}), pero se encontraron {ncols}."
        )

def comprobar_csv(filepath):
    if not filepath.lower().endswith(".csv"):
        raise ValueError(f"El archivo '{filepath}' no tiene extensión .csv")

    try:
        data=pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo '{filepath}' como CSV. Detalle: {e}")

    return data
    
def main(input_file, anotado):
    data = comprobar_csv(input_file)
    comprobar_datos(data, anotado)

    modelo = crear_modelo()
    if anotado:
        X = torch.tensor(data.iloc[:,:-1].values, dtype=torch.float32)
        
        # Debug: Verificar los datos de entrada
        print(f"Datos de entrada - Filas: {data.shape[0]}, Columnas: {data.shape[1]}")
        print(f"Primeras 5 filas de X:\n{X[:5]}")
        print(f"Estadísticas de X - Min: {X.min().item():.4f}, Max: {X.max().item():.4f}, Mean: {X.mean().item():.4f}")
        
        # Verificar las etiquetas originales
        etiquetas_originales = data.iloc[:,-1].values
        print(f"Primeras 10 etiquetas originales: {etiquetas_originales[:10]}")
        print(f"Etiquetas únicas en el archivo: {np.unique(etiquetas_originales)}")
        
        result = np.array([asignar_niveles(x) for x in etiquetas_originales])
        
        # Debug: Verificar la conversión de etiquetas
        print(f"Result shape: {result.shape}")
        print(f"Primeras 10 conversiones de niveles: {result[:10]}")
        
        # Verificar si hay etiquetas no reconocidas (-1)
        unique_n1 = np.unique(result[:, 0])
        unique_n2 = np.unique(result[:, 1])
        unique_n3 = np.unique(result[:, 2])
        print(f"Valores únicos nivel 1: {unique_n1}")
        print(f"Valores únicos nivel 2: {unique_n2}")
        print(f"Valores únicos nivel 3: {unique_n3}")
        
        # Contar etiquetas no reconocidas
        no_reconocidas_n1 = np.sum(result[:, 0] == -1)
        no_reconocidas_n2 = np.sum(result[:, 1] == -1)
        no_reconocidas_n3 = np.sum(result[:, 2] == -1)
        print(f"Etiquetas no reconocidas - N1: {no_reconocidas_n1}, N2: {no_reconocidas_n2}, N3: {no_reconocidas_n3}")
        
        # verificar si result es como debe
        if len(result.shape) == 1 and len(result) == 3:
            result = result.reshape(1, -1)
        elif len(result.shape) == 1:
            n_samples = len(result) // 3
            result = result.reshape(n_samples, 3)
        
        y1 = result[:, 0]
        y2 = result[:, 1] 
        y3 = result[:, 2]
        
        # Convertir a tensors
        y1 = torch.tensor(y1, dtype=torch.long)
        y2 = torch.tensor(y2, dtype=torch.long)
        y3 = torch.tensor(y3, dtype=torch.long)
        
        # Debug: Verificar las predicciones del modelo
        try:
            y_pred_raw = modelo.predict(X)
            print(f"Predicciones brutas - Nivel 1: {np.unique(y_pred_raw[0].numpy())}")
            print(f"Predicciones brutas - Nivel 2: {np.unique(y_pred_raw[1])}")
            print(f"Predicciones brutas - Nivel 3: {np.unique(y_pred_raw[2])}")
            
            acc1, acc2, accuracy = modelo.accuracy_niveles(X, y1, y2, y3)
            fscore1, fscore2, fscore = modelo.fscore_niveles(X, y1, y2, y3)
            y_pred = modelo.predict_celulas(X)
            
            # Debug: Mostrar algunas predicciones vs etiquetas reales
            print(f"\nComparación de primeras 10 muestras:")
            for i in range(min(10, len(etiquetas_originales))):
                print(f"Real: {etiquetas_originales[i]} -> Predicho: {y_pred[i]}")
            
            print(f"\nMétricas calculadas:")
            print(f"Accuracy N1: {acc1:.4f}, N2: {acc2:.4f}, N3: {accuracy:.4f}")
            print(f"F-Score N1: {fscore1:.4f}, N2: {fscore2:.4f}, N3: {fscore:.4f}")
            
            # anadir columna de clases al DataFrame
            data["Clases_modelo"] = y_pred
            
            # fichero tmaporal
            import tempfile
            output_fd, output_path = tempfile.mkstemp(suffix='_classified.csv')
            os.close(output_fd)
            data.to_csv(output_path, index=False)
            
            # devuelve path fichero y metricas
            return {
                'file_path': output_path,
                'metrics': {
                    'accuracy_nivel1': round(float(acc1), 4),
                    'accuracy_nivel2': round(float(acc2), 4), 
                    'accuracy_nivel3': round(float(accuracy), 4),
                    'fscore_nivel1': round(float(fscore1), 4),
                    'fscore_nivel2': round(float(fscore2), 4),
                    'fscore_nivel3': round(float(fscore), 4),
                    'total_samples': len(data)
                }
            }
            
        except Exception as e:
            print(f"Error in metrics calculation: {e}")
            print(f"y1 shape: {y1.shape}, y2 shape: {y2.shape}, y3 shape: {y3.shape}")
            import traceback
            traceback.print_exc()
            raise e
            
    else:
        print("Procesando archivo no anotado...")
        X = torch.tensor(data.values, dtype=torch.float32)
        y_pred = modelo.predict_celulas(X)
        
        # anadir columna de clases al DataFrame
        data["Clases_modelo"] = y_pred
        
        # fichero tmeporal
        import tempfile
        output_fd, output_path = tempfile.mkstemp(suffix='_classified.csv')
        os.close(output_fd)
        data.to_csv(output_path, index=False)
        
        return output_path

def procesar_archivo_csv(archivo_path, es_anotado=False):
    return main(archivo_path, es_anotado)
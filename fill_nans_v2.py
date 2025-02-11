import numpy as np 
import pandas as pd
import random
import plotly.express as px

def interpolate_with_rules(
    df: pd.DataFrame,
    target_column: str,
    min_value: float,
    max_value: float,
    noise: float,
    max_step: float,
    other_weight: float,
    seed: int = 42,
    other_columns: list[str] = None,
    fill_mode: str = "linear",  # Puede ser "linear", "exponential" o "sigmoidal"
    cumm: bool = False  # Si True, se impone la restricción acumulativa.
) -> pd.DataFrame:
    """
    Interpola los valores NaN en la columna 'target_column' de un DataFrame, combinando la interpolación
    (lineal, exponencial o sigmoidal) con información derivada de otras columnas.
    
    Para cada bloque consecutivo de NaN en 'target_column', se realiza:
      1) Se identifica el límite izquierdo (L) y el límite derecho (R). Si no hay valor anterior se usa min_value;
         si no hay valor siguiente se usa max_value.
      2) Se interpola entre L y R utilizando el modo especificado en 'fill_mode':
             - "linear": interpolación lineal.
             - "exponential": interpolación exponencial.
             - "sigmoidal": interpolación sigmoidal.
         (Para bloques que inician en el índice 0, se interpola de derecha a izquierda).
      3) A cada valor interpolado se le añade un ruido uniforme en el rango [-noise, noise].
      4) Se obtiene un "valor de otras columnas" para la misma fila, calculado como la media de los valores
         de las columnas indicadas (ignorando NaN). Si no se proporcionan o no hay datos válidos, se usa el valor base.
      5) Se combina el valor interpolado con el obtenido de las otras columnas usando:
             valor_final = (1 - other_weight) * valor_interpolado + other_weight * valor_otras
      6) Se fuerza que el valor resultante esté entre [min_value, max_value] y que la diferencia
         con el valor anterior (o siguiente) no exceda max_step. Además, si 'cumm' es True se impone que:
            - En el caso de rellenar de izquierda a derecha, el nuevo valor no sea inferior al anterior.
            - En el caso de rellenar de derecha a izquierda, el nuevo valor no sea mayor que el valor a su derecha.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna objetivo y las demás columnas que se usarán para complementar la interpolación.
    target_column : str
        Nombre de la columna objetivo que se interpolará.
    min_value : float
        Valor mínimo permitido (y usado en la frontera izquierda si es necesario).
    max_value : float
        Valor máximo permitido (y usado en la frontera derecha si es necesario).
    noise : float
        Magnitud del ruido (uniforme en [-noise, noise]) a añadir a cada valor interpolado.
    max_step : float
        Diferencia máxima permitida entre un valor interpolado y el anterior (o siguiente) ya conocido.
    other_weight : float
        Peso (en el rango [0, 1]) de la influencia de la información proveniente de las otras columnas.
    seed : int, opcional
        Semilla para el generador de números aleatorios.
    other_columns : list[str], opcional
        Lista de columnas a utilizar para complementar la interpolación. Si es None, se usan todas las columnas
        excepto la columna objetivo.
    fill_mode : str, opcional
        Modo de interpolación: "linear", "exponential" o "sigmoidal".
    cumm : bool, opcional
        Si es True, se impone que:
         - En el modo de rellenar de izquierda a derecha, cada nuevo valor no sea inferior al anterior.
         - En el modo de rellenar de derecha a izquierda, cada nuevo valor no sea mayor que el valor a su derecha.
         Si es False, no se aplica esta restricción.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la columna 'target_column' interpolada según las reglas descritas.
    """
    # Inicializa la semilla para reproducibilidad
    random.seed(seed)
    
    # Trabajamos sobre una copia para no modificar el DataFrame original
    df_new = df.copy()
    arr = df_new[target_column].to_numpy(dtype=float, copy=True)
    n = len(arr)
    
    # Si no se han especificado otras columnas, se usan todas las columnas excepto target_column.
    if other_columns is None:
        other_columns = [col for col in df_new.columns if col != target_column]
    
    def clamp_and_step(value, previous_value, reverse=False):
        """
        Ajusta 'value' para que:
          - La diferencia con 'previous_value' no exceda 'max_step' (en función del sentido de iteración).
          - Si 'cumm' es True, se impone que:
               * En modo normal (reverse=False): 'value' no sea inferior a 'previous_value'.
               * En modo inverso (reverse=True): 'value' no sea mayor que 'previous_value'.
          - Se mantenga dentro del rango [min_value, max_value].
        """
        if reverse:
            # Relleno de derecha a izquierda.
            diff = previous_value - value  # Se desea que value sea como máximo igual a previous_value.
            if cumm:
                if diff < 0:  # value > previous_value.
                    value = previous_value
                elif diff > max_step:
                    value = previous_value - max_step
            else:
                if diff > max_step:
                    value = previous_value - max_step
                elif diff < -max_step:
                    value = previous_value + max_step
        else:
            # Relleno de izquierda a derecha.
            diff = value - previous_value
            if cumm:
                if diff < 0:
                    value = previous_value
                elif diff > max_step:
                    value = previous_value + max_step
            else:
                if diff > max_step:
                    value = previous_value + max_step
                elif diff < -max_step:
                    value = previous_value - max_step
        
        if value < min_value:
            value = min_value
        elif value > max_value:
            value = max_value
        return value
    
    def compute_base_value(L, R, frac, mode, reverse=False):
        """
        Calcula el valor base interpolado entre L y R dado una fracción 'frac' y el modo de interpolación.
        Si reverse es True, se interpola de derecha a izquierda.
        """
        # Para el modo sigmoidal se utiliza una función logística normalizada.
        # Usamos una constante k que controla la pendiente (se ha elegido k=10).
        k = 10
        if reverse:
            diff = L - R
            if mode == "linear":
                return R + diff * frac
            elif mode == "exponential":
                return R + diff * ((np.exp(frac) - 1) / (np.e - 1))
            elif mode == "sigmoidal":
                S = 1 / (1 + np.exp(-k * (frac - 0.5)))
                S0 = 1 / (1 + np.exp(-k * (0 - 0.5)))
                S1 = 1 / (1 + np.exp(-k * (1 - 0.5)))
                norm_frac = (S - S0) / (S1 - S0)
                return R + diff * norm_frac
            else:
                raise ValueError(f"fill_mode '{mode}' no soportado.")
        else:
            diff = R - L
            if mode == "linear":
                return L + diff * frac
            elif mode == "exponential":
                return L + diff * ((np.exp(frac) - 1) / (np.e - 1))
            elif mode == "sigmoidal":
                S = 1 / (1 + np.exp(-k * (frac - 0.5)))
                S0 = 1 / (1 + np.exp(-k * (0 - 0.5)))
                S1 = 1 / (1 + np.exp(-k * (1 - 0.5)))
                norm_frac = (S - S0) / (S1 - S0)
                return L + diff * norm_frac
            else:
                raise ValueError(f"fill_mode '{mode}' no soportado.")
    
    i = 0
    while i < n:
        # Si el valor ya está definido (no es NaN), lo dejamos y pasamos al siguiente
        if not np.isnan(arr[i]):
            i += 1
            continue
        
        # Determinar el bloque de NaN (desde 'start' hasta 'end')
        start = i
        while i < n and np.isnan(arr[i]):
            i += 1
        end = i - 1
        size = end - start + 1
        
        # Definir los límites L y R usando los valores conocidos (no NaN)
        if start > 0 and not np.isnan(arr[start - 1]):
            L = arr[start - 1]
        else:
            L = min_value
        if end < (n - 1) and not np.isnan(arr[end + 1]):
            R = arr[end + 1]
        else:
            R = max_value
        
        # Si el bloque inicia en el índice 0, se interpola de derecha a izquierda
        if start == 0:
            step_count = size + 1
            previous_value = R  # Empezamos desde el límite derecho
            for offset in range(size):
                idx = end - offset
                frac = (offset + 1) / step_count
                base_value = compute_base_value(L, R, frac, fill_mode, reverse=True)
                base_value += random.uniform(-noise, noise)
                
                # Valor derivado de las columnas adicionales para la fila actual
                row_other = df_new.loc[df_new.index[idx], other_columns]
                if row_other.dropna().empty:
                    other_value = base_value
                else:
                    other_value = row_other.mean()
                
                combined_value = (1 - other_weight) * base_value + other_weight * other_value
                final_value = clamp_and_step(combined_value, previous_value, reverse=True)
                arr[idx] = final_value
                previous_value = final_value
        else:
            # Para bloques que no inician en el índice 0, se interpola de izquierda a derecha
            step_count = size + 1
            previous_value = L  # Empezamos desde el límite izquierdo
            for offset in range(size):
                idx = start + offset
                frac = (offset + 1) / step_count
                base_value = compute_base_value(L, R, frac, fill_mode, reverse=False)
                base_value += random.uniform(-noise, noise)
                
                # Valor derivado de las columnas adicionales para la fila actual
                row_other = df_new.loc[df_new.index[idx], other_columns]
                if row_other.dropna().empty:
                    other_value = base_value
                else:
                    other_value = row_other.mean()
                
                combined_value = (1 - other_weight) * base_value + other_weight * other_value
                final_value = clamp_and_step(combined_value, previous_value, reverse=False)
                arr[idx] = final_value
                previous_value = final_value
    
    # Se actualiza la columna objetivo en el DataFrame
    df_new[target_column] = arr
    return df_new

def test_interpolation(data = None, target_column = "target", min_value = 0, max_value = 1, noise = 0, 
                       max_step = 0.2, other_weight = 0.5, other_columns = None, seed = 42,
                       fill_mode="linear", cumm=False):
    """
    Función de prueba para visualizar la interpolación en un DataFrame de ejemplo.
    Permite especificar el modo de interpolación (fill_mode) y si se impone la restricción acumulativa (cumm).
    """
    # Creamos un DataFrame de ejemplo con una columna objetivo y dos columnas adicionales.
    if data is None:
        data = {
            "target": [np.nan, np.nan, 0.5, 0.65, 0.65, np.nan, np.nan, np.nan, 0.8, np.nan],
            "other1": [0.1, 0.2, 0.3, 0.4, np.nan, 0.6, 0.7, 0.8, 0.8, 0.75],
            "other2": [0.2, 0.35, 0.4, np.nan, 0.6, 1, 1, 0.6, 0.8, 0.8]
        }
        data = pd.DataFrame(data)
    
    # Interpolación sin influencia de otras columnas (other_weight = 0)
    df_no_other = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,  # Sin influencia de otras columnas
        seed=seed,
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    other_weight_2 = 0.5
    # Interpolación con influencia de todas las columnas (excepto target)
    df_with_other = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight_2,
        seed=seed,
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    # Interpolación utilizando solo la columna "other1"
    df_with_other_col_1 = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        other_columns=[data.columns[1]],
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    # Interpolación utilizando solo la columna "other2"
    df_with_other_col_2 = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        other_columns=[data.columns[2]],
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    # Interpolación con restricción acumulativa
    df_cumm = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode=fill_mode,
        cumm=True
    )
    
    # Interpolación exponencial con restricción acumulativa activada
    df_exp = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode="exponential",
        cumm=True
    )
    
    # Interpolación sigmoidal con restricción acumulativa activada
    df_sigmoidal = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode="sigmoidal",
        cumm=True
    )
    
    # Graficamos los resultados
    fig = px.line(data[target_column], markers=True, title="Interpolación de valores NaN", height=800)
    fig.update_traces(marker=dict(symbol='x', size=20), line=dict(width=8))
    
    # Añadimos las otras columnas del DataFrame original
    for col in data.columns[1:]:
        fig.add_scatter(x=data.index, y=data[col], mode='lines+markers', name=col)
    
    # Añadimos las líneas de las interpolaciones
    fig.add_scatter(x=df_no_other.index, y=df_no_other[target_column], mode='lines+markers',
                    name='Sin influencia de otras columnas')
    fig.add_scatter(x=df_with_other.index, y=df_with_other[target_column], mode='lines+markers',
                    name=f'Con influencia de otras columnas (α={other_weight_2})')
    fig.add_scatter(x=df_with_other_col_1.index, y=df_with_other_col_1[target_column], mode='lines+markers',
                    name=f'Con influencia de {[data.columns[1]]} (α={other_weight})')
    fig.add_scatter(x=df_with_other_col_2.index, y=df_with_other_col_2[target_column], mode='lines+markers',
                    name=f'Con influencia de {[data.columns[2]]} (α={other_weight})')
    fig.add_scatter(x=df_cumm.index, y=df_cumm[target_column], mode='lines+markers',
                    name='Con restricción acumulativa')
    fig.add_scatter(x=df_exp.index, y=df_exp[target_column], mode='lines+markers',
                    name='Interpolación exponencial (con restricción acumulativa)')
    fig.add_scatter(x=df_sigmoidal.index, y=df_sigmoidal[target_column], mode='lines+markers',
                    name='Interpolación sigmoidal (con restricción acumulativa)')
    
    # Mostramos la gráfica
    fig.show()


# Ejemplo de uso:
if __name__ == "__main__":
    # Creamos un DataFrame de ejemplo
    data = {
        "target": [0.0, 0.05, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan, 0.4, 0.5,
                   0.52, 0.5, np.nan, np.nan, 0.7, 0.71, 0.73, np.nan, np.nan, np.nan, np.nan, np.nan],
        "other1": [0.1, 0.1, 0.1, 0.11, 0.2, 0.22, 0.22, 0.25, 0.28, 0.3,
                   0.33, 0.37, 0.4, 0.45, 0.5, 0.55, 0.55, 0.65, 0.66, 0.7, 0.75, 0.75],
        "other2": [0.0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64, 0.72,
                   0.8, 0.84, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0]
    }
    data = pd.DataFrame(data)

    ##### CONFIGURACIÓN #####
    target_column = "target"
    min_value = 0
    max_value = 1
    noise = 0       
    max_step = 0.2
    other_weight = 0.2
    other_columns = None
    seed = 42
    fill_mode="linear"  # También puede ser "exponential" o "sigmoidal"
    cumm=False
    #########################
    
    df_no_other = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    other_weight_2 = 0.5
    df_with_other = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight_2,
        seed=seed,
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    df_with_other_col_1 = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        other_columns=[data.columns[1]],
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    df_with_other_col_2 = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        other_columns=[data.columns[2]],
        fill_mode=fill_mode,
        cumm=cumm
    )
    
    df_cumm = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode=fill_mode,
        cumm=True
    )
    
    df_exp = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode="exponential",
        cumm=True
    )
    
    df_sigmoidal = interpolate_with_rules(
        data,
        target_column=target_column,
        min_value=min_value,
        max_value=max_value,
        noise=noise,
        max_step=max_step,
        other_weight=other_weight,
        seed=seed,
        fill_mode="sigmoidal",
        cumm=True
    )
    
    fig = px.line(data["target"], markers=True, title="Interpolación de valores NaN", height=800)
    fig.update_traces(marker=dict(symbol='x', size=20), line=dict(width=8))
    
    for col in data.columns[1:]:
        fig.add_scatter(x=data.index, y=data[col], mode='lines+markers', name=col)
    
    fig.add_scatter(x=df_no_other.index, y=df_no_other["target"], mode='lines+markers',
                    name='Sin influencia de otras columnas')
    fig.add_scatter(x=df_with_other.index, y=df_with_other["target"], mode='lines+markers',
                    name=f'Con influencia de otras columnas (α={other_weight_2})')
    fig.add_scatter(x=df_with_other_col_1.index, y=df_with_other_col_1["target"], mode='lines+markers',
                    name=f'Con influencia de {[data.columns[1]]} (α={other_weight})')
    fig.add_scatter(x=df_with_other_col_2.index, y=df_with_other_col_2["target"], mode='lines+markers',
                    name=f'Con influencia de {[data.columns[2]]} (α={other_weight})')
    fig.add_scatter(x=df_cumm.index, y=df_cumm["target"], mode='lines+markers',
                    name='Con restricción acumulativa')
    fig.add_scatter(x=df_exp.index, y=df_exp["target"], mode='lines+markers',
                    name='Interpolación exponencial (con restricción acumulativa)')
    fig.add_scatter(x=df_sigmoidal.index, y=df_sigmoidal["target"], mode='lines+markers',
                    name='Interpolación sigmoidal (con restricción acumulativa)')
    
    fig.show()

__all__ = ["interpolate_with_rules_v4", "test_interpolation"]

import numpy as np
import pandas as pd
import random

def interpolate_with_rules(
    s: pd.Series,
    min_value: float,
    max_value: float,
    noise: float,
    max_step: float,
    seed: int = 42
) -> pd.Series:
    """
    Interpola los valores NaN en 's' por bloques consecutivos, siguiendo:
      1) Relleno de izquierda a derecha, salvo si el bloque inicia en la
         primera posición (índice 0), entonces se rellena de derecha a izquierda.
      2) Si no hay valor anterior (extremo izquierdo), se usa min_value.
      3) Si no hay valor posterior (extremo derecho), se usa max_value.
      4) Después de interpolar cada valor, se añade un ruido uniforme en
         [-noise, +noise].
      5) Se fuerza a que el valor quede entre [min_value, max_value].
      6) Se fuerza a que la diferencia con el último valor conocido o rellenado
         no exceda max_step.
    
    Parámetros
    ----------
    s : pd.Series
        Serie con datos numéricos (puede tener NaN).
    min_value : float
        Valor mínimo permitido en la interpolación y valor usado en la
        frontera si el bloque de NaN empieza en el extremo izquierdo.
    max_value : float
        Valor máximo permitido y valor usado en la frontera si el bloque de NaN
        está en el extremo derecho.
    noise : float
        Cantidad de ruido aleatorio que se añade a cada valor interpolado
        (en el rango [-noise, noise]).
    max_step : float
        La máxima diferencia permitida entre el valor interpolado y
        el valor anterior (o siguiente, si rellenas de derecha a izquierda).
    seed : int
        Semilla para el generador de números aleatorios.

    Retorna
    -------
    pd.Series
        Serie sin NaN, con los valores rellenados según la lógica descrita.
    """

    random.seed(seed)
    arr = s.to_numpy(dtype=float, copy=True)
    n = len(arr)

    def clamp_and_step(value, valor_anterior):
        diff = value - valor_anterior
        if abs(diff) > max_step:
            if diff > 0:
                value = valor_anterior + max_step
            else:
                value = valor_anterior - max_step
        
        if value < min_value:
            value = min_value
        elif value > max_value:
            value = max_value
        
        return value

    i = 0
    while i < n:
        if not np.isnan(arr[i]):
            i += 1
            continue
        
        start = i
        while i < n and np.isnan(arr[i]):
            i += 1
        end = i - 1
        size = end - start + 1

        if start > 0 and not np.isnan(arr[start - 1]):
            L = arr[start - 1]
        else:
            L = min_value
        
        if end < (n - 1) and not np.isnan(arr[end + 1]):
            R = arr[end + 1]
        else:
            R = max_value

        if start == 0:
            step_count = size + 1
            diff = L - R
            valor_anterior = R  

            for offset in range(size):
                idx = end - offset
                frac = (offset + 1) / step_count
                val_interp = R + diff * frac
                val_interp += random.uniform(-noise, noise)
                val_clamped = clamp_and_step(val_interp, valor_anterior)
                
                arr[idx] = val_clamped
                valor_anterior = arr[idx]

        else:
            step_count = size + 1
            diff = R - L
            valor_anterior = L

            for offset in range(size):
                idx = start + offset
                frac = (offset + 1) / step_count
                val_interp = L + diff * frac
                val_interp += random.uniform(-noise, noise)
                val_clamped = clamp_and_step(val_interp, valor_anterior)

                arr[idx] = val_clamped
                valor_anterior = arr[idx]

    return pd.Series(arr, index=s.index, name=s.name)

if __name__ == "__main__":
    data = [np.nan, np.nan, 0.5, 0.65, np.nan, np.nan, np.nan, np.nan, 0.8, np.nan]
    s = pd.Series(data, name="Ejemplo")

    s_filled = interpolate_with_rules(s, min_value=0, max_value=1, noise=0, max_step=2, seed=42)
    s_filled = s_filled.round(2)
    s.name, s_filled.name = ["Original", "Rellenado"]
    print(pd.concat([s, s_filled], axis=1))

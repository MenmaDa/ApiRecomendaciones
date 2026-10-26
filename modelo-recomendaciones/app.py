from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


app = FastAPI(
    title="Nexus ERP Recommendation Model",
    version="1.0.0"
)


class ProductoCatalogo(BaseModel):
    id: str
    nombre: str
    categoria: str
    tipo: str
    precio: float
    puntuacionCalidad: int
    stock: Optional[int] = None
    activo: bool = True


class Venta(BaseModel):
    facturaId: str
    clienteId: str
    productoServicioId: str


class RecomendacionRequest(BaseModel):
    clienteId: str
    catalogo: list[ProductoCatalogo]
    ventas: list[Venta]
    presupuesto: Optional[float] = None
    limite: int = Field(default=5, ge=1, le=10)


def crear_reglas(ventas: pd.DataFrame) -> pd.DataFrame:
    if ventas.empty or ventas["factura_id"].nunique() < 2:
        return pd.DataFrame()

    canasta = pd.crosstab(
        ventas["factura_id"],
        ventas["producto_servicio_id"]
    ).astype(bool)

    try:
        frecuentes = apriori(
            canasta,
            min_support=0.15,
            use_colnames=True
        )

        if frecuentes.empty:
            return pd.DataFrame()

        return association_rules(
            frecuentes,
            metric="confidence",
            min_threshold=0.20
        )

    except ValueError:
        return pd.DataFrame()


def aplicar_puntaje(resultado: pd.DataFrame) -> pd.DataFrame:
    precio_seguro = resultado["precio"].replace(0, 1)
    relacion_calidad_precio = (
        resultado["puntuacion_calidad"] / precio_seguro
    )

    max_relacion = relacion_calidad_precio.max()

    if max_relacion > 0:
        puntaje_precio = relacion_calidad_precio / max_relacion * 15
    else:
        puntaje_precio = 0

    resultado["puntaje"] = (
        resultado["confianza"] * 60
        + (resultado["puntuacion_calidad"] / 5) * 25
        + puntaje_precio
    )

    return resultado


@app.get("/health")
def health():
    return {"estado": "ok"}


@app.post("/recomendaciones")
def recomendar(request: RecomendacionRequest):
    catalogo = pd.DataFrame([
        {
            "id": str(producto.id),
            "nombre": producto.nombre,
            "categoria": producto.categoria,
            "tipo": producto.tipo,
            "precio": producto.precio,
            "puntuacion_calidad": producto.puntuacionCalidad,
            "stock": producto.stock,
            "activo": producto.activo
        }
        for producto in request.catalogo
    ])

    ventas = pd.DataFrame([
        {
            "factura_id": str(venta.facturaId),
            "cliente_id": str(venta.clienteId),
            "producto_servicio_id": str(venta.productoServicioId)
        }
        for venta in request.ventas
    ])

    if catalogo.empty:
        return []

    cliente_id = str(request.clienteId)
    compras = set()

    if not ventas.empty:
        compras = set(
            ventas.loc[
                ventas["cliente_id"] == cliente_id,
                "producto_servicio_id"
            ]
        )

    reglas = crear_reglas(ventas)
    candidatas = []

    # Recomendaciones según compras relacionadas.
    if not reglas.empty:
        for _, regla in reglas.iterrows():
            antecedentes = set(regla["antecedents"])
            consecuentes = set(regla["consequents"])

            if antecedentes.issubset(compras):
                for producto_id in consecuentes:
                    producto_id = str(producto_id)

                    if producto_id not in compras:
                        candidatas.append({
                            "producto_servicio_id": producto_id,
                            "confianza": float(regla["confidence"])
                        })

    if candidatas:
        recomendaciones = pd.DataFrame(candidatas).groupby(
            "producto_servicio_id",
            as_index=False
        )["confianza"].max()

        resultado = recomendaciones.merge(
            catalogo,
            left_on="producto_servicio_id",
            right_on="id"
        )
    else:
        resultado = pd.DataFrame()

    # Filtros: activo, stock y presupuesto.
    if not resultado.empty:
        resultado = resultado[resultado["activo"] == True]

        resultado = resultado[
            (resultado["tipo"] == "SERVICIO")
            | (resultado["stock"].fillna(0) > 0)
        ]

        if request.presupuesto is not None:
            resultado = resultado[
                resultado["precio"] <= request.presupuesto
            ]

        resultado = aplicar_puntaje(resultado)

        resultado["motivo"] = (
            "Clients with similar purchases also selected this option."
        )

    # Alternativas por calidad/precio cuando no existan suficientes reglas.
    faltantes = request.limite - len(resultado)

    if faltantes > 0:
        ya_recomendados = (
            set(resultado["id"])
            if not resultado.empty
            else set()
        )

        alternativas = catalogo[
            (~catalogo["id"].isin(compras))
            & (~catalogo["id"].isin(ya_recomendados))
            & (catalogo["activo"] == True)
        ].copy()

        alternativas = alternativas[
            (alternativas["tipo"] == "SERVICIO")
            | (alternativas["stock"].fillna(0) > 0)
        ]

        if request.presupuesto is not None:
            alternativas = alternativas[
                alternativas["precio"] <= request.presupuesto
            ]

        if not alternativas.empty:
            alternativas["confianza"] = 0
            alternativas = aplicar_puntaje(alternativas)

            alternativas["motivo"] = (
                "Suggested for its quality-to-price ratio and availability."
            )

            resultado = pd.concat(
                [resultado, alternativas],
                ignore_index=True
            )

    if resultado.empty:
        return []

    salida = resultado[
        [
            "id",
            "nombre",
            "categoria",
            "tipo",
            "precio",
            "puntuacion_calidad",
            "confianza",
            "puntaje",
            "motivo"
        ]
    ].sort_values(
        "puntaje",
        ascending=False
    ).head(request.limite)

    salida = salida.rename(columns={
        "id": "productoServicioId",
        "puntuacion_calidad": "puntuacionCalidad"
    })

    return salida.to_dict(orient="records")
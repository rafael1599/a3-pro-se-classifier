"""Genera figuras ilustrativas para la Sección 11 del informe TP3."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_balanceo():
    """11.4: balanceo por submuestreo."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    clases = ["IFP", "Counsel", "Extension"]
    antes = [354, 353, 288]
    despues = [278, 278, 278]
    colores = ["#3b82f6", "#10b981", "#f59e0b"]

    axes[0].bar(clases, antes, color=colores, edgecolor="black", linewidth=0.6)
    axes[0].set_title("Antes: cantidades dispares", fontsize=12, weight="bold")
    axes[0].set_ylabel("Casos por tipo")
    axes[0].set_ylim(0, 400)
    for i, v in enumerate(antes):
        axes[0].text(i, v + 8, str(v), ha="center", weight="bold")

    axes[1].bar(clases, despues, color=colores, edgecolor="black", linewidth=0.6)
    axes[1].set_title("Despues: 278 casos por tipo", fontsize=12, weight="bold")
    axes[1].set_ylim(0, 400)
    for i, v in enumerate(despues):
        axes[1].text(i, v + 8, str(v), ha="center", weight="bold")

    fig.suptitle("Balanceo: igualar para que el sistema no se incline a la clase mayoritaria",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_balanceo.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)


def fig_imputacion():
    """11.5: promedio vs mediana con outlier."""
    fig, ax = plt.subplots(figsize=(10, 3.6))

    valores = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1000]
    promedio = np.mean(valores)
    mediana = np.median(valores)

    for v in valores[:-1]:
        ax.plot(v, 1, "o", markersize=15, color="#3b82f6", markeredgecolor="black")
    ax.plot(valores[-1], 1, "o", markersize=20, color="#ef4444",
            markeredgecolor="black", label="Caso extremo")

    ax.axvline(promedio, color="#f59e0b", linewidth=2.5, linestyle="--",
               label=f"Promedio = {promedio:.0f} (distorsionado)")
    ax.axvline(mediana, color="#10b981", linewidth=2.5, linestyle="-",
               label=f"Mediana = {mediana:.0f} (representa al tipico)")

    ax.set_xlim(-50, 1100)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Sueldos en una sala (en miles)")
    ax.set_title("Cuando hay un caso extremo, el promedio engania: la mediana es mas honesta",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_imputacion.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)


def fig_cache():
    """11.10: patrón base de datos primero."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def caja(x, y, w, h, label, color):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.08",
                             linewidth=1.6, edgecolor="black", facecolor=color)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=11, weight="bold")

    def flecha(x1, y1, x2, y2, label=None, color="black"):
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle="->", mutation_scale=18,
                              color=color, linewidth=2)
        ax.add_patch(arr)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25, label,
                    ha="center", fontsize=9.5, style="italic")

    caja(0.3, 2.3, 1.8, 1.4, "Usuario\nbusca", "#dbeafe")
    caja(2.8, 2.3, 2.0, 1.4, "Nuestra\nbase de datos", "#bbf7d0")
    caja(5.6, 4.0, 2.4, 1.4, "Si esta:\ndevolver", "#bbf7d0")
    caja(5.6, 0.6, 2.4, 1.4, "Si no esta:\nAPI publica", "#fed7aa")
    caja(8.8, 0.6, 2.8, 1.4, "Guardar para\nla proxima", "#fde68a")

    flecha(2.1, 3.0, 2.8, 3.0)
    flecha(4.8, 3.2, 5.6, 4.4, "lo tenemos")
    flecha(4.8, 2.8, 5.6, 1.6, "no lo tenemos")
    flecha(8.0, 1.3, 8.8, 1.3)

    ax.text(6, 5.6, "Analogia: revisar la heladera antes de ir al supermercado",
            ha="center", fontsize=12, weight="bold")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_cache.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    fig_balanceo()
    fig_imputacion()
    fig_cache()
    print("Listo. Figuras en:", FIG_DIR)

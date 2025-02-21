from typing import Type

import pydantic
from loguru import logger
from matplotlib import colorbar
from matplotlib import pyplot as plt

from utils.SmoothAHAplot.src.aha import aha_interpolation, aha_segmental_values
from utils.SmoothAHAplot.src.aha.parameters.parameters import PLOT_COMPONENTS


class AHAPlotting(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    ax: plt.Axes
    fig: plt.Figure
    biomarker_name: str

    def __init__(self, **data: dict) -> None:
        super().__init__(**data)

    def create_plot(self, missing_segments):
    # def create_plot(self, segment_values, missing_segments=None):
        if missing_segments is None:
            missing_segments = []

        self._clear_bullseye_plot()

        return self.ax

    def _clear_bullseye_plot(self) -> None:
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_ylim(
            [PLOT_COMPONENTS["bound_range"]["inner"], PLOT_COMPONENTS["bound_range"]["outer"]]
        )
        self.ax.grid(None)

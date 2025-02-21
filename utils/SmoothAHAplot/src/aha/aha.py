import base64
import plotly.tools as tls
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from utils.SmoothAHAplot.src.aha import aha_annotation, aha_segmental_values
from utils.SmoothAHAplot.src.aha.parameters.parameters import PLOT_COMPONENTS
from utils.SmoothAHAplot.src.aha.plot import bounds, plotting


class AHA:
    """
    Class for producing smooth 17 and 18 segment left ventricle plot, recommended by American Heart
    Association:
        http://www.pmod.com/files/download/v34/doc/pcardp/3615.htm
    Inspired with the 'bullseye plot':
        https://matplotlib.org/gallery/specialty_plots/leftventricle_bulleye.html
    Available at:
        https://github.com/MaciejPMarciniak/smoothAHAplot

    Two methods are included, adjusted to plot myocardial work and strain values with proper scales.
    """

    def __init__(
        self,
        segments: dict[str, float],
        plot_type: str
    ):
        self._segments = aha_segmental_values.AHASegmentalValues(segments=segments)
        self._plot_type = plot_type
        # self._output_path = plot_output_path
        self.fig = plt.Figure
        self.ax = plt.Axes

    @property
    def segment_values(self) -> list[int]:
        return self.segments.segmental_values

    @property
    def segments(self) -> aha_segmental_values.AHASegmentalValues:
        return self._segments

    def bullseye_smooth(self, missing_value):

        self.fig, self.ax = plt.subplots(
            figsize=PLOT_COMPONENTS["figure_size"],
            nrows=1,
            ncols=1,
            subplot_kw={"projection": "polar"},
            layout="constrained",
        )

        ax_annotation = aha_annotation.AHAAnnotation(segments=self.segments, ax=self.ax)
        self.ax = ax_annotation.annotate_aha_segments(missing_value)
        ax_bounds = bounds.AHAPlotBounds(len(self.segments), ax=self.ax)
        self.ax = ax_bounds.draw_aha_bounds()
        ax_plotting = plotting.AHAPlotting(ax=self.ax, fig=self.fig, biomarker_name=self._plot_type)
        ax_plotting.create_plot(self.segments)
        uri = fig_to_uri(self.fig)
        return uri


def fig_to_uri(fig):
    """Convert a Matplotlib figure to a URI that can be displayed in Dash."""
    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Close the figure to prevent it from consuming memory
    plt.close(fig)
    # Encode the BytesIO object as base64 and decode to a string
    data = base64.b64encode(buf.getbuffer()).decode("utf8")
    # Create the URI
    uri = "data:image/png;base64,{}".format(data)
    return uri
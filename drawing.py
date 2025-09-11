import enum
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union

import einops
import numpy as np
import plotly.express as px
import tqdm
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image, ImageDraw, ImageOps

from image_color import blur_image
from misc import get_color_hash, get_img_hash

Arr = np.ndarray


class LineType(enum.Enum):
    STRAIGHT = enum.auto()
    BEZIER = enum.auto()


class ShapeType(enum.Enum):
    CIRCLE = enum.auto()
    RECT = enum.auto()
    TRI = enum.auto()
    HEX = enum.auto()

    @property
    def n_sides(self) -> int | None:
        if self == ShapeType.CIRCLE:
            return None
        elif self == ShapeType.RECT:
            return 4
        elif self == ShapeType.TRI:
            return 3
        elif self == ShapeType.HEX:
            return 6


@dataclass
class BezierCurve:
    """Represents a cubic Bezier curve with its control points."""

    p0: Float[Arr, "2"]  # Start point
    p1: Float[Arr, "2"]  # First control point
    p2: Float[Arr, "2"]  # Second control point
    p3: Float[Arr, "2"]  # End point

    def to_svg_path(self) -> str:
        """Convert to SVG path command."""
        return f"M {self.p0[1]:.1f} {self.p0[0]:.1f} C {self.p1[1]:.1f} {self.p1[0]:.1f}, {self.p2[1]:.1f} {self.p2[0]:.1f}, {self.p3[1]:.1f} {self.p3[0]:.1f}"

    @property
    def line_length(self) -> int:
        chord_length = np.linalg.norm(self.p3 - self.p0)
        control_length = (
            np.linalg.norm(self.p1 - self.p0)
            + np.linalg.norm(self.p2 - self.p1)
            + np.linalg.norm(self.p3 - self.p2)
        )
        return 1 + max(1, int((chord_length + control_length) / 2))

    def interpolate_points(self, num_steps: int | None = None) -> Float[Arr, "2 n_steps"]:
        """Interpolate points along the curve for pixel-based operations."""
        num_steps = num_steps or int(self.line_length)
        t_values = np.linspace(0, 1, num_steps, endpoint=True)
        coords = (
            ((1 - t_values) ** 3)[None, :] * self.p0[:, None]
            + (3 * (1 - t_values) ** 2 * t_values)[None, :] * self.p1[:, None]
            + (3 * (1 - t_values) * t_values**2)[None, :] * self.p2[:, None]
            + (t_values**3)[None, :] * self.p3[:, None]
        )
        return coords

    def reversed(self) -> "BezierCurve":
        return BezierCurve(p0=self.p3, p1=self.p2, p2=self.p1, p3=self.p0)

    def scale(self, scale_factor: float) -> "BezierCurve":
        return BezierCurve(
            p0=self.p0 * scale_factor,
            p1=self.p1 * scale_factor,
            p2=self.p2 * scale_factor,
            p3=self.p3 * scale_factor,
        )

    def offset(self, offset: Float[Arr, "2"]) -> "BezierCurve":
        return BezierCurve(
            p0=self.p0 + offset, p1=self.p1 + offset, p2=self.p2 + offset, p3=self.p3 + offset
        )


@dataclass
class Circle:
    """Represents a circle with center and radius."""

    center: Float[Arr, "2"]  # Center point
    radius: float  # Radius

    def to_svg_element(self, color: str = "black") -> str:
        """Convert to SVG circle element."""
        return f'<circle cx="{self.center[1]:.1f}" cy="{self.center[0]:.1f}" r="{self.radius:.1f}" stroke="{color}" fill="none"/>'

    @property
    def line_length(self) -> int:
        return 1 + max(1, int(2 * np.pi * self.radius))

    def interpolate_points(self, num_steps: int | None = None) -> Float[Arr, "2 n_steps"]:
        """Interpolate points around the circle for pixel-based operations."""
        num_steps = num_steps or int(self.line_length)
        angles = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
        coords = self.center[:, None] + self.radius * np.array([np.cos(angles), np.sin(angles)])
        return coords

    def scale(self, scale_factor: float) -> "Circle":
        return Circle(center=self.center * scale_factor, radius=self.radius * scale_factor)

    def offset(self, offset: Float[Arr, "2"]) -> "Circle":
        return Circle(center=self.center + offset, radius=self.radius)


@dataclass
class PiecewiseLinear:
    """Represents a piecewise linear shape (polygons, straight lines, etc.) with coordinates."""

    coords: Float[Arr, "2 n_points"]  # Array of points

    def to_svg_path(self) -> str:
        """Convert to SVG path command for piecewise linear shape."""
        if self.coords.shape[1] == 0:
            return ""

        # Start with move command
        path = f"M {self.coords[1, 0]:.1f} {self.coords[0, 0]:.1f}"

        # Add line commands for each subsequent point
        for i in range(1, self.coords.shape[1]):
            path += f" L {self.coords[1, i]:.1f} {self.coords[0, i]:.1f}"

        return path

    @property
    def line_length(self) -> int:
        return sum(
            np.linalg.norm(self.coords[:, i + 1] - self.coords[:, i])
            for i in range(self.coords.shape[1] - 1)
        )

    def interpolate_points(self, num_steps: int | None = None) -> Float[Arr, "2 n_steps"]:
        """Vectorized interpolation for piecewise linear shapes."""
        if self.coords.shape[1] == 0:
            return np.empty((2, 0))

        # For piecewise linear, we can interpolate along the total length
        num_steps = num_steps or int(self.line_length)

        # Pre-calculate segment information using vectorized operations
        segment_vectors = np.diff(self.coords, axis=1)  # Direction vectors
        segment_lengths = np.linalg.norm(segment_vectors, axis=0)  # Lengths
        segment_starts = self.coords[:, :-1]  # Start points of each segment

        # Generate parameter values along the total length
        t_values = np.linspace(0, segment_lengths.sum(), num_steps, endpoint=True)

        # Pre-calculate cumulative segment lengths for efficient lookup
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

        # Find which segment each t value belongs to using vectorized operations
        # Use searchsorted for efficient binary search instead of linear search
        segment_indices = np.searchsorted(cumulative_lengths[1:], t_values, side="right")
        segment_indices = np.clip(segment_indices, 0, len(segment_lengths) - 1)

        # Calculate local t values within each segment
        local_t_values = t_values - cumulative_lengths[segment_indices]
        segment_t_values = local_t_values / segment_lengths[segment_indices]

        # Vectorized interpolation: start_point + t * (end_point - start_point)
        start_points = segment_starts[:, segment_indices]
        end_points = self.coords[:, segment_indices + 1]

        # Expand segment_t_values for broadcasting
        segment_t_values_expanded = segment_t_values[None, :]

        # Vectorized linear interpolation
        interpolated_points = start_points + segment_t_values_expanded * (end_points - start_points)

        return interpolated_points

    def scale(self, scale_factor: float) -> "PiecewiseLinear":
        return PiecewiseLinear(coords=self.coords * scale_factor)

    def offset(self, offset: Float[Arr, "2"]) -> "PiecewiseLinear":
        return PiecewiseLinear(coords=self.coords + offset[:, None])


@dataclass
class Shape:
    """
    Most general class for shapes: contains characters, connected arcs, and all uniform shapes.
    """

    line_type: LineType | None = None
    shape_type: ShapeType | None = None
    character_path: str | None = None

    # Not parameter-specific
    max_out_of_bounds: int | None = (
        None  # No lines are allowed to go more than this far out of bounds, at any point
    )

    # Used for shapes and lines (not characters)
    size_range: tuple[float, float] = (0.05, 0.15)  # size range

    # Only used for lines
    endpoint_angle_range: tuple[float, float] = (np.pi * 0.2, np.pi * 0.6)  # zero is boring!
    bezier_control_factor_range: tuple[float, float] = (0.4, 0.6)  # curviness ratios
    bezier_end_angle_range: tuple[float, float] = (-np.pi * 0.6, np.pi * 0.6)  # can be zero

    def __post_init__(self):
        assert (
            sum(x is not None for x in [self.shape_type, self.character_path, self.line_type]) == 1
        ), "Must specify one of shape_type, character_path, or line_type"
        if self.line_type is not None:
            assert self.shape_type is None, "Can't specify both line_type and shape_type"
            assert self.character_path is None, "Can't specify both line_type and character_path"

        if self.line_type is not None:
            pass  # TODO - add in appropriate restrictions here

    def get_random_params(
        self,
        start_coords: Float[Arr, "2"] | None,
        start_dir: Float[Arr, "2"] | None,
        canvas_length: float,
        end_coords: Float[Arr, "2"] | None = None,
    ) -> dict:
        """Gets random parameters for the shape."""

        # Chinese characters have no free parameters except for position
        if self.character_path is not None:
            return {}

        # Non-line shapes have a single free parameter: size
        elif self.shape_type is not None:
            return {"size": np.random.uniform(*self.size_range)}

        else:
            # Get angles. The end angle represents the angle we're moving in the direction of, if we moved
            # straight from start to end (which is why it's close to `start_angle`).
            start_angle = np.arctan2(*start_dir)
            angle_delta = np.random.uniform(*self.endpoint_angle_range) * (
                1 if np.random.rand() < 0.5 else -1
            )
            end_angle = start_angle + angle_delta

            # Get end point
            if end_coords is None:
                radius = (
                    np.random.rand() * (self.size_range[1] - self.size_range[0])
                    + self.size_range[0]
                )
                radius *= canvas_length
                end_coords_delta = radius * np.array([np.sin(end_angle), np.cos(end_angle)])
                end_coords = start_coords + end_coords_delta

            if self.line_type == LineType.STRAIGHT:
                return {"end_coords": end_coords}
            elif self.line_type == LineType.BEZIER:
                end_angle_direction = end_angle + np.random.uniform(*self.bezier_end_angle_range)
                return {
                    "end_coords": end_coords,
                    "end_dir": np.array([np.sin(end_angle_direction), np.cos(end_angle_direction)]),
                    "start_strength": np.random.uniform(*self.bezier_control_factor_range),
                    "end_strength": np.random.uniform(*self.bezier_control_factor_range),
                }

    def get_drawing_coords_list(
        self,
        n_shapes: int,
        start_dir: Float[Arr, "2"] | None,
        start_coords: Float[Arr, "2"] | None,
        canvas_y: int,
        canvas_x: int,
        outer_bound: float | None,
        inner_bound: float | None,
        end_coords: Float[Arr, "2"] | None = None,
        max_n_repeats_without_valid_line: int = 100,
    ) -> list[
        tuple[
            Union[BezierCurve, Circle, PiecewiseLinear], Float[Arr, "2 n_pixels"], Float[Arr, "2"]
        ],
    ]:
        """Generates `n_shapes` random shapes, and returns a list of their coords.

        Args:
            n_shapes: Number of shapes to generate
            start_dir: Direction to start the shape in (None for shapes)
            start_coords: Coordinates to start the shape at (None for shapes)
            canvas_y: Height of the canvas
            canvas_x: Width of the canvas
            outer_bound: Max value out of bounds we allow ANY pixel to be
            inner_bound: Min negative value out of bounds we allow the LAST pixel to be
            end_coords: Coordinates to end the shape at (if we are restricting this)
            max_n_repeats_without_valid_line: Maximum number of attempts to generate a valid shape

        Returns:
            List of the following:
            - params: Dictionary of parameters used to generate the shape
            - coords: (2, n_pixels) array of float coordinates for the shape OR BezierCurve object
            - coords_uncropped: (2, n_pixels) without cropping at sides (useful for final drawing)
            - pixels: (2, n_pixels) array of integer (y, x) coordinates for the shape
        """
        canvas_length = max(canvas_x, canvas_y)

        coords_list = []
        counter = 0

        while len(coords_list) < n_shapes:
            # Get random parameterization for this shape
            params = self.get_random_params(start_coords, start_dir, canvas_length, end_coords)

            # For shapes, we ignore the start coords and randomize them
            if self.shape_type is not None:
                # Random position for shapes
                shape_start_coords = np.random.rand(2) * np.array([canvas_y, canvas_x])
            else:
                # Use provided start coords for lines
                shape_start_coords = start_coords

            # Get the actual coordinates for this shape (interpolating with step size of 1)
            coords_uncropped, _, end_dir = self.draw_curve(
                shape_start_coords, start_dir, canvas_y=canvas_y, canvas_x=canvas_x, **params
            )
            interpolated_coords = coords_uncropped.interpolate_points()

            # Crop parts of `coords` which go off the edge, and only keep ones which are in bounds anywhere
            coords = mask_coords(
                interpolated_coords,
                canvas_y,
                canvas_x,
                outer_bound=outer_bound if end_coords is None else None,
                inner_bound=inner_bound if end_coords is None else None,
                remove=True,
            )
            if coords.shape[-1] > 0:
                coords_list.append((coords_uncropped, coords, end_dir))

            counter += 1
            if counter / (len(coords_list) + 10) > max_n_repeats_without_valid_line:
                raise ValueError(
                    f"No valid shapes: only found {len(coords_list)}/{max_n_repeats_without_valid_line}. Params are {start_coords=}, {start_dir=}, {end_coords=}, {canvas_length=}"
                )

        return coords_list

    def draw_curve(
        self,
        start_coords: Float[Arr, "2"],
        start_dir: Float[Arr, "2"] | None,
        canvas_y: int | None = None,
        canvas_x: int | None = None,
        end_coords: Float[Arr, "2"] | None = None,
        **kwargs,
    ) -> tuple[
        Union[BezierCurve, Circle, PiecewiseLinear], Int[Arr, "2 n_pixels"], Float[Arr, "2"]
    ]:
        """
        Draw arc, Bezier curve, or shape and return interpolated pixel coordinates.

        Args:
            start_coords: (y, x) starting coordinates as floats
            start_dir: normalized (y, x) direction vector (None for shapes)
            canvas_y: Height of the canvas (required for shapes)
            canvas_x: Width of the canvas (required for shapes)
            end_coords: (y, x) ending coordinates as floats (None for shapes)
            **kwargs: For 'arc': radius, angle, orientation (True=anticlockwise)
                    For 'bezier': end_coords, start_length, end_length, end_dir
                    For 'shape': size

        Returns:
            coords: (2, num_pixels) array of float coordinates OR BezierCurve object
            pixels: (2, num_pixels) array of integer (y, x) coordinates
            final_dir: (2,) array with final normalized (y, x) direction
        """
        # We start by getting 3 things:
        # - Length of curve
        # - Final direction
        # - Function which interpolates the curve in the region [0, 1]

        if self.shape_type is not None:
            # Handle shapes
            assert "size" in kwargs
            assert canvas_y is not None and canvas_x is not None, (
                "Canvas dimensions required for shapes"
            )
            size = kwargs["size"]

            # Calculate shape size in pixels
            canvas_length = max(canvas_y, canvas_x)  # Use actual canvas dimensions
            shape_size = size * canvas_length

            if self.shape_type == ShapeType.CIRCLE:
                # Generate circle object
                radius = shape_size / 2
                center = start_coords
                coords = Circle(center=center, radius=radius)
                final_dir = np.array([1.0, 0.0])  # Arbitrary direction for shapes

            elif self.shape_type == ShapeType.RECT:
                # Generate square coordinates
                half_size = shape_size / 2
                center = start_coords
                # Square vertices: top-left, top-right, bottom-right, bottom-left, back to top-left
                square_coords = np.array(
                    [
                        [-half_size, -half_size],  # top-left
                        [half_size, -half_size],  # top-right
                        [half_size, half_size],  # bottom-right
                        [-half_size, half_size],  # bottom-left
                        [-half_size, -half_size],  # back to start
                    ]
                ).T
                coords = PiecewiseLinear(coords=center[:, None] + square_coords)
                final_dir = np.array([1.0, 0.0])

            elif self.shape_type == ShapeType.TRI:
                # Generate equilateral triangle coordinates
                half_width = shape_size / 2
                height = half_width * np.sqrt(3)  # Height of equilateral triangle
                center = start_coords
                # Triangle vertices: bottom-left, bottom-right, top, back to bottom-left
                triangle_coords = np.array(
                    [
                        [-half_width, -height / 3],  # bottom-left
                        [half_width, -height / 3],  # bottom-right
                        [0, 2 * height / 3],  # top
                        [-half_width, -height / 3],  # back to start
                    ]
                ).T
                coords = PiecewiseLinear(coords=center[:, None] + triangle_coords)
                final_dir = np.array([1.0, 0.0])

            elif self.shape_type == ShapeType.HEX:
                # Generate regular hexagon coordinates (rotated so corners are left/right)
                half_width = shape_size / 2
                height = half_width * np.sqrt(3)  # Height of regular hexagon
                center = start_coords
                # Hexagon vertices: going clockwise from leftmost point (rotated 90 degrees)
                hex_coords = np.array(
                    [
                        [0, -half_width],  # left
                        [-height / 2, -half_width / 2],  # bottom-left
                        [-height / 2, half_width / 2],  # bottom-right
                        [0, half_width],  # right
                        [height / 2, half_width / 2],  # top-right
                        [height / 2, -half_width / 2],  # top-left
                        [0, -half_width],  # back to start
                    ]
                ).T
                coords = PiecewiseLinear(coords=center[:, None] + hex_coords)
                final_dir = np.array([1.0, 0.0])
            else:
                raise ValueError(f"Unsupported shape type: {self.shape_type}")

        elif self.line_type == LineType.STRAIGHT:
            assert set(kwargs.keys()) == set()

            # Create PiecewiseLinear object for straight line & end direction
            coords = PiecewiseLinear(coords=np.column_stack([start_coords, end_coords]))
            final_dir = start_dir

        elif self.line_type == LineType.BEZIER:
            assert set(kwargs.keys()) == {"start_strength", "end_strength", "end_dir"}
            start_strength, end_strength, end_dir = (
                kwargs["start_strength"],
                kwargs["end_strength"],
                kwargs["end_dir"],
            )

            # Get chord length, to be used for computing control points
            chord = end_coords - start_coords
            chord_length = np.linalg.norm(chord)

            # Default for `end_dir` is to be pointing in the direction of the direct path from start to end
            if end_dir is None:
                end_dir = chord / chord_length

            # Control points
            p0 = np.array(start_coords)
            p1 = np.array(start_coords) + start_strength * chord_length * start_dir
            p2 = np.array(end_coords) - end_strength * chord_length * end_dir
            p3 = np.array(end_coords)

            # Create BezierCurve object & end direction
            coords = BezierCurve(p0=p0, p1=p1, p2=p2, p3=p3)
            final_dir = np.array(end_dir)

        else:
            raise ValueError(f"Invalid line type: {self.line_type}")

        # For all shape types, we need to interpolate points for pixel generation
        coords_interpolated = coords.interpolate_points()

        # Round to pixels and normalize final direction
        pixels = np.round(coords_interpolated).astype(int)
        pixels = np.unique(pixels, axis=1)
        final_dir = final_dir / np.linalg.norm(final_dir)

        return coords, pixels, final_dir


@dataclass
class TargetImage:
    image_path: str | dict[tuple[int, int, int], str]
    weight_image_path: str | None
    palette: list[tuple[int, int, int]]
    x: int
    output_x: int
    blur_rad: float | None = 4
    display_dithered: bool = False

    def __post_init__(self):
        # Check colors are valid (raise error if not)
        _ = [get_color_string(color) for color in self.palette]

        # Load in image (either a single image which we'll dither, or a dict of images for each colour)
        if isinstance(self.image_path, str):
            image = Image.open(self.image_path).convert("L" if len(self.palette) == 1 else "RGB")
            width, height = image.size
        else:
            image = {
                color: Image.open(img_path).convert("L" if len(self.palette) == 1 else "RGB")
                for color, img_path in self.image_path.items()
            }
            assert len(set(i.size for i in image.values())) == 1, (
                "All images must have the same size"
            )
            width, height = image.values()[0].size

        # Optionally load in (and turn to an array) the weight image
        self.weight_image = None
        if self.weight_image_path is not None:
            weight_image = Image.open(self.weight_image_path).convert("L")
            self.weight_image = (
                np.asarray(weight_image.resize((self.x, self.y))).astype(np.float32) / 255
            )

        # Get dimensions (for target and output images)
        self.y = int(self.x * height / width)
        self.output_sf = self.output_x / self.x

        # Optionally perform dithering, and get `self.image_dict` for use in `Drawing`
        if len(self.palette) == 1:
            # Case 1: basic monochrome image (only black lines)
            assert isinstance(image, Image.Image), (
                "Image must be a single image if there is only one color"
            )
            image_arr = np.asarray(image.resize((self.x, self.y)))
            self.image_dict = {self.palette[0]: 1.0 - image_arr.astype(np.float32) / 255}
        elif isinstance(image, dict):
            # Case 2: multiple images (one per color), e.g. Bowie
            self.image_dict = {
                color: np.asarray(color_img.resize((self.x, self.y))).astype(np.float32) / 255
                for color, color_img in image.items()
            }
            density_sum = sum([img.sum() for img in self.image_dict.values()])
            for color, img in self.image_dict.items():
                print(f"{get_color_string(color)}, density = {img.sum() / density_sum:.4f}")
        else:
            # Case 3: single color image, to be dithered
            assert (255, 255, 255) not in self.palette, "White should not be in palette"
            image_arr = np.asarray(image.resize((self.x, self.y)))
            image_dithered = FS_dither(image_arr, [(255, 255, 255)] + self.palette)
            self.image_dict = {
                color: (get_img_hash(image_dithered) == get_color_hash(np.array(color))).astype(
                    np.float32
                )
                for color in self.palette
            }
            if self.blur_rad is not None:
                self.image_dict = {
                    color: blur_image(img, self.blur_rad) for color, img in self.image_dict.items()
                }

            nonwhite_density_sum = sum(
                [img.sum() for color, img in self.image_dict.items() if color != (255, 255, 255)]
            )
            for color, img in self.image_dict.items():
                print(
                    f"{get_color_string(color)}, density = {img.sum() / nonwhite_density_sum:.4f}"
                )

        # Display the images for each color (again we split based on whether the input was a string or dictionary)
        if self.display_dithered:
            background_colors = [
                np.array([255, 255, 255]) if sum(color) < 255 + 160 else np.array([0, 0, 0])
                for color in self.palette
            ]
            dithered_images = np.concatenate(
                [
                    bg + img[:, :, None] * (np.array(color) - bg)
                    for bg, (color, img) in zip(background_colors, self.image_dict.items())
                ],
                axis=1,
            )
            px.imshow(
                dithered_images,
                height=290,
                width=100 + 200 * len(self.palette),
                title=" | ".join([str(x) for x in self.palette]),
            ).update_layout(margin=dict(l=10, r=10, t=40, b=10)).show()


@dataclass
class Drawing:
    target: TargetImage

    shape: Shape

    n_shapes: int | list[int]
    n_random: int
    darkness: float | list[float]
    negative_penalty: float

    # Outer bound means we don't allow any lines to go further than this far out of bounds. Inner bound
    # means we don't allow any lines to END closer than this to the edge. Inner is important because without
    # it, we might finish 1 pixel away from the end and then we'd be totally fucked.
    outer_bound: float | None
    inner_bound: float | None

    # If we use zoom fractions, then we zoom the image out before optionally appending our final lines
    zoom_fractions: tuple[float, float] | None = None

    def __post_init__(self):
        if self.outer_bound is not None:
            assert self.outer_bound >= 0, "Outer bound must be non-negative"
            assert self.inner_bound is not None and self.inner_bound > 0, (
                "Inner bound must be supplied if outer bound is"
            )

    def create_img(
        self,
        seed: int = 0,
        use_borders: bool = False,
        name: str | None = None,
    ) -> tuple[
        Image.Image,
        dict[
            str, Union[Float[Arr, "n_pixels 2"], list[Union[BezierCurve, Circle, PiecewiseLinear]]]
        ],
        tuple[int, int],
    ]:
        np.random.seed(seed)

        # If any parameters were given as a single number, convert them to lists
        if isinstance(self.darkness, float):
            self.darkness = [self.darkness] * len(self.target.palette)
        if isinstance(self.n_shapes, int):
            self.n_shapes = [self.n_shapes]
        assert len(self.n_shapes) == len(self.target.palette), (
            "Should give num shapes for each color"
        )

        # Create dicts to store coords for each color
        all_coords = {}
        all_start_end_positions = {}  # TODO - this is redundant if I can get start/end posns/dirs from `coords`?

        for color, n_shapes, darkness in zip(
            self.target.palette, self.n_shapes, self.darkness, strict=True
        ):
            if n_shapes == 0:
                continue

            image = self.target.image_dict[color]
            color_string = get_color_string(color)
            all_coords[color_string] = []

            # If using borders then start at the closest pixel to the border, if not
            # then start at the darkest pixel
            if use_borders:
                blurred_image = blur_image(image, rad=5, mode="linear")
                pixels_are_dark = blurred_image > 0.8 * blurred_image.max()
                pixels_mesh = np.stack(
                    np.meshgrid(np.arange(self.target.y), np.arange(self.target.x)), axis=-1
                )
                pixel_distances_from_border = np.stack(
                    [
                        pixels_mesh[:, :, 0],
                        self.target.y - pixels_mesh[:, :, 0],
                        pixels_mesh[:, :, 1],
                        self.target.x - pixels_mesh[:, :, 1],
                    ]
                )
                valid_pixel_distances_from_border = np.where(
                    pixels_are_dark, pixel_distances_from_border.min(axis=0), np.inf
                )
                start_coords = np.stack(
                    np.unravel_index(np.argmin(valid_pixel_distances_from_border), image.shape)
                )
            else:
                start_coords = np.stack(np.unravel_index(np.argmax(image), image.shape))

            # Initially point inwards
            start_coords_offset = start_coords - np.array([self.target.y, self.target.x]) / 2
            start_dir = -start_coords_offset / (np.linalg.norm(start_coords_offset) + 1e-6)

            current_coords = start_coords.copy()
            current_dir = start_dir.copy()

            # Get the n normal shapes
            for step in tqdm.tqdm(range(n_shapes), desc=f"Drawing {color_string}"):
                best_coords, best_coords_uncropped, best_end_dir = self.get_best_shape(
                    image, start_dir=current_dir, start_coords=current_coords
                )

                # Subtract it from the target image, and write it to the canvas
                # For all shape types, we need to interpolate points for pixel operations
                if isinstance(best_coords, (BezierCurve, Circle, PiecewiseLinear)):
                    # Estimate length for interpolation
                    if isinstance(best_coords, BezierCurve):
                        chord_length = np.linalg.norm(best_coords.p3 - best_coords.p0)
                        control_length = (
                            np.linalg.norm(best_coords.p1 - best_coords.p0)
                            + np.linalg.norm(best_coords.p2 - best_coords.p1)
                            + np.linalg.norm(best_coords.p3 - best_coords.p2)
                        )
                        line_length = (chord_length + control_length) / 2
                    elif isinstance(best_coords, Circle):
                        line_length = 2 * np.pi * best_coords.radius
                    else:  # PiecewiseLinear
                        total_length = 0
                        for i in range(best_coords.coords.shape[1] - 1):
                            total_length += np.linalg.norm(
                                best_coords.coords[:, i + 1] - best_coords.coords[:, i]
                            )
                        line_length = total_length

                    num_steps = 1 + max(1, int(line_length))
                    best_pixels = best_coords.interpolate_points(num_steps)
                    best_pixels = np.round(best_pixels).astype(np.int32)
                    best_pixels = best_pixels[
                        :,
                        (best_pixels >= 0).all(axis=0)
                        & (best_pixels < np.array(image.shape)[:, None]).all(axis=0),
                    ]
                    best_pixels = np.unique(best_pixels, axis=1)
                else:
                    best_pixels = best_coords.astype(np.int32)

                image[best_pixels[0], best_pixels[1]] -= darkness
                # all_coords[color_string].append(best_coords_uncropped)
                all_coords[color_string].append(best_coords)

                # For shapes, each shape is independent (no connection between shapes)
                # For lines, this end dir is the new start dir (same for position)
                if self.shape.shape_type is None:
                    if isinstance(best_coords, (BezierCurve, Circle, PiecewiseLinear)):
                        if isinstance(best_coords, BezierCurve):
                            current_coords = best_coords.p3  # End point of Bezier curve
                        elif isinstance(best_coords, Circle):
                            # For circles, we don't really have an "end point", so we'll use the center
                            current_coords = best_coords.center
                        else:  # PiecewiseLinear
                            current_coords = best_coords.coords[
                                :, -1
                            ]  # Last point of coordinate array
                    else:
                        current_coords = best_coords[:, -1]  # Last point of coordinate array
                    current_dir = best_end_dir

            all_start_end_positions[color_string] = {
                "start_coords": start_coords,
                "start_dir": start_dir,
                "end_coords": current_coords,
                "end_dir": current_dir,
            }

        # If using zoom fractions, then zoom out now (updating target x and y accordingly)
        if self.zoom_fractions is not None:
            zoom_pixels_y = int(self.target.y * self.zoom_fractions[0])
            zoom_pixels_x = int(self.target.x * self.zoom_fractions[1])
            offset = np.array([zoom_pixels_y, zoom_pixels_x])
            for color in self.target.palette:
                color_string = get_color_string(color)
                # Offset coords
                for shape in all_coords[color_string]:
                    shape = shape.offset(offset)
                all_start_end_positions[color_string]["start_coords"] += offset
                all_start_end_positions[color_string]["end_coords"] += offset
                # Pad image with zeros
                self.target.image_dict[color] = np.pad(
                    self.target.image_dict[color],
                    ((zoom_pixels_y, zoom_pixels_y), (zoom_pixels_x, zoom_pixels_x)),
                    mode="constant",
                    constant_values=0,
                )
            target_y = self.target.y + 2 * zoom_pixels_y
            target_x = self.target.x + 2 * zoom_pixels_x

        else:
            target_y = self.target.y
            target_x = self.target.x

        # If we're using borders, then we need to add a shape at the start & end,
        # which we choose to be as close to the border as possible
        border_lengths = {}
        if use_borders:
            for color_string, coords in all_coords.items():
                image = self.target.image_dict[color]

                # Get start shape: starting from the starting position but moving backwards, to the closest border
                best_coords_start, _, _ = self.get_best_shape(
                    image,
                    start_dir=-all_start_end_positions[color_string]["start_dir"],
                    start_coords=all_start_end_positions[color_string]["start_coords"],
                    end_coords=get_closest_point_on_border(
                        all_start_end_positions[color_string]["start_coords"], target_y, target_x
                    )[1],
                    use_bounds=False,
                )

                # Handle the start shape coordinates (Bezier curves need to be reversed)
                coords = [
                    best_coords_start.reversed()
                    if isinstance(best_coords_start, BezierCurve)
                    else best_coords_start
                ] + (coords if isinstance(coords, list) else [coords])

                # Get end shape: starting from the end position and moving to the closest border
                best_coords_end, _, _ = self.get_best_shape(
                    image,
                    start_dir=all_start_end_positions[color_string]["end_dir"],
                    start_coords=all_start_end_positions[color_string]["end_coords"],
                    end_coords=get_closest_point_on_border(
                        all_start_end_positions[color_string]["end_coords"], target_y, target_x
                    )[1],
                    use_bounds=False,
                )

                # Handle the end shape coordinates
                if isinstance(best_coords_end, (BezierCurve, Circle, PiecewiseLinear)):
                    coords = coords + [best_coords_end]
                else:
                    coords = np.concatenate([coords, best_coords_end.T], axis=0)

                all_coords[color_string] = coords

                if isinstance(best_coords_start, (BezierCurve, Circle, PiecewiseLinear)):
                    border_lengths[color_string] = (1, 1)  # One shape each
                else:
                    border_lengths[color_string] = (
                        best_coords_start.shape[1],
                        best_coords_end.shape[1],
                    )

                # Test: are the min coord distances very small, and are the start/end coords on the border?
                if isinstance(coords, list):
                    # For Bezier curves, we can't easily test this without interpolating
                    pass
                else:
                    diffs = np.diff(coords, axis=0)
                    min_diff = np.min(np.linalg.norm(diffs, axis=1))
                    assert min_diff < 1.0, f"Found unexpectedly large coord diffs: {min_diff:.3f}"
                    for coord in [coords[0], coords[-1]]:
                        _, border_coord = get_closest_point_on_border(coord, target_y, target_x)
                        border_coord_diff = np.linalg.norm(coord - border_coord)
                        assert border_coord_diff < 3.0, (
                            f"Found unexpected coord: {coord} with diff to border {border_coord} of {border_coord_diff:.3f}"
                        )
        else:
            border_lengths = {color_string: (0, 0) for color_string in all_coords.keys()}

        # Create canvas and draw on it
        canvas, svg = self.make_canvas_and_crop_coords(all_coords, target_y, target_x)[:2]

        # Optionally save things
        if name is not None:
            canvas.save(f"outputs_drawing/{name}.png")
            with open(f"outputs_drawing/{name}.svg", "w") as f:
                f.write(svg)
            # Save coordinates - for shape objects, we need to handle them differently
            if isinstance(next(iter(all_coords.values())), list):
                # Convert shape objects to a format that can be saved
                save_coords = {}
                for color_string, shapes in all_coords.items():
                    save_coords[color_string] = {
                        "shapes": [
                            {
                                "type": type(shape).__name__,
                                "data": {
                                    "p0": shape.p0.tolist() if hasattr(shape, "p0") else None,
                                    "p1": shape.p1.tolist() if hasattr(shape, "p1") else None,
                                    "p2": shape.p2.tolist() if hasattr(shape, "p2") else None,
                                    "p3": shape.p3.tolist() if hasattr(shape, "p3") else None,
                                    "center": shape.center.tolist()
                                    if hasattr(shape, "center")
                                    else None,
                                    "radius": shape.radius if hasattr(shape, "radius") else None,
                                    "coords": shape.coords.tolist()
                                    if hasattr(shape, "coords")
                                    else None,
                                },
                            }
                            for shape in shapes
                        ]
                    }
                np.savez(f"outputs_drawing/{name}.npz", **save_coords)
            else:
                np.savez(f"outputs_drawing/{name}.npz", **all_coords)
            json.dump(
                {"border_lengths": border_lengths, "target_y": target_y, "target_x": target_x},
                open(f"outputs_drawing/{name}.json", "w"),
            )

        return canvas, svg, all_coords, border_lengths, target_y, target_x

    def get_best_shape(
        self,
        image: Float[Arr, "y x"],
        start_dir: Float[Arr, "2"],
        start_coords: Float[Arr, "2"],
        end_coords: Float[Arr, "2"] | None = None,
        use_bounds: bool = True,
    ) -> tuple[
        Union[BezierCurve, Circle, PiecewiseLinear], Float[Arr, "2 n_pixels"], Float[Arr, "2"]
    ]:
        # Get our random parameterized shapes

        coords_list = self.shape.get_drawing_coords_list(
            n_shapes=self.n_random,
            start_dir=start_dir,
            start_coords=start_coords,
            canvas_y=image.shape[0],
            canvas_x=image.shape[1],
            outer_bound=self.outer_bound if use_bounds else None,
            inner_bound=self.inner_bound if use_bounds else None,
            end_coords=end_coords,
        )

        # Turn them into integer pixels, and concat them
        pixels = []
        n_pixels = []
        for coords, _, _ in coords_list:
            if isinstance(coords, (BezierCurve, Circle, PiecewiseLinear)):
                # For shape objects, interpolate points for pixel operations
                if isinstance(coords, BezierCurve):
                    chord_length = np.linalg.norm(coords.p3 - coords.p0)
                    control_length = (
                        np.linalg.norm(coords.p1 - coords.p0)
                        + np.linalg.norm(coords.p2 - coords.p1)
                        + np.linalg.norm(coords.p3 - coords.p2)
                    )
                    line_length = (chord_length + control_length) / 2
                elif isinstance(coords, Circle):
                    line_length = 2 * np.pi * coords.radius
                else:  # PiecewiseLinear
                    total_length = 0
                    for i in range(coords.coords.shape[1] - 1):
                        total_length += np.linalg.norm(
                            coords.coords[:, i + 1] - coords.coords[:, i]
                        )
                    line_length = total_length

                num_steps = 1 + max(1, int(line_length))
                interpolated_coords = coords.interpolate_points(num_steps)
                interpolated_coords = interpolated_coords[
                    :,
                    (interpolated_coords >= 0).all(axis=0)
                    & (interpolated_coords < np.array(image.shape)[:, None]).all(axis=0),
                ].astype(np.int32)
                pixels.append(interpolated_coords)
                n_pixels.append(interpolated_coords.shape[1])
            else:
                pixels.append(coords.astype(np.int32))
                n_pixels.append(coords.shape[1])

        pixels = np.stack([pad_to_length(p, max(n_pixels)) for p in pixels])  # (n_rand, 2, n_pix)

        # Get the pixels values of the target image at these coords
        pixel_values = image[pixels[:, 0], pixels[:, 1]]  # (n_rand, n_pix)
        pixel_values_mask = np.any(pixels != 0, axis=1)  # (n_rand, n_pix)

        # Apply negative penalty and weighting
        if self.negative_penalty > 0.0:
            # pixel_values[pixel_values < 0.0] *= 1 + self.negative_penalty
            pixel_values -= self.negative_penalty * np.maximum(0.0, self.darkness - pixel_values)

        if self.target.weight_image is not None:
            pixel_weights = self.target.weight_image[pixels[:, 0], pixels[:, 1]]  # (n_rand, n_pix)
            pixel_values_mask = pixel_values_mask.astype(pixel_values.dtype) * pixel_weights

        # Average over each pixel array
        pixel_values = (pixel_values * pixel_values_mask).sum(-1) / (
            pixel_values_mask.sum(-1) + 1e-8
        )

        # Pick the darkest shape to draw
        best_idx = np.argmax(pixel_values)
        return coords_list[best_idx]

    def make_canvas_and_crop_coords(
        self,
        all_coords: dict[
            str, Union[Float[Arr, "n_pixels 2"], list[Union[BezierCurve, Circle, PiecewiseLinear]]]
        ],
        target_y: int,
        target_x: int,
        bounding_x: tuple[float, float] = (0.0, 1.0),
        bounding_y: tuple[float, float] = (0.0, 1.0),
        fractions: float | dict[str, float] | None = None,
    ) -> tuple[Image.Image, dict[str, Float[Arr, "n_pixels 2"]], str]:
        """
        Function which makes a canvas to display, and at the same time crops coordinates & gives us
        rescaled coords (in 0-1 range) to be used for GCode generation. Also returns an SVG
        representation with a transparent background.
        """

        all_coords_rescaled = {}

        output_x = self.target.output_x
        output_y = int(output_x * target_y / target_x)
        output_sf = output_x / target_x

        size = np.array([target_y, target_x]) - 1
        bounding_min = np.array([bounding_y[0], bounding_x[0]])
        bounding_max = np.array([bounding_y[1], bounding_x[1]])
        bounding_lengths = bounding_max - bounding_min
        output_size = np.array([output_y * bounding_lengths[0], output_x * bounding_lengths[1]])

        canvas = Image.new("RGB", (int(output_size[1]), int(output_size[0])), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        svg_lines = []

        for color_string, coords in all_coords.items():
            assert isinstance(coords, list), "Should all be lists by now?"

            # Possibly crop the coordinates down to a subset of them
            if fractions is not None:
                fraction = fractions if isinstance(fractions, float) else fractions[color_string]
                coords = coords[: int(len(coords) * fraction)].copy()

            for shape in coords:
                scaled_shape = shape.scale(output_sf)
                if isinstance(shape, BezierCurve):
                    svg_lines.append(
                        f'<path d="{scaled_shape.to_svg_path()}" stroke="{color_string}" fill="none"/>'
                    )

                    # For canvas drawing, interpolate points and draw line segments
                    chord_length = np.linalg.norm(scaled_shape.p3 - scaled_shape.p0)
                    control_length = (
                        np.linalg.norm(scaled_shape.p1 - scaled_shape.p0)
                        + np.linalg.norm(scaled_shape.p2 - scaled_shape.p1)
                        + np.linalg.norm(scaled_shape.p3 - scaled_shape.p2)
                    )
                    line_length = (chord_length + control_length) / 2
                    num_steps = 1 + max(1, int(line_length))
                    curve_points = scaled_shape.interpolate_points(num_steps)

                    # Draw line segments on canvas
                    for i in range(curve_points.shape[1] - 1):
                        y0, x0 = curve_points[:, i]
                        y1, x1 = curve_points[:, i + 1]
                        draw.line([(x0, y0), (x1, y1)], fill=color_string, width=1)

                elif isinstance(shape, Circle):
                    svg_lines.append(scaled_shape.to_svg_element(color_string))

                    # For canvas drawing, interpolate points and draw circle
                    num_steps = 1 + max(1, int(2 * np.pi * scaled_shape.radius))
                    circle_points = shape.interpolate_points(num_steps)
                    circle_points = output_sf * (circle_points - (size * bounding_min)[:, None])

                    # Draw circle segments on canvas
                    for i in range(circle_points.shape[1] - 1):
                        y0, x0 = circle_points[:, i]
                        y1, x1 = circle_points[:, i + 1]
                        draw.line([(x0, y0), (x1, y1)], fill=color_string, width=1)

                elif isinstance(shape, PiecewiseLinear):
                    # Scale and offset the piecewise linear shape
                    # scaled_coords = output_sf * (shape.coords - (size * bounding_min)[:, None])
                    scaled_coords = output_sf * shape.coords

                    # Add SVG path element
                    scaled_shape = PiecewiseLinear(coords=scaled_coords)
                    svg_lines.append(
                        f'<path d="{scaled_shape.to_svg_path()}" stroke="{color_string}" fill="none"/>'
                    )

                    # For canvas drawing, interpolate points and draw line segments
                    total_length = 0
                    for i in range(scaled_coords.shape[1] - 1):
                        total_length += np.linalg.norm(
                            scaled_coords[:, i + 1] - scaled_coords[:, i]
                        )
                    num_steps = 1 + max(1, int(total_length))
                    line_points = scaled_shape.interpolate_points(num_steps)

                    # Draw line segments on canvas
                    for i in range(line_points.shape[1] - 1):
                        y0, x0 = line_points[:, i]
                        y1, x1 = line_points[:, i + 1]
                        draw.line([(x0, y0), (x1, y1)], fill=color_string, width=1)

            # For rescaled coordinates, we need to interpolate all shapes
            all_shape_points = []
            for shape in coords:
                # Estimate length for interpolation
                if isinstance(shape, BezierCurve):
                    chord_length = np.linalg.norm(shape.p3 - shape.p0)
                    control_length = (
                        np.linalg.norm(shape.p1 - shape.p0)
                        + np.linalg.norm(shape.p2 - shape.p1)
                        + np.linalg.norm(shape.p3 - shape.p2)
                    )
                    line_length = (chord_length + control_length) / 2
                elif isinstance(shape, Circle):
                    line_length = 2 * np.pi * shape.radius
                else:  # PiecewiseLinear
                    total_length = 0
                    for i in range(shape.coords.shape[1] - 1):
                        total_length += np.linalg.norm(shape.coords[:, i + 1] - shape.coords[:, i])
                    line_length = total_length

                num_steps = 1 + max(1, int(line_length))
                shape_points = shape.interpolate_points(num_steps)
                all_shape_points.append(shape_points.T)  # Transpose to (n_points, 2)

            if all_shape_points:
                # Concatenate all shape points
                all_points = np.concatenate(all_shape_points, axis=0)
                all_coords_rescaled[color_string] = all_points / output_size.max()
            else:
                all_coords_rescaled[color_string] = np.empty((0, 2))

        # Resize canvas to height 500
        canvas_y = 500
        canvas_x = (
            500
            * (output_x * (bounding_x[1] - bounding_x[0]))
            / (output_y * (bounding_y[1] - bounding_y[0]))
        )
        canvas = canvas.resize((int(canvas_x), canvas_y))

        # Check all_coords_rescaled is within the bounding box
        min_y, min_x, max_y, max_x = _get_min_max_coords(all_coords_rescaled)

        print(
            f"  Bounding box (rescaled, inner):  [{min_x:.6f}-{max_x:.6f}, {min_y:.6f}-{max_y:.6f}], AR = {(max_y - min_y) / (max_x - min_x):.6f}"
        )

        svg = f'<svg width="{int(output_size[1])}" height="{int(output_size[0])}" xmlns="http://www.w3.org/2000/svg" style="background-color: transparent;">{"".join(svg_lines)}</svg>'

        return canvas, svg, all_coords_rescaled, svg


def get_closest_point_on_border(
    coords: Float[Arr, "2"], max_dim_0: int, max_dim_1: int, min_dim_0: int = 0, min_dim_1: int = 0
) -> tuple[int, Float[Arr, "2"]]:
    border_diffs = [
        max_dim_0 - coords[0],
        coords[1] - min_dim_1,
        coords[0] - min_dim_0,
        max_dim_1 - coords[1],
    ]
    assert min(border_diffs) >= 0, (
        f"Coords {coords} are out of bounds {min_dim_0}-{max_dim_0}, {min_dim_1}-{max_dim_1}"
    )
    closest_border = np.argmin(border_diffs).item()
    return closest_border, [
        np.array([max_dim_0, coords[1]]),
        np.array([coords[0], min_dim_1]),
        np.array([min_dim_0, coords[1]]),
        np.array([coords[0], max_dim_1]),
    ][closest_border]


def get_color_string(color: tuple[int, int, int]):
    color_string = {
        (0, 0, 0): "black",
        (0, 215, 225): "aqua",
        (0, 120, 240): "dodgerblue",
        (0, 0, 128): "darkblue",
        (255, 255, 255): "white",
        (255, 0, 0): "red",
    }.get(color, None)

    if color_string is None:
        raise ValueError(f"Color {color} not found in color string")

    return color_string


def return_to_origin(
    x: float,
    y: float,
    side: int,  # 0 = right, moving anticlockwise
    include_start: bool = True,
) -> list[tuple[float, float]]:
    """Returns a list of (y, x) tuples which traces a path back to the origin."""
    path = [(x, y), (x, 0) if side % 2 == 0 else (0, y), (0, 0)]
    return path if include_start else path[1:]


def mask_coords(
    coords: Float[Arr, "2 n_pixels"],
    max_y: int,
    max_x: int,
    outer_bound: float | None,
    inner_bound: float | None,
    remove: bool = False,
) -> Float[Arr, "2 n_pixels"]:
    """Masks coordinates that go out of bounds."""
    assert coords.shape[0] == 2, "Coords should have shape (2, n_pixels)"

    # Return empty array if either (1) ANY pixels are too far out of bounds or (2) we END too close to an edge
    if outer_bound is not None:
        max_out_of_bounds = np.max(
            [
                -coords[0].min() / max_y,
                (coords[0].max() - max_y) / max_y,
                -coords[1].min() / max_x,
                (coords[1].max() - max_x) / max_x,
            ]
        )
        if max_out_of_bounds > outer_bound:
            return coords[:, :0]

    if inner_bound is not None:
        end_out_of_bounds = np.max(
            [
                -coords[0, -1] / max_y,
                (coords[0, -1] - max_y) / max_y,
                -coords[1, -1] / max_x,
                (coords[1, -1] - max_x) / max_x,
            ]
        )
        if end_out_of_bounds > -inner_bound:
            return coords[:, :0]

    # Remove all out of bounds coordinates
    out_of_bounds = (coords[0] < 0) | (coords[0] >= max_y) | (coords[1] < 0) | (coords[1] >= max_x)
    coords = coords[:, ~out_of_bounds] if remove else np.where(out_of_bounds, 0.0, coords)
    return coords


def pad_to_length(arr: np.ndarray, length: int, axis: int = -1, fill_value: float = 0):
    target_shape = list(arr.shape)
    assert length >= target_shape[axis]
    target_shape[axis] = length - target_shape[axis]
    return np.concatenate(
        [arr, np.full_like(arr, fill_value=fill_value, shape=tuple(target_shape))], axis=axis
    )


def FS_dither(
    image: Int[Arr, "y x 3"],
    palette: list[tuple[int, int, int]],
    pixels_per_batch: int = 32,
    num_overlap_rows: int = 6,
) -> tuple[Float[Arr, "y x 3"], float]:
    t0 = time.time()

    image_dithered: Float[Arr, "y x 3"] = image.astype(np.float32)
    y, x = image_dithered.shape[:2]

    num_batches = math.ceil(y / pixels_per_batch)
    rows_to_extend_by = num_batches - (y % num_batches)

    # Add a batch dimension
    image_dithered = einops.rearrange(
        np.concatenate([image_dithered, np.zeros((rows_to_extend_by, x, 3))]),
        "(batch y) x rgb -> y x batch rgb",
        batch=num_batches,
    )
    # Concat the last `num_overlap_rows` to the start of the image
    end_of_each_batch = np.concatenate(
        [np.zeros((num_overlap_rows, x, 1, 3)), image_dithered[-num_overlap_rows:, :, :-1]], axis=-2
    )
    image_dithered = np.concatenate([end_of_each_batch, image_dithered], axis=0)

    image_dithered = FS_dither_batch(image_dithered, palette)

    image_dithered = einops.rearrange(
        image_dithered[num_overlap_rows - 1 : -1],
        "y x batch rgb -> (batch y) x rgb",
    )[1 : y + 1]

    print(f"FS dithering complete in {time.time() - t0:.2f}s")

    return image_dithered


def FS_dither_batch(
    image_dithered: Float[Arr, "y x batch 3"],
    palette: list[tuple[int, int, int]],
) -> Int[Arr, "y x batch 3"]:
    # Define the constants we'll multiply with when "shifting the errors" in dithering
    AB = np.array([3, 5]) / 16
    ABC = np.array([3, 5, 1]) / 16
    BC = np.array([5, 1]) / 16

    palette = np.array(palette)  # [palette 3]

    # Set up stuff
    palette_sq = einops.rearrange(palette, "palette rgb -> palette 1 rgb")
    y, x, batch = image_dithered.shape[:3]
    is_clamp = True

    # loop over each row, from first to second last
    for y_ in range(y - 1):
        row = image_dithered[y_].astype(np.float32)  # [x batch 3]
        next_row = np.zeros_like(row)  # [x batch 3]

        # deal with the first pixel in the row
        old_color = row[0]  # [batch 3]
        color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)  # [palette batch]
        color = palette[color_diffs.argmin(axis=0)]  # [batch 3]
        color_diff = old_color - color  # [batch 3]
        row[0] = color
        row[1] += (7 / 16) * color_diff
        next_row[[0, 1]] += einops.einsum(BC, color_diff, "two, batch rgb -> two batch rgb")

        # loop over each pixel in the row, from second to second last
        for x_ in range(1, x - 1):
            old_color = row[x_]  # [batch 3]
            color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)  # [colors batch]
            color = palette[color_diffs.argmin(axis=0)]
            color_diff = old_color - color
            row[x_] = color
            row[x_ + 1] += (7 / 16) * color_diff
            next_row[[x_ - 1, x_, x_ + 1]] += einops.einsum(
                ABC, color_diff, "three, batch rgb -> three batch rgb"
            )

        # deal with the last pixel in the row
        old_color = row[-1]
        color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)
        color = palette[color_diffs.argmin(axis=0)]
        color_diff = old_color - color
        row[-1] = color
        next_row[[-2, -1]] += einops.einsum(AB, color_diff, "two, batch rgb -> two batch rgb")

        # update the rows, i.e. changing current row and propagating errors to next row
        image_dithered[y_] = np.clip(row, 0, 255)
        image_dithered[y_ + 1] += next_row
        if is_clamp:
            image_dithered[y_ + 1] = np.clip(image_dithered[y_ + 1], 0, 255)

    # deal with the last row
    row = image_dithered[-1]
    for x_ in range(x - 1):
        old_color = row[x_]
        color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)
        color = palette[color_diffs.argmin(axis=0)]
        color_diff = old_color - color
        row[x_] = color
        row[x_ + 1] += color_diff

    # deal with the last pixel in the last row
    old_color = row[-1]
    color_diffs = ((palette_sq - old_color) ** 2).sum(axis=-1)
    color = palette[color_diffs.argmin(axis=0)]
    row[-1] = color
    if is_clamp:
        row = np.clip(row, 0, 255)
    image_dithered[-1] = row
    # pbar.close()

    return image_dithered.astype(np.int32)


def _get_min_max_coords(
    coords: dict[Any, Float[Arr, "2 n_pixels"]],
) -> tuple[float, float, float, float]:
    min_dim_0 = min(coords[:, 0].min().item() for coords in coords.values())
    min_dim_1 = min(coords[:, 1].min().item() for coords in coords.values())
    max_dim_0 = max(coords[:, 0].max().item() for coords in coords.values())
    max_dim_1 = max(coords[:, 1].max().item() for coords in coords.values())
    return min_dim_0, min_dim_1, max_dim_0, max_dim_1


def make_gcode(
    all_coords: dict[str, Int[Arr, "n_coords 2"]],
    image_bounding_box: tuple[float, float],  # area we want to place `all_coords` within
    gcode_bounding_box: tuple[float, float],  # drawing area for gcode
    border_lengths: dict[
        str, tuple[int, int]
    ],  # amount of pixels we draw with pen-up from start & end of each colour
    margin: float = 0.0,
    tiling: tuple[int, int] = (1, 1),
    speed: int = 10_000,
    plot_gcode: bool = False,
    rotate: bool = False,
    pen_height: int = 450,
    demo: bool = False,  # turn this on and we only print the first 1000 lines, with all pen up & half speed
) -> dict[str, list[str]]:
    """
    Generates G-code for multiple different copies of the image.
    """
    gcode_all = defaultdict(list)
    times_all = defaultdict(list)

    # We assume bounding box has been zeroed
    gcode_bounding_box = [(0.0, 0.0), gcode_bounding_box]
    (x0, y0), (x1, y1) = gcode_bounding_box

    # Start by changing (y, x) representations to (x, y)
    all_coords = {k: v[:, ::-1] for k, v in all_coords.items()}
    image_bounding_box = tuple(image_bounding_box)[::-1]

    # Optionally rotate (we do this if the bounding box aspect ratio makes this favourable)
    if rotate:
        for color, coords in all_coords.items():
            all_coords[color] = np.array([coords[:, 1], image_bounding_box[0] - coords[:, 0]]).T
        image_bounding_box = tuple(image_bounding_box)[::-1]
    else:
        for color, coords in all_coords.items():
            all_coords[color] = np.array([coords[:, 0], image_bounding_box[1] - coords[:, 1]]).T

    # After the flipping and optional rotations, check our bounds are still valid
    all_coords_max = np.max(np.stack([v.max(axis=0) for v in all_coords.values()]), axis=0)
    assert np.all(all_coords_max <= image_bounding_box), (
        f"Out of bounds: {all_coords_max} > {image_bounding_box}"
    )

    for x_iter in range(tiling[0]):
        for y_iter in range(tiling[1]):
            print(f"{x_iter}, {y_iter}")
            _x0 = x0 + (x1 - x0) * x_iter / tiling[0]
            _y0 = y0 + (y1 - y0) * y_iter / tiling[1]
            _x1 = x0 + (x1 - x0) * (x_iter + 1) / tiling[0]
            _y1 = y0 + (y1 - y0) * (y_iter + 1) / tiling[1]
            _bounding_box = ((_x0, _y0), (_x1 - margin, _y1 - margin))

            gcode, times = make_gcode_single(
                all_coords,
                image_bounding_box=image_bounding_box,
                gcode_bounding_box=_bounding_box,
                border_lengths=border_lengths,
                speed=speed,
                pen_height=pen_height,
            )
            for k in ["bounding_box"] + list(all_coords.keys()):
                gcode_all[k].extend(gcode[k])
                times_all[k].append(times[k])

    print()
    for color, times in times_all.items():
        print(
            f"{color:<13} ... time = sum({', '.join(f'{t:05.2f}' for t in times)}) = {sum(times):.2f} minutes"
        )

    if demo:
        for color, gcode in gcode_all.items():
            gcode_all[color] = gcode[:1000]
            gcode_all[color] = [
                g.replace(f"F{speed}", f"F{int(speed / 2)}")
                for g in gcode_all[color]
                if not g.startswith("M3S0")
            ]

    if plot_gcode:
        output_area = 600 * 600
        output_x = (output_area * (x1 - x0) / (y1 - y0)) ** 0.5
        output_y = (output_area * (y1 - y0) / (x1 - x0)) ** 0.5
        ((x0, y0), (x1, y1)) = gcode_bounding_box
        sf = output_x / (x1 - x0)
        canvas_all = Image.new("RGB", (int(output_x), int(output_y)), (255, 255, 255))
        draw_all = ImageDraw.Draw(canvas_all)
        # for color, all_lines in coords_gcode_scale_all.items():
        for color, gcode in gcode_all.items():
            all_lines = _create_coords_from_gcode(gcode)
            canvas = Image.new("RGB", (int(output_x), int(output_y)), (255, 255, 255))
            for lines, pen_up in all_lines:
                draw = ImageDraw.Draw(canvas)
                points = list(zip(sf * (lines[:, 0] - x0), sf * (y1 - lines[:, 1])))
                width = 4 if pen_up else 1
                fill = "#aaa" if pen_up else "black" if color == "bounding_box" else color
                draw.line(points, fill=fill, width=width)
                draw_all.line(points, fill=fill, width=width)
            display(ImageOps.expand(canvas, border=(3, 0, 0, 3), fill="white"))
        display(ImageOps.expand(canvas_all, border=(3, 0, 0, 3), fill="white"))

    return gcode_all


def make_gcode_single(
    all_coords: dict[tuple[int, int, int], Int[Arr, "n_coords 2"]],
    image_bounding_box: tuple[float, float],
    gcode_bounding_box: tuple[tuple[float, float], tuple[float, float]],
    border_lengths: dict[str, tuple[int, int]],
    speed: int = 10_000,
    pen_height: int = 400,
) -> dict[tuple[int, int, int], list[str]]:
    """
    Creates G-code for a single tile image. This gets concatenated for multiple tiles.
    """

    all_coords = {k: v.copy() for k, v in all_coords.items()}

    # Figure out which side each color starts and ends on
    start_end_sides = {}
    for color, coords in all_coords.items():
        start_end_sides[color] = {}
        for side_type, side_idx in zip(("start", "end"), (0, -1)):
            coord = coords[side_idx]
            border, border_coord = get_closest_point_on_border(coord, *image_bounding_box)
            assert np.linalg.norm(coord - border_coord) < 3.0, (
                f"Found unexpected coord: {coord} with diff to border {border_coord} of {np.linalg.norm(coord - border_coord):.3f}"
            )
            start_end_sides[color][side_type] = border % 4

    # Print out the bounding box (in original coordinates), to get a sanity check on how much of the space we're using.
    min_x, min_y, max_x, max_y = _get_min_max_coords(all_coords)
    print(
        f"  Bounding box (orig, outer):  [{0.0:.3f}-{image_bounding_box[0]:.3f}, {0.0:.3f}-{image_bounding_box[1]:.3f}], AR = {(image_bounding_box[1]) / image_bounding_box[0]:.3f}"
    )
    print(
        f"  Bounding box (orig, inner):  [{min_x:.3f}-{max_x:.3f}, {min_y:.3f}-{max_y:.3f}], AR = {(max_y - min_y) / (max_x - min_x):.3f}"
    )

    # Rescale coordinates to fit within GCode bounding box (we scale as large as possible while staying inside it)
    (x0, y0), (x1, y1) = gcode_bounding_box
    sf = min((x1 - x0) / image_bounding_box[0], (y1 - y0) / image_bounding_box[1])
    all_coords_gcode_scale = {
        color: np.array([x0, y0]) + coords * sf for color, coords in all_coords.items()
    }

    # Print new bounding box, in GCode terms (note that we still use the originally provided gcode bounding box
    # rather than cropping further, since we might want the empty space!).
    min_x, min_y, max_x, max_y = _get_min_max_coords(all_coords_gcode_scale)
    all_coords_gcode_scale["bounding_box"] = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    )
    print(
        f"  Bounding box (outer): [{x0:06.2f}-{x1:06.2f}, {y0:06.2f}-{y1:06.2f}], AR = {(y1 - y0) / (x1 - x0):.3f}"
    )
    print(
        f"  Bounding box (inner): [{min_x:06.2f}-{max_x:06.2f}, {min_y:06.2f}-{max_y:06.2f}], AR = {(max_y - min_y) / (max_x - min_x):.3f}"
    )

    # Create dicts to store gcode and time for each colour to be drawn
    gcode = {}
    times = {}

    # Fill in bounding box lines, i.e. just moving around the corners of the bounding box
    gcode["bounding_box"] = ["M3S250 ; raise"]
    # for x, y in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]:
    for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
        gcode["bounding_box"].append(f"G1 X{x:.3f} Y{y:.3f} F{speed}")

    # For all colors, update the gcode movements so that they follow this pattern:
    #   (1) Raise pen
    #   (2) Move from origin to starting position
    #   (3) Lower pen
    #   (4) Draw (i.e. the coords_list we've already computed for this color)
    #   (5) Raise pen
    #   (6) Move back to origin
    for color, coords_list in all_coords_gcode_scale.items():
        border_start, border_end = border_lengths.get(color, (0, 0))
        border_end = len(coords_list) - 1 - border_end

        # (1, 2) Raise pen & move to starting position
        gcode[color] = [f"M3S{pen_height} ; raise (before moving to starting position)"]
        start_xy_seq = return_to_origin(
            x=coords_list[0, 0],
            y=coords_list[0, 1],
            side=2 if color == "bounding_box" else start_end_sides[color]["start"],
            include_start=False,
        )[::-1]
        gcode[color].extend([f"G1 X{x:.3f} Y{y:.3f} F{speed}" for x, y in start_xy_seq])

        # (4) Add all the drawing coordinates (this includes step 3 & 5, lowering & raising)
        x = y = None
        for i, (x, y) in enumerate(coords_list):
            gcode[color].append(f"G1 X{x:.3f} Y{y:.3f} F{speed}")
            if i == border_start:
                gcode[color].append("M3S0 ; lower (to start drawing)")
            elif i == border_end:
                gcode[color].append(f"M3S{pen_height} ; raise (to end drawing)")

        # (6) End the drawing by moving back to the origin
        end_xy_seq = return_to_origin(
            x=coords_list[-1, 0],
            y=coords_list[-1, 1],
            side=2 if color == "bounding_box" else start_end_sides[color]["end"],
            include_start=False,
        )
        gcode[color].extend([f"G1 X{x:.3f} Y{y:.3f} F{speed}" for x, y in end_xy_seq])

        # Filter GCode to remove any duplicates which are adjacent to each other
        duplicate_indices = set(
            [i for i, (g0, g1) in enumerate(zip(gcode[color][:-1], gcode[color][1:])) if g0 == g1]
        )
        if duplicate_indices:
            gcode[color] = [g for i, g in enumerate(gcode[color]) if i not in duplicate_indices]

        # Update normalized coords to reflect the journey to & from the origin (including whether some of the
        # start and end coords were actually pen-up)
        all_coords_gcode_scale[color] = [
            (False, np.concatenate([np.array(start_xy_seq), coords_list[:border_start]])),
            (True, coords_list[border_start:border_end]),
            (False, np.concatenate([coords_list[border_end:], np.array(end_xy_seq)])),
        ]
        # print(color, border_start, border_end, border_lengths, len(coords_list))

        # Print total time this will take
        coords_concatenated = np.concatenate(
            [coords for _, coords in all_coords_gcode_scale[color]]
        )
        distances = np.linalg.norm(np.diff(coords_concatenated, axis=0), axis=1)
        distance_for_one_minute = 1400  # TODO - improve this estimate
        times[color] = distances.sum() / distance_for_one_minute

    return gcode, times


def _create_coords_from_gcode(gcode: list[str]) -> list[tuple[Float[Arr, "length 2"], bool]]:
    """Creates list of coords (and whether pen is up/down) from GCode. Good for sanity checking!"""

    assert gcode[0].startswith("M3S"), "GCode should always start with raise/lower op, to be sure!"

    coords, segments, pen_up = [], [], True

    for cmd in gcode:
        if cmd.startswith("M3S"):
            if coords:
                segments.append((np.array(coords), pen_up))
                coords = [coords[-1]] if coords else []
            pen_up = not cmd.startswith("M3S0")
        elif cmd.startswith("G1"):
            x, y = re.findall(r"X([\d.-]+) Y([\d.-]+)", cmd)[0]
            coords.append([float(x), float(y)])

    if coords:
        segments.append((np.array(coords), pen_up))

    return segments

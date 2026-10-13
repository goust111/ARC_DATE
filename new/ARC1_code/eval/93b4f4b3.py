import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg
import itertools
import random
from functools import lru_cache


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    offsets = tuple(
        sorted((int(r - r0), int(c - c0)) for r, c in coords)
    )
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), offsets


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each left-template hole with the color of the right-side object whose shape matches that hole.

    Input structure
    The input is split into two equal-width panels.  The left panel is a single-color
    template on the background, with disconnected background hole-regions; the right
    panel contains one monochromatic object for each hole-region.

    Rule
    Return the left panel only.  For every background hole-region in the left panel,
    normalize its mask to its bounding box, find the unique right-panel object with
    the same normalized mask, and recolor that hole-region with the object's color.

    Key insight
    The right-panel objects are not copied by position; their shapes are a palette
    that names which color belongs in each matching left-panel hole.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    assert w % 2 == 0
    mid = w // 2
    bg = get_bg(g)
    left = g[:, :mid]
    right = g[:, mid:]

    frame_colors = set(map(int, np.unique(left))) - {bg}
    assert len(frame_colors) == 1

    shape_to_color: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}
    for obj in find_components(right, bg=bg, connectivity=4, monochromatic=True):
        key = _shape_key(obj["coords"])
        assert key not in shape_to_color
        shape_to_color[key] = int(obj["color"])

    slots = components_mask(left == bg, connectivity=4)
    assert len(slots) == len(shape_to_color)

    out = left.copy()
    for coords in slots:
        key = _shape_key(coords)
        assert key in shape_to_color
        out[coords[:, 0], coords[:, 1]] = shape_to_color[key]

    assert not np.any(out == bg)
    return out

# --------------- generator (o5x) ---------------

PANEL_WIDTH_RANGE = (5, 6)  # Canvas width per panel.
BAND_COUNT_RANGE = (3, 4)  # Number of horizontal bands.
BAND_HEIGHT_RANGE = (2, 3)  # Uniform inner height per band.
FRAME_THICKNESS = 1  # Border and separator thickness.
SHAPE_MIN_WIDTH = 2  # Shape bboxes are at least this wide.
SHAPE_MIN_FILL_RATIO = 0.55  # Minimum density inside each shape bbox.
SHAPE_CONNECTIVITY = 4  # Holes and right-side objects are 4-connected.
BACKGROUND_COLOR = 0
COLOR_POOL = range(1, 10)
MAX_GENERATION_ATTEMPTS = 300


def _gen_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    r0, c0, _, _ = bbox_coords(coords)
    offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return (int(mask.shape[0]), int(mask.shape[1])), offsets


@lru_cache(None)
def _gen_shape_catalog(height: int, max_width: int) -> tuple[tuple[int, int, tuple, np.ndarray], ...]:
    catalog = []
    for width in range(SHAPE_MIN_WIDTH, max_width + 1):
        min_area = int(np.ceil(SHAPE_MIN_FILL_RATIO * height * width))
        n_cells = height * width
        for bits in range(1, 1 << n_cells):
            area = bits.bit_count()
            if area < min_area:
                continue
            mask = np.array(
                [(bits >> i) & 1 for i in range(n_cells)], dtype=bool
            ).reshape(height, width)
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if len(components_mask(mask, connectivity=SHAPE_CONNECTIVITY)) != 1:
                continue
            catalog.append((width, area, _gen_mask_key(mask), mask))
    return tuple(catalog)


def _gen_choose_shapes(height: int, inner_width: int, count: int) -> list[np.ndarray] | None:
    groups = {}
    for width, area, key, mask in _gen_shape_catalog(height, inner_width):
        groups.setdefault((width, area), []).append((key, mask))

    chosen = []
    used_keys = set()
    for _ in range(count):
        available = [
            [item for item in items if item[0] not in used_keys]
            for items in groups.values()
        ]
        available = [items for items in available if items]
        if not available:
            return None
        key, mask = random.choice(random.choice(available))
        used_keys.add(key)
        chosen.append(mask.copy())
    return chosen


@lru_cache(None)
def _gen_derangements(count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(perm for perm in itertools.permutations(range(count)) if all(i != perm[i] for i in range(count)))


def _gen_layout() -> tuple[int, int, int, int, int]:
    """Choose panel dimensions and return derived height and inner width."""
    panel_width = random.randint(*PANEL_WIDTH_RANGE)
    n_bands = random.randint(*BAND_COUNT_RANGE)
    band_height = random.randint(*BAND_HEIGHT_RANGE)
    height = n_bands * (band_height + FRAME_THICKNESS) + FRAME_THICKNESS
    inner_width = panel_width - 2 * FRAME_THICKNESS
    return panel_width, n_bands, band_height, height, inner_width


def _gen_palette(n_bands: int) -> tuple[int, list[int]]:
    """Choose one frame color and one distinct fill color per band."""
    colors = random.sample(COLOR_POOL, n_bands + 1)
    return colors[0], colors[1:]


def _gen_render_panels(panel_width, height, band_height, masks, frame_color, fill_colors):
    """Render expected output, input-left holes, and deranged input-right objects."""
    output = np.full((height, panel_width), frame_color, dtype=int)
    left = output.copy()

    for band_i, mask in enumerate(masks):
        r0 = band_i * (band_height + FRAME_THICKNESS) + FRAME_THICKNESS
        c0 = random.randint(FRAME_THICKNESS, panel_width - mask.shape[1] - FRAME_THICKNESS)
        rr, cc = np.where(mask)
        output[r0 + rr, c0 + cc] = fill_colors[band_i]
        left[r0 + rr, c0 + cc] = BACKGROUND_COLOR

    right = np.zeros((height, panel_width), dtype=int)
    perm = random.choice(_gen_derangements(len(masks)))
    for band_i, shape_i in enumerate(perm):
        mask = masks[shape_i]
        r0 = band_i * (band_height + FRAME_THICKNESS) + FRAME_THICKNESS
        c0 = random.randint(FRAME_THICKNESS, panel_width - mask.shape[1] - FRAME_THICKNESS)
        rr, cc = np.where(mask)
        right[r0 + rr, c0 + cc] = fill_colors[shape_i]
    return output, left, right


def _gen_validated_input(inp, output):
    """Return the candidate input only when the solver reproduces the oracle output."""
    try:
        return inp if np.array_equal(main(inp), output) else None
    except Exception:
        return None


def _gen_attempt() -> np.ndarray | None:
    panel_width, n_bands, band_height, height, inner_width = _gen_layout()
    frame_color, fill_colors = _gen_palette(n_bands)

    masks = _gen_choose_shapes(band_height, inner_width, n_bands)
    if masks is None:
        return None

    output, left, right = _gen_render_panels(
        panel_width, height, band_height, masks, frame_color, fill_colors
    )
    return _gen_validated_input(np.concatenate([left, right], axis=1), output)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise ValueError("failed to generate valid input")

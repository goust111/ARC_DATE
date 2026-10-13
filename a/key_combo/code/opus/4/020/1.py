from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _install_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        candidate = base / "new" / "solvers" / "arc_utils.py"
        if candidate.exists():
            utils_dir = str(candidate.parent)
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            return

    fallback = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if fallback.exists():
        utils_dir = str(fallback.parent)
        if utils_dir not in sys.path:
            sys.path.insert(0, utils_dir)
        return

    raise ImportError("could not locate new/solvers/arc_utils.py")


_install_arc_utils_path()

from arc_utils import Color, find_components  # noqa: E402


GLYPH_COLORS = (Color.BLUE, Color.RED, Color.YELLOW, Color.GREY, Color.PINK)

GLYPH_LIBRARY: dict[int, tuple[tuple[tuple[int, ...], ...], ...]] = {
    3: (
        ((1, 0), (1, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 0)),
        ((1, 1), (0, 1)),
    ),
    4: (
        ((1, 1), (1, 1)),
        ((0, 1, 0), (1, 1, 1)),
        ((1, 0, 0), (1, 1, 1)),
        ((0, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 0, 0)),
        ((1, 1, 1), (0, 0, 1)),
    ),
    5: (
        ((1, 1, 0), (1, 1, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 0, 0, 0), (1, 1, 1, 1)),
        ((0, 1, 0, 0), (1, 1, 1, 1)),
        ((0, 0, 1, 0), (1, 1, 1, 1)),
        ((0, 0, 0, 1), (1, 1, 1, 1)),
    ),
    6: (
        ((1, 1, 1), (1, 1, 1)),
        ((1, 1, 0, 0), (1, 1, 1, 1)),
        ((0, 1, 1, 0), (1, 1, 1, 1)),
        ((0, 0, 1, 1), (1, 1, 1, 1)),
        ((1, 0, 1, 0), (1, 1, 1, 1)),
        ((0, 1, 0, 1), (1, 1, 1, 1)),
    ),
    7: (
        ((1, 1, 1, 0), (1, 1, 1, 1)),
        ((1, 1, 0, 1), (1, 1, 1, 1)),
        ((1, 0, 1, 1), (1, 1, 1, 1)),
        ((0, 1, 1, 1), (1, 1, 1, 1)),
    ),
    8: (
        ((1, 1, 1, 1), (1, 1, 1, 1)),
    ),
}

AREA_WEIGHTS = {
    3: 6,
    4: 6,
    5: 6,
    6: 4,
    7: 2,
    8: 1,
}

MODE_CHOICES = ("no_bounce", "single_bounce", "multi_bounce", "joint", "joint")
MAX_TOTAL_LENGTH = 22
MAX_GEN_ATTEMPTS = 400


def _shape_area(shape: tuple[tuple[int, ...], ...]) -> int:
    return sum(sum(row) for row in shape)


def _shape_width(shape: tuple[tuple[int, ...], ...]) -> int:
    return len(shape[0])


def _shape_to_grid(shape: tuple[tuple[int, ...], ...], color: int) -> np.ndarray:
    mask = np.array(shape, dtype=int)
    return np.where(mask == 1, int(color), Color.BLACK).astype(int)


def _decode_glyphs(grid: np.ndarray) -> list[tuple[int, int]]:
    strip = np.asarray(grid[-2:, :], dtype=int)
    comps = find_components(strip, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    comps = sorted(comps, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))

    assert 3 <= len(comps) <= 5, "expected 3 to 5 glyphs in the bottom strip"

    instructions: list[tuple[int, int]] = []
    seen_colors: set[int] = set()
    prev_c1: int | None = None

    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = map(int, comp["bbox"])
        width = c1 - c0 + 1

        assert color in GLYPH_COLORS, "glyph colors must come from the visible instruction palette"
        assert color not in seen_colors, "glyph colors must be unique"
        assert (r0, r1) == (0, 1), "every glyph must span both strip rows"
        assert 2 <= width <= 4, "every glyph must be 2 to 4 columns wide"
        if prev_c1 is not None:
            assert c0 == prev_c1 + 1, "glyphs must be flush with no gaps"

        instructions.append((int(comp["area"]), color))
        seen_colors.add(color)
        prev_c1 = c1

    return instructions


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(np.asarray(grid, dtype=int) == Color.GREEN)
    assert coords.shape == (1, 2), "expected exactly one green start marker"
    r, c = map(int, coords[0])
    assert r < grid.shape[0] - 2, "start marker must be above the glyph strip"
    assert 0 < c < grid.shape[1] - 1, "start marker cannot begin on a side wall"
    return r, c


def _trace_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    instructions: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    height, width = map(int, shape)
    r, c = map(int, start)
    dc = 1

    trace: list[tuple[int, int, int]] = []
    bounces = 0
    joint_events = 0

    for seg_idx, (length, color) in enumerate(instructions):
        for seg_step in range(int(length)):
            if c + dc < 0 or c + dc >= width:
                dc = -dc
                bounces += 1
                if seg_idx > 0 and seg_step == 0:
                    joint_events += 1

            r -= 1
            c += dc
            if r < 0 or c < 0 or c >= width or r >= height:
                raise ValueError("path left the canvas before consuming all decoded instructions")
            trace.append((r, c, int(color)))

    meta = {
        "bounces": int(bounces),
        "joint_events": int(joint_events),
        "path_length": int(len(trace)),
    }
    return trace, meta


def _render_output(grid: np.ndarray, instructions: list[tuple[int, int]], start: tuple[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    trace, _meta = _trace_path(out.shape, start, instructions)
    for r, c, color in trace:
        out[r, c] = int(color)
    out[start[0], start[1]] = Color.GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the bottom glyph strip into segment lengths and paint the reflected diagonal path.

    Input structure:
    - The grid has a black background, a single green start marker above the bottom two rows, and a bottom two-row strip of 3 to 5 touching monochrome glyphs.
    - Each glyph is a 4-connected component with a unique visible color from the instruction palette, spans both strip rows, and is 2 to 4 columns wide.
    - The glyph strip is read strictly from left to right.

    Rule:
    - For each bottom-strip glyph, count its foreground pixels and pair that count with the glyph color to obtain an ordered instruction list `(length, color)`.
    - Starting from the cell just above-right of the green marker, move one cell up per step; whenever the next horizontal move would leave the canvas, reverse only the horizontal direction before taking that step.
    - Paint exactly `length` cells with each instruction color, then continue with the next instruction while keeping the current diagonal heading; keep the green marker and the bottom glyph strip unchanged.

    Key insight:
    - The glyph geometry is only a code: its visible color selects the paint color, while its pixel count, not its width or silhouette, determines how many cells the next diagonal segment occupies.
    """
    g = np.asarray(grid, dtype=int)
    instructions = _decode_glyphs(g)
    start = _find_start(g)
    return _render_output(g, instructions, start)


def _weighted_area_choice() -> int:
    areas = list(AREA_WEIGHTS)
    weights = [AREA_WEIGHTS[area] for area in areas]
    return int(random.choices(areas, weights=weights, k=1)[0])


def _sample_two_shapes_same_area(area: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    shapes = list(GLYPH_LIBRARY[area])
    pairs = [
        (a, b)
        for i, a in enumerate(shapes)
        for b in shapes[i + 1 :]
        if _shape_width(a) != _shape_width(b)
    ]
    if pairs:
        return random.choice(pairs)
    return tuple(random.sample(shapes, 2))  # type: ignore[return-value]


def _sample_glyph_sequence() -> tuple[list[np.ndarray], list[tuple[int, int]], int] | None:
    for _ in range(200):
        glyph_count = random.randint(3, 5)
        colors = random.sample(GLYPH_COLORS, glyph_count)
        chosen_shapes: list[tuple[tuple[int, ...], ...] | None] = [None] * glyph_count

        force_repeat = glyph_count >= 3 and random.random() < 0.75
        if force_repeat:
            repeat_area = random.choice([4, 5, 6])
            a_shape, b_shape = _sample_two_shapes_same_area(repeat_area)
            i, j = random.sample(range(glyph_count), 2)
            chosen_shapes[i] = a_shape
            chosen_shapes[j] = b_shape

        for idx in range(glyph_count):
            if chosen_shapes[idx] is not None:
                continue
            area = _weighted_area_choice()
            chosen_shapes[idx] = random.choice(GLYPH_LIBRARY[area])

        shapes = [shape for shape in chosen_shapes if shape is not None]
        total_length = sum(_shape_area(shape) for shape in shapes)
        strip_width = sum(_shape_width(shape) for shape in shapes)

        if not (10 <= total_length <= MAX_TOTAL_LENGTH):
            continue
        if strip_width > 20:
            continue

        instructions = [(int(_shape_area(shape)), int(color)) for shape, color in zip(shapes, colors)]
        glyphs = [_shape_to_grid(shape, color) for shape, color in zip(shapes, colors)]
        return glyphs, instructions, strip_width
    return None


def _mode_ok(meta: dict[str, int], mode: str) -> bool:
    if mode == "no_bounce":
        return meta["bounces"] == 0
    if mode == "single_bounce":
        return meta["bounces"] == 1
    if mode == "multi_bounce":
        return meta["bounces"] >= 2
    if mode == "joint":
        return meta["joint_events"] >= 1
    raise ValueError(f"unknown generation mode: {mode}")


def _find_layout(
    instructions: list[tuple[int, int]],
    strip_width: int,
    mode: str,
) -> tuple[int, int, tuple[int, int], dict[str, int]] | None:
    total_length = sum(length for length, _color in instructions)
    min_height = max(total_length + 3, 15)
    if min_height > 30:
        return None

    candidates: list[tuple[int, int, tuple[int, int], dict[str, int]]] = []
    max_height = min(total_length + 7, 30)
    min_width = max(strip_width, 8)
    max_width = 24

    for height in range(min_height, max_height + 1):
        start_row_limit = min(height - 3, total_length + 3)
        for start_row in range(total_length, start_row_limit + 1):
            for width in range(min_width, max_width + 1):
                for start_col in range(1, width - 1):
                    try:
                        _trace, meta = _trace_path((height, width), (start_row, start_col), instructions)
                    except ValueError:
                        continue
                    if _mode_ok(meta, mode):
                        candidates.append((height, width, (start_row, start_col), meta))

    if not candidates:
        return None
    return random.choice(candidates)


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    sampled = _sample_glyph_sequence()
    if sampled is None:
        return None

    glyphs, instructions, strip_width = sampled
    mode = random.choice(MODE_CHOICES)
    layout = _find_layout(instructions, strip_width, mode)
    if layout is None:
        return None

    height, width, start, meta = layout
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[start[0], start[1]] = Color.GREEN

    strip_start = random.randint(0, width - strip_width)
    cursor = strip_start
    for glyph in glyphs:
        glyph_width = glyph.shape[1]
        patch = grid[-2:, cursor : cursor + glyph_width]
        mask = glyph != Color.BLACK
        patch[mask] = glyph[mask]
        cursor += glyph_width

    decoded = _decode_glyphs(grid)
    if decoded != instructions:
        return None

    expected = _render_output(grid, instructions, start)

    if meta["path_length"] != sum(length for length, _color in instructions):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_input()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input for the glyph-bounce puzzle")

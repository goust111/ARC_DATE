from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SOURCE = 5
PALETTE = (BLUE, RED, GREEN, YELLOW)
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 18
GRID_MAX = 24
INNER_MIN = 5
INNER_MAX = 8
GEN_RETRIES = 400

_GEN_CALL_COUNT = 0


BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _contains(outer: BBox, inner: BBox) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _border_cells(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r, c1) for r in range(r0 + 1, r1 + 1)]
    cells += [(r1, c) for c in range(c1 - 1, c0 - 1, -1)]
    cells += [(r, c0) for r in range(r1 - 1, r0, -1)]
    return cells


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _source_side(source: tuple[int, int], outer: BBox) -> str:
    r, c = source
    r0, c0, r1, c1 = outer
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("source must occupy one non-corner cell on the outer frame")


def _step_for_side(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    raise AssertionError(f"unknown side {side}")


def _ray_cells(shape: tuple[int, int], source: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = source
    dr, dc = step
    cells = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells
        cells.append((r, c))


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(np.bincount(g.ravel(), minlength=10).argmax())
    assert bg == BG, "background must be black"

    source_coords = np.argwhere(g == SOURCE)
    assert source_coords.shape[0] == 1, "expected exactly one grey source pixel"
    source = tuple(map(int, source_coords[0]))

    colors = [int(color) for color in np.unique(g) if color not in (BG, SOURCE)]
    assert 3 <= len(colors) <= 4, "expected 3 or 4 nested frame colors"

    layers = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "each frame color must appear"
        box = _bbox_coords(coords)
        border = _border_cells(box)
        border_set = set(border)
        coord_set = {tuple(map(int, rc)) for rc in coords}
        assert coord_set <= border_set, "frame pixels must lie on their bbox border"
        missing = [cell for cell in border if int(g[cell]) != color]
        layers.append({"color": color, "bbox": box, "missing": missing})

    layers.sort(key=lambda layer: _area(layer["bbox"]), reverse=True)
    assert len({_area(layer["bbox"]) for layer in layers}) == len(layers), "layer bboxes must be unique"

    for outer, inner in zip(layers, layers[1:]):
        assert _contains(outer["bbox"], inner["bbox"]), "layers must be strictly nested"

    outer = layers[0]
    assert outer["missing"] == [source], "outer frame must have a single source gap"
    for layer in layers[1:]:
        assert not layer["missing"], "only the outer frame may contain a gap"

    side = _source_side(source, outer["bbox"])
    step = _step_for_side(side)
    line = _ray_cells(g.shape, source, step)
    assert line, "ray must have non-empty runway"

    return {
        "grid": g,
        "source": source,
        "side": side,
        "step": step,
        "line": line,
        "layers": layers,
        "layer_colors": [int(layer["color"]) for layer in layers],
    }


def _trace_solution(grid: np.ndarray, info: dict) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = info["grid"]
    out = g.copy()
    layers = info["layers"]
    allowed = set(info["layer_colors"])
    current = int(layers[0]["color"])
    seen = {current}
    painted: list[tuple[int, int]] = []
    segments: list[tuple[int, int]] = []
    run = 0

    for r, c in info["line"]:
        value = int(g[r, c])
        if value == BG:
            out[r, c] = current
            painted.append((r, c))
            run += 1
            continue

        assert value in allowed, "ray may only meet nested frame pixels"
        if value not in seen:
            assert run > 0, "every color segment must occupy at least one background pixel"
            segments.append((current, run))
            seen.add(value)
            current = value
            run = 0

    assert run > 0, "the innermost color must continue to the grid boundary"
    segments.append((current, run))
    assert len(segments) == len(layers), "every layer color must govern one visible segment"
    assert seen == allowed, "ray must enter every nested layer"
    return out, painted, segments


def main(grid: np.ndarray) -> np.ndarray:
    """Emit a straight ray through nested rectangular frames, recoloring each deeper segment by the frame it enters.

    Input structure:
    - The background is black.
    - There are 3 or 4 strictly nested one-pixel rectangular frames, all different colors.
    - The outermost frame has exactly one non-corner gap occupied by a grey source pixel.

    Rule:
    - Fire one cardinal ray inward from the grey source, perpendicular to the gapped side.
    - The source itself stays grey, frame pixels stay unchanged, and only background cells on the ray may be painted.
    - After the ray enters through the outer gap it uses the outer frame color, then each time it reaches a deeper frame border, the following background segment switches to that frame's color; once the innermost color is reached it continues unchanged to the grid edge.

    Key insight:
    - The visible line is segmented by the nested borders, and each segment takes the color of the deepest frame the ray has entered so far rather than a fixed depth-to-color code.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    out, _painted, _segments = _trace_solution(grid, info)
    return out


def _choose_layer_count(branch: int) -> int:
    return 3 if branch % 2 == 0 else 4


def _choose_side(branch: int) -> str:
    return SIDES[(branch // 2) % len(SIDES)]


def _gap_patterns(layer_count: int) -> list[tuple[int, ...]]:
    if layer_count == 3:
        return [(2, 2), (2, 3), (3, 2), (3, 3)]
    return [(2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)]


def _build_boxes(
    rng: np.random.Generator, shape: tuple[int, int], layer_count: int
) -> list[BBox] | None:
    h, w = shape
    patterns = _gap_patterns(layer_count)
    gap_pattern = patterns[int(rng.integers(0, len(patterns)))]
    total_inset = sum(gap_pattern)

    inner_h_max = min(INNER_MAX, h - 2 * total_inset - 2)
    inner_w_max = min(INNER_MAX, w - 2 * total_inset - 2)
    if inner_h_max < INNER_MIN or inner_w_max < INNER_MIN:
        return None

    inner_h = int(rng.integers(INNER_MIN, inner_h_max + 1))
    inner_w = int(rng.integers(INNER_MIN, inner_w_max + 1))
    outer_h = inner_h + 2 * total_inset
    outer_w = inner_w + 2 * total_inset

    max_r0 = h - outer_h - 1
    max_c0 = w - outer_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None

    r0 = int(rng.integers(1, max_r0 + 1))
    c0 = int(rng.integers(1, max_c0 + 1))
    boxes = [(r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)]
    for gap in gap_pattern:
        pr0, pc0, pr1, pc1 = boxes[-1]
        boxes.append((pr0 + gap, pc0 + gap, pr1 - gap, pc1 - gap))
    return boxes


def _source_from_side(
    rng: np.random.Generator, side: str, outer: BBox, inner: BBox
) -> tuple[int, int]:
    r0, c0, r1, c1 = outer
    ir0, ic0, ir1, ic1 = inner
    if side in ("top", "bottom"):
        candidates = list(range(ic0 + 1, ic1))
        assert candidates
        c = int(candidates[int(rng.integers(0, len(candidates)))])
        return (r0, c) if side == "top" else (r1, c)
    candidates = list(range(ir0 + 1, ir1))
    assert candidates
    r = int(candidates[int(rng.integers(0, len(candidates)))])
    return (r, c0) if side == "left" else (r, c1)


def _build_input(
    rng: np.random.Generator, branch: int
) -> np.ndarray | None:
    layer_count = _choose_layer_count(branch)
    side = _choose_side(branch)
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    boxes = _build_boxes(rng, (h, w), layer_count)
    if boxes is None:
        return None

    colors = list(map(int, rng.permutation(PALETTE)[:layer_count]))
    grid = np.full((h, w), BG, dtype=int)
    for color, box in zip(colors, boxes):
        _draw_frame(grid, box, color)

    source = _source_from_side(rng, side, boxes[0], boxes[-1])
    grid[source] = SOURCE
    return grid


def _validate_example(inp: np.ndarray) -> bool:
    try:
        info = _analyze_grid(inp)
        out, painted, segments = _trace_solution(inp, info)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, out):
        return False

    diff = actual != inp
    if not np.any(diff):
        return False
    if np.any(inp[diff] != BG):
        return False

    line = set(info["line"])
    if any(cell not in line for cell in painted):
        return False

    segment_colors = [color for color, _length in segments]
    layer_colors = [int(layer["color"]) for layer in info["layers"]]
    if segment_colors != layer_colors:
        return False
    if any(length <= 0 for _color, length in segments):
        return False

    crossings = {color: 0 for color in layer_colors}
    for r, c in info["line"]:
        value = int(inp[r, c])
        if value in crossings:
            crossings[value] += 1
    if crossings[layer_colors[0]] < 1:
        return False
    for color in layer_colors[1:]:
        if crossings[color] < 2:
            return False

    # The source side must matter: moving the source to another side would change the ray line.
    outer = info["layers"][0]["bbox"]
    source = info["source"]
    if _source_side(source, outer) != info["side"]:
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    branch = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    rng = np.random.default_rng()

    for _ in range(GEN_RETRIES):
        grid = _build_input(rng, branch)
        if grid is not None and _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid nested-ray recolor input")

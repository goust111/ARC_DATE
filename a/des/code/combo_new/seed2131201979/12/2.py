import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREY = 5
PALETTE_POOL = [1, 3, 4, 6, 7, 8, 9]

GRID_MIN = 10
GRID_MAX = 24
LEGEND_WIDTH = 3
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_legend(grid: np.ndarray) -> tuple[np.ndarray, list[int], tuple[int, int, int, int]]:
    candidates = []
    for comp in _components(grid == GREY):
        box = _bbox(comp)
        border = _border_cells(box)
        if set(comp) != border:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        inner: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(grid[r, c])
                if color not in (BG, RED, GREY):
                    inner.append((r, c, color))
        if not inner:
            continue
        corner_dist = min(r0 + c0, r0 + (grid.shape[1] - 1 - c1), (grid.shape[0] - 1 - r1) + c0, (grid.shape[0] - 1 - r1) + (grid.shape[1] - 1 - c1))
        candidates.append((len(inner), -corner_dist, box, inner))

    assert candidates, "expected a grey framed legend"
    _, _, box, inner = max(candidates)
    r0, c0, r1, c1 = box
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    order: list[int] = []
    seen: set[int] = set()
    for _r, _c, color in sorted(inner):
        if color not in seen:
            seen.add(color)
            order.append(color)
    assert order, "legend must contain at least one ordered color"
    return legend_mask, order, box


def _line_cells(axis: str, index: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if axis == "row":
        return [(index, c) for c in range(w)]
    return [(r, index) for r in range(h)]


def _eligible_cells(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    axis: str,
    index: int,
) -> list[tuple[int, int]]:
    cells = []
    for r, c in _line_cells(axis, index, grid.shape):
        if legend_mask[r, c]:
            continue
        if int(grid[r, c]) in (RED, GREY):
            continue
        cells.append((r, c))
    return cells


def _target_cells(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    axis: str,
    index: int,
    direction: str,
    count: int,
) -> list[tuple[int, int]]:
    cells = _eligible_cells(grid, legend_mask, axis, index)
    if direction in ("right", "down"):
        cells = list(reversed(cells))
    return cells[:count]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact each arrow-marked row or column toward the arrow direction, using the corner legend strip to decide color order.

    Input structure:
    - The background is black.
    - Single red pixels sit on the outer border and act as arrows: left-edge arrows push right, right-edge arrows push left, top-edge arrows push down, and bottom-edge arrows push up.
    - The row or column selected by an arrow contains scattered non-black colored pixels, separated by black gaps.
    - A grey rectangular frame in a corner encloses a vertical color strip whose top-to-bottom order lists all movable non-black colors.

    Rule:
    - Locate the grey framed legend and read the inner strip from top to bottom to obtain the color ordering.
    - For each red border arrow, collect all non-black, non-red, non-grey pixels on its row or column, excluding the legend area.
    - Sort those collected colors by the legend-strip order, then place the sorted pixels tightly from the target end indicated by the arrow direction.
    - Clear the old positions of the moved pixels and erase the red arrow itself to black.
    - Keep the legend region unchanged.

    Key insight:
    - The red arrow tells you which side the line compacts toward, but the final arrangement within that compacted block comes from the separate corner legend rather than from the colors themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    legend_mask, order, _ = _find_legend(g)
    rank = {color: idx for idx, color in enumerate(order)}

    arrows: list[tuple[str, int, str, tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != RED or legend_mask[r, c]:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if r in (0, h - 1) and c in (0, w - 1):
                continue
            if c == 0:
                arrows.append(("row", r, "right", (r, c)))
            elif c == w - 1:
                arrows.append(("row", r, "left", (r, c)))
            elif r == 0:
                arrows.append(("col", c, "down", (r, c)))
            else:
                arrows.append(("col", c, "up", (r, c)))

    out = g.copy()
    cells_to_clear: set[tuple[int, int]] = {arrow_pos for *_meta, arrow_pos in arrows}
    placements: list[tuple[int, int, int]] = []

    for axis, index, direction, _arrow_pos in arrows:
        colors: list[int] = []
        for r, c in _line_cells(axis, index, g.shape):
            if legend_mask[r, c]:
                continue
            color = int(g[r, c])
            if color in (BG, RED, GREY):
                continue
            cells_to_clear.add((r, c))
            assert color in rank, "all movable colors must appear in the legend strip"
            colors.append(color)

        if not colors:
            continue

        sorted_colors = sorted(colors, key=lambda color: rank[color])
        targets = _target_cells(g, legend_mask, axis, index, direction, len(sorted_colors))
        assert len(targets) == len(sorted_colors), "line must have enough free cells to compact into"
        for (r, c), color in zip(targets, sorted_colors):
            placements.append((r, c, color))

    for r, c in cells_to_clear:
        if not legend_mask[r, c]:
            out[r, c] = BG
    for r, c, color in placements:
        out[r, c] = color
    return out


def _legend_box(h: int, w: int, order_len: int, corner: str) -> tuple[int, int, int, int]:
    box_h = order_len + 2
    box_w = LEGEND_WIDTH
    if corner == "tl":
        r0, c0 = 1, 1
    elif corner == "tr":
        r0, c0 = 1, w - box_w - 1
    elif corner == "bl":
        r0, c0 = h - box_h - 1, 1
    else:
        r0, c0 = h - box_h - 1, w - box_w - 1
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _paint_legend(grid: np.ndarray, box: tuple[int, int, int, int], order: list[int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    strip_col = c0 + 1
    for idx, color in enumerate(order):
        grid[r0 + 1 + idx, strip_col] = int(color)


def _choose_dimensions(
    orientation: str,
    total_lines: int,
    order_len: int,
    max_pixels: int,
) -> tuple[int, int] | None:
    if orientation == "horizontal":
        min_h = max(GRID_MIN, order_len + total_lines + 4, max_pixels + 2)
        min_w = max(GRID_MIN, max_pixels + 3)
    else:
        min_h = max(GRID_MIN, order_len + 4, max_pixels + 3)
        min_w = max(GRID_MIN, total_lines + LEGEND_WIDTH + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    return random.randint(min_h, GRID_MAX), random.randint(min_w, GRID_MAX)


def _make_line_specs(
    orientation: str,
    shape: tuple[int, int],
    legend_box: tuple[int, int, int, int],
    side_counts: tuple[int, int],
) -> list[dict] | None:
    h, w = shape
    r0, c0, r1, c1 = legend_box
    specs: list[dict] = []
    if orientation == "horizontal":
        available = [r for r in range(1, h - 1) if not (r0 <= r <= r1)]
        if len(available) < sum(side_counts):
            return None
        chosen = random.sample(available, sum(side_counts))
        random.shuffle(chosen)
        for r in chosen[: side_counts[0]]:
            specs.append({"axis": "row", "index": r, "direction": "right", "arrow": (r, 0)})
        for r in chosen[side_counts[0] :]:
            specs.append({"axis": "row", "index": r, "direction": "left", "arrow": (r, w - 1)})
    else:
        available = [c for c in range(1, w - 1) if not (c0 <= c <= c1)]
        if len(available) < sum(side_counts):
            return None
        chosen = random.sample(available, sum(side_counts))
        random.shuffle(chosen)
        for c in chosen[: side_counts[0]]:
            specs.append({"axis": "col", "index": c, "direction": "down", "arrow": (0, c)})
        for c in chosen[side_counts[0] :]:
            specs.append({"axis": "col", "index": c, "direction": "up", "arrow": (h - 1, c)})
    random.shuffle(specs)
    return specs


def _build_line_color_lists(order: list[int], counts: list[int]) -> list[list[int]] | None:
    if len(order) < 2:
        return None
    line_colors: list[list[int]] = []
    for count in counts:
        colors = random.sample(order, 2)
        while len(colors) < count:
            colors.append(random.choice(order))
        random.shuffle(colors)
        line_colors.append(colors)

    used = {color for colors in line_colors for color in colors}
    missing = [color for color in order if color not in used]
    for color in missing:
        placed = False
        for _ in range(100):
            line_idx = random.randrange(len(line_colors))
            slot_idx = random.randrange(len(line_colors[line_idx]))
            trial = line_colors[line_idx].copy()
            trial[slot_idx] = color
            if len(set(trial)) >= 2:
                line_colors[line_idx] = trial
                placed = True
                break
        if not placed:
            return None
    return line_colors


def _choose_input_positions(
    eligible: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    target_set = set(targets)
    outside = [pos for pos in eligible if pos not in target_set]
    assert outside, "need at least one cell outside the solved block"
    for _ in range(80):
        picked = random.sample(eligible, count)
        if any(pos not in target_set for pos in picked):
            return picked
    fixed = [random.choice(outside)]
    remaining = [pos for pos in eligible if pos != fixed[0]]
    fixed.extend(random.sample(remaining, count - 1))
    return fixed


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["horizontal", "vertical"])
    side_counts = (random.randint(1, 2), random.randint(1, 2))
    total_lines = sum(side_counts)
    line_pixel_counts = [random.randint(2, 5) for _ in range(total_lines)]
    max_pixels = max(line_pixel_counts)
    max_order = min(len(PALETTE_POOL), sum(line_pixel_counts))
    if max_order < 3:
        return None
    order_len = random.randint(3, min(6, max_order))

    dims = _choose_dimensions(orientation, total_lines, order_len, max_pixels)
    if dims is None:
        return None
    h, w = dims

    order = random.sample(PALETTE_POOL, order_len)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_box = _legend_box(h, w, order_len, corner)
    specs = _make_line_specs(orientation, (h, w), legend_box, side_counts)
    if specs is None:
        return None

    line_colors = _build_line_color_lists(order, line_pixel_counts)
    if line_colors is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _paint_legend(grid, legend_box, order)
    for spec in specs:
        ar, ac = spec["arrow"]
        grid[ar, ac] = RED

    legend_mask, detected_order, _ = _find_legend(grid)
    if detected_order != order:
        return None

    for spec, colors in zip(specs, line_colors):
        eligible = _eligible_cells(grid, legend_mask, spec["axis"], spec["index"])
        if len(eligible) <= len(colors):
            return None
        targets = _target_cells(grid, legend_mask, spec["axis"], spec["index"], spec["direction"], len(colors))
        picked = _choose_input_positions(eligible, targets, len(colors))
        shuffled = colors.copy()
        random.shuffle(shuffled)
        if set(picked) == set(targets):
            return None
        for (r, c), color in zip(picked, shuffled):
            grid[r, c] = int(color)

    if set(order) != {int(v) for v in np.unique(grid) if int(v) not in (BG, RED, GREY)}:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.any(out == RED):
        return None
    if not np.array_equal(out[legend_mask], grid[legend_mask]):
        return None

    post_legend_mask, post_order, _ = _find_legend(out)
    if not np.array_equal(post_legend_mask, legend_mask) or post_order != order:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")

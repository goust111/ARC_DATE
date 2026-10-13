from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _edge_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _infer_period(colors: list[int]) -> int:
    n = len(colors)
    for p in range(1, n):
        if all(colors[i] == colors[i + p] for i in range(n - p)):
            return p
    return n


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _components_mask(g != bg)
    assert len(components) == 2, "expected one rectangle and one marker"
    singletons = [comp for comp in components if len(comp) == 1]
    solids = [comp for comp in components if len(comp) > 1]
    assert len(singletons) == 1 and len(solids) == 1

    marker_r, marker_c = map(int, singletons[0][0])
    rect_coords = solids[0]
    r0, c0, r1, c1 = _bbox_coords(rect_coords)
    rect_h, rect_w = r1 - r0 + 1, c1 - c0 + 1
    assert rect_h >= 3 and rect_w >= 3

    rect_mask = np.zeros_like(g, dtype=bool)
    rect_mask[rect_coords[:, 0], rect_coords[:, 1]] = True
    assert rect_mask[r0 : r1 + 1, c0 : c1 + 1].all(), "rectangle must be solid"
    assert int(rect_mask.sum()) == rect_h * rect_w

    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    edge = _edge_mask(rect.shape)
    border_values = np.unique(rect[edge])
    assert border_values.size == 1, "expected a single border color"
    border_color = int(border_values[0])

    inner = rect[1:-1, 1:-1]
    assert inner.size > 0
    assert np.all(inner != bg), "interior must be fully painted"

    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    min_sum = inner_r0 + inner_c0
    max_sum = inner_r1 + inner_c1

    visible_colors = []
    for diag_sum in range(min_sum, max_sum + 1):
        values = {
            int(g[rr, cc])
            for rr in range(inner_r0, inner_r1 + 1)
            for cc in (diag_sum - rr,)
            if inner_c0 <= cc <= inner_c1
        }
        assert len(values) == 1, "interior must be constant on anti-diagonals"
        visible_colors.append(values.pop())

    period = _infer_period(visible_colors)
    assert period < len(visible_colors), "visible diagonals must reveal a repeating period"
    period_colors = visible_colors[:period]

    options = []
    if marker_r < r0 and c0 <= marker_c <= c1:
        options.append(("up", (marker_r, c0, r1, c1)))
    if marker_r > r1 and c0 <= marker_c <= c1:
        options.append(("down", (r0, c0, marker_r, c1)))
    if marker_c < c0 and r0 <= marker_r <= r1:
        options.append(("left", (r0, marker_c, r1, c1)))
    if marker_c > c1 and r0 <= marker_r <= r1:
        options.append(("right", (r0, c0, r1, marker_c)))
    assert len(options) == 1, "marker must uniquely select one stretch direction"
    direction, expanded_box = options[0]

    return {
        "grid": g,
        "bg": bg,
        "marker": (marker_r, marker_c),
        "orig_box": (r0, c0, r1, c1),
        "expanded_box": expanded_box,
        "direction": direction,
        "border_color": border_color,
        "period_colors": period_colors,
        "anchor_sum": min_sum,
    }


def _field_color(r: int, c: int, period_colors: list[int], anchor_sum: int) -> int:
    period = len(period_colors)
    return int(period_colors[(r + c - anchor_sum) % period])


def _render_output_from_analysis(info: dict) -> np.ndarray:
    g = info["grid"]
    bg = int(info["bg"])
    border_color = int(info["border_color"])
    period_colors = list(info["period_colors"])
    anchor_sum = int(info["anchor_sum"])
    r0, c0, r1, c1 = info["expanded_box"]

    out = np.full_like(g, bg)
    out[r0 : r1 + 1, c0 : c1 + 1] = border_color
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[r, c] = _field_color(r, c, period_colors, anchor_sum)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the striped rectangle toward the aligned marker and continue its anti-diagonal color field.

    Input structure:
    - The background is the dominant color.
    - One solid rectangular object has a single border color and a fully painted interior.
    - Inside the rectangle, colors are constant on anti-diagonals and repeat with a fixed period.
    - A lone marker pixel sits directly above, below, left, or right of the rectangle, aligned to its span.

    Rule:
    - Treat the marker as the new outer row or column in its aligned direction and expand the rectangle until that side reaches the marker.
    - Redraw only the new outer boundary with the original border color; the old moved side becomes interior.
    - Fill every interior cell of the expanded rectangle by extending the same anti-diagonal periodic field inferred from the original interior.
    - Leave every cell outside the expanded rectangle as background.

    Key insight:
    - The marker controls geometry only; the added area is not copied from a neighboring row or column, but inferred from the rectangle's global anti-diagonal periodic pattern.
    """
    info = _parse_instance(np.asarray(grid, dtype=int))
    return _render_output_from_analysis(info)


PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
PERIOD_CHOICES = (2, 3, 4)
ORIG_SIDE_MIN = 6
ORIG_SIDE_MAX = 10
EXPANSION_MIN = 2
EXPANSION_MAX = 6
GRID_MARGIN_MIN = 1
GRID_PADDING_MAX = 10
GENERATION_RETRIES = 400


def _render_rectangle(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    border_color: int,
    stripe_colors: list[int],
    phase: int,
) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(border_color)
    period = len(stripe_colors)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            grid[r, c] = int(stripe_colors[(r + c + phase) % period])


def _build_case(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    h = int(spec["grid_h"])
    w = int(spec["grid_w"])
    bg = int(spec["bg"])
    border_color = int(spec["border_color"])
    marker_color = int(spec["marker_color"])
    stripe_colors = list(spec["stripe_colors"])
    phase = int(spec["phase"])
    orig_box = spec["orig_box"]
    expanded_box = spec["expanded_box"]
    marker = spec["marker"]

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    _render_rectangle(inp, orig_box, border_color, stripe_colors, phase)
    inp[marker] = marker_color
    _render_rectangle(out, expanded_box, border_color, stripe_colors, phase)
    return inp, out


def _new_interior_cells(
    orig_box: tuple[int, int, int, int],
    expanded_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    or0, oc0, or1, oc1 = orig_box
    nr0, nc0, nr1, nc1 = expanded_box
    cells = []
    for r in range(nr0 + 1, nr1):
        for c in range(nc0 + 1, nc1):
            if not (or0 + 1 <= r <= or1 - 1 and oc0 + 1 <= c <= oc1 - 1):
                cells.append((r, c))
    return cells


def _simple_copy_matches(
    out: np.ndarray,
    direction: str,
    new_cells: list[tuple[int, int]],
) -> bool:
    if direction == "up":
        return all(int(out[r, c]) == int(out[r + 1, c]) for r, c in new_cells)
    if direction == "down":
        return all(int(out[r, c]) == int(out[r - 1, c]) for r, c in new_cells)
    if direction == "left":
        return all(int(out[r, c]) == int(out[r, c + 1]) for r, c in new_cells)
    return all(int(out[r, c]) == int(out[r, c - 1]) for r, c in new_cells)


def _moved_side_cells(orig_box: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = orig_box
    if direction == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if direction == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if direction == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _counts_unique_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts[int(bg)]
    return all(bg_count > count for color, count in counts.items() if int(color) != int(bg))


def _case_is_semantic(spec: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    bg = int(spec["bg"])
    border_color = int(spec["border_color"])
    orig_box = spec["orig_box"]
    expanded_box = spec["expanded_box"]
    direction = spec["direction"]

    if not _counts_unique_bg(inp, bg):
        return False

    new_cells = _new_interior_cells(orig_box, expanded_box)
    if len(new_cells) < 4:
        return False
    if len({int(out[r, c]) for r, c in new_cells}) < 2:
        return False
    if _simple_copy_matches(out, direction, new_cells):
        return False

    moved_side = _moved_side_cells(orig_box, direction)
    if not moved_side:
        return False
    if not any(int(out[r, c]) != border_color for r, c in moved_side):
        return False

    parsed = _parse_instance(inp)
    if parsed["direction"] != direction:
        return False
    if tuple(parsed["expanded_box"]) != tuple(expanded_box):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _random_case_spec() -> dict | None:
    direction = random.choice(DIRECTIONS)
    orig_h = random.randint(ORIG_SIDE_MIN, ORIG_SIDE_MAX)
    orig_w = random.randint(ORIG_SIDE_MIN, ORIG_SIDE_MAX)
    visible_diagonals = orig_h + orig_w - 5
    valid_periods = [p for p in PERIOD_CHOICES if p < visible_diagonals]
    if not valid_periods:
        return None
    period = random.choice(valid_periods)
    expansion = random.randint(EXPANSION_MIN, EXPANSION_MAX)

    expanded_h = orig_h + expansion if direction in ("up", "down") else orig_h
    expanded_w = orig_w + expansion if direction in ("left", "right") else orig_w

    grid_h_max = min(30, expanded_h + GRID_PADDING_MAX)
    grid_w_max = min(30, expanded_w + GRID_PADDING_MAX)
    grid_h_min = expanded_h + 2 * GRID_MARGIN_MIN
    grid_w_min = expanded_w + 2 * GRID_MARGIN_MIN
    if grid_h_min > grid_h_max or grid_w_min > grid_w_max:
        return None

    grid_h = random.randint(grid_h_min, grid_h_max)
    grid_w = random.randint(grid_w_min, grid_w_max)
    top = random.randint(GRID_MARGIN_MIN, grid_h - expanded_h - GRID_MARGIN_MIN)
    left = random.randint(GRID_MARGIN_MIN, grid_w - expanded_w - GRID_MARGIN_MIN)
    expanded_box = (top, left, top + expanded_h - 1, left + expanded_w - 1)

    nr0, nc0, nr1, nc1 = expanded_box
    if direction == "up":
        orig_box = (nr0 + expansion, nc0, nr1, nc1)
        marker = (nr0, random.randint(nc0, nc1))
    elif direction == "down":
        orig_box = (nr0, nc0, nr1 - expansion, nc1)
        marker = (nr1, random.randint(nc0, nc1))
    elif direction == "left":
        orig_box = (nr0, nc0 + expansion, nr1, nc1)
        marker = (random.randint(nr0, nr1), nc0)
    else:
        orig_box = (nr0, nc0, nr1, nc1 - expansion)
        marker = (random.randint(nr0, nr1), nc1)

    colors = random.sample(PALETTE, period + 2)
    border_color = int(colors[0])
    marker_color = int(colors[1])
    stripe_colors = list(map(int, colors[2:]))
    phase = random.randrange(period)

    return {
        "grid_h": grid_h,
        "grid_w": grid_w,
        "bg": 0,
        "border_color": border_color,
        "marker_color": marker_color,
        "stripe_colors": stripe_colors,
        "phase": phase,
        "direction": direction,
        "orig_box": orig_box,
        "expanded_box": expanded_box,
        "marker": marker,
    }


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        spec = _random_case_spec()
        if spec is None:
            continue
        inp, out = _build_case(spec)
        if _case_is_semantic(spec, inp, out):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 8")

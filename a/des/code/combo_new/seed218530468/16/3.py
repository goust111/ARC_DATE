from __future__ import annotations

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


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
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _frame_color(grid: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    colors = [(int(v), int(c)) for v, c in zip(vals, cnts, strict=False) if int(v) != bg]
    assert colors, "expected at least one non-background color"
    return max(colors, key=lambda item: item[1])[0]


def _rectangular_frames(grid: np.ndarray, frame_color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == frame_color):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0 = min(rows)
        r1 = max(rows)
        c0 = min(cols)
        c1 = max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        ring = set()
        for rr in range(r0, r1 + 1):
            ring.add((int(rr), int(c0)))
            ring.add((int(rr), int(c1)))
        for cc in range(c0, c1 + 1):
            ring.add((int(r0), int(cc)))
            ring.add((int(r1), int(cc)))

        if set(comp) != ring:
            continue
        frames.append((int(r0), int(c0), int(r1), int(c1)))

    frames.sort()
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill one half of each framed rectangle using the unique marker color inside it.

    Input structure:
    - A black background surrounds several disjoint 1-pixel-thick rectangular frames of one shared color.
    - Each frame encloses exactly one single-pixel marker whose color is neither the background nor the frame color.
    - Every rectangle is strictly horizontal or strictly vertical, never square.

    Rule:
    - Find each rectangular frame and its unique interior marker pixel.
    - If the rectangle is horizontal, use the marker's column as the split line and fill the interior half on the marker's side (left or right) with the marker color.
    - If the rectangle is vertical, use the marker's row as the split line and fill the interior half on the marker's side (top or bottom) with the marker color.
    - Keep the frame unchanged and include the marker pixel inside the filled half.

    Key insight:
    - The rectangle's long axis decides whether the split is left/right or top/bottom, while the marker's off-center position chooses which half gets colored.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frame = _frame_color(g, bg)
    out = g.copy()

    frames = _rectangular_frames(g, frame)
    assert frames, "expected at least one rectangular frame"

    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere((interior != bg) & (interior != frame))
        assert marker_locs.shape == (1, 2), "each frame must contain exactly one marker"

        mr_local, mc_local = map(int, marker_locs[0])
        marker_color = int(interior[mr_local, mc_local])
        inner_h, inner_w = map(int, interior.shape)
        abs_r = r0 + 1 + mr_local
        abs_c = c0 + 1 + mc_local

        if inner_w > inner_h:
            if mc_local < inner_w / 2.0:
                out[r0 + 1 : r1, c0 + 1 : abs_c + 1] = marker_color
            else:
                out[r0 + 1 : r1, abs_c:c1] = marker_color
        else:
            if mr_local < inner_h / 2.0:
                out[r0 + 1 : abs_r + 1, c0 + 1 : c1] = marker_color
            else:
                out[abs_r:r1, c0 + 1 : c1] = marker_color

    return out


GEN_ATTEMPTS = 500
PLACE_TRIES = 300
GRID_MIN = 12
GRID_MAX = 30
MIN_COUNT = 2
MAX_COUNT = 5
PALETTE = tuple(range(1, 10))


def _balanced_labels(n: int, first: str, second: str, rng: np.random.Generator) -> list[str]:
    left = n // 2
    right = n - left
    if n % 2 == 1 and rng.integers(0, 2) == 0:
        left, right = right, left
    labels = [first] * left + [second] * right
    rng.shuffle(labels)
    return labels


def _sample_inner_dims(orientation: str, rng: np.random.Generator) -> tuple[int, int]:
    short_axis = int(rng.integers(2, 6))
    long_min = max(3, short_axis + 1)
    long_max = short_axis + 6
    long_axis = int(rng.integers(long_min, long_max + 1))
    if orientation == "H":
        return short_axis, long_axis
    return long_axis, short_axis


def _side_index(length: int, side: str, rng: np.random.Generator) -> int:
    if side in ("left", "top"):
        hi = max(1, length // 2)
        return int(rng.integers(0, hi))
    lo = length // 2 if length % 2 == 0 else length // 2 + 1
    return int(rng.integers(lo, length))


def _boxes_too_close(
    box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return True
    return False


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, frame_color: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _expected_from_specs(
    grid: np.ndarray, specs: list[dict[str, int | str]]
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        marker_r = int(spec["marker_r"])
        marker_c = int(spec["marker_c"])
        color = int(spec["marker_color"])
        orientation = str(spec["orientation"])
        side = str(spec["side"])

        if orientation == "H":
            if side == "left":
                out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : marker_c + 1] = color
            else:
                out[r0 + 1 : r0 + 1 + inner_h, marker_c : c0 + 1 + inner_w] = color
        else:
            if side == "top":
                out[r0 + 1 : marker_r + 1, c0 + 1 : c0 + 1 + inner_w] = color
            else:
                out[marker_r : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = color
    return out


def _build_requests(count: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    n_h = count // 2
    n_v = count // 2
    if count % 2 == 1:
        if rng.integers(0, 2) == 0:
            n_h += 1
        else:
            n_v += 1

    requests = [("H", side) for side in _balanced_labels(n_h, "left", "right", rng)]
    requests.extend(("V", side) for side in _balanced_labels(n_v, "top", "bottom", rng))
    rng.shuffle(requests)
    return requests


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    area = h * w
    count_cap = min(MAX_COUNT, max(MIN_COUNT, 2 + area // 220))
    count = int(rng.integers(MIN_COUNT, count_cap + 1))

    colors = list(PALETTE)
    rng.shuffle(colors)
    frame_color = int(colors[0])
    marker_colors = [int(c) for c in colors[1 : 1 + count]]
    if len(marker_colors) != count:
        return None

    grid = np.zeros((h, w), dtype=int)
    requests = _build_requests(count, rng)
    placed_boxes: list[tuple[int, int, int, int]] = []
    specs: list[dict[str, int | str]] = []

    for (orientation, side), marker_color in zip(requests, marker_colors, strict=False):
        placed = False
        for _ in range(PLACE_TRIES):
            inner_h, inner_w = _sample_inner_dims(orientation, rng)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h > h - 2 or outer_w > w - 2:
                continue

            r0 = int(rng.integers(1, h - outer_h))
            c0 = int(rng.integers(1, w - outer_w))
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if _boxes_too_close(box, placed_boxes, gap=1):
                continue

            if orientation == "H":
                marker_r = r0 + 1 + int(rng.integers(0, inner_h))
                marker_c = c0 + 1 + _side_index(inner_w, side, rng)
            else:
                marker_r = r0 + 1 + _side_index(inner_h, side, rng)
                marker_c = c0 + 1 + int(rng.integers(0, inner_w))

            _draw_frame(grid, r0, c0, inner_h, inner_w, frame_color)
            grid[marker_r, marker_c] = marker_color
            placed_boxes.append(box)
            specs.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "marker_r": marker_r,
                    "marker_c": marker_c,
                    "marker_color": marker_color,
                    "orientation": orientation,
                    "side": side,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

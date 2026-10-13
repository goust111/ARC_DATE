import random

import numpy as np


BG_COLOR = 0
PIPE_COLOR = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_GRID_SIZE = 18
MAX_GRID_SIZE = 30
MIN_PIPE_LENGTH = 5
CENTER_GAP = 4
CROSS_MARGIN = 7
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _most_common_non_bg_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(values) > 0, "expected at least one non-background color"
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_markers(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int, int, tuple[int, int], tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    pipe = _most_common_non_bg_color(g, bg)

    marker_mask = (g != bg) & (g != pipe)
    markers: list[tuple[int, int, int, tuple[int, int], tuple[int, int]]] = []
    for comp in _components(marker_mask):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "marker component must be monochrome"
        color = int(next(iter(colors)))

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert len(comp) == 3, "marker strip must have exactly three cells"

        found: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if r1 - r0 == 2 and c0 == c1:
            center = (r0 + 1, c0)
            if c1 + 1 < w and all(int(g[r, c1 + 1]) == pipe for r in range(r0, r1 + 1)):
                found.append(((0, 1), (center[0], c1 + 1)))
            if c0 - 1 >= 0 and all(int(g[r, c0 - 1]) == pipe for r in range(r0, r1 + 1)):
                found.append(((0, -1), (center[0], c0 - 1)))
        elif c1 - c0 == 2 and r0 == r1:
            center = (r0, c0 + 1)
            if r1 + 1 < h and all(int(g[r1 + 1, c]) == pipe for c in range(c0, c1 + 1)):
                found.append(((1, 0), (r1 + 1, center[1])))
            if r0 - 1 >= 0 and all(int(g[r0 - 1, c]) == pipe for c in range(c0, c1 + 1)):
                found.append(((-1, 0), (r0 - 1, center[1])))
        else:
            raise AssertionError("marker strip must be 1x3 or 3x1")

        assert len(found) == 1, "marker strip must touch exactly one pipe entrance"
        markers.append((r0, c0, color, found[0][0], found[0][1]))

    markers.sort(key=lambda item: (item[0], item[1]))
    return bg, pipe, markers


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    _bg, pipe, markers = _extract_markers(g)
    for _r0, _c0, color, (dr, dc), (r, c) in markers:
        step = 1
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == pipe:
            out[rr, cc] = int(color) if step % 2 == 1 else int(pipe)
            rr += dr
            cc += dc
            step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each marked straight pipe through crossings and stripe its center channel.

    Input structure:
    - Background is black.
    - Gray pipes are 3 cells wide and run horizontally or vertically.
    - Each pipe has exactly one colored 3-cell marker strip attached to one endpoint, while the other endpoint stays gray.
    - Horizontal and vertical pipes may overlap in 3x3 cross intersections.

    Rule:
    - Detect every colored endpoint marker and infer the pipe's travel direction from the side where the gray pipe begins.
    - Starting from the marker-side entrance, follow the pipe straight ahead through any crossings without turning.
    - Repaint only the central row or central column of that pipe as alternating marker-color / gray stripes, with the first pipe cell after the marker colored.
    - Process pipes in the order of the marker strip's first appearance in the grid (topmost row, then leftmost column); later pipes overwrite earlier stripes at shared center pixels.

    Key insight:
    - The colored strips define both which gray bar is a pipe and which way flow enters it, while intersections are resolved by continuing straight and repainting the shared center cell in marker order.
    """
    return _transform(grid)


def _step_from_side(pos: int, size: int, side: str) -> int:
    if side in ("left", "top"):
        return pos
    if side in ("right", "bottom"):
        return size - 1 - pos
    raise ValueError(f"unknown side: {side}")


def _max_spaced_count(values: list[int], gap: int) -> int:
    count = 0
    last = -10**9
    for value in sorted(values):
        if value - last >= gap:
            count += 1
            last = value
    return count


def _sample_spaced_values(
    rng: random.Random,
    values: list[int],
    count: int,
    gap: int,
) -> list[int] | None:
    values = sorted(values)
    combos: list[tuple[int, ...]] = []

    def dfs(start: int, chosen: list[int]) -> None:
        if len(chosen) == count:
            combos.append(tuple(chosen))
            return
        for idx in range(start, len(values)):
            value = values[idx]
            if chosen and value - chosen[-1] < gap:
                continue
            dfs(idx + 1, chosen + [value])

    dfs(0, [])
    if not combos:
        return None
    return list(rng.choice(combos))


def _draw_pipe(grid: np.ndarray, spec: dict[str, int | str]) -> None:
    color = int(spec["color"])
    if spec["axis"] == "h":
        row = int(spec["center"])
        c0 = int(spec["c0"])
        c1 = int(spec["c1"])
        grid[row - 1 : row + 2, c0 : c1 + 1] = PIPE_COLOR
        if spec["marker_side"] == "left":
            grid[row - 1 : row + 2, c0 - 1] = color
        else:
            grid[row - 1 : row + 2, c1 + 1] = color
    else:
        col = int(spec["center"])
        r0 = int(spec["r0"])
        r1 = int(spec["r1"])
        grid[r0 : r1 + 1, col - 1 : col + 2] = PIPE_COLOR
        if spec["marker_side"] == "top":
            grid[r0 - 1, col - 1 : col + 2] = color
        else:
            grid[r1 + 1, col - 1 : col + 2] = color


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)

    h_side = rng.choice(("left", "right"))
    v_side = rng.choice(("top", "bottom"))

    cross_rows = [r for r in range(CROSS_MARGIN, h - CROSS_MARGIN) if _step_from_side(r, h, v_side) % 2 == 1]
    cross_cols = [c for c in range(CROSS_MARGIN, w - CROSS_MARGIN) if _step_from_side(c, w, h_side) % 2 == 1]
    if not cross_rows or not cross_cols:
        return None

    cross_row = rng.choice(cross_rows)
    cross_col = rng.choice(cross_cols)

    pipes: list[dict[str, int | str]] = []
    if h_side == "left":
        pipes.append(
            {
                "axis": "h",
                "center": cross_row,
                "c0": 1,
                "c1": rng.randint(cross_col + 2, w - 2),
                "marker_side": "left",
            }
        )
    else:
        pipes.append(
            {
                "axis": "h",
                "center": cross_row,
                "c0": rng.randint(1, cross_col - 2),
                "c1": w - 2,
                "marker_side": "right",
            }
        )

    if v_side == "top":
        pipes.append(
            {
                "axis": "v",
                "center": cross_col,
                "r0": 1,
                "r1": rng.randint(cross_row + 2, h - 2),
                "marker_side": "top",
            }
        )
    else:
        pipes.append(
            {
                "axis": "v",
                "center": cross_col,
                "r0": rng.randint(1, cross_row - 2),
                "r1": h - 2,
                "marker_side": "bottom",
            }
        )

    extra_axis = rng.choice(("h", "v"))
    if extra_axis == "h":
        row_candidates = [r for r in range(2, h - 2) if abs(r - cross_row) >= CENTER_GAP]
        max_extra = min(3, _max_spaced_count(row_candidates, CENTER_GAP))
        if max_extra == 0:
            return None
        count = rng.randint(1, max_extra)
        rows = _sample_spaced_values(rng, row_candidates, count, CENTER_GAP)
        if rows is None:
            return None
        for row in rows:
            side = rng.choice(("left", "right"))
            if side == "left":
                pipes.append(
                    {
                        "axis": "h",
                        "center": row,
                        "c0": 1,
                        "c1": rng.randint(MIN_PIPE_LENGTH, cross_col - 2),
                        "marker_side": "left",
                    }
                )
            else:
                pipes.append(
                    {
                        "axis": "h",
                        "center": row,
                        "c0": rng.randint(cross_col + 2, w - MIN_PIPE_LENGTH - 1),
                        "c1": w - 2,
                        "marker_side": "right",
                    }
                )
    else:
        col_candidates = [c for c in range(2, w - 2) if abs(c - cross_col) >= CENTER_GAP]
        max_extra = min(3, _max_spaced_count(col_candidates, CENTER_GAP))
        if max_extra == 0:
            return None
        count = rng.randint(1, max_extra)
        cols = _sample_spaced_values(rng, col_candidates, count, CENTER_GAP)
        if cols is None:
            return None
        for col in cols:
            side = rng.choice(("top", "bottom"))
            if side == "top":
                pipes.append(
                    {
                        "axis": "v",
                        "center": col,
                        "r0": 1,
                        "r1": rng.randint(MIN_PIPE_LENGTH, cross_row - 2),
                        "marker_side": "top",
                    }
                )
            else:
                pipes.append(
                    {
                        "axis": "v",
                        "center": col,
                        "r0": rng.randint(cross_row + 2, h - MIN_PIPE_LENGTH - 1),
                        "r1": h - 2,
                        "marker_side": "bottom",
                    }
                )

    colors = rng.sample(MARKER_PALETTE, len(pipes))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for spec, color in zip(pipes, colors):
        spec["color"] = color
        _draw_pipe(grid, spec)

    try:
        bg, pipe, markers = _extract_markers(grid)
        out = _transform(grid)
    except AssertionError:
        return None

    if bg != BG_COLOR or pipe != PIPE_COLOR:
        return None
    if len(markers) != len(pipes):
        return None
    if len({marker[2] for marker in markers}) != len(markers):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

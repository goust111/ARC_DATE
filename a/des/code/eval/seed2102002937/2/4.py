import random
from collections import Counter, deque

import numpy as np


SIDES = ("top", "bottom", "left", "right")
BG = 0

# Generator configuration.
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
FRAME_COUNT_RANGE = (2, 4)
FRAME_OUTER_RANGE = (4, 8)
FRAME_GAP_RANGE = (1, 3)
MARKER_DISTANCE_RANGE = (1, 2)
MARKER_SIZE_RANGE = (1, 2)
PLACEMENT_TRIES = 160
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _same_color_component(
    grid: np.ndarray,
    start: tuple[int, int],
    seen: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])

    q = deque([(sr, sc)])
    comp = []
    seen_local = {(sr, sc)} if seen is None else None
    if seen is not None:
        seen[sr, sc] = True

    while q:
        r, c = q.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == color:
                if seen is not None:
                    if seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                else:
                    if (nr, nc) in seen_local:
                        continue
                    seen_local.add((nr, nc))
                q.append((nr, nc))

    return comp


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            coords = _same_color_component(grid, (r, c), seen)
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": int(grid[r, c]),
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_contiguous(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return False
    return bool(np.all(np.diff(indices) == 1))


def _detect_frame(comp: dict) -> dict | None:
    coords = comp["coords"]
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    occ = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        occ[r - r0, c - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(occ & ~border):
        return None

    top_missing = np.flatnonzero(~occ[0, :])
    bottom_missing = np.flatnonzero(~occ[-1, :])
    left_missing = np.flatnonzero(~occ[:, 0])
    right_missing = np.flatnonzero(~occ[:, -1])

    sides = []
    if top_missing.size:
        sides.append(("top", top_missing))
    if bottom_missing.size:
        sides.append(("bottom", bottom_missing))
    if left_missing.size:
        sides.append(("left", left_missing))
    if right_missing.size:
        sides.append(("right", right_missing))
    if len(sides) != 1:
        return None

    side, missing = sides[0]
    if not _is_contiguous(missing):
        return None
    if side in ("top", "bottom"):
        if missing[0] == 0 or missing[-1] == w - 1:
            return None
        if missing.size > FRAME_GAP_RANGE[1]:
            return None
        gap0 = c0 + int(missing[0])
        gap1 = c0 + int(missing[-1])
    else:
        if missing[0] == 0 or missing[-1] == h - 1:
            return None
        if missing.size > FRAME_GAP_RANGE[1]:
            return None
        gap0 = r0 + int(missing[0])
        gap1 = r0 + int(missing[-1])

    if side != "top" and not occ[0, :].all():
        return None
    if side != "bottom" and not occ[-1, :].all():
        return None
    if side != "left" and not occ[:, 0].all():
        return None
    if side != "right" and not occ[:, -1].all():
        return None

    return {
        "frame_color": color,
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap": (gap0, gap1),
    }


def _find_marker(
    grid: np.ndarray,
    bg: int,
    frame: dict,
) -> tuple[int, list[tuple[int, int]]] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    gap0, gap1 = frame["gap"]

    if side == "top":
        row_groups = [[r0 - 1], [r0 - 2]] if r0 >= 2 else [[r0 - 1]]
        col_groups = [list(range(gap0, gap1 + 1))]
    elif side == "bottom":
        row_groups = [[r1 + 1], [r1 + 2]] if r1 + 2 < h else [[r1 + 1]] if r1 + 1 < h else []
        col_groups = [list(range(gap0, gap1 + 1))]
    elif side == "left":
        row_groups = [list(range(gap0, gap1 + 1))]
        col_groups = [[c0 - 1], [c0 - 2]] if c0 >= 2 else [[c0 - 1]]
    else:
        row_groups = [list(range(gap0, gap1 + 1))]
        col_groups = [[c1 + 1], [c1 + 2]] if c1 + 2 < w else [[c1 + 1]] if c1 + 1 < w else []

    for rows in row_groups:
        for cols in col_groups:
            for r in rows:
                if not (0 <= r < h):
                    continue
                for c in cols:
                    if not (0 <= c < w):
                        continue
                    color = int(grid[r, c])
                    if color == bg:
                        continue
                    seen = np.zeros(grid.shape, dtype=bool)
                    coords = _same_color_component(grid, (r, c), seen)
                    if not (1 <= len(coords) <= MARKER_SIZE_RANGE[1]):
                        continue

                    rows2 = [p[0] for p in coords]
                    cols2 = [p[1] for p in coords]
                    rr0, rr1 = min(rows2), max(rows2)
                    cc0, cc1 = min(cols2), max(cols2)

                    if side == "top":
                        if rr1 >= r0 or cc0 < gap0 or cc1 > gap1:
                            continue
                        distance = r0 - rr1
                    elif side == "bottom":
                        if rr0 <= r1 or cc0 < gap0 or cc1 > gap1:
                            continue
                        distance = rr0 - r1
                    elif side == "left":
                        if cc1 >= c0 or rr0 < gap0 or rr1 > gap1:
                            continue
                        distance = c0 - cc1
                    else:
                        if cc0 <= c1 or rr0 < gap0 or rr1 > gap1:
                            continue
                        distance = cc0 - c1

                    if MARKER_DISTANCE_RANGE[0] <= distance <= MARKER_DISTANCE_RANGE[1]:
                        return color, coords

    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames = []
    for comp in _components_by_color(g, bg):
        frame = _detect_frame(comp)
        if frame is not None:
            frames.append(frame)

    for frame in frames:
        marker = _find_marker(g, bg, frame)
        if marker is None:
            continue

        fill_color, marker_coords = marker
        r0, c0, r1, c1 = frame["bbox"]
        frame_color = frame["frame_color"]
        gap0, gap1 = frame["gap"]

        if frame["side"] == "top":
            out[r0, gap0 : gap1 + 1] = frame_color
        elif frame["side"] == "bottom":
            out[r1, gap0 : gap1 + 1] = frame_color
        elif frame["side"] == "left":
            out[gap0 : gap1 + 1, c0] = frame_color
        else:
            out[gap0 : gap1 + 1, c1] = frame_color

        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        for r, c in marker_coords:
            out[r, c] = bg

    return out


def _draw_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        h = spec["h"]
        w = spec["w"]
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        color = spec["frame_color"]

        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        gap0, gap1 = spec["gap"]
        if spec["side"] == "top":
            grid[r0, gap0 : gap1 + 1] = BG
        elif spec["side"] == "bottom":
            grid[r1, gap0 : gap1 + 1] = BG
        elif spec["side"] == "left":
            grid[gap0 : gap1 + 1, c0] = BG
        else:
            grid[gap0 : gap1 + 1, c1] = BG

        for r, c in spec["marker_coords"]:
            grid[r, c] = spec["fill_color"]

    return grid


def _draw_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = _draw_input(shape, specs)
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        h = spec["h"]
        w = spec["w"]
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        gap0, gap1 = spec["gap"]
        color = spec["frame_color"]

        if spec["side"] == "top":
            out[r0, gap0 : gap1 + 1] = color
        elif spec["side"] == "bottom":
            out[r1, gap0 : gap1 + 1] = color
        elif spec["side"] == "left":
            out[gap0 : gap1 + 1, c0] = color
        else:
            out[gap0 : gap1 + 1, c1] = color

        out[r0 + 1 : r1, c0 + 1 : c1] = spec["fill_color"]
        for r, c in spec["marker_coords"]:
            out[r, c] = BG

    return out


def _sample_object(
    rng: random.Random,
    h_grid: int,
    w_grid: int,
    frame_color: int,
    fill_color: int,
    occupied: np.ndarray,
) -> dict | None:
    for _ in range(PLACEMENT_TRIES):
        side = rng.choice(SIDES)
        gap_len = rng.randint(*FRAME_GAP_RANGE)

        if side in ("top", "bottom"):
            outer_w_min = max(FRAME_OUTER_RANGE[0], gap_len + 2)
            if outer_w_min > FRAME_OUTER_RANGE[1]:
                continue
            outer_h = rng.randint(*FRAME_OUTER_RANGE)
            outer_w = rng.randint(outer_w_min, FRAME_OUTER_RANGE[1])
            side_span = outer_w
        else:
            outer_h_min = max(FRAME_OUTER_RANGE[0], gap_len + 2)
            if outer_h_min > FRAME_OUTER_RANGE[1]:
                continue
            outer_h = rng.randint(outer_h_min, FRAME_OUTER_RANGE[1])
            outer_w = rng.randint(*FRAME_OUTER_RANGE)
            side_span = outer_h

        gap_start = rng.randint(1, side_span - gap_len - 1)
        marker_len = rng.randint(1, min(MARKER_SIZE_RANGE[1], gap_len))
        marker_offset = rng.randint(0, gap_len - marker_len)
        marker_dist = rng.randint(*MARKER_DISTANCE_RANGE)

        if side == "top":
            r0_min = marker_dist + 1
            r0_max = h_grid - outer_h - 1
            c0_min = 1
            c0_max = w_grid - outer_w - 1
        elif side == "bottom":
            r0_min = 1
            r0_max = h_grid - outer_h - marker_dist - 1
            c0_min = 1
            c0_max = w_grid - outer_w - 1
        elif side == "left":
            r0_min = 1
            r0_max = h_grid - outer_h - 1
            c0_min = marker_dist + 1
            c0_max = w_grid - outer_w - 1
        else:
            r0_min = 1
            r0_max = h_grid - outer_h - 1
            c0_min = 1
            c0_max = w_grid - outer_w - marker_dist - 1

        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if side in ("top", "bottom"):
            gap0 = c0 + gap_start
            gap1 = gap0 + gap_len - 1
            marker_c0 = gap0 + marker_offset
            marker_c1 = marker_c0 + marker_len - 1
            marker_row = r0 - marker_dist if side == "top" else r1 + marker_dist
            marker_coords = [(marker_row, c) for c in range(marker_c0, marker_c1 + 1)]
        else:
            gap0 = r0 + gap_start
            gap1 = gap0 + gap_len - 1
            marker_r0 = gap0 + marker_offset
            marker_r1 = marker_r0 + marker_len - 1
            marker_col = c0 - marker_dist if side == "left" else c1 + marker_dist
            marker_coords = [(r, marker_col) for r in range(marker_r0, marker_r1 + 1)]

        marker_rows = [p[0] for p in marker_coords]
        marker_cols = [p[1] for p in marker_coords]
        union_r0 = min(r0, min(marker_rows))
        union_c0 = min(c0, min(marker_cols))
        union_r1 = max(r1, max(marker_rows))
        union_c1 = max(c1, max(marker_cols))

        pad_r0 = union_r0 - 1
        pad_c0 = union_c0 - 1
        pad_r1 = union_r1 + 1
        pad_c1 = union_c1 + 1
        if pad_r0 < 0 or pad_c0 < 0 or pad_r1 >= h_grid or pad_c1 >= w_grid:
            continue
        if occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
            continue

        occupied[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
        return {
            "r0": r0,
            "c0": c0,
            "h": outer_h,
            "w": outer_w,
            "side": side,
            "gap": (gap0, gap1),
            "frame_color": frame_color,
            "fill_color": fill_color,
            "marker_coords": marker_coords,
        }

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h_grid = rng.randint(*GRID_H_RANGE)
    w_grid = rng.randint(*GRID_W_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    frame_colors = rng.sample(list(range(1, 10)), frame_count)
    fill_pool = [c for c in range(1, 10) if c not in frame_colors]
    occupied = np.zeros((h_grid, w_grid), dtype=bool)
    specs = []

    for frame_color in frame_colors:
        fill_color = rng.choice(fill_pool)
        spec = _sample_object(rng, h_grid, w_grid, frame_color, fill_color, occupied)
        if spec is None:
            return None
        specs.append(spec)

    grid = _draw_input((h_grid, w_grid), specs)
    expected = _draw_output((h_grid, w_grid), specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

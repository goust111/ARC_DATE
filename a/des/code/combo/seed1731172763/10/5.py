import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TOUCH_TO_COLOR = {
    0: BLUE,
    1: GREEN,
    2: YELLOW,
    3: ORANGE,
    4: RED,
}

INPUT_COLORS = (6, 8, 9)
SIDES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _spans_from_runs(length: int, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = []
    prev_end = -1
    for start, end in runs:
        if prev_end + 1 < start:
            spans.append((prev_end + 1, start))
        prev_end = end
    if prev_end + 1 < length:
        spans.append((prev_end + 1, length))
    return spans


def _find_panel_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r] == GREY)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert sep_rows and sep_cols, "expected grey separator rows and columns"

    row_spans = _spans_from_runs(h, _group_runs(sep_rows))
    col_spans = _spans_from_runs(w, _group_runs(sep_cols))
    assert row_spans and col_spans, "expected at least one panel"

    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "expected equal-sized panels"
    side = int(next(iter(row_sizes)))
    assert side == int(next(iter(col_sizes))), "expected square panels"
    assert side >= 2, "expected nontrivial panels"
    return row_spans, col_spans


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _touching_sides(mask: np.ndarray) -> set[str]:
    return {
        side
        for side, touches in (
            ("top", bool(mask[0, :].any())),
            ("bottom", bool(mask[-1, :].any())),
            ("left", bool(mask[:, 0].any())),
            ("right", bool(mask[:, -1].any())),
        )
        if touches
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each panel object by how many panel borders it touches."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_spans, col_spans = _find_panel_spans(g)
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0:r1, c0:c1]
            assert panel.shape[0] == panel.shape[1], "expected square panels"
            assert not np.any(panel == GREY), "separator color must stay outside panels"

            fg = panel != BLACK
            assert fg.any(), "expected one foreground object in each panel"
            components = _connected_components(fg)
            assert len(components) == 1, "expected one 4-connected object per panel"

            colors = np.unique(panel[fg])
            assert len(colors) == 1, "expected each panel object to be monochrome"

            touches = len(_touching_sides(fg))
            panel_out = out[r0:r1, c0:c1]
            panel_out[fg] = TOUCH_TO_COLOR[touches]

    return out


# --------------- generator ---------------

PANEL_SIDE_RANGE = (5, 8)
PANEL_LAYOUTS = [(2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
GEN_ATTEMPTS = 300
MASK_ATTEMPTS = 120


def _allowed_cells(side: int, touched: set[str]) -> list[tuple[int, int]]:
    cells = []
    for r in range(side):
        for c in range(side):
            if "top" not in touched and r == 0:
                continue
            if "bottom" not in touched and r == side - 1:
                continue
            if "left" not in touched and c == 0:
                continue
            if "right" not in touched and c == side - 1:
                continue
            cells.append((r, c))
    return cells


def _anchor_candidates(side: int, touched: set[str], edge: str) -> list[tuple[int, int]]:
    allowed = set(_allowed_cells(side, touched))
    if edge == "top":
        line = [(0, c) for c in range(side)]
    elif edge == "bottom":
        line = [(side - 1, c) for c in range(side)]
    elif edge == "left":
        line = [(r, 0) for r in range(side)]
    else:
        line = [(r, side - 1) for r in range(side)]
    return [cell for cell in line if cell in allowed]


def _manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    path = [(r, c)]
    row_first = random.random() < 0.5
    if row_first:
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            path.append((nr, c))
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            path.append((rr, nc))
    else:
        step = 1 if cc >= c else -1
        for nc in range(c + step, cc + step, step):
            path.append((r, nc))
        step = 1 if rr >= r else -1
        for nr in range(r + step, rr + step, step):
            path.append((nr, cc))
    return path


def _frontier(cells: set[tuple[int, int]], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            nbr = (r + dr, c + dc)
            if nbr in allowed and nbr not in cells:
                frontier.add(nbr)
    return list(frontier)


def _generate_panel_mask(side: int, touch_count: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        touched = set(random.sample(SIDES, touch_count))
        allowed_list = _allowed_cells(side, touched)
        allowed = set(allowed_list)
        interior = [(r, c) for r, c in allowed_list if 0 < r < side - 1 and 0 < c < side - 1]
        if not interior:
            continue

        cells = {random.choice(interior)}
        ok = True
        for edge in random.sample(list(touched), len(touched)):
            candidates = _anchor_candidates(side, touched, edge)
            if not candidates:
                ok = False
                break
            anchor = random.choice(candidates)
            attach = random.choice(tuple(cells))
            cells.update(_manhattan_path(anchor, attach))
        if not ok:
            continue

        min_target = max(len(cells), 2 if touch_count == 0 else len(cells))
        area_cap = int(round(len(allowed) * random.uniform(0.28, 0.62)))
        area_cap = min(len(allowed), max(min_target, area_cap + random.randint(0, max(1, side // 2))))
        target = random.randint(min_target, area_cap)
        while len(cells) < target:
            candidates = _frontier(cells, allowed)
            if not candidates:
                break
            cells.add(random.choice(candidates))

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if len(_connected_components(mask)) != 1:
            continue
        if _touching_sides(mask) != touched:
            continue
        return mask

    raise RuntimeError(f"failed to build panel mask for touch_count={touch_count}")


def _separator_positions(panel_count: int, side: int) -> list[int]:
    return [k * side + (k - 1) for k in range(1, panel_count)]


def _build_grids(
    n_rows: int,
    n_cols: int,
    side: int,
    specs: list[tuple[np.ndarray, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h = n_rows * side + (n_rows - 1)
    w = n_cols * side + (n_cols - 1)
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)

    for r in _separator_positions(n_rows, side):
        inp[r, :] = GREY
        out[r, :] = GREY
    for c in _separator_positions(n_cols, side):
        inp[:, c] = GREY
        out[:, c] = GREY

    idx = 0
    for pr in range(n_rows):
        for pc in range(n_cols):
            top = pr * (side + 1)
            left = pc * (side + 1)
            mask, color, touch_count = specs[idx]
            panel_in = inp[top : top + side, left : left + side]
            panel_out = out[top : top + side, left : left + side]
            panel_in[mask] = color
            panel_out[mask] = TOUCH_TO_COLOR[touch_count]
            idx += 1

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(PANEL_LAYOUTS)
    side = random.randint(*PANEL_SIDE_RANGE)
    n_panels = n_rows * n_cols

    touch_counts = [0, 1, 2, 3, 4]
    touch_counts.extend(random.randint(0, 4) for _ in range(n_panels - len(touch_counts)))
    random.shuffle(touch_counts)

    specs = []
    for touch_count in touch_counts:
        mask = _generate_panel_mask(side, touch_count)
        color = random.choice(INPUT_COLORS)
        specs.append((mask, color, touch_count))

    inp, expected = _build_grids(n_rows, n_cols, side, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

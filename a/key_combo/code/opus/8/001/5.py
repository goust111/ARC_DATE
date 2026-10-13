import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
BOTTOM_H = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _project_bottom_markers(bottom: np.ndarray, panel_w: int) -> np.ndarray:
    projected = np.zeros((bottom.shape[0], panel_w), dtype=int)
    for r, c in np.argwhere(bottom == GREEN):
        projected[int(r), int(c) % panel_w] = GREEN
    return projected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse the two top panels into complement blockers, then shoot upward rays from the lower markers.

    Input structure:
    - The grid has an upper region split into equal-width left and right panels, followed by four lower rows.
    - The left top panel contains blue foreground cells and the right top panel contains red foreground cells.
    - The lower rows contain isolated green marker pixels on a black background; marker columns are interpreted modulo one panel width.

    Rule:
    - Align the two top panels cell by cell and output a single panel-width top region.
    - A top output cell becomes gray iff the aligned left and right cells are both black; otherwise it becomes black.
    - Project every lower green marker into the compressed panel-width output and keep those markers.
    - From each lower marker, cast a ray upward through its column, painting black cells green until a gray blocker is reached.

    Key insight:
    The occupied-cell complement of the two aligned panels is not the final picture; it is the blocker layer that determines how far each upward ray can travel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > BOTTOM_H and w % 2 == 0

    panel_w = w // 2
    top_h = h - BOTTOM_H
    left = g[:top_h, :panel_w]
    right = g[:top_h, panel_w:]

    out = np.zeros((h, panel_w), dtype=int)
    blockers = (left == BLACK) & (right == BLACK)
    out_top = out[:top_h]
    out_top[blockers] = GREY

    bottom = _project_bottom_markers(g[top_h:], panel_w)
    out[top_h:] = bottom

    for r0, c0 in np.argwhere(bottom == GREEN):
        c = int(c0)
        for r in range(top_h - 1, -1, -1):
            if out[r, c] == GREY:
                break
            out[r, c] = GREEN

    return out


def _far_from_other_components(occupied: np.ndarray, r: int, c: int) -> bool:
    h, w = occupied.shape
    if not _in_bounds(h, w, r, c) or occupied[r, c]:
        return False
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(h, w, nr, nc) and occupied[nr, nc]:
            return False
    return True


def _add_component(panel: np.ndarray, occupied: np.ndarray, color: int, cells) -> None:
    for r, c in cells:
        panel[r, c] = color
        occupied[r, c] = True


def _grow_random_component(
    h: int, w: int, occupied: np.ndarray, size: int, banned_cols: set[int]
) -> list[tuple[int, int]] | None:
    starts = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if c not in banned_cols and _far_from_other_components(occupied, r, c)
    ]
    random.shuffle(starts)
    for start in starts:
        shape = {start}
        while len(shape) < size:
            frontier = []
            for r, c in shape:
                for nr, nc in _neighbors4(r, c):
                    if (
                        _in_bounds(h, w, nr, nc)
                        and nc not in banned_cols
                        and (nr, nc) not in shape
                        and _far_from_other_components(occupied, nr, nc)
                    ):
                        frontier.append((nr, nc))
            if not frontier:
                break
            shape.add(random.choice(frontier))
        if len(shape) == size:
            return list(shape)
    return None


def _place_extra_components(
    panel: np.ndarray,
    occupied: np.ndarray,
    color: int,
    count: int,
    banned_cols: set[int],
) -> bool:
    h, w = panel.shape
    for _ in range(count):
        size = random.randint(2, 5)
        cells = _grow_random_component(h, w, occupied, size, banned_cols)
        if cells is None:
            return False
        _add_component(panel, occupied, color, cells)
    return True


def _safe_split(top_h: int) -> int:
    lo = max(2, top_h - 5)
    hi = min(5, top_h - 2)
    return random.randint(lo, hi)


def _build_top_panels(top_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    left = np.zeros((top_h, panel_w), dtype=int)
    right = np.zeros((top_h, panel_w), dtype=int)
    occ_left = np.zeros_like(left, dtype=bool)
    occ_right = np.zeros_like(right, dtype=bool)

    safe_col = random.randrange(panel_w)
    stop_choices = [c for c in range(panel_w) if abs(c - safe_col) >= 2]
    if not stop_choices:
        return None
    stop_col = random.choice(stop_choices)

    split = _safe_split(top_h)
    if random.random() < 0.5:
        safe_left = [(r, safe_col) for r in range(split)]
        safe_right = [(r, safe_col) for r in range(split, top_h)]
    else:
        safe_right = [(r, safe_col) for r in range(split)]
        safe_left = [(r, safe_col) for r in range(split, top_h)]
    _add_component(left, occ_left, BLUE, safe_left)
    _add_component(right, occ_right, RED, safe_right)

    stop_owner_left = random.random() < 0.5
    stop_len = random.randint(2, min(4, top_h - 1))
    stop_cells = [(r, stop_col) for r in range(top_h - stop_len, top_h)]
    if stop_owner_left:
        _add_component(left, occ_left, BLUE, stop_cells)
    else:
        _add_component(right, occ_right, RED, stop_cells)

    left_extra = random.randint(0, 1 if stop_owner_left else 2)
    right_extra = random.randint(0, 1 if not stop_owner_left else 2)
    banned_cols = {safe_col, stop_col}
    if not _place_extra_components(left, occ_left, BLUE, left_extra, banned_cols):
        return None
    if not _place_extra_components(right, occ_right, RED, right_extra, banned_cols):
        return None

    return left, right, safe_col, stop_col


def _marker_columns(panel_w: int, safe_col: int, stop_col: int) -> list[int]:
    count = random.randint(2, min(4, panel_w))
    cols = [safe_col, stop_col]
    extras = [c for c in range(panel_w) if c not in cols]
    random.shuffle(extras)
    cols.extend(extras[: count - len(cols)])
    random.shuffle(cols)
    return cols


def _make_input() -> np.ndarray | None:
    top_h = random.randint(6, 10)
    panel_w = random.randint(4, 8)
    built = _build_top_panels(top_h, panel_w)
    if built is None:
        return None
    left, right, safe_col, stop_col = built

    complement = (left == BLACK) & (right == BLACK)
    if int(complement.sum()) < 2:
        return None
    if complement[:, safe_col].any():
        return None
    if not complement[:, stop_col].any():
        return None
    if complement[top_h - 1, stop_col]:
        return None

    grid = np.zeros((top_h + BOTTOM_H, 2 * panel_w), dtype=int)
    grid[:top_h, :panel_w] = left
    grid[:top_h, panel_w:] = right

    for col in _marker_columns(panel_w, safe_col, stop_col):
        row = top_h + random.randrange(BOTTOM_H)
        side = random.randrange(2)
        grid[row, col + side * panel_w] = GREEN

    out = main(grid)
    if out.shape != (top_h + BOTTOM_H, panel_w):
        return None
    if out[0, safe_col] != GREEN:
        return None
    blocker_rows = np.where(out[:top_h, stop_col] == GREY)[0]
    if len(blocker_rows) == 0:
        return None
    first_blocker = int(blocker_rows[-1])
    if not np.all(out[first_blocker + 1 : top_h, stop_col] == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import numpy as np


BG = 0
CONTROL_COLOR = 5
CORNER_COLOR = 9
WORK_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRY_LIMIT = 400


def _corner_name_and_pos(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    h, w = grid.shape
    corners = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }
    hits = [(name, pos) for name, pos in corners.items() if int(grid[pos]) == CORNER_COLOR]
    assert len(hits) == 1, "expected exactly one maroon corner marker"
    return hits[0]


def _triangle_mask(shape: tuple[int, int], corner: str, side: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    if corner == "TL":
        return rr + cc < side
    if corner == "TR":
        return rr + (w - 1 - cc) < side
    if corner == "BL":
        return (h - 1 - rr) + cc < side
    if corner == "BR":
        return (h - 1 - rr) + (w - 1 - cc) < side
    raise ValueError(f"unknown corner {corner}")


def _marked_columns(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    cols: list[int] = []
    for c in range(w):
        if int(grid[h - 1, c]) == BG:
            continue
        if np.any(grid[: h - 1, c] != BG):
            cols.append(c)
    return cols


def _propagate_marked_columns(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)
    for c in _marked_columns(g):
        current = int(g[h - 1, c])
        for r in range(h - 1, -1, -1):
            color = int(g[r, c])
            if color != BG:
                current = color
            if current != BG:
                out[r, c] = current
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each bottom-marked column upward, then keep only the part inside a count-sized corner triangle.

    Input structure:
    - The background is black.
    - The bottom row contains 2-4 marked columns; each real marked column has a non-black bottom pixel and at least one more non-black pixel above it.
    - Exactly one corner contains a maroon marker pixel that selects the triangle vertex, and grey singleton control pixels are scattered outside the marked columns.

    Rule:
    - Identify the marked columns from the bottom row and propagate colors upward within each such column by scanning bottom-to-top and filling black cells with the most recently seen non-black color.
    - Count the grey control pixels; this count is the side length of a right triangle rooted at the maroon corner, defined by cells whose two edge-distances sum to less than that count.
    - Output only the propagated column pixels that lie inside that triangle; everything else becomes black, including the maroon corner and all grey control points.

    Key insight:
    - The bottom-row markers and the grey controls drive two independent computations: column propagation determines color bands, while the control count determines a geometric crop that is applied only at the end.
    """
    g = np.asarray(grid, dtype=int)
    corner, _ = _corner_name_and_pos(g)
    side = int(np.count_nonzero(g == CONTROL_COLOR))
    assert side > 0, "expected at least one grey control pixel"

    propagated = _propagate_marked_columns(g)
    mask = _triangle_mask(g.shape, corner, side)
    out = np.zeros_like(g)
    out[mask] = propagated[mask]
    return out


def _corner_col(corner: str, w: int) -> int:
    return 0 if corner in {"TL", "BL"} else w - 1


def _triangle_columns(corner: str, w: int, side: int) -> list[int]:
    if corner in {"TL", "BL"}:
        return list(range(min(side, w)))
    return list(range(max(0, w - side), w))


def _choose_marked_columns(h: int, w: int, corner: str, side: int, rng: np.random.Generator) -> list[int] | None:
    del h
    corner_col = _corner_col(corner, w)
    near_cols = [c for c in _triangle_columns(corner, w, side) if c != corner_col]
    all_cols = [c for c in range(w) if c != corner_col]
    far_cols = [c for c in all_cols if c not in near_cols]
    if len(near_cols) < 2:
        return None

    mark_count = int(rng.integers(2, 5))
    max_inside = min(mark_count, len(near_cols))
    if max_inside < 2:
        return None

    if mark_count > 2 and far_cols and rng.random() < 0.7:
        inside_count = 2
    else:
        inside_count = int(rng.integers(2, max_inside + 1))

    outside_count = mark_count - inside_count
    if outside_count > len(far_cols):
        return None

    inside = list(rng.choice(near_cols, size=inside_count, replace=False))
    outside = list(rng.choice(far_cols, size=outside_count, replace=False)) if outside_count else []
    cols = sorted(int(c) for c in inside + outside)
    return cols


def _sample_band_rows(h: int, extra_count: int, rng: np.random.Generator) -> list[int]:
    rows = sorted(int(r) for r in rng.choice(np.arange(h - 1), size=extra_count, replace=False))
    return rows


def _place_controls(
    grid: np.ndarray,
    corner: str,
    marked_cols: list[int],
    side: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    corner_r = 0 if corner in {"TL", "TR"} else h - 1
    corner_c = _corner_col(corner, w)
    blocked = np.argwhere(grid != BG)
    occupied = {tuple(map(int, rc)) for rc in blocked}

    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if c in marked_cols or c == corner_c:
                continue
            if r in {corner_r, h - 1}:
                continue
            if (r, c) in occupied:
                continue
            candidates.append((r, c))
    rng.shuffle(candidates)

    placed: list[tuple[int, int]] = []
    for r, c in candidates:
        ok = True
        for rr, cc in placed:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                ok = False
                break
        if not ok:
            continue
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                ok = False
                break
        if not ok:
            continue
        grid[r, c] = CONTROL_COLOR
        placed.append((r, c))
        if len(placed) == side:
            break
    return len(placed) == side


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(13, 19))
    w = int(rng.integers(13, 19))
    side_max = min(8, h - 2, w - 1)
    if side_max < 3:
        return None

    corner = str(rng.choice(["TL", "TR", "BL", "BR"]))
    side = int(rng.integers(3, side_max + 1))
    marked_cols = _choose_marked_columns(h, w, corner, side, rng)
    if marked_cols is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    corner_r = 0 if corner in {"TL", "TR"} else h - 1
    corner_c = _corner_col(corner, w)
    grid[corner_r, corner_c] = CORNER_COLOR

    bottom_colors = list(rng.choice(WORK_COLORS, size=len(marked_cols), replace=False))
    for c, bottom_color in zip(marked_cols, bottom_colors):
        grid[h - 1, c] = int(bottom_color)
        extra_count = int(rng.integers(1, 4))
        rows = _sample_band_rows(h, extra_count, rng)
        colors = [int(rng.choice(WORK_COLORS)) for _ in range(extra_count)]
        if all(color == int(bottom_color) for color in colors):
            colors[0] = int(rng.choice([col for col in WORK_COLORS if col != int(bottom_color)]))
        for r, color in zip(rows, colors):
            grid[r, c] = int(color)

    if not _place_controls(grid, corner, marked_cols, side, rng):
        return None

    if int(np.count_nonzero(grid == CONTROL_COLOR)) != side:
        return None
    if int(np.count_nonzero(grid == CORNER_COLOR)) != 1:
        return None

    triangle_cols = set(_triangle_columns(corner, w, side))
    if len([c for c in marked_cols if c in triangle_cols]) < 2:
        return None

    out = main(grid)
    surviving_cols = [c for c in marked_cols if np.any(out[:, c] != BG)]
    if len(surviving_cols) < 2:
        return None
    if np.any(out == CONTROL_COLOR) or np.any(out == CORNER_COLOR):
        return None
    if not np.any(out != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
MARK = 8

_BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
)


def _coords_to_patch(coords: np.ndarray) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[coords[:, 0], coords[:, 1]] = 1
    return patch


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=int).copy()
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _rotate_coords(coords: np.ndarray, k: int) -> np.ndarray:
    patch = _coords_to_patch(_normalize_coords(coords))
    return _normalize_coords(np.argwhere(np.rot90(patch, k=k) != 0))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _has_row_width_variation(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    widths = [int(np.sum(rows == r)) for r in np.unique(rows)]
    return len(widths) >= 2 and len(set(widths)) >= 2


def _can_place(
    occupied: np.ndarray,
    coords: np.ndarray,
    r0: int,
    c0: int,
    *,
    pad: int,
) -> bool:
    h, w = occupied.shape
    placed = coords + np.array([r0, c0], dtype=int)
    if (
        np.any(placed[:, 0] < 0)
        or np.any(placed[:, 0] >= h)
        or np.any(placed[:, 1] < 0)
        or np.any(placed[:, 1] >= w)
    ):
        return False
    for r, c in placed:
        rr0 = max(0, int(r) - pad)
        rr1 = min(h, int(r) + pad + 1)
        cc0 = max(0, int(c) - pad)
        cc1 = min(w, int(c) + pad + 1)
        if np.any(occupied[rr0:rr1, cc0:cc1]):
            return False
    return True


def _stamp(occupied: np.ndarray, coords: np.ndarray, r0: int, c0: int) -> None:
    placed = coords + np.array([r0, c0], dtype=int)
    occupied[placed[:, 0], placed[:, 1]] = True


def _paint_objects(
    h: int,
    w: int,
    objects: list[dict[str, int | np.ndarray]],
    *,
    use_input_rows: bool,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = BLUE
    grid[h - 1, :] = RED
    for obj in objects:
        r0 = int(obj["input_r0"] if use_input_rows else obj["landing_r0"])
        c0 = int(obj["c0"])
        coords = np.asarray(obj["coords"], dtype=int) + np.array([r0, c0], dtype=int)
        color = int(obj["color"])
        grid[coords[:, 0], coords[:, 1]] = color
    return grid


def _find_objects(grid: np.ndarray) -> list[dict[str, int | np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int | np.ndarray]] = []
    for r in range(1, h - 1):
        for c in range(w):
            color = int(grid[r, c])
            if color not in (BLUE, RED) or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 1 <= nr < h - 1 and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": np.array([r0, c0, r1, c1], dtype=int),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _slide_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full_like(g, BG)
    out[0, :] = BLUE
    out[h - 1, :] = RED
    for obj in _find_objects(g):
        coords = np.asarray(obj["coords"], dtype=int)
        r0, _c0, r1, _c1 = map(int, obj["bbox"])
        color = int(obj["color"])
        dr = 1 - r0 if color == BLUE else (h - 2) - r1
        moved = coords + np.array([dr, 0], dtype=int)
        assert np.all(out[moved[:, 0], moved[:, 1]] == BG)
        out[moved[:, 0], moved[:, 1]] = color
    return out


def _mark_foreground_columns(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    selected = np.any(out[1:-1, :] != BG, axis=0)
    out[(out == BG) & selected[np.newaxis, :]] = MARK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide blue and red objects to their matching border bars, mark the resulting foreground-bearing columns, then tile the marked grid 2x2.

    Input structure:
    - The background is black.
    - The top row is a full blue bar and the bottom row is a full red bar.
    - The interior contains 4-connected monochrome blue or red objects that are separated from the border rows.

    Rule:
    - Move every blue object straight upward until its bbox top reaches row 1.
    - Move every red object straight downward until its bbox bottom reaches row H-2.
    - In the slid grid, find every interior column containing at least one blue or red pixel and recolor all remaining black cells in those columns to teal.
    - Tile the whole marked grid twice vertically and twice horizontally.

    Key insight:
    - The border bars define the slide destinations, and the column marking must be computed after the objects have been relocated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    assert h >= 5
    assert np.all(g[0, :] == BLUE)
    assert np.all(g[h - 1, :] == RED)
    assert set(map(int, np.unique(g))).issubset({BG, BLUE, RED})

    slid = _slide_grid(g)
    marked = _mark_foreground_columns(slid)
    return np.tile(marked, (2, 2))


# --------------- generator ---------------

_GEN_RETRIES = 400


def _random_shape(rng: np.random.Generator, *, force_complex: bool) -> tuple[np.ndarray, bool]:
    choices = list(_BASE_SHAPES)
    rng.shuffle(choices)
    for base in choices:
        coords = np.asarray(base, dtype=int)
        for k in rng.permutation(4):
            rotated = _rotate_coords(coords, int(k))
            complex_shape = _has_row_width_variation(rotated)
            if not force_complex or complex_shape:
                return rotated, complex_shape
    raise RuntimeError("no shape available")


def _wrong_no_slide_output(inp: np.ndarray) -> np.ndarray:
    return np.tile(_mark_foreground_columns(inp), (2, 2))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(13, 16))
    w = int(rng.integers(8, 16))
    n_objects = int(rng.integers(2, 5))
    blue_count = int(rng.integers(1, n_objects))
    colors = [BLUE] * blue_count + [RED] * (n_objects - blue_count)
    rng.shuffle(colors)

    objects: list[dict[str, int | np.ndarray]] = []
    output_occupied = np.zeros((h, w), dtype=bool)
    complex_needed = True

    for idx, color in enumerate(colors):
        placed = False
        remaining = len(colors) - idx
        force_complex = complex_needed and remaining == 1
        for _ in range(80):
            coords, complex_shape = _random_shape(rng, force_complex=force_complex)
            shape_h = int(coords[:, 0].max()) + 1
            shape_w = int(coords[:, 1].max()) + 1
            landing_r0 = 1 if color == BLUE else h - 1 - shape_h
            cols = np.arange(0, w - shape_w + 1, dtype=int)
            rng.shuffle(cols)
            for c0 in cols:
                if not _can_place(output_occupied, coords, landing_r0, int(c0), pad=1):
                    continue
                _stamp(output_occupied, coords, landing_r0, int(c0))
                objects.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "landing_r0": int(landing_r0),
                        "c0": int(c0),
                        "input_r0": -1,
                    }
                )
                complex_needed = complex_needed and (not complex_shape)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    slid = _paint_objects(h, w, objects, use_input_rows=False)
    selected = np.any(slid[1:-1, :] != BG, axis=0)
    n_selected = int(np.sum(selected))
    if not (2 <= n_selected <= w - 2):
        return None

    marked = _mark_foreground_columns(slid)
    if not np.any(marked == MARK):
        return None

    expected = np.tile(marked, (2, 2))
    input_occupied = np.zeros((h, w), dtype=bool)
    order = rng.permutation(len(objects))
    for idx in order:
        obj = objects[int(idx)]
        coords = np.asarray(obj["coords"], dtype=int)
        shape_h = int(coords[:, 0].max()) + 1
        c0 = int(obj["c0"])
        if int(obj["color"]) == BLUE:
            candidates = np.arange(2, (h // 2) - shape_h + 1, dtype=int)
        else:
            candidates = np.arange((h // 2) + 1, int(obj["landing_r0"]), dtype=int)
        if candidates.size == 0:
            return None
        rng.shuffle(candidates)
        placed = False
        for r0 in candidates:
            if not _can_place(input_occupied, coords, int(r0), c0, pad=1):
                continue
            _stamp(input_occupied, coords, int(r0), c0)
            obj["input_r0"] = int(r0)
            placed = True
            break
        if not placed:
            return None

    inp = _paint_objects(h, w, objects, use_input_rows=True)
    if np.array_equal(expected, _wrong_no_slide_output(inp)):
        return None
    if np.array_equal(expected, np.tile(slid, (2, 2))):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if len(_find_objects(inp)) != len(objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")

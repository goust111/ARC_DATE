from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GUIDE = 3
GRAY = 5

TILE_SIZE = 5
GEN_RETRIES = 200

_GEN_EDGE_LENGTH_PAIRS = ((1, 2), (2, 1), (1, 3), (3, 1))
_GEN_GUIDE_SEGMENTS = ((), (1,), (2,), (3,), (1, 2), (2, 3), (1, 2, 3))
_ROW_CANDIDATES_BY_K: dict[int, list[np.ndarray]] | None = None


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                            seen[nr, nc] = True
                            queue.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _find_guide(grid: np.ndarray) -> tuple[str, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square tile"
    n = int(g.shape[0])
    assert n * n <= 30, "expanded ARC canvas must stay within 30x30"

    row_guides = [r for r in range(n) if np.all(g[r] == GUIDE)]
    col_guides = [c for c in range(n) if np.all(g[:, c] == GUIDE)]
    assert len(row_guides) + len(col_guides) == 1, "expected exactly one full green guide row or column"
    if row_guides:
        return "row", int(row_guides[0])
    return "col", int(col_guides[0])


def _tile_band(grid: np.ndarray) -> tuple[np.ndarray, str, int]:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    axis, k = _find_guide(g)
    tiled = np.full((n * n, n * n), BG, dtype=int)

    if axis == "row":
        r0 = k * n
        for block_col in range(n):
            c0 = block_col * n
            tiled[r0 : r0 + n, c0 : c0 + n] = g
    else:
        c0 = k * n
        for block_row in range(n):
            r0 = block_row * n
            tiled[r0 : r0 + n, c0 : c0 + n] = g

    tiled[tiled == GUIDE] = GRAY
    return tiled, axis, k


def _recolor_tiled_band(tiled: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    gray_components = _components8(tiled == GRAY)
    shape_keys = [_shape_key(component) for component in gray_components]
    counts = Counter(shape_keys)

    out = np.full_like(tiled, BG)
    records: list[dict[str, object]] = []
    for component, shape_key in zip(gray_components, shape_keys):
        color = RED if counts[shape_key] == 1 else BLUE
        out[component[:, 0], component[:, 1]] = color
        records.append(
            {
                "coords": component,
                "shape_key": shape_key,
                "color": color,
            }
        )
    return out, records


def _oracle_output(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    tiled, axis, k = _tile_band(grid)
    out, records = _recolor_tiled_band(tiled)
    return out, {"tiled": tiled, "axis": axis, "guide_index": k, "records": records}


def _crosses_tile_boundary(coords: np.ndarray, n: int, axis: str) -> bool:
    values = coords[:, 1] if axis == "row" else coords[:, 0]
    return len({int(value // n) for value in values}) > 1


def _near_guide(coords: np.ndarray, n: int, axis: str, k: int) -> bool:
    local = coords[:, 0] % n if axis == "row" else coords[:, 1] % n
    return bool(np.any(np.abs(local - k) <= 1))


def _naive_no_merge_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    axis, k = _find_guide(g)
    tile = g.copy()
    tile[tile == GUIDE] = GRAY
    components = _components8(tile == GRAY)
    keys = [_shape_key(component) for component in components]
    counts = Counter(keys)

    out = np.full((n * n, n * n), BG, dtype=int)
    for block in range(n):
        if axis == "row":
            r0 = k * n
            c0 = block * n
        else:
            r0 = block * n
            c0 = k * n
        for component, shape_key in zip(components, keys):
            total_count = counts[shape_key] * n
            color = RED if total_count == 1 else BLUE
            out[r0 + component[:, 0], c0 + component[:, 1]] = color
    return out


def _merge_status_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    tiled, axis, _k = _tile_band(g)
    components = _components8(tiled == GRAY)

    out = np.full_like(tiled, BG)
    for component in components:
        color = RED if _crosses_tile_boundary(component, n, axis) else BLUE
        out[component[:, 0], component[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the guide-selected band, merge grey regions across tile boundaries, and recolor each merged component by post-tiling shape frequency.

    Input structure:
    - The input is a square tile with black background, grey foreground objects, and exactly one full green guide row or guide column.
    - Grey objects are 8-connected within the tile and some may touch the tile edges parallel to the guide-selected tiling direction.
    - The guide line is the only complete green row or column and it indicates which block row or block column survives after expansion.

    Rule:
    - Let n be the tile size and k the guide row or column index.
    - Create an n^2 by n^2 black canvas made of an n by n array of tile-sized blocks.
    - If the guide is a row, copy the tile into every block of block-row k; if the guide is a column, copy the tile into every block of block-column k.
    - Replace every copied green guide cell with grey, then find all 8-connected grey components in the expanded band.
    - Normalize each component to its own bounding-box mask, group components by that normalized shape, color singleton shapes red and repeated shapes blue, and keep black background unchanged.

    Key insight:
    - The decisive components are the ones after tiling: edge-touching pieces and the converted guide line can join across block boundaries, which changes the shape-frequency counts that decide the recolor.
    """
    g = np.asarray(grid, dtype=int)
    out, _summary = _oracle_output(g)
    return out


def _row_far_options(k: int) -> tuple[int, ...]:
    if k < TILE_SIZE // 2:
        return (TILE_SIZE - 1,)
    if k > TILE_SIZE // 2:
        return (0,)
    return (0, TILE_SIZE - 1)


def _build_row_candidate(k: int, far_row: int, left_len: int, right_len: int, guide_cols: tuple[int, ...]) -> np.ndarray:
    g = np.full((TILE_SIZE, TILE_SIZE), BG, dtype=int)
    g[k, :] = GUIDE
    g[far_row, :left_len] = GRAY
    g[far_row, TILE_SIZE - right_len :] = GRAY

    guide_row = k + 1 if far_row > k else k - 1
    if 0 <= guide_row < TILE_SIZE:
        g[guide_row, list(guide_cols)] = GRAY
    return g


def _generated_grid_valid(grid: np.ndarray, *, check_main: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.shape != (TILE_SIZE, TILE_SIZE):
        return False

    try:
        axis, k = _find_guide(g)
        out, summary = _oracle_output(g)
    except AssertionError:
        return False

    input_components = _components8(g == GRAY)
    input_shape_counts = Counter(_shape_key(component) for component in input_components)
    if len(input_components) < 2 or len(input_shape_counts) < 2:
        return False

    if axis == "row":
        touches_relevant_edge = any(np.any((component[:, 1] == 0) | (component[:, 1] == TILE_SIZE - 1)) for component in input_components)
    else:
        touches_relevant_edge = any(np.any((component[:, 0] == 0) | (component[:, 0] == TILE_SIZE - 1)) for component in input_components)
    if not touches_relevant_edge:
        return False

    colors = set(map(int, np.unique(out))) - {BG}
    if colors != {BLUE, RED}:
        return False

    if np.array_equal(out, _naive_no_merge_output(g)):
        return False
    if np.array_equal(out, _merge_status_output(g)):
        return False

    red_far = False
    blue_crosses_boundary = False
    for record in summary["records"]:
        coords = np.asarray(record["coords"], dtype=int)
        color = int(record["color"])
        if color == RED and not _near_guide(coords, TILE_SIZE, axis, k):
            red_far = True
        if color == BLUE and _crosses_tile_boundary(coords, TILE_SIZE, axis):
            blue_crosses_boundary = True
    if not red_far or not blue_crosses_boundary:
        return False

    if check_main and not np.array_equal(main(g), out):
        return False
    return True


def _row_candidates_by_k() -> dict[int, list[np.ndarray]]:
    global _ROW_CANDIDATES_BY_K
    if _ROW_CANDIDATES_BY_K is not None:
        return _ROW_CANDIDATES_BY_K

    candidates: dict[int, list[np.ndarray]] = defaultdict(list)
    seen: set[bytes] = set()
    for k in range(TILE_SIZE):
        for far_row in _row_far_options(k):
            for left_len, right_len in _GEN_EDGE_LENGTH_PAIRS:
                for guide_cols in _GEN_GUIDE_SEGMENTS:
                    candidate = _build_row_candidate(k, far_row, left_len, right_len, guide_cols)
                    if not _generated_grid_valid(candidate, check_main=False):
                        continue
                    key = candidate.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates[k].append(candidate)

    assert all(candidates[k] for k in range(TILE_SIZE)), "expected valid generators for every guide index"
    _ROW_CANDIDATES_BY_K = dict(candidates)
    return _ROW_CANDIDATES_BY_K


def generate_input() -> np.ndarray:
    candidates_by_k = _row_candidates_by_k()
    for _ in range(GEN_RETRIES):
        k = random.randrange(TILE_SIZE)
        base = random.choice(candidates_by_k[k]).copy()
        if random.random() < 0.5:
            base = base.T
        if _generated_grid_valid(base, check_main=True):
            return base
    raise RuntimeError("failed to generate a valid tiling-frequency ARC input")

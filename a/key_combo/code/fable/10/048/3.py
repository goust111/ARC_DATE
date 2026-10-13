import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
MAROON = 9

_DIRS = {
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0),
}

_GEN_RETRIES = 1200
_MIN_GAP = 1


def _find_components(grid: np.ndarray, colors: Iterable[int]) -> list[dict]:
    colors = set(int(c) for c in colors)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _extract_squares(grid: np.ndarray) -> list[dict]:
    squares = []
    for idx, comp in enumerate(_find_components(grid, [GREY, PINK])):
        r0, c0, r1, c1 = comp["bbox"]
        size_r = r1 - r0 + 1
        size_c = c1 - c0 + 1
        assert size_r == size_c, f"component is not square: {comp['bbox']}"
        assert size_r % 2 == 1, f"square side must be odd: {comp['bbox']}"
        assert comp["area"] == size_r * size_c, f"square must be solid: {comp['bbox']}"
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch == comp["color"])), f"square must be monochromatic: {comp['bbox']}"
        squares.append(
            {
                "id": idx,
                "color": int(comp["color"]),
                "size": int(size_r),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "cr": int((r0 + r1) // 2),
                "cc": int((c0 + c1) // 2),
            }
        )
    return squares


def _edge_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right, f"source is not on the border: {(r, c)}"
    assert (on_top + on_bottom + on_left + on_right) == 1, f"corner source is ambiguous: {(r, c)}"
    if on_left:
        return _DIRS["E"]
    if on_right:
        return _DIRS["W"]
    if on_top:
        return _DIRS["S"]
    return _DIRS["N"]


def _source_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    sources = []
    for r, c in np.argwhere(grid == MAROON):
        rr = int(r)
        cc = int(c)
        assert rr in (0, h - 1) or cc in (0, w - 1), f"maroon source must be on border: {(rr, cc)}"
        assert not ((rr in (0, h - 1)) and (cc in (0, w - 1))), f"corner source is ambiguous: {(rr, cc)}"
        sources.append((rr, cc))
    assert sources, "expected at least one maroon source"
    return sorted(sources)


def _next_square(r: int, c: int, dr: int, dc: int, squares: list[dict]) -> dict | None:
    best = None
    best_key = None
    if dr == 0:
        for sq in squares:
            if not (sq["r0"] <= r <= sq["r1"]):
                continue
            if dc == 1 and sq["c0"] > c:
                key = sq["c0"]
            elif dc == -1 and sq["c1"] < c:
                key = -sq["c1"]
            else:
                continue
            if best is None or key < best_key:
                best = sq
                best_key = key
    else:
        for sq in squares:
            if not (sq["c0"] <= c <= sq["c1"]):
                continue
            if dr == 1 and sq["r0"] > r:
                key = sq["r0"]
            elif dr == -1 and sq["r1"] < r:
                key = -sq["r1"]
            else:
                continue
            if best is None or key < best_key:
                best = sq
                best_key = key
    return best


def _trace_ray(
    grid: np.ndarray,
    squares: list[dict],
    source: tuple[int, int],
    ignore_reflections: frozenset[int] = frozenset(),
) -> dict:
    h, w = grid.shape
    r, c = source
    dr, dc = _edge_direction((h, w), r, c)
    pink_hit_index = 0
    hits = []
    mirrors = []
    while True:
        sq = _next_square(r, c, dr, dc, squares)
        if sq is None:
            if dr == 0:
                endpoint = (r, w - 1 if dc == 1 else 0)
            else:
                endpoint = (h - 1 if dr == 1 else 0, c)
            return {
                "source": (int(source[0]), int(source[1])),
                "endpoint": (int(endpoint[0]), int(endpoint[1])),
                "absorbed": False,
                "axis": "H" if dr == 0 else "V",
                "mirror_count": len(mirrors),
                "hits": hits,
                "mirrors": mirrors,
            }

        hits.append((sq["color"], sq["id"]))
        if sq["color"] == GREY:
            if dr == 0:
                endpoint = (r, sq["c0"] - 1 if dc == 1 else sq["c1"] + 1)
            else:
                endpoint = (sq["r0"] - 1 if dr == 1 else sq["r1"] + 1, c)
            return {
                "source": (int(source[0]), int(source[1])),
                "endpoint": (int(endpoint[0]), int(endpoint[1])),
                "absorbed": True,
                "axis": "H" if dr == 0 else "V",
                "mirror_count": len(mirrors),
                "hits": hits,
                "mirrors": mirrors,
            }

        mirrors.append(sq["id"])
        reflect = pink_hit_index not in ignore_reflections
        if dr == 0:
            new_r = int(2 * sq["cr"] - r) if reflect else int(r)
            r = new_r
            c = sq["c1"] if dc == 1 else sq["c0"]
        else:
            new_c = int(2 * sq["cc"] - c) if reflect else int(c)
            c = new_c
            r = sq["r1"] if dr == 1 else sq["r0"]
        pink_hit_index += 1


def main(grid: np.ndarray) -> np.ndarray:
    """Trace each border maroon light ray through pink reflecting squares and grey absorbers, then mark its terminal cell.

    Input structure:
    - The background is black.
    - The only non-background objects are solid odd-sized squares: pink squares are mirrors and grey squares are absorbers.
    - One or more maroon source pixels lie on the grid border, never in a corner.

    Rule:
    - A source on the left or right border travels horizontally along its row; a source on the top or bottom border travels vertically along its column.
    - Whenever the ray intersects a pink square, its row or column is mirrored about that square's center line, and the ray continues in the same travel direction from the opposite side of the square.
    - Whenever the ray hits a grey square, it stops and the background cell immediately outside the struck face is colored maroon; if no later square intersects the current ray line, the maroon endpoint is placed on the opposite border cell.

    Key insight:
    - The squares are optical devices, not targets: each pink square changes which later squares the ray can intersect, so the full ordered reflection chain must be followed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    squares = _extract_squares(g)
    sources = _source_cells(g)

    out = g.copy()
    for source in sources:
        trace = _trace_ray(g, squares, source)
        rr, cc = trace["endpoint"]
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], f"endpoint out of bounds: {trace}"
        out[rr, cc] = MAROON
    return out


def _can_place_square(placed: list[dict], r0: int, c0: int, size: int, h: int, w: int) -> bool:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    if r0 < 1 or c0 < 1 or r1 > h - 2 or c1 > w - 2:
        return False
    for sq in placed:
        if not (r1 + _MIN_GAP < sq["r0"] or sq["r1"] + _MIN_GAP < r0 or c1 + _MIN_GAP < sq["c0"] or sq["c1"] + _MIN_GAP < c0):
            return False
    return True


def _add_square(placed: list[dict], color: int, center_r: int, center_c: int, size: int, h: int, w: int) -> bool:
    half = size // 2
    r0 = center_r - half
    c0 = center_c - half
    if not _can_place_square(placed, r0, c0, size, h, w):
        return False
    placed.append(
        {
            "color": int(color),
            "size": int(size),
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r0 + size - 1),
            "c1": int(c0 + size - 1),
            "cr": int(center_r),
            "cc": int(center_c),
        }
    )
    return True


def _paint_scene(h: int, w: int, squares: list[dict], sources: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for sq in squares:
        grid[sq["r0"] : sq["r1"] + 1, sq["c0"] : sq["c1"] + 1] = sq["color"]
    for r, c in sources:
        assert grid[r, c] == BLACK
        grid[r, c] = MAROON
    return grid


def _random_symmetry(grid: np.ndarray) -> np.ndarray:
    choice = random.randrange(8)
    if choice == 0:
        return grid.copy()
    if choice == 1:
        return np.rot90(grid, 1).copy()
    if choice == 2:
        return np.rot90(grid, 2).copy()
    if choice == 3:
        return np.rot90(grid, 3).copy()
    if choice == 4:
        return np.fliplr(grid).copy()
    if choice == 5:
        return np.flipud(grid).copy()
    if choice == 6:
        return np.transpose(grid).copy()
    return np.fliplr(np.transpose(grid)).copy()


def _endpoint_signature(trace: dict) -> tuple:
    return (trace["endpoint"], trace["absorbed"], tuple(trace["hits"]))


def _validate_scene(grid: np.ndarray) -> bool:
    squares = _extract_squares(grid)
    sources = _source_cells(grid)
    if len(sources) != 3:
        return False

    traces = [_trace_ray(grid, squares, source) for source in sources]
    endpoints = [trace["endpoint"] for trace in traces]
    if len(set(endpoints)) != len(endpoints):
        return False
    if any(endpoint in sources for endpoint in endpoints):
        return False
    if len({trace["axis"] for trace in traces}) != 2:
        return False
    if not any(trace["absorbed"] and trace["mirror_count"] >= 2 for trace in traces):
        return False
    if not any((not trace["absorbed"]) and trace["mirror_count"] >= 1 for trace in traces):
        return False
    if not any(trace["absorbed"] and trace["mirror_count"] == 0 for trace in traces):
        return False
    if np.array_equal(main(grid), grid):
        return False

    multi = next(trace for trace in traces if trace["absorbed"] and trace["mirror_count"] >= 2)
    mod0 = _trace_ray(grid, squares, multi["source"], frozenset({0}))
    mod1 = _trace_ray(grid, squares, multi["source"], frozenset({1}))
    if _endpoint_signature(mod0) == _endpoint_signature(multi):
        return False
    if _endpoint_signature(mod1) == _endpoint_signature(multi):
        return False

    exit_reflected = next(trace for trace in traces if (not trace["absorbed"]) and trace["mirror_count"] >= 1)
    mod_exit = _trace_ray(grid, squares, exit_reflected["source"], frozenset({0}))
    if _endpoint_signature(mod_exit) == _endpoint_signature(exit_reflected):
        return False

    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(26, 29)
    w = random.randint(25, 29)
    placed: list[dict] = []

    size_a = random.choice([3, 5])
    size_d = random.choice([3, 5])
    size_g = random.choice([3, 5])
    half_a = size_a // 2
    center_a = random.randint(6, 9)
    offset_a = random.choice([d for d in range(-half_a, half_a + 1) if d != 0])
    source_row1 = center_a + offset_a
    row_after_a = center_a - offset_a
    offset_b = 1 if offset_a > 0 else -1
    center_b = row_after_a - offset_b
    row_after_b = row_after_a - 2 * offset_b
    if not (2 <= row_after_b <= 13):
        return None

    if not _add_square(placed, PINK, center_a, 3 + half_a, size_a, h, w):
        return None
    if not _add_square(placed, PINK, center_b, 10, 3, h, w):
        return None
    if not _add_square(placed, GREY, row_after_b, 14, 3, h, w):
        return None

    half_g = size_g // 2
    center_col_g = w - 5
    if not _add_square(placed, GREY, h - 5, center_col_g, size_g, h, w):
        return None

    half_d = size_d // 2
    center_col_d = w - 9
    valid_offsets_d = [d for d in range(1, half_d + 1)]
    random.shuffle(valid_offsets_d)
    source_col2 = None
    for offset_d in valid_offsets_d:
        src_col = center_col_d + offset_d
        new_col = center_col_d - offset_d
        if not (1 <= src_col <= w - 2 and 1 <= new_col <= w - 2):
            continue
        if (center_col_g - half_g) <= new_col <= (center_col_g + half_g):
            continue
        source_col2 = src_col
        break
    if source_col2 is None:
        return None

    center_row_d = random.randint(12, 14)
    if not _add_square(placed, PINK, center_row_d, center_col_d, size_d, h, w):
        return None

    grey_f_center_row = center_row_d + half_d + 4
    if not _add_square(placed, GREY, grey_f_center_row, source_col2, 3, h, w):
        return None

    direct_row3 = (h - 5) + random.choice([d for d in range(-half_g, half_g + 1)])

    for _ in range(random.choice([0, 1])):
        distractor_size = random.choice([3, 5])
        half = distractor_size // 2
        ok = False
        for _ in range(30):
            center_r = random.randint(2 + half, h - 3 - half)
            center_c = random.randint(2 + half, w - 3 - half)
            color = random.choice([GREY, PINK])
            if _add_square(placed, color, center_r, center_c, distractor_size, h, w):
                ok = True
                break
        if not ok:
            return None

    sources = [(source_row1, 0), (0, source_col2), (direct_row3, w - 1)]
    if len(set(sources)) != 3:
        return None

    grid = _paint_scene(h, w, placed, sources)
    grid = _random_symmetry(grid)
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-ray scene")

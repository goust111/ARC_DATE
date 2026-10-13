from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
BLACK = 0

MISSING_TO_COLOR = {
    "TL": 3,
    "TR": 1,
    "BL": 4,
    "BR": 2,
}
COLOR_TO_MISSING = {color: missing for missing, color in MISSING_TO_COLOR.items()}
MISSING_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _band_cells(n: int, k: int) -> list[tuple[int, int]]:
    cells = [(k, c) for c in range(k + 1)]
    cells.extend((r, k) for r in range(k))
    return cells


def _render_from_head(seq: list[int]) -> np.ndarray:
    n = len(seq)
    out = np.zeros((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            out[r, c] = int(seq[max(r, c)])
    return out


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _marker_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _missing_corner(patch: np.ndarray) -> str:
    assert patch.shape == (2, 2)
    grey_mask = patch == GREY
    assert int(grey_mask.sum()) == 3
    missing = np.argwhere(~grey_mask)
    assert missing.shape == (1, 2)
    key = tuple(map(int, missing[0]))
    for name, offset in MISSING_OFFSETS.items():
        if key == offset:
            return name
    raise AssertionError(f"invalid missing corner: {key}")


def _marker_band_index(coords: np.ndarray) -> int:
    return min(max(int(r), int(c)) for r, c in coords)


def _head_from_grid(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    head: list[int] = []
    for k in range(n):
        vals = {int(g[r, c]) for r, c in _band_cells(n, k)}
        vals.discard(BLACK)
        assert len(vals) <= 1, f"band {k} is not monochrome"
        head.append(vals.pop() if vals else BLACK)
    return head


def _least_period(seq: list[int]) -> int:
    n = len(seq)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return n


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode damaged diagonal-band colors from gray L markers, repair the head sequence, and extrapolate a doubled square.

    Input structure:
    - The input is an N x N square on black background using the diagonal-index rule cell(r,c) = head[max(r,c)].
    - The visible head colors are non-black, while a trailing suffix may already be black.
    - One or two full diagonal bands have been damaged to black, and gray 2x2 L-triomino markers sit on nearby black cells.

    Rule:
    - Find every gray marker as a 2x2 box with exactly one missing corner.
    - Decode its repaired band color from the missing corner: top-left -> green(3), top-right -> blue(1), bottom-left -> yellow(4), bottom-right -> red(2).
    - The marker repairs the damaged diagonal band whose index is the minimum max(r,c) among that marker's gray pixels; erase all gray pixels back to black.
    - Read the repaired head sequence from the diagonal bands, trim the trailing black suffix, repeat the valid head cyclically to length 2L, and fill a 2L x 2L square again by cell(r,c) = cycle[max(r,c)].

    Key insight:
    - The gray L markers are not decoration: their missing-corner orientation restores hidden entries of the one-dimensional head sequence, and only the repaired sequence can be extrapolated correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    markers = _find_color_components(g, GREY)
    assert 1 <= len(markers) <= 2, "expected one or two L markers"

    repaired = g.copy()
    repairs: dict[int, int] = {}
    for coords in markers:
        assert coords.shape == (3, 2), "marker must be a triomino"
        r0, c0, r1, c1 = _marker_bbox(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "marker bbox must be 2x2"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        missing = _missing_corner(patch)
        color = int(MISSING_TO_COLOR[missing])
        k = _marker_band_index(coords)
        assert k not in repairs, "markers must repair distinct bands"
        repairs[k] = color
        repaired[coords[:, 0], coords[:, 1]] = BLACK

    head = _head_from_grid(repaired)
    for k, color in repairs.items():
        assert head[k] == BLACK, f"marker band {k} must be fully damaged before repair"
        head[k] = color

    nonzero = [i for i, color in enumerate(head) if color != BLACK]
    assert nonzero, "expected a non-empty head sequence"
    L = max(nonzero) + 1
    assert all(color != BLACK for color in head[:L]), "valid head prefix must be fully repaired"
    assert all(color == BLACK for color in head[L:]), "only a black suffix may remain"

    cycle = [int(head[i % L]) for i in range(2 * L)]
    return _render_from_head(cycle)


def _marker_patch(missing: str) -> np.ndarray:
    patch = np.full((2, 2), GREY, dtype=int)
    mr, mc = MISSING_OFFSETS[missing]
    patch[mr, mc] = BLACK
    return patch


def _marker_cells(r0: int, c0: int, missing: str) -> list[tuple[int, int]]:
    mr, mc = MISSING_OFFSETS[missing]
    cells: list[tuple[int, int]] = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != (mr, mc):
                cells.append((r0 + dr, c0 + dc))
    return cells


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return True
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _enumerate_marker_positions(
    grid: np.ndarray,
    k: int,
    missing: str,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    n = grid.shape[0]
    out: list[tuple[int, int]] = []
    for r0 in range(n - 1):
        for c0 in range(n - 1):
            cells = _marker_cells(r0, c0, missing)
            if _touches_existing(cells, occupied):
                continue
            if any(int(grid[r, c]) != BLACK for r, c in cells):
                continue
            if min(max(r, c) for r, c in cells) != k:
                continue
            out.append((r0, c0))
    random.shuffle(out)
    return out


def _place_markers(
    grid: np.ndarray,
    damaged: list[int],
    colors: list[int],
) -> bool:
    occupied: set[tuple[int, int]] = set()
    order = list(range(len(damaged)))
    random.shuffle(order)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        k = damaged[idx]
        missing = COLOR_TO_MISSING[int(colors[idx])]
        for r0, c0 in _enumerate_marker_positions(grid, k, missing, occupied):
            patch = _marker_patch(missing)
            cells = _marker_cells(r0, c0, missing)
            grid[r0 : r0 + 2, c0 : c0 + 2] = np.maximum(grid[r0 : r0 + 2, c0 : c0 + 2], patch)
            occupied.update(cells)
            if dfs(pos + 1):
                return True
            for r, c in cells:
                occupied.remove((r, c))
                grid[r, c] = BLACK
        return False

    return dfs(0)


def _validate_instance(grid: np.ndarray, head: list[int], damaged: list[int]) -> bool:
    n = len(head)
    full_head = head[:]
    expected = _render_from_head([full_head[i % len(full_head)] for i in range(2 * len(full_head))])
    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    markers = _find_color_components(grid, GREY)
    if len(markers) != len(damaged):
        return False
    if sorted(_marker_band_index(coords) for coords in markers) != sorted(damaged):
        return False

    erased = grid.copy()
    erased[erased == GREY] = BLACK
    visible = _head_from_grid(erased)
    if any(visible[k] != BLACK for k in damaged):
        return False
    if visible[-1] != BLACK and (len(visible) not in damaged):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        n = random.randint(6, 8)
        suffix_len = random.randint(1, min(2, n - 4))
        L = n - suffix_len
        num_damaged = random.choice([1, 2]) if L >= 4 else 1
        damaged = list(range(L - num_damaged, L))

        distinct = random.randint(2, min(4, L))
        palette = random.sample([1, 2, 3, 4], k=distinct)
        head = palette[:]
        head.extend(random.choice(palette) for _ in range(L - distinct))
        random.shuffle(head)
        if _least_period(head) != L:
            continue

        seq = head + [BLACK] * suffix_len
        grid = _render_from_head(seq)
        for k in damaged:
            for r, c in _band_cells(n, k):
                grid[r, c] = BLACK

        if not _place_markers(grid, damaged, [head[k] for k in damaged]):
            continue

        if not _validate_instance(grid, head, damaged):
            continue
        return grid

    raise RuntimeError("generate_input: failed to build a valid puzzle instance")

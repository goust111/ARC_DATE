from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _boundary_values(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])


def _find_layout(grid: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int, int, int]]]:
    bg = int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])
    rects: list[tuple[int, int, np.ndarray]] = []
    library_candidates: list[tuple[int, int, np.ndarray]] = []

    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        h, w = patch.shape
        if int(h * w) != len(comp) or h != w:
            continue

        if h >= 6 and len(np.unique(_boundary_values(patch))) == 1:
            inner = patch[1:-1, 1:-1]
            if inner.size and not bool(np.any(inner == bg)) and len(np.unique(inner)) >= 3:
                library_candidates.append((int(c0), int(r0), patch))
                continue

        rects.append((int(r0), int(c0), patch))

    assert library_candidates, "library block not found"
    _, _, framed = min(library_candidates, key=lambda item: (item[0], item[1], -item[2].shape[0]))
    library = framed[1:-1, 1:-1]
    n = int(library.shape[0])

    boards: list[tuple[int, int, int, int]] = []
    for r0, c0, patch in rects:
        if patch.shape != (n, n):
            continue
        colors, counts = np.unique(patch, return_counts=True)
        if len(colors) != 2:
            continue
        counts_list = sorted(int(x) for x in counts.tolist())
        if counts_list != [1, n * n - 1]:
            continue
        marker_color = int(colors[int(np.where(counts == 1)[0][0])])
        marker_locs = np.argwhere(patch == marker_color)
        if marker_locs.shape[0] != 1:
            continue
        mr, mc = map(int, marker_locs[0])
        boards.append((int(r0), int(c0), int(mr), int(mc)))

    assert boards, "no checkerboard blocks found"
    return bg, library, boards


def _cross_from_library(library: np.ndarray, r: int, c: int, bg: int) -> np.ndarray:
    n = int(library.shape[0])
    patch = np.full((n, n), int(bg), dtype=int)
    patch[int(r), :] = library[int(r), :]
    patch[:, int(c)] = library[:, int(c)]
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marker-bearing square with the row/column cross selected from the shared library.
    """
    g = np.asarray(grid, dtype=int)
    bg, library, boards = _find_layout(g)
    n = int(library.shape[0])

    out = g.copy()
    for r0, c0, mr, mc in boards:
        out[int(r0) : int(r0) + n, int(c0) : int(c0) + n] = _cross_from_library(library, mr, mc, bg)
    return out


BG = 0
LIBRARY_BORDER = 8
BOARD_FILL = 5
MARKER = 9
LIBRARY_PALETTE = (1, 2, 3, 4, 6, 7)
MAX_GENERATION_ATTEMPTS = 300


def _sample_library(n: int) -> np.ndarray:
    for _ in range(100):
        palette = random.sample(list(LIBRARY_PALETTE), random.randint(3, len(LIBRARY_PALETTE)))
        lib = np.array([[random.choice(palette) for _ in range(n)] for _ in range(n)], dtype=int)
        if len(np.unique(lib)) < 3:
            continue
        if min(len(np.unique(lib[r, :])) for r in range(n)) < 2:
            continue
        if min(len(np.unique(lib[:, c])) for c in range(n)) < 2:
            continue
        return lib
    raise RuntimeError("failed to sample a non-degenerate library")


def _build_expected(
    inp: np.ndarray,
    library: np.ndarray,
    boards: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r0, c0, mr, mc in boards:
        n = int(library.shape[0])
        out[int(r0) : int(r0) + n, int(c0) : int(c0) + n] = _cross_from_library(library, mr, mc, BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(4, 7)
    library = _sample_library(n)

    n_boards = random.randint(2, min(6, n * n))
    marker_positions = random.sample([(r, c) for r in range(n) for c in range(n)], n_boards)

    board_rows = random.randint(max(1, math.ceil(n_boards / 3)), min(3, n_boards))
    board_cols = int(math.ceil(n_boards / board_rows))
    v_gap = random.randint(2, 4)
    h_gap = random.randint(2, 4)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 4)
    region_gap = random.randint(3, 6)

    library_side = n + 2
    board_h = board_rows * n + (board_rows - 1) * v_gap
    board_w = board_cols * n + (board_cols - 1) * h_gap
    h = max(library_side, board_h) + top_margin + bottom_margin + random.randint(0, 4)
    w = left_margin + library_side + region_gap + board_w + right_margin + random.randint(0, 6)

    lib_r0 = random.randint(top_margin, h - bottom_margin - library_side)
    lib_c0 = left_margin
    board_top = random.randint(top_margin, h - bottom_margin - board_h)
    board_left = random.randint(lib_c0 + library_side + region_gap, w - right_margin - board_w)

    inp = np.full((h, w), BG, dtype=int)
    inp[int(lib_r0) : int(lib_r0) + library_side, int(lib_c0) : int(lib_c0) + library_side] = LIBRARY_BORDER
    inp[int(lib_r0) + 1 : int(lib_r0) + 1 + n, int(lib_c0) + 1 : int(lib_c0) + 1 + n] = library

    slots = [(rr, cc) for rr in range(board_rows) for cc in range(board_cols)]
    chosen_slots = random.sample(slots, n_boards)
    boards: list[tuple[int, int, int, int]] = []
    for (slot_r, slot_c), (mr, mc) in zip(chosen_slots, marker_positions):
        r0 = int(board_top + slot_r * (n + v_gap))
        c0 = int(board_left + slot_c * (n + h_gap))
        inp[int(r0) : int(r0) + n, int(c0) : int(c0) + n] = BOARD_FILL
        inp[int(r0) + int(mr), int(c0) + int(mc)] = MARKER
        boards.append((int(r0), int(c0), int(mr), int(mc)))

    expected = _build_expected(inp, library, boards)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")

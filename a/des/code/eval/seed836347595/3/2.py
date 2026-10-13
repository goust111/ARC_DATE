from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid, dtype=int).ravel()).argmax())


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _grey_rect(g: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == GREY)
    assert pts.size, "expected a grey occluder"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey occluder must be solid"
    return r0, c0, r1, c1


def _piece_arrays(g: np.ndarray, bg: int, rect: tuple[int, int, int, int]) -> list[tuple[np.ndarray, list[tuple[int, int]]]]:
    r0, c0, r1, c1 = rect
    foreground = (g != bg) & (g != GREY)
    pieces: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
    for comp in _components(foreground):
        touches_context = any(r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 for r, c in comp)
        if touches_context:
            continue
        br0, bc0, br1, bc1 = _bbox(comp)
        arr = np.full((br1 - br0 + 1, bc1 - bc0 + 1), bg, dtype=int)
        for r, c in comp:
            arr[r - br0, c - bc0] = int(g[r, c])
        pieces.append((arr, comp))
    assert pieces, "expected detached edge pieces"
    return pieces


def _placements(piece: np.ndarray, target_shape: tuple[int, int], bg: int) -> list[tuple[int, int, np.ndarray]]:
    ph, pw = piece.shape
    th, tw = target_shape
    mask = piece != bg
    assert mask.any()
    out: list[tuple[int, int, np.ndarray]] = []
    for r in range(th - ph + 1):
        for c in range(tw - pw + 1):
            out.append((r, c, mask))
    return out


def _pack_pieces(pieces: list[np.ndarray], target_shape: tuple[int, int], bg: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(pieces)), key=lambda i: int(np.count_nonzero(pieces[i] != bg)), reverse=True)
    placement_options = {i: _placements(pieces[i], target_shape, bg) for i in order}
    filled = np.zeros(target_shape, dtype=bool)
    chosen: dict[int, tuple[int, int]] = {}
    total_area = sum(int(np.count_nonzero(p != bg)) for p in pieces)
    if total_area != target_shape[0] * target_shape[1]:
        return None

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return bool(filled.all())
        best_i = None
        best_fits: list[tuple[int, int, np.ndarray]] | None = None
        for i in order:
            if i in chosen:
                continue
            fits = []
            for r, c, mask in placement_options[i]:
                region = filled[r : r + mask.shape[0], c : c + mask.shape[1]]
                if not np.any(region & mask):
                    fits.append((r, c, mask))
            if best_fits is None or len(fits) < len(best_fits):
                best_i = i
                best_fits = fits
        if best_i is None or not best_fits:
            return False
        for r, c, mask in best_fits:
            filled[r : r + mask.shape[0], c : c + mask.shape[1]][mask] = True
            chosen[best_i] = (r, c)
            if backtrack(pos + 1):
                return True
            del chosen[best_i]
            filled[r : r + mask.shape[0], c : c + mask.shape[1]][mask] = False
        return False

    if not backtrack(0):
        return None
    return [chosen[i] for i in range(len(pieces))]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    rect = _grey_rect(g)
    r0, c0, r1, c1 = rect
    target_shape = (r1 - r0 + 1, c1 - c0 + 1)
    piece_records = _piece_arrays(g, bg, rect)
    pieces = [record[0] for record in piece_records]
    positions = _pack_pieces(pieces, target_shape, bg)
    assert positions is not None, "pieces must tile the grey occluder"

    out = g.copy()
    for _piece, comp in piece_records:
        for r, c in comp:
            out[r, c] = bg
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for piece, (pr, pc) in zip(pieces, positions):
        mask = piece != bg
        region = out[r0 + pr : r0 + pr + piece.shape[0], c0 + pc : c0 + pc + piece.shape[1]]
        region[mask] = piece[mask]
    return out


def _make_partition(h: int, w: int, n: int) -> list[np.ndarray]:
    bases = sorted(random.sample(range(2, w - 1), n - 1))
    cuts = []
    for r in range(h):
        row = []
        last = 0
        for base in bases:
            jitter = random.choice((-1, 0, 1))
            val = min(w - 1, max(last + 1, base + jitter))
            row.append(val)
            last = val
        cuts.append(row)
    labels = np.zeros((h, w), dtype=int)
    for r in range(h):
        bounds = [0] + cuts[r] + [w]
        for i in range(n):
            labels[r, bounds[i] : bounds[i + 1]] = i
    pieces = []
    for i in range(n):
        mask = labels == i
        pts = np.argwhere(mask)
        br0, bc0 = pts.min(axis=0)
        br1, bc1 = pts.max(axis=0)
        pieces.append(mask[int(br0) : int(br1) + 1, int(bc0) : int(bc1) + 1])
    return pieces


def _place_piece(grid: np.ndarray, piece: np.ndarray, top: int, left: int, colors: np.ndarray) -> None:
    h, w = piece.shape
    region = grid[top : top + h, left : left + w]
    region[piece] = colors[piece]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 8)
    w = random.randint(6, 10)
    n = random.randint(3, min(5, w // 2 + 1))
    piece_masks = _make_partition(h, w, n)
    hidden = np.array([[random.choice(COLORS) for _ in range(w)] for _ in range(h)], dtype=int)

    canvas_h = h + 18
    canvas_w = w + 18
    r0 = random.randint(7, canvas_h - h - 7)
    c0 = random.randint(7, canvas_w - w - 7)
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    grid[r0 : r0 + h, c0 : c0 + w] = GREY

    for c in range(w):
        grid[r0 - 1, c0 + c] = hidden[0, c]
        grid[r0 + h, c0 + c] = hidden[-1, c]
    for r in range(h):
        grid[r0 + r, c0 - 1] = hidden[r, 0]
        grid[r0 + r, c0 + w] = hidden[r, -1]

    occupied = grid != BG
    placements = [
        (1, 1),
        (1, canvas_w // 2),
        (canvas_h - 9, 1),
        (canvas_h - 9, canvas_w // 2),
        (canvas_h // 2, 1),
    ]
    for i, mask in enumerate(piece_masks):
        pts = np.argwhere(mask)
        br0, bc0 = pts.min(axis=0)
        colors = hidden[int(br0) : int(br0) + mask.shape[0], int(bc0) : int(bc0) + mask.shape[1]]
        top, left = placements[i]
        if top + mask.shape[0] >= canvas_h or left + mask.shape[1] >= canvas_w:
            return None
        pad = occupied[max(0, top - 1) : min(canvas_h, top + mask.shape[0] + 1), max(0, left - 1) : min(canvas_w, left + mask.shape[1] + 1)]
        if pad.any():
            return None
        _place_piece(grid, mask, top, left, colors)
        occupied = grid != BG

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

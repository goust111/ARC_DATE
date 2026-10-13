from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
SHAPE_COLORS = (BLUE, RED)
GEN_ATTEMPTS = 400


def _mirror_swap(block: np.ndarray) -> np.ndarray:
    mirrored = np.fliplr(block)
    swapped = mirrored.copy()
    swapped[mirrored == BLUE] = RED
    swapped[mirrored == RED] = BLUE
    return swapped


def _full_grey_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [int(r) for r in np.where(np.all(grid == GREY, axis=1))[0]]
    cols = [int(c) for c in np.where(np.all(grid == GREY, axis=0))[0]]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one full grey row and one full grey column"
    return rows[0], cols[0]


def _score_axis(block: np.ndarray, axis_k: int) -> int:
    h, w = block.shape
    score = 0
    for color in SHAPE_COLORS:
        for r, c in np.argwhere(block == color):
            cc = axis_k - int(c)
            if 0 <= cc < w and int(block[int(r), cc]) == color:
                score += 1
    return score


def _best_axis(block: np.ndarray) -> tuple[int, int]:
    h, w = block.shape
    del h
    scores = [(k, _score_axis(block, k)) for k in range(2 * (w - 1) + 1)]
    scores.sort(key=lambda item: (-item[1], item[0]))
    best_k, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1
    return int(best_k), int(best_score - second_score)


def _symmetrize(block: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = block.shape
    axis_k, _gap = _best_axis(block)
    out = np.zeros_like(block)
    total_counts = {color: int(np.count_nonzero(block == color)) for color in SHAPE_COLORS}
    for r in range(h):
        seen: set[tuple[int, int]] = set()
        for c in range(w):
            cc = axis_k - c
            if cc < 0 or cc >= w:
                continue
            pair = tuple(sorted((int(c), int(cc))))
            if pair in seen:
                continue
            seen.add(pair)
            counts = {color: 0 for color in SHAPE_COLORS}
            for col in pair:
                color = int(block[r, col])
                if color in counts:
                    counts[color] += 1
            present = [color for color, count in counts.items() if count > 0]
            if not present:
                continue
            if len(present) == 1:
                chosen = present[0]
            else:
                chosen = max(present, key=lambda color: (counts[color], total_counts[color], -color))
            for col in pair:
                out[r, col] = chosen
    return out, axis_k


def _bounded_zero_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                components.append(cells)
    return components


def _fill_bounded_regions(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for cells in _bounded_zero_components(grid):
        for r, c in cells:
            out[r, c] = YELLOW
    return out


def _tight_mask_signature(block: np.ndarray) -> tuple[tuple[int, ...], ...]:
    mask = block != BG
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1].astype(int)
    return tuple(tuple(int(v) for v in row) for row in crop)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise the two diagonal two-color motifs by shared vertical symmetry, mirror-swap them into the empty quadrants, then flood-fill the newly enclosed background.

    Input structure:
    - A black canvas is split by one full grey middle row and one full grey middle column into four quadrants.
    - The upper-left and lower-right quadrants contain noisy red/blue motifs; the other two quadrants are black.
    - Each noisy motif is meant to be vertically symmetric within its own quadrant, with a few asymmetric extra or missing pixels.

    Rule:
    - In each non-empty quadrant, find the vertical axis that best fits both color layers together, remove unpaired pixels, and complete missing same-color mirror partners.
    - Copy the cleaned upper-left motif into the upper-right by horizontal reflection across the grey column and swap red with blue.
    - Copy the cleaned lower-right motif into the lower-left the same way, again swapping red and blue.
    - Keep the grey cross unchanged, then fill every 4-connected black region that is not 4-connected to the outer border with yellow.

    Key insight:
    - The puzzle is not plain mirroring: the diagonal motifs must first be restored by a shared symmetry axis, and only the restored motifs determine the mirrored color-swapped copies and the bounded regions that get filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mid_r, mid_c = _full_grey_cross(g)
    tl = g[:mid_r, :mid_c]
    tr = g[:mid_r, mid_c + 1 :]
    bl = g[mid_r + 1 :, :mid_c]
    br = g[mid_r + 1 :, mid_c + 1 :]

    assert np.all(tr == BG) and np.all(bl == BG), "expected the off-diagonal quadrants to be empty"

    clean_tl, _ = _symmetrize(tl)
    clean_br, _ = _symmetrize(br)

    out = np.zeros_like(g)
    out[mid_r, :] = GREY
    out[:, mid_c] = GREY
    out[:mid_r, :mid_c] = clean_tl
    out[mid_r + 1 :, mid_c + 1 :] = clean_br
    out[:mid_r, mid_c + 1 :] = _mirror_swap(clean_tl)
    out[mid_r + 1 :, :mid_c] = _mirror_swap(clean_br)
    return _fill_bounded_regions(out)


def _make_clean_mask(kind: str, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if kind == "frame":
        return mask
    if kind == "double_v":
        cols = [w // 2] if w % 2 == 1 else [w // 2 - 1, w // 2]
        for col in cols:
            mask[1:-1, col] = True
        return mask
    if kind == "double_h":
        row = h // 2
        mask[row, 1:-1] = True
        return mask
    raise ValueError(f"unknown kind: {kind}")


def _colorize(mask: np.ndarray, kind: str) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=int)
    scheme = random.choice(("row_split", "row_parity", "bar_vs_ring"))
    if scheme == "row_split":
        split = random.randint(1, h - 2)
        primary, secondary = random.sample(SHAPE_COLORS, 2)
        for r in range(h):
            color = primary if r <= split else secondary
            out[r, mask[r]] = color
    elif scheme == "row_parity":
        primary, secondary = random.sample(SHAPE_COLORS, 2)
        offset = random.randint(0, 1)
        for r in range(h):
            color = primary if (r + offset) % 2 == 0 else secondary
            out[r, mask[r]] = color
    else:
        ring_color, bar_color = random.sample(SHAPE_COLORS, 2)
        out[mask] = ring_color
        if kind == "double_v":
            cols = [w // 2] if w % 2 == 1 else [w // 2 - 1, w // 2]
            for col in cols:
                out[1:-1, col] = bar_color
        elif kind == "double_h":
            out[h // 2, 1:-1] = bar_color
        else:
            split = h // 2
            out[: split + 1][mask[: split + 1]] = ring_color
            out[split + 1 :][mask[split + 1 :]] = bar_color
    assert set(np.unique(out)) >= {BG, BLUE, RED}
    return out


def _embed_local(local: np.ndarray, qh: int, qw: int, top: int, left: int) -> np.ndarray:
    quad = np.zeros((qh, qw), dtype=int)
    h, w = local.shape
    quad[top : top + h, left : left + w] = local
    return quad


def _local_bounded_area(block: np.ndarray) -> int:
    return sum(len(comp) for comp in _bounded_zero_components(block))


def _true_axis_k(left: int, width: int) -> int:
    return 2 * left + width - 1


def _make_noisy(clean_quad: np.ndarray, axis_k: int) -> np.ndarray | None:
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(clean_quad != BG)]
    orbit_reps: dict[tuple[int, int, int], tuple[int, int]] = {}
    for r, c in occupied:
        cc = axis_k - c
        if cc == c or cc < 0 or cc >= clean_quad.shape[1]:
            continue
        rep = (int(r), min(int(c), int(cc)), max(int(c), int(cc)))
        orbit_reps.setdefault(rep, (int(r), int(c)))
    candidates = list(orbit_reps.values())
    random.shuffle(candidates)
    desired = random.randint(2, 3)
    if len(candidates) < desired:
        return None

    for _ in range(60):
        deleted = random.sample(candidates, desired)
        noisy = clean_quad.copy()
        for r, c in deleted:
            noisy[r, c] = BG
        if _local_bounded_area(noisy) != 0:
            continue
        cleaned, solved_axis = _symmetrize(noisy)
        if solved_axis != axis_k or not np.array_equal(cleaned, clean_quad):
            continue
        return noisy
    return None


def _make_quadrant(qh: int, qw: int) -> tuple[np.ndarray, np.ndarray]:
    kinds = ("frame", "double_v", "double_h")
    for _ in range(200):
        kind = random.choice(kinds)
        if kind == "frame":
            w = random.randint(5, qw - 2)
            h = random.randint(5, qh - 2)
        elif kind == "double_v":
            parity = random.choice((0, 1))
            w_min = 8 if parity == 0 else 7
            if qw - 2 < w_min:
                continue
            possible = [ww for ww in range(w_min, qw - 1) if ww % 2 == parity]
            if not possible:
                continue
            w = random.choice(possible)
            h = random.randint(5, qh - 2)
        else:
            if qh - 2 < 7:
                continue
            w = random.randint(5, qw - 2)
            h = random.randint(7, qh - 2)
        top = random.randint(1, qh - h - 1)
        left = random.randint(1, qw - w - 1)
        local_mask = _make_clean_mask(kind, h, w)
        local_color = _colorize(local_mask, kind)
        clean_quad = _embed_local(local_color, qh, qw, top, left)
        if _local_bounded_area(clean_quad) < 4:
            continue
        axis_k = _true_axis_k(left, w)
        noisy_quad = _make_noisy(clean_quad, axis_k)
        if noisy_quad is None:
            continue
        if _local_bounded_area(noisy_quad) != 0:
            continue
        best_k, gap = _best_axis(noisy_quad)
        if best_k != axis_k or gap < 2:
            continue
        cleaned, solved_axis = _symmetrize(noisy_quad)
        if solved_axis != axis_k or not np.array_equal(cleaned, clean_quad):
            continue
        diff = int(np.count_nonzero(noisy_quad != clean_quad))
        if diff not in (2, 3):
            continue
        if not np.any((clean_quad != BG) & (noisy_quad == BG)):
            continue
        return noisy_quad, clean_quad
    raise RuntimeError("failed to build a valid quadrant")


def _validate_generated(inp: np.ndarray, out: np.ndarray) -> None:
    mid_r, mid_c = _full_grey_cross(inp)
    tl_in = inp[:mid_r, :mid_c]
    br_in = inp[mid_r + 1 :, mid_c + 1 :]
    assert np.all(inp[:mid_r, mid_c + 1 :] == BG)
    assert np.all(inp[mid_r + 1 :, :mid_c] == BG)
    assert _local_bounded_area(inp) == 0

    tl_clean, _ = _symmetrize(tl_in)
    br_clean, _ = _symmetrize(br_in)
    prefill = np.zeros_like(inp)
    prefill[mid_r, :] = GREY
    prefill[:, mid_c] = GREY
    prefill[:mid_r, :mid_c] = tl_clean
    prefill[mid_r + 1 :, mid_c + 1 :] = br_clean
    prefill[:mid_r, mid_c + 1 :] = _mirror_swap(tl_clean)
    prefill[mid_r + 1 :, :mid_c] = _mirror_swap(br_clean)
    assert _tight_mask_signature(tl_clean) != _tight_mask_signature(br_clean)
    assert np.array_equal(np.where(out == YELLOW, BG, out), prefill)
    expected = _fill_bounded_regions(prefill)
    assert np.array_equal(out, expected)

    yellow = np.argwhere(out == YELLOW)
    assert len(yellow) >= 4
    assert any(c > mid_c for r, c in yellow if r < mid_r) or any(c < mid_c for r, c in yellow if r > mid_r)
    assert any(len(comp) >= 4 for comp in _bounded_zero_components(prefill))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        qh = random.randint(8, 12)
        qw = random.randint(8, 12)
        tl_noisy, tl_clean = _make_quadrant(qh, qw)
        br_noisy, br_clean = _make_quadrant(qh, qw)
        if _tight_mask_signature(tl_clean) == _tight_mask_signature(br_clean):
            continue

        h = 2 * qh + 1
        w = 2 * qw + 1
        grid = np.zeros((h, w), dtype=int)
        grid[qh, :] = GREY
        grid[:, qw] = GREY
        grid[:qh, :qw] = tl_noisy
        grid[qh + 1 :, qw + 1 :] = br_noisy

        out = main(grid)
        _validate_generated(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid input")

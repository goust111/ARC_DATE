from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
PERIOD_CHOICES = (3, 4)
GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
FG_CHOICES = tuple(range(1, 10))
NOISE_LENGTHS = (2, 3, 4)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _fg_color(grid: np.ndarray) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != 0]
    assert colors, "expected at least one foreground color"
    assert len(colors) == 1, "expected a single foreground color"
    return colors[0]


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    out = 0
    if r > 0 and mask[r - 1, c]:
        out += 1
    if r + 1 < h and mask[r + 1, c]:
        out += 1
    if c > 0 and mask[r, c - 1]:
        out += 1
    if c + 1 < w and mask[r, c + 1]:
        out += 1
    return out


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in DIAGONALS:
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
            return True
    return False


def _suspect_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if mask[r, c] and _deg4(mask, r, c) == 0 and _has_diag(mask, r, c):
                out[r, c] = True
    return out


def _period_score(mask: np.ndarray, p: int) -> tuple[float, int]:
    total = int(mask.sum())
    if total == 0:
        return (-1.0, -1)
    overlap = int(np.sum(mask[p:, :] & mask[:-p, :]))
    overlap += int(np.sum(mask[:, p:] & mask[:, :-p]))
    residue_counts = np.zeros((p, p), dtype=int)
    for r, c in np.argwhere(mask):
        residue_counts[r % p, c % p] += 1
    witnesses = int(np.min(residue_counts[residue_counts > 0])) if np.any(residue_counts > 0) else 0
    return (overlap / total, witnesses)


def _infer_period(mask: np.ndarray) -> int:
    scored = []
    for p in PERIOD_CHOICES:
        score, witnesses = _period_score(mask, p)
        scored.append((score, witnesses, -p, p))
    scored.sort(reverse=True)
    best = scored[0]
    return int(best[-1])


def _infer_tile(mask: np.ndarray, p: int) -> np.ndarray:
    tile = np.zeros((p, p), dtype=bool)
    for r, c in np.argwhere(mask):
        tile[r % p, c % p] = True
    return tile


def _rescues_from_period(
    full_mask: np.ndarray, suspects: np.ndarray, tile: np.ndarray, p: int, r: int, c: int
) -> bool:
    if not tile[r % p, c % p]:
        return False
    for dr, dc in DIAGONALS:
        rr, cc = r + dr, c + dc
        if not _in_bounds(full_mask.shape, rr, cc) or not full_mask[rr, cc]:
            continue
        if not tile[rr % p, cc % p]:
            continue
        br1 = (r, cc)
        br2 = (rr, c)
        if not (_in_bounds(full_mask.shape, br1[0], br1[1]) and _in_bounds(full_mask.shape, br2[0], br2[1])):
            continue
        if full_mask[br1] or full_mask[br2]:
            continue
        if tile[br1[0] % p, br1[1] % p] and tile[br2[0] % p, br2[1] % p]:
            return True
    return False


def _periodic_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    p = tile.shape[0]
    rows = np.arange(shape[0])[:, None] % p
    cols = np.arange(shape[1])[None, :] % p
    return tile[rows, cols]


def _period_solution(full_mask: np.ndarray, p: int) -> tuple[tuple[float, int, int, int], np.ndarray, np.ndarray]:
    suspects = _suspect_mask(full_mask)
    base = full_mask & (~suspects)
    tile = _infer_tile(base, p)
    keep = np.zeros_like(full_mask, dtype=bool)
    for _ in range(2):
        keep[:, :] = False
        for r, c in np.argwhere(suspects):
            keep[r, c] = _rescues_from_period(full_mask, suspects, tile, p, int(r), int(c))
        tile = _infer_tile(base | keep, p)
    support = base | keep
    overlap, witnesses = _period_score(support, p)
    score = (overlap, int(keep.sum()), witnesses, -p)
    return score, tile, keep


def _has_period(mask: np.ndarray, p: int) -> bool:
    return bool(np.array_equal(mask[p:, :], mask[:-p, :]) and np.array_equal(mask[:, p:], mask[:, :-p]))


def _minimal_period(mask: np.ndarray) -> int:
    lim = min(mask.shape)
    for p in range(1, lim + 1):
        if _has_period(mask, p):
            return p
    return lim


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove diagonal-only noise and restore the smallest repeating foreground tile.

    Input structure:
    - A black background contains one foreground color forming a square-periodic pattern.
    - Some true pattern cells are erased to black, and some extra foreground pixels form 45-degree diagonal chains.
    - Diagonal-chain pixels have no orthogonal foreground neighbors, but they may share a residue class with the real period.

    Rule:
    - Mark every foreground pixel with zero orthogonal neighbors and at least one diagonal foreground neighbor as a suspect.
    - Infer the period from the remaining non-suspect foreground samples, then project those samples onto the period tile.
    - Re-keep a suspect only when periodic consistency shows it is one corner of a locally missing periodic motif rather than free diagonal noise.
    - Delete the confirmed noise and fill every cell whose period-tile residue should be foreground.

    Key insight:
    Diagonal isolation alone is not enough: true pattern cells can look like diagonal noise after erasure, so the solution must combine local diagonal structure with global period inference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    fg = _fg_color(g)
    full_mask = g == fg

    scored = []
    for p in PERIOD_CHOICES:
        score, tile, keep = _period_solution(full_mask, p)
        scored.append((score, p, tile, keep))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    _score, _p, tile, _keep = scored[0]

    filled_mask = _periodic_fill(g.shape, tile)
    out = np.zeros_like(g)
    out[filled_mask] = fg
    return out


def _build_tile(p: int) -> np.ndarray:
    tile = np.zeros((p, p), dtype=bool)
    if p == 3:
        block_h, block_w = 2, 2
    else:
        block_h, block_w = random.choice(((2, 2), (2, 3), (3, 2)))
    r0 = random.randint(0, p - block_h)
    c0 = random.randint(0, p - block_w)
    tile[r0 : r0 + block_h, c0 : c0 + block_w] = True
    return tile


def _render_output(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    return _periodic_fill(shape, tile)


def _pick_rescue_boxes(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    candidates = []
    for r in range(mask.shape[0] - 1):
        for c in range(mask.shape[1] - 1):
            if bool(np.all(mask[r : r + 2, c : c + 2])):
                candidates.append((r, c))
    random.shuffle(candidates)
    chosen = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                break
    return chosen


def _apply_rescues(inp_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = inp_mask.copy()
    rescue_boxes = _pick_rescue_boxes(out, random.choice((1, 1, 2)))
    protected = []
    for r, c in rescue_boxes:
        out[r, c + 1] = False
        out[r + 1, c] = False
        protected.extend([(r, c), (r + 1, c + 1)])
    return out, protected


def _extra_holes(inp_mask: np.ndarray, protected: set[tuple[int, int]], tile: np.ndarray, p: int) -> np.ndarray:
    out = inp_mask.copy()
    fg_cells = [(int(r), int(c)) for r, c in np.argwhere(out)]
    random.shuffle(fg_cells)
    max_holes = random.randint(2, 6)
    residue_counts = np.zeros(tile.shape, dtype=int)
    suspects = _suspect_mask(out)
    base = out & (~suspects)
    for r, c in np.argwhere(base):
        residue_counts[r % p, c % p] += 1
    removed = 0
    for r, c in fg_cells:
        if removed >= max_holes:
            break
        if (r, c) in protected:
            continue
        rr, cc = r % p, c % p
        if residue_counts[rr, cc] <= 2:
            continue
        out[r, c] = False
        residue_counts[rr, cc] -= 1
        removed += 1
    return out


def _chain_cells(r: int, c: int, length: int, dc_sign: int) -> list[tuple[int, int]]:
    return [(r + i, c + dc_sign * i) for i in range(length)]


def _valid_noise_chain(
    mask: np.ndarray, tile: np.ndarray, p: int, cells: list[tuple[int, int]], require_legal: bool
) -> bool:
    if any(not _in_bounds(mask.shape, r, c) for r, c in cells):
        return False
    if any(mask[r, c] for r, c in cells):
        return False

    legal_count = sum(1 for r, c in cells if tile[r % p, c % p])
    if require_legal and legal_count == 0:
        return False

    temp = mask.copy()
    for r, c in cells:
        temp[r, c] = True

    suspects = _suspect_mask(temp)
    for r, c in cells:
        if not suspects[r, c]:
            return False
        if _rescues_from_period(temp, suspects, tile, p, r, c):
            return False
    return True


def _add_noise(inp_mask: np.ndarray, tile: np.ndarray, p: int) -> np.ndarray | None:
    out = inp_mask.copy()
    needs = [False, random.choice((False, True))]
    for require_legal in needs:
        placed = False
        for length in random.sample(NOISE_LENGTHS, k=len(NOISE_LENGTHS)):
            for dc_sign in random.sample((-1, 1), k=2):
                starts = [
                    (r, c)
                    for r in range(out.shape[0])
                    for c in range(out.shape[1])
                ]
                random.shuffle(starts)
                for r, c in starts:
                    cells = _chain_cells(int(r), int(c), int(length), int(dc_sign))
                    if not _valid_noise_chain(out, tile, p, cells, require_legal=require_legal):
                        continue
                    for rr, cc in cells:
                        out[rr, cc] = True
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return out


def _grid_from_mask(mask: np.ndarray, fg: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = fg
    return out


def _validate_case(inp: np.ndarray, clean_mask: np.ndarray, tile: np.ndarray, p: int) -> bool:
    fg = _fg_color(inp)
    out = main(inp)
    expected = _grid_from_mask(clean_mask, fg)
    if not np.array_equal(out, expected):
        return False

    mask = inp == fg
    suspects = _suspect_mask(mask)
    base = mask & (~suspects)
    if not np.any(base):
        return False

    true_score, true_tile, keep = _period_solution(mask, p)
    other_p = 7 - p
    other_score, _other_tile, _other_keep = _period_solution(mask, other_p)
    if true_score <= other_score:
        return False
    if not np.array_equal(true_tile, tile):
        return False

    residue_counts = np.zeros((p, p), dtype=int)
    for r, c in np.argwhere(base):
        residue_counts[r % p, c % p] += 1
    for r, c in np.argwhere(tile):
        if residue_counts[r, c] < 2:
            return False

    rescued = 0
    deleted = 0
    for r, c in np.argwhere(suspects):
        if clean_mask[r, c]:
            if out[r, c] != fg:
                return False
            rescued += 1
        else:
            if out[r, c] != 0:
                return False
            deleted += 1
    if rescued == 0 or deleted == 0 or int(keep.sum()) == 0:
        return False

    if _minimal_period(clean_mask) != p:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    p = random.choice(PERIOD_CHOICES)
    fg = random.choice(FG_CHOICES)
    h = random.randint(max(GRID_H_RANGE[0], 3 * p + 1), GRID_H_RANGE[1])
    w = random.randint(max(GRID_W_RANGE[0], 3 * p + 1), GRID_W_RANGE[1])

    tile = _build_tile(p)
    clean_mask = _render_output((h, w), tile)

    if _minimal_period(clean_mask) != p:
        return None

    inp_mask, protected_list = _apply_rescues(clean_mask)
    protected = set(protected_list)
    inp_mask = _extra_holes(inp_mask, protected, tile, p)
    inp_mask = _add_noise(inp_mask, tile, p)
    if inp_mask is None:
        return None

    inp = _grid_from_mask(inp_mask, fg)
    if _validate_case(inp, clean_mask, tile, p):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-noise puzzle")

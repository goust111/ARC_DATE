from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox


SPECIAL_CONFIGS = [
    (2, 5, 2, 3),
    (2, 5, 3, 3),
    (3, 5, 2, 3),
    (3, 3, 3, 3),
    (4, 3, 3, 3),
]
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _nonzero_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != 0)
    if coords.size == 0:
        raise ValueError("grid must contain a non-zero rectangular region")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _candidate_periods(length: int) -> list[int]:
    return [d for d in range(1, length + 1) if length % d == 0 and length // d >= 2]


def _class_coords(height: int, width: int, tile_h: int, tile_w: int, tr: int, tc: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(tr, height, tile_h)
        for c in range(tc, width, tile_w)
    ]


def _tile_signature(tile: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in tile)


def _minimal_axis_period(tile: np.ndarray, axis: int) -> int:
    length = tile.shape[axis]
    for period in range(1, length + 1):
        if length % period != 0:
            continue
        ok = True
        for idx in range(length):
            base = tile[:, idx % period] if axis == 1 else tile[idx % period, :]
            cur = tile[:, idx] if axis == 1 else tile[idx, :]
            if not np.array_equal(cur, base):
                ok = False
                break
        if ok:
            return period
    return length


def _make_tile(tile_h: int, tile_w: int, colors: list[int]) -> np.ndarray:
    half = tile_w // 2
    for _ in range(300):
        rows = []
        for _r in range(tile_h):
            left = [random.choice(colors) for _ in range(half)]
            center = [random.choice(colors)]
            row = np.array(left + center + left[::-1], dtype=int)
            rows.append(row)
        tile = np.stack(rows, axis=0)
        if len(set(int(v) for v in tile.ravel())) < 2:
            continue
        if _minimal_axis_period(tile, axis=0) != tile_h:
            continue
        if _minimal_axis_period(tile, axis=1) != tile_w:
            continue
        return tile
    raise RuntimeError("failed to build a tile with the required minimal periods")


def _repair_with_period(region: np.ndarray, tile_h: int, tile_w: int, *, trace: bool = False):
    work = np.array(region, dtype=int, copy=True)
    height, width = work.shape
    fill_step = np.full((height, width), -1, dtype=int)
    round_id = 0

    while True:
        proposals: dict[tuple[int, int], int] = {}
        for r, c in np.argwhere(work == 0):
            rr = int(r)
            cc = int(c)
            tile_colors = {
                int(work[r2, c2])
                for r2 in range(rr % tile_h, height, tile_h)
                for c2 in range(cc % tile_w, width, tile_w)
                if int(work[r2, c2]) != 0
            }
            if len(tile_colors) > 1:
                return None if not trace else (None, None)

            sym_color = int(work[rr, width - 1 - cc])
            candidates = set(tile_colors)
            if sym_color != 0:
                candidates.add(sym_color)
            if len(candidates) > 1:
                return None if not trace else (None, None)
            if len(candidates) == 1:
                proposals[(rr, cc)] = next(iter(candidates))

        if not proposals:
            break

        round_id += 1
        for (r, c), color in proposals.items():
            work[r, c] = int(color)
            fill_step[r, c] = round_id

    if np.any(work == 0):
        return None if not trace else (None, None)

    for r in range(height):
        for c in range(width):
            if int(work[r, c]) != int(work[r, width - 1 - c]):
                return None if not trace else (None, None)

    tile = np.zeros((tile_h, tile_w), dtype=int)
    for tr in range(tile_h):
        for tc in range(tile_w):
            vals = {int(work[r, c]) for r, c in _class_coords(height, width, tile_h, tile_w, tr, tc)}
            if len(vals) != 1:
                return None if not trace else (None, None)
            tile[tr, tc] = next(iter(vals))

    rebuilt = np.tile(tile, (height // tile_h, width // tile_w))
    if not np.array_equal(rebuilt, work):
        return None if not trace else (None, None)

    return work if not trace else (work, fill_step)


def _solve_region(region: np.ndarray) -> np.ndarray:
    height, width = region.shape
    candidates = []
    for tile_h in _candidate_periods(height):
        for tile_w in _candidate_periods(width):
            repaired = _repair_with_period(region, tile_h, tile_w)
            if repaired is not None:
                candidates.append((tile_h * tile_w, tile_h, tile_w, repaired))
    if not candidates:
        raise ValueError("no valid joint tiling/symmetry solution found")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the erased rectangular pattern so the result is both periodically tiled and vertically mirror-symmetric.

    Input structure:
    - A black background contains one non-black rectangular region with some cells erased back to black.
    - The hidden clean region is made by tiling a smaller rectangle at least 2 times vertically and at least 2 times horizontally.
    - The clean region is also symmetric about its vertical center column, so mirrored cells match in color.

    Rule:
    - Find the bounding box of all non-black cells and crop to that rectangle.
    - Infer the smallest tile height and width that divide the rectangle while still allowing repeated copies in both axes.
    - Repeatedly repair erased cells by either copying the unique color from the same tile residue class in other repeats or copying the color from the mirrored cell across the vertical center line.
    - Keep propagating both constraints until the rectangle is fully repaired, then return the repaired rectangle only.

    Key insight:
    - Neither periodic repetition nor mirror symmetry is sufficient alone; alternating the two constraints turns initially deadlocked holes into uniquely recoverable cells.
    """
    region = crop_bbox(np.asarray(grid, dtype=int), _nonzero_bbox(np.asarray(grid, dtype=int)))
    return _solve_region(region)


def _full_region(tile_h: int, tile_w: int, rep_h: int, rep_w: int) -> np.ndarray:
    color_count = 2 if random.random() < 0.6 else 3
    colors = random.sample(PALETTE, k=color_count)
    tile = _make_tile(tile_h, tile_w, colors)
    return np.tile(tile, (rep_h, rep_w))


def _residue_of(pos: tuple[int, int], tile_h: int, tile_w: int) -> tuple[int, int]:
    return pos[0] % tile_h, pos[1] % tile_w


def _mirror(pos: tuple[int, int], width: int) -> tuple[int, int]:
    return pos[0], width - 1 - pos[1]


def _special_residues(tile_h: int, tile_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    non_center = [(tr, tc) for tr in range(tile_h) for tc in range(tile_w) if tc != tile_w // 2]
    random.shuffle(non_center)
    first = non_center[0]
    second_choices = [
        rc for rc in non_center[1:]
        if rc != first and rc != (first[0], tile_w - 1 - first[1])
    ]
    if not second_choices:
        raise RuntimeError("not enough distinct non-center residue classes")
    return first, random.choice(second_choices)


def _initial_mask(region: np.ndarray, tile_h: int, tile_w: int, rep_h: int, rep_w: int) -> np.ndarray:
    height, width = region.shape
    missing = np.zeros((height, width), dtype=bool)
    class_size = rep_h * rep_w
    removed_by_residue = {(tr, tc): 0 for tr in range(tile_h) for tc in range(tile_w)}

    sym_residue, chain_residue = _special_residues(tile_h, tile_w)
    for r, c in _class_coords(height, width, tile_h, tile_w, *sym_residue):
        missing[r, c] = True
        removed_by_residue[sym_residue] += 1

    chain_cells = _class_coords(height, width, tile_h, tile_w, *chain_residue)
    for r, c in chain_cells:
        missing[r, c] = True
        removed_by_residue[chain_residue] += 1

    anchor = random.choice(chain_cells)
    mirror_anchor = _mirror(anchor, width)
    mirror_residue = _residue_of(mirror_anchor, tile_h, tile_w)
    if not missing[mirror_anchor]:
        missing[mirror_anchor] = True
        removed_by_residue[mirror_residue] += 1

    protected_residues = {
        sym_residue,
        (sym_residue[0], tile_w - 1 - sym_residue[1]),
        chain_residue,
        mirror_residue,
    }

    def can_add_pair(pos: tuple[int, int]) -> bool:
        if missing[pos]:
            return False
        mirror_pos = _mirror(pos, width)
        if pos == mirror_pos or missing[mirror_pos]:
            return False
        res_a = _residue_of(pos, tile_h, tile_w)
        res_b = _residue_of(mirror_pos, tile_h, tile_w)
        if res_a in protected_residues or res_b in protected_residues:
            return False
        if removed_by_residue[res_a] + 1 >= class_size:
            return False
        if removed_by_residue[res_b] + 1 >= class_size:
            return False
        return True

    def add_pair(pos: tuple[int, int]) -> None:
        mirror_pos = _mirror(pos, width)
        res_a = _residue_of(pos, tile_h, tile_w)
        res_b = _residue_of(mirror_pos, tile_h, tile_w)
        missing[pos] = True
        missing[mirror_pos] = True
        removed_by_residue[res_a] += 1
        removed_by_residue[res_b] += 1

    pair_cells = [(r, c) for r in range(height) for c in range(width // 2) if can_add_pair((r, c))]
    random.shuffle(pair_cells)
    added_pairs = 0
    for pos in pair_cells:
        if added_pairs >= 2:
            break
        if can_add_pair(pos):
            add_pair(pos)
            added_pairs += 1

    target_ratio = random.uniform(0.22, 0.29)
    target_missing = math.ceil(target_ratio * region.size)
    extra_cells = [(r, c) for r in range(height) for c in range(width // 2) if can_add_pair((r, c))]
    random.shuffle(extra_cells)
    for pos in extra_cells:
        if int(missing.sum()) >= target_missing:
            break
        if can_add_pair(pos):
            add_pair(pos)

    return missing


def _border_nonzero(masked: np.ndarray) -> bool:
    return (
        np.any(masked[0, :] != 0)
        and np.any(masked[-1, :] != 0)
        and np.any(masked[:, 0] != 0)
        and np.any(masked[:, -1] != 0)
    )


def _semantic_counts(region: np.ndarray, tile_h: int, tile_w: int) -> tuple[int, int, int]:
    height, width = region.shape
    sym_only = 0
    tile_only = 0
    deadlock = 0
    for r, c in np.argwhere(region == 0):
        rr = int(r)
        cc = int(c)
        tile_colors = {
            int(region[r2, c2])
            for r2 in range(rr % tile_h, height, tile_h)
            for c2 in range(cc % tile_w, width, tile_w)
            if int(region[r2, c2]) != 0
        }
        tile_known = len(tile_colors) == 1
        sym_known = int(region[rr, width - 1 - cc]) != 0
        if sym_known and not tile_known:
            sym_only += 1
        if tile_known and not sym_known:
            tile_only += 1
        if not tile_known and not sym_known:
            deadlock += 1
    return sym_only, tile_only, deadlock


def _unique_period(clean: np.ndarray, masked: np.ndarray, tile_h: int, tile_w: int) -> bool:
    hits = []
    for cand_h in _candidate_periods(clean.shape[0]):
        for cand_w in _candidate_periods(clean.shape[1]):
            repaired = _repair_with_period(masked, cand_h, cand_w)
            if repaired is not None:
                hits.append((cand_h, cand_w, np.array_equal(repaired, clean)))
    exact = [(h, w) for h, w, same in hits if same]
    return exact == [(tile_h, tile_w)]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    tile_h, tile_w, rep_h, rep_w = random.choice(SPECIAL_CONFIGS)
    clean = _full_region(tile_h, tile_w, rep_h, rep_w)
    missing = _initial_mask(clean, tile_h, tile_w, rep_h, rep_w)
    masked = clean.copy()
    masked[missing] = 0

    ratio = float(missing.mean())
    if not (0.20 <= ratio <= 0.30):
        raise RuntimeError("missing ratio out of range")
    if not _border_nonzero(masked):
        raise RuntimeError("erasure collapsed the detectable bounding box")

    repaired, fill_step = _repair_with_period(masked, tile_h, tile_w, trace=True)
    if repaired is None or fill_step is None or not np.array_equal(repaired, clean):
        raise RuntimeError("propagation did not recover the clean region")
    if not _unique_period(clean, masked, tile_h, tile_w):
        raise RuntimeError("period inference is not unique")

    sym_only, tile_only, deadlock = _semantic_counts(masked, tile_h, tile_w)
    if sym_only < 2 or tile_only < 2 or deadlock < 1:
        raise RuntimeError("semantic recovery categories are insufficient")
    if int(np.sum(fill_step >= 2)) < 1:
        raise RuntimeError("missing chain dependency across iterations")

    canvas_h = random.randint(clean.shape[0] + 2, min(30, clean.shape[0] + 10))
    canvas_w = random.randint(clean.shape[1] + 2, min(30, clean.shape[1] + 10))
    top = random.randint(1, canvas_h - clean.shape[0] - 1)
    left = random.randint(1, canvas_w - clean.shape[1] - 1)
    inp = np.zeros((canvas_h, canvas_w), dtype=int)
    inp[top : top + clean.shape[0], left : left + clean.shape[1]] = masked

    if not np.array_equal(main(inp), clean):
        raise RuntimeError("final self-check against main() failed")
    return inp, clean


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, _out = _generate_case()
            return inp
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid ARC input")

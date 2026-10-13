from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = Color.BLACK
OLD = Color.YELLOW
OBS = Color.ORANGE
LOW = Color.RED
HIGH = Color.GREEN

LATTICE_SHAPES = ((2, 3), (3, 2))
ROW_GAP_RANGE = (3, 5)
COL_GAP_RANGE = (3, 5)
MARGIN_RANGE = (2, 4)
OBSTACLE_TOTAL_RANGE = (10, 20)
GEN_RETRY_LIMIT = 400


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _enumerate_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for h, w in ((3, 3), (3, 4), (4, 3), (4, 4)):
        cell_count = h * w
        max_area = min(8, cell_count - 2)
        for bits in range(1 << cell_count):
            area = int(bits.bit_count())
            if not 4 <= area <= max_area:
                continue
            mask = np.array([(bits >> i) & 1 for i in range(cell_count)], dtype=bool).reshape(h, w)
            if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if not _mask_connected(mask):
                continue
            masks.append(mask)
    assert masks
    return masks


SPRITE_MASKS = _enumerate_sprite_masks()


def _sprite_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _sprite_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = OLD
    return patch


def _extract_sprite_mask(grid: np.ndarray, obj: dict) -> np.ndarray:
    patch = crop_bbox(grid, obj["bbox"])
    return patch == OLD


def _analyze_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[int], list[int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, OLD, OBS}), "unexpected input colors"

    yellow_objects = find_components(g, bg=BG, connectivity=4, monochromatic=True, colors=[OLD])
    assert 3 <= len(yellow_objects) <= 5, "expected 3-5 existing yellow sprites"

    sprite_mask = _extract_sprite_mask(g, yellow_objects[0])
    assert 4 <= int(sprite_mask.sum()) <= 8, "sprite area must be 4-8"
    assert max(sprite_mask.shape) <= 4, "sprite bbox must be at most 4x4"

    for obj in yellow_objects[1:]:
        assert np.array_equal(_extract_sprite_mask(g, obj), sprite_mask), "yellow sprites must be identical"

    offsets = _sprite_offsets(sprite_mask)
    rows = sorted({int(obj["bbox"][0]) for obj in yellow_objects})
    cols = sorted({int(obj["bbox"][1]) for obj in yellow_objects})
    occupied = {(int(obj["bbox"][0]), int(obj["bbox"][1])) for obj in yellow_objects}

    assert len(rows) >= 2 and len(cols) >= 2, "lattice needs at least two rows and columns"
    assert len(rows) * len(cols) > len(occupied), "there must be at least one missing lattice slot"
    assert max(sum((r, c) in occupied for c in cols) for r in rows) == len(cols), "need one full lattice row"
    assert max(sum((r, c) in occupied for r in rows) for c in cols) == len(rows), "need one full lattice column"

    h, w = sprite_mask.shape
    for r0 in rows:
        for c0 in cols:
            assert 0 <= r0 <= g.shape[0] - h and 0 <= c0 <= g.shape[1] - w, "sprite bbox must stay in bounds"
    return g, sprite_mask, offsets, rows, cols, occupied


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing lattice sprites, crop them against orange obstacles, and recolor them by survival ratio.

    Input structure:
    - The background is black, with 3-5 identical yellow 4-connected sprites on a rectangular row/column lattice.
    - The yellow sprites share one common sprite mask when cropped to their tight bounding boxes.
    - Orange obstacle pixels appear only in empty lattice slots and nearby empty space; they do not overlap the existing yellow sprites.

    Rule:
    - Recover the common yellow sprite mask and the lattice row starts and column starts from the existing yellow sprites.
    - For every missing lattice position in that Cartesian product, stamp the same sprite mask there.
    - Any stamped pixel that would land on an orange obstacle is omitted; the obstacle remains orange.
    - Count the surviving stamped pixels s versus the full sprite size S. Recolor the whole new stamped remainder green if s > S/2, else red.
    - Keep the original yellow sprites and all orange obstacle pixels unchanged.

    Key insight:
    - The task is not just lattice completion: each missing sprite must be reconstructed with the shared mask, clipped exactly where orange obstacles overlap it, then recolored from that clipped survival count.
    """
    g, _sprite_mask, offsets, rows, cols, occupied = _analyze_input(grid)
    out = g.copy()
    sprite_size = len(offsets)

    for r0 in rows:
        for c0 in cols:
            if (r0, c0) in occupied:
                continue
            survivors: list[tuple[int, int]] = []
            for dr, dc in offsets:
                rr, cc = r0 + dr, c0 + dc
                assert int(g[rr, cc]) in {BG, OBS}, "missing slot must contain only background or obstacles"
                if int(g[rr, cc]) != OBS:
                    survivors.append((rr, cc))
            new_color = HIGH if len(survivors) * 2 > sprite_size else LOW
            for rr, cc in survivors:
                out[rr, cc] = new_color
    return out


def _lattice_cells(row_starts: list[int], col_starts: list[int], offsets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0 in row_starts:
        for c0 in col_starts:
            for dr, dc in offsets:
                cells.add((r0 + dr, c0 + dc))
    return cells


def _expanded_slot_cells(r0: int, c0: int, h: int, w: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    cells = []
    for rr in range(max(0, r0 - 1), min(H, r0 + h + 1)):
        for cc in range(max(0, c0 - 1), min(W, c0 + w + 1)):
            cells.append((rr, cc))
    return cells


def _missing_anchors(rows: list[int], cols: list[int], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in rows for c0 in cols if (r0, c0) not in occupied]


def _build_expected(
    grid: np.ndarray,
    offsets: list[tuple[int, int]],
    missing_specs: list[dict],
) -> np.ndarray:
    out = grid.copy()
    sprite_size = len(offsets)
    for spec in missing_specs:
        r0, c0 = spec["anchor"]
        survivors = []
        killed = spec["killed"]
        for dr, dc in offsets:
            if (dr, dc) not in killed:
                survivors.append((r0 + dr, c0 + dc))
        color = HIGH if len(survivors) * 2 > sprite_size else LOW
        for rr, cc in survivors:
            out[rr, cc] = color
    return out


def _full_completion(grid: np.ndarray, patch: np.ndarray, missing: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r0, c0 in missing:
        paste(out, patch, r0, c0, transparent=BG)
    return out


def _cropped_completion(grid: np.ndarray, offsets: list[tuple[int, int]], missing: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r0, c0 in missing:
        for dr, dc in offsets:
            rr, cc = r0 + dr, c0 + dc
            if int(grid[rr, cc]) != OBS:
                out[rr, cc] = OLD
    return out


def _validate_case(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        g, sprite_mask, offsets, rows, cols, occupied = _analyze_input(grid)
    except AssertionError:
        return False

    missing = _missing_anchors(rows, cols, occupied)
    if len(missing) != 2:
        return False

    sprite_size = len(offsets)
    survivals = []
    nearby_totals = []
    for r0, c0 in missing:
        overlap = sum(int(g[r0 + dr, c0 + dc] == OBS) for dr, dc in offsets)
        if not (1 <= overlap < sprite_size):
            return False
        survivals.append(sprite_size - overlap)
        nearby_totals.append(sum(int(g[rr, cc] == OBS) for rr, cc in _expanded_slot_cells(r0, c0, *sprite_mask.shape, g.shape)))

    if len(set(survivals)) != 2:
        return False
    if not (min(survivals) * 2 <= sprite_size and max(survivals) * 2 > sprite_size):
        return False
    killed = [sprite_size - s for s in survivals]
    if (killed[0] - killed[1]) * (nearby_totals[0] - nearby_totals[1]) >= 0:
        return False
    if not (OBSTACLE_TOTAL_RANGE[0] <= int(np.count_nonzero(g == OBS)) <= OBSTACLE_TOTAL_RANGE[1]):
        return False

    patch = _sprite_patch(sprite_mask)
    if np.array_equal(_full_completion(g, patch, missing), expected):
        return False
    if np.array_equal(_cropped_completion(g, offsets, missing), expected):
        return False
    if not (np.any(expected == LOW) and np.any(expected == HIGH)):
        return False

    try:
        actual = main(g)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    sprite_mask = random.choice(SPRITE_MASKS)
    offsets = _sprite_offsets(sprite_mask)
    sprite_size = len(offsets)
    h, w = sprite_mask.shape
    patch = _sprite_patch(sprite_mask)

    lattice_rows, lattice_cols = random.choice(LATTICE_SHAPES)
    row_gap = random.randint(*ROW_GAP_RANGE)
    col_gap = random.randint(*COL_GAP_RANGE)
    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    row_step = h + row_gap
    col_step = w + col_gap
    H = top + (lattice_rows - 1) * row_step + h + bottom
    W = left + (lattice_cols - 1) * col_step + w + right
    if max(H, W) > 30:
        return None

    row_starts = [top + i * row_step for i in range(lattice_rows)]
    col_starts = [left + j * col_step for j in range(lattice_cols)]
    full_row = random.randrange(lattice_rows)
    full_col = random.randrange(lattice_cols)

    occupied_idx = {(full_row, j) for j in range(lattice_cols)} | {(i, full_col) for i in range(lattice_rows)}
    missing_idx = [(i, j) for i in range(lattice_rows) for j in range(lattice_cols) if (i, j) not in occupied_idx]
    if len(missing_idx) != 2:
        return None

    low_survive = random.randint(1, sprite_size // 2)
    high_survive = random.randint(sprite_size // 2 + 1, sprite_size - 1)
    survival_targets = [low_survive, high_survive]
    random.shuffle(survival_targets)

    grid = np.zeros((H, W), dtype=int)
    for i, j in occupied_idx:
        paste(grid, patch, row_starts[i], col_starts[j], transparent=BG)

    lattice_mask_cells = _lattice_cells(row_starts, col_starts, offsets)
    missing_specs = []
    for (i, j), survive in zip(missing_idx, survival_targets):
        r0, c0 = row_starts[i], col_starts[j]
        killed = set(random.sample(offsets, sprite_size - survive))
        for dr, dc in killed:
            grid[r0 + dr, c0 + dc] = OBS
        missing_specs.append(
            {
                "anchor": (r0, c0),
                "survive": survive,
                "killed": killed,
                "candidate_fillers": [],
            }
        )

    for spec in missing_specs:
        r0, c0 = spec["anchor"]
        candidates = []
        for rr, cc in _expanded_slot_cells(r0, c0, h, w, grid.shape):
            if (rr, cc) in lattice_mask_cells:
                continue
            if int(grid[rr, cc]) != BG:
                continue
            candidates.append((rr, cc))
        random.shuffle(candidates)
        spec["candidate_fillers"] = candidates

    kill_counts = [len(spec["killed"]) for spec in missing_specs]
    capacities = [len(spec["candidate_fillers"]) for spec in missing_specs]
    base_obstacles = sum(kill_counts)
    feasible_fillers = []
    for fill0 in range(capacities[0] + 1):
        for fill1 in range(capacities[1] + 1):
            total = base_obstacles + fill0 + fill1
            near0 = kill_counts[0] + fill0
            near1 = kill_counts[1] + fill1
            if not (OBSTACLE_TOTAL_RANGE[0] <= total <= OBSTACLE_TOTAL_RANGE[1]):
                continue
            if (kill_counts[0] - kill_counts[1]) * (near0 - near1) < 0:
                feasible_fillers.append((fill0, fill1))
    if not feasible_fillers:
        return None

    fill_counts = random.choice(feasible_fillers)
    for spec, fill_count in zip(missing_specs, fill_counts):
        for rr, cc in spec["candidate_fillers"][:fill_count]:
            grid[rr, cc] = OBS

    expected = _build_expected(grid, offsets, missing_specs)
    return grid if _validate_case(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FG = 3
WALL = 5
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def count_holes_in_crop(crop: np.ndarray) -> int:
    holes = 0
    h, w = crop.shape
    for comp in components_mask(crop == BG):
        if np.all((comp[:, 0] > 0) & (comp[:, 0] < h - 1) & (comp[:, 1] > 0) & (comp[:, 1] < w - 1)):
            holes += 1
    return holes


def _shape(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rot in range(4):
        cur = np.rot90(mask, rot)
        for candidate in (cur, np.fliplr(cur), np.flipud(cur)):
            key = (candidate.shape, candidate.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(candidate.copy())
    return out


def _mask_hole_count(mask: np.ndarray) -> int:
    patch = np.where(mask, FG, BG)
    return count_holes_in_crop(patch)


_BASE_SHAPES = {
    0: [
        ("110", "111", "010"),
        ("111", "101", "100"),
        ("0110", "1111", "0010"),
        ("1111", "0011", "0010"),
        ("010", "111", "011"),
    ],
    1: [
        ("111", "101", "111"),
        ("1111", "1001", "1011", "1111"),
        ("11111", "10001", "10111", "10001", "11111"),
        ("1111", "1011", "1001", "1111"),
    ],
    2: [
        ("11111", "10101", "11111", "11111", "11111"),
        ("11111", "10001", "11111", "11101", "11111"),
        ("11111", "10111", "11111", "11101", "11111"),
    ],
    3: [
        ("11111", "10101", "11111", "10111", "11111"),
        ("11111", "10001", "11111", "10101", "11111"),
    ],
}


def _build_shape_bank() -> dict[int, list[np.ndarray]]:
    bank: dict[int, list[np.ndarray]] = {}
    for holes, patterns in _BASE_SHAPES.items():
        unique: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
        for rows in patterns:
            base = _shape(rows)
            for mask in _all_transforms(base):
                if len(components_mask(mask)) != 1:
                    continue
                if _mask_hole_count(mask) != holes:
                    continue
                unique[(mask.shape, mask.tobytes())] = mask
        bank[holes] = list(unique.values())
    return bank


SHAPE_BANK = _build_shape_bank()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the chamber indexed by the unique-hole block and erase every other chamber's block.

    Input structure:
    - Black background cells and a connected grey wall divide the grid into 4 to 6 non-wall chambers.
    - Each chamber contains exactly one 4-connected green foreground block separated from the wall by background.
    - Different blocks may contain different numbers of enclosed background holes inside their own bounding boxes.

    Rule:
    - Count, for each green block, how many 4-connected background hole components lie fully inside that block's bounding box.
    - Find the single hole count value that appears exactly once; that block is the outlier.
    - Treat grey as the only barrier, locate the non-wall chamber containing that outlier block, and recolor every black cell in that chamber to green.
    - Recolor all green pixels in every other chamber to black, while keeping every grey wall pixel unchanged.

    Key insight:
    - The hole-count outlier does not get output directly; it serves as an address that selects which surrounding chamber should be flood-filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blocks = components_mask(g == FG)
    assert 4 <= len(blocks) <= 6

    hole_counts: list[int] = []
    for coords in blocks:
        crop = crop_bbox(g, bbox_coords(coords))
        hole_counts.append(count_holes_in_crop(crop))

    freq = Counter(hole_counts)
    unique_values = [value for value, count in freq.items() if count == 1]
    assert len(unique_values) == 1
    target_value = unique_values[0]

    target_blocks = [coords for coords, holes in zip(blocks, hole_counts) if holes == target_value]
    assert len(target_blocks) == 1
    target_block = target_blocks[0]
    target_cell = tuple(map(int, target_block[0]))

    chambers = components_mask(g != WALL)
    target_chamber: np.ndarray | None = None
    for chamber in chambers:
        if np.any(np.all(chamber == np.array(target_cell, dtype=int), axis=1)):
            target_chamber = chamber
            break
    assert target_chamber is not None

    target_mask = np.zeros_like(g, dtype=bool)
    target_mask[target_chamber[:, 0], target_chamber[:, 1]] = True

    out = g.copy()
    out[target_mask & (out == BG)] = FG
    out[(~target_mask) & (out == FG)] = BG
    return out


def _build_wall_layout() -> np.ndarray | None:
    chamber_count = random.choice([4, 5, 6])
    if chamber_count == 4:
        rows, cols = 2, 2
    else:
        rows, cols = random.choice([(2, 3), (3, 2)])

    row_spans = [random.randint(7, 8) for _ in range(rows)]
    col_spans = [random.randint(7, 8) for _ in range(cols)]
    h = sum(row_spans) + rows + 1
    w = sum(col_spans) + cols + 1
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    room_rows: list[tuple[int, int]] = []
    room_cols: list[tuple[int, int]] = []

    cursor = 1
    for idx, span in enumerate(row_spans):
        room_rows.append((cursor, cursor + span - 1))
        cursor += span
        if idx < rows - 1:
            grid[cursor, :] = WALL
            cursor += 1

    cursor = 1
    for idx, span in enumerate(col_spans):
        room_cols.append((cursor, cursor + span - 1))
        cursor += span
        if idx < cols - 1:
            grid[:, cursor] = WALL
            cursor += 1

    if chamber_count == 5:
        doorway_vertical = random.choice([False, True])
        if doorway_vertical and cols >= 2:
            wall_idx = random.randrange(cols - 1)
            col = room_cols[wall_idx][1] + 1
            row_idx = random.randrange(rows)
            r0, r1 = room_rows[row_idx]
            width = random.choice([2, 3])
            start = random.randint(r0 + 1, r1 - width)
            grid[start : start + width, col] = BG
        else:
            wall_idx = random.randrange(rows - 1)
            row = room_rows[wall_idx][1] + 1
            col_idx = random.randrange(cols)
            c0, c1 = room_cols[col_idx]
            width = random.choice([2, 3])
            start = random.randint(c0 + 1, c1 - width)
            grid[row, start : start + width] = BG

    if len(components_mask(grid == WALL)) != 1:
        return None
    if len(components_mask(grid == BG)) != chamber_count:
        return None
    return grid


def _sample_hole_counts(chamber_count: int, target_idx: int) -> list[int]:
    values = [0, 1, 2, 3]
    unique_value = random.choice(values)
    others = [v for v in values if v != unique_value]
    counts = [unique_value]

    if chamber_count == 4:
        common = random.choice(others)
        counts += [common] * 3
    elif chamber_count == 5 and random.random() < 0.5:
        common = random.choice(others)
        counts += [common] * 4
    elif chamber_count == 5:
        a, b = random.sample(others, 2)
        counts += [a, a, b, b]
    elif chamber_count == 6 and random.random() < 0.5:
        common = random.choice(others)
        counts += [common] * 5
    else:
        a, b = random.sample(others, 2)
        counts += [a, a, a, b, b]

    random.shuffle(counts[1:])
    out = [0] * chamber_count
    out[target_idx] = unique_value
    rest = [value for value in counts[1:]]
    random.shuffle(rest)
    ptr = 0
    for idx in range(chamber_count):
        if idx == target_idx:
            continue
        out[idx] = rest[ptr]
        ptr += 1
    return out


def _sample_distinct_masks(hole_counts: list[int]) -> list[np.ndarray] | None:
    chosen: list[np.ndarray | None] = [None] * len(hole_counts)
    for hole_value in sorted(set(hole_counts)):
        indices = [idx for idx, value in enumerate(hole_counts) if value == hole_value]
        candidates = SHAPE_BANK[hole_value][:]
        if len(candidates) < len(indices):
            return None
        picks = random.sample(candidates, len(indices))
        for idx, mask in zip(indices, picks):
            chosen[idx] = mask.copy()
    return [mask for mask in chosen if mask is not None]


def _valid_origins(chamber_mask: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch.shape
    coords = np.argwhere(chamber_mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    origins: list[tuple[int, int]] = []
    for rr in range(r0 + 1, r1 - h + 2):
        for cc in range(c0 + 1, c1 - w + 2):
            if rr - 1 < 0 or cc - 1 < 0 or rr + h >= chamber_mask.shape[0] or cc + w >= chamber_mask.shape[1]:
                continue
            if np.all(chamber_mask[rr - 1 : rr + h + 1, cc - 1 : cc + w + 1]):
                origins.append((rr, cc))
    return origins


def _place_blocks(layout: np.ndarray, chamber_masks: list[np.ndarray], block_masks: list[np.ndarray]) -> np.ndarray | None:
    grid = layout.copy()
    order = list(range(len(chamber_masks)))
    random.shuffle(order)
    for idx in order:
        origins = _valid_origins(chamber_masks[idx], block_masks[idx])
        if not origins:
            return None
        r0, c0 = random.choice(origins)
        patch = block_masks[idx]
        view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[patch] = FG
    return grid


def _component_label(shape: tuple[int, int], components: list[np.ndarray]) -> np.ndarray:
    label = np.full(shape, -1, dtype=int)
    for idx, comp in enumerate(components):
        label[comp[:, 0], comp[:, 1]] = idx
    return label


def _green_not_touching_wall(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r, c in np.argwhere(grid == FG):
        for dr, dc in NEIGHBORS8:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == WALL:
                return False
    return True


def _target_info(grid: np.ndarray):
    blocks = components_mask(grid == FG)
    hole_counts = [count_holes_in_crop(crop_bbox(grid, bbox_coords(coords))) for coords in blocks]
    freq = Counter(hole_counts)
    unique_values = [value for value, count in freq.items() if count == 1]
    if len(unique_values) != 1:
        return None
    target_value = unique_values[0]
    target_blocks = [coords for coords, holes in zip(blocks, hole_counts) if holes == target_value]
    if len(target_blocks) != 1:
        return None
    return blocks, hole_counts, target_blocks[0]


def _semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if len(components_mask(grid == WALL)) != 1:
        return False
    if not _green_not_touching_wall(grid):
        return False

    chambers = components_mask(grid != WALL)
    if not (4 <= len(chambers) <= 6):
        return False
    chamber_label = _component_label(grid.shape, chambers)

    green_components = components_mask(grid == FG)
    if len(green_components) != len(chambers):
        return False

    chamber_counts = [0] * len(chambers)
    for comp in green_components:
        ids = {int(chamber_label[r, c]) for r, c in comp}
        if len(ids) != 1 or -1 in ids:
            return False
        chamber_counts[next(iter(ids))] += 1
    if any(count != 1 for count in chamber_counts):
        return False

    target_data = _target_info(grid)
    if target_data is None:
        return False
    _blocks, _hole_counts, target_block = target_data
    target_cell = tuple(map(int, target_block[0]))
    target_idx = int(chamber_label[target_cell])

    bg_areas = [int(np.count_nonzero(grid[comp[:, 0], comp[:, 1]] == BG)) for comp in chambers]
    target_area = bg_areas[target_idx]
    if target_area == min(bg_areas) or target_area == max(bg_areas):
        return False

    out = main(grid)
    if not np.array_equal(out[grid == WALL], grid[grid == WALL]):
        return False
    for idx, chamber in enumerate(chambers):
        vals = out[chamber[:, 0], chamber[:, 1]]
        if idx == target_idx:
            if not np.all(vals == FG):
                return False
        else:
            if np.any(vals == FG):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _build_wall_layout()
    if layout is None:
        return None

    chamber_masks = []
    for chamber in components_mask(layout == BG):
        mask = np.zeros_like(layout, dtype=bool)
        mask[chamber[:, 0], chamber[:, 1]] = True
        chamber_masks.append(mask)

    chamber_areas = [int(mask.sum()) for mask in chamber_masks]
    candidates = [idx for idx, area in enumerate(chamber_areas) if area != min(chamber_areas) and area != max(chamber_areas)]
    if not candidates:
        return None

    target_idx = random.choice(candidates)
    hole_counts = _sample_hole_counts(len(chamber_masks), target_idx)
    block_masks = _sample_distinct_masks(hole_counts)
    if block_masks is None:
        return None

    grid = _place_blocks(layout, chamber_masks, block_masks)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

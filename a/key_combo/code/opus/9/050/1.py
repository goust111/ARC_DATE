from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
SIZE_OPTIONS = {
    0: (4, 5, 6),
    1: (4, 5, 6),
    2: (4, 5, 6),
    3: (5, 6),
}
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 300
CARVE_ATTEMPTS = 40
CARVE_KEEP_PROB = 0.6


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _mask_component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _count_holes_in_mask(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _count_holes_in_patch(patch: np.ndarray, color: int) -> int:
    return _count_holes_in_mask(np.asarray(patch == color, dtype=bool))


def _find_components(grid: np.ndarray, bg: int):
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    comps = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            coords = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
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
                    "area": int(len(coords)),
                }
            )
    return comps


def _rotate_patch_clockwise(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(np.asarray(patch, dtype=int), -int(turns)).copy()


def _tight_crop_nonzero(patch: np.ndarray, bg: int = BG) -> np.ndarray:
    box = _bbox(np.asarray(patch != bg, dtype=bool))
    assert box is not None
    return _crop_bbox(patch, box)


def _pack_crops(items: list[tuple[int, np.ndarray]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: item[0])
    max_h = max(int(crop.shape[0]) for _, crop in ordered)
    total_w = sum(int(crop.shape[1]) for _, crop in ordered)
    out = np.full((max_h, total_w), BG, dtype=int)
    c0 = 0
    for _, crop in ordered:
        h, w = crop.shape
        out[:h, c0 : c0 + w] = crop
        c0 += w
    return out


def _shape_touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_shape(mask: np.ndarray, hole_count: int) -> bool:
    if not mask.any():
        return False
    if _mask_component_count(mask) != 1:
        return False
    if _count_holes_in_mask(mask) != hole_count:
        return False
    if not _shape_touches_all_sides(mask):
        return False
    return True


def _pick_hole_seeds(size: int, hole_count: int):
    positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(120):
        random.shuffle(positions)
        seeds = []
        for r, c in positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == hole_count:
                    return seeds
    return None


def _boundary_cells(mask: np.ndarray):
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            edge_or_bg_neighbor = False
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    edge_or_bg_neighbor = True
                    break
            if edge_or_bg_neighbor:
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _generate_mask_for_holes(hole_count: int) -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        size = random.choice(SIZE_OPTIONS[hole_count])
        mask = np.ones((size, size), dtype=bool)
        seeds = _pick_hole_seeds(size, hole_count)
        if hole_count and seeds is None:
            continue
        for r, c in seeds or []:
            mask[r, c] = False

        if not _valid_shape(mask, hole_count):
            continue

        target_area = random.randint(max(size + 3, int(size * size * 0.5)), int(mask.sum()))
        for _ in range(CARVE_ATTEMPTS):
            if int(mask.sum()) <= target_area:
                break
            changed = False
            for r, c in _boundary_cells(mask):
                trial = mask.copy()
                trial[r, c] = False
                if _valid_shape(trial, hole_count) and random.random() < CARVE_KEEP_PROB:
                    mask = trial
                    changed = True
                    break
            if not changed:
                break

        if hole_count and np.array_equal(mask, np.rot90(mask, -hole_count)):
            continue
        if hole_count == 0 and np.array_equal(mask, np.rot90(mask, -1)):
            continue
        if int(mask.sum()) >= size * size:
            continue
        return mask
    raise RuntimeError(f"failed to build shape for hole count {hole_count}")


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _sprite_output_patch(patch: np.ndarray, hole_count: int) -> np.ndarray:
    rotated = _rotate_patch_clockwise(patch, hole_count)
    return _tight_crop_nonzero(rotated, bg=BG)


def _non_monotonic_permutation():
    options = [
        perm
        for perm in itertools.permutations((0, 1, 2, 3))
        if perm != (0, 1, 2, 3) and perm != (3, 2, 1, 0)
    ]
    return random.choice(options)


def _build_expected_from_placed(placed):
    items = []
    for item in placed:
        items.append((item["left"], _sprite_output_patch(item["patch"], item["holes"])))
    return _pack_crops(items)


def _build_unrotated_pack(placed):
    items = []
    for item in placed:
        items.append((item["left"], _tight_crop_nonzero(item["patch"], bg=BG)))
    return _pack_crops(items)


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each monochrome sprite by a hole-count-driven angle, then tightly pack the rotated crops left-to-right.

    Input structure:
    - The grid has black background and exactly four separated monochrome 4-connected sprites.
    - Each sprite uses a different non-black color, has a tight square bounding box, and may contain 0, 1, 2, or 3 enclosed black hole components.
    - The four sprites are scattered with non-overlapping boxes and distinct original horizontal positions.

    Rule:
    - For each sprite, count its enclosed 4-connected black hole regions inside its own square bounding box.
    - Rotate that sprite clockwise by hole_count * 90 degrees, then crop the rotated result to its tight non-black bounding box.
    - Sort sprites by the original left column of their input bounding boxes and concatenate the cropped rotated sprites horizontally with top alignment, padding shorter crops with black at the bottom.

    Key insight:
    - The rotation angle is encoded by topology: the number of enclosed holes determines how many quarter-turns to apply before the final tight packing.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg)
    assert len(comps) == 4
    assert len(_colors_present(g, bg)) == 4

    items = []
    seen_holes = set()
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 - r0 == c1 - c0
        patch = _crop_bbox(g, comp["bbox"])
        color = int(comp["color"])
        hole_count = _count_holes_in_patch(patch, color)
        assert 0 <= hole_count <= 3
        seen_holes.add(hole_count)
        items.append((c0, _sprite_output_patch(patch, hole_count)))

    assert seen_holes == {0, 1, 2, 3}
    return _pack_crops(items)


def _gen_attempt() -> np.ndarray | None:
    order = _non_monotonic_permutation()
    colors = random.sample(PALETTE, 4)
    sprites = []
    sizes = []
    for holes, color in zip(order, colors):
        mask = _generate_mask_for_holes(holes)
        patch = _mask_to_patch(mask, color)
        sprites.append({"holes": holes, "patch": patch, "size": int(patch.shape[0]), "color": color})
        sizes.append(int(patch.shape[0]))

    if len(set(sizes)) == 1:
        return None

    gaps = [random.randint(1, 3) for _ in range(3)]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + sum(sizes) + sum(gaps) + right_margin
    if width > 30:
        return None

    max_size = max(sizes)
    height = random.randint(max_size + 4, min(22, max_size + 9))
    if height > 30:
        return None

    center_row = random.randint(max_size // 2 + 1, height - (max_size // 2) - 2)
    grid = np.zeros((height, width), dtype=int)

    cur_left = left_margin
    placed = []
    for idx, sprite in enumerate(sprites):
        size = sprite["size"]
        ideal_top = center_row - size // 2
        top = max(1, min(height - size - 1, ideal_top + random.randint(-1, 1)))
        patch = sprite["patch"]
        grid[top : top + size, cur_left : cur_left + size] = patch
        placed.append({"holes": sprite["holes"], "patch": patch, "left": cur_left, "top": top})
        cur_left += size
        if idx < len(gaps):
            cur_left += gaps[idx]

    expected = _build_expected_from_placed(placed)
    unrotated = _build_unrotated_pack(placed)
    if np.array_equal(expected, unrotated):
        return None
    if len({item["holes"] for item in placed}) != 4:
        return None
    if [item["holes"] for item in placed] == [0, 1, 2, 3]:
        return None
    if [item["holes"] for item in placed] == [3, 2, 1, 0]:
        return None
    if len(set(int(_sprite_output_patch(item["patch"], item["holes"]).shape[0]) for item in placed)) == 1:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    comps = _find_components(grid, bg=BG)
    if len(comps) != 4:
        return None
    detected_holes = sorted(_count_holes_in_patch(_crop_bbox(grid, comp["bbox"]), int(comp["color"])) for comp in comps)
    if detected_holes != [0, 1, 2, 3]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")

from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if _ARC_UTILS_PATH.exists():
    _spec = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
    _arc_utils = importlib.util.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_arc_utils)
    get_bg = _arc_utils.get_bg
    crop_bbox = _arc_utils.crop_bbox
else:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


BBox = tuple[int, int, int, int]

MAX_SIDE = 30
LEGEND_WIDTH = 2
MIN_LEGEND_ROWS = 3
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120


def _block(grid: np.ndarray, box: BBox) -> np.ndarray:
    return crop_bbox(grid, box)


def _sorted_box_pair(a: BBox, b: BBox) -> tuple[BBox, BBox]:
    return (a, b) if (a[0], a[1], a[2], a[3]) <= (b[0], b[1], b[2], b[3]) else (b, a)


def _moat_is_bg(grid: np.ndarray, box: BBox, bg: int) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    window = grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    window[r0 - rr0 : r1 - rr0 + 1, c0 - cc0 : c1 - cc0 + 1] = bg
    return bool(np.all(window == bg))


def _candidate_legend_boxes(grid: np.ndarray, bg: int) -> list[BBox]:
    h, w = grid.shape
    candidates: list[BBox] = []
    for c0 in range(w - 1):
        active = np.any(grid[:, c0 : c0 + LEGEND_WIDTH] != bg, axis=1)
        r = 0
        while r < h:
            if not active[r]:
                r += 1
                continue
            r0 = r
            while r + 1 < h and active[r + 1]:
                r += 1
            r1 = r
            if r1 - r0 + 1 >= MIN_LEGEND_ROWS:
                box = (r0, c0, r1, c0 + 1)
                block = _block(grid, box)
                visible_colors = set(map(int, block[block != bg].ravel()))
                if (
                    np.all(np.any(block != bg, axis=1))
                    and np.any(block[:, 0] != bg)
                    and np.any(block[:, 1] != bg)
                    and np.any(block == bg)
                    and len(visible_colors) >= 2
                    and _moat_is_bg(grid, box, bg)
                ):
                    candidates.append(box)
            r += 1
    return candidates


def _merge_legend_blocks(a: np.ndarray, b: np.ndarray, bg: int) -> np.ndarray | None:
    if a.shape != b.shape or a.shape[1] != LEGEND_WIDTH:
        return None
    missing = (a == bg) & (b == bg)
    clash = (a != bg) & (b != bg) & (a != b)
    if np.any(missing) or np.any(clash):
        return None
    merged = a.copy()
    merged[a == bg] = b[a == bg]
    if np.any(merged == bg):
        return None
    return merged


def _find_legend_pair(grid: np.ndarray, bg: int) -> tuple[tuple[BBox, BBox], np.ndarray]:
    candidates = _candidate_legend_boxes(grid, bg)
    matches: list[tuple[tuple[BBox, BBox], np.ndarray]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            box_a = candidates[i]
            box_b = candidates[j]
            if (box_a[2] - box_a[0]) != (box_b[2] - box_b[0]):
                continue
            merged = _merge_legend_blocks(_block(grid, box_a), _block(grid, box_b), bg)
            if merged is None:
                continue
            matches.append((_sorted_box_pair(box_a, box_b), merged))
    unique: dict[tuple[BBox, BBox], np.ndarray] = {}
    for pair, merged in matches:
        unique[pair] = merged
    assert len(unique) == 1, f"expected exactly one compatible legend pair, found {len(unique)}"
    pair, merged = next(iter(unique.items()))
    return pair, merged


def _apply_pairs(grid: np.ndarray, legend_boxes: tuple[BBox, BBox], pairs: list[tuple[int, int]], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    legend_mask = np.zeros(out.shape, dtype=bool)
    for r0, c0, r1, c1 in legend_boxes:
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    targets = np.argwhere((out != bg) & (~legend_mask))
    for r, c in targets:
        color = int(out[r, c])
        for src, dst in pairs:
            if color == src:
                color = dst
        out[r, c] = color
    return out


def _visible_pairs(block: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for row in block:
        left = int(row[0])
        right = int(row[1])
        if left != bg and right != bg:
            pairs.append((left, right))
    return pairs


def main(input_grid: np.ndarray) -> np.ndarray:
    """Repair the complementary damaged two-column legends, then execute their ordered recolor chain on the target.

    Input structure:
    - The background is the dominant color.
    - Two isolated vertical legend blocks have width 2 and the same height; they are damaged copies of the same ordered substitution table, with some cells erased to background.
    - Every other non-background pixel belongs to the target object area to be recolored.

    Rule:
    - Locate the unique pair of same-height width-2 legend blocks whose cells are mutually compatible under complementary repair.
    - Overlay the two copies cell by cell, filling each missing legend entry from the other copy to recover the complete two-column legend.
    - Read the repaired legend from top to bottom as ordered substitutions left->right and apply that chain to every non-background pixel outside both legend blocks, while leaving the two damaged legend copies unchanged.

    Key insight:
    - Neither damaged copy is trustworthy on its own: only the complementary merge reconstructs the full ordered substitution sequence, and that recovered row order controls the final color after chaining.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    bg = int(get_bg(grid))
    legend_boxes, merged = _find_legend_pair(grid, bg)
    pairs = [(int(row[0]), int(row[1])) for row in merged]
    return _apply_pairs(grid, legend_boxes, pairs, bg)


def _apply_damage_state(copy_a: np.ndarray, copy_b: np.ndarray, row: int, state: int, bg: int) -> None:
    if state == 0:
        return
    if state == 1:
        copy_a[row, 0] = bg
    elif state == 2:
        copy_a[row, 1] = bg
    elif state == 3:
        copy_b[row, 0] = bg
    elif state == 4:
        copy_b[row, 1] = bg
    elif state == 5:
        copy_a[row, 0] = bg
        copy_b[row, 1] = bg
    elif state == 6:
        copy_a[row, 1] = bg
        copy_b[row, 0] = bg
    else:
        raise ValueError(f"unknown damage state {state}")


def _choose_palette(rng: random.Random, bg: int) -> list[int]:
    return [color for color in range(10) if color != bg]


def _build_legend(rng: random.Random, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    palette = _choose_palette(rng, bg)
    chain_len = rng.randint(3, 6)
    chain_colors = rng.sample(palette, chain_len + 1)
    merged = np.array([(chain_colors[i], chain_colors[i + 1]) for i in range(chain_len)], dtype=int)
    for _ in range(80):
        copy_a = merged.copy()
        copy_b = merged.copy()

        critical_rows = [0, 1]
        for row in critical_rows:
            _apply_damage_state(copy_a, copy_b, row, rng.choice([5, 6]), bg)

        for row in range(2, chain_len):
            _apply_damage_state(copy_a, copy_b, row, rng.choice([0, 1, 2, 3, 4, 5, 6]), bg)

        if not np.all(np.any(copy_a != bg, axis=1)):
            continue
        if not np.all(np.any(copy_b != bg, axis=1)):
            continue
        if not np.any(copy_a[:, 0] != bg) or not np.any(copy_a[:, 1] != bg):
            continue
        if not np.any(copy_b[:, 0] != bg) or not np.any(copy_b[:, 1] != bg):
            continue
        if not (np.any(copy_a == bg) and np.any(copy_b == bg)):
            continue
        return merged, copy_a, copy_b, chain_colors
    raise RuntimeError("failed to build a non-degenerate damaged legend")


def _reserve(mask: np.ndarray, box: BBox, pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(mask: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = mask.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if mask[r, c]:
            return False
    return True


def _draw_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _shape_offsets(kind: str, rng: random.Random) -> list[tuple[int, int]]:
    if kind == "single":
        return [(0, 0)]
    if kind == "hline":
        length = rng.randint(2, 5)
        return [(0, dc) for dc in range(length)]
    if kind == "vline":
        length = rng.randint(2, 5)
        return [(dr, 0) for dr in range(length)]
    if kind == "square2":
        return [(dr, dc) for dr in range(2) for dc in range(2)]
    if kind == "square3":
        return [(dr, dc) for dr in range(3) for dc in range(3)]
    if kind == "plus3":
        return [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
    if kind == "tee3":
        return [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]
    raise ValueError(f"unknown shape kind {kind}")


def _place_legend(grid: np.ndarray, blocked: np.ndarray, block: np.ndarray, rng: random.Random) -> BBox:
    h, w = grid.shape
    bh, bw = block.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        if rng.random() < 0.35:
            r0 = rng.choice([0, h - bh]) if h > bh else 0
        else:
            r0 = rng.randint(0, h - bh)
        if rng.random() < 0.35:
            c0 = rng.choice([0, w - bw]) if w > bw else 0
        else:
            c0 = rng.randint(0, w - bw)
        box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        region = blocked[max(0, r0 - 1) : min(h, r0 + bh + 1), max(0, c0 - 1) : min(w, c0 + bw + 1)]
        if np.any(region):
            continue
        grid[r0 : r0 + bh, c0 : c0 + bw] = block
        _reserve(blocked, box, pad=1)
        return box
    raise RuntimeError("failed to place legend")


def _place_shape(grid: np.ndarray, blocked: np.ndarray, offsets: list[tuple[int, int]], color: int, rng: random.Random) -> bool:
    h, w = grid.shape
    max_r = max(r for r, _ in offsets)
    max_c = max(c for _, c in offsets)
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(0, h - max_r - 1)
        c0 = rng.randint(0, w - max_c - 1)
        coords = [(r0 + dr, c0 + dc) for dr, dc in offsets]
        if not _can_place(blocked, coords):
            continue
        _draw_coords(grid, coords, color)
        rr = [r for r, _ in coords]
        cc = [c for _, c in coords]
        _reserve(blocked, (min(rr), min(cc), max(rr), max(cc)), pad=1)
        return True
    return False


def _build_target(grid: np.ndarray, blocked: np.ndarray, merged: np.ndarray, chain_colors: list[int], bg: int, rng: random.Random) -> bool:
    required_colors = [chain_colors[0], chain_colors[1]]
    spare_colors = [c for c in range(10) if c not in set(chain_colors)]
    if spare_colors:
        required_colors.append(rng.choice(spare_colors))

    guaranteed_shapes = ["single", "hline", "vline"]
    for color, kind in zip(required_colors, guaranteed_shapes):
        if not _place_shape(grid, blocked, _shape_offsets(kind, rng), color, rng):
            return False

    shape_kinds = ["single", "hline", "vline", "square2", "square3", "plus3", "tee3"]
    usable_colors = chain_colors[:-1] + required_colors[2:]
    extra_count = rng.randint(4, 10)
    for _ in range(extra_count):
        color = rng.choice(usable_colors)
        kind = rng.choice(shape_kinds)
        _place_shape(grid, blocked, _shape_offsets(kind, rng), color, rng)

    legend_colors = set(map(int, merged.ravel()))
    legend_boxes, merged2 = _find_legend_pair(grid, bg)
    assert np.array_equal(merged, merged2)
    out = _apply_pairs(grid, legend_boxes, [(int(a), int(b)) for a, b in merged], int(get_bg(grid)))
    legend_mask = np.zeros(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in legend_boxes:
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    changed = (grid != out) & (~legend_mask)
    if not np.any(changed):
        return False
    changed_colors = set(map(int, grid[changed].ravel()))
    return bool(changed_colors & legend_colors)


def _semantic_checks(grid: np.ndarray) -> None:
    bg = int(get_bg(grid))
    legend_boxes, merged = _find_legend_pair(grid, bg)
    pairs = [(int(row[0]), int(row[1])) for row in merged]
    out = main(grid)
    expected = _apply_pairs(grid, legend_boxes, pairs, bg)
    assert np.array_equal(out, expected)

    legend_mask = np.zeros(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in legend_boxes:
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.array_equal(out[legend_mask], grid[legend_mask])
    assert np.any((out != grid) & (~legend_mask))

    block_a = _block(grid, legend_boxes[0])
    block_b = _block(grid, legend_boxes[1])
    partial_a = _apply_pairs(grid, legend_boxes, _visible_pairs(block_a, bg), bg)
    partial_b = _apply_pairs(grid, legend_boxes, _visible_pairs(block_b, bg), bg)
    assert not np.array_equal(out, partial_a)
    assert not np.array_equal(out, partial_b)

    reversed_out = _apply_pairs(grid, legend_boxes, list(reversed(pairs)), bg)
    assert not np.array_equal(out, reversed_out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        bg = rng.randrange(10)
        merged, copy_a, copy_b, chain_colors = _build_legend(rng, bg)
        h = rng.randint(max(12, merged.shape[0] + 4), MAX_SIDE)
        w = rng.randint(12, MAX_SIDE)

        grid = np.full((h, w), bg, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        try:
            _place_legend(grid, blocked, copy_a, rng)
            _place_legend(grid, blocked, copy_b, rng)
        except RuntimeError:
            continue

        if not _build_target(grid, blocked, merged, chain_colors, bg, rng):
            continue

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue

        assert grid.ndim == 2 and max(grid.shape) <= MAX_SIDE
        return grid

    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

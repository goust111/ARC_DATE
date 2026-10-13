import random
from collections import deque

import numpy as np


BG = 0
TARGET = 1
MARKER = 2
MOVER = 3
PUSHER = 4
GUIDE = 5
RAY = 9

GEN_RETRIES = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _center2(bbox: tuple[int, int, int, int], dim: int) -> int:
    if dim == 0:
        return int(bbox[0] + bbox[2])
    return int(bbox[1] + bbox[3])


def _span(bbox: tuple[int, int, int, int], dim: int) -> int:
    if dim == 0:
        return int(bbox[2] - bbox[0] + 1)
    return int(bbox[3] - bbox[1] + 1)


def _solid_rect_patch(patch: np.ndarray, color: int) -> bool:
    return bool(np.all(patch == int(color)))


def _order_triplet(coords: np.ndarray) -> tuple[tuple[int, int], int, int]:
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    d1 = coords[1] - coords[0]
    d2 = coords[2] - coords[1]
    assert np.array_equal(d1, d2), "guide markers must be equally spaced"
    dr, dc = map(int, d1)
    assert (dr == 0) ^ (dc == 0), "guide markers must be horizontal or vertical"
    assert dr >= 0 and dc >= 0, "row-major ordering fixes the forward direction"
    step = abs(dr) + abs(dc)
    assert step > 0
    if dr != 0:
        return (1, 0), step, 0
    return (0, 1), step, 1


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))

    guide = np.argwhere(g == GUIDE)
    assert len(guide) == 3, "expected exactly three guide markers"
    unit, step, axis = _order_triplet(guide)
    perp = 1 - axis

    valid_colors = {BG, TARGET, MARKER, MOVER, PUSHER, GUIDE}
    assert set(map(int, np.unique(g))).issubset(valid_colors)

    target_comps = _components((g == TARGET) | (g == MARKER))
    source_comps = _components(g == MOVER)
    pusher_comps = _components(g == PUSHER)
    assert len(target_comps) == 1
    assert len(source_comps) == 1
    assert len(pusher_comps) == 1

    target = {
        "coords": target_comps[0],
        "bbox": _bbox_from_coords(target_comps[0]),
        "colors": {TARGET, MARKER},
    }
    source = {
        "coords": source_comps[0],
        "bbox": _bbox_from_coords(source_comps[0]),
        "colors": {MOVER},
    }
    pusher = {
        "coords": pusher_comps[0],
        "bbox": _bbox_from_coords(pusher_comps[0]),
        "colors": {PUSHER},
    }

    src_patch = _crop(g, source["bbox"])
    push_patch = _crop(g, pusher["bbox"])
    target_patch = _crop(g, target["bbox"])

    assert source["colors"] == {MOVER}
    assert _solid_rect_patch(src_patch, MOVER)
    assert _span(source["bbox"], perp) == 2

    assert pusher["colors"] == {PUSHER}
    assert _solid_rect_patch(push_patch, PUSHER)

    assert target["colors"].issubset({TARGET, MARKER})
    assert MARKER in target["colors"]
    assert np.all(target_patch != BG)

    assert _center2(source["bbox"], perp) == _center2(target["bbox"], perp) == _center2(pusher["bbox"], perp)
    assert _center2(target["bbox"], axis) > _center2(source["bbox"], axis) > _center2(pusher["bbox"], axis)

    if axis == 1:
        gap = target["bbox"][1] - source["bbox"][3] - 1
        rel0 = source["bbox"][0] - target["bbox"][0]
        rel1 = source["bbox"][2] - target["bbox"][0]
    else:
        gap = target["bbox"][0] - source["bbox"][2] - 1
        rel0 = source["bbox"][1] - target["bbox"][1]
        rel1 = source["bbox"][3] - target["bbox"][1]

    assert gap == step, "guide spacing must equal source-to-target gap"
    assert gap >= 1

    low_overhang = rel0
    high_overhang = (_span(target["bbox"], perp) - 1) - rel1
    assert low_overhang == high_overhang and low_overhang >= 2

    return {
        "grid": g,
        "axis": axis,
        "perp": perp,
        "unit": unit,
        "step": step,
        "source": source,
        "target": target,
        "pusher": pusher,
        "target_patch": target_patch,
        "split_low": rel0,
        "split_high": rel1,
        "gap": gap,
    }


def _split_fragments(meta: dict) -> tuple[list[np.ndarray], list[int], int]:
    patch = meta["target_patch"]
    axis = meta["axis"]
    rel0 = meta["split_low"]
    rel1 = meta["split_high"]

    if axis == 1:
        first = patch[: rel0 + 1, :].copy()
        second = patch[rel1:, :].copy()
    else:
        first = patch[:, : rel0 + 1].copy()
        second = patch[:, rel1:].copy()

    counts = [int(np.count_nonzero(first == MARKER)), int(np.count_nonzero(second == MARKER))]
    assert min(counts) > 0
    assert counts[0] != counts[1]
    winner = 0 if counts[0] > counts[1] else 1
    return [first, second], counts, winner


def _draw_rays(crop: np.ndarray, unit: tuple[int, int], step: int) -> np.ndarray:
    out = crop.copy()
    markers = np.argwhere(crop == MARKER)
    h, w = crop.shape
    for r, c in markers:
        rr = int(r)
        cc = int(c)
        for dist in range(1, step + 1):
            nr = rr + dist * unit[0]
            nc = cc + dist * unit[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if out[nr, nc] != MARKER:
                out[nr, nc] = RAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the split target fragment with more markers, then extend marker rays using the guide-triplet step.

    Input structure:
    - Black background contains three axis-aligned foreground objects on one horizontal or vertical axis.
    - The middle object is a solid 2-cell-thick mover, one end object is a solid pusher, and the far end object is a marked solid target rectangle.
    - The target uses blue fill with red marker pixels inside it, and exactly three grey pixels form an equally spaced collinear guide triplet parallel to the object axis.

    Rule:
    - Read the grey triplet in row-major order to get the forward axis direction and the spacing D between consecutive guides.
    - Treat the middle rectangle as moving toward the marked target: its 2-cell path splits the target into two outward-shifted fragments, which corresponds to taking the two target sub-rectangles on opposite sides of the mover path.
    - Count red markers in the two fragments, crop the unique fragment with the larger count, and then from every red marker draw a maroon ray forward for D steps inside that crop, without overwriting red markers.

    Key insight:
    - The grey triplet is the shared parameter source: its direction fixes both the split orientation and the ray direction, and its spacing fixes both the source-target gap and the ray length.
    """
    meta = _analyze_scene(np.asarray(grid, dtype=int))
    fragments, _counts, winner = _split_fragments(meta)
    return _draw_rays(fragments[winner], meta["unit"], meta["step"])


def _target_patch_horizontal(k: int, width: int, winner_side: int, step: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((2 * k + 2, width), TARGET, dtype=int)
    counts = random.sample(range(1, k + 1), 2)
    if winner_side == 0:
        top_count, bottom_count = max(counts), min(counts)
    else:
        top_count, bottom_count = min(counts), max(counts)

    top_rows = random.sample(list(range(0, k)), top_count)
    bottom_rows = random.sample(list(range(k + 2, 2 * k + 2)), bottom_count)

    top_cols = [0] + [random.randint(0, width - 2) for _ in range(top_count - 1)]
    bottom_cols = [0] + [random.randint(0, width - 2) for _ in range(bottom_count - 1)]

    for r, c in zip(top_rows, top_cols):
        patch[r, c] = MARKER
    for r, c in zip(bottom_rows, bottom_cols):
        patch[r, c] = MARKER

    winner_counts = (top_count, bottom_count)
    assert max(winner_counts) > min(winner_counts)
    assert width >= step + 1
    return patch, winner_counts


def _target_patch_vertical(k: int, height: int, winner_side: int, step: int) -> tuple[np.ndarray, tuple[int, int]]:
    patch = np.full((height, 2 * k + 2), TARGET, dtype=int)
    counts = random.sample(range(1, k + 1), 2)
    if winner_side == 0:
        left_count, right_count = max(counts), min(counts)
    else:
        left_count, right_count = min(counts), max(counts)

    left_cols = random.sample(list(range(0, k)), left_count)
    right_cols = random.sample(list(range(k + 2, 2 * k + 2)), right_count)

    left_rows = [0] + [random.randint(0, height - 2) for _ in range(left_count - 1)]
    right_rows = [0] + [random.randint(0, height - 2) for _ in range(right_count - 1)]

    for r, c in zip(left_rows, left_cols):
        patch[r, c] = MARKER
    for r, c in zip(right_rows, right_cols):
        patch[r, c] = MARKER

    winner_counts = (left_count, right_count)
    assert max(winner_counts) > min(winner_counts)
    assert height >= step + 1
    return patch, winner_counts


def _place_triplet_horizontal(grid: np.ndarray, row: int, step: int) -> bool:
    h, w = grid.shape
    starts = list(range(0, w - 2 * step))
    random.shuffle(starts)
    for c0 in starts:
        cells = [(row, c0 + i * step) for i in range(3)]
        if all(grid[r, c] == BG for r, c in cells):
            for r, c in cells:
                grid[r, c] = GUIDE
            return True
    return False


def _place_triplet_vertical(grid: np.ndarray, col: int, step: int) -> bool:
    h, w = grid.shape
    starts = list(range(0, h - 2 * step))
    random.shuffle(starts)
    for r0 in starts:
        cells = [(r0 + i * step, col) for i in range(3)]
        if all(grid[r, c] == BG for r, c in cells):
            for r, c in cells:
                grid[r, c] = GUIDE
            return True
    return False


def _gen_horizontal() -> np.ndarray | None:
    step = random.randint(1, 4)
    k = random.choice([2, 3])
    source_w = random.randint(2, 4)
    target_w = random.randint(step + 1, min(7, step + 4))
    pusher_h = random.choice([2, 4])
    pusher_w = random.randint(2, 5)

    left_margin = random.randint(1, 3)
    right_margin = random.randint(source_w + 1, source_w + 4)
    min_w = left_margin + pusher_w + source_w + step + target_w + right_margin
    lo_w = max(14, min_w)
    hi_w = min(30, min_w + 5)
    if lo_w > hi_w:
        return None
    width = random.randint(lo_w, hi_w)

    target_h = 2 * k + 2
    lo_h = max(12, target_h + 3)
    hi_h = min(24, target_h + 8)
    if lo_h > hi_h:
        return None
    height = random.randint(lo_h, hi_h)
    target_r0 = random.randint(1, height - target_h - 2)
    mover_r0 = target_r0 + k
    pusher_r0 = mover_r0 + 1 - pusher_h // 2
    if pusher_r0 < 0 or pusher_r0 + pusher_h > height:
        return None

    pusher_c0 = left_margin
    mover_c0 = pusher_c0 + pusher_w
    target_c0 = mover_c0 + source_w + step
    if target_c0 + target_w > width - source_w:
        return None

    grid = np.full((height, width), BG, dtype=int)
    patch, counts = _target_patch_horizontal(k, target_w, random.randint(0, 1), step)

    grid[target_r0 : target_r0 + target_h, target_c0 : target_c0 + target_w] = patch
    grid[mover_r0 : mover_r0 + 2, mover_c0 : mover_c0 + source_w] = MOVER
    grid[pusher_r0 : pusher_r0 + pusher_h, pusher_c0 : pusher_c0 + pusher_w] = PUSHER

    guide_rows = [target_r0 - 1, target_r0 + target_h]
    random.shuffle(guide_rows)
    if not any(_place_triplet_horizontal(grid, row, step) for row in guide_rows):
        return None

    assert min(counts) > 0 and counts[0] != counts[1]
    return grid


def _gen_vertical() -> np.ndarray | None:
    step = random.randint(1, 4)
    k = random.choice([2, 3])
    source_h = random.randint(2, 4)
    target_h = random.randint(step + 1, min(7, step + 4))
    pusher_w = random.choice([2, 4])
    pusher_h = random.randint(2, 5)

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(source_h + 1, source_h + 4)
    min_h = top_margin + pusher_h + source_h + step + target_h + bottom_margin
    lo_h = max(14, min_h)
    hi_h = min(30, min_h + 5)
    if lo_h > hi_h:
        return None
    height = random.randint(lo_h, hi_h)

    target_w = 2 * k + 2
    lo_w = max(12, target_w + 3)
    hi_w = min(24, target_w + 8)
    if lo_w > hi_w:
        return None
    width = random.randint(lo_w, hi_w)
    target_c0 = random.randint(1, width - target_w - 2)
    mover_c0 = target_c0 + k
    pusher_c0 = mover_c0 + 1 - pusher_w // 2
    if pusher_c0 < 0 or pusher_c0 + pusher_w > width:
        return None

    pusher_r0 = top_margin
    mover_r0 = pusher_r0 + pusher_h
    target_r0 = mover_r0 + source_h + step
    if target_r0 + target_h > height - source_h:
        return None

    grid = np.full((height, width), BG, dtype=int)
    patch, counts = _target_patch_vertical(k, target_h, random.randint(0, 1), step)

    grid[target_r0 : target_r0 + target_h, target_c0 : target_c0 + target_w] = patch
    grid[mover_r0 : mover_r0 + source_h, mover_c0 : mover_c0 + 2] = MOVER
    grid[pusher_r0 : pusher_r0 + pusher_h, pusher_c0 : pusher_c0 + pusher_w] = PUSHER

    guide_cols = [target_c0 - 1, target_c0 + target_w]
    random.shuffle(guide_cols)
    if not any(_place_triplet_vertical(grid, col, step) for col in guide_cols):
        return None

    assert min(counts) > 0 and counts[0] != counts[1]
    return grid


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        meta = _analyze_scene(grid)
        fragments, counts, winner = _split_fragments(meta)
        expected = _draw_rays(fragments[winner], meta["unit"], meta["step"])
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if not np.any(expected == RAY):
        return False
    if np.array_equal(expected, fragments[winner]):
        return False
    if np.array_equal(expected, fragments[1 - winner]):
        return False
    wrong = _draw_rays(fragments[1 - winner], meta["unit"], meta["step"])
    if np.array_equal(expected, wrong):
        return False
    if counts[0] == counts[1] or min(counts) == 0:
        return False
    if meta["gap"] != meta["step"]:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        if random.random() < 0.5:
            grid = _gen_horizontal()
        else:
            grid = _gen_vertical()
        if grid is not None and _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

import itertools
import math
import random

import numpy as np


GREEN = 3
PALETTE = tuple(range(10))
INPUT_COLORS = tuple(color for color in PALETTE if color != GREEN)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

    return len(seen) == int(mask.sum())


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _background_color(grid)
    g = np.asarray(grid, dtype=int)
    objects = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        mask = g[r0 : r1 + 1, c0 : c1 + 1] == color
        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        )

    assert objects and len(objects) % 2 == 1
    shape = objects[0]["mask"].shape
    assert all(obj["mask"].shape == shape for obj in objects)
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["color"]))
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Vote across aligned same-size colored sprites to form a green consensus, then sort the originals by disagreement count.

    Input structure:
    - A dominant background surrounds several separated monochrome foreground objects.
    - Each non-background color belongs to exactly one object, and all object bboxes have the same height and width.
    - When those bbox-sized shapes are aligned, most pixel positions agree across the objects, with only a few per-object deviations.

    Rule:
    - Extract each object's bbox mask and align all masks to a common origin.
    - For every aligned pixel position, mark it as consensus foreground iff a strict majority of objects contain a foreground pixel there.
    - Draw that consensus mask in green at the far left of the output.
    - Count, for each original object, how many aligned pixels differ from the consensus mask; place the original objects from left to right in ascending disagreement count, with one background column between consecutive placed shapes.

    Key insight:
    - The colored objects are noisy votes for a shared template, and the output order is determined by each voter's Hamming distance from that majority-vote template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects = _extract_objects(g)
    h, w = objects[0]["mask"].shape
    masks = np.stack([obj["mask"] for obj in objects], axis=0)
    consensus = masks.sum(axis=0) > (len(objects) // 2)

    for obj in objects:
        obj["deviation"] = int(np.count_nonzero(obj["mask"] != consensus))

    ordered = sorted(
        objects,
        key=lambda obj: (obj["deviation"], obj["bbox"][1], obj["bbox"][0], obj["color"]),
    )

    out_w = (len(objects) + 1) * w + len(objects)
    out = np.full((h, out_w), bg, dtype=int)
    _paint_mask(out, consensus, 0, 0, GREEN)

    left = w + 1
    for obj in ordered:
        _paint_mask(out, obj["mask"], 0, left, obj["color"])
        left += w + 1

    return out


def _sample_bbox_dims(n_objects: int) -> tuple[int, int]:
    while True:
        if n_objects == 3:
            h = random.randint(3, 7)
            w = random.randint(3, 6)
            min_area = 10
        else:
            h = random.randint(4, 7)
            w = random.randint(3, 4)
            min_area = 14

        if h * w >= min_area and (n_objects + 1) * w + n_objects <= 30:
            return h, w


def _spanning_skeleton(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)

    col = random.randrange(w)
    vertical_cols = [col]
    mask[0, col] = True
    for r in range(1, h):
        col = max(0, min(w - 1, col + random.choice((-1, 0, 1))))
        vertical_cols.append(col)
        mask[r, col] = True

    anchor_r = random.randrange(h)
    row = anchor_r
    col = vertical_cols[anchor_r]
    while col > 0:
        row = max(0, min(h - 1, row + random.choice((-1, 0, 1))))
        col -= 1
        mask[row, col] = True

    row = anchor_r
    col = vertical_cols[anchor_r]
    while col < w - 1:
        row = max(0, min(h - 1, row + random.choice((-1, 0, 1))))
        col += 1
        mask[row, col] = True

    return mask


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((r, c))
                    break
    return cells


def _locked_positions(mask: np.ndarray) -> set[tuple[int, int]]:
    top_cols = list(np.flatnonzero(mask[0, :]))
    bottom_cols = list(np.flatnonzero(mask[-1, :]))
    left_rows = list(np.flatnonzero(mask[:, 0]))
    right_rows = list(np.flatnonzero(mask[:, -1]))

    return {
        (0, random.choice(top_cols)),
        (mask.shape[0] - 1, random.choice(bottom_cols)),
        (random.choice(left_rows), 0),
        (random.choice(right_rows), mask.shape[1] - 1),
    }


def _random_consensus_mask(h: int, w: int, min_capacity: int) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    for _ in range(80):
        mask = _spanning_skeleton(h, w)
        if not (_is_connected(mask) and _spans_bbox(mask)):
            continue

        base_area = int(mask.sum())
        area_lo = max(base_area + 1, math.ceil(0.35 * h * w))
        area_hi = min(h * w - 3, max(area_lo, math.floor(0.7 * h * w)))
        if area_hi < area_lo:
            continue

        target_area = random.randint(area_lo, area_hi)
        while int(mask.sum()) < target_area:
            frontier = _frontier(mask)
            if not frontier:
                break
            r, c = random.choice(frontier)
            mask[r, c] = True

        if not (_is_connected(mask) and _spans_bbox(mask)):
            continue

        locks = _locked_positions(mask)
        free_capacity = h * w - len(locks)
        if free_capacity < min_capacity:
            continue
        if int(mask.sum()) - len(locks) < 3:
            continue
        if int((~mask).sum()) < 3:
            continue
        return mask, locks

    return None


def _has_true_neighbor(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            return True
    return False


def _can_remove(mask: np.ndarray, pos: tuple[int, int], locks: set[tuple[int, int]]) -> bool:
    r, c = pos
    if not mask[r, c] or pos in locks:
        return False
    trial = mask.copy()
    trial[r, c] = False
    return _is_connected(trial)


def _can_add(mask: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    return (not mask[r, c]) and _has_true_neighbor(mask, r, c)


def _sample_deviation_counts(n_objects: int, free_capacity: int) -> tuple[int, ...] | None:
    options = [
        counts
        for counts in itertools.combinations(range(6), n_objects)
        if counts[-1] >= 2 and sum(counts) <= free_capacity
    ]
    if not options:
        return None
    return random.choice(options)


def _build_variant_masks(
    consensus: np.ndarray,
    deviation_counts: tuple[int, ...],
    locks: set[tuple[int, int]],
) -> list[np.ndarray] | None:
    h, w = consensus.shape
    all_positions = [(r, c) for r in range(h) for c in range(w)]
    available = set(all_positions)
    variants: list[np.ndarray | None] = [None] * len(deviation_counts)

    for idx, dev_count in sorted(enumerate(deviation_counts), key=lambda item: item[1], reverse=True):
        mask = consensus.copy()
        for _ in range(dev_count):
            removable = [pos for pos in available if _can_remove(mask, pos, locks)]
            addable = [pos for pos in available if _can_add(mask, pos)]

            pools = []
            if removable:
                pools.append(removable)
            if addable:
                pools.append(addable)
            if not pools:
                return None

            pool = random.choice(pools)
            pos = random.choice(pool)
            mask[pos] = not mask[pos]
            available.remove(pos)

        if not (_is_connected(mask) and _spans_bbox(mask)):
            return None
        variants[idx] = mask

    built = [mask for mask in variants if mask is not None]
    assert len(built) == len(deviation_counts)
    return built


def _sample_layout(n_objects: int, h: int, w: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    row_choices = [1, 2] if n_objects == 3 else [2, 3]

    for _ in range(80):
        rows = random.choice(row_choices)
        cols = math.ceil(n_objects / rows)

        row_heights = [h + random.randint(0, 3) for _ in range(rows)]
        col_widths = [w + random.randint(0, 4) for _ in range(cols)]
        top = random.randint(1, 4)
        bottom = random.randint(1, 4)
        left = random.randint(1, 4)
        right = random.randint(1, 4)
        row_gaps = [random.randint(1, 3) for _ in range(rows - 1)]
        col_gaps = [random.randint(1, 4) for _ in range(cols - 1)]

        height = top + sum(row_heights) + sum(row_gaps) + bottom
        width = left + sum(col_widths) + sum(col_gaps) + right
        if height > 30 or width > 30:
            continue

        row_starts = []
        cursor = top
        for i, band in enumerate(row_heights):
            row_starts.append(cursor)
            cursor += band
            if i < len(row_gaps):
                cursor += row_gaps[i]

        col_starts = []
        cursor = left
        for j, band in enumerate(col_widths):
            col_starts.append(cursor)
            cursor += band
            if j < len(col_gaps):
                cursor += col_gaps[j]

        slots = []
        for i, band_h in enumerate(row_heights):
            for j, band_w in enumerate(col_widths):
                r0 = row_starts[i] + random.randint(0, band_h - h)
                c0 = col_starts[j] + random.randint(0, band_w - w)
                slots.append((r0, c0))

        return height, width, random.sample(slots, n_objects)

    return None


def _build_expected_output(bg: int, consensus: np.ndarray, objects: list[dict]) -> np.ndarray:
    h, w = consensus.shape
    out = np.full((h, (len(objects) + 1) * w + len(objects)), bg, dtype=int)
    _paint_mask(out, consensus, 0, 0, GREEN)

    left = w + 1
    for obj in sorted(objects, key=lambda obj: (obj["deviation"], obj["pos"][1], obj["pos"][0], obj["color"])):
        _paint_mask(out, obj["mask"], 0, left, obj["color"])
        left += w + 1
    return out


def _source_already_sorted(objects: list[dict]) -> bool:
    source_order = [obj["id"] for obj in sorted(objects, key=lambda obj: (obj["pos"][1], obj["pos"][0]))]
    deviation_order = [obj["id"] for obj in sorted(objects, key=lambda obj: obj["deviation"])]
    return source_order == deviation_order


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice((3, 5))
    h, w = _sample_bbox_dims(n_objects)

    min_capacity = 10 if n_objects == 5 else 5
    consensus_data = _random_consensus_mask(h, w, min_capacity=min_capacity)
    if consensus_data is None:
        return None
    consensus, locks = consensus_data

    deviation_counts = _sample_deviation_counts(n_objects, h * w - len(locks))
    if deviation_counts is None:
        return None

    variant_masks = _build_variant_masks(consensus, deviation_counts, locks)
    if variant_masks is None:
        return None

    stack = np.stack(variant_masks, axis=0)
    if not np.array_equal(stack.sum(axis=0) > (n_objects // 2), consensus):
        return None

    bg = random.choice(INPUT_COLORS)
    colors = random.sample([color for color in INPUT_COLORS if color != bg], n_objects)

    layout = _sample_layout(n_objects, h, w)
    if layout is None:
        return None
    height, width, positions = layout

    objects = []
    for idx, (mask, color, pos) in enumerate(zip(variant_masks, colors, positions)):
        objects.append(
            {
                "id": idx,
                "mask": mask,
                "color": int(color),
                "deviation": int(np.count_nonzero(mask != consensus)),
                "pos": pos,
            }
        )

    if len({obj["deviation"] for obj in objects}) != n_objects:
        return None
    if _source_already_sorted(objects):
        return None

    grid = np.full((height, width), bg, dtype=int)
    for obj in objects:
        _paint_mask(grid, obj["mask"], obj["pos"][0], obj["pos"][1], obj["color"])

    if 2 * int(np.count_nonzero(grid != bg)) >= grid.size:
        return None

    expected = _build_expected_output(bg, consensus, objects)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
MAROON = 9
PALETTE = (2, 3, 4, 5, 6, 7, 8)

MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400

LAYOUTS_BY_PAIR_COUNT = {
    2: [(2, 3), (3, 2), (1, 6), (6, 1)],
    3: [(3, 3)],
    4: [(3, 4), (4, 3), (2, 6), (6, 2)],
}


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    vals = [int(x) for x in indices]
    assert vals, "expected at least one non-separator band"
    out: list[tuple[int, int]] = []
    start = prev = vals[0]
    for value in vals[1:]:
        if value == prev + 1:
            prev = value
        else:
            out.append((start, prev))
            start = prev = value
    out.append((start, prev))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        moves = []
        if r < er:
            moves.append((1, 0))
        elif r > er:
            moves.append((-1, 0))
        if c < ec:
            moves.append((0, 1))
        elif c > ec:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        mask[r, c] = True


def _grow_connected(mask: np.ndarray, target_area: int) -> bool:
    h, w = mask.shape
    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for rr, cc in _neighbors4(int(r), int(c), h, w):
                if not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            return False
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True
    return True


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _random_connected_subset(allowed: np.ndarray, size: int) -> np.ndarray | None:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
    if len(coords) < size:
        return None
    h, w = allowed.shape
    for _ in range(40):
        start = random.choice(coords)
        chosen = {start}
        frontier = {
            nbr
            for nbr in _neighbors4(start[0], start[1], h, w)
            if allowed[nbr]
        }
        while len(chosen) < size:
            frontier -= chosen
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            chosen.add(cell)
            for nbr in _neighbors4(cell[0], cell[1], h, w):
                if allowed[nbr] and nbr not in chosen:
                    frontier.add(nbr)
        if len(chosen) == size:
            out = np.zeros_like(allowed, dtype=bool)
            for r, c in chosen:
                out[r, c] = True
            return out
    return None


def _generate_full_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        current = random.choice(anchors)
        for anchor in random.sample(anchors, len(anchors)):
            _random_path(mask, current, anchor)
            current = anchor

        min_target = max(int(mask.sum()) + 1, int(np.ceil(area * 0.45)))
        max_target = area - 1
        if min_target > max_target:
            continue
        target_area = random.randint(min_target, max_target)
        if not _grow_connected(mask, target_area):
            continue
        if not _touches_all_edges(mask):
            continue
        if height >= 3 and width >= 3 and not mask[1:-1, 1:-1].any():
            continue
        if mask.all():
            continue
        return mask
    return None


def _split_regions(grid: np.ndarray) -> list[dict[str, object]]:
    blue_rows = np.all(grid == BLUE, axis=1)
    blue_cols = np.all(grid == BLUE, axis=0)
    row_bands = _bands(np.where(~blue_rows)[0])
    col_bands = _bands(np.where(~blue_cols)[0])
    regions: list[dict[str, object]] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(region == BLUE), "blue pixels should only appear on separators"
            colors = set(map(int, np.unique(region))) - {BLACK, BLUE}
            assert len(colors) == 1, "each region must carry exactly one semantic color"
            color = colors.pop()
            regions.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "grid": region,
                    "color": color,
                }
            )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the missing shape from each complete/damaged sample pair and draw it
    in maroon at the center of the matching solid-color canvas region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _split_regions(g)

    by_color: dict[int, list[dict[str, object]]] = {}
    for region in regions:
        color = int(region["color"])
        by_color.setdefault(color, []).append(region)

    for color, triple in by_color.items():
        assert len(triple) == 3, f"color {color} must appear in exactly three regions"
        canvases = [region for region in triple if np.all(region["grid"] == color)]
        assert len(canvases) == 1, f"color {color} must have exactly one canvas region"
        canvas = canvases[0]
        samples = [region for region in triple if region is not canvas]
        assert len(samples) == 2

        counts = [int(np.sum(region["grid"] == color)) for region in samples]
        assert counts[0] != counts[1], "complete and damaged samples must differ"
        full = samples[0] if counts[0] > counts[1] else samples[1]
        damaged = samples[1] if full is samples[0] else samples[0]

        full_mask = np.asarray(full["grid"] == color, dtype=bool)
        damaged_mask = np.asarray(damaged["grid"] == color, dtype=bool)
        full_crop = _crop_to_bbox(full_mask)
        damaged_crop = _crop_to_bbox(damaged_mask)
        assert full_crop.shape == damaged_crop.shape, "aligned sample bboxes must match"
        assert np.all(damaged_crop <= full_crop), "damaged sample must be a subset of the full sample"

        missing = full_crop & ~damaged_crop
        missing = _crop_to_bbox(missing)
        canvas_grid = np.asarray(canvas["grid"])
        ch, cw = canvas_grid.shape
        mh, mw = missing.shape
        assert mh <= ch and mw <= cw, "canvas must be able to contain the missing shape"
        top = (ch - mh) // 2
        left = (cw - mw) // 2
        r0, c0, _, _ = canvas["bbox"]
        patch = out[r0 : r0 + ch, c0 : c0 + cw]
        patch[top : top + mh, left : left + mw][missing] = MAROON

    return out


def _sample_band_sizes(count: int) -> list[int] | None:
    if count >= 6:
        min_size = 4
    elif count >= 4:
        min_size = 5
    elif count >= 2:
        min_size = 6
    else:
        min_size = 8
    max_size = min(10, (MAX_GRID_SIDE - (count - 1)) // count)
    if max_size < min_size:
        return None
    return [random.randint(min_size, max_size) for _ in range(count)]


def _band_starts(sizes: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts


def _region_boxes(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    row_starts = _band_starts(row_sizes)
    col_starts = _band_starts(col_sizes)
    for r0, rh in zip(row_starts, row_sizes, strict=False):
        for c0, cw in zip(col_starts, col_sizes, strict=False):
            boxes.append((r0, c0, rh, cw))
    return boxes


def _paint_mask(region: np.ndarray, mask: np.ndarray, color: int, offset_r: int, offset_c: int) -> None:
    sub = region[offset_r : offset_r + mask.shape[0], offset_c : offset_c + mask.shape[1]]
    sub[mask] = color


def _build_group_regions(
    full_box: tuple[int, int, int, int],
    damaged_box: tuple[int, int, int, int],
    canvas_box: tuple[int, int, int, int],
    color: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    _, _, full_h, full_w = full_box
    _, _, damaged_h, damaged_w = damaged_box
    _, _, canvas_h, canvas_w = canvas_box

    max_bbox_h = min(full_h, damaged_h, canvas_h + 2)
    max_bbox_w = min(full_w, damaged_w, canvas_w + 2)
    if max_bbox_h < 3 or max_bbox_w < 3:
        return None

    bbox_h_choices = [dim for dim in range(3, max_bbox_h + 1)]
    bbox_w_choices = [dim for dim in range(3, max_bbox_w + 1)]
    for _ in range(80):
        bbox_h = random.choice(bbox_h_choices)
        bbox_w = random.choice(bbox_w_choices)
        full_mask = _generate_full_mask(bbox_h, bbox_w)
        if full_mask is None:
            continue

        interior_allowed = np.zeros_like(full_mask, dtype=bool)
        interior_allowed[1:-1, 1:-1] = full_mask[1:-1, 1:-1]
        interior_count = int(interior_allowed.sum())
        if interior_count == 0:
            continue
        max_missing = min(interior_count, max(1, int(full_mask.sum() * 0.4)))
        missing_size = random.randint(1, max_missing)
        missing_mask = _random_connected_subset(interior_allowed, missing_size)
        if missing_mask is None:
            continue

        damaged_mask = full_mask & ~missing_mask
        if not damaged_mask.any():
            continue
        if _crop_to_bbox(full_mask).shape != _crop_to_bbox(damaged_mask).shape:
            continue
        missing_crop = _crop_to_bbox(missing_mask)
        miss_h, miss_w = missing_crop.shape
        if miss_h > canvas_h or miss_w > canvas_w:
            continue

        full_region = np.full((full_h, full_w), BLACK, dtype=int)
        damaged_region = np.full((damaged_h, damaged_w), BLACK, dtype=int)
        canvas_region = np.full((canvas_h, canvas_w), color, dtype=int)
        canvas_out = canvas_region.copy()

        full_r = random.randint(0, full_h - bbox_h)
        full_c = random.randint(0, full_w - bbox_w)
        damaged_r = random.randint(0, damaged_h - bbox_h)
        damaged_c = random.randint(0, damaged_w - bbox_w)
        _paint_mask(full_region, full_mask, color, full_r, full_c)
        _paint_mask(damaged_region, damaged_mask, color, damaged_r, damaged_c)

        top = (canvas_h - miss_h) // 2
        left = (canvas_w - miss_w) // 2
        canvas_out[top : top + miss_h, left : left + miss_w][missing_crop] = MAROON
        return full_region, damaged_region, canvas_region, canvas_out
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice(sorted(LAYOUTS_BY_PAIR_COUNT))
    rows, cols = random.choice(LAYOUTS_BY_PAIR_COUNT[pair_count])
    row_sizes = _sample_band_sizes(rows)
    col_sizes = _sample_band_sizes(cols)
    if row_sizes is None or col_sizes is None:
        return None

    height = sum(row_sizes) + max(0, rows - 1)
    width = sum(col_sizes) + max(0, cols - 1)
    if height > MAX_GRID_SIDE or width > MAX_GRID_SIDE:
        return None

    boxes = _region_boxes(row_sizes, col_sizes)
    assert len(boxes) == 3 * pair_count
    random.shuffle(boxes)
    colors = random.sample(PALETTE, pair_count)

    inp = np.full((height, width), BLUE, dtype=int)
    out = inp.copy()

    for i, color in enumerate(colors):
        triple = boxes[3 * i : 3 * i + 3]
        triple = sorted(triple, key=lambda box: box[2] * box[3], reverse=True)
        sample_a, sample_b, canvas_box = triple
        if random.random() < 0.5:
            sample_a, sample_b = sample_b, sample_a

        regions = _build_group_regions(sample_a, sample_b, canvas_box, color)
        if regions is None:
            return None
        full_region, damaged_region, canvas_region, canvas_out = regions

        for box, patch in (
            (sample_a, full_region),
            (sample_b, damaged_region),
            (canvas_box, canvas_region),
        ):
            r0, c0, rh, cw = box
            inp[r0 : r0 + rh, c0 : c0 + cw] = patch
            out[r0 : r0 + rh, c0 : c0 + cw] = patch

        r0, c0, rh, cw = canvas_box
        out[r0 : r0 + rh, c0 : c0 + cw] = canvas_out

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")

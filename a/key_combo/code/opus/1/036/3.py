from __future__ import annotations

from collections import deque
import random

import numpy as np


_NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GENERATION_RETRIES = 500


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS_8:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _compress_rows(crop: np.ndarray) -> np.ndarray:
    kept = [crop[0].copy()]
    for row in crop[1:]:
        if not np.array_equal(row, kept[-1]):
            kept.append(row.copy())
    return np.stack(kept, axis=0)


def _assemble_bottom_aligned(crops: list[np.ndarray], bg: int = 0) -> np.ndarray:
    max_h = max(crop.shape[0] for crop in crops)
    total_w = sum(crop.shape[1] for crop in crops)
    out = np.full((max_h, total_w), bg, dtype=int)

    left = 0
    for crop in crops:
        h, w = crop.shape
        out[max_h - h :, left : left + w] = crop
        left += w

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop each sprite, compress duplicate row runs inside each crop, and concatenate by compressed height.

    Input structure:
    - A black background contains multiple separated monochrome 8-connected sprites.
    - The sprites are spread horizontally, and each sprite is read only inside its own bounding box.
    - Adjacent rows inside a sprite crop may repeat exactly, so different sprites can hide different row-run patterns.

    Rule:
    - Find every 8-connected foreground sprite and crop it to its bounding box.
    - For each crop, scan rows from top to bottom and merge every maximal run of identical adjacent rows into one retained row.
    - Sort the compressed crops by compressed height ascending, breaking ties by the original bbox left column.
    - Concatenate the sorted compressed crops from left to right with no gap, bottom-aligning them and padding above shorter crops with black.

    Key insight:
    - The raw sprite sizes do not determine the order; only per-sprite row compression reveals the sortable heights.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all(g >= 0)

    components = _find_components(g, bg=0)
    assert components

    items = []
    for comp in components:
        bbox = tuple(comp["bbox"])
        crop = _crop_bbox(g, bbox)
        compressed = _compress_rows(crop)
        items.append((bbox, compressed))

    ordered = sorted(items, key=lambda item: (item[1].shape[0], item[0][1]))
    return _assemble_bottom_aligned([compressed for _bbox, compressed in ordered], bg=0)


def _interval_catalog(width: int) -> list[tuple[int, int]]:
    return [(start, end) for start in range(width) for end in range(start, width)]


def _intervals_touch(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] + 1 and b[0] <= a[1] + 1


def _random_interval_sequence(width: int, compressed_h: int) -> list[tuple[int, int]] | None:
    patterns = _interval_catalog(width)
    for _ in range(250):
        seq = [random.choice(patterns)]
        while len(seq) < compressed_h:
            prev = seq[-1]
            choices = [pat for pat in patterns if pat != prev and _intervals_touch(prev, pat)]
            if not choices:
                break
            seq.append(random.choice(choices))
        if len(seq) != compressed_h:
            continue
        if min(start for start, _end in seq) != 0:
            continue
        if max(end for _start, end in seq) != width - 1:
            continue
        if len(set(seq)) < 2:
            continue
        return seq
    return None


def _random_run_lengths(raw_h: int, compressed_h: int) -> list[int]:
    cuts = sorted(random.sample(range(1, raw_h), compressed_h - 1))
    edges = [0, *cuts, raw_h]
    return [edges[i + 1] - edges[i] for i in range(compressed_h)]


def _make_sprite(width: int, raw_h: int, compressed_h: int, color: int) -> np.ndarray | None:
    for _ in range(250):
        seq = _random_interval_sequence(width, compressed_h)
        if seq is None:
            continue

        runs = _random_run_lengths(raw_h, compressed_h)
        rows = []
        for interval, run in zip(seq, runs):
            rows.extend([interval] * run)

        crop = np.zeros((raw_h, width), dtype=int)
        for r, (start, end) in enumerate(rows):
            crop[r, start : end + 1] = color

        comps = _find_components(crop, bg=0)
        if len(comps) != 1:
            continue
        if tuple(comps[0]["bbox"]) != (0, 0, raw_h - 1, width - 1):
            continue
        if _compress_rows(crop).shape[0] != compressed_h:
            continue
        return crop

    return None


def _sample_left_positions(count: int, width: int) -> tuple[int, list[int]] | None:
    min_grid_w = count * width + (count - 1)
    if min_grid_w > 30:
        return None

    max_extra = min(30 - min_grid_w, (count - 1) * 2 + 2)
    grid_w = min_grid_w + random.randint(0, max_extra)
    slack = grid_w - min_grid_w

    buckets = [0] * (count + 1)
    for _ in range(slack):
        buckets[random.randrange(count + 1)] += 1

    lefts = [buckets[0]]
    for i in range(1, count):
        gap = 1 + buckets[i]
        lefts.append(lefts[-1] + width + gap)

    return grid_w, lefts


def _sort_indices(items: list[dict], key_name: str, reverse: bool = False) -> tuple[int, ...]:
    if reverse:
        return tuple(
            idx
            for idx, _item in sorted(
                enumerate(items),
                key=lambda pair: (-pair[1][key_name], pair[1]["left"]),
            )
        )
    return tuple(
        idx
        for idx, _item in sorted(
            enumerate(items),
            key=lambda pair: (pair[1][key_name], pair[1]["left"]),
        )
    )


def _validate_instance(inp: np.ndarray, items: list[dict], expected: np.ndarray) -> bool:
    if len(items) not in (4, 5):
        return False

    widths = {item["width"] for item in items}
    raw_heights = {item["raw_height"] for item in items}
    compressed_heights = [item["compressed_height"] for item in items]
    if len(widths) != 1 or len(raw_heights) != 1:
        return False
    if len(set(compressed_heights)) != len(compressed_heights):
        return False
    if min(compressed_heights) < 3:
        return False
    if max(compressed_heights) >= next(iter(raw_heights)):
        return False

    if compressed_heights == sorted(compressed_heights):
        return False
    if compressed_heights == sorted(compressed_heights, reverse=True):
        return False

    target_order = _sort_indices(items, "compressed_height")
    if target_order in {
        tuple(range(len(items))),
        tuple(range(len(items) - 1, -1, -1)),
        _sort_indices(items, "color"),
        _sort_indices(items, "color", reverse=True),
        _sort_indices(items, "area"),
        _sort_indices(items, "area", reverse=True),
        _sort_indices(items, "top"),
        _sort_indices(items, "top", reverse=True),
    }:
        return False

    if len(set(item["top"] for item in items)) < 2:
        return False

    raw_ltr = _assemble_bottom_aligned([item["crop"] for item in items])
    compressed_ltr = _assemble_bottom_aligned([item["compressed"] for item in items])
    compressed_rtl = _assemble_bottom_aligned([item["compressed"] for item in items[::-1]])
    raw_sorted = _assemble_bottom_aligned([items[idx]["crop"] for idx in target_order])

    if any(np.array_equal(expected, wrong) for wrong in (raw_ltr, compressed_ltr, compressed_rtl, raw_sorted)):
        return False

    if len(_find_components(inp, bg=0)) != len(items):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([4, 5])
    raw_h = random.choice([8, 9]) if count == 5 else random.choice([7, 8, 9])
    width = 3 if count == 5 else random.choice([3, 4])

    possible_compressed = list(range(3, raw_h))
    if len(possible_compressed) < count:
        return None

    compressed_choices = sorted(random.sample(possible_compressed, count))
    random.shuffle(compressed_choices)

    colors = random.sample(_COLOR_POOL, count)
    crops = []
    for compressed_h, color in zip(compressed_choices, colors):
        crop = _make_sprite(width, raw_h, compressed_h, color)
        if crop is None:
            return None
        crops.append(crop)

    placement = _sample_left_positions(count, width)
    if placement is None:
        return None
    grid_w, lefts = placement

    grid_h = raw_h + random.randint(2, 6)
    if grid_h > 30:
        return None
    max_top = grid_h - raw_h
    tops = [random.randint(0, max_top) for _ in range(count)]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    items = []
    for crop, color, top, left in zip(crops, colors, tops, lefts):
        inp[top : top + raw_h, left : left + width] = crop
        compressed = _compress_rows(crop)
        items.append(
            {
                "color": color,
                "top": top,
                "left": left,
                "area": int(np.count_nonzero(crop)),
                "width": int(width),
                "raw_height": int(raw_h),
                "compressed_height": int(compressed.shape[0]),
                "crop": crop,
                "compressed": compressed,
            }
        )

    order = _sort_indices(items, "compressed_height")
    expected = _assemble_bottom_aligned([items[idx]["compressed"] for idx in order])

    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_instance(inp, items, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

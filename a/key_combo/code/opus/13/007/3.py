import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

INPUT_COLORS = [1, 2, 6, 7]
FILL_BY_PARITY = {1: 3, 0: 4}  # odd rank -> green, even rank -> yellow
OUTLINE_COLOR = 5
BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 400


TEMPLATES = {
    0: [
        np.array(
            [
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=int,
        ),
    ],
    1: [
        np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 0, 0],
            ],
            dtype=int,
        ),
    ],
    2: [
        np.array(
            [
                [0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=int,
        ),
    ],
    3: [
        np.array(
            [
                [0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=int,
        ),
    ],
}


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_masks(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    deltas = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in deltas:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.zeros((h, w), dtype=bool)
            rr, cc = zip(*coords)
            comp[np.array(rr), np.array(cc)] = True
            out.append(comp)
    return out


def _hole_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    bg = ~mask
    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                stack.append((nr, nc))
    return _component_masks(bg & ~reachable, connectivity=4)


def _object_records(grid: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BACKGROUND):
        for comp in _component_masks(grid == color, connectivity=4):
            coords = np.argwhere(comp)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            obj_mask = comp[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            holes = _hole_masks(obj_mask)
            records.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": obj_mask.copy(),
                    "holes": [h.copy() for h in holes],
                }
            )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder objects by enclosed-hole count, then fill holes by sorted parity and add gray outlines.

    Input structure:
    - The grid uses black as background and contains 3-4 separated single-color foreground objects.
    - Each object is a 4-connected monochrome component cropped by its own minimal bounding box.
    - Objects may enclose 0-3 black holes, and the hole counts are distinct across objects.

    Rule:
    - Count each object's enclosed 4-connected background regions inside its bounding box.
    - Sort the objects by that hole count from fewest to most, then place their cropped bounding boxes left-to-right, bottom-aligned, with one background column between neighbors and a one-cell outer margin.
    - Fill the holes of sorted odd-position objects with green, the holes of sorted even-position objects with yellow, and finally draw a one-pixel gray outline on background cells that 8-neighbor each placed object-plus-filled-holes region.

    Key insight:
    - The same topological measurement serves two roles at once: it determines the packing order and selects the regions whose fill color is decided by the post-sort position parity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    records = _object_records(g)
    assert 3 <= len(records) <= 4

    hole_counts = [len(rec["holes"]) for rec in records]
    assert len(set(hole_counts)) == len(hole_counts)

    ordered = sorted(records, key=lambda rec: len(rec["holes"]))
    max_h = max(int(rec["mask"].shape[0]) for rec in ordered)
    total_w = sum(int(rec["mask"].shape[1]) for rec in ordered)
    out_h = max_h + 2
    out_w = total_w + (len(ordered) - 1) + 2
    out = np.zeros((out_h, out_w), dtype=int)

    col = 1
    for rank, rec in enumerate(ordered, start=1):
        mask = rec["mask"]
        h, w = mask.shape
        row = out_h - 1 - h
        out[row : row + h, col : col + w][mask] = int(rec["color"])
        fill_color = FILL_BY_PARITY[rank % 2]
        for hole in rec["holes"]:
            out[row : row + h, col : col + w][hole] = fill_color
        col += w
        if rank != len(ordered):
            col += 1

    solids = out != BACKGROUND
    outline = np.zeros_like(solids)
    for r, c in np.argwhere(solids):
        for dr, dc in NEIGHBORS8:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < out_h and 0 <= nc < out_w and out[nr, nc] == BACKGROUND:
                outline[nr, nc] = True
    out[outline] = OUTLINE_COLOR
    return out


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _is_single_component(mask: np.ndarray) -> bool:
    return len(_component_masks(mask, connectivity=4)) == 1


def _valid_object_mask(mask: np.ndarray, expected_holes: int) -> bool:
    if not np.any(mask):
        return False
    mask = _crop_mask(mask)
    h, w = mask.shape
    area = _mask_area(mask)
    if not (10 <= area <= 25):
        return False
    if h > 8 or w > 7:
        return False
    if area == h * w:
        return False
    if not _is_single_component(mask):
        return False
    return len(_hole_masks(mask)) == expected_holes


def _mutate_mask(mask: np.ndarray, expected_holes: int) -> np.ndarray:
    cur = mask.copy()
    for _ in range(random.randint(1, 4)):
        candidates = []
        h, w = cur.shape
        for r in range(h):
            for c in range(w):
                trial = cur.copy()
                trial[r, c] = not trial[r, c]
                if not np.any(trial):
                    continue
                if _valid_object_mask(trial, expected_holes):
                    candidates.append(_crop_mask(trial))
        if candidates and random.random() < 0.75:
            cur = random.choice(candidates)
    return _crop_mask(cur)


def _sample_object_mask(hole_count: int) -> np.ndarray:
    for _ in range(120):
        mask = random.choice(TEMPLATES[hole_count]).astype(bool)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = _mutate_mask(mask, hole_count)
        if _valid_object_mask(mask, hole_count):
            return _crop_mask(mask)
    raise RuntimeError(f"failed to sample object with {hole_count} holes")


def _choose_hole_counts(n_objects: int) -> list[int]:
    if n_objects == 4:
        return [0, 1, 2, 3]
    options = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]
    return random.choice(options).copy()


def _place_input_objects(objects: list[dict]) -> np.ndarray | None:
    for _ in range(120):
        shuffled = objects[:]
        random.shuffle(shuffled)
        if [obj["holes"] for obj in shuffled] == sorted(obj["holes"] for obj in shuffled):
            continue

        widths = [int(obj["mask"].shape[1]) for obj in shuffled]
        heights = [int(obj["mask"].shape[0]) for obj in shuffled]
        gaps = [random.randint(1, 3) for _ in range(len(shuffled) - 1)]
        min_w = sum(widths) + sum(gaps)
        max_extra_w = 30 - min_w
        if max_extra_w < 0:
            continue
        left = random.randint(0, min(2, max_extra_w))
        right_cap = max_extra_w - left
        right = random.randint(0, min(2, right_cap))
        w = min_w + left + right

        min_h = max(heights)
        h_lo = max(min_h, 9)
        h_hi = min(16, min_h + 5)
        if h_lo > h_hi:
            continue
        h = random.randint(h_lo, h_hi)

        grid = np.zeros((h, w), dtype=int)
        col = left
        for idx, obj in enumerate(shuffled):
            mask = obj["mask"]
            oh, ow = mask.shape
            row = random.randint(0, h - oh)
            grid[row : row + oh, col : col + ow][mask] = int(obj["color"])
            col += ow
            if idx != len(shuffled) - 1:
                col += gaps[idx]

        records = sorted(_object_records(grid), key=lambda rec: rec["bbox"][1])
        left_to_right = [len(rec["holes"]) for rec in records]
        if left_to_right == sorted(left_to_right):
            continue
        return grid
    return None


def _generate_attempt() -> np.ndarray | None:
    n_objects = random.choice([3, 4])
    hole_counts = _choose_hole_counts(n_objects)
    colors = random.sample(INPUT_COLORS, n_objects)

    objects = []
    for hole_count, color in zip(hole_counts, colors):
        mask = _sample_object_mask(hole_count)
        objects.append({"holes": hole_count, "color": color, "mask": mask})

    grid = _place_input_objects(objects)
    if grid is None:
        return None

    records = _object_records(grid)
    counts = sorted(len(rec["holes"]) for rec in records)
    if counts != sorted(hole_counts):
        return None
    if len(set(counts)) != len(counts):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not np.any(out == 3) or not np.any(out == 4) or not np.any(out == 5):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

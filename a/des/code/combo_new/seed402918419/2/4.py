from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GROUP_ORDER = ("both", "horizontal", "vertical")
PALETTE = tuple(range(1, 10))
ODD_DIMS = (3, 5, 7)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_SHAPE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _cells_to_mask(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    raise AssertionError("component mask must have at least one symmetry axis")


def _reflect_horizontal(seed: np.ndarray) -> np.ndarray:
    return np.vstack([seed[:-1], seed[::-1]])


def _reflect_vertical(seed: np.ndarray) -> np.ndarray:
    return np.hstack([seed[:, :-1], seed[:, ::-1]])


def _grow_connected_cells(
    height: int,
    width: int,
    anchor: tuple[int, int],
    min_cells: int,
    max_cells: int,
) -> set[tuple[int, int]] | None:
    target = random.randint(min_cells, max_cells)
    cells = {anchor}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _sample_symmetric_mask(kind: str) -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.choice(ODD_DIMS)
        w = random.choice(ODD_DIMS)

        if kind == "horizontal":
            seed_h = h // 2 + 1
            anchor = (seed_h - 1, random.randrange(w))
            cells = _grow_connected_cells(seed_h, w, anchor, max(2, seed_h), seed_h * w)
            if cells is None:
                continue
            full = _reflect_horizontal(_cells_to_mask(cells, (seed_h, w)))
        elif kind == "vertical":
            seed_w = w // 2 + 1
            anchor = (random.randrange(h), seed_w - 1)
            cells = _grow_connected_cells(h, seed_w, anchor, max(2, seed_w), h * seed_w)
            if cells is None:
                continue
            full = _reflect_vertical(_cells_to_mask(cells, (h, seed_w)))
        elif kind == "both":
            seed_h = h // 2 + 1
            seed_w = w // 2 + 1
            anchor = (seed_h - 1, seed_w - 1)
            cells = _grow_connected_cells(seed_h, seed_w, anchor, 2, seed_h * seed_w)
            if cells is None:
                continue
            full = _reflect_vertical(_reflect_horizontal(_cells_to_mask(cells, (seed_h, seed_w))))
        else:
            raise ValueError(kind)

        cropped = _crop_mask(full)
        if min(cropped.shape) < 2 or int(cropped.sum()) < 4:
            continue
        if not _is_connected_mask(cropped):
            continue
        if _classify_mask(cropped) != kind:
            continue
        return cropped
    raise RuntimeError(f"failed to sample a {kind} object")


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == BG:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != BG:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            for rr, cc in coords:
                patch[rr - r0, cc - c0] = g[rr, cc]

            comps.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                    "mask": patch != BG,
                }
            )
    return comps


def _build_row(patches: list[np.ndarray]) -> np.ndarray:
    row_h = max(int(p.shape[0]) for p in patches)
    row_w = sum(int(p.shape[1]) for p in patches)
    row = np.zeros((row_h, row_w), dtype=int)
    col = 0
    for patch in patches:
        h, w = patch.shape
        row[:h, col : col + w] = patch
        col += w
    return row


def main(grid: np.ndarray) -> np.ndarray:
    """Group scattered objects by reflection symmetry and pack each symmetry class into one row.

    Input structure:
    - Background is black.
    - Several small 4-connected non-black objects are scattered on the grid and do not touch one another.
    - Each object has exactly one of three symmetry types: horizontal-only, vertical-only, or both horizontal and vertical.
    - The same number of objects appears in each symmetry type.

    Rule:
    - Extract every non-black connected component and classify it by the symmetry of its occupancy mask, ignoring color.
    - Within each symmetry class, keep the input reading order (top-to-bottom, then left-to-right) and place the cropped object boxes side by side with no gap; objects are top-aligned inside their merged row.
    - Stack the three merged rows in the fixed order both-axes, horizontal-only, vertical-only, inserting one all-black row between neighboring groups and padding shorter rows on the right with black.

    Key insight:
    - The colors and original positions are distractions; the decisive hidden feature is each object's mirror symmetry, which determines its group before any packing happens.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g)
    assert comps, "expected at least one object"
    comps.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))

    grouped: dict[str, list[np.ndarray]] = {kind: [] for kind in GROUP_ORDER}
    for comp in comps:
        kind = _classify_mask(np.asarray(comp["mask"], dtype=bool))
        grouped[kind].append(np.asarray(comp["patch"], dtype=int))

    counts = [len(grouped[kind]) for kind in GROUP_ORDER]
    assert all(counts), "expected all three symmetry groups to be present"
    assert len(set(counts)) == 1, "expected equal numbers of objects per symmetry group"

    rows = [_build_row(grouped[kind]) for kind in GROUP_ORDER]
    out_h = sum(int(row.shape[0]) for row in rows) + (len(rows) - 1)
    out_w = max(int(row.shape[1]) for row in rows)
    out = np.zeros((out_h, out_w), dtype=int)

    r = 0
    for i, row in enumerate(rows):
        h, w = row.shape
        out[r : r + h, :w] = row
        r += h
        if i != len(rows) - 1:
            r += 1
    return out


def _sample_objects(count_per_type: int) -> list[dict]:
    used: set[tuple[str, tuple[int, int], bytes]] = set()
    colors = random.sample(PALETTE, 3 * count_per_type)
    objects: list[dict] = []
    color_idx = 0

    for kind in GROUP_ORDER:
        made = 0
        while made < count_per_type:
            mask = _sample_symmetric_mask(kind)
            sig = (kind, tuple(map(int, mask.shape)), mask.tobytes())
            if sig in used:
                continue
            used.add(sig)
            patch = np.zeros(mask.shape, dtype=int)
            patch[mask] = colors[color_idx]
            color_idx += 1
            objects.append({"kind": kind, "mask": mask, "patch": patch})
            made += 1

    random.shuffle(objects)
    return objects


def _slot_layout(total_objects: int) -> tuple[int, int]:
    if total_objects == 6:
        return random.choice(((2, 3), (3, 2), (3, 3)))
    return (3, 3)


def _place_objects(objects: list[dict]) -> np.ndarray | None:
    rows, cols = _slot_layout(len(objects))
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    slot_h = max_h + 3
    slot_w = max_w + 3
    base_h = rows * slot_h
    base_w = cols * slot_w
    if base_h > 30 or base_w > 30 or rows * cols < len(objects):
        return None

    pad_h = 30 - base_h
    pad_w = 30 - base_w
    top_pad = random.randint(0, min(2, pad_h))
    bottom_pad = random.randint(0, min(2, pad_h - top_pad))
    left_pad = random.randint(0, min(2, pad_w))
    right_pad = random.randint(0, min(2, pad_w - left_pad))

    grid = np.zeros((base_h + top_pad + bottom_pad, base_w + left_pad + right_pad), dtype=int)
    slots = [(sr, sc) for sr in range(rows) for sc in range(cols)]
    for obj, (sr, sc) in zip(objects, random.sample(slots, len(objects))):
        patch = np.asarray(obj["patch"], dtype=int)
        h, w = patch.shape
        r0 = top_pad + sr * slot_h
        c0 = left_pad + sc * slot_w
        dr = random.randint(1, slot_h - h - 1)
        dc = random.randint(1, slot_w - w - 1)
        dest = grid[r0 + dr : r0 + dr + h, c0 + dc : c0 + dc + w]
        dest[patch != BG] = patch[patch != BG]

    return grid


def _valid_sample(grid: np.ndarray, count_per_type: int) -> bool:
    comps = _find_components(grid)
    if len(comps) != 3 * count_per_type:
        return False
    try:
        kinds = [_classify_mask(np.asarray(comp["mask"], dtype=bool)) for comp in comps]
    except AssertionError:
        return False
    counts = Counter(kinds)
    if any(int(counts.get(kind, 0)) != count_per_type for kind in GROUP_ORDER):
        return False
    out = main(grid)
    return max(grid.shape) <= 30 and max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    count_per_type = random.choice((2, 3))
    objects = _sample_objects(count_per_type)
    grid = _place_objects(objects)
    if grid is None:
        return None
    if not _valid_sample(grid, count_per_type):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-grouping sample")

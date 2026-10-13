from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MAX_GENERATION_ATTEMPTS = 300
MAX_PLACEMENT_TRIES = 250

_BASE_PROTOTYPES = [
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
]


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _outline_of(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _fill_outline(outline: np.ndarray) -> np.ndarray:
    h, w = outline.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not outline[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not outline[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return outline | ~outside


def _resize_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    row_idx = (np.arange(dst_h) * src_h) // dst_h
    col_idx = (np.arange(dst_w) * src_w) // dst_w
    return mask[row_idx][:, col_idx]


def _component_list(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            queue: deque[tuple[int, int]] = deque([(r0, c0)])
            coords: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in neigh:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1, c1), mask


def _boundary_pixels(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(coords)
    boundary: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr, dc in DIR4:
            if (r + dr, c + dc) not in pts:
                boundary.add((r, c))
                break
    return boundary


def _parse_legend(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    grey_comps = _component_list(grid == GREY, connectivity=8)
    assert grey_comps, "expected grey outline entries on the right"

    entries: list[dict] = []
    legend_mask = np.zeros((h, w), dtype=bool)
    for coords in grey_comps:
        (r0, c0, r1, c1), outline = _mask_from_coords(coords)
        prototype = _fill_outline(outline)
        swatches: list[tuple[int, int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c1 + 1, w):
                color = int(grid[r, c])
                if color not in (BLACK, GREY):
                    swatches.append((r, c, color))
        swatches.sort(key=lambda x: (x[1], x[0]))
        assert len(swatches) >= 2, "expected two color swatches next to each legend entry"
        outer = swatches[0][2]
        inner = swatches[1][2]
        for r, c in coords:
            legend_mask[r, c] = True
        legend_mask[swatches[0][0], swatches[0][1]] = True
        legend_mask[swatches[1][0], swatches[1][1]] = True
        entries.append(
            {
                "bbox": (r0, c0, r1, c1),
                "prototype": prototype,
                "outer": outer,
                "inner": inner,
            }
        )
    return entries, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each large solid object to the grey outline prototype on the right
    after nearest-neighbor downscaling, then recolor boundary and interior
    layers using the two swatches beside that prototype.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    entries, legend_mask = _parse_legend(g)
    out = g.copy()

    object_mask = (g != BLACK) & (g != GREY) & ~legend_mask
    comps = _component_list(object_mask, connectivity=4)
    assert comps, "expected at least one large object"

    for coords in comps:
        (r0, c0, r1, c1), mask = _mask_from_coords(coords)
        matches: list[dict] = []
        for entry in entries:
            proto = entry["prototype"]
            shrunk = _resize_nearest(mask, proto.shape)
            if np.array_equal(shrunk, proto):
                matches.append(entry)
        assert len(matches) == 1, "each object must match exactly one legend prototype"
        entry = matches[0]
        boundary = _boundary_pixels(coords)
        for r, c in coords:
            out[r, c] = entry["outer"] if (r, c) in boundary else entry["inner"]

    return out


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return _trim(out)


def _valid_proto(mask: np.ndarray) -> bool:
    outline = _outline_of(mask)
    return np.array_equal(_fill_outline(outline), mask)


def _scaled_object(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _resizes_to(src: np.ndarray, dst: np.ndarray) -> bool:
    return np.array_equal(_resize_nearest(src, dst.shape), dst)


def _prototypes_compatible(a: np.ndarray, b: np.ndarray) -> bool:
    return not (_resizes_to(a, b) or _resizes_to(b, a))


def _can_place(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    sub = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = color


def _build_expected(inp: np.ndarray, objects: list[dict], entries: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        match = entries[obj["entry_idx"]]
        mask = obj["mask"]
        top = obj["top"]
        left = obj["left"]
        coords = [(top + r, left + c) for r, c in zip(*np.where(mask))]
        boundary = _boundary_pixels(coords)
        for r, c in coords:
            out[r, c] = match["outer"] if (r, c) in boundary else match["inner"]
    return out


def _gen_attempt() -> np.ndarray | None:
    base_pool = [mask for mask in _BASE_PROTOTYPES if _valid_proto(mask)]
    n_entries = random.randint(2, 4)
    chosen: list[np.ndarray] = []
    seen = set()
    pick_attempts = 0
    while len(chosen) < n_entries and pick_attempts < 200:
        pick_attempts += 1
        cand = _random_transform(random.choice(base_pool))
        key = (cand.shape, cand.tobytes())
        if key in seen:
            continue
        if not all(_prototypes_compatible(cand, prev) for prev in chosen):
            continue
        chosen.append(cand)
        seen.add(key)
    if len(chosen) < n_entries:
        return None

    color_order = list(COLORS)
    random.shuffle(color_order)
    entries: list[dict] = []
    for idx, proto in enumerate(chosen):
        outer = color_order[2 * idx]
        inner = color_order[2 * idx + 1]
        entries.append(
            {
                "prototype": proto,
                "outline": _outline_of(proto),
                "outer": outer,
                "inner": inner,
            }
        )

    n_objects = random.randint(n_entries, min(5, n_entries + 1))
    assignment = list(range(n_entries))
    while len(assignment) < n_objects:
        assignment.append(random.randrange(n_entries))
    random.shuffle(assignment)

    objects: list[dict] = []
    for entry_idx in assignment:
        proto = entries[entry_idx]["prototype"]
        scale_choices = [2, 3, 4] if n_objects <= 3 else [2, 3]
        random.shuffle(scale_choices)
        big_mask = None
        for scale in scale_choices:
            cand = _scaled_object(proto, scale)
            if np.any(~np.isin(cand.astype(int), [0, 1])):
                continue
            if len(_boundary_pixels([(r, c) for r, c in zip(*np.where(cand))])) < int(cand.sum()):
                big_mask = cand
                break
        if big_mask is None:
            big_mask = _scaled_object(proto, 4)
        object_color_choices = [c for c in COLORS if c not in {entries[entry_idx]["outer"], entries[entry_idx]["inner"]}]
        objects.append(
            {
                "entry_idx": entry_idx,
                "mask": big_mask,
                "color": random.choice(object_color_choices),
            }
        )

    entry_heights = [int(e["outline"].shape[0]) for e in entries]
    entry_widths = [int(e["outline"].shape[1]) + 2 for e in entries]
    table_width = max(entry_widths)
    gap_rows = [random.randint(1, 2) for _ in range(max(0, n_entries - 1))]
    table_height = sum(entry_heights) + sum(gap_rows)

    object_heights = [int(obj["mask"].shape[0]) for obj in objects]
    object_widths = [int(obj["mask"].shape[1]) for obj in objects]
    max_obj_h = max(object_heights)
    max_obj_w = max(object_widths)
    total_obj_area = sum(int(obj["mask"].sum()) for obj in objects)

    grid_h_lo = max(table_height + 4, max_obj_h + 4, 17)
    if grid_h_lo > 34:
        return None
    grid_h = random.randint(grid_h_lo, min(38, grid_h_lo + 6))

    left_region_w_lo = max(max_obj_w + 3, int(np.ceil(total_obj_area / max(6, grid_h - 4))) + 7)
    left_region_w_lo = max(left_region_w_lo, 12)
    left_region_w_hi = min(34, left_region_w_lo + 8)
    if left_region_w_lo > left_region_w_hi:
        return None
    left_region_w = random.randint(left_region_w_lo, left_region_w_hi)

    separator = random.randint(2, 4)
    right_margin = random.randint(1, 2)
    grid_w = left_region_w + separator + table_width + right_margin
    if grid_w > 42:
        return None

    table_top = random.randint(1, grid_h - table_height - 1)
    table_left = grid_w - right_margin - table_width
    left_region_h = grid_h - 2
    left_region_top = 1
    left_region_left = 0

    blocked = np.zeros((grid_h, left_region_w), dtype=bool)
    order = sorted(range(n_objects), key=lambda i: objects[i]["mask"].sum(), reverse=True)
    for idx in order:
        mask = objects[idx]["mask"]
        oh, ow = mask.shape
        placed = False
        for _ in range(MAX_PLACEMENT_TRIES):
            top = random.randint(left_region_top, left_region_top + left_region_h - oh)
            left = random.randint(left_region_left, left_region_left + left_region_w - ow)
            if not _can_place(blocked, top, left, oh, ow, pad=1):
                continue
            objects[idx]["top"] = top
            objects[idx]["left"] = left
            _mark(blocked, top, left, oh, ow, pad=1)
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    row = table_top
    for i, entry in enumerate(entries):
        outline = entry["outline"]
        eh, ew = outline.shape
        _paint_mask(inp, row, table_left, outline, GREY)
        swatch_row = row + eh // 2
        inp[swatch_row, table_left + ew] = entry["outer"]
        inp[swatch_row, table_left + ew + 1] = entry["inner"]
        entry["legend_top"] = row
        entry["legend_left"] = table_left
        row += eh
        if i < len(gap_rows):
            row += gap_rows[i]

    for obj in objects:
        _paint_mask(inp, obj["top"], obj["left"], obj["mask"], obj["color"])

    expected = _build_expected(inp, objects, entries)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

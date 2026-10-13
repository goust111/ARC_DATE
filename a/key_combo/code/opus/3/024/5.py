import random
from collections import deque

import numpy as np


BG = 4
BLACK = 0
RED = 2
SEP = 5

MAX_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_from_cells(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def components4(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def crop_bbox(grid: np.ndarray, bbox):
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_separator_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    cols = [c for c in range(w) if np.all(grid[:, c] == SEP)]
    assert len(cols) == 1, f"expected one separator column, got {cols}"
    return int(cols[0])


def component_holes_from_cells(grid: np.ndarray, cells, bg: int = BG):
    comp_bbox = bbox_from_cells(cells)
    r0, c0, r1, c1 = comp_bbox
    patch_h = r1 - r0 + 1
    patch_w = c1 - c0 + 1

    fg = np.zeros((patch_h, patch_w), dtype=bool)
    for r, c in cells:
        fg[r - r0, c - c0] = True

    outside = np.zeros((patch_h, patch_w), dtype=bool)
    q = deque()
    for rr in range(patch_h):
        for cc in range(patch_w):
            if rr not in (0, patch_h - 1) and cc not in (0, patch_w - 1):
                continue
            if not fg[rr, cc] and int(grid[r0 + rr, c0 + cc]) == bg and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < patch_h and 0 <= nc < patch_w:
                if fg[nr, nc] or outside[nr, nc]:
                    continue
                if int(grid[r0 + nr, c0 + nc]) != bg:
                    continue
                outside[nr, nc] = True
                q.append((nr, nc))

    hole_mask = np.zeros((patch_h, patch_w), dtype=bool)
    for rr in range(patch_h):
        for cc in range(patch_w):
            if fg[rr, cc]:
                continue
            if int(grid[r0 + rr, c0 + cc]) == bg and not outside[rr, cc]:
                hole_mask[rr, cc] = True

    holes = []
    for comp in components4(hole_mask):
        global_cells = [(r0 + rr, c0 + cc) for rr, cc in comp]
        holes.append(global_cells)
    return holes


def hole_template(cells):
    r0, c0, r1, c1 = bbox_from_cells(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1, c1), mask


def build_output_from_components(grid: np.ndarray, moved_components):
    out = grid.copy()
    sep = find_separator_col(out)
    left = out[:, :sep]
    left[left == BLACK] = BG

    sorted_components = sorted(moved_components, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    active_templates = []

    for item in sorted_components:
        cells = item["cells"]
        for r, c in cells:
            out[r, c] = BLACK

        holes = component_holes_from_cells(out, cells, bg=BG)
        for hole in holes:
            hb = bbox_from_cells(hole)
            # Support the intended "separator-adjacent nozzle" interpretation, and
            # also the physically common case where the hole touches the inner wall
            # one cell left of the separator.
            if hb[3] not in (sep - 1, sep - 2):
                continue
            _, mask = hole_template(hole)
            active_templates.append(
                {
                    "priority": item["bbox"][0],
                    "top": hb[0],
                    "mask": mask,
                }
            )

    claimed = np.zeros(out.shape, dtype=bool)
    right_start = sep + 1
    h, w = out.shape
    for tpl in sorted(active_templates, key=lambda x: x["priority"]):
        top = tpl["top"]
        mask = tpl["mask"]
        th, tw = mask.shape
        col = right_start
        while col < w:
            for rr in range(th):
                gr = top + rr
                if not (0 <= gr < h):
                    continue
                for cc in range(tw):
                    gc = col + cc
                    if gc >= w:
                        continue
                    if mask[rr, cc] and not claimed[gr, gc]:
                        claimed[gr, gc] = True
                        out[gr, gc] = RED
            col += tw
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes to the grey separator, then tile each separator-side hole shape across the right half.

    Input structure:
    - Yellow (4) is the background and one full-height grey (5) column is the separator.
    - Left of the separator there are 2-3 black (0) 4-connected components, each containing one enclosed yellow hole.
    - The right half starts as background only.

    Rule:
    - Sort the black components by bbox top row and translate each one horizontally so its right edge becomes flush with the column just left of the separator.
    - For each translated component, detect enclosed background regions by flood-filling from the component bbox border; only holes whose right edge reaches the separator-side nozzle are active.
    - Convert each active hole into a relative pixel mask from its own bbox and repeat that mask horizontally from the first column right of the separator to the grid edge, painting red (2) and clipping partial tiles.
    - If two tiled patterns want the same right-half pixel, the component with the smaller bbox top row keeps it; the translated left half stays unchanged.

    Key insight:
    - The hole is not a binary trigger: after the slide, the hole's exact geometry becomes the tile motif that is stamped across the right side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = find_separator_col(g)
    left = g[:, :sep]
    comps = components4(left == BLACK)
    ordered = sorted(comps, key=lambda cells: bbox_from_cells(cells))

    moved_components = []
    occupied = np.zeros_like(left, dtype=bool)
    for cells in ordered:
        b = bbox_from_cells(cells)
        delta = (sep - 1) - b[3]
        assert delta >= 0, "component crosses the separator"
        shifted = [(r, c + delta) for r, c in cells]
        for r, c in shifted:
            assert 0 <= c < sep
            assert not occupied[r, c], "translated components must not overlap"
            occupied[r, c] = True
        moved_components.append({"cells": shifted, "bbox": bbox_from_cells(shifted)})

    return build_output_from_components(g, moved_components)


HOLE_PATTERNS = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]


def rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    out = np.rot90(mask, k=k)
    coords = np.argwhere(out)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1].copy()


def random_hole_mask(rng: random.Random) -> np.ndarray:
    base = HOLE_PATTERNS[rng.randrange(len(HOLE_PATTERNS))]
    mask = rotate_mask(base, rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.fliplr(mask).copy()
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    return mask


def count_component_holes_in_patch(patch: np.ndarray):
    cells = [(int(r), int(c)) for r, c in np.argwhere(patch == BLACK)]
    holes = component_holes_from_cells(patch, cells, bg=BG)
    return holes


def shape_is_valid(patch: np.ndarray) -> bool:
    comps = components4(patch == BLACK)
    if len(comps) != 1:
        return False
    holes = count_component_holes_in_patch(patch)
    return len(holes) == 1


def irregularize_patch(patch: np.ndarray, hole_mask: np.ndarray, hole_r0: int, hole_c0: int, rng: random.Random):
    out = patch.copy()
    h, w = out.shape
    hole_cells = {(hole_r0 + r, hole_c0 + c) for r, c in np.argwhere(hole_mask)}

    boundary = []
    for r in range(h):
        for c in range(w):
            if out[r, c] != BLACK:
                continue
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if any(abs(r - hr) + abs(c - hc) == 1 for hr, hc in hole_cells):
                continue
            boundary.append((r, c))

    rng.shuffle(boundary)
    for r, c in boundary[: rng.randint(0, 2)]:
        trial = out.copy()
        trial[r, c] = BG
        if shape_is_valid(trial):
            out = trial
    return out


def make_shape_patch(hole_mask: np.ndarray, active: bool, rng: random.Random):
    hh, hw = hole_mask.shape
    top_margin = 1
    bottom_margin = 1
    left_margin = 1
    right_margin = 1 if active else 2

    patch = np.full((hh + top_margin + bottom_margin, hw + left_margin + right_margin), BLACK, dtype=int)
    hole_r0 = top_margin
    hole_c0 = left_margin
    patch[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + hw][hole_mask] = BG
    patch = irregularize_patch(patch, hole_mask, hole_r0, hole_c0, rng)

    holes = count_component_holes_in_patch(patch)
    assert len(holes) == 1
    return patch


def split_extra(total: int, parts: int, rng: random.Random):
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def expected_from_meta(shape, sep, entries):
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP
    claimed = np.zeros((h, w), dtype=bool)

    ordered = sorted(entries, key=lambda item: item["row"])
    for item in ordered:
        patch = item["patch"]
        r0 = item["row"]
        ph, pw = patch.shape
        shift_col = sep - pw
        for rr in range(ph):
            for cc in range(pw):
                if patch[rr, cc] == BLACK:
                    out[r0 + rr, shift_col + cc] = BLACK

    for item in ordered:
        if not item["active"]:
            continue
        mask = item["hole_mask"]
        mh, mw = mask.shape
        top = item["row"] + 1
        col = sep + 1
        while col < w:
            for rr in range(mh):
                gr = top + rr
                if not (0 <= gr < h):
                    continue
                for cc in range(mw):
                    gc = col + cc
                    if gc >= w:
                        continue
                    if mask[rr, cc] and not claimed[gr, gc]:
                        claimed[gr, gc] = True
                        out[gr, gc] = RED
            col += mw

    return out


def generate_case(rng: random.Random):
    n_shapes = rng.choice([2, 3])
    active_flags = [True]
    if n_shapes == 2:
        active_flags.append(rng.random() < 0.45)
    else:
        active_flags.extend([rng.random() < 0.65 for _ in range(n_shapes - 1)])
    if not any(flag is False for flag in active_flags):
        active_flags[rng.randrange(1, n_shapes)] = False
    rng.shuffle(active_flags)

    entries = []
    for active in active_flags:
        hole_mask = random_hole_mask(rng)
        patch = make_shape_patch(hole_mask, active, rng)
        entries.append({"active": active, "hole_mask": hole_mask, "patch": patch})

    max_patch_w = max(item["patch"].shape[1] for item in entries)
    max_hole_w = max(item["hole_mask"].shape[1] for item in entries)
    sep_min = max_patch_w + 4
    sep = rng.randint(sep_min, min(11, sep_min + 2))
    right_w = rng.randint(max(max_hole_w + 3, 6), 10)
    width = sep + 1 + right_w

    total_heights = sum(item["patch"].shape[0] for item in entries)
    min_height = total_heights + (n_shapes + 1)
    height_low = max(16, min_height)
    height_high = max(height_low, min(24, min_height + 4))
    height = rng.randint(height_low, height_high)
    slack = height - min_height
    gaps = [1 + extra for extra in split_extra(slack, n_shapes + 1, rng)]

    row = gaps[0]
    for i, item in enumerate(entries):
        patch = item["patch"]
        max_gap = min(4, sep - patch.shape[1])
        gap_seed = patch.shape[0] + 2 * patch.shape[1] + int(item["hole_mask"].sum()) + int(item["active"]) + row
        gap = 1 + (gap_seed % max_gap)
        item["row"] = row
        item["col"] = sep - patch.shape[1] - gap
        row += patch.shape[0] + gaps[i + 1]

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep] = SEP
    for item in entries:
        patch = item["patch"]
        r0, c0 = item["row"], item["col"]
        ph, pw = patch.shape
        region = grid[r0 : r0 + ph, c0 : c0 + pw]
        assert np.all(region == BG)
        region[patch == BLACK] = BLACK

    return grid, sep, entries


def validate_generated_case(grid: np.ndarray, sep: int, entries) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid[:, sep] == SEP) != grid.shape[0]:
        return False
    if np.any(grid[:, sep + 1 :] != BG):
        return False

    comps = components4(grid[:, :sep] == BLACK)
    if len(comps) != len(entries):
        return False

    ordered = sorted(comps, key=bbox_from_cells)
    shifted_boxes = []
    expected_active = sum(1 for item in entries if item["active"])
    expected_inactive = len(entries) - expected_active
    seen_active = 0
    seen_inactive = 0
    for cells in ordered:
        holes = component_holes_from_cells(grid, cells, bg=BG)
        if len(holes) != 1:
            return False
        b = bbox_from_cells(cells)
        delta = (sep - 1) - b[3]
        moved_hole = [(r, c + delta) for r, c in holes[0]]
        hole_bbox = bbox_from_cells(moved_hole)
        if hole_bbox[3] in (sep - 1, sep - 2):
            seen_active += 1
        else:
            seen_inactive += 1

        shifted = [(r, c + delta) for r, c in cells]
        sb = bbox_from_cells(shifted)
        for ob in shifted_boxes:
            if not (sb[2] < ob[0] or ob[2] < sb[0] or sb[3] < ob[1] or ob[3] < sb[1]):
                return False
        shifted_boxes.append(sb)

    expected = expected_from_meta(grid.shape, sep, entries)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if seen_active != expected_active or seen_inactive != expected_inactive:
        return False
    if seen_active == 0 or seen_inactive == 0:
        return False
    if np.count_nonzero(actual[:, sep + 1 :] == RED) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid, sep, entries = generate_case(rng)
        if validate_generated_case(grid, sep, entries):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

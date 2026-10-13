import random
from collections import Counter

import numpy as np


BG = 0
PATTERN = 1
SEED = 2
GREEN = 3
MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray, color: int):
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                local[rr - r0, cc - c0] = True
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": len(coords),
                }
            )
    return comps


def _touches_corner(comp: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _select_seed_component(components: list[dict], shape: tuple[int, int]) -> dict:
    ranked = sorted(
        components,
        key=lambda comp: (
            0 if _touches_corner(comp, shape) else 1,
            comp["area"],
            comp["bbox"][0],
            comp["bbox"][1],
        ),
    )
    return ranked[0]


def _choose_roles(grid: np.ndarray):
    bg = _get_bg(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    by_color = {color: _find_components(grid, color) for color in colors}
    pattern_color = max(
        colors,
        key=lambda color: (
            len(by_color[color]),
            max((comp["area"] for comp in by_color[color]), default=0),
        ),
    )
    seed_color = min(
        [color for color in colors if color != pattern_color],
        key=lambda color: (
            0
            if any(_touches_corner(comp, grid.shape) for comp in by_color[color])
            else 1,
            len(by_color[color]),
            sum(comp["area"] for comp in by_color[color]),
        ),
    )
    return bg, pattern_color, seed_color, by_color


def _pattern_group(components: list[dict]) -> list[dict]:
    shape_counts = Counter(comp["mask"].shape for comp in components)
    target_shape = max(
        shape_counts,
        key=lambda shape: (shape_counts[shape], shape[0] * shape[1]),
    )
    group = [comp for comp in components if comp["mask"].shape == target_shape]
    assert len(group) >= 2
    return group


def _reference_index(components: list[dict]) -> int:
    masks = [comp["mask"] for comp in components]
    scores = []
    for i, mask_i in enumerate(masks):
        total = 0
        for mask_j in masks:
            total += int(np.count_nonzero(mask_i != mask_j))
        scores.append(total)
    order = sorted(
        range(len(components)),
        key=lambda i: (
            scores[i],
            components[i]["bbox"][0],
            components[i]["bbox"][1],
            -components[i]["area"],
        ),
    )
    return order[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare near-identical pattern copies against the reference, read each lone
    differing local coordinate as a translation vector, and stamp the corner
    seed at those offsets in green.
    """
    g = np.asarray(grid, dtype=int)
    bg, pattern_color, seed_color, by_color = _choose_roles(g)

    seed_component = _select_seed_component(by_color[seed_color], g.shape)
    sr0, sc0, _sr1, _sc1 = seed_component["bbox"]
    seed_rel = [tuple(map(int, coord)) for coord in np.argwhere(seed_component["mask"])]

    pattern_components = _pattern_group(by_color[pattern_color])
    ref_idx = _reference_index(pattern_components)
    ref_mask = pattern_components[ref_idx]["mask"]

    out = g.copy()
    for idx, comp in enumerate(pattern_components):
        if idx == ref_idx:
            continue
        diff = np.argwhere(ref_mask != comp["mask"])
        assert diff.shape[0] == 1, "each copy should differ from the reference by one pixel"
        dr, dc = map(int, diff[0])
        for rr, cc in seed_rel:
            tr = sr0 + dr + rr
            tc = sc0 + dc + cc
            if 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1] and out[tr, tc] in (bg, GREEN):
                out[tr, tc] = GREEN

    return out


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        steps = []
        if r < er:
            steps.append((1, 0))
        elif r > er:
            steps.append((-1, 0))
        if c < ec:
            steps.append((0, 1))
        elif c > ec:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        mask[r, c] = True


def _full_bbox_blob(height: int, width: int, min_area: int, max_area: int):
    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        top = (0, random.randrange(width))
        bottom = (height - 1, random.randrange(width))
        left = (random.randrange(height), 0)
        right = (random.randrange(height), width - 1)

        _carve_path(mask, top, bottom)
        existing = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        _carve_path(mask, left, random.choice(existing))
        existing = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        _carve_path(mask, right, random.choice(existing))

        base_area = int(mask.sum())
        if base_area > max_area:
            continue

        target_area = random.randint(max(min_area, base_area), max_area)
        while int(mask.sum()) < target_area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), height, width):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            rr, cc = random.choice(tuple(frontier))
            mask[rr, cc] = True

        if _mask_bbox(mask) != (0, 0, height - 1, width - 1):
            continue
        if not _is_connected(mask):
            continue
        return mask
    return None


def _removable_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    if int(mask.sum()) <= 1:
        return out
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        candidate = mask.copy()
        candidate[rr, cc] = False
        if _mask_bbox(candidate) != (0, 0, h - 1, w - 1):
            continue
        if _is_connected(candidate):
            out.append((rr, cc))
    return out


def _addable_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
                out.append((r, c))
    return out


def _shifted_bboxes_separate(
    pos_a: tuple[int, int], pos_b: tuple[int, int], seed_shape: tuple[int, int]
) -> bool:
    sh, sw = seed_shape
    return abs(pos_a[0] - pos_b[0]) >= sh or abs(pos_a[1] - pos_b[1]) >= sw


def _select_differences(
    addables: list[tuple[int, int]],
    removables: list[tuple[int, int]],
    seed_shape: tuple[int, int],
):
    candidates = [((r, c), "add") for r, c in addables] + [((r, c), "remove") for r, c in removables]
    sh, sw = seed_shape
    candidates = [item for item in candidates if item[0][0] >= sh or item[0][1] >= sw]
    if len(candidates) < 2:
        return None

    max_copies = min(4, len(candidates))
    for _ in range(300):
        count = random.randint(2, max_copies)
        require_mixed = bool(addables and removables and random.random() < 0.8)
        pool = candidates[:]
        random.shuffle(pool)
        chosen = []
        for item in pool:
            pos, _kind = item
            if all(_shifted_bboxes_separate(pos, other_pos, seed_shape) for other_pos, _ in chosen):
                chosen.append(item)
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue
        if require_mixed and len({kind for _pos, kind in chosen}) < 2:
            continue
        return chosen
    return None


def _apply_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = color


def _copy_mask(reference: np.ndarray, pos: tuple[int, int], kind: str) -> np.ndarray:
    out = reference.copy()
    r, c = pos
    out[r, c] = kind == "add"
    return out


def _expected_output(grid: np.ndarray, seed_mask: np.ndarray, vectors: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for dr, dc in vectors:
        for rr, cc in np.argwhere(seed_mask):
            out[dr + int(rr), dc + int(cc)] = GREEN
    return out


def _gen_attempt():
    seed_h = random.randint(2, 3)
    seed_w = random.randint(2, 4)
    seed_min = max(3, seed_h + seed_w - 1)
    seed_max = min(seed_h * seed_w, seed_min + 2)
    seed_mask = _full_bbox_blob(seed_h, seed_w, seed_min, seed_max)
    if seed_mask is None:
        return None

    pat_h = random.randint(5, 8)
    pat_w = random.randint(5, 8)
    pat_min = max(seed_h + seed_w + 2, (pat_h * pat_w) // 3)
    pat_max = pat_h * pat_w - 2
    if pat_min > pat_max:
        return None
    reference_mask = _full_bbox_blob(pat_h, pat_w, pat_min, pat_max)
    if reference_mask is None:
        return None

    addables = _addable_positions(reference_mask)
    removables = _removable_positions(reference_mask)
    if not addables or not removables:
        return None

    chosen = _select_differences(addables, removables, seed_mask.shape)
    if chosen is None:
        return None

    vectors = [pos for pos, _kind in chosen]
    n_objects = len(chosen) + 1
    slot_cols = random.choice([2, 3] if n_objects >= 4 else [2])
    slot_rows = (n_objects + slot_cols - 1) // slot_cols
    gap_r = random.randint(2, 4)
    gap_c = random.randint(2, 5)
    start_row = pat_h + seed_h + random.randint(1, 3)
    start_col = random.randint(1, 4)

    height = start_row + slot_rows * pat_h + (slot_rows - 1) * gap_r + random.randint(1, 3)
    width = start_col + slot_cols * pat_w + (slot_cols - 1) * gap_c + random.randint(1, 4)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _apply_mask(grid, seed_mask, 0, 0, SEED)

    placements = []
    for idx in range(n_objects):
        row = idx // slot_cols
        col = idx % slot_cols
        top = start_row + row * (pat_h + gap_r)
        left = start_col + col * (pat_w + gap_c)
        placements.append((top, left))

    _apply_mask(grid, reference_mask, placements[0][0], placements[0][1], PATTERN)
    for (pos, kind), (top, left) in zip(chosen, placements[1:]):
        _apply_mask(grid, _copy_mask(reference_mask, pos, kind), top, left, PATTERN)

    expected = _expected_output(grid, seed_mask, vectors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

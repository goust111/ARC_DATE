from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _random_frontier(length: int, limit: int) -> list[int]:
    cut = random.randint(1, limit - 1)
    frontier: list[int] = []
    for _ in range(length):
        cut = max(1, min(limit - 1, cut + random.choice((-1, 0, 1))))
        frontier.append(cut)
    if random.random() < 0.5:
        frontier.reverse()
    return frontier


def _make_template(size: int) -> np.ndarray:
    rr, cc = np.indices((size, size))

    for _ in range(200):
        anchor = random.choice(PALETTE)
        non_anchor = list(random.sample([c for c in PALETTE if c != anchor], random.randint(3, 4)))
        canvas = np.full((size, size), non_anchor[0], dtype=int)
        style = random.choice(("row_frontier", "col_frontier", "corner_regions"))

        if style == "row_frontier":
            cuts = _random_frontier(size, size)
            for r, cut in enumerate(cuts):
                canvas[r, :cut] = non_anchor[1]
        elif style == "col_frontier":
            cuts = _random_frontier(size, size)
            for c, cut in enumerate(cuts):
                canvas[:cut, c] = non_anchor[1]
        else:
            corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
            first, second = random.sample(corners, 2)
            jitter = np.random.uniform(-0.35, 0.35, size=(size, size))
            d_first = np.abs(rr - first[0]) + np.abs(cc - first[1]) + jitter
            d_second = np.abs(rr - second[0]) + np.abs(cc - second[1]) - jitter
            canvas = np.where(d_first <= d_second, non_anchor[0], non_anchor[1]).astype(int)

        rect_h = random.randint(2, size - 1)
        rect_w = random.randint(2, size - 1)
        rect_r = random.randint(0, size - rect_h)
        rect_c = random.randint(0, size - rect_w)
        canvas[rect_r : rect_r + rect_h, rect_c : rect_c + rect_w] = non_anchor[2]

        if len(non_anchor) == 4:
            if random.random() < 0.5:
                band_c = random.randint(1, size - 2)
                band_w = random.randint(1, min(2, size - band_c))
                canvas[:, band_c : band_c + band_w] = non_anchor[3]
            else:
                band_r = random.randint(1, size - 2)
                band_h = random.randint(1, min(2, size - band_r))
                canvas[band_r : band_r + band_h, :] = non_anchor[3]

        canvas[0, 0] = anchor

        if np.any(canvas[1:, :] == anchor) or np.any(canvas[:, 1:][canvas[:, 1:] == anchor]):
            continue

        colors = [int(c) for c in np.unique(canvas)]
        if len(colors) < 4:
            continue
        if any(int(np.count_nonzero(canvas == color)) < 2 for color in colors if color != anchor):
            continue
        if any(int(np.count_nonzero(canvas[1:, 1:] == color)) == 0 for color in colors if color != anchor):
            continue
        if np.count_nonzero(canvas == anchor) != 1:
            continue
        return canvas

    raise RuntimeError("failed to construct a non-degenerate template")


def _rotate_coord(r: int, c: int, size: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return size - 1 - c, r
    if k == 2:
        return size - 1 - r, size - 1 - c
    return c, size - 1 - r


def _find_anchor_color(crops: list[np.ndarray], bg: int) -> int:
    all_colors = sorted({int(color) for crop in crops for color in np.unique(crop) if int(color) != bg})
    candidates: list[int] = []
    for color in all_colors:
        valid = True
        for crop in crops:
            hits = np.argwhere(crop == color)
            if hits.shape[0] != 1:
                valid = False
                break
            r, c = map(int, hits[0])
            h, w = crop.shape
            if (r, c) not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
                valid = False
                break
        if valid:
            candidates.append(color)
    assert len(candidates) == 1, "anchor color must be uniquely identifiable"
    return int(candidates[0])


def _rotation_to_standard(crop: np.ndarray, anchor_color: int) -> int:
    h, w = crop.shape
    assert h == w, "each copy must be square"
    corners = [
        ((0, 0), 0),
        ((0, w - 1), 1),
        ((h - 1, w - 1), 2),
        ((h - 1, 0), 3),
    ]
    matches = [rot for (r, c), rot in corners if int(crop[r, c]) == anchor_color]
    assert len(matches) == 1, "anchor color must occupy exactly one corner"
    return int(matches[0])


def _majority_vote(crops: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(crops, axis=0)
    n, h, w = stack.shape
    out = np.zeros((h, w), dtype=int)
    max_color = max(10, int(stack.max()) + 1)

    for r in range(h):
        for c in range(w):
            counts = np.bincount(stack[:, r, c], minlength=max_color)
            best = int(counts.max())
            winners = np.flatnonzero(counts == best)
            assert winners.size == 1, "each cell must have a unique majority color"
            out[r, c] = int(winners[0])

    assert out.shape == (h, w)
    assert n >= 3
    return out


def _extract_crops(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    bg = get_bg(grid)
    components = components_mask(grid != bg)
    assert len(components) >= 3, "expected multiple separated copies"

    crops: list[np.ndarray] = []
    sizes = set()
    for coords in components:
        box = bbox_coords(coords)
        crop = crop_bbox(grid, box)
        h, w = crop.shape
        assert h == w, "each copy bbox must be square"
        crops.append(crop)
        sizes.add(h)

    assert len(sizes) == 1, "all copies must share the same square size"
    return bg, crops


def _analyze_grid(grid: np.ndarray) -> tuple[int, list[np.ndarray], int, list[np.ndarray], list[int]]:
    bg, crops = _extract_crops(np.asarray(grid, dtype=int))
    anchor_color = _find_anchor_color(crops, bg)
    rotations = [_rotation_to_standard(crop, anchor_color) for crop in crops]
    normalized = [np.rot90(crop, k) for crop, k in zip(crops, rotations)]
    return bg, crops, anchor_color, normalized, rotations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate damaged copies back to a shared anchor orientation, then recover the original by cellwise majority vote.

    Input structure:
    - A black background contains several separated square copies of the same multicolor template.
    - Each copy is rotated by 0/90/180/270 degrees and has a few cells erased to black or repainted.
    - Exactly one color is a corner anchor: it appears once per copy and always sits on the same template corner before rotation.

    Rule:
    - Identify the anchor color and read which corner of each copy currently contains it.
    - Rotate every copy until the anchor returns to the top-left corner, giving one shared reference orientation.
    - For every cell position in that aligned stack, choose the unique majority color and output the voted template.

    Key insight:
    - No single copy is trustworthy: the corner anchor tells you how much to rotate, and only the aligned majority across copies reconstructs the true pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _bg, _crops, _anchor_color, normalized, _rotations = _analyze_grid(g)
    return _majority_vote(normalized)


def _choose_layout(copy_count: int, size: int) -> tuple[int, int]:
    layouts = []
    for rows in range(1, copy_count + 1):
        cols = (copy_count + rows - 1) // rows
        base_h = rows * size + (rows + 1)
        base_w = cols * size + (cols + 1)
        if base_h <= 30 and base_w <= 30:
            layouts.append((rows, cols))
    assert layouts, "at least one layout must fit the ARC size limit"
    return random.choice(layouts)


def _place_copies(copies: list[np.ndarray]) -> np.ndarray:
    size = copies[0].shape[0]
    copy_count = len(copies)
    rows, cols = _choose_layout(copy_count, size)
    base_h = rows * size + (rows + 1)
    base_w = cols * size + (cols + 1)
    grid_h = random.randint(base_h, 30)
    grid_w = random.randint(base_w, 30)

    row_gaps = [1 + x for x in _split_total(grid_h - base_h, rows + 1)]
    col_gaps = [1 + x for x in _split_total(grid_w - base_w, cols + 1)]

    row_starts: list[int] = []
    pos = row_gaps[0]
    for idx in range(rows):
        row_starts.append(pos)
        pos += size + row_gaps[idx + 1]

    col_starts: list[int] = []
    pos = col_gaps[0]
    for idx in range(cols):
        col_starts.append(pos)
        pos += size + col_gaps[idx + 1]

    slots = [(r, c) for r in row_starts for c in col_starts]
    chosen_slots = random.sample(slots, copy_count)
    random.shuffle(chosen_slots)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for patch, (r0, c0) in zip(copies, chosen_slots):
        grid[r0 : r0 + size, c0 : c0 + size] = patch
    return grid


def _damage_plan(size: int, copy_count: int) -> list[set[tuple[int, int]]]:
    cells = [(r, c) for r in range(size) for c in range(size) if (r, c) != (0, 0)]
    max_hits = copy_count // 2
    targets = [random.randint(2, min(size, 4)) for _ in range(copy_count)]

    for _ in range(200):
        hit_counts = {cell: 0 for cell in cells}
        per_copy = [set() for _ in range(copy_count)]
        order = list(range(copy_count))
        random.shuffle(order)
        ok = True

        for idx in order:
            attempts = 0
            while len(per_copy[idx]) < targets[idx] and attempts < 500:
                cell = random.choice(cells)
                attempts += 1
                if cell in per_copy[idx] or hit_counts[cell] >= max_hits:
                    continue
                per_copy[idx].add(cell)
                hit_counts[cell] += 1
            if len(per_copy[idx]) < targets[idx]:
                ok = False
                break

        if ok and all(per_copy):
            return per_copy

    raise RuntimeError("failed to schedule majority-safe damage")


def _copy_is_valid(copy: np.ndarray, size: int) -> bool:
    components = components_mask(copy != BG)
    if len(components) != 1:
        return False
    box = bbox_coords(components[0])
    return crop_bbox(copy, box).shape == (size, size)


def _build_copies(truth: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    size = truth.shape[0]
    copy_count = random.choice((3, 5))
    per_copy_damage = _damage_plan(size, copy_count)
    all_events = [(idx, cell) for idx, cells in enumerate(per_copy_damage) for cell in cells]

    damage_kind: dict[tuple[int, tuple[int, int]], str] = {}
    random.shuffle(all_events)
    if all_events:
        damage_kind[all_events[0]] = "erase"
    if len(all_events) >= 2:
        damage_kind[all_events[1]] = "recolor"
    for event in all_events[2:]:
        damage_kind[event] = random.choice(("erase", "recolor"))

    for _ in range(200):
        rotations = [random.randrange(4) for _ in range(copy_count)]
        if len(set(rotations)) < 2:
            continue

        copies: list[np.ndarray] = []
        ok = True
        for idx, rot in enumerate(rotations):
            copy = np.rot90(truth, rot).copy()
            for r, c in per_copy_damage[idx]:
                rr, cc = _rotate_coord(r, c, size, rot)
                if damage_kind[(idx, (r, c))] == "erase":
                    copy[rr, cc] = BG
                else:
                    cur = int(copy[rr, cc])
                    options = [color for color in PALETTE if color != cur and color != int(truth[0, 0])]
                    copy[rr, cc] = random.choice(options)
            if not _copy_is_valid(copy, size):
                ok = False
                break
            copies.append(copy)

        if ok:
            return copies, rotations

    raise RuntimeError("failed to construct valid damaged copies")


def _validate_case(inp: np.ndarray, truth: np.ndarray, rotations: list[int]) -> bool:
    if inp.ndim != 2 or truth.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(truth.shape) > 30:
        return False
    if len(set(rotations)) < 2:
        return False

    try:
        bg, crops, anchor_color, normalized, inferred_rotations = _analyze_grid(inp)
        voted = _majority_vote(normalized)
        raw_vote = _majority_vote(crops)
    except Exception:
        return False

    if bg != BG:
        return False
    if anchor_color != int(truth[0, 0]):
        return False
    if not np.array_equal(voted, truth):
        return False
    if not np.array_equal(main(inp), truth):
        return False
    if np.array_equal(raw_vote, truth):
        return False
    if len(set(inferred_rotations)) < 2:
        return False
    if any(np.array_equal(crop, truth) for crop in normalized):
        return False
    if np.count_nonzero(truth == anchor_color) != 1:
        return False

    for color in np.unique(truth):
        if int(color) in (BG, anchor_color):
            continue
        if int(np.count_nonzero(truth == color)) < 2:
            return False
        if int(np.count_nonzero(truth[1:, 1:] == color)) == 0:
            return False

    return True


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    size = random.randint(4, 7)
    truth = _make_template(size)
    copies, rotations = _build_copies(truth)
    inp = _place_copies(copies)
    assert _validate_case(inp, truth, rotations)
    return inp, truth


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, _truth = _generate_case()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input grid")

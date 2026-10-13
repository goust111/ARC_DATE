from __future__ import annotations

import random

import numpy as np


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _connected(mask: np.ndarray, value: bool = True) -> bool:
    h, w = map(int, mask.shape)
    coords = np.argwhere(mask == value)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) == bool(value):
                cell = (int(rr), int(cc))
                if cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == int(coords.shape[0])


def _covers_full_bbox(mask: np.ndarray, value: bool = True) -> bool:
    coords = np.argwhere(mask == value)
    if coords.size == 0:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()) == 0
        and int(rows.max()) == int(mask.shape[0]) - 1
        and int(cols.min()) == 0
        and int(cols.max()) == int(mask.shape[1]) - 1
    )


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes()


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            coords = np.array(cells, dtype=int)
            r0, c0 = map(int, coords.min(axis=0))
            r1, c1 = map(int, coords.max(axis=0))
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(coords.shape[0]),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary colored fragments into filled rectangles at the larger fragment locations.

    Input structure:
    - bg = Black.
    - The grid contains several colored fragments, each a single 8-connected monochrome region.
    - Every fragment's bounding box has the same height and width, and each fragment spans that full box.
    - Fragments come in pairs whose cropped shape masks are exact complements inside that common box size.

    Rule:
    - Extract every fragment and crop it to its own bounding box to obtain a binary shape mask.
    - Pair each fragment with the unique other fragment whose mask is the cellwise complement of its mask.
    - Within each pair, compare areas and take the smaller fragment's color as the output color.
    - Remove the smaller fragment from its original location, then fill the larger fragment's entire bounding box as a solid rectangle in the smaller fragment's color.

    Key insight:
    - The important relation is mask complementarity inside equal bounding boxes, not the fragment colors or positions; the smaller piece supplies the color while the larger piece supplies the destination rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0
    components = _extract_components(g, bg)
    assert components and len(components) % 2 == 0, "expected an even number of fragments"

    box_shapes = {tuple(map(int, comp["mask"].shape)) for comp in components}
    assert len(box_shapes) == 1, "all fragment bounding boxes must share one size"

    by_key: dict[bytes, list[int]] = {}
    for idx, comp in enumerate(components):
        by_key.setdefault(_mask_key(comp["mask"]), []).append(idx)

    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for idx, comp in enumerate(components):
        if idx in used:
            continue
        complement_key = _mask_key(~comp["mask"])
        candidates = [j for j in by_key.get(complement_key, []) if j not in used and j != idx]
        assert len(candidates) == 1, "each fragment must have one unique complementary partner"
        mate = int(candidates[0])
        used.add(idx)
        used.add(mate)
        pairs.append((idx, mate))

    out = g.copy()
    for ia, ib in pairs:
        a = components[ia]
        b = components[ib]
        assert a["mask"].shape == b["mask"].shape
        assert not np.any(a["mask"] & b["mask"])
        assert bool(np.all(a["mask"] | b["mask"]))
        assert a["area"] != b["area"], "pair areas must be unequal"

        small, large = (a, b) if int(a["area"]) < int(b["area"]) else (b, a)
        out[small["coords"][:, 0], small["coords"][:, 1]] = bg
        r0, c0, r1, c1 = map(int, large["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(small["color"])

    return out


# --------------- generator ---------------

BG = 0
BOX_HEIGHT_OPTIONS = (4, 5, 6)
BOX_WIDTH_OPTIONS = (4, 5, 6)
PAIR_COUNT_OPTIONS = (2, 3, 4)
MIN_MARGIN = 1
MAX_EXTRA_SPACE = 8
SHAPE_ATTEMPTS = 200
PAIR_ATTEMPTS = 200
GENERATION_ATTEMPTS = 300


def _random_partition(total: int, parts: int) -> list[int]:
    values = [0] * int(parts)
    for _ in range(int(total)):
        values[random.randrange(int(parts))] += 1
    return values


def _random_connected_mask(h: int, w: int, target_area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}

    while len(cells) < int(target_area):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for rr, cc in _neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((int(rr), int(cc)))
        assert frontier, "frontier should remain nonempty before the mask fills the box"

        hit_top = any(r == 0 for r, _ in cells)
        hit_bottom = any(r == h - 1 for r, _ in cells)
        hit_left = any(c == 0 for _, c in cells)
        hit_right = any(c == w - 1 for _, c in cells)

        choices = list(frontier)
        weights = []
        for r, c in choices:
            orth_neighbors = 0
            all_neighbors = 0
            for rr, cc in _neighbors8(r, c):
                if (rr, cc) in cells:
                    all_neighbors += 1
                    if rr == r or cc == c:
                        orth_neighbors += 1
            border_bonus = 0
            border_bonus += 2 if (not hit_top and r == 0) else 0
            border_bonus += 2 if (not hit_bottom and r == h - 1) else 0
            border_bonus += 2 if (not hit_left and c == 0) else 0
            border_bonus += 2 if (not hit_right and c == w - 1) else 0
            weights.append(1 + 2 * orth_neighbors + all_neighbors + border_bonus)

        cells.add(random.choices(choices, weights=weights, k=1)[0])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_mask(h: int, w: int) -> np.ndarray | None:
    area = int(h) * int(w)
    min_area = max(4, int(np.ceil(area * 0.30)))
    max_area = min(area - 4, int(np.floor(area * 0.70)))
    if min_area > max_area:
        return None

    choices = [a for a in range(min_area, max_area + 1) if a * 2 != area]
    if not choices:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.choice(choices)
        mask = _random_connected_mask(h, w, target_area)
        if not _covers_full_bbox(mask, True):
            continue
        if not _covers_full_bbox(mask, False):
            continue
        if not _connected(mask, True):
            continue
        if not _connected(mask, False):
            continue
        return mask
    return None


def _layout_positions(n_fragments: int, box_h: int, box_w: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    candidates: list[tuple[int, int]] = []
    for rows in range(1, int(n_fragments) + 1):
        cols = (int(n_fragments) + rows - 1) // rows
        base_h = rows * int(box_h) + (rows + 1) * MIN_MARGIN
        base_w = cols * int(box_w) + (cols + 1) * MIN_MARGIN
        if base_h <= 30 and base_w <= 30:
            candidates.append((rows, cols))
    if not candidates:
        return None

    rows, cols = random.choice(candidates)
    base_h = rows * int(box_h) + (rows + 1) * MIN_MARGIN
    base_w = cols * int(box_w) + (cols + 1) * MIN_MARGIN
    extra_h = random.randint(0, min(MAX_EXTRA_SPACE, 30 - base_h))
    extra_w = random.randint(0, min(MAX_EXTRA_SPACE, 30 - base_w))
    gaps_h = [MIN_MARGIN + x for x in _random_partition(extra_h, rows + 1)]
    gaps_w = [MIN_MARGIN + x for x in _random_partition(extra_w, cols + 1)]

    grid_h = rows * int(box_h) + sum(gaps_h)
    grid_w = cols * int(box_w) + sum(gaps_w)

    row_starts: list[int] = []
    pos = gaps_h[0]
    for idx in range(rows):
        row_starts.append(int(pos))
        pos += int(box_h) + gaps_h[idx + 1]

    col_starts: list[int] = []
    pos = gaps_w[0]
    for idx in range(cols):
        col_starts.append(int(pos))
        pos += int(box_w) + gaps_w[idx + 1]

    slots = [(int(r0), int(c0)) for r0 in row_starts for c0 in col_starts]
    chosen = random.sample(slots, int(n_fragments))
    random.shuffle(chosen)
    return int(grid_h), int(grid_w), chosen


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    shifted = coords + np.array([top, left], dtype=int)
    grid[shifted[:, 0], shifted[:, 1]] = int(color)
    return shifted


def _build_example() -> np.ndarray | None:
    box_h = random.choice(BOX_HEIGHT_OPTIONS)
    box_w = random.choice(BOX_WIDTH_OPTIONS)
    n_pairs = random.choice(PAIR_COUNT_OPTIONS)
    n_fragments = 2 * int(n_pairs)

    layout = _layout_positions(n_fragments, box_h, box_w)
    if layout is None:
        return None
    grid_h, grid_w, positions = layout

    used_keys: set[bytes] = set()
    pair_masks: list[np.ndarray] = []
    for _ in range(n_pairs):
        mask = None
        for _ in range(PAIR_ATTEMPTS):
            candidate = _sample_mask(box_h, box_w)
            if candidate is None:
                return None
            key = _mask_key(candidate)
            comp_key = _mask_key(~candidate)
            if key in used_keys or comp_key in used_keys:
                continue
            used_keys.add(key)
            used_keys.add(comp_key)
            mask = candidate
            break
        if mask is None:
            return None
        pair_masks.append(mask)

    colors = random.sample(range(1, 10), n_fragments)
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = inp.copy()

    pos_iter = iter(positions)
    for pair_idx, mask in enumerate(pair_masks):
        comp = ~mask
        top_a, left_a = next(pos_iter)
        top_b, left_b = next(pos_iter)
        color_a = colors[2 * pair_idx]
        color_b = colors[2 * pair_idx + 1]

        if random.random() < 0.5:
            mask_a, mask_b = mask, comp
        else:
            mask_a, mask_b = comp, mask

        coords_a = _paint_mask(inp, top_a, left_a, mask_a, color_a)
        coords_b = _paint_mask(inp, top_b, left_b, mask_b, color_b)

        area_a = int(mask_a.sum())
        area_b = int(mask_b.sum())
        assert area_a != area_b

        expected[coords_a[:, 0], coords_a[:, 1]] = color_a
        expected[coords_b[:, 0], coords_b[:, 1]] = color_b

        if area_a > area_b:
            large_top, large_left = top_a, left_a
            small_coords = coords_b
            fill_color = color_b
        else:
            large_top, large_left = top_b, left_b
            small_coords = coords_a
            fill_color = color_a

        expected[small_coords[:, 0], small_coords[:, 1]] = BG
        expected[large_top : large_top + box_h, large_left : large_left + box_w] = fill_color

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")

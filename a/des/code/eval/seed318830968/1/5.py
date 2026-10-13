from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _crop_bool_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    pieces: list[np.ndarray] = []

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
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            piece = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                piece[rr - r0, cc - c0] = True
            pieces.append(piece)

    return pieces


def _parse_marker_row(row: np.ndarray, bg: int) -> tuple[int, int] | None:
    row = np.asarray(row, dtype=int)
    w = row.shape[0]
    c = 0
    while c < w and int(row[c]) == bg:
        c += 1
    if c == w:
        return None

    marker_color = int(row[c])
    groups: list[int] = []
    while c < w:
        if int(row[c]) == bg:
            return None
        start = c
        while c < w and int(row[c]) == marker_color:
            c += 1
        if np.any(row[start:c] != marker_color):
            return None
        groups.append(c - start)
        gap_start = c
        while c < w and int(row[c]) == bg:
            c += 1
        if c < w and c - gap_start != 1:
            return None
        if c < w and int(row[c]) != marker_color:
            return None

    expected = list(range(1, len(groups) + 1))
    if groups != expected:
        return None
    return marker_color, len(groups)


def _find_marker_row(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    for r in range(grid.shape[0] - 1, -1, -1):
        parsed = _parse_marker_row(grid[r], bg)
        if parsed is not None:
            marker_color, count = parsed
            return r, marker_color, count
    raise AssertionError("marker row not found")


def _placement_bits(coords: list[tuple[int, int]], rect_w: int, off_r: int, off_c: int) -> int:
    bits = 0
    for rr, cc in coords:
        bits |= 1 << ((off_r + rr) * rect_w + (off_c + cc))
    return bits


def _enumerate_packings(fragment_masks: list[np.ndarray], limit: int = 3) -> set[tuple[int, int, tuple[int, ...]]]:
    total_area = sum(int(mask.sum()) for mask in fragment_masks)
    fragments = []
    for mask in fragment_masks:
        coords = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
        fragments.append(
            {
                "coords": coords,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "area": int(mask.sum()),
            }
        )

    packings: set[tuple[int, int, tuple[int, ...]]] = set()
    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        placements: list[list[int]] = []
        feasible = True
        for frag in fragments:
            if frag["h"] > rect_h or frag["w"] > rect_w:
                feasible = False
                break
            frag_bits: list[int] = []
            for off_r in range(rect_h - frag["h"] + 1):
                for off_c in range(rect_w - frag["w"] + 1):
                    frag_bits.append(_placement_bits(frag["coords"], rect_w, off_r, off_c))
            if not frag_bits:
                feasible = False
                break
            placements.append(frag_bits)
        if not feasible:
            continue

        order = sorted(range(len(fragments)), key=lambda i: (len(placements[i]), -fragments[i]["area"]))

        def search(idx: int, used: int, chosen: list[int]) -> None:
            if len(packings) >= limit:
                return
            if idx == len(order):
                packings.add((rect_h, rect_w, tuple(sorted(chosen))))
                return
            frag_idx = order[idx]
            for bits in placements[frag_idx]:
                if bits & used:
                    continue
                search(idx + 1, used | bits, chosen + [bits])
                if len(packings) >= limit:
                    return

        search(0, 0, [])
        if len(packings) >= limit:
            return packings
    return packings


def _reconstruct_rectangle(fragment_masks: list[np.ndarray]) -> tuple[int, int]:
    packings = _enumerate_packings(fragment_masks, limit=8)
    dims = {(int(h), int(w)) for h, w, _signature in packings}
    assert len(dims) == 1, f"ambiguous rectangle dimensions: {dims}"
    return next(iter(dims))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored fragments into rectangles, sort the rectangles by area,
    and pack them left-to-right at the top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    marker_row, marker_color, marker_count = _find_marker_row(g, bg)
    content = g[:marker_row, :]

    colors = [int(v) for v in np.unique(content) if int(v) not in (bg, marker_color)]
    assert len(colors) == marker_count

    rectangles: list[dict[str, int]] = []
    for color in colors:
        fragments = _component_masks(content == color)
        assert 2 <= len(fragments) <= 3
        rect_h, rect_w = _reconstruct_rectangle(fragments)
        rectangles.append(
            {
                "color": int(color),
                "h": int(rect_h),
                "w": int(rect_w),
                "area": int(rect_h * rect_w),
            }
        )

    rectangles.sort(key=lambda item: (item["area"], item["h"], item["w"]))
    out_h = max(item["h"] for item in rectangles)
    out_w = sum(item["w"] for item in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for rect in rectangles:
        out[: rect["h"], col : col + rect["w"]] = rect["color"]
        col += rect["w"] + 1
    return out


# --------------- generator (gpt-5) ---------------

BG = 0
MARKER_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GROUP_COUNT_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (2, 6)
RECT_WIDTH_RANGE = (3, 7)
FRAGMENT_COUNT_CHOICES = (2, 3)
MIN_FRAGMENT_AREA = 2

INPUT_SIZE_MIN = 18
INPUT_SIZE_MAX = 30
PLACE_ATTEMPTS = 250
GROUP_ATTEMPTS = 180
GENERATION_ATTEMPTS = 300


def _is_rectangular(mask: np.ndarray) -> bool:
    return int(mask.sum()) == int(mask.shape[0] * mask.shape[1])


def _random_connected_partition(rect_h: int, rect_w: int, n_parts: int) -> list[np.ndarray] | None:
    cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    seeds = random.sample(cells, n_parts)
    owner = -np.ones((rect_h, rect_w), dtype=int)
    for idx, (r, c) in enumerate(seeds):
        owner[r, c] = idx

    unassigned = set(cells) - set(seeds)
    while unassigned:
        options: list[tuple[tuple[int, int], int]] = []
        for r, c in unassigned:
            seen_regions = set()
            for dr, dc in _NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rect_h and 0 <= nc < rect_w and owner[nr, nc] != -1:
                    seen_regions.add(int(owner[nr, nc]))
            for region in seen_regions:
                options.append(((r, c), region))
        if not options:
            return None
        (r, c), region = random.choice(options)
        owner[r, c] = region
        unassigned.remove((r, c))

    pieces = [owner == idx for idx in range(n_parts)]
    if any(int(piece.sum()) < MIN_FRAGMENT_AREA for piece in pieces):
        return None
    if all(_is_rectangular(_crop_bool_mask(piece)) for piece in pieces):
        return None
    return [_crop_bool_mask(piece) for piece in pieces]


def _make_group(used_areas: set[int]) -> dict | None:
    for _ in range(GROUP_ATTEMPTS):
        rect_h = random.randint(*RECT_HEIGHT_RANGE)
        rect_w = random.randint(*RECT_WIDTH_RANGE)
        area = rect_h * rect_w
        if area in used_areas:
            continue
        n_parts = random.choice(FRAGMENT_COUNT_CHOICES)
        if area < n_parts * MIN_FRAGMENT_AREA:
            continue

        fragments = _random_connected_partition(rect_h, rect_w, n_parts)
        if fragments is None:
            continue
        packings = _enumerate_packings(fragments, limit=2)
        if len(packings) != 1:
            continue
        only = next(iter(packings))
        if only[:2] != (rect_h, rect_w):
            continue

        return {
            "h": rect_h,
            "w": rect_w,
            "area": area,
            "n_parts": n_parts,
            "fragments": fragments,
        }
    return None


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _marker_width(n_groups: int) -> int:
    return n_groups * (n_groups + 1) // 2 + (n_groups - 1)


def _choose_canvas(patches: list[np.ndarray], n_groups: int) -> tuple[int, int] | None:
    total_bbox_area = sum(int(np.prod(patch.shape)) for patch in patches)
    marker_w = _marker_width(n_groups)
    min_h = max(INPUT_SIZE_MIN, max(int(patch.shape[0]) for patch in patches) + 4)
    min_w = max(INPUT_SIZE_MIN, marker_w + 2, max(int(patch.shape[1]) for patch in patches) + 2)

    base = int(np.ceil(np.sqrt(max(1, total_bbox_area) * 3.2)))
    min_h = max(min_h, base)
    min_w = max(min_w, base + random.randint(0, 2))
    if min_h > INPUT_SIZE_MAX or min_w > INPUT_SIZE_MAX:
        return None

    height = random.randint(min_h, INPUT_SIZE_MAX)
    width = random.randint(min_w, INPUT_SIZE_MAX)
    return height, width


def _place_patches(grid: np.ndarray, patches: list[np.ndarray]) -> bool:
    occupied = np.zeros_like(grid, dtype=bool)
    # Reserve a blank separator row plus the bottom marker row.
    occupied[-2:, :] = True

    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: int((patches[idx] != BG).sum()), reverse=True)

    for idx in order:
        patch = patches[idx]
        mask = patch != BG
        ph, pw = patch.shape
        max_r = grid.shape[0] - 2 - ph
        max_c = grid.shape[1] - pw
        if max_r < 0 or max_c < 0:
            return False

        placed = False
        blocked = _dilate8(occupied)
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(0, max_r)
            c0 = random.randint(0, max_c)
            if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask):
                continue
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            view[mask] = patch[mask]
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
            placed = True
            break
        if not placed:
            return False
    return True


def _paint_marker_row(grid: np.ndarray, n_groups: int) -> None:
    row = grid.shape[0] - 1
    width = _marker_width(n_groups)
    start = random.randint(0, grid.shape[1] - width)
    c = start
    for size in range(1, n_groups + 1):
        grid[row, c : c + size] = MARKER_COLOR
        c += size
        if size != n_groups:
            c += 1


def _expected_output(groups: list[dict[str, object]]) -> np.ndarray:
    ordered = sorted(groups, key=lambda item: (int(item["area"]), int(item["h"]), int(item["w"])))
    out_h = max(int(item["h"]) for item in ordered)
    out_w = sum(int(item["w"]) for item in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for item in ordered:
        color = int(item["color"])
        h = int(item["h"])
        w = int(item["w"])
        out[:h, col : col + w] = color
        col += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.randint(*GROUP_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, n_groups)

    groups = []
    used_areas: set[int] = set()
    for color in colors:
        group = _make_group(used_areas)
        if group is None:
            return None
        group["color"] = int(color)
        groups.append(group)
        used_areas.add(int(group["area"]))

    output = _expected_output(groups)
    if output.shape[0] > INPUT_SIZE_MAX or output.shape[1] > INPUT_SIZE_MAX:
        return None

    patches = []
    for group in groups:
        color = int(group["color"])
        fragments = list(group["fragments"])
        random.shuffle(fragments)
        for mask in fragments:
            patches.append(_render_patch(mask, color))

    canvas = _choose_canvas(patches, n_groups)
    if canvas is None:
        return None
    height, width = canvas
    grid = np.full((height, width), BG, dtype=int)
    if not _place_patches(grid, patches):
        return None
    _paint_marker_row(grid, n_groups)

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

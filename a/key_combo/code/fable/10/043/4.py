from __future__ import annotations

import random
from collections import Counter

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    neighbors = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _piece_from_coords(color_patch: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    ordered_abs = sorted((int(r), int(c)) for r, c in coords)
    rel = tuple((r - r0, c - c0) for r, c in ordered_abs)
    colors = tuple(int(color_patch[r, c]) for r, c in ordered_abs)
    assert all(color != bg for color in colors)
    return {
        "coords": rel,
        "colors": colors,
        "area": len(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _enumerate_piece_placements(piece: dict, slot_mask: np.ndarray) -> list[tuple[tuple[int, int, int], ...]]:
    slot_h, slot_w = slot_mask.shape
    placements: list[tuple[tuple[int, int, int], ...]] = []
    for r0 in range(slot_h - piece["h"] + 1):
        for c0 in range(slot_w - piece["w"] + 1):
            placed = []
            ok = True
            for (dr, dc), color in zip(piece["coords"], piece["colors"]):
                rr = r0 + dr
                cc = c0 + dc
                if not slot_mask[rr, cc]:
                    ok = False
                    break
                placed.append((rr, cc, int(color)))
            if ok:
                placements.append(tuple(placed))
    return placements


def _pack_slot(
    slot_mask: np.ndarray,
    pieces: list[dict],
    bg: int,
    limit: int = 2,
) -> tuple[int, np.ndarray | None, list[int]]:
    placements = [_enumerate_piece_placements(piece, slot_mask) for piece in pieces]
    candidate_counts = [len(opts) for opts in placements]

    covered = np.zeros(slot_mask.shape, dtype=bool)
    painted = np.full(slot_mask.shape, bg, dtype=int)
    found: dict[bytes, np.ndarray] = {}

    def rec(remaining: tuple[int, ...]) -> None:
        if len(found) >= limit:
            return
        if not remaining:
            if np.array_equal(covered, slot_mask):
                key = painted.tobytes()
                if key not in found:
                    found[key] = painted.copy()
            return

        chosen = None
        chosen_opts: list[tuple[tuple[int, int, int], ...]] | None = None
        for idx in remaining:
            valid = []
            for option in placements[idx]:
                if all(not covered[r, c] for r, c, _ in option):
                    valid.append(option)
            if not valid:
                return
            if chosen_opts is None or len(valid) < len(chosen_opts):
                chosen = idx
                chosen_opts = valid

        assert chosen is not None and chosen_opts is not None
        nxt = tuple(idx for idx in remaining if idx != chosen)
        for option in chosen_opts:
            changed = []
            for r, c, color in option:
                covered[r, c] = True
                painted[r, c] = color
                changed.append((r, c))
            rec(nxt)
            for r, c in changed:
                covered[r, c] = False
                painted[r, c] = bg
            if len(found) >= limit:
                return

    rec(tuple(range(len(pieces))))
    sample = next(iter(found.values())) if found else None
    return len(found), sample, candidate_counts


def _find_mask_and_pattern(
    grid: np.ndarray,
    bg: int,
) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    non_bg = grid != bg
    whole_components = _components_from_mask(non_bg, connectivity=8)
    whole_label = np.full(grid.shape, -1, dtype=int)
    whole_area: list[int] = []
    for idx, comp in enumerate(whole_components):
        whole_label[comp[:, 0], comp[:, 1]] = idx
        whole_area.append(int(comp.shape[0]))

    color_totals = Counter(map(int, grid.flatten()))
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for comp in _components_from_mask(grid == color, connectivity=4):
            area = int(comp.shape[0])
            if area <= 1:
                continue
            rect = _bbox(comp)
            r0, c0, r1, c1 = rect
            if area != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if int(color_totals[color]) != area:
                continue
            cid = int(whole_label[comp[0, 0], comp[0, 1]])
            if cid < 0 or whole_area[cid] == area:
                continue
            candidates.append((int(color), rect, whole_components[cid]))

    assert len(candidates) == 1, "expected exactly one embedded solid mask rectangle"
    return candidates[0]


def _slot_mask_from_mirror(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    pattern_bbox: tuple[int, int, int, int],
    bg: int,
    mask_color: int,
) -> np.ndarray:
    r0, c0, r1, c1 = rect
    _pr0, pc0, _pr1, pc1 = pattern_bbox
    slot = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mc = pc0 + pc1 - c
            value = int(grid[r, mc])
            assert value != mask_color, "mask may not overlap its own mirror partner"
            slot[r - r0, c - c0] = value != bg
    return slot


def _recover_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    mask_color, rect, pattern_coords = _find_mask_and_pattern(g, bg)
    pattern_mask = np.zeros(g.shape, dtype=bool)
    pattern_mask[pattern_coords[:, 0], pattern_coords[:, 1]] = True
    pattern_bbox = _bbox(pattern_coords)
    slot_mask = _slot_mask_from_mirror(g, rect, pattern_bbox, bg, mask_color)

    outside_mask = (g != bg) & ~pattern_mask
    fragment_components = _components_from_mask(outside_mask, connectivity=8)
    assert fragment_components, "expected scattered fragments outside the masked figure"

    pieces = []
    for comp in fragment_components:
        coords = [(int(r), int(c)) for r, c in comp]
        pieces.append(_piece_from_coords(g, coords, bg))

    assert sum(piece["area"] for piece in pieces) == int(slot_mask.sum())

    n_solutions, solved, _ = _pack_slot(slot_mask, pieces, bg, limit=2)
    assert n_solutions == 1 and solved is not None, "slot assembly must have a unique output"
    return solved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the masked rectangle by mirroring only the silhouette and filling that slot with the scattered fragments.

    Input structure:
    - A central multi-color figure has left-right symmetric occupancy, but mirrored cells need not share colors.
    - One solid rectangle of a unique mask color covers part of that figure; some masked cells belong to the figure and some are background.
    - Outside the figure, several 8-connected colored fragments are scattered with gaps between them.

    Rule:
    - Find the embedded solid mask rectangle and the larger figure containing it.
    - Use the figure's left-right silhouette symmetry to determine which cells inside the mask rectangle should be foreground.
    - Collect every outside fragment and translate the fragments, without rotation or reflection and without overlap, so that they exactly cover that inferred slot shape.
    - Output only the mask rectangle crop: fragment colors where the slot is filled, background elsewhere.

    Key insight:
    - Mirror symmetry supplies geometry only: it reveals where matter belongs inside the mask, not which colors belong there.
    - The colors come exclusively from the scattered fragments, so both the mirrored silhouette and the jigsaw packing are necessary.
    """
    return _recover_output(grid)


GEN_ATTEMPTS = 800
BG_CHOICES = tuple(range(10))
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (20, 30)
PATTERN_H_RANGE = (8, 13)
PATTERN_W_OPTIONS = (9, 11, 13, 15)
MASK_H_RANGE = (3, 6)
MASK_W_RANGE = (3, 5)
HIDDEN_AREA_RANGE = (6, 14)
PIECE_COUNT_RANGE = (2, 4)


def _build_symmetric_shape(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    axis = width // 2
    widths = []
    cur = rng.randint(max(1, axis // 2), axis)
    for r in range(height):
        if r in (0, height - 1):
            cur = min(cur, rng.randint(1, max(1, axis - 1)))
        widths.append(cur)
        step = rng.choice([-2, -1, 0, 1, 2])
        cur = max(1, min(axis, cur + step))
    if max(widths) - min(widths) < 2:
        return None

    shape = np.zeros((height, width), dtype=bool)
    for r, half_w in enumerate(widths):
        shape[r, axis - half_w : axis + half_w + 1] = True
    return shape


def _local_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = _components_from_mask(mask, connectivity=8)
    return [[(int(r), int(c)) for r, c in comp] for comp in comps]


def _find_mask_rectangles(shape: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = shape.shape
    axis = width // 2
    rects = []
    for h in range(MASK_H_RANGE[0], min(MASK_H_RANGE[1], height) + 1):
        for w in range(MASK_W_RANGE[0], min(MASK_W_RANGE[1], width - 1) + 1):
            for r0 in range(height - h + 1):
                for c0 in range(width - w + 1):
                    c1 = c0 + w - 1
                    if c1 >= axis and c0 <= axis:
                        continue
                    sub = shape[r0 : r0 + h, c0 : c0 + w]
                    area = int(sub.sum())
                    if area < HIDDEN_AREA_RANGE[0] or area > HIDDEN_AREA_RANGE[1]:
                        continue
                    if area == h * w:
                        continue
                    comps = _local_components(sub)
                    if len(comps) != 1:
                        continue
                    rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return rects


def _partition_connected(
    coords: list[tuple[int, int]],
    n_parts: int,
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    coords_set = set(coords)
    if n_parts <= 1 or n_parts > len(coords_set):
        return None
    seeds = rng.sample(coords, n_parts)
    regions = [{seed} for seed in seeds]
    remaining = coords_set - set(seeds)

    while remaining:
        candidates = []
        for cell in remaining:
            choices = []
            cr, cc = cell
            for idx, region in enumerate(regions):
                if any((cr + dr, cc + dc) in region for dr, dc in NEIGHBORS8):
                    choices.append(idx)
            if choices:
                candidates.append((cell, choices))
        if not candidates:
            return None
        cell, choices = rng.choice(candidates)
        regions[rng.choice(choices)].add(cell)
        remaining.remove(cell)

    out = [sorted(region) for region in regions]
    if any(len(region) == 0 for region in out):
        return None
    return out


def _mirror_patch(pattern: np.ndarray, rect: tuple[int, int, int, int], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = rect
    width = pattern.shape[1]
    mirrored = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            mirrored[r - r0, c - c0] = int(pattern[r, width - 1 - c])
    return mirrored


def _sample_pattern_and_pieces(
    rng: random.Random,
) -> tuple[int, int, np.ndarray, tuple[int, int, int, int], np.ndarray, list[dict]] | None:
    bg = rng.choice(BG_CHOICES)
    height = rng.randint(*PATTERN_H_RANGE)
    width = rng.choice(PATTERN_W_OPTIONS)
    shape = _build_symmetric_shape(height, width, rng)
    if shape is None:
        return None

    rects = _find_mask_rectangles(shape)
    if not rects:
        return None
    rect = rng.choice(rects)
    r0, c0, r1, c1 = rect
    slot_mask = shape[r0 : r1 + 1, c0 : c1 + 1].copy()

    mask_color = rng.choice([color for color in range(10) if color != bg])
    palette_pool = [color for color in range(10) if color not in (bg, mask_color)]
    palette = rng.sample(palette_pool, rng.randint(3, min(5, len(palette_pool))))
    pattern = np.full((height, width), bg, dtype=int)
    for r in range(height):
        for c in range(width):
            if shape[r, c]:
                pattern[r, c] = rng.choice(palette)

    expected = pattern[r0 : r1 + 1, c0 : c1 + 1].copy()
    mirrored_colors = _mirror_patch(pattern, rect, bg)
    if not np.any(expected[slot_mask] != mirrored_colors[slot_mask]):
        return None

    hidden_coords = [(int(r), int(c)) for r, c in np.argwhere(slot_mask)]
    max_parts = min(PIECE_COUNT_RANGE[1], len(hidden_coords) - 1)
    if max_parts < PIECE_COUNT_RANGE[0]:
        return None

    for _ in range(60):
        n_parts = rng.randint(PIECE_COUNT_RANGE[0], max_parts)
        regions = _partition_connected(hidden_coords, n_parts, rng)
        if regions is None:
            continue
        if max(len(region) for region in regions) == 1:
            continue
        pieces = [_piece_from_coords(expected, region, bg) for region in regions]
        n_solutions, solved, candidate_counts = _pack_slot(slot_mask, pieces, bg, limit=2)
        if n_solutions != 1 or solved is None or not np.array_equal(solved, expected):
            continue
        if not any(count > 1 for count in candidate_counts):
            continue
        if not any(piece["area"] < piece["h"] * piece["w"] for piece in pieces):
            continue
        return bg, mask_color, pattern, rect, expected, pieces

    return None


def _can_place_with_gap(
    occupied: np.ndarray,
    piece: dict,
    r0: int,
    c0: int,
) -> bool:
    height, width = occupied.shape
    for dr, dc in piece["coords"]:
        rr = r0 + dr
        cc = c0 + dc
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < height and 0 <= nc < width and occupied[nr, nc]:
                    return False
    return True


def _paint_piece(grid: np.ndarray, occupied: np.ndarray, piece: dict, r0: int, c0: int) -> None:
    for (dr, dc), color in zip(piece["coords"], piece["colors"]):
        rr = r0 + dr
        cc = c0 + dc
        grid[rr, cc] = int(color)
        occupied[rr, cc] = True


def _scatter_pieces(
    grid: np.ndarray,
    occupied: np.ndarray,
    pieces: list[dict],
    rng: random.Random,
) -> bool:
    height, width = grid.shape
    order = list(range(len(pieces)))
    rng.shuffle(order)
    for idx in order:
        piece = pieces[idx]
        options = []
        for r0 in range(height - piece["h"] + 1):
            for c0 in range(width - piece["w"] + 1):
                if _can_place_with_gap(occupied, piece, r0, c0):
                    options.append((r0, c0))
        if not options:
            return False
        r0, c0 = rng.choice(options)
        _paint_piece(grid, occupied, piece, r0, c0)
    return True


def _build_global_input(
    bg: int,
    mask_color: int,
    pattern: np.ndarray,
    rect: tuple[int, int, int, int],
    pieces: list[dict],
    rng: random.Random,
) -> np.ndarray | None:
    ph, pw = pattern.shape
    min_h = max(GRID_H_RANGE[0], ph + 8)
    max_h = min(GRID_H_RANGE[1], 30)
    min_w = max(GRID_W_RANGE[0], pw + 10)
    max_w = min(GRID_W_RANGE[1], 30)
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(80):
        height = rng.randint(min_h, max_h)
        width = rng.randint(min_w, max_w)
        grid = np.full((height, width), bg, dtype=int)
        occupied = np.zeros((height, width), dtype=bool)

        pr0 = (height - ph) // 2
        pc0 = (width - pw) // 2
        for r in range(ph):
            for c in range(pw):
                value = int(pattern[r, c])
                if value != bg:
                    grid[pr0 + r, pc0 + c] = value
                    occupied[pr0 + r, pc0 + c] = True

        r0, c0, r1, c1 = rect
        gr0 = pr0 + r0
        gc0 = pc0 + c0
        gr1 = pr0 + r1
        gc1 = pc0 + c1
        grid[gr0 : gr1 + 1, gc0 : gc1 + 1] = mask_color
        occupied[gr0 : gr1 + 1, gc0 : gc1 + 1] = True

        if _scatter_pieces(grid, occupied, pieces, rng):
            return grid

    return None


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    pattern: np.ndarray,
    rect: tuple[int, int, int, int],
    pieces: list[dict],
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    r0, c0, r1, c1 = rect
    slot_mask = pattern[r0 : r1 + 1, c0 : c1 + 1] != bg
    if not (0 < int(slot_mask.sum()) < slot_mask.size):
        return False
    if len(pieces) < 2:
        return False
    if sum(piece["area"] for piece in pieces) != int(slot_mask.sum()):
        return False

    mirrored = _mirror_patch(pattern, rect, bg)
    if not np.any(expected[slot_mask] != mirrored[slot_mask]):
        return False

    n_solutions, solved, candidate_counts = _pack_slot(slot_mask, pieces, bg, limit=2)
    if n_solutions != 1 or solved is None or not np.array_equal(solved, expected):
        return False
    if not any(count > 1 for count in candidate_counts):
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    sampled = _sample_pattern_and_pieces(rng)
    if sampled is None:
        return None
    bg, mask_color, pattern, rect, expected, pieces = sampled
    grid = _build_global_input(bg, mask_color, pattern, rect, pieces, rng)
    if grid is None:
        return None
    if not _semantic_checks(grid, expected, bg, pattern, rect, pieces):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid silhouette-slot puzzle input")

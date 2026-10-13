from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
FRAME_CHOICES = ((4, 4), (4, 6), (6, 4), (6, 6))
MAX_GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            comp_id += 1
            queue = deque([(int(r), int(c))])
            labels[r, c] = int(comp_id)
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = int(comp_id)
                        queue.append((int(nr), int(nc)))
    return labels, int(comp_id)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _apply_dihedral(coords: np.ndarray, op: int) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    out = pts.copy()
    if bool(op & 4):
        out[:, 1] = -out[:, 1]
    for _ in range(int(op & 3)):
        out = np.stack((out[:, 1], -out[:, 0]), axis=1)
    out -= out.min(axis=0)
    return out


def _canonical_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.asarray(coords, dtype=int)
    best = None
    for op in range(8):
        variant = _apply_dihedral(pts, int(op))
        sig = tuple(sorted((int(r), int(c)) for r, c in variant.tolist()))
        if best is None or sig < best:
            best = sig
    assert best is not None
    return best


def _reflect_cell(cell: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
    r, c = cell
    return int(rows - 1 - r), int(cols - 1 - c)


def _all_orbits(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            cell = (int(r), int(c))
            if cell in seen:
                continue
            ref = _reflect_cell(cell, int(rows), int(cols))
            seen.add(cell)
            seen.add(ref)
            orbits.append((cell, ref))
    return orbits


def _boundary_seed_orbits(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            on_tb = int(r) in (0, int(rows - 1))
            on_lr = int(c) in (0, int(cols - 1))
            if int(on_tb) + int(on_lr) != 1:
                continue
            cell = (int(r), int(c))
            ref = _reflect_cell(cell, int(rows), int(cols))
            if cell == ref or cell in seen or ref in seen:
                continue
            seen.add(cell)
            seen.add(ref)
            out.append((cell, ref))
    return out


def _touches(cell: tuple[int, int], region: set[tuple[int, int]]) -> bool:
    r, c = cell
    return any((int(r + dr), int(c + dc)) in region for dr, dc in NEIGHBORS4)


def _weighted_index(weights: list[float], rng: random.Random) -> int:
    total = float(sum(weights))
    x = float(rng.random()) * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += float(w)
        if x <= acc:
            return int(i)
    return int(len(weights) - 1)


def _choose_seed_orbits(rows: int, cols: int, pair_count: int, rng: random.Random):
    options = _boundary_seed_orbits(int(rows), int(cols))
    assert len(options) >= int(pair_count)
    for _ in range(200):
        chosen = rng.sample(options, int(pair_count))
        seeds = [cell for orbit in chosen for cell in orbit]
        ok = True
        for i, a in enumerate(seeds):
            for b in seeds[i + 1 :]:
                if abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) < 2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return chosen
    return rng.sample(options, int(pair_count))


def _grow_partition(rows: int, cols: int, pair_count: int, rng: random.Random):
    seed_orbits = _choose_seed_orbits(int(rows), int(cols), int(pair_count), rng)
    left_regions = [{orbit[0]} for orbit in seed_orbits]
    right_regions = [{orbit[1]} for orbit in seed_orbits]
    seed_cells = {cell for orbit in seed_orbits for cell in orbit}
    unassigned = [orbit for orbit in _all_orbits(int(rows), int(cols)) if orbit[0] not in seed_cells and orbit[1] not in seed_cells]

    def assign(orbit: tuple[tuple[int, int], tuple[int, int]], region_idx: int, side: int) -> None:
        first, second = orbit
        if int(side) == 0:
            left_regions[int(region_idx)].add(first)
            right_regions[int(region_idx)].add(second)
        else:
            right_regions[int(region_idx)].add(first)
            left_regions[int(region_idx)].add(second)
        unassigned.remove(orbit)

    order = list(range(int(pair_count)))
    rng.shuffle(order)
    for region_idx in order:
        choices: list[tuple[tuple[tuple[int, int], tuple[int, int]], int]] = []
        for orbit in list(unassigned):
            first, _second = orbit
            if _touches(first, left_regions[int(region_idx)]):
                choices.append((orbit, 0))
            if _touches(first, right_regions[int(region_idx)]):
                choices.append((orbit, 1))
        if not choices:
            return None
        orbit, side = rng.choice(choices)
        assign(orbit, int(region_idx), int(side))

    while unassigned:
        frontier: list[tuple[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]] = []
        for orbit in list(unassigned):
            first, _second = orbit
            sides: list[tuple[int, int]] = []
            for region_idx in range(int(pair_count)):
                if _touches(first, left_regions[int(region_idx)]):
                    sides.append((int(region_idx), 0))
                if _touches(first, right_regions[int(region_idx)]):
                    sides.append((int(region_idx), 1))
            if sides:
                frontier.append((orbit, sides))
        if not frontier:
            return None
        orbit, sides = rng.choice(frontier)
        weights = [1.0 / max(1, len(left_regions[int(region_idx)])) for region_idx, _side in sides]
        chosen = sides[_weighted_index(weights, rng)]
        assign(orbit, int(chosen[0]), int(chosen[1]))

    regions: list[set[tuple[int, int]]] = []
    seeds: list[tuple[int, int]] = []
    for idx in range(int(pair_count)):
        regions.append(left_regions[int(idx)])
        seeds.append(seed_orbits[int(idx)][0])
        regions.append(right_regions[int(idx)])
        seeds.append(seed_orbits[int(idx)][1])
    return regions, seeds


def _region_pixels(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cellset = set(cells)
    pixels: set[tuple[int, int]] = set()
    for r, c in cellset:
        pixels.add((int(2 * r + 1), int(2 * c + 1)))
        if (int(r + 1), int(c)) in cellset:
            pixels.add((int(2 * r + 2), int(2 * c + 1)))
        if (int(r), int(c + 1)) in cellset:
            pixels.add((int(2 * r + 1), int(2 * c + 2)))
    return pixels


def _marker_pixel_from_seed(seed: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
    r, c = seed
    if int(r) == 0:
        return 0, int(2 * c + 1)
    if int(r) == int(rows - 1):
        return int(2 * rows), int(2 * c + 1)
    if int(c) == 0:
        return int(2 * r + 1), 0
    return int(2 * r + 1), int(2 * cols)


def _render_frame(
    rows: int,
    cols: int,
    regions: list[set[tuple[int, int]]],
    seeds: list[tuple[int, int]],
    colors: list[int],
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    h, w = int(2 * rows + 1), int(2 * cols + 1)
    frame = np.full((h, w), GREY, dtype=int)
    region_arrays: list[np.ndarray] = []
    marker_pixels: list[tuple[int, int]] = []
    for cells, seed, color in zip(regions, seeds, colors):
        pixels = np.array(sorted(_region_pixels(cells)), dtype=int)
        region_arrays.append(pixels)
        frame[pixels[:, 0], pixels[:, 1]] = BLACK
        marker = _marker_pixel_from_seed(seed, int(rows), int(cols))
        marker_pixels.append(marker)
        frame[int(marker[0]), int(marker[1])] = int(color)
    return frame, region_arrays, marker_pixels


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    labels, count = _label_components(grid != BLACK)
    assert count > 0
    h, w = grid.shape
    center_r = float(h - 1) / 2.0
    center_c = float(w - 1) / 2.0
    best_key = None
    best_bbox = None
    for comp_id in range(1, int(count) + 1):
        coords = np.argwhere(labels == int(comp_id))
        grey_count = int(np.count_nonzero(grid[labels == int(comp_id)] == GREY))
        if grey_count == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        comp_center_r = float(r0 + r1) / 2.0
        comp_center_c = float(c0 + c1) / 2.0
        center_penalty = abs(comp_center_r - center_r) + abs(comp_center_c - center_c)
        key = (int(grey_count), int(area), -float(center_penalty))
        if best_key is None or key > best_key:
            best_key = key
            best_bbox = (int(r0), int(c0), int(r1), int(c1))
    assert best_bbox is not None
    return best_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each scattered fragment into the matching framed slot and recolor it by the slot reached through point symmetry.

    Input structure:
    - The background is black.
    - A large grey rectangular frame sits near the center; inside it, grey walls partition the black interior into several enclosed regions.
    - Each region has exactly one colored marker pixel on its boundary, and the marker color also appears on one monochrome fragment outside the frame.
    - Every outside fragment has the same shape as its matching region up to rotation or reflection.

    Rule:
    - Detect the central framed partition and split its black interior into connected regions.
    - Match each region to the outside fragment with the same marker color, validating the match by shape up to the dihedral transforms.
    - Reflect the marker position through the frame center; the color at the reflected marker determines the final fill color for that region.
    - Remove all outside fragments and fill each region with its reflected partner color while keeping the frame and markers unchanged.

    Key insight:
    - Color identifies which fragment belongs to which slot, but the fill color comes from a second relation: the marker's 180-degree symmetric counterpart on the frame boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    r0, c0, r1, c1 = _find_frame_bbox(g)
    bbox_mask = np.zeros_like(g, dtype=bool)
    bbox_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    region_mask = np.zeros_like(g, dtype=bool)
    region_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == BLACK
    region_labels, region_count = _label_components(region_mask)
    assert region_count > 0

    marker_pixels = np.argwhere(bbox_mask & (g != BLACK) & (g != GREY))
    assert len(marker_pixels) == int(region_count)

    region_to_marker: dict[int, tuple[int, int, int]] = {}
    marker_color_at: dict[tuple[int, int], int] = {}
    for r, c in marker_pixels:
        adjacent = {
            int(region_labels[nr, nc])
            for nr, nc in _neighbors4(int(r), int(c))
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(region_labels[nr, nc]) > 0
        }
        assert len(adjacent) == 1
        region_id = next(iter(adjacent))
        region_to_marker[int(region_id)] = (int(r), int(c), int(g[int(r), int(c)]))
        marker_color_at[(int(r), int(c))] = int(g[int(r), int(c)])

    assert len(region_to_marker) == int(region_count)

    outside_mask = ~bbox_mask
    out = g.copy()
    out[outside_mask & (g != BLACK)] = BLACK

    for region_id in range(1, int(region_count) + 1):
        rr, cc, marker_color = region_to_marker[int(region_id)]
        region_coords = np.argwhere(region_labels == int(region_id))
        piece_coords = np.argwhere((g == int(marker_color)) & outside_mask)
        assert len(piece_coords) > 0
        assert _canonical_signature(region_coords) == _canonical_signature(piece_coords)

        sym_r = int(r0 + r1 - rr)
        sym_c = int(c0 + c1 - cc)
        fill_color = marker_color_at.get((int(sym_r), int(sym_c)))
        if fill_color is None:
            reflected_adjacent = {
                int(region_labels[nr, nc])
                for nr, nc in _neighbors4(int(sym_r), int(sym_c))
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(region_labels[nr, nc]) > 0
            }
            assert len(reflected_adjacent) == 1
            reflected_region = next(iter(reflected_adjacent))
            fill_color = int(region_to_marker[int(reflected_region)][2])

        out[region_labels == int(region_id)] = int(fill_color)

    return out


def _piece_coords_for_region(region_pixels: np.ndarray, rng: random.Random) -> np.ndarray:
    base = np.asarray(region_pixels, dtype=int) - np.asarray(region_pixels, dtype=int).min(axis=0)
    op = rng.randrange(8)
    return _apply_dihedral(base, int(op))


def _frame_position(h: int, w: int, fh: int, fw: int, rng: random.Random) -> tuple[int, int]:
    center_r = int((h - fh) // 2)
    center_c = int((w - fw) // 2)
    r_lo = max(1, int(center_r - 2))
    r_hi = min(int(h - fh - 1), int(center_r + 2))
    c_lo = max(1, int(center_c - 2))
    c_hi = min(int(w - fw - 1), int(center_c + 2))
    fr = int(rng.randint(r_lo, r_hi)) if r_lo <= r_hi else int(center_r)
    fc = int(rng.randint(c_lo, c_hi)) if c_lo <= c_hi else int(center_c)
    return fr, fc


def _place_pieces(
    canvas: np.ndarray,
    pieces: list[tuple[int, np.ndarray]],
    frame_rect: tuple[int, int, int, int],
    rng: random.Random,
) -> dict[int, np.ndarray] | None:
    h, w = canvas.shape
    fr0, fc0, fr1, fc1 = frame_rect
    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[max(0, fr0 - 1) : min(h, fr1 + 2), max(0, fc0 - 1) : min(w, fc1 + 2)] = True
    placements: dict[int, np.ndarray] = {}

    def can_place(abs_coords: np.ndarray) -> bool:
        for r, c in abs_coords.tolist():
            if not (0 <= int(r) < h and 0 <= int(c) < w):
                return False
            if bool(forbidden[int(r), int(c)]):
                return False
            r0 = max(0, int(r) - 1)
            r1 = min(h, int(r) + 2)
            c0 = max(0, int(c) - 1)
            c1 = min(w, int(c) + 2)
            if bool(np.any(canvas[r0:r1, c0:c1] != BLACK)):
                return False
        return True

    for color, coords in sorted(pieces, key=lambda item: -len(item[1])):
        piece_h = int(coords[:, 0].max()) + 1
        piece_w = int(coords[:, 1].max()) + 1
        for _ in range(400):
            top = rng.randint(0, int(h - piece_h))
            left = rng.randint(0, int(w - piece_w))
            abs_coords = coords + np.array([top, left], dtype=int)
            if not can_place(abs_coords):
                continue
            canvas[abs_coords[:, 0], abs_coords[:, 1]] = int(color)
            placements[int(color)] = abs_coords
            for r, c in abs_coords.tolist():
                forbidden[max(0, int(r) - 1) : min(h, int(r) + 2), max(0, int(c) - 1) : min(w, int(c) + 2)] = True
            break
        else:
            return None
    return placements


def _expected_output(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    frame: np.ndarray,
    region_pixels: list[np.ndarray],
    marker_pixels: list[tuple[int, int]],
    colors: list[int],
    placements: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for coords in placements.values():
        out[coords[:, 0], coords[:, 1]] = BLACK

    color_at_marker = {tuple(map(int, marker)): int(color) for marker, color in zip(marker_pixels, colors)}
    fh, fw = frame.shape
    for pixels, marker in zip(region_pixels, marker_pixels):
        mr, mc = marker
        fill = color_at_marker[(int(fh - 1 - mr), int(fw - 1 - mc))]
        abs_pixels = pixels + np.array([frame_top, frame_left], dtype=int)
        out[abs_pixels[:, 0], abs_pixels[:, 1]] = int(fill)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(FRAME_CHOICES)
    pair_count = 2 if (rows, cols) == (6, 6) else rng.choice((2, 3))
    grown = _grow_partition(int(rows), int(cols), int(pair_count), rng)
    if grown is None:
        return None
    regions, seeds = grown
    region_count = len(regions)
    colors = rng.sample(list(COLOR_POOL), int(region_count))
    frame, region_pixels, marker_pixels = _render_frame(int(rows), int(cols), regions, seeds, colors)

    fh, fw = frame.shape
    h_min = max(int(fh + 10), 24)
    w_min = max(int(fw + 10), 24)
    h = rng.randint(int(h_min), 30)
    w = rng.randint(int(w_min), 30)
    grid = np.zeros((h, w), dtype=int)

    frame_top, frame_left = _frame_position(int(h), int(w), int(fh), int(fw), rng)
    grid[int(frame_top) : int(frame_top + fh), int(frame_left) : int(frame_left + fw)] = frame

    pieces = [(int(color), _piece_coords_for_region(pixels, rng)) for color, pixels in zip(colors, region_pixels)]
    placements = _place_pieces(
        grid,
        pieces,
        (int(frame_top), int(frame_left), int(frame_top + fh - 1), int(frame_left + fw - 1)),
        rng,
    )
    if placements is None:
        return None

    expected = _expected_output(grid, int(frame_top), int(frame_left), frame, region_pixels, marker_pixels, colors, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid puzzle")

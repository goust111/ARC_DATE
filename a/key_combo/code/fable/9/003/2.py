from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
ALL_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc2 + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _blob_infos(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    infos: dict[int, tuple[int, int]] = {}
    for comp in _components_from_mask(grid != bg, connectivity=8):
        colors = [int(grid[r, c]) for r, c in comp]
        counts = Counter(colors)
        if len(counts) != 2:
            continue
        singles = [color for color, count in counts.items() if count == 1]
        if len(singles) != 1:
            continue
        seed_color = int(singles[0])
        body_candidates = [color for color, count in counts.items() if count > 1]
        if len(body_candidates) != 1:
            continue
        body_color = int(body_candidates[0])
        seed_coord = next((coord for coord in comp if int(grid[coord]) == seed_color), None)
        assert seed_coord is not None
        sr, sc = seed_coord
        if not all(_in_bounds(grid.shape, sr + dr, sc + dc) and int(grid[sr + dr, sc + dc]) == body_color for dr, dc in DIR4):
            continue
        body_area = int(counts[body_color])
        if body_area < 3:
            continue
        if body_color in infos:
            raise AssertionError("each body color should define at most one valid blob")
        infos[body_color] = (body_area, seed_color)
    return infos


def _segment_endpoints(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        if len(cols) != len(coords) or cols[-1] - cols[0] + 1 != len(coords):
            return None
        return (row, cols[0]), (row, cols[-1])
    if len(cols) == 1:
        col = cols[0]
        if len(rows) != len(coords) or rows[-1] - rows[0] + 1 != len(coords):
            return None
        return (rows[0], col), (rows[-1], col)
    return None


def _line_fits(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int], length: int) -> bool:
    r, c = start
    dr, dc = direction
    end_r = r + (length - 1) * dr
    end_c = c + (length - 1) * dc
    return _in_bounds(shape, r, c) and _in_bounds(shape, end_r, end_c)


def _backward_margin(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = start
    dr, dc = direction
    margin = 0
    rr = r - dr
    cc = c - dc
    while _in_bounds(shape, rr, cc):
        margin += 1
        rr -= dr
        cc -= dc
    return margin


def _marker_infos(grid: np.ndarray, bg: int, lengths_by_color: dict[int, int]) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    infos: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    for color, n in lengths_by_color.items():
        candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        mask = grid == int(color)
        for comp in _components_from_mask(mask, connectivity=4):
            if len(comp) < 2:
                continue
            endpoints = _segment_endpoints(comp)
            if endpoints is None:
                continue
            a, b = endpoints
            dr = int(np.sign(b[0] - a[0]))
            dc = int(np.sign(b[1] - a[1]))
            for head, direction in ((a, (-dr, -dc)), (b, (dr, dc))):
                if direction == (0, 0):
                    continue
                if _line_fits(grid.shape, head, direction, n):
                    candidates.append((_backward_margin(grid.shape, head, direction), head, direction))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        best_margin = candidates[0][0]
        best = [item for item in candidates if item[0] == best_margin]
        if len(best) != 1:
            raise AssertionError("matched marker direction must be unambiguous")
        _, head, direction = best[0]
        infos[int(color)] = (head, direction)
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite each blob's area as a seed-colored ray emitted from the same-colored arrow marker.

    Input structure:
    - The background is the dominant color.
    - Some foreground objects are two-color blobs: a larger monochrome body plus one embedded seed pixel.
    - Some foreground objects are same-color short straight arrow markers whose color matches a blob body color and whose stem orientation determines a direction.

    Rule:
    - Clear the whole grid to the background color.
    - For every body color that has both a valid blob and a valid same-colored arrow marker, count the body's pixels `n` (excluding the seed).
    - Starting at the arrow's head pixel, draw a straight segment of length `n` in the stem-indicated direction, using the seed color from the matched blob.
    - Remove every original foreground object, including unmatched blobs and unmatched markers.

    Key insight:
    - The blob contributes only three attributes: which arrow color it matches, how long the new segment is, and which color that segment should be.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    blobs = _blob_infos(g, bg)
    markers = _marker_infos(g, bg, {color: area for color, (area, _seed) in blobs.items()})

    out = np.full(g.shape, bg, dtype=int)
    for color in sorted(set(blobs) & set(markers)):
        length, seed_color = blobs[color]
        (r, c), (dr, dc) = markers[color]
        for step in range(length):
            rr = r + step * dr
            cc = c + step * dc
            assert _in_bounds(out.shape, rr, cc), "generated segment must stay inside the grid"
            if int(out[rr, cc]) not in (bg, seed_color):
                raise AssertionError("output segments should not collide")
            out[rr, cc] = int(seed_color)
    return out


GRID_MIN = 18
GRID_MAX = 28
MATCHED_COUNT_RANGE = (2, 4)
AREA_RANGE = (8, 13)
STEM_LENGTHS = (1, 2)
GENERATION_ATTEMPTS = 400


def _dilate(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(pad)):
        grown = out.copy()
        for r, c in np.argwhere(out):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIR8:
                nr = rr + dr
                nc = cc + dc
                if _in_bounds(out.shape, nr, nc):
                    grown[nr, nc] = True
        out = grown
    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, occupied: np.ndarray, bg: int) -> bool:
    ph, pw = patch.shape
    mask = patch != bg
    if not mask.any():
        return False
    candidates: list[tuple[int, int]] = []
    blocked = _dilate(occupied, pad=1)
    for r0 in range(grid.shape[0] - ph + 1):
        for c0 in range(grid.shape[1] - pw + 1):
            if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    for rr in range(ph):
        for cc in range(pw):
            if mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = int(patch[rr, cc])
                occupied[r0 + rr, c0 + cc] = True
    return True


def _grow_blob_patch(area: int, body_color: int, seed_color: int, bg: int) -> np.ndarray:
    size = 5
    seed = (2, 2)
    body = {
        (1, 1), (1, 2), (1, 3),
        (2, 1),         (2, 3),
        (3, 1), (3, 2), (3, 3),
    }
    while len(body) < area:
        frontier = []
        for r in range(size):
            for c in range(size):
                if (r, c) == seed or (r, c) in body:
                    continue
                if any((r + dr, c + dc) in body for dr, dc in DIR4):
                    frontier.append((r, c))
        if not frontier:
            raise AssertionError("failed to grow blob")
        body.add(random.choice(frontier))
    cells = list(body) + [seed]
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[seed[0] - r0, seed[1] - c0] = int(seed_color)
    return patch


def _arrow_cells(head: tuple[int, int], direction: tuple[int, int], stem_len: int) -> list[tuple[int, int]]:
    r, c = head
    dr, dc = direction
    return [(r - step * dr, c - step * dc) for step in range(stem_len + 1)]


def _line_cells(head: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = head
    dr, dc = direction
    return [(r + step * dr, c + step * dc) for step in range(length)]


def _can_place_cells(shape: tuple[int, int], cells: list[tuple[int, int]], occupied: np.ndarray, pad: int = 1) -> bool:
    probe = np.zeros(shape, dtype=bool)
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        probe[r, c] = True
    return not bool((_dilate(probe, pad=pad) & occupied).any())


def _head_candidates(shape: tuple[int, int], direction: tuple[int, int], length: int, stem_len: int) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    candidates: list[tuple[int, int]] = []
    if (dr, dc) == (0, 1):
        max_c = min(w - length, stem_len + 2)
        for c in range(stem_len, max_c + 1):
            for r in range(1, h - 1):
                candidates.append((r, c))
    elif (dr, dc) == (0, -1):
        min_c = max(length - 1, w - stem_len - 3)
        for c in range(min_c, w - stem_len):
            for r in range(1, h - 1):
                candidates.append((r, c))
    elif (dr, dc) == (1, 0):
        max_r = min(h - length, stem_len + 2)
        for r in range(stem_len, max_r + 1):
            for c in range(1, w - 1):
                candidates.append((r, c))
    else:
        min_r = max(length - 1, h - stem_len - 3)
        for r in range(min_r, h - stem_len):
            for c in range(1, w - 1):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _choose_directions(count: int) -> list[tuple[int, int]]:
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    picks = [random.choice(dirs) for _ in range(count)]
    if count >= 2 and len(set(picks)) == 1:
        picks[-1] = dirs[(dirs.index(picks[-1]) + random.randint(1, 3)) % 4]
    return picks


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]

    want_blob_only = random.random() < 0.7
    want_marker_only = random.random() < 0.7
    matched_count = random.randint(*MATCHED_COUNT_RANGE)
    total_body_colors = matched_count + int(want_blob_only) + int(want_marker_only)
    if total_body_colors > len(palette):
        return None

    body_colors = random.sample(palette, total_body_colors)
    matched_colors = body_colors[:matched_count]
    extra_index = matched_count
    blob_only_color = body_colors[extra_index] if want_blob_only else None
    extra_index += int(want_blob_only)
    marker_only_color = body_colors[extra_index] if want_marker_only else None

    seed_palette = [color for color in palette if color not in body_colors]
    if not seed_palette:
        return None

    input_grid = np.full((h, w), bg, dtype=int)
    output_grid = np.full((h, w), bg, dtype=int)
    input_occupied = np.zeros((h, w), dtype=bool)
    output_occupied = np.zeros((h, w), dtype=bool)

    pair_specs = []
    for body_color, direction in zip(matched_colors, _choose_directions(matched_count)):
        area = random.randint(*AREA_RANGE)
        seed_color = random.choice(seed_palette)
        stem_len = random.choice(STEM_LENGTHS)
        placed = False
        for head in _head_candidates((h, w), direction, area, stem_len):
            line = _line_cells(head, direction, area)
            arrow = _arrow_cells(head, direction, stem_len)
            tail = arrow[-1]
            if _line_fits((h, w), tail, (-direction[0], -direction[1]), area):
                continue
            if not _can_place_cells((h, w), line, output_occupied, pad=1):
                continue
            if not _can_place_cells((h, w), arrow, input_occupied, pad=1):
                continue
            for r, c in line:
                output_grid[r, c] = int(seed_color)
                output_occupied[r, c] = True
            for r, c in arrow:
                input_grid[r, c] = int(body_color)
                input_occupied[r, c] = True
            pair_specs.append((area, body_color, seed_color))
            placed = True
            break
        if not placed:
            return None

    for area, body_color, seed_color in pair_specs:
        patch = _grow_blob_patch(area, body_color, seed_color, bg)
        if not _place_patch(input_grid, patch, input_occupied, bg):
            return None

    if blob_only_color is not None:
        area = random.randint(*AREA_RANGE)
        seed_color = random.choice(seed_palette)
        patch = _grow_blob_patch(area, blob_only_color, seed_color, bg)
        if not _place_patch(input_grid, patch, input_occupied, bg):
            return None

    if marker_only_color is not None:
        direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        stem_len = random.choice(STEM_LENGTHS)
        for head in _head_candidates((h, w), direction, random.randint(*AREA_RANGE), stem_len):
            arrow = _arrow_cells(head, direction, stem_len)
            if _can_place_cells((h, w), arrow, input_occupied, pad=1):
                for r, c in arrow:
                    input_grid[r, c] = int(marker_only_color)
                    input_occupied[r, c] = True
                break
        else:
            return None

    try:
        actual = main(input_grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/9/003 task 2")

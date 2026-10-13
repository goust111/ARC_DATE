from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
SUBJECT_COLOR = 1
MARKER_COLOR = 2
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (15, 22)
MARKER_COUNT_RANGE = (2, 4)
NOISE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 500

RING_SHAPE_PROBABILITY = 0.45
INNER_NOISE_PROBABILITY = 0.75
NOISY_RING_MARKER_PROBABILITY = 0.55

_RNG = random.Random()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _render_tile(shape: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(shape)
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), BG_COLOR, dtype=int)
    for r, c in shape:
        tile[r - r0, c - c0] = SUBJECT_COLOR
    return tile


def _subject_from_grid(grid: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    comps = _components(grid != BG_COLOR)
    assert comps, "expected at least one non-background component"
    sizes = [len(comp) for comp in comps]
    max_size = max(sizes)
    assert sizes.count(max_size) == 1, "largest non-background component must be unique"
    subject_cells = set(comps[sizes.index(max_size)])
    sr, sc = next(iter(subject_cells))
    return subject_cells, int(grid[sr, sc])


def _count_singleton_markers(grid: np.ndarray) -> int:
    return sum(len(comp) == 1 for comp in _components(grid == MARKER_COLOR))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove same-color singleton noise, crop the main blue figure, and repeat that crop once per red marker.

    Input structure:
    - A black background grid contains one irregular multi-pixel blue figure that is 4-connected.
    - Additional isolated blue singletons lie away from that figure and act only as noise for cropping.
    - Several isolated red singleton pixels lie outside the figure's bounding box and act only as repetition markers.

    Rule:
    - Find the unique largest 4-connected non-background component; this is the blue subject.
    - Ignore every other blue pixel, then crop the subject's tight axis-aligned bounding box while keeping black holes or gaps inside that box.
    - Count the red singleton markers in the input.
    - Concatenate the cropped subject tile horizontally once per red marker, with no gaps or extra padding.

    Key insight:
    - Blue noise pixels and red marker pixels are both isolated singletons, so color alone separates their roles: same-color singletons must be deleted before cropping, while red singletons must be counted to choose the repetition count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    subject_cells, subject_color = _subject_from_grid(g)
    tile = np.full_like(g, BG_COLOR)
    for r, c in subject_cells:
        tile[r, c] = subject_color

    r0, c0, r1, c1 = _bbox(subject_cells)
    cropped = tile[r0 : r1 + 1, c0 : c1 + 1]

    marker_count = _count_singleton_markers(g)
    assert marker_count >= 1, "expected at least one red marker"
    return np.tile(cropped, (1, marker_count))


def _shift_shape(shape: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape}


def _cell_is_available(cell: tuple[int, int], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in occupied:
        return False
    return all((r + dr, c + dc) not in occupied for dr, dc in DIR4)


def _candidate_cells(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    *,
    must_be_inside: tuple[int, int, int, int] | None = None,
    must_be_outside: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if must_be_inside is not None and not _inside_bbox(cell, must_be_inside):
                continue
            if must_be_outside is not None and _inside_bbox(cell, must_be_outside):
                continue
            if _cell_is_available(cell, occupied, h, w):
                out.append(cell)
    return out


def _deep_empty_candidates(shape: set[tuple[int, int]], bbox_h: int, bbox_w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(bbox_h):
        for c in range(bbox_w):
            cell = (r, c)
            if cell in shape:
                continue
            if all((r + dr, c + dc) not in shape for dr, dc in DIR4):
                candidates.append(cell)
    return candidates


def _removable(cell: tuple[int, int], cells: set[tuple[int, int]], bbox_h: int, bbox_w: int) -> bool:
    r, c = cell
    if r == 0 and sum(rr == 0 for rr, _ in cells) == 1:
        return False
    if r == bbox_h - 1 and sum(rr == bbox_h - 1 for rr, _ in cells) == 1:
        return False
    if c == 0 and sum(cc == 0 for _, cc in cells) == 1:
        return False
    if c == bbox_w - 1 and sum(cc == bbox_w - 1 for _, cc in cells) == 1:
        return False
    trial = set(cells)
    trial.remove(cell)
    return _connected(trial)


def _generate_carved_shape(rng: random.Random) -> set[tuple[int, int]] | None:
    for _ in range(200):
        bbox_h = rng.randint(3, 6)
        bbox_w = rng.randint(3, 7)
        target_min = max(6, bbox_h + bbox_w - 1)
        target_max = min(15, bbox_h * bbox_w - 1)
        if target_min > target_max:
            continue

        cells = {(r, c) for r in range(bbox_h) for c in range(bbox_w)}
        target_size = rng.randint(target_min, target_max)
        while len(cells) > target_size:
            removable = [cell for cell in cells if _removable(cell, cells, bbox_h, bbox_w)]
            if not removable:
                break
            weights = [3 if 0 < r < bbox_h - 1 and 0 < c < bbox_w - 1 else 1 for r, c in removable]
            chosen = rng.choices(removable, weights=weights, k=1)[0]
            cells.remove(chosen)

        if len(cells) != target_size:
            continue
        if not _connected(cells):
            continue
        if _bbox(cells) != (0, 0, bbox_h - 1, bbox_w - 1):
            continue
        return cells
    return None


def _border_cells(bbox_h: int, bbox_w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(bbox_h)
        for c in range(bbox_w)
        if r in {0, bbox_h - 1} or c in {0, bbox_w - 1}
    }


def _generate_ring_shape(rng: random.Random) -> set[tuple[int, int]] | None:
    for _ in range(200):
        bbox_h, bbox_w = rng.choice(((5, 5), (5, 6), (6, 5)))
        cells = _border_cells(bbox_h, bbox_w)
        target_size = rng.randint(max(12, bbox_h + bbox_w - 1), 15)
        while len(cells) > target_size:
            removable = [cell for cell in cells if _removable(cell, cells, bbox_h, bbox_w)]
            if not removable:
                break
            cells.remove(rng.choice(removable))

        if len(cells) != target_size:
            continue
        if not _deep_empty_candidates(cells, bbox_h, bbox_w):
            continue
        return cells
    return None


def _pick_cells(
    rng: random.Random,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    count: int,
    *,
    must_be_inside: tuple[int, int, int, int] | None = None,
    must_be_outside: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]] | None:
    picked: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = _candidate_cells(
            h,
            w,
            occupied | set(picked),
            must_be_inside=must_be_inside,
            must_be_outside=must_be_outside,
        )
        if not candidates:
            return None
        picked.append(rng.choice(candidates))
    return picked


def _wrong_crop_all_blue(grid: np.ndarray, marker_count: int) -> np.ndarray:
    blue = np.argwhere(grid == SUBJECT_COLOR)
    r0, c0 = blue.min(axis=0)
    r1, c1 = blue.max(axis=0)
    tile = np.where(grid[r0 : r1 + 1, c0 : c1 + 1] == SUBJECT_COLOR, SUBJECT_COLOR, BG_COLOR)
    return np.tile(tile, (1, marker_count))


def _expected_output(shape: set[tuple[int, int]], marker_count: int) -> np.ndarray:
    return np.tile(_render_tile(shape), (1, marker_count))


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    marker_count = rng.randint(*MARKER_COUNT_RANGE)
    noise_count = rng.randint(*NOISE_COUNT_RANGE)
    use_ring_shape = rng.random() < RING_SHAPE_PROBABILITY
    want_inner_noise = use_ring_shape and rng.random() < INNER_NOISE_PROBABILITY

    shape = _generate_ring_shape(rng) if use_ring_shape else _generate_carved_shape(rng)
    if shape is None:
        return None

    tile = _render_tile(shape)
    bbox_h, bbox_w = tile.shape
    if bbox_w * marker_count > 30:
        return None

    grid_h = rng.randint(max(GRID_HEIGHT_RANGE[0], bbox_h + 5), GRID_HEIGHT_RANGE[1])
    grid_w = rng.randint(max(GRID_WIDTH_RANGE[0], bbox_w + 7), GRID_WIDTH_RANGE[1])

    top = rng.randint(1, grid_h - bbox_h - 2)
    left = rng.randint(1, grid_w - bbox_w - 2)
    subject = _shift_shape(shape, top, left)
    clean_bbox = _bbox(subject)

    occupied = set(subject)
    noise_cells: list[tuple[int, int]] = []
    if want_inner_noise:
        inner_candidates = [
            (top + r, left + c)
            for r, c in _deep_empty_candidates(shape, bbox_h, bbox_w)
            if _cell_is_available((top + r, left + c), occupied, grid_h, grid_w)
        ]
        if not inner_candidates:
            return None
        inner_noise = rng.choice(inner_candidates)
        noise_cells.append(inner_noise)
        occupied.add(inner_noise)

    extra_noise = _pick_cells(
        rng,
        grid_h,
        grid_w,
        occupied,
        noise_count - len(noise_cells),
        must_be_outside=clean_bbox,
    )
    if extra_noise is None:
        return None
    noise_cells.extend(extra_noise)
    occupied.update(extra_noise)

    noisy_bbox = _bbox(subject | set(noise_cells))
    if noisy_bbox == clean_bbox:
        return None

    marker_cells: list[tuple[int, int]] = []
    if rng.random() < NOISY_RING_MARKER_PROBABILITY:
        noisy_ring = [
            cell
            for cell in _candidate_cells(grid_h, grid_w, occupied, must_be_inside=noisy_bbox)
            if not _inside_bbox(cell, clean_bbox)
        ]
        if noisy_ring:
            marker = rng.choice(noisy_ring)
            marker_cells.append(marker)
            occupied.add(marker)

    more_markers = _pick_cells(
        rng,
        grid_h,
        grid_w,
        occupied,
        marker_count - len(marker_cells),
        must_be_outside=clean_bbox,
    )
    if more_markers is None:
        return None
    marker_cells.extend(more_markers)

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    for r, c in subject | set(noise_cells):
        grid[r, c] = SUBJECT_COLOR
    for r, c in marker_cells:
        grid[r, c] = MARKER_COLOR

    expected = _expected_output(shape, marker_count)

    subject_comps = _components(grid != BG_COLOR)
    comp_sizes = sorted((len(comp) for comp in subject_comps), reverse=True)
    if not comp_sizes or comp_sizes[0] != len(subject):
        return None
    if len(comp_sizes) > 1 and comp_sizes[1] >= comp_sizes[0]:
        return None
    if any(_inside_bbox(cell, clean_bbox) for cell in marker_cells):
        return None
    if not any(not _inside_bbox(cell, clean_bbox) for cell in noise_cells):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(_wrong_crop_all_blue(grid, marker_count), expected):
        return None
    if np.array_equal(np.tile(tile, (1, marker_count + len(noise_cells))), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case(_RNG)
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid input")

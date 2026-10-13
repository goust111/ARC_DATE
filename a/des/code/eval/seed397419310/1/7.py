from __future__ import annotations

import random

import numpy as np


BG = 0
ZERO_MARK = 9
OUTPUT_MARK = 8
TRACK_COLORS = (1, 2, 3, 4, 5, 6, 7)

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
PLACE_ATTEMPTS = 240


def _components(grid: np.ndarray, active: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            if not bool(active[r, c]) or bool(seen[r, c]):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or not bool(active[nr, nc]):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, np.array(cells, dtype=int)))
    return comps


def _parse_tracks(grid: np.ndarray) -> tuple[dict[int, tuple[int, int, int]], np.ndarray]:
    h, w = grid.shape
    track_mask = np.zeros((h, w), dtype=bool)
    color_to_track: dict[int, tuple[int, int, int]] = {}

    rows = [int(r) for r in np.flatnonzero(np.any(grid == ZERO_MARK, axis=1))]
    assert rows, "expected at least one ruler track"

    for r in rows:
        zero_cols = np.flatnonzero(grid[r] == ZERO_MARK)
        assert zero_cols.size == 1, "each ruler row must have exactly one zero marker"
        zc = int(zero_cols[0])
        assert not np.any(grid[r, :zc] != BG), "ruler rows must be empty to the left of the zero marker"
        assert zc + 1 < w, "zero marker must have track cells on its right"

        color = int(grid[r, zc + 1])
        assert color not in (BG, ZERO_MARK, OUTPUT_MARK), "track color must be distinct from markers and background"

        end = zc + 1
        while end < w and int(grid[r, end]) == color:
            end += 1
        assert not np.any(grid[r, end:] != BG), "ruler track must be one contiguous run"

        length = int(end - (zc + 1))
        assert length >= 1, "track must contain at least one measurable cell"
        assert color not in color_to_track, "each track color must be unique"

        color_to_track[color] = (r, zc, length)
        track_mask[r, zc] = True
        track_mask[r, zc + 1 : zc + 1 + length] = True

    return color_to_track, track_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map each colored object to the same-colored ruler track, convert its area into a
    tick position, mark that position on the track, and erase the original object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    color_to_track, track_mask = _parse_tracks(g)
    object_mask = (g != BG) & (~track_mask)
    components = _components(g, object_mask)
    assert components, "expected at least one colored object above the tracks"

    out = g.copy()
    out[object_mask] = BG

    for color, cells in components:
        assert color in color_to_track, "every object color must match a ruler track"
        row, zero_col, length = color_to_track[color]
        area = int(cells.shape[0])
        assert 1 <= area <= length, "object area must fit on its matching track"
        out[row, zero_col + area] = OUTPUT_MARK

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _transform_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    pts = np.array(list(cells), dtype=int)
    variants: set[tuple[tuple[int, int], ...]] = set()

    for flip in (1, -1):
        base = pts.copy()
        base[:, 1] *= flip
        current = base
        for _ in range(4):
            current = np.column_stack((-current[:, 1], current[:, 0]))
            variant = _normalize_cells({(int(r), int(c)) for r, c in current})
            variants.add(variant)

    chosen = random.choice(sorted(variants))
    return np.array(chosen, dtype=int)


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}

        while len(cells) < area and frontier:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add((int(pick[0]), int(pick[1])))
            pr, pc = pick
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (int(pr + dr), int(pc + dc))
                if nxt not in cells:
                    frontier.add(nxt)

        if len(cells) == area:
            return _transform_shape(cells)
    return None


def _choose_object_plan(lengths: list[int]) -> tuple[str, bool, list[list[int]]]:
    mode = random.choice(("all_unique", "multi_unique", "multi_duplicate"))
    include_unused = len(lengths) >= 3 and bool(random.getrandbits(1))

    sizes_by_track: list[list[int]] = [[] for _ in lengths]
    used_tracks = list(range(len(lengths)))
    if include_unused:
        unused = random.choice(used_tracks)
        used_tracks.remove(unused)

    focus_track = random.choice(used_tracks)

    def rand_size(track_idx: int) -> int:
        return random.randint(2, min(10, int(lengths[track_idx])))

    for idx in used_tracks:
        sizes_by_track[idx] = [rand_size(idx)]

    if mode == "multi_unique":
        max_size = min(10, int(lengths[focus_track]))
        count = random.randint(2, min(3, max_size - 1))
        sizes_by_track[focus_track] = random.sample(list(range(2, max_size + 1)), count)
    elif mode == "multi_duplicate":
        max_size = min(10, int(lengths[focus_track]))
        dup_size = rand_size(focus_track)
        sizes = [dup_size, dup_size]
        if random.random() < 0.5:
            others = [x for x in range(2, max_size + 1) if x != dup_size]
            if others:
                sizes.append(random.choice(others))
        sizes_by_track[focus_track] = sizes

    return mode, include_unused, sizes_by_track


def _can_place(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if bool(occupied[rr, cc]):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(occupied[nr, nc]):
                    return False
    return True


def _place_objects(
    grid: np.ndarray,
    object_specs: list[tuple[int, int]],
    object_zone_h: int,
) -> bool:
    occupied = np.zeros(grid.shape, dtype=bool)

    shuffled = object_specs[:]
    random.shuffle(shuffled)
    shuffled.sort(key=lambda item: (-int(item[1]), int(item[0])))

    for color, area in shuffled:
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            shape = _random_polyomino(int(area))
            if shape is None:
                continue
            shape_h = int(shape[:, 0].max()) + 1
            shape_w = int(shape[:, 1].max()) + 1
            if shape_h > object_zone_h or shape_w > grid.shape[1]:
                continue

            r0 = random.randint(0, object_zone_h - shape_h)
            c0 = random.randint(0, grid.shape[1] - shape_w)
            coords = shape + np.array([r0, c0], dtype=int)
            if not _can_place(coords, occupied):
                continue

            grid[coords[:, 0], coords[:, 1]] = int(color)
            occupied[coords[:, 0], coords[:, 1]] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    track_count = random.randint(2, 5)
    track_colors = random.sample(TRACK_COLORS, track_count)
    lengths = [random.randint(5, 15) for _ in range(track_count)]
    mode, include_unused, sizes_by_track = _choose_object_plan(lengths)

    total_objects = sum(len(sizes) for sizes in sizes_by_track)
    band_h = 2 * track_count - 1
    object_zone_h = 6 + total_objects + random.randint(0, 3)
    h = band_h + 1 + object_zone_h

    min_w = max(18, max(lengths) + 3, 12 + total_objects)
    max_w = max(min_w, min(32, min_w + 8))
    w = random.randint(min_w, max_w)

    grid = np.full((h, w), BG, dtype=int)
    band_top = h - band_h

    for idx, (color, length) in enumerate(zip(track_colors, lengths)):
        row = band_top + 2 * idx
        zero_col = random.randint(0, w - length - 1)
        grid[row, zero_col] = ZERO_MARK
        grid[row, zero_col + 1 : zero_col + 1 + length] = int(color)

    object_specs: list[tuple[int, int]] = []
    for color, sizes in zip(track_colors, sizes_by_track):
        for size in sizes:
            object_specs.append((int(color), int(size)))

    if not object_specs:
        return None
    if not _place_objects(grid, object_specs, object_zone_h):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert np.count_nonzero(out == OUTPUT_MARK) >= 1

    if mode == "multi_duplicate":
        duplicate_found = any(len(sizes) != len(set(sizes)) for sizes in sizes_by_track)
        assert duplicate_found
    if mode == "multi_unique":
        unique_multi_found = any(len(sizes) >= 2 and len(sizes) == len(set(sizes)) for sizes in sizes_by_track)
        assert unique_multi_found
    if include_unused:
        assert any(len(sizes) == 0 for sizes in sizes_by_track)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-track ARC input")

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
AZURE = 8
GEN_VALVE_COLORS = (2, 4, 3, 6)  # red, yellow, green, magenta

ORTHO_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_STEPS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250
BLOB_PLACEMENT_ATTEMPTS = 250
GRID_MIN = 12
GRID_MAX = 24
FUEL_MIN = 3
FUEL_MAX = 7
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
PAIR_COUNT_WEIGHTS = (2, 3, 3, 2)


@dataclass
class PairSpec:
    color: int
    orientation: str
    valve_r: int
    valve_c: int
    end0_r: int
    end0_c: int
    end1_r: int
    end1_c: int
    fuel_size: int

    def line_cells(self) -> list[tuple[int, int]]:
        if self.orientation == "H":
            return [(self.valve_r, c) for c in range(self.end0_c, self.end1_c + 1)]
        return [(r, self.valve_c) for r in range(self.end0_r, self.end1_r + 1)]

    def ray_cells(self, height: int, width: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        if self.orientation == "H":
            for step in range(1, self.fuel_size + 1):
                up_r = self.valve_r - step
                down_r = self.valve_r + step
                if up_r >= 0:
                    cells.append((up_r, self.valve_c))
                if down_r < height:
                    cells.append((down_r, self.valve_c))
        else:
            for step in range(1, self.fuel_size + 1):
                left_c = self.valve_c - step
                right_c = self.valve_c + step
                if left_c >= 0:
                    cells.append((self.valve_r, left_c))
                if right_c < width:
                    cells.append((self.valve_r, right_c))
        return cells

    def footprint_cells(self, height: int, width: int) -> list[tuple[int, int]]:
        return self.line_cells() + self.ray_cells(height, width)


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = ORTHO_STEPS if connectivity == 4 else EIGHT_STEPS
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _transform_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    if random.random() < 0.5:
        arr[:, 1] *= -1
    for _ in range(random.randrange(4)):
        arr = np.column_stack((-arr[:, 1], arr[:, 0]))
    return _normalize_coords(arr)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _cells_to_mask(cells: list[tuple[int, int]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if cells:
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        mask[rows, cols] = True
    return mask


def _valid_segment(seq: np.ndarray, valve_idx: int, valve_color: int, bg: int) -> bool:
    return (
        int(seq[0]) == AZURE
        and int(seq[-1]) == AZURE
        and int(seq[valve_idx]) == valve_color
        and int(np.count_nonzero(seq == AZURE)) == 2
        and int(np.count_nonzero(seq == valve_color)) == 1
        and int(np.count_nonzero(seq != bg)) == 3
    )


def _find_pair_for_valve(grid: np.ndarray, valve_r: int, valve_c: int, valve_color: int, bg: int) -> PairSpec:
    h, w = grid.shape
    candidates: list[PairSpec] = []

    azure_cols = np.flatnonzero(grid[valve_r] == AZURE)
    left_cols = [int(c) for c in azure_cols if int(c) < valve_c]
    right_cols = [int(c) for c in azure_cols if int(c) > valve_c]
    for c0 in left_cols:
        for c1 in right_cols:
            segment = grid[valve_r, c0 : c1 + 1]
            if _valid_segment(segment, valve_c - c0, valve_color, bg):
                candidates.append(
                    PairSpec(
                        color=int(valve_color),
                        orientation="H",
                        valve_r=int(valve_r),
                        valve_c=int(valve_c),
                        end0_r=int(valve_r),
                        end0_c=int(c0),
                        end1_r=int(valve_r),
                        end1_c=int(c1),
                        fuel_size=0,
                    )
                )

    azure_rows = np.flatnonzero(grid[:, valve_c] == AZURE)
    top_rows = [int(r) for r in azure_rows if int(r) < valve_r]
    bottom_rows = [int(r) for r in azure_rows if int(r) > valve_r]
    for r0 in top_rows:
        for r1 in bottom_rows:
            segment = grid[r0 : r1 + 1, valve_c]
            if _valid_segment(segment, valve_r - r0, valve_color, bg):
                candidates.append(
                    PairSpec(
                        color=int(valve_color),
                        orientation="V",
                        valve_r=int(valve_r),
                        valve_c=int(valve_c),
                        end0_r=int(r0),
                        end0_c=int(valve_c),
                        end1_r=int(r1),
                        end1_c=int(valve_c),
                        fuel_size=0,
                    )
                )

    assert len(candidates) == 1, "each valve must belong to exactly one azure pair"
    return candidates[0]


def _parse_scene(grid: np.ndarray) -> list[tuple[PairSpec, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) not in {bg, AZURE}]
    assert colors, "expected at least one valve color"

    parsed: list[tuple[PairSpec, np.ndarray]] = []
    for color in colors:
        comps = _components(g == color, connectivity=4)
        singles = [comp for comp in comps if len(comp) == 1]
        blobs = [comp for comp in comps if 3 <= len(comp) <= 7]
        assert len(comps) == 2, "each valve color must have exactly two components"
        assert len(singles) == 1 and len(blobs) == 1, "expected one singleton valve and one compact blob"

        valve_r, valve_c = map(int, singles[0][0])
        pair = _find_pair_for_valve(g, valve_r, valve_c, color, bg)
        pair.fuel_size = int(len(blobs[0]))
        parsed.append((pair, blobs[0]))

    parsed.sort(key=lambda item: item[0].color)
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _bg_color(g)

    parsed = _parse_scene(g)
    out = g.copy()

    for _pair, blob in parsed:
        out[blob[:, 0], blob[:, 1]] = bg

    for pair, _blob in parsed:
        for r, c in pair.line_cells():
            out[r, c] = AZURE
        out[pair.valve_r, pair.valve_c] = pair.color
        for r, c in pair.ray_cells(h, w):
            out[r, c] = pair.color

    return out


def _random_compact_polyomino(size: int) -> np.ndarray:
    patterns = {
        3: np.asarray([(0, 0), (0, 1), (1, 0)], dtype=int),
        4: np.asarray([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=int),
        5: np.asarray([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], dtype=int),
        6: np.asarray([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], dtype=int),
        7: np.asarray([(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)], dtype=int),
    }
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ORTHO_STEPS:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in cells:
                        continue
                    test = cells | {(nr, nc)}
                    rows = [rr for rr, _cc in test]
                    cols = [cc for _rr, cc in test]
                    if max(rows) - min(rows) + 1 <= 4 and max(cols) - min(cols) + 1 <= 4:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) == size:
            arr = _normalize_coords(np.asarray(list(cells), dtype=int))
            return _transform_coords(arr)
    return _transform_coords(patterns[int(size)])


def _choose_pair_count() -> int:
    return int(random.choices(PAIR_COUNT_OPTIONS, weights=PAIR_COUNT_WEIGHTS, k=1)[0])


def _ray_axis_options(limit: int, fuel_size: int, truncated: bool) -> list[int]:
    full = [i for i in range(limit) if i >= fuel_size and limit - 1 - i >= fuel_size]
    if truncated:
        return [i for i in range(limit) if i not in set(full)]
    return full


def _sample_pair(
    height: int,
    width: int,
    color: int,
    fuel_size: int,
    orientation: str,
    truncated: bool,
    buffer_mask: np.ndarray,
) -> PairSpec | None:
    line_cap = max(3, min(8, max(height, width) - 2))
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        if orientation == "H":
            row_options = _ray_axis_options(height, fuel_size, truncated)
            if not row_options or width < 3:
                return None
            valve_r = int(random.choice(row_options))
            valve_c = int(random.randint(1, width - 2))
            left_cap = min(line_cap, valve_c)
            right_cap = min(line_cap, width - 1 - valve_c)
            if left_cap < 1 or right_cap < 1:
                continue
            left_span = int(random.randint(1, left_cap))
            right_span = int(random.randint(1, right_cap))
            spec = PairSpec(
                color=int(color),
                orientation="H",
                valve_r=valve_r,
                valve_c=valve_c,
                end0_r=valve_r,
                end0_c=valve_c - left_span,
                end1_r=valve_r,
                end1_c=valve_c + right_span,
                fuel_size=int(fuel_size),
            )
        else:
            col_options = _ray_axis_options(width, fuel_size, truncated)
            if not col_options or height < 3:
                return None
            valve_c = int(random.choice(col_options))
            valve_r = int(random.randint(1, height - 2))
            top_cap = min(line_cap, valve_r)
            bottom_cap = min(line_cap, height - 1 - valve_r)
            if top_cap < 1 or bottom_cap < 1:
                continue
            top_span = int(random.randint(1, top_cap))
            bottom_span = int(random.randint(1, bottom_cap))
            spec = PairSpec(
                color=int(color),
                orientation="V",
                valve_r=valve_r,
                valve_c=valve_c,
                end0_r=valve_r - top_span,
                end0_c=valve_c,
                end1_r=valve_r + bottom_span,
                end1_c=valve_c,
                fuel_size=int(fuel_size),
            )

        footprint = spec.footprint_cells(height, width)
        if any(buffer_mask[r, c] for r, c in footprint):
            continue
        return spec
    return None


def _place_blobs(
    grid: np.ndarray,
    specs: list[PairSpec],
    occupied_mask: np.ndarray,
    line_mask: np.ndarray,
) -> bool:
    h, w = grid.shape
    forbidden = _dilate_mask(occupied_mask, 1) | _dilate_mask(line_mask, 1)
    order = specs[:]
    random.shuffle(order)
    for spec in order:
        placed = False
        for _ in range(BLOB_PLACEMENT_ATTEMPTS):
            shape = _random_compact_polyomino(spec.fuel_size)
            r0, c0, r1, c1 = _bbox(shape)
            sh = r1 - r0 + 1
            sw = c1 - c0 + 1
            anchor_r = int(random.randint(0, h - sh))
            anchor_c = int(random.randint(0, w - sw))
            coords = shape + np.asarray([anchor_r, anchor_c], dtype=int)
            if any(forbidden[int(r), int(c)] for r, c in coords):
                continue
            grid[coords[:, 0], coords[:, 1]] = spec.color
            blob_mask = np.zeros((h, w), dtype=bool)
            blob_mask[coords[:, 0], coords[:, 1]] = True
            forbidden |= _dilate_mask(blob_mask, 1)
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    if bg != BLACK:
        return False
    if np.any((g != bg) & (g != AZURE) & (g < 0)):
        return False

    try:
        parsed = _parse_scene(g)
    except AssertionError:
        return False

    if len(parsed) < 1 or len(parsed) > len(GEN_VALVE_COLORS):
        return False

    if int(np.count_nonzero(g == AZURE)) != 2 * len(parsed):
        return False

    line_mask = np.zeros_like(g, dtype=bool)
    for pair, _blob in parsed:
        for r, c in pair.line_cells():
            line_mask[r, c] = True
    line_guard = _dilate_mask(line_mask, 1)
    for _pair, blob in parsed:
        for r, c in blob:
            if line_guard[int(r), int(c)]:
                return False

    try:
        out = main(g)
    except AssertionError:
        return False

    if out.shape != g.shape or np.array_equal(out, g):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    pair_count = _choose_pair_count()
    min_dim = max(GRID_MIN, 10 + 2 * pair_count)
    height = int(random.randint(min_dim, GRID_MAX))
    width = int(random.randint(min_dim, GRID_MAX))
    colors = random.sample(list(GEN_VALVE_COLORS), pair_count)

    specs: list[PairSpec] = []
    occupied = np.zeros((height, width), dtype=bool)
    buffer_mask = np.zeros((height, width), dtype=bool)
    line_mask = np.zeros((height, width), dtype=bool)

    for color in colors:
        fuel_size = int(random.randint(FUEL_MIN, FUEL_MAX))
        orientation = random.choice(("H", "V"))
        truncated = bool(random.getrandbits(1))
        spec = _sample_pair(height, width, color, fuel_size, orientation, truncated, buffer_mask)
        if spec is None:
            return None
        specs.append(spec)

        footprint = spec.footprint_cells(height, width)
        footprint_mask = _cells_to_mask(footprint, height, width)
        occupied |= footprint_mask
        buffer_mask |= _dilate_mask(footprint_mask, 1)
        line_mask |= _cells_to_mask(spec.line_cells(), height, width)

    grid = np.full((height, width), BLACK, dtype=int)
    for spec in specs:
        grid[spec.end0_r, spec.end0_c] = AZURE
        grid[spec.end1_r, spec.end1_c] = AZURE
        grid[spec.valve_r, spec.valve_c] = spec.color

    if not _place_blobs(grid, specs, occupied, line_mask):
        return None
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")

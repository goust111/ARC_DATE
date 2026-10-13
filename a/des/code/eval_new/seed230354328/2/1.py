from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((int(r0), int(c)))
        pts.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        pts.add((int(r), int(c0)))
        pts.add((int(r), int(c1)))
    return pts


def _parse_legend(grid: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    for r in range(h):
        if not np.any(grid[r] != BG):
            continue
        entries: list[tuple[int, int]] = []
        covered = np.zeros(w, dtype=bool)
        c = 0
        valid = True
        while c < w:
            while c < w and int(grid[r, c]) == BG:
                c += 1
            if c >= w:
                break
            if int(grid[r, c]) == GREY:
                valid = False
                break
            if c + 1 >= w or int(grid[r, c + 1]) in {BG, GREY}:
                valid = False
                break
            entries.append((int(grid[r, c]), int(grid[r, c + 1])))
            covered[c] = True
            covered[c + 1] = True
            c += 2
            sep_count = 0
            while c < w and int(grid[r, c]) == GREY:
                covered[c] = True
                sep_count += 1
                c += 1
            if c < w and int(grid[r, c]) != BG and sep_count == 0:
                valid = False
                break
        if not valid or not entries:
            continue
        if not np.array_equal(covered, grid[r] != BG):
            continue
        match_colors = [match for match, _fill in entries]
        if len(set(match_colors)) != len(match_colors):
            continue
        return int(r), entries
    raise AssertionError("failed to locate a valid top legend row")


def _find_frames(grid: np.ndarray, legend_row: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        if max(rows) == legend_row and min(rows) == legend_row:
            continue
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(comp) == _border_cells(r0, c0, r1, c1):
            frames.append((int(r0), int(c0), int(r1), int(c1)))
    frames.sort()
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _ordered_seed_colors(inner: np.ndarray, legend_order: list[int]) -> list[int]:
    present = {
        int(v)
        for v in np.unique(inner)
        if int(v) not in {BG, GREY}
    }
    ordered = [int(color) for color in legend_order if int(color) in present]
    assert ordered, "every frame must contain at least one colored fragment"
    assert set(ordered) == present, "every fragment color must appear in the legend"
    return ordered


def _label_regions(inner: np.ndarray, seed_colors: list[int]) -> np.ndarray:
    h, w = inner.shape
    owner = np.full((h, w), -1, dtype=int)
    dist = np.full((h, w), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()

    for label, color in enumerate(seed_colors):
        coords = np.argwhere(inner == int(color))
        assert coords.size > 0
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            owner[rr, cc] = int(label)
            dist[rr, cc] = 0
            queue.append((rr, cc))

    while queue:
        r, c = queue.popleft()
        next_dist = int(dist[r, c]) + 1
        label = int(owner[r, c])
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and int(inner[nr, nc]) == BG and dist[nr, nc] == -1:
                owner[nr, nc] = label
                dist[nr, nc] = next_dist
                queue.append((int(nr), int(nc)))

    assert not np.any((inner == BG) & (owner < 0)), "all black interior cells must be claimed"
    return owner


def _solve_frame(inner: np.ndarray, color_to_fill: dict[int, int], legend_order: list[int]) -> np.ndarray:
    seed_colors = _ordered_seed_colors(inner, legend_order)
    owner = _label_regions(inner, seed_colors)
    out = np.full(inner.shape, BG, dtype=int)
    for label, color in enumerate(seed_colors):
        out[owner == int(label)] = int(color_to_fill[int(color)])
    assert bool(np.all(out != BG)), "frame interiors should be completely filled"
    return out


def _single_source_distances(inner: np.ndarray, color: int) -> np.ndarray:
    h, w = inner.shape
    dist = np.full((h, w), -1, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(inner == int(color)):
        rr = int(r)
        cc = int(c)
        dist[rr, cc] = 0
        queue.append((rr, cc))
    while queue:
        r, c = queue.popleft()
        next_dist = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == -1 and int(inner[nr, nc]) == BG:
                dist[nr, nc] = next_dist
                queue.append((int(nr), int(nc)))
    return dist


def _frame_has_tie(inner: np.ndarray, legend_order: list[int]) -> bool:
    seed_colors = _ordered_seed_colors(inner, legend_order)
    if len(seed_colors) < 2:
        return False
    dists = [_single_source_distances(inner, color) for color in seed_colors]
    for r, c in np.argwhere(inner == BG):
        rr = int(r)
        cc = int(c)
        values = [int(dist[rr, cc]) for dist in dists if int(dist[rr, cc]) >= 0]
        if not values:
            continue
        best = min(values)
        if sum(int(value == best) for value in values) >= 2:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed region by Voronoi-partitioning its black interior from the colored fragments and then translating fragment colors through the legend.

    Input structure:
    - bg = black background.
    - The top legend row contains repeated two-pixel entries: match color on the left, fill color on the right, with grey separator pixels between entries.
    - Below the legend are several disjoint grey hollow rectangles, each a one-pixel-thick frame.
    - Inside each frame, a few non-grey colored fragment regions sit on a large black interior.

    Rule:
    - Read the legend left-to-right as a mapping from fragment color to output fill color.
    - For each frame, collect the fragment colors present inside it and use those fragments as multi-source seeds.
    - Expand the seeds through only the black interior with 4-neighbor distance; each black cell belongs to the nearest seed color, and exact ties are won by the earlier legend entry.
    - Recolor every cell in each claimed region, including the original fragment pixels, with that fragment color's mapped fill color.
    - Keep the grey frame border and the legend row unchanged.

    Key insight:
    - The fragments do two jobs at once: their positions define a Voronoi-style partition of the frame interior, while the legend independently translates each fragment color into the final paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    legend_row, entries = _parse_legend(g)
    legend_order = [int(match) for match, _fill in entries]
    color_to_fill = {int(match): int(fill) for match, fill in entries}

    out = g.copy()
    for r0, c0, r1, c1 in _find_frames(g, legend_row):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _solve_frame(inner, color_to_fill, legend_order)
    return out


_SHAPE_LIBRARY = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
)

GEN_RETRIES = 500
GRID_SIDE_MIN = 14
GRID_SIDE_MAX = 30
FRAME_COUNT_RANGE = (1, 4)
LEGEND_ENTRY_RANGE = (2, 4)
FRAME_INNER_MIN = 3
FRAME_INNER_MAX = 8
TIE_FRAME_MIN = 5
PADDING_ROW = 1


def _normalize_shape(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))


def _random_shape() -> tuple[tuple[int, int], ...]:
    cells = list(random.choice(_SHAPE_LIBRARY))
    for _ in range(random.randint(0, 3)):
        cells = [(int(c), int(-r)) for r, c in cells]
    if random.random() < 0.5:
        cells = [(int(r), int(-c)) for r, c in cells]
    if random.random() < 0.5:
        cells = [(int(-r), int(c)) for r, c in cells]
    return _normalize_shape(cells)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _can_place_shape(occupied: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    h, w = occupied.shape
    cells = {(int(top + r), int(left + c)) for r, c in shape}
    for rr, cc in cells:
        if not (0 <= rr < h and 0 <= cc < w) or occupied[rr, cc]:
            return False
    for rr, cc in cells:
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _place_shape(inner: np.ndarray, occupied: np.ndarray, color: int, shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = inner.shape
    sh, sw = _shape_dims(shape)
    positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place_shape(occupied, shape, top, left):
            continue
        for dr, dc in shape:
            rr = int(top + dr)
            cc = int(left + dc)
            inner[rr, cc] = int(color)
            occupied[rr, cc] = True
        return True
    return False


def _place_tie_pair(inner: np.ndarray, occupied: np.ndarray, colors: list[int]) -> bool:
    h, w = inner.shape
    options: list[tuple[int, int, int, int]] = []
    if w >= TIE_FRAME_MIN:
        for r in range(h):
            for c1 in range(w):
                for c2 in range(c1 + 2, w):
                    if (c2 - c1) % 2 == 0:
                        options.append((0, r, c1, c2))
    if h >= TIE_FRAME_MIN:
        for c in range(w):
            for r1 in range(h):
                for r2 in range(r1 + 2, h):
                    if (r2 - r1) % 2 == 0:
                        options.append((1, c, r1, r2))
    random.shuffle(options)
    for orient, fixed, a, b in options:
        if orient == 0:
            cells = [(int(fixed), int(a), int(colors[0])), (int(fixed), int(b), int(colors[1]))]
        else:
            cells = [(int(a), int(fixed), int(colors[0])), (int(b), int(fixed), int(colors[1]))]
        if any(occupied[r, c] for r, c, _color in cells):
            continue
        ok = True
        for r, c, _color in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for r, c, color in cells:
            inner[r, c] = int(color)
            occupied[r, c] = True
        return True
    return False


def _build_frame_inner(inner_h: int, inner_w: int, colors: list[int], force_tie: bool) -> np.ndarray | None:
    inner = np.full((inner_h, inner_w), BG, dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)

    if force_tie:
        if len(colors) != 2 or not _place_tie_pair(inner, occupied, colors):
            return None
        return inner

    for color in colors:
        for _ in range(50):
            shape = _random_shape()
            sh, sw = _shape_dims(shape)
            if sh > inner_h or sw > inner_w:
                continue
            if _place_shape(inner, occupied, int(color), shape):
                break
        else:
            return None
    return inner


def _build_frame_specs(legend_order: list[int], want_tie: bool) -> list[dict] | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    specs: list[dict] = []
    tie_index = random.randrange(frame_count) if want_tie else None

    single_cycle = legend_order.copy()
    random.shuffle(single_cycle)
    single_ptr = 0

    for idx in range(frame_count):
        force_tie = idx == tie_index
        if force_tie:
            colors = random.sample(legend_order, 2)
            inner_h = random.randint(TIE_FRAME_MIN, FRAME_INNER_MAX)
            inner_w = random.randint(TIE_FRAME_MIN, FRAME_INNER_MAX)
        elif want_tie:
            color_count = random.randint(1, min(3, len(legend_order)))
            colors = random.sample(legend_order, color_count)
            inner_h = random.randint(max(FRAME_INNER_MIN, 3 + (color_count > 1)), FRAME_INNER_MAX)
            inner_w = random.randint(max(FRAME_INNER_MIN, 3 + (color_count > 1)), FRAME_INNER_MAX)
        else:
            colors = [single_cycle[single_ptr % len(single_cycle)]]
            single_ptr += 1
            inner_h = random.randint(FRAME_INNER_MIN, FRAME_INNER_MAX)
            inner_w = random.randint(FRAME_INNER_MIN, FRAME_INNER_MAX)

        inner = _build_frame_inner(inner_h, inner_w, colors, force_tie)
        if inner is None:
            return None
        specs.append(
            {
                "inner": inner,
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )
    return specs


def _pick_grid_shape(legend_width: int, frame_specs: list[dict]) -> tuple[int, int] | None:
    total_area = sum((spec["outer_h"] + 1) * (spec["outer_w"] + 1) for spec in frame_specs)
    max_h = max(int(spec["outer_h"]) for spec in frame_specs)
    max_w = max(int(spec["outer_w"]) for spec in frame_specs)
    side = int(np.ceil(total_area ** 0.5))
    min_h = max(GRID_SIDE_MIN, 3 + max_h, side)
    min_w = max(GRID_SIDE_MIN, legend_width + 1, max_w + 1, side)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None
    return random.randint(min_h, GRID_SIDE_MAX), random.randint(min_w, GRID_SIDE_MAX)


def _place_frames(grid_shape: tuple[int, int], frame_specs: list[dict]) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(frame_specs)
    order = sorted(
        range(len(frame_specs)),
        key=lambda idx: int(frame_specs[idx]["outer_h"]) * int(frame_specs[idx]["outer_w"]),
        reverse=True,
    )
    for idx in order:
        spec = frame_specs[idx]
        fh = int(spec["outer_h"])
        fw = int(spec["outer_w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(PADDING_ROW + 1, h - fh + 1):
            for c0 in range(w - fw + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + fh + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + fw + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (int(r0), int(c0))
        occupied[r0 : r0 + fh, c0 : c0 + fw] = True
    assert all(pos is not None for pos in positions)
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _render_legend(grid: np.ndarray, entries: list[tuple[int, int]], start_col: int) -> None:
    c = int(start_col)
    for idx, (match, fill) in enumerate(entries):
        grid[0, c] = int(match)
        grid[0, c + 1] = int(fill)
        c += 2
        if idx + 1 < len(entries):
            grid[0, c] = GREY
            c += 1


def _render_frames(grid: np.ndarray, frame_specs: list[dict], positions: list[tuple[int, int]]) -> None:
    for spec, (r0, c0) in zip(frame_specs, positions, strict=False):
        fh = int(spec["outer_h"])
        fw = int(spec["outer_w"])
        grid[r0, c0 : c0 + fw] = GREY
        grid[r0 + fh - 1, c0 : c0 + fw] = GREY
        grid[r0 : r0 + fh, c0] = GREY
        grid[r0 : r0 + fh, c0 + fw - 1] = GREY
        inner = spec["inner"]
        ih, iw = inner.shape
        assert fh == ih + 2 and fw == iw + 2
        grid[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = inner


def _expected_output(
    grid: np.ndarray,
    frame_specs: list[dict],
    positions: list[tuple[int, int]],
    color_to_fill: dict[int, int],
    legend_order: list[int],
) -> np.ndarray:
    out = grid.copy()
    for spec, (r0, c0) in zip(frame_specs, positions, strict=False):
        inner = spec["inner"]
        ih, iw = inner.shape
        out[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw] = _solve_frame(inner, color_to_fill, legend_order)
    return out


def _gen_attempt() -> np.ndarray | None:
    want_tie = random.choice([False, True])
    entry_count = random.randint(*LEGEND_ENTRY_RANGE)
    palette = [color for color in range(10) if color not in {BG, GREY}]
    if entry_count * 2 > len(palette):
        return None
    match_colors = random.sample(palette, entry_count)
    fill_pool = [color for color in palette if color not in match_colors]
    fill_colors = random.sample(fill_pool, entry_count)
    entries = [(int(match), int(fill)) for match, fill in zip(match_colors, fill_colors, strict=False)]
    legend_order = [int(match) for match, _fill in entries]
    color_to_fill = {int(match): int(fill) for match, fill in entries}

    frame_specs = _build_frame_specs(legend_order, want_tie)
    if frame_specs is None:
        return None

    legend_width = 3 * entry_count - 1
    grid_shape = _pick_grid_shape(legend_width, frame_specs)
    if grid_shape is None:
        return None
    positions = _place_frames(grid_shape, frame_specs)
    if positions is None:
        return None

    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    legend_start = random.randint(0, w - legend_width)
    _render_legend(grid, entries, legend_start)
    _render_frames(grid, frame_specs, positions)

    has_tie = any(_frame_has_tie(spec["inner"], legend_order) for spec in frame_specs)
    if has_tie != want_tie:
        return None

    expected = _expected_output(grid, frame_specs, positions, color_to_fill, legend_order)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the new ARC task")

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3)
BAR_LENGTHS = (3, 5, 7)
CORE_LENGTHS = (5, 7)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            out.append(np.array(cells, dtype=int))

    return out


def _bar_cells(center: tuple[int, int], length: int, orient: str) -> set[tuple[int, int]]:
    cr, cc = center
    half = length // 2
    if orient == "H":
        return {(cr, cc + dc) for dc in range(-half, half + 1)}
    return {(cr + dr, cc) for dr in range(-half, half + 1)}


def _swap_orient(orient: str) -> str:
    return "V" if orient == "H" else "H"


def _rotated_bar_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(coords.shape[0])

    assert min(height, width) == 1, "grey components must be one-cell-thick bars"
    assert area == max(height, width), "grey bars must be solid"
    length = max(height, width)
    assert length % 2 == 1, "grey bar lengths must be odd"

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    half = length // 2

    if height == 1:
        return [(r, cc) for r in range(cr - half, cr + half + 1)]
    return [(cr, c) for c in range(cc - half, cc + half + 1)]


def _region_map(shape: tuple[int, int], obstacles: set[tuple[int, int]]) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = shape
    labels = np.full((h, w), -2, dtype=int)
    for r, c in obstacles:
        labels[r, c] = -1

    regions: list[list[tuple[int, int]]] = []
    region_id = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] != -2:
                continue
            queue = deque([(r, c)])
            labels[r, c] = region_id
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(shape, nr, nc) and labels[nr, nc] == -2:
                        labels[nr, nc] = region_id
                        queue.append((nr, nc))

            regions.append(cells)
            region_id += 1

    return labels, regions


def _solve(grid: np.ndarray, *, rotate_bars: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    barrier_cells: set[tuple[int, int]] = set()
    for comp in _components_of_color(g, GREY):
        target_cells = _rotated_bar_cells(comp) if rotate_bars else [tuple(map(int, rc)) for rc in comp]
        for r, c in target_cells:
            assert _in_bounds(g.shape, r, c), "rotated bar must stay in bounds"
            assert (r, c) not in barrier_cells, "rotated bars must not overlap"
            barrier_cells.add((r, c))
            out[r, c] = GREY

    marker_cells = [tuple(map(int, rc)) for rc in np.argwhere((g != BG) & (g != GREY))]
    for r, c in marker_cells:
        assert out[r, c] == BG, "markers should not lie on rotated bars"
        out[r, c] = int(g[r, c])

    _labels, regions = _region_map(g.shape, barrier_cells)
    for cells in regions:
        counts = Counter(int(g[r, c]) for r, c in cells if int(g[r, c]) not in (BG, GREY))
        if not counts:
            continue
        ranked = counts.most_common()
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "marked regions need a unique majority"
        fill_color = int(ranked[0][0])
        for r, c in cells:
            if out[r, c] == BG:
                out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the grey bars, re-cut the background into regions, and fill each marked region by its marker-color majority.

    Input structure:
    - Black is the background.
    - Grey cells form several disconnected one-cell-thick horizontal or vertical bars of odd length.
    - Other non-black cells are isolated single-pixel markers that sit in the open space between the bars.

    Rule:
    - Rotate every grey bar 90 degrees around its center cell, preserving its length and color.
    - Treat the rotated grey bars as obstacles and flood-fill the remaining cells into 4-connected regions, with marker cells belonging to their containing region.
    - For each region containing markers, count the marker colors in that region and fill every non-marker cell of the region with the unique majority marker color.
    - Regions without markers stay black, marker pixels keep their original colors, and grey bars stay at their rotated positions.

    Key insight:
    - The bar rotation changes which open cells are connected, so marker ownership must be computed after the rotation; only then can the correct regional majority color be chosen.
    """
    return _solve(grid, rotate_bars=True)


def _cells_touch_side(cells: set[tuple[int, int]], others: set[tuple[int, int]]) -> bool:
    return any((nr, nc) in others for r, c in cells for nr, nc in _neighbors4(r, c))


def _scene_bar_sets(specs: list[dict], *, rotated: bool) -> list[set[tuple[int, int]]]:
    parts = []
    for spec in specs:
        orient = spec["out_orient"] if rotated else _swap_orient(spec["out_orient"])
        parts.append(_bar_cells(spec["center"], spec["length"], orient))
    return parts


def _valid_specs(shape: tuple[int, int], specs: list[dict]) -> bool:
    for rotated in (False, True):
        parts = _scene_bar_sets(specs, rotated=rotated)
        for i, cells in enumerate(parts):
            if any(not _in_bounds(shape, r, c) for r, c in cells):
                return False
            for other in parts[:i]:
                if cells & other:
                    return False
                if _cells_touch_side(cells, other):
                    return False
    return True


def _build_core_specs(shape: tuple[int, int], extra_count: int) -> list[dict] | None:
    h, w = shape
    tl_h = random.choice(CORE_LENGTHS)
    tl_v = random.choice(CORE_LENGTHS)
    br_h = random.choice(CORE_LENGTHS)
    br_v = random.choice(CORE_LENGTHS)

    specs = [
        {"center": (tl_v, tl_h // 2), "length": tl_h, "out_orient": "H"},
        {"center": (tl_v // 2, tl_h), "length": tl_v, "out_orient": "V"},
        {"center": (h - 1 - br_v, w - 1 - br_h // 2), "length": br_h, "out_orient": "H"},
        {"center": (h - 1 - br_v // 2, w - 1 - br_h), "length": br_v, "out_orient": "V"},
    ]
    if not _valid_specs(shape, specs):
        return None

    for _ in range(extra_count):
        placed = False
        for _ in range(200):
            spec = {
                "center": (random.randint(3, h - 4), random.randint(3, w - 4)),
                "length": random.choice(BAR_LENGTHS),
                "out_orient": random.choice(("H", "V")),
            }
            if _valid_specs(shape, specs + [spec]):
                specs.append(spec)
                placed = True
                break
        if not placed:
            return None

    return specs


def _render_input(shape: tuple[int, int], specs: list[dict], markers: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in _bar_cells(spec["center"], spec["length"], _swap_orient(spec["out_orient"])):
            grid[r, c] = GREY
    for (r, c), color in markers.items():
        assert grid[r, c] == BG
        grid[r, c] = color
    return grid


def _same_region_signature(labels: np.ndarray, points: list[tuple[int, int]]) -> tuple[bool, ...]:
    out = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            out.append(int(labels[points[i]]) == int(labels[points[j]]))
    return tuple(out)


def _marker_candidates(
    shape: tuple[int, int],
    region_cells: list[tuple[int, int]],
    input_bars: set[tuple[int, int]],
    output_bars: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in region_cells:
        if (r, c) in input_bars or (r, c) in output_bars:
            continue
        if any(_in_bounds(shape, nr, nc) and (nr, nc) in input_bars for nr, nc in _neighbors4(r, c)):
            continue
        candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _pick_cells(
    candidates: list[tuple[int, int]],
    count: int,
    taken: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    picked: list[tuple[int, int]] = []
    blocked = set(taken)
    for cell in candidates:
        if cell in blocked:
            continue
        if any(nb in blocked for nb in _neighbors4(*cell)):
            continue
        picked.append(cell)
        blocked.add(cell)
        for nb in _neighbors4(*cell):
            blocked.add(nb)
        if len(picked) == count:
            return picked
    return None


def _marker_plan() -> tuple[Counter[int], Counter[int]]:
    major_big, major_small, other = random.sample(MARKER_COLORS, 3)
    big_region = Counter({major_big: random.choice((3, 4)), major_small: 1, other: 1})
    small_region = Counter({major_big: 1, major_small: random.choice((2, 3, 4)), other: 1})
    return big_region, small_region


def _validate_generated_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    grey_comps = _components_of_color(g, GREY)
    assert 4 <= len(grey_comps) <= 6

    marker_positions = [tuple(map(int, rc)) for rc in np.argwhere((g != BG) & (g != GREY))]
    marker_counts = Counter(int(g[r, c]) for r, c in marker_positions)
    assert 8 <= len(marker_positions) <= 14
    assert set(marker_counts) == set(MARKER_COLORS)
    assert all(marker_counts[color] >= 2 for color in MARKER_COLORS)

    input_bars: set[tuple[int, int]] = set()
    for comp in grey_comps:
        assert int(comp.shape[0]) in BAR_LENGTHS
        r0, c0, r1, c1 = _bbox_coords(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert min(height, width) == 1
        assert max(height, width) == int(comp.shape[0])
        cells = {tuple(map(int, rc)) for rc in comp}
        assert not (input_bars & cells)
        input_bars.update(cells)

    for r, c in marker_positions:
        assert all(nb not in input_bars for nb in _neighbors4(r, c)), "markers must avoid input bars by side-neighbor"

    output = main(g)
    no_rotate_output = _solve(g, rotate_bars=False)
    assert not np.array_equal(output, no_rotate_output), "rotation must affect the answer"

    rotated_bars = {tuple(map(int, rc)) for rc in np.argwhere(output == GREY)}
    assert len(rotated_bars) == len(input_bars)

    out_labels, out_regions = _region_map(g.shape, rotated_bars)
    in_labels, _ = _region_map(g.shape, input_bars)
    assert len(out_regions) >= 3
    assert _same_region_signature(in_labels, marker_positions) != _same_region_signature(out_labels, marker_positions)

    marked_regions = 0
    mixed_regions = 0
    majority_colors: set[int] = set()

    for cells in out_regions:
        counts = Counter(int(g[r, c]) for r, c in cells if int(g[r, c]) in MARKER_COLORS)
        if not counts:
            for r, c in cells:
                assert int(output[r, c]) == BG
            continue

        marked_regions += 1
        if len(counts) > 1:
            mixed_regions += 1
        ranked = counts.most_common()
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1]
        fill_color = int(ranked[0][0])
        majority_colors.add(fill_color)

        for r, c in cells:
            if int(g[r, c]) in MARKER_COLORS:
                assert int(output[r, c]) == int(g[r, c])
            else:
                assert int(output[r, c]) == fill_color

    assert marked_regions >= 2
    assert marked_regions < len(out_regions), "keep at least one unmarked region"
    assert mixed_regions >= 1
    assert len(majority_colors) >= 2


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        shape = (random.randint(14, 18), random.randint(14, 18))
        specs = _build_core_specs(shape, extra_count=random.choice((0, 1, 2)))
        if specs is None:
            continue

        input_bars = set().union(*_scene_bar_sets(specs, rotated=False))
        output_bars = set().union(*_scene_bar_sets(specs, rotated=True))
        _out_labels, out_regions = _region_map(shape, output_bars)
        if len(out_regions) < 3:
            continue

        sized_regions = sorted(
            [(idx, cells) for idx, cells in enumerate(out_regions)],
            key=lambda item: len(item[1]),
            reverse=True,
        )
        big_id, big_cells = sized_regions[0]
        big_candidates = _marker_candidates(shape, big_cells, input_bars, output_bars)
        if len(big_candidates) < 6:
            continue

        other_options = []
        for rid, cells in sized_regions[1:]:
            candidates = _marker_candidates(shape, cells, input_bars, output_bars)
            if len(candidates) >= 4:
                other_options.append((rid, cells, candidates))
        if not other_options:
            continue

        small_id, _small_cells, small_candidates = random.choice(other_options)
        assert big_id != small_id
        big_plan, small_plan = _marker_plan()

        taken: set[tuple[int, int]] = set()
        markers: dict[tuple[int, int], int] = {}
        ok = True

        for color, count in big_plan.items():
            picks = _pick_cells(big_candidates, count, taken)
            if picks is None:
                ok = False
                break
            for cell in picks:
                markers[cell] = color
            taken.update(picks)
        if not ok:
            continue

        for color, count in small_plan.items():
            picks = _pick_cells(small_candidates, count, taken)
            if picks is None:
                ok = False
                break
            for cell in picks:
                markers[cell] = color
            taken.update(picks)
        if not ok:
            continue

        grid = _render_input(shape, specs, markers)
        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate valid input")

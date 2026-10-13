from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GRAY = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _diag_neighbors(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIAGONALS]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _component_dict(color: int, coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return {
        "color": int(color),
        "coords": np.array(coords, dtype=int),
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "area": len(coords),
    }


def _find_monochrome_components(grid: np.ndarray, *, excluded: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not _in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(_component_dict(color, coords))
    return comps


def _trace_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    bar_mask: np.ndarray,
) -> dict | None:
    h, w = shape
    r, c = start
    dr, dc = direction
    path = [(r, c)]
    collisions: list[tuple[int, int]] = []

    for _ in range(h * w * 2):
        nr, nc = r + dr, c + dc
        if not _in_bounds(shape, nr, nc):
            return {"path": path, "collisions": collisions}
        if bar_mask[nr, nc]:
            collisions.append((nr, nc))
            dr = -dr
            nr, nc = r + dr, c + dc
            if not _in_bounds(shape, nr, nc):
                return {"path": path, "collisions": collisions}
            if bar_mask[nr, nc]:
                return None
        r, c = nr, nc
        path.append((r, c))
    return None


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_monochrome_components(g, excluded={bg, GRAY})
    assert components, "expected a colored start pixel and at least one reflector bar"

    gray_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)]
    assert gray_coords, "expected gray guide/placeholder pixels"
    gray_set = set(gray_coords)
    for r, c in gray_coords:
        assert all((nr, nc) not in gray_set for nr, nc in _neighbors8(r, c)), "gray pixels must be isolated"

    singleton_components = [comp for comp in components if int(comp["area"]) == 1]
    assert len(singleton_components) == 1, "expected exactly one colored singleton start pixel"

    start_comp = singleton_components[0]
    start = tuple(map(int, start_comp["coords"][0]))
    guide_candidates = [
        pos
        for pos in _diag_neighbors(*start)
        if _in_bounds(g.shape, *pos) and int(g[pos]) == GRAY
    ]
    assert len(guide_candidates) == 1, "guide must be the unique gray diagonal neighbor of the start"
    guide = guide_candidates[0]

    bars = [comp for comp in components if comp is not start_comp]
    assert bars, "expected at least one reflector bar"
    bar_mask = np.zeros(g.shape, dtype=bool)
    for bar in bars:
        r0, c0, r1, c1 = bar["bbox"]
        width = c1 - c0 + 1
        assert r0 == r1, "reflectors must be horizontal bars"
        assert int(bar["area"]) == width and width >= 2, "reflectors must be solid horizontal bars"
        for rr, cc in bar["coords"]:
            bar_mask[int(rr), int(cc)] = True

    return {
        "grid": g,
        "bg": bg,
        "start": start,
        "start_color": int(start_comp["color"]),
        "guide": guide,
        "direction": (guide[0] - start[0], guide[1] - start[1]),
        "bars": bars,
        "bar_mask": bar_mask,
        "gray_coords": gray_coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light only the gray placeholders that lie on a diagonally moving ray reflected by horizontal bars.

    Input structure:
    - The background is black.
    - There is exactly one colored start pixel, with one gray diagonal-adjacent guide pixel that gives the initial diagonal direction.
    - The other colored objects are solid horizontal bars, and the remaining gray pixels are isolated placeholder pixels.

    Rule:
    - Fire an invisible diagonal ray from the start in the guide direction.
    - Whenever the next ray step would enter a bar, reflect by flipping only the vertical direction component, then continue until the ray exits the grid.
    - Recolor only the gray placeholder pixels visited by that path to the start color, erase the gray guide pixel, and leave the start and bars unchanged.

    Key insight:
    - The output never shows the ray itself; only the subset of gray singletons that lie on the hidden reflected path become visible in the start color.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    trace = _trace_path(scene["grid"].shape, scene["start"], scene["direction"], scene["bar_mask"])
    assert trace is not None, "ray path must be well-defined"

    out = scene["grid"].copy()
    out[scene["guide"]] = scene["bg"]
    for cell in trace["path"][1:]:
        if cell != scene["guide"] and int(scene["grid"][cell]) == GRAY:
            out[cell] = scene["start_color"]
    return out


def _bar_cells_clear(bar_mask: np.ndarray, cells: list[tuple[int, int]], blocked: set[tuple[int, int]]) -> bool:
    h, w = bar_mask.shape
    for r, c in cells:
        if (r, c) in blocked or bar_mask[r, c]:
            return False
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not _in_bounds((h, w), nr, nc):
                    continue
                if (nr, nc) == (r, c):
                    continue
                if bar_mask[nr, nc]:
                    return False
    return True


def _choose_bar_segment(
    shape: tuple[int, int],
    collision: tuple[int, int],
    bar_mask: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    _h, w = shape
    r, c = collision
    candidates: list[list[tuple[int, int]]] = []
    lengths = [3, 4, 5, 6]
    random.shuffle(lengths)
    for length in lengths:
        offsets = list(range(length))
        random.shuffle(offsets)
        for left_len in offsets:
            c0 = c - left_len
            c1 = c0 + length - 1
            if c0 < 0 or c1 >= w:
                continue
            cells = [(r, cc) for cc in range(c0, c1 + 1)]
            if _bar_cells_clear(bar_mask, cells, blocked):
                candidates.append(cells)
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _build_bars(
    shape: tuple[int, int],
    start: tuple[int, int],
    guide: tuple[int, int],
    direction: tuple[int, int],
    bounce_count: int,
) -> tuple[list[list[tuple[int, int]]], dict] | None:
    bar_mask = np.zeros(shape, dtype=bool)
    bars: list[list[tuple[int, int]]] = []
    blocked = {start, guide}

    for bounce_idx in range(bounce_count):
        trace = _trace_path(shape, start, direction, bar_mask)
        if trace is None:
            return None
        path = trace["path"]
        remaining_bounces = bounce_count - bounce_idx - 1
        candidates: list[tuple[int, tuple[int, int]]] = []
        for idx in range(2, len(path) - 1):
            prev = path[idx - 1]
            collision = path[idx]
            dr = collision[0] - prev[0]
            dc = collision[1] - prev[1]
            reflected_next = (prev[0] - dr, prev[1] + dc)
            if not _in_bounds(shape, *reflected_next) or bar_mask[reflected_next]:
                continue
            if _chebyshev(collision, start) <= 1 or _chebyshev(collision, guide) <= 1:
                continue
            horizontal_room = (shape[1] - 1 - collision[1]) if dc == 1 else collision[1]
            if horizontal_room < 2 * remaining_bounces + 1:
                continue
            candidates.append((idx, collision))
        random.shuffle(candidates)

        placed = False
        for idx, collision in candidates:
            segment = _choose_bar_segment(shape, collision, bar_mask, blocked)
            if segment is None:
                continue
            for cell in segment:
                bar_mask[cell] = True
            updated = _trace_path(shape, start, direction, bar_mask)
            if (
                updated is not None
                and len(updated["collisions"]) == bounce_idx + 1
                and updated["collisions"][-1] == collision
            ):
                bars.append(segment)
                placed = True
                break
            for cell in segment:
                bar_mask[cell] = False
        if not placed:
            return None

    final_trace = _trace_path(shape, start, direction, bar_mask)
    if final_trace is None or len(final_trace["collisions"]) != bounce_count:
        return None
    return bars, final_trace


def _basic_gray_candidate(
    cell: tuple[int, int],
    shape: tuple[int, int],
    start: tuple[int, int],
    guide: tuple[int, int],
    bar_mask: np.ndarray,
) -> bool:
    return (
        _in_bounds(shape, *cell)
        and not bar_mask[cell]
        and cell != start
        and cell != guide
        and _chebyshev(cell, guide) > 1
        and _chebyshev(cell, start) > 1
    )


def _select_from_groups(
    groups: list[tuple[str, list[tuple[int, int]]]],
    start: tuple[int, int],
    guide: tuple[int, int],
) -> list[tuple[str, tuple[int, int]]] | None:
    chosen: list[tuple[str, tuple[int, int]]] = []

    def fits(cell: tuple[int, int]) -> bool:
        if _chebyshev(cell, guide) <= 1 or _chebyshev(cell, start) <= 1:
            return False
        return all(_chebyshev(cell, other) > 1 for _label, other in chosen)

    def dfs(i: int) -> bool:
        if i == len(groups):
            return True
        label, candidates = groups[i]
        random.shuffle(candidates)
        for cell in candidates:
            if not fits(cell):
                continue
            chosen.append((label, cell))
            if dfs(i + 1):
                return True
            chosen.pop()
        return False

    return chosen if dfs(0) else None


def _take_spaced_cells(
    pool: list[tuple[int, int]],
    count: int,
    start: tuple[int, int],
    guide: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    picked: set[tuple[int, int]] = set()
    random.shuffle(pool)
    for cell in pool:
        if len(picked) >= count:
            break
        if _chebyshev(cell, start) <= 1 or _chebyshev(cell, guide) <= 1:
            continue
        if any(_chebyshev(cell, other) <= 1 for other in occupied | picked):
            continue
        picked.add(cell)
    return picked


def _choose_placeholders(
    shape: tuple[int, int],
    start: tuple[int, int],
    guide: tuple[int, int],
    direction: tuple[int, int],
    bars: list[list[tuple[int, int]]],
    full_bar_mask: np.ndarray,
    true_trace: dict,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    true_path = true_trace["path"][1:]
    true_set = set(true_path)
    groups: list[tuple[str, list[tuple[int, int]]]] = []
    alt_decoy_pool: list[tuple[int, int]] = []

    for segment in bars:
        alt_mask = full_bar_mask.copy()
        for cell in segment:
            alt_mask[cell] = False
        alt_trace = _trace_path(shape, start, direction, alt_mask)
        if alt_trace is None:
            return None
        alt_set = set(alt_trace["path"][1:])

        true_candidates = [
            cell
            for cell in true_trace["path"][2:]
            if cell not in alt_set and _basic_gray_candidate(cell, shape, start, guide, full_bar_mask)
        ]
        alt_candidates = [
            cell
            for cell in alt_trace["path"][1:]
            if cell not in true_set and _basic_gray_candidate(cell, shape, start, guide, full_bar_mask)
        ]
        if not true_candidates or not alt_candidates:
            return None
        groups.append(("lit", list(dict.fromkeys(true_candidates))))
        groups.append(("decoy", list(dict.fromkeys(alt_candidates))))
        alt_decoy_pool.extend(alt_candidates)

    chosen = _select_from_groups(groups, start, guide)
    if chosen is None:
        return None

    lit = {cell for label, cell in chosen if label == "lit"}
    decoy = {cell for label, cell in chosen if label == "decoy"}
    occupied_gray = lit | decoy

    min_lit_total = max(2, len(bars))
    min_decoy_total = max(2, len(bars))

    extra_lit_pool = [
        cell
        for cell in true_trace["path"][2:]
        if cell not in occupied_gray and _basic_gray_candidate(cell, shape, start, guide, full_bar_mask)
    ]
    lit |= _take_spaced_cells(extra_lit_pool, max(0, min_lit_total - len(lit)), start, guide, occupied_gray)
    occupied_gray = lit | decoy

    extra_decoy_pool = [
        cell
        for cell in dict.fromkeys(alt_decoy_pool)
        if cell not in occupied_gray and cell not in true_set
    ]
    decoy |= _take_spaced_cells(extra_decoy_pool, max(0, min_decoy_total - len(decoy)), start, guide, occupied_gray)
    occupied_gray = lit | decoy

    if len(decoy) < min_decoy_total:
        all_off_path = [
            (r, c)
            for r in range(shape[0])
            for c in range(shape[1])
            if (
                (r, c) not in true_set
                and _basic_gray_candidate((r, c), shape, start, guide, full_bar_mask)
                and (r, c) not in occupied_gray
            )
        ]
        decoy |= _take_spaced_cells(all_off_path, min_decoy_total - len(decoy), start, guide, occupied_gray)

    occupied_gray = lit | decoy
    extra_lit_count = random.randint(0, 2)
    lit |= _take_spaced_cells(extra_lit_pool, extra_lit_count, start, guide, occupied_gray)
    occupied_gray = lit | decoy

    extra_decoy_count = random.randint(0, 2)
    all_off_path = [
        (r, c)
        for r in range(shape[0])
        for c in range(shape[1])
        if (
            (r, c) not in true_set
            and _basic_gray_candidate((r, c), shape, start, guide, full_bar_mask)
            and (r, c) not in occupied_gray
        )
    ]
    decoy |= _take_spaced_cells(all_off_path, extra_decoy_count, start, guide, occupied_gray)

    if len(lit) < min_lit_total or len(decoy) < min_decoy_total:
        return None
    return lit, decoy


def _render_scene(
    shape: tuple[int, int],
    start: tuple[int, int],
    start_color: int,
    guide: tuple[int, int],
    bars: list[list[tuple[int, int]]],
    lit: set[tuple[int, int]],
    decoy: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[start] = int(start_color)
    grid[guide] = GRAY
    bar_palette = [color for color in NON_SPECIAL_COLORS if color != start_color]
    for segment in bars:
        color = random.choice(bar_palette)
        for cell in segment:
            grid[cell] = color
    for cell in lit | decoy:
        assert int(grid[cell]) == BLACK
        grid[cell] = GRAY
    return grid


def _semantic_checks(
    grid: np.ndarray,
    lit: set[tuple[int, int]],
    decoy: set[tuple[int, int]],
    bars: list[list[tuple[int, int]]],
) -> bool:
    scene = _parse_scene(grid)
    trace = _trace_path(grid.shape, scene["start"], scene["direction"], scene["bar_mask"])
    if trace is None:
        return False

    path_set = set(trace["path"][1:])
    if lit and not all(cell in path_set for cell in lit):
        return False
    if not decoy or any(cell in path_set for cell in decoy):
        return False

    base_lit = {cell for cell in lit | decoy if cell in path_set}
    for segment in bars:
        variant_mask = scene["bar_mask"].copy()
        for cell in segment:
            variant_mask[cell] = False
        variant_trace = _trace_path(grid.shape, scene["start"], scene["direction"], variant_mask)
        if variant_trace is None:
            return False
        variant_path_set = set(variant_trace["path"][1:])
        variant_lit = {cell for cell in lit | decoy if cell in variant_path_set}
        if variant_lit == base_lit:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(14, 28)
    direction = random.choice(DIAGONALS)
    dr, dc = direction

    if dc == 1:
        start_c = random.randint(1, max(1, w // 3))
    else:
        start_c = random.randint(max(1, (2 * w) // 3), w - 2)
    if dr == 1:
        start_r = random.randint(1, h - 4)
    else:
        start_r = random.randint(3, h - 2)

    start = (start_r, start_c)
    guide = (start_r + dr, start_c + dc)
    if not _in_bounds((h, w), *guide):
        return None

    bounce_count = random.choices([1, 2, 3], weights=[2, 4, 4], k=1)[0]
    built = _build_bars((h, w), start, guide, direction, bounce_count)
    if built is None:
        return None
    bars, true_trace = built

    full_bar_mask = np.zeros((h, w), dtype=bool)
    for segment in bars:
        for cell in segment:
            full_bar_mask[cell] = True

    placeholders = _choose_placeholders((h, w), start, guide, direction, bars, full_bar_mask, true_trace)
    if placeholders is None:
        return None
    lit, decoy = placeholders

    start_color = random.choice(NON_SPECIAL_COLORS)
    grid = _render_scene((h, w), start, start_color, guide, bars, lit, decoy)

    expected = grid.copy()
    expected[guide] = BLACK
    for cell in lit:
        expected[cell] = start_color

    if not np.array_equal(main(grid), expected):
        return None
    if not _semantic_checks(grid, lit, decoy, bars):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bounce-light ARC input")

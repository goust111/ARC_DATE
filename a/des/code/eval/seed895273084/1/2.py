from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from itertools import combinations
import random

import numpy as np


BG = 0
AZURE = 8
YELLOW = 4

DIR_TO_GROUP = {
    (-1, 0): 0,  # up
    (0, 1): 1,   # right
    (1, 0): 2,   # down
    (0, -1): 3,  # left
}
GROUP_TO_DIR = {group: vec for vec, group in DIR_TO_GROUP.items()}

BODY_COLORS = (1, 2, 3, 5, 6, 7, 9)

TARGET_H_RANGE = (4, 6)
TARGET_W_RANGE = (5, 7)
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
MIN_PIECES = 4
MAX_PIECES = 7
MAX_CORE_SIZE = 4
GEN_ATTEMPTS = 400
SPEC_ATTEMPTS = 120
LAYOUT_TEMPLATE_TRIES = 10
PIECE_PLACE_TRIES = 250
BLUEPRINT_TARGET = 24
BLUEPRINT_ATTEMPTS = 3000

_EXTRA_POOL = (
    (0, 1),
    (1, 0),
    (1, 1),
    (0, 2),
    (1, 2),
    (2, 0),
    (2, 1),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(g: np.ndarray) -> int:
    counts = Counter(int(v) for v in g.ravel())
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _connected(cells: set[tuple[int, int]]) -> bool:
    todo = deque([next(iter(cells))])
    seen = {todo[0]}
    while todo:
        r, c = todo.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return seen == cells


def _on_edge(cells: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    return any(nei not in cells for nei in _neighbors4(*cell))


def _shape_is_rectangle(cells: tuple[tuple[int, int], ...]) -> bool:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _build_template_library() -> dict[int, dict[int, tuple[tuple[tuple[int, int], ...], ...]]]:
    library: dict[int, dict[int, tuple[tuple[tuple[int, int], ...], ...]]] = {}
    for group, direction in GROUP_TO_DIR.items():
        base = {(0, 0), direction}
        pool = [cell for cell in _EXTRA_POOL if cell not in base]
        size_map: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}
        for core_size in range(1, MAX_CORE_SIZE + 1):
            variants: set[tuple[tuple[int, int], ...]] = set()
            for combo in combinations(pool, core_size):
                cells = base | set(combo)
                if not _connected(cells):
                    continue
                if not _on_edge(cells, (0, 0)) or not _on_edge(cells, direction):
                    continue
                variants.add(tuple(sorted(cells)))
            ordered = sorted(
                variants,
                key=lambda cells: (_shape_is_rectangle(cells), len(cells), cells),
            )
            size_map[core_size] = tuple(ordered)
        library[group] = size_map
    return library


TEMPLATE_LIBRARY = _build_template_library()
SIZE_COMBOS = {
    count: tuple(combinations(range(1, MAX_CORE_SIZE + 1), count))
    for count in range(MAX_CORE_SIZE + 1)
}


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            todo = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while todo:
                cr, cc = todo.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        todo.append((nr, nc))
            comps.append(comp)
    return comps


def _corner_slice(shape: tuple[int, int], corner: str, rect_h: int, rect_w: int) -> tuple[slice, slice]:
    h, w = shape
    if corner == "tl":
        return slice(0, rect_h), slice(0, rect_w)
    if corner == "tr":
        return slice(0, rect_h), slice(w - rect_w, w)
    if corner == "bl":
        return slice(h - rect_h, h), slice(0, rect_w)
    return slice(h - rect_h, h), slice(w - rect_w, w)


def _detect_corner_marker(g: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    corners = {
        "tl": (0, 0, 1, 1),
        "tr": (0, w - 1, 1, -1),
        "bl": (h - 1, 0, -1, 1),
        "br": (h - 1, w - 1, -1, -1),
    }
    for corner, (r0, c0, rs, cs) in corners.items():
        color = int(g[r0, c0])
        if color == bg:
            continue
        rect_h = 0
        while 0 <= r0 + rs * rect_h < h and int(g[r0 + rs * rect_h, c0]) == color:
            rect_h += 1
        rect_w = 0
        while 0 <= c0 + cs * rect_w < w and int(g[r0, c0 + cs * rect_w]) == color:
            rect_w += 1
        if rect_h < 2 or rect_w < 2:
            continue
        rslice, cslice = _corner_slice((h, w), corner, rect_h, rect_w)
        if not np.all(g[rslice, cslice] == color):
            continue

        ok = True
        if corner in ("tl", "tr"):
            r_out = rect_h
        else:
            r_out = h - rect_h - 1
        if 0 <= r_out < h:
            row = g[r_out, cslice]
            if np.any(row == color):
                ok = False
        if corner in ("tl", "bl"):
            c_out = rect_w
        else:
            c_out = w - rect_w - 1
        if 0 <= c_out < w:
            col = g[rslice, c_out]
            if np.any(col == color):
                ok = False
        if not ok:
            continue

        mask = np.zeros_like(g, dtype=bool)
        mask[rslice, cslice] = True
        candidates.append((rect_h, rect_w, mask))

    assert len(candidates) == 1, "expected exactly one solid corner size marker"
    return candidates[0]


def _piece_info(g: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    colors = {int(g[r, c]) for r, c in coords}
    assert len(colors) >= 3, "piece must be multi-color"

    anchors = [(r, c) for r, c in coords if int(g[r, c]) == AZURE]
    markers = [(r, c) for r, c in coords if int(g[r, c]) == YELLOW]
    assert len(anchors) == 1 and len(markers) == 1, "each piece must have one anchor and one direction pixel"
    ar, ac = anchors[0]
    mr, mc = markers[0]
    direction = (mr - ar, mc - ac)
    assert direction in DIR_TO_GROUP, "direction pixel must be 4-neighbor adjacent to the anchor"

    body = [(r, c) for r, c in coords if int(g[r, c]) not in (AZURE, YELLOW)]
    assert body, "piece must contain body pixels"
    counts = Counter(int(g[r, c]) for r, c in body)
    max_count = max(counts.values())
    dominant = [color for color, count in counts.items() if count == max_count]
    assert len(dominant) == 1, "piece must have a unique largest body color"
    dominant_color = int(dominant[0])

    rel_cells: list[tuple[int, int, int]] = []
    for r, c in coords:
        color = dominant_color if int(g[r, c]) in (AZURE, YELLOW) else int(g[r, c])
        rel_cells.append((r - ar, c - ac, color))
    rel_cells.sort()

    return {
        "group": DIR_TO_GROUP[direction],
        "core_size": len(body),
        "cells": tuple(rel_cells),
    }


def _first_feasible_anchor(
    out_h: int,
    out_w: int,
    occ: np.ndarray,
    cells: tuple[tuple[int, int, int], ...],
) -> tuple[int, int] | None:
    for r in range(out_h):
        for c in range(out_w):
            ok = True
            for dr, dc, _ in cells:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < out_h and 0 <= cc < out_w) or occ[rr, cc]:
                    ok = False
                    break
            if ok:
                return (r, c)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    out_h, out_w, marker_mask = _detect_corner_marker(g, bg)
    piece_mask = (g != bg) & (~marker_mask)
    components = _find_components(piece_mask)
    assert components, "expected at least one piece"

    pieces = [_piece_info(g, comp) for comp in components]
    pieces.sort(key=lambda piece: (int(piece["group"]), int(piece["core_size"])))

    seen_group_sizes: set[tuple[int, int]] = set()
    for piece in pieces:
        key = (int(piece["group"]), int(piece["core_size"]))
        assert key not in seen_group_sizes, "piece order inside a group must be unambiguous"
        seen_group_sizes.add(key)

    out = np.full((out_h, out_w), bg, dtype=int)
    occ = np.zeros((out_h, out_w), dtype=bool)
    for piece in pieces:
        cells = piece["cells"]
        assert isinstance(cells, tuple)
        anchor = _first_feasible_anchor(out_h, out_w, occ, cells)
        assert anchor is not None, "every ordered piece must fit somewhere in the target rectangle"
        ar, ac = anchor
        for dr, dc, color in cells:
            rr = ar + dr
            cc = ac + dc
            occ[rr, cc] = True
            out[rr, cc] = int(color)

    assert bool(np.all(occ)), "pieces must exactly fill the target rectangle"
    return out


def _choose_specs(area: int) -> list[tuple[int, int]] | None:
    max_pieces = min(MAX_PIECES, area // 3)
    if max_pieces < MIN_PIECES:
        return None
    for _ in range(SPEC_ATTEMPTS):
        n_pieces = random.randint(MIN_PIECES, max_pieces)
        distinct_groups = 4 if random.random() < 0.4 else random.randint(2, 4)
        active_groups = random.sample(range(4), distinct_groups)
        groups = active_groups[:] + [random.choice(active_groups) for _ in range(n_pieces - distinct_groups)]
        random.shuffle(groups)
        groups.sort()
        counts = [groups.count(group) for group in range(4)]
        if max(counts) > MAX_CORE_SIZE:
            continue

        target_core_sum = area - 2 * n_pieces
        if not (n_pieces <= target_core_sum <= n_pieces * MAX_CORE_SIZE):
            continue

        options: list[list[tuple[int, ...]]] = []

        def rec(group: int, remaining: int, acc: list[tuple[int, ...]]) -> None:
            if group == 4:
                if remaining == 0:
                    options.append(acc[:])
                return
            for combo in SIZE_COMBOS[counts[group]]:
                combo_sum = sum(combo)
                if combo_sum > remaining:
                    continue
                acc.append(combo)
                rec(group + 1, remaining - combo_sum, acc)
                acc.pop()

        rec(0, target_core_sum, [])
        if not options:
            continue

        sizes_by_group = random.choice(options)
        specs: list[tuple[int, int]] = []
        for group in range(4):
            for core_size in sizes_by_group[group]:
                specs.append((group, int(core_size)))
        return specs
    return None


def _template_candidates(group: int, core_size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    variants = list(TEMPLATE_LIBRARY[group][core_size])
    irregular = [cells for cells in variants if not _shape_is_rectangle(cells)]
    regular = [cells for cells in variants if _shape_is_rectangle(cells)]
    random.shuffle(irregular)
    random.shuffle(regular)
    ordered = irregular + regular
    return tuple(ordered[: min(len(ordered), LAYOUT_TEMPLATE_TRIES)])


def _first_feasible_mask(
    out_h: int,
    out_w: int,
    shape: tuple[tuple[int, int], ...],
    occ_mask: int,
) -> tuple[int, int, int] | None:
    for r in range(out_h):
        for c in range(out_w):
            mask = 0
            ok = True
            for dr, dc in shape:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < out_h and 0 <= cc < out_w):
                    ok = False
                    break
                bit = 1 << (rr * out_w + cc)
                if occ_mask & bit:
                    ok = False
                    break
                mask |= bit
            if ok:
                return (r, c, mask)
    return None


def _search_layout(
    out_h: int,
    out_w: int,
    specs: tuple[tuple[int, int], ...],
    candidates: tuple[tuple[tuple[tuple[int, int], ...], ...], ...],
) -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] | None:
    full_mask = (1 << (out_h * out_w)) - 1

    @lru_cache(None)
    def rec(index: int, occ_mask: int) -> tuple[tuple[tuple[int, int], tuple[tuple[int, int], ...]], ...] | None:
        if index == len(specs):
            return tuple() if occ_mask == full_mask else None
        for shape in candidates[index]:
            placed = _first_feasible_mask(out_h, out_w, shape, occ_mask)
            if placed is None:
                continue
            ar, ac, mask = placed
            rest = rec(index + 1, occ_mask | mask)
            if rest is not None:
                return (((ar, ac), shape),) + rest
        return None

    result = rec(0, 0)
    if result is None:
        return None
    return list(result)


def _choose_blueprint() -> tuple[int, int, tuple[tuple[int, int, tuple[int, int], tuple[tuple[int, int], ...]], ...]] | None:
    for _ in range(GEN_ATTEMPTS):
        out_h = random.randint(*TARGET_H_RANGE)
        out_w = random.randint(*TARGET_W_RANGE)
        specs = _choose_specs(out_h * out_w)
        if specs is None:
            continue
        spec_tuple = tuple(specs)
        candidates = tuple(_template_candidates(group, core_size) for group, core_size in spec_tuple)
        if any(not options for options in candidates):
            continue
        placed = _search_layout(out_h, out_w, spec_tuple, candidates)
        if placed is None:
            continue

        irregular_count = sum(not _shape_is_rectangle(shape) for _, shape in placed)
        if irregular_count == 0:
            continue

        blueprint = tuple(
            (group, core_size, anchor, shape)
            for (group, core_size), (anchor, shape) in zip(spec_tuple, placed)
        )
        return out_h, out_w, blueprint
    return None


def _build_blueprints() -> tuple[tuple[int, int, tuple[tuple[int, int, tuple[int, int], tuple[tuple[int, int], ...]], ...]], ...]:
    blueprints: list[tuple[int, int, tuple[tuple[int, int, tuple[int, int], tuple[tuple[int, int], ...]], ...]]] = []
    seen: set[tuple[object, ...]] = set()
    for _ in range(BLUEPRINT_ATTEMPTS):
        blueprint = _choose_blueprint()
        if blueprint is None:
            continue
        out_h, out_w, entries = blueprint
        signature = (out_h, out_w, entries)
        if signature in seen:
            continue
        seen.add(signature)
        blueprints.append(blueprint)
        if len(blueprints) >= BLUEPRINT_TARGET:
            break
    assert blueprints, "failed to precompute at least one valid layout blueprint"
    return tuple(blueprints)


def _instantiate_layout(
    blueprint: tuple[int, int, tuple[tuple[int, int, tuple[int, int], tuple[tuple[int, int], ...]], ...]]
) -> tuple[int, int, list[dict[str, object]], np.ndarray]:
    out_h, out_w, entries = blueprint
    pieces: list[dict[str, object]] = []
    expected = np.full((out_h, out_w), BG, dtype=int)

    for group, core_size, anchor, shape in entries:
        dominant = int(random.choice(BODY_COLORS))
        accent = None
        if core_size >= 3 and random.random() < 0.6:
            accent = int(random.choice([color for color in BODY_COLORS if color != dominant]))
        direction = GROUP_TO_DIR[group]
        extras = [cell for cell in shape if cell not in ((0, 0), direction)]
        random.shuffle(extras)
        accent_cells = {extras[0]} if accent is not None and extras else set()

        input_cells: list[tuple[int, int, int]] = []
        ar, ac = anchor
        for dr, dc in shape:
            if (dr, dc) == (0, 0):
                in_color = AZURE
                out_color = dominant
            elif (dr, dc) == direction:
                in_color = YELLOW
                out_color = dominant
            elif (dr, dc) in accent_cells:
                in_color = int(accent)
                out_color = int(accent)
            else:
                in_color = dominant
                out_color = dominant
            input_cells.append((dr, dc, in_color))
            expected[ar + dr, ac + dc] = out_color

        pieces.append(
            {
                "group": group,
                "core_size": core_size,
                "input_cells": tuple(sorted(input_cells)),
                "bbox_area": (
                    (max(r for r, _, _ in input_cells) - min(r for r, _, _ in input_cells) + 1)
                    * (max(c for _, c, _ in input_cells) - min(c for _, c, _ in input_cells) + 1)
                ),
            }
        )

    assert bool(np.all(expected != BG))
    return out_h, out_w, pieces, expected


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_corner_marker(grid: np.ndarray, blocked: np.ndarray, out_h: int, out_w: int, color: int) -> None:
    h, w = grid.shape
    corner = random.choice(("tl", "tr", "bl", "br"))
    rslice, cslice = _corner_slice((h, w), corner, out_h, out_w)
    grid[rslice, cslice] = color
    marker_cells = [(r, c) for r in range(*rslice.indices(h)) for c in range(*cslice.indices(w))]
    _mark_blocked(blocked, marker_cells)


def _scatter_scene(out_h: int, out_w: int, pieces: list[dict[str, object]]) -> np.ndarray | None:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        if out_h + 4 >= h or out_w + 4 >= w:
            continue

        grid = np.full((h, w), BG, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        marker_color = int(random.choice(BODY_COLORS))
        _place_corner_marker(grid, blocked, out_h, out_w, marker_color)

        order = sorted(range(len(pieces)), key=lambda idx: int(pieces[idx]["bbox_area"]), reverse=True)
        placed_ok = True
        for idx in order:
            cells = pieces[idx]["input_cells"]
            min_r = min(dr for dr, _, _ in cells)
            max_r = max(dr for dr, _, _ in cells)
            min_c = min(dc for _, dc, _ in cells)
            max_c = max(dc for _, dc, _ in cells)

            low_r = 1 - min_r
            high_r = (h - 2) - max_r
            low_c = 1 - min_c
            high_c = (w - 2) - max_c
            if low_r > high_r or low_c > high_c:
                placed_ok = False
                break

            anchors = [
                (random.randint(low_r, high_r), random.randint(low_c, high_c))
                for _ in range(PIECE_PLACE_TRIES)
            ]
            found = False
            for ar, ac in anchors:
                coords = [(ar + dr, ac + dc) for dr, dc, _ in cells]
                if any(blocked[r, c] for r, c in coords):
                    continue
                for dr, dc, color in cells:
                    grid[ar + dr, ac + dc] = int(color)
                _mark_blocked(blocked, coords)
                found = True
                break
            if not found:
                placed_ok = False
                break

        if placed_ok:
            return grid
    return None


BLUEPRINTS = _build_blueprints()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        out_h, out_w, pieces, expected = _instantiate_layout(random.choice(BLUEPRINTS))
        grid = _scatter_scene(out_h, out_w, pieces)
        if grid is None:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GEN_ATTEMPTS} attempts")

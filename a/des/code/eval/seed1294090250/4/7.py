from __future__ import annotations

import random
from itertools import combinations

import numpy as np


# ARC color roles used by this puzzle.
BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PIPE = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

PRIMARY_SOURCE_COLORS = (RED, BLUE, YELLOW, CYAN)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_2x2(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != 4:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _mix_pair(a: int, b: int) -> int:
    if a == b:
        return int(a)
    pair = frozenset((int(a), int(b)))
    if pair == frozenset((RED, BLUE)):
        return MAGENTA
    if pair == frozenset((RED, YELLOW)):
        return ORANGE
    if pair == frozenset((BLUE, YELLOW)):
        return GREEN
    return MAROON


def _mix_colors(colors: list[int] | set[int]) -> int:
    ordered = list(dict.fromkeys(int(c) for c in colors))
    assert ordered, "component must have at least one source color"
    mixed = ordered[0]
    for color in ordered[1:]:
        mixed = _mix_pair(mixed, int(color))
    return int(mixed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each 2x2 dye source through the grey pipe network and recolor each
    pipe component by the mix of all sources attached to that component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    h, w = g.shape

    source_cells_mask = np.zeros((h, w), dtype=bool)
    sources: list[dict[str, object]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for comp in _components(g == int(color)):
            if _is_solid_2x2(comp):
                for r, c in comp:
                    source_cells_mask[r, c] = True
                sources.append({"color": int(color), "cells": comp})

    assert sources, "expected at least one 2x2 source block"

    non_source_fg = (g != bg) & (~source_cells_mask)
    remaining_colors = sorted(int(c) for c in np.unique(g[non_source_fg]))
    assert remaining_colors, "expected a pipe color"
    assert len(remaining_colors) == 1, "expected only one non-source foreground color"
    pipe_color = remaining_colors[0]

    pipe_comps = _components(g == int(pipe_color))
    pipe_label = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(pipe_comps):
        for r, c in comp:
            pipe_label[r, c] = idx

    comp_sources: list[set[int]] = [set() for _ in pipe_comps]
    for source in sources:
        touched: set[int] = set()
        for r, c in source["cells"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and pipe_label[nr, nc] >= 0:
                    touched.add(int(pipe_label[nr, nc]))
        assert touched, "each source block must touch the pipe network"
        for idx in touched:
            comp_sources[idx].add(int(source["color"]))

    out = g.copy()
    for idx, comp in enumerate(pipe_comps):
        if not comp_sources[idx]:
            continue
        mixed = _mix_colors(comp_sources[idx])
        for r, c in comp:
            out[r, c] = int(mixed)
    return out


def _delta_to_dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(b[0] - a[0]), int(b[1] - a[1])


def _opposite(direction: tuple[int, int]) -> tuple[int, int]:
    return -int(direction[0]), -int(direction[1])


def _can_add_cell(cell: tuple[int, int], prev: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if cell in occupied:
        return False
    cr, cc = cell
    for nr, nc in _neighbors4(cr, cc):
        if (nr, nc) in occupied and (nr, nc) != prev:
            return False
    return True


def _build_backbone() -> list[tuple[int, int]] | None:
    for _ in range(120):
        n_segments = random.randint(1, 3)
        ordered = [(0, 0)]
        occupied = {(0, 0)}
        prev_dir: tuple[int, int] | None = None
        ok = True
        for _seg in range(n_segments):
            dirs = DIRS[:]
            if prev_dir is not None:
                dirs = [d for d in dirs if d != _opposite(prev_dir)]
                if random.random() < 0.7:
                    turned = [d for d in dirs if d != prev_dir]
                    if turned:
                        dirs = turned
            random.shuffle(dirs)

            placed = False
            for direction in dirs:
                trial = ordered[-1]
                trail: list[tuple[int, int]] = []
                trial_occupied = set(occupied)
                length = random.randint(2, 4)
                for _step in range(length):
                    nxt = (trial[0] + direction[0], trial[1] + direction[1])
                    if not _can_add_cell(nxt, trial, trial_occupied):
                        trail = []
                        break
                    trail.append(nxt)
                    trial_occupied.add(nxt)
                    trial = nxt
                if trail:
                    ordered.extend(trail)
                    occupied = trial_occupied
                    prev_dir = direction
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok or len(ordered) < 3:
            continue
        r0, c0, r1, c1 = _bbox(ordered)
        if r1 - r0 <= 10 and c1 - c0 <= 10:
            return ordered
    return None


def _degrees(pipe: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {
        cell: sum((nr, nc) in pipe for nr, nc in _neighbors4(cell[0], cell[1]))
        for cell in pipe
    }


def _available_branch_dirs(attach: tuple[int, int], pipe: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    ar, ac = attach
    for direction in DIRS:
        nxt = (ar + direction[0], ac + direction[1])
        if _can_add_cell(nxt, attach, pipe):
            out.append(direction)
    return out


def _extend_branch(
    pipe: set[tuple[int, int]],
    attach: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> bool:
    cur = attach
    for _ in range(length):
        nxt = (cur[0] + direction[0], cur[1] + direction[1])
        if not _can_add_cell(nxt, cur, pipe):
            return False
        pipe.add(nxt)
        cur = nxt
    return True


def _block_options(endpoint: tuple[int, int], outward: tuple[int, int]) -> list[set[tuple[int, int]]]:
    r, c = endpoint
    if outward == (-1, 0):
        return [
            {(r - 2, c - 1), (r - 2, c), (r - 1, c - 1), (r - 1, c)},
            {(r - 2, c), (r - 2, c + 1), (r - 1, c), (r - 1, c + 1)},
        ]
    if outward == (1, 0):
        return [
            {(r + 1, c - 1), (r + 1, c), (r + 2, c - 1), (r + 2, c)},
            {(r + 1, c), (r + 1, c + 1), (r + 2, c), (r + 2, c + 1)},
        ]
    if outward == (0, -1):
        return [
            {(r - 1, c - 2), (r - 1, c - 1), (r, c - 2), (r, c - 1)},
            {(r, c - 2), (r, c - 1), (r + 1, c - 2), (r + 1, c - 1)},
        ]
    assert outward == (0, 1)
    return [
        {(r - 1, c + 1), (r - 1, c + 2), (r, c + 1), (r, c + 2)},
        {(r, c + 1), (r, c + 2), (r + 1, c + 1), (r + 1, c + 2)},
    ]


def _valid_block_option(
    block: set[tuple[int, int]],
    endpoint: tuple[int, int],
    pipe: set[tuple[int, int]],
    used_blocks: set[tuple[int, int]],
) -> bool:
    for cell in block:
        if cell in pipe or cell in used_blocks:
            return False
    for cell in block:
        for nbr in _neighbors4(cell[0], cell[1]):
            if nbr in pipe and nbr != endpoint:
                return False
            if nbr in used_blocks and nbr not in block:
                return False
    return True


def _place_source_blocks(
    pipe: set[tuple[int, int]],
    source_colors: list[int],
) -> list[dict[str, object]] | None:
    deg = _degrees(pipe)
    endpoints = [cell for cell, d in deg.items() if d == 1]
    if len(endpoints) < len(source_colors):
        return None

    endpoint_info = []
    for endpoint in endpoints:
        pipe_neighbor = next(n for n in _neighbors4(endpoint[0], endpoint[1]) if n in pipe)
        outward = _opposite(_delta_to_dir(endpoint, pipe_neighbor))
        options = _block_options(endpoint, outward)
        endpoint_info.append((endpoint, outward, options))

    random.shuffle(endpoint_info)
    for subset in combinations(endpoint_info, len(source_colors)):
        subset = list(subset)
        random.shuffle(subset)
        shuffled_colors = list(source_colors)
        random.shuffle(shuffled_colors)
        placed: list[dict[str, object]] = []
        used_blocks: set[tuple[int, int]] = set()

        def backtrack(i: int) -> bool:
            if i == len(subset):
                return True
            endpoint, _outward, options = subset[i]
            candidates = options[:]
            random.shuffle(candidates)
            for block in candidates:
                if not _valid_block_option(block, endpoint, pipe, used_blocks):
                    continue
                used_blocks.update(block)
                placed.append({"color": shuffled_colors[i], "cells": set(block)})
                if backtrack(i + 1):
                    return True
                placed.pop()
                used_blocks.difference_update(block)
            return False

        if backtrack(0):
            return placed
    return None


def _generate_component(source_colors: list[int]) -> dict[str, object] | None:
    need = len(source_colors)
    assert 1 <= need <= 4
    for _ in range(180):
        backbone = _build_backbone()
        if backbone is None:
            continue
        pipe = set(backbone)

        branch_plan = 0
        if need == 4:
            branch_plan = 2
        elif need == 3:
            branch_plan = 1
        elif random.random() < 0.45:
            branch_plan = 1

        if branch_plan:
            internal = backbone[1:-1]
            random.shuffle(internal)
            if not internal:
                continue
            attach = None
            dirs: list[tuple[int, int]] = []
            if branch_plan == 2:
                for cell in internal:
                    available = _available_branch_dirs(cell, pipe)
                    if len(available) >= 2:
                        attach = cell
                        dirs = random.sample(available, 2)
                        break
                if attach is None:
                    continue
                ok = True
                for direction in dirs:
                    if not _extend_branch(pipe, attach, direction, random.randint(2, 4)):
                        ok = False
                        break
                if not ok:
                    continue
            else:
                found = False
                for cell in internal:
                    available = _available_branch_dirs(cell, pipe)
                    if not available:
                        continue
                    direction = random.choice(available)
                    trial_pipe = set(pipe)
                    if _extend_branch(trial_pipe, cell, direction, random.randint(2, 4)):
                        pipe = trial_pipe
                        found = True
                        break
                if not found:
                    continue

        blocks = _place_source_blocks(pipe, source_colors)
        if blocks is None:
            continue

        occupied = set(pipe)
        for block in blocks:
            occupied.update(block["cells"])
        r0, c0, r1, c1 = _bbox(occupied)
        shift_r, shift_c = -r0, -c0
        shifted_pipe = {(r + shift_r, c + shift_c) for r, c in pipe}
        shifted_blocks = [
            {
                "color": int(block["color"]),
                "cells": {(r + shift_r, c + shift_c) for r, c in block["cells"]},
            }
            for block in blocks
        ]

        return {
            "pipe": shifted_pipe,
            "sources": shifted_blocks,
            "height": int(r1 - r0 + 1),
            "width": int(c1 - c0 + 1),
            "mix": _mix_colors(source_colors),
        }
    return None


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int, offset: tuple[int, int]) -> None:
    orow, ocol = offset
    for r, c in cells:
        grid[orow + r, ocol + c] = int(color)


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _choose_primary_case() -> tuple[str, list[int]]:
    case = random.choice(["single", "special_pair", "other_pair", "tripleplus"])
    if case == "single":
        return case, [random.choice(PRIMARY_SOURCE_COLORS)]
    if case == "special_pair":
        return case, list(random.choice([(RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)]))
    if case == "other_pair":
        return case, list(random.choice([(RED, CYAN), (BLUE, CYAN), (YELLOW, CYAN)]))
    count = random.choice([3, 4])
    return case, random.sample(list(PRIMARY_SOURCE_COLORS), count)


def _make_spec() -> list[dict[str, object]] | None:
    _case, primary_colors = _choose_primary_case()
    primary = _generate_component(primary_colors)
    if primary is None:
        return None

    specs = [primary]
    remaining = [c for c in PRIMARY_SOURCE_COLORS if c not in primary_colors]
    if remaining and random.random() < 0.6:
        extra_count = 1 if len(remaining) == 1 else random.choice([1, 2])
        extra_colors = random.sample(remaining, extra_count)
        extra = _generate_component(extra_colors)
        if extra is not None:
            specs.append(extra)
    return specs


def _render_pair(specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(int(spec["height"]) for spec in specs)
    max_w = max(int(spec["width"]) for spec in specs)
    min_h = max(18, max_h + 2)
    min_w = max(18, max_w + 2)
    if len(specs) == 2:
        min_h = max(min_h, max_h + 2)
        min_w = max(min_w, max_w * 2 + 3)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(160):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        bboxes: list[tuple[int, int, int, int]] = []
        offsets: list[tuple[int, int]] = []
        ok = True
        for spec in specs:
            sh = int(spec["height"])
            sw = int(spec["width"])
            placed = False
            for _inner in range(120):
                r0 = random.randint(0, h - sh)
                c0 = random.randint(0, w - sw)
                bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if all(not _bbox_overlaps(bbox, other, margin=1) for other in bboxes):
                    bboxes.append(bbox)
                    offsets.append((r0, c0))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        for spec, offset in zip(specs, offsets, strict=False):
            _paint_cells(inp, spec["pipe"], PIPE, offset)
            _paint_cells(out, spec["pipe"], int(spec["mix"]), offset)
            for source in spec["sources"]:
                _paint_cells(inp, source["cells"], int(source["color"]), offset)
                _paint_cells(out, source["cells"], int(source["color"]), offset)

        if _dominant_color(inp) != BG:
            continue
        return inp, out
    return None


def _gen_attempt() -> np.ndarray | None:
    specs = _make_spec()
    if specs is None:
        return None
    rendered = _render_pair(specs)
    if rendered is None:
        return None
    inp, expected = rendered
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dye-pipe input")

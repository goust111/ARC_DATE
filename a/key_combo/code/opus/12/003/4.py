from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREEN = 3
MOVE_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

TIP_DIRS = {
    "up": ((-1, -1), (-1, 1)),
    "down": ((1, -1), (1, 1)),
    "left": ((-1, -1), (1, -1)),
    "right": ((-1, 1), (1, 1)),
}
ORIENTATION_BY_DIRS = {
    frozenset(dirs): orientation for orientation, dirs in TIP_DIRS.items()
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _green_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    green = grid == GREEN
    visited = np.zeros_like(green, dtype=bool)
    components: list[set[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not green[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and green[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _ray_boundary_cells(
    tip: tuple[int, int], orientation: str, height: int
) -> set[tuple[int, int]]:
    r, c = tip
    cells = {(r, c)}
    for dr, dc in TIP_DIRS[orientation]:
        for d in range(1, height + 1):
            cells.add((r + dr * d, c + dc * d))
    return cells


def _interior_cells(
    tip: tuple[int, int], orientation: str, height: int
) -> list[tuple[int, int]]:
    r, c = tip
    cells: list[tuple[int, int]] = []
    if orientation == "down":
        for d in range(1, height + 1):
            rr = r + d
            for cc in range(c - d + 1, c + d):
                cells.append((rr, cc))
    elif orientation == "up":
        for d in range(1, height + 1):
            rr = r - d
            for cc in range(c - d + 1, c + d):
                cells.append((rr, cc))
    elif orientation == "right":
        for d in range(1, height + 1):
            cc = c + d
            for rr in range(r - d + 1, r + d):
                cells.append((rr, cc))
    elif orientation == "left":
        for d in range(1, height + 1):
            cc = c - d
            for rr in range(r - d + 1, r + d):
                cells.append((rr, cc))
    else:
        raise AssertionError(f"unexpected orientation: {orientation}")
    return cells


def _progress(cell: tuple[int, int], orientation: str) -> int:
    r, c = cell
    if orientation == "down":
        return r
    if orientation == "up":
        return -r
    if orientation == "right":
        return c
    if orientation == "left":
        return -c
    raise AssertionError(f"unexpected orientation: {orientation}")


def _axis_key(cell: tuple[int, int], orientation: str) -> int:
    r, c = cell
    return c if orientation in ("up", "down") else r


def _axis_lines(
    tip: tuple[int, int], orientation: str, height: int
) -> dict[int, list[tuple[int, int]]]:
    lines: dict[int, list[tuple[int, int]]] = {}
    for cell in _interior_cells(tip, orientation, height):
        lines.setdefault(_axis_key(cell, orientation), []).append(cell)
    for key in lines:
        lines[key].sort(key=lambda cell: _progress(cell, orientation))
    return lines


def _detect_wedges(grid: np.ndarray) -> list[dict]:
    wedges: list[dict] = []
    for comp in _green_components(grid):
        tip_candidates = []
        for r, c in comp:
            dirs = []
            for dr, dc in DIAGONALS:
                if (r + dr, c + dc) in comp:
                    dirs.append((dr, dc))
            orientation = ORIENTATION_BY_DIRS.get(frozenset(dirs))
            if orientation is not None:
                tip_candidates.append(((r, c), orientation))

        assert len(tip_candidates) == 1, "each wedge must have exactly one tip"
        tip, orientation = tip_candidates[0]

        lengths = []
        for dr, dc in TIP_DIRS[orientation]:
            d = 1
            while (tip[0] + dr * d, tip[1] + dc * d) in comp:
                d += 1
            lengths.append(d - 1)
        assert lengths[0] == lengths[1] and lengths[0] >= 2
        height = lengths[0]

        boundary = _ray_boundary_cells(tip, orientation, height)
        assert comp == boundary, "green markers must be full diagonal rays"

        interior = _interior_cells(tip, orientation, height)
        wedges.append(
            {
                "tip": tip,
                "orientation": orientation,
                "height": height,
                "boundary": boundary,
                "interior": interior,
                "lines": _axis_lines(tip, orientation, height),
            }
        )

    assert wedges, "expected at least one wedge"

    occupied: set[tuple[int, int]] = set()
    for wedge in wedges:
        interior = set(wedge["interior"])
        assert occupied.isdisjoint(interior), "wedge interiors must not overlap"
        occupied |= interior
    return wedges


def _settle_wedge(
    source: np.ndarray, out: np.ndarray, wedge: dict
) -> None:
    interior = wedge["interior"]
    for r, c in interior:
        assert source[r, c] != GREEN
        out[r, c] = BLACK

    for ordered in wedge["lines"].values():
        colors = [
            int(source[r, c])
            for r, c in ordered
            if int(source[r, c]) not in (BLACK, GREEN)
        ]
        dests = ordered[len(ordered) - len(colors) :]
        for (r, c), color in zip(dests, colors):
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Settle each wedge's interior pixels toward the wedge opening while keeping the green boundaries fixed.

    Input structure:
    - The grid has a black background plus 2-3 green V-shaped diagonal wedge boundaries.
    - Each wedge is defined by one tip pixel and two full 45-degree green side rays of equal length.
    - Non-green foreground pixels lie only inside the triangular interior of those wedges.

    Rule:
    - Detect each green wedge, and read its opening direction from the tip: the side opposite the tip is the open face.
    - Use that opening direction as gravity. Pixels move only along that axis, staying on the same perpendicular slice inside the wedge.
    - For each slice, compact the slice's foreground pixels against the open face, preserving their order along the gravity axis.
    - Clear vacated interior cells to black and leave every green boundary pixel unchanged.

    Key insight:
    - The same diagonal green geometry both identifies which cells belong to a wedge and encodes the gravity direction that its interior pixels must follow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wedges = _detect_wedges(g)
    interior_union = {cell for wedge in wedges for cell in wedge["interior"]}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            value = int(g[r, c])
            if value not in (BLACK, GREEN):
                assert (r, c) in interior_union, "foreground pixels must lie inside wedges"

    out = g.copy()
    for wedge in wedges:
        _settle_wedge(g, out, wedge)
    return out


GRID_MIN = 12
GRID_MAX = 16
WEDGE_COUNT_RANGE = (2, 3)
HEIGHT_RANGE = (3, 5)
GEN_ATTEMPTS = 400


def _wedge_geometry(
    shape: tuple[int, int], tip: tuple[int, int], orientation: str, height: int
) -> dict | None:
    boundary = _ray_boundary_cells(tip, orientation, height)
    interior = _interior_cells(tip, orientation, height)
    cells = boundary | set(interior)
    if not all(_in_bounds(shape, r, c) for r, c in cells):
        return None
    return {
        "tip": tip,
        "orientation": orientation,
        "height": height,
        "boundary": boundary,
        "interior": interior,
        "lines": _axis_lines(tip, orientation, height),
        "footprint": cells,
    }


def _candidate_wedges(
    shape: tuple[int, int], orientation: str, height: int
) -> list[dict]:
    h, w = shape
    candidates = []
    for r in range(h):
        for c in range(w):
            wedge = _wedge_geometry(shape, (r, c), orientation, height)
            if wedge is not None:
                candidates.append(wedge)
    return candidates


def _well_separated(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r1, c1 in a:
        for r2, c2 in b:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _choose_wedge_specs() -> list[tuple[str, int]]:
    count = random.randint(*WEDGE_COUNT_RANGE)
    first = random.choice(tuple(TIP_DIRS))
    second = random.choice([d for d in TIP_DIRS if d != first])
    orientations = [first, second]
    while len(orientations) < count:
        orientations.append(random.choice(tuple(TIP_DIRS)))
    random.shuffle(orientations)
    return [(orientation, random.randint(*HEIGHT_RANGE)) for orientation in orientations]


def _place_wedges(shape: tuple[int, int], specs: list[tuple[str, int]]) -> list[dict] | None:
    caches = {
        spec: _candidate_wedges(shape, spec[0], spec[1])
        for spec in set(specs)
    }
    for _ in range(120):
        used: set[tuple[int, int]] = set()
        placed: list[dict] = []
        ok = True
        for spec in specs:
            candidates = [
                wedge
                for wedge in caches[spec]
                if wedge["footprint"].isdisjoint(used)
                and _well_separated(wedge["footprint"], used)
            ]
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            placed.append(chosen)
            used |= chosen["footprint"]
        if ok:
            return placed
    return None


def _line_combinations(
    capacity: int,
    count: int,
    *,
    unsettled: bool,
    require_gap: bool,
) -> list[tuple[int, ...]]:
    settled = tuple(range(capacity - count, capacity))
    combos = []
    for combo in itertools.combinations(range(capacity), count):
        if unsettled and combo == settled:
            continue
        if require_gap and count >= 2:
            if not any(combo[i + 1] - combo[i] > 1 for i in range(count - 1)):
                continue
        combos.append(combo)
    return combos


def _build_wedge_payload(
    wedge: dict, *, force_stack_gap: bool
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    lines = wedge["lines"]
    keys = list(lines)
    capacities = {key: len(lines[key]) for key in keys}
    mandatory = max(keys, key=lambda key: capacities[key])

    total = random.randint(2, 5)
    counts = {key: 0 for key in keys}
    if force_stack_gap:
        counts[mandatory] = 2
    else:
        counts[mandatory] = random.randint(1, min(2, capacities[mandatory] - 1))
    remaining = total - counts[mandatory]

    while remaining > 0:
        candidates = []
        for key in keys:
            limit = capacities[key]
            if key == mandatory:
                limit -= 1
            if counts[key] < limit:
                candidates.append(key)
        assert candidates, "wedge must have enough capacity for its payload"
        key = random.choice(candidates)
        counts[key] += 1
        remaining -= 1

    colors = random.sample(MOVE_COLORS, total)
    random.shuffle(colors)

    output_by_key: dict[int, list[int]] = {}
    input_by_key: dict[int, list[int]] = {}
    cursor = 0
    for key in random.sample([k for k in keys if counts[k] > 0], sum(v > 0 for v in counts.values())):
        count = counts[key]
        ordered_colors = colors[cursor : cursor + count]
        cursor += count
        output_by_key[key] = ordered_colors

        capacity = capacities[key]
        need_gap = force_stack_gap and key == mandatory
        need_unsettled = key == mandatory or random.random() < 0.65
        combos = _line_combinations(
            capacity,
            count,
            unsettled=need_unsettled,
            require_gap=need_gap,
        )
        if not combos and need_unsettled:
            combos = _line_combinations(
                capacity,
                count,
                unsettled=False,
                require_gap=need_gap,
            )
        if not combos and need_gap:
            combos = _line_combinations(
                capacity,
                count,
                unsettled=False,
                require_gap=False,
            )
        assert combos, "every occupied line must admit an input arrangement"
        input_by_key[key] = list(random.choice(combos))

    return output_by_key, input_by_key


def _paint_payload(
    grid: np.ndarray,
    wedge: dict,
    line_colors: dict[int, list[int]],
    line_indices: dict[int, list[int]] | None = None,
) -> None:
    for key, colors in line_colors.items():
        ordered = wedge["lines"][key]
        if line_indices is None:
            start = len(ordered) - len(colors)
            positions = list(range(start, len(ordered)))
        else:
            positions = line_indices[key]
        for idx, color in zip(positions, colors):
            r, c = ordered[idx]
            grid[r, c] = color


def _validate_example(inp: np.ndarray, planned_out: np.ndarray) -> None:
    wedges = _detect_wedges(inp)
    assert 2 <= len(wedges) <= 3
    assert len({w["orientation"] for w in wedges}) >= 2

    moved_pixels = 0
    saw_stack = False
    all_interior = {cell for wedge in wedges for cell in wedge["interior"]}
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            value = int(inp[r, c])
            if value not in (BLACK, GREEN):
                assert (r, c) in all_interior

    out = main(inp)
    assert np.array_equal(out, planned_out)
    assert np.array_equal(main(out), out)

    for wedge in wedges:
        payload = [
            int(inp[r, c])
            for r, c in wedge["interior"]
            if int(inp[r, c]) not in (BLACK, GREEN)
        ]
        assert 2 <= len(payload) <= 5
        assert len(payload) == len(set(payload))
        if any(int(inp[r, c]) != int(out[r, c]) for r, c in wedge["interior"]):
            moved_pixels += 1
        for ordered in wedge["lines"].values():
            in_count = sum(int(inp[r, c]) not in (BLACK, GREEN) for r, c in ordered)
            out_count = sum(int(out[r, c]) not in (BLACK, GREEN) for r, c in ordered)
            assert in_count == out_count
            if in_count >= 2:
                saw_stack = True
                out_tail = ordered[len(ordered) - out_count :]
                assert all(int(out[r, c]) not in (BLACK, GREEN) for r, c in out_tail)
    assert moved_pixels == len(wedges), "every wedge should demonstrate motion"
    assert saw_stack, "at least one slice should require stacking"


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    specs = _choose_wedge_specs()
    wedges = _place_wedges(shape, specs)
    if wedges is None:
        return None

    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)
    for wedge in wedges:
        for r, c in wedge["boundary"]:
            inp[r, c] = GREEN
            out[r, c] = GREEN

    special = random.randrange(len(wedges))
    for i, wedge in enumerate(wedges):
        output_by_key, input_by_key = _build_wedge_payload(
            wedge, force_stack_gap=(i == special)
        )
        _paint_payload(inp, wedge, output_by_key, input_by_key)
        _paint_payload(out, wedge, output_by_key)

    _validate_example(inp, out)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wedge-settling example")

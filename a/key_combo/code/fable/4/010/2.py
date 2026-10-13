from __future__ import annotations

import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background must be uniquely most common"
    return int(winners[0])


def _legend(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], dict[int, int]]:
    """Read the contiguous two-column legend anchored at (0, 0)."""
    assert grid.shape[1] >= 2, "expected two legend columns"
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        source, target = int(grid[r, 0]), int(grid[r, 1])
        if source == bg or target == bg:
            break
        pairs.append((source, target))

    assert len(pairs) >= 2, "expected a nontrivial color-cycle legend"
    sources = [source for source, _target in pairs]
    targets = [target for _source, target in pairs]
    assert len(set(sources)) == len(sources), "legend sources must be unique"
    assert set(targets) == set(sources), "legend targets must be exactly the source colors"

    mapping = dict(pairs)
    start = sources[0]
    visited: set[int] = set()
    color = start
    for _ in sources:
        assert color not in visited, "legend must form one cycle"
        visited.add(color)
        color = mapping[color]
    assert color == start and visited == set(sources), "legend must form one closed cycle"
    return pairs, mapping


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _advance(color: int, steps: int, mapping: dict[int, int]) -> int:
    for _ in range(int(steps)):
        color = mapping[color]
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """Advance every object's color through the legend cycle by its area.

    Input structure:
    - A dominant background contains a contiguous two-column legend at the upper left.
    - Each legend row maps its left color to the next color on the right, and all
      rows together form one closed cycle.
    - Outside the legend are separated monochrome 4-connected objects whose colors
      are legend source colors.

    Rule:
    - Clear the legend. For each monochrome 4-connected object, count its pixels k,
      follow the displayed color mapping exactly k times from its original color,
      and recolor the entire object with the resulting color.
    - Preserve the canvas size and every object's shape and position.

    Key insight:
    - The legend is a one-step evolution operator rather than a direct final-color
      lookup: object area determines how many successive mapping steps to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    bg = _background(g)
    pairs, mapping = _legend(g, bg)

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[: len(pairs), :2] = True
    outside_foreground = (g != bg) & ~legend_mask
    assert outside_foreground.any(), "expected at least one object outside the legend"
    assert set(map(int, g[outside_foreground])).issubset(mapping), (
        "objects must use legend source colors"
    )

    objects = g.copy()
    objects[legend_mask] = bg
    out = objects.copy()
    object_count = 0
    for source in mapping:
        # Detect every object on the unchanged, legend-free input. Recolored
        # objects must not be mistaken for objects of a later cycle color.
        for coords in _components4(objects == source):
            object_count += 1
            target = _advance(source, len(coords), mapping)
            out[coords[:, 0], coords[:, 1]] = target
    assert object_count > 0
    return out


# ------------------------------- generator ---------------------------------

_PALETTE = tuple(range(10))
_CYCLE_LENGTHS = (3, 4, 5)
_MAX_GENERATION_ATTEMPTS = 300


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch.copy()


def _random_polyomino(area: int) -> np.ndarray:
    """Grow a compact random 4-connected shape and crop it to its bbox."""
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _DIR4:
                    p = (r + dr, c + dc)
                    if p not in cells:
                        frontier.add(p)
            options = list(frontier)
            weights = [
                1 + 2 * sum((r + dr, c + dc) in cells for dr, dc in _DIR4)
                for r, c in options
            ]
            cells.add(random.choices(options, weights=weights, k=1)[0])

        r0 = min(r for r, _c in cells)
        c0 = min(c for _r, c in cells)
        shifted = {(r - r0, c - c0) for r, c in cells}
        height = max(r for r, _c in shifted) + 1
        width = max(c for _r, c in shifted) + 1
        if height <= 5 and width <= 5:
            patch = np.zeros((height, width), dtype=bool)
            for r, c in shifted:
                patch[r, c] = True
            return _transform_patch(patch)
    raise RuntimeError("failed to grow a compact polyomino")


def _object_specs(cycle: list[int]) -> list[dict[str, object]]:
    """Force semantic counterexamples, then add varied objects."""
    specs: list[dict[str, object]] = [
        {"source": cycle[0], "patch": np.ones((1, 1), dtype=bool), "role": "one_step_bait"},
        {"source": cycle[0], "patch": _transform_patch(np.ones((1, 2), dtype=bool)), "role": "two_steps"},
        {
            "source": cycle[1],
            "patch": _transform_patch(np.array([[1, 1], [1, 0]], dtype=bool)),
            "role": "bbox_counterexample",
        },
    ]
    for source in cycle[2:]:
        area = random.randint(4, 8)
        specs.append({"source": source, "patch": _random_polyomino(area), "role": "coverage"})
    for _ in range(random.randint(1, 3)):
        area = random.randint(3, 9)
        specs.append(
            {"source": random.choice(cycle), "patch": _random_polyomino(area), "role": "extra"}
        )
    random.shuffle(specs)
    return specs


def _blocked_cells(grid: np.ndarray, bg: int, forbidden: np.ndarray) -> np.ndarray:
    """Reserve an 8-neighbor moat around every already placed foreground cell."""
    blocked = forbidden.copy()
    height, width = grid.shape
    for r, c in np.argwhere(grid != bg):
        r, c = int(r), int(c)
        blocked[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] = True
    return blocked


def _place_patch(
    grid: np.ndarray, patch: np.ndarray, color: int, bg: int, forbidden: np.ndarray
) -> np.ndarray | None:
    height, width = grid.shape
    ph, pw = patch.shape
    blocked = _blocked_cells(grid, bg, forbidden)
    local = np.argwhere(patch)
    candidates: list[tuple[int, int]] = []
    for r0 in range(height - ph + 1):
        for c0 in range(width - pw + 1):
            rr = local[:, 0] + r0
            cc = local[:, 1] + c0
            if not blocked[rr, cc].any():
                candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    coords = local + np.array([r0, c0], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _expected_from_records(
    grid: np.ndarray,
    legend_rows: int,
    mapping: dict[int, int],
    records: list[dict[str, object]],
    bg: int,
    *, use_bbox_area: bool = False,
    fixed_steps: int | None = None,
) -> np.ndarray:
    """Independent construction oracle used to reject semantic shortcuts."""
    out = grid.copy()
    out[:legend_rows, :2] = bg
    for record in records:
        coords = np.asarray(record["coords"], dtype=int)
        source = int(record["source"])
        if fixed_steps is not None:
            steps = int(fixed_steps)
        elif use_bbox_area:
            steps = int(np.ptp(coords[:, 0]) + 1) * int(np.ptp(coords[:, 1]) + 1)
        else:
            steps = len(coords)
        target = _advance(source, steps, mapping)
        out[coords[:, 0], coords[:, 1]] = target
    return out


def _gen_attempt() -> np.ndarray | None:
    cycle_length = random.choice(_CYCLE_LENGTHS)
    bg = random.choice(_PALETTE)
    cycle = random.sample([color for color in _PALETTE if color != bg], cycle_length)
    mapping = {cycle[i]: cycle[(i + 1) % cycle_length] for i in range(cycle_length)}

    specs = _object_specs(cycle)
    height = random.randint(16, 25)
    width = random.randint(16, 25)
    grid = np.full((height, width), bg, dtype=int)

    legend_order = cycle.copy()
    random.shuffle(legend_order)
    for r, source in enumerate(legend_order):
        grid[r, 0] = source
        grid[r, 1] = mapping[source]

    forbidden = np.zeros(grid.shape, dtype=bool)
    forbidden[: cycle_length + 2, :4] = True
    records: list[dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: int(np.sum(item["patch"])), reverse=True):
        coords = _place_patch(grid, np.asarray(spec["patch"], dtype=bool), int(spec["source"]), bg, forbidden)
        if coords is None:
            return None
        records.append({**spec, "coords": coords})

    values, counts = np.unique(grid, return_counts=True)
    if int(values[np.argmax(counts)]) != bg or np.count_nonzero(counts == counts.max()) != 1:
        return None

    expected = _expected_from_records(grid, cycle_length, mapping, records, bg)
    one_step = _expected_from_records(grid, cycle_length, mapping, records, bg, fixed_steps=1)
    bbox_count = _expected_from_records(grid, cycle_length, mapping, records, bg, use_bbox_area=True)
    if np.array_equal(expected, one_step) or np.array_equal(expected, bbox_count):
        return None

    forced = {str(record["role"]): record for record in records if record["role"] != "extra"}
    bait = forced.get("one_step_bait")
    two_steps = forced.get("two_steps")
    bbox_case = forced.get("bbox_counterexample")
    if bait is None or two_steps is None or bbox_case is None:
        return None
    if int(bait["source"]) != int(two_steps["source"]):
        return None
    if _advance(int(bait["source"]), 1, mapping) == _advance(int(two_steps["source"]), 2, mapping):
        return None
    bbox_coords = np.asarray(bbox_case["coords"], dtype=int)
    bbox_area = int(np.ptp(bbox_coords[:, 0]) + 1) * int(np.ptp(bbox_coords[:, 1]) + 1)
    if bbox_area == len(bbox_coords) or (bbox_area - len(bbox_coords)) % cycle_length == 0:
        return None
    if {int(record["source"]) for record in records} != set(cycle):
        return None

    try:
        solved = main(grid)
    except (AssertionError, KeyError):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-iteration input")

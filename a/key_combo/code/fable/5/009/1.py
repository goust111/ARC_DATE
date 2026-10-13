from __future__ import annotations

from collections import Counter

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color, following the ARC background convention."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    """Return 4-connected True components as lists of (row, column) coordinates."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []

    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        component: list[Coord] = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        components.append(component)
    return components


def _launch_direction(launcher: Coord, shape: tuple[int, int]) -> Coord:
    """Decode the unique inward direction of a non-corner boundary launcher."""
    r, c = launcher
    height, width = shape
    directions: list[Coord] = []
    if r == 0:
        directions.append((1, 0))
    if r == height - 1:
        directions.append((-1, 0))
    if c == 0:
        directions.append((0, 1))
    if c == width - 1:
        directions.append((0, -1))
    assert len(directions) == 1, "each launcher must lie on exactly one edge, not at a corner"
    return directions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Translate two shape-separated marker counts into periodic inward rays.

    Input structure:
    - The dominant color is background. Every other color has one non-corner
      launcher pixel on an edge and scattered interior markers of that color.
    - After excluding the launcher, every 4-connected marker component is either
      an isolated singleton or a straight two-cell domino; marker colors do not touch.

    Rule:
    - For each launcher color, let p be the number of singleton components and k
      the number of domino components, then erase all of those scattered markers.
    - Keep the launcher and paint its color inward at distances p, 2p, ..., k*p;
      every other cell becomes background.

    Key insight:
    - Marker shape supplies two independent count channels: singletons encode the
      ray period, dominoes encode its number of repetitions, while the edge fixes direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty two-dimensional grid"
    height, width = g.shape
    bg = _background(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert colors, "expected at least one launcher color"

    decoded: list[tuple[int, Coord, Coord, int, int]] = []
    for color in colors:
        color_mask = g == color
        edge_mask = np.zeros_like(color_mask)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        boundary_cells = np.argwhere(color_mask & edge_mask)
        assert boundary_cells.shape[0] == 1, "each color must have exactly one boundary launcher"
        launcher = (int(boundary_cells[0, 0]), int(boundary_cells[0, 1]))
        direction = _launch_direction(launcher, g.shape)

        marker_mask = color_mask.copy()
        marker_mask[launcher] = False
        components = _components4(marker_mask)
        p = 0
        k = 0
        for component in components:
            if len(component) == 1:
                p += 1
            elif len(component) == 2:
                (r0, c0), (r1, c1) = component
                assert abs(r0 - r1) + abs(c0 - c1) == 1, "two-cell marks must be straight dominoes"
                k += 1
            else:
                raise AssertionError("marker components must be singletons or dominoes")
        assert p >= 1 and k >= 1, "both marker-shape count channels must be present"
        decoded.append((color, launcher, direction, p, k))

    out = np.full_like(g, bg)
    for color, launcher, _direction, _p, _k in decoded:
        assert int(out[launcher]) == bg, "launchers must occupy distinct cells"
        out[launcher] = color

    for color, (r, c), (dr, dc), p, k in decoded:
        for repetition in range(1, k + 1):
            rr = r + dr * p * repetition
            cc = c + dc * p * repetition
            assert 0 <= rr < height and 0 <= cc < width, "encoded ray must fit inside the grid"
            assert int(out[rr, cc]) == bg, "colored ray beats must not collide"
            out[rr, cc] = color
    return out


def _side_geometry(side: str, index: int, height: int, width: int) -> tuple[Coord, Coord]:
    if side == "top":
        return (0, index), (1, 0)
    if side == "bottom":
        return (height - 1, index), (-1, 0)
    if side == "left":
        return (index, 0), (0, 1)
    if side == "right":
        return (index, width - 1), (0, -1)
    raise ValueError(f"unknown side: {side}")


def _ray_cells(launcher: Coord, direction: Coord, shape: tuple[int, int]) -> set[Coord]:
    height, width = shape
    r, c = launcher
    dr, dc = direction
    cells: set[Coord] = set()
    r += dr
    c += dc
    while 0 <= r < height and 0 <= c < width:
        cells.add((r, c))
        r += dr
        c += dc
    return cells


def _render_expected(shape: tuple[int, int], bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    """Independent generator oracle using the sampled semantic parameters directly."""
    out = np.full(shape, bg, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        r, c = spec["launcher"]  # type: ignore[misc]
        out[int(r), int(c)] = color
    for spec in specs:
        color = int(spec["color"])
        r, c = spec["launcher"]  # type: ignore[misc]
        dr, dc = spec["direction"]  # type: ignore[misc]
        p, k = int(spec["p"]), int(spec["k"])
        for beat in range(1, k + 1):
            rr, cc = int(r) + int(dr) * p * beat, int(c) + int(dc) * p * beat
            assert int(out[rr, cc]) == bg
            out[rr, cc] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    # The four pairs form a visible 2x2 channel cross.  The first two also have
    # equal marker-pixel totals despite different (p, k), defeating total-count shortcuts.
    small_p = int(rng.choice((2, 3)))
    large_p = small_p + 2
    small_k = int(rng.choice((2, 3, 4) if small_p == 2 else (2, 3)))
    large_k = small_k + 1
    pairs = [
        (small_p, large_k),
        (large_p, small_k),
        (small_p, small_k),
        (large_p, large_k),
    ]
    color_count = int(rng.integers(3, 5))
    pairs = pairs[:color_count]
    rng.shuffle(pairs)

    max_distance = max(p * k for p, k in pairs)
    min_side = max(20, max_distance + 3)
    height = int(rng.integers(min_side, 31))
    width = int(rng.integers(min_side, 31))

    palette = list(map(int, rng.choice(np.arange(10), size=color_count + 1, replace=False)))
    bg = palette[0]
    colors = palette[1:]

    # Every generated grid includes both horizontal and vertical firing directions;
    # the particular edge on each axis remains random.
    sides = [str(rng.choice(("top", "bottom"))), str(rng.choice(("left", "right")))]
    sides.extend(str(rng.choice(("top", "bottom", "left", "right"))) for _ in range(color_count - 2))
    rng.shuffle(sides)

    specs: list[dict[str, object]] | None = None
    for _ in range(300):
        candidate_specs: list[dict[str, object]] = []
        for color, (p, k), side in zip(colors, pairs, sides):
            if side in {"top", "bottom"}:
                index = int(rng.integers(2, width - 2))
            else:
                index = int(rng.integers(2, height - 2))
            launcher, direction = _side_geometry(side, index, height, width)
            candidate_specs.append(
                {"color": color, "p": p, "k": k, "launcher": launcher, "direction": direction}
            )

        launchers = [spec["launcher"] for spec in candidate_specs]
        if any(
            max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1
            for i, a in enumerate(launchers)
            for b in launchers[i + 1 :]
        ):
            continue

        rays = [
            _ray_cells(spec["launcher"], spec["direction"], (height, width))  # type: ignore[arg-type]
            for spec in candidate_specs
        ]
        if any(
            other_launcher in ray
            for ray_index, ray in enumerate(rays)
            for other_index, other_launcher in enumerate(launchers)
            if ray_index != other_index
        ):
            continue

        targets: list[Coord] = []
        for spec in candidate_specs:
            r, c = spec["launcher"]  # type: ignore[misc]
            dr, dc = spec["direction"]  # type: ignore[misc]
            p, k = int(spec["p"]), int(spec["k"])
            targets.extend(
                (int(r) + int(dr) * p * beat, int(c) + int(dc) * p * beat)
                for beat in range(1, k + 1)
            )
        if len(targets) != len(set(targets)) or any(target in launchers for target in targets):
            continue
        specs = candidate_specs
        break
    if specs is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    occupied: set[Coord] = set()
    forbidden: set[Coord] = set()
    for spec in specs:
        launcher = spec["launcher"]  # type: ignore[assignment]
        color = int(spec["color"])
        grid[launcher] = color
        occupied.add(launcher)
        forbidden |= _ray_cells(launcher, spec["direction"], grid.shape)  # type: ignore[arg-type]

    jobs: list[tuple[int, int, str | None]] = []
    domino_job_indices: list[int] = []
    for spec_index, spec in enumerate(specs):
        jobs.extend((spec_index, 1, None) for _ in range(int(spec["p"])))
        for _ in range(int(spec["k"])):
            domino_job_indices.append(len(jobs))
            jobs.append((spec_index, 2, str(rng.choice(("horizontal", "vertical")))))
    # Both domino orientations occur in every sample, so neither branch is hidden.
    if len(domino_job_indices) >= 2:
        i0, i1 = domino_job_indices[:2]
        jobs[i0] = (jobs[i0][0], 2, "horizontal")
        jobs[i1] = (jobs[i1][0], 2, "vertical")
    rng.shuffle(jobs)

    objects: list[list[list[Coord]]] = [[] for _ in specs]
    for spec_index, area, orientation in jobs:
        placed: list[Coord] | None = None
        for _ in range(1500):
            r = int(rng.integers(1, height - 1))
            c = int(rng.integers(1, width - 1))
            cells = [(r, c)]
            if area == 2:
                dr, dc = ((0, 1) if orientation == "horizontal" else (1, 0))
                cells.append((r + dr, c + dc))
            if any(not (1 <= rr < height - 1 and 1 <= cc < width - 1) for rr, cc in cells):
                continue
            if any(cell in forbidden for cell in cells):
                continue
            if any(
                max(abs(rr - orr), abs(cc - occ)) <= 1
                for rr, cc in cells
                for orr, occ in occupied
            ):
                continue
            placed = cells
            break
        if placed is None:
            return None
        color = int(specs[spec_index]["color"])
        for cell in placed:
            grid[cell] = color
            occupied.add(cell)
        objects[spec_index].append(placed)

    # Semantic validation against sampled objects, rather than only a solver round trip.
    if _background(grid) != bg:
        return None
    for spec, color_objects in zip(specs, objects):
        singles = [obj for obj in color_objects if len(obj) == 1]
        dominoes = [obj for obj in color_objects if len(obj) == 2]
        if len(singles) != int(spec["p"]) or len(dominoes) != int(spec["k"]):
            return None
        if any(cell in forbidden for obj in color_objects for cell in obj):
            return None
    sampled_pairs = [(int(spec["p"]), int(spec["k"])) for spec in specs]
    if not any(p0 == p1 and k0 != k1 for p0, k0 in sampled_pairs for p1, k1 in sampled_pairs):
        return None
    if not any(k0 == k1 and p0 != p1 for p0, k0 in sampled_pairs for p1, k1 in sampled_pairs):
        return None
    if not any(
        p0 + 2 * k0 == p1 + 2 * k1 and (p0, k0) != (p1, k1)
        for p0, k0 in sampled_pairs
        for p1, k1 in sampled_pairs
    ):
        return None

    expected = _render_expected(grid.shape, bg, specs)
    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
        return None

    # Counterfactual checks show that changing either channel necessarily changes
    # the prescribed output, independently of marker placement.
    probe = specs[0]
    p, k = int(probe["p"]), int(probe["k"])
    beat_offsets = {p * beat for beat in range(1, k + 1)}
    changed_period_offsets = {(p - 1) * beat for beat in range(1, k + 1)}
    changed_repetition_offsets = {p * beat for beat in range(1, k)}
    if beat_offsets == changed_period_offsets or beat_offsets == changed_repetition_offsets:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate valid instance of the puzzle."""
    rng = np.random.default_rng()
    for _ in range(200):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid dual-channel counting-ray input")

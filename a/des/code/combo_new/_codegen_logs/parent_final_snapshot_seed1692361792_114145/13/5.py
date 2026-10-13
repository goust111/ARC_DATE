import random
from collections import deque

import numpy as np


BG = 0
EDGES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400

BOUNDARY_BASE = np.array(
    [
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ],
    dtype=int,
)


def _edge_center(shape: tuple[int, int], edge: str) -> tuple[int, int]:
    h, w = shape
    if edge == "top":
        return 0, w // 2
    if edge == "bottom":
        return h - 1, w // 2
    if edge == "left":
        return h // 2, 0
    if edge == "right":
        return h // 2, w - 1
    raise ValueError(edge)


def _center_distance_sq(shape: tuple[int, int], r: int, c: int) -> int:
    h, w = shape
    return (2 * r - (h - 1)) ** 2 + (2 * c - (w - 1)) ** 2


def _edge_of_seed(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    edges = []
    if r == 0:
        edges.append("top")
    if r == h - 1:
        edges.append("bottom")
    if c == 0:
        edges.append("left")
    if c == w - 1:
        edges.append("right")
    assert len(edges) == 1, "seed pixels must lie on exactly one non-corner edge"
    return edges[0]


def _components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            touches = bool(np.any(arr[:, 0] == 0) or np.any(arr[:, 0] == h - 1) or np.any(arr[:, 1] == 0) or np.any(arr[:, 1] == w - 1))
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "touches_border": touches,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _component_to_template(comp: dict[str, object]) -> dict[str, object]:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    return {
        "color": int(comp["color"]),
        "rel_cells": tuple(rel),
        "shape": (h, w),
        "center_rel": (h // 2, w // 2),
        "top_left": (int(r0), int(c0)),
        "abs_cells": {tuple(map(int, coord)) for coord in coords.tolist()},
    }


def _scene_parts(grid: np.ndarray) -> tuple[list[dict[str, int]], dict[int, dict[str, object]], set[tuple[int, int]]]:
    templates: dict[int, dict[str, object]] = {}
    seeds: list[dict[str, int]] = []
    obstacle: set[tuple[int, int]] = set()
    for comp in _components(grid):
        if bool(comp["touches_border"]):
            assert int(comp["area"]) == 1, "edge markers must be singleton seed pixels"
            r, c = map(int, np.asarray(comp["coords"], dtype=int)[0])
            seeds.append({"row": r, "col": c, "color": int(comp["color"])})
            continue
        template = _component_to_template(comp)
        color = int(template["color"])
        assert color not in templates, "each color must map to exactly one central template"
        templates[color] = template
        obstacle |= set(template["abs_cells"])
    assert seeds, "expected at least one edge seed"
    assert templates, "expected at least one interior template"
    for seed in seeds:
        assert int(seed["color"]) in templates, "every seed color must match a central template"
    return seeds, templates, obstacle


def _stamp_positions(
    template: dict[str, object],
    seed: tuple[int, int],
    grid_shape: tuple[int, int],
    obstacle: set[tuple[int, int]],
) -> tuple[list[list[tuple[int, int]]], str, bool]:
    h, w = grid_shape
    sr, sc = map(int, seed)
    edge = _edge_of_seed(grid_shape, sr, sc)
    th, tw = template["shape"]
    cr, cc = template["center_rel"]

    if edge == "top":
        start_r, start_c = 0, sc - cc
        step_r, step_c = th, 0
        opposite_touch = lambda cell: cell[0] == h - 1
    elif edge == "bottom":
        start_r, start_c = h - th, sc - cc
        step_r, step_c = -th, 0
        opposite_touch = lambda cell: cell[0] == 0
    elif edge == "left":
        start_r, start_c = sr - cr, 0
        step_r, step_c = 0, tw
        opposite_touch = lambda cell: cell[1] == w - 1
    else:
        start_r, start_c = sr - cr, w - tw
        step_r, step_c = 0, -tw
        opposite_touch = lambda cell: cell[1] == 0

    stamps: list[list[tuple[int, int]]] = []
    cur_r, cur_c = start_r, start_c
    while True:
        cur_cells: list[tuple[int, int]] = []
        out_of_bounds = False
        for rr, cc0 in template["rel_cells"]:
            ar = cur_r + int(rr)
            ac = cur_c + int(cc0)
            if not (0 <= ar < h and 0 <= ac < w):
                out_of_bounds = True
                break
            cur_cells.append((ar, ac))
        if out_of_bounds:
            return stamps, "boundary", bool(stamps and any(opposite_touch(cell) for cell in stamps[-1]))
        if any(cell in obstacle for cell in cur_cells):
            return stamps, "template", bool(stamps and any(opposite_touch(cell) for cell in stamps[-1]))
        stamps.append(cur_cells)
        cur_r += step_r
        cur_c += step_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each central monochrome template inward from matching edge seeds, with the seed edge selecting the growth direction.

    Input structure:
    - The background is black.
    - Single colored seed pixels sit on the outer grid edges.
    - The interior contains one connected monochrome template per active color, and each seed color matches exactly one template color.

    Rule:
    - For every seed pixel, find the central template with the same color.
    - The seed edge sets the stamping direction: top grows downward, bottom upward, left rightward, and right leftward.
    - Stamp translated copies of that template starting from the seed-aligned edge position, then continue one template-height or template-width farther each step.
    - Stop before a stamp would go out of bounds or overlap any of the original central templates.
    - If stamp regions overlap, apply seeds from farthest to nearest relative to the grid center so later, nearer seeds overwrite earlier paint.
    - Keep the original central templates unchanged.

    Key insight:
    - A template has two roles at once: it supplies the repeated shape for same-color edge seeds, and the untouched interior templates also act as stopping obstacles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    seeds, templates, obstacle = _scene_parts(g)

    out = g.copy()
    ordered = sorted(
        seeds,
        key=lambda seed: (
            -_center_distance_sq(g.shape, int(seed["row"]), int(seed["col"])),
            int(seed["row"]),
            int(seed["col"]),
        ),
    )
    for seed in ordered:
        color = int(seed["color"])
        template = templates[color]
        stamps, _reason, _touch = _stamp_positions(template, (int(seed["row"]), int(seed["col"])), g.shape, obstacle)
        for cells in stamps:
            for r, c in cells:
                out[r, c] = color
    return out


def _random_path(a: tuple[int, int], b: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = rng.choice(moves)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, k).copy()


def _boundary_shape(edge: str) -> tuple[tuple[int, int], ...]:
    if edge in {"top", "left"}:
        arr = BOUNDARY_BASE
    elif edge == "bottom":
        arr = _rotate_mask(BOUNDARY_BASE, 1)
    else:
        arr = _rotate_mask(BOUNDARY_BASE, 3)
    coords = []
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if int(arr[r, c]) == 1:
                coords.append((r, c))
    return tuple(coords)


def _random_odd_shape(size: int, required_edges: set[str], rng: random.Random) -> tuple[tuple[int, int], ...]:
    assert size in {3, 5}
    center = (size // 2, size // 2)
    target_area_low = 5 if size == 3 else 7
    target_area_high = 8 if size == 3 else 13

    for _ in range(200):
        anchors = {center}
        for edge in required_edges:
            anchors.add(_edge_center((size, size), edge))
        if not any(r == 0 for r, _ in anchors):
            anchors.add((0, rng.randrange(size)))
        if not any(r == size - 1 for r, _ in anchors):
            anchors.add((size - 1, rng.randrange(size)))
        if not any(c == 0 for _, c in anchors):
            anchors.add((rng.randrange(size), 0))
        if not any(c == size - 1 for _, c in anchors):
            anchors.add((rng.randrange(size), size - 1))

        cells = {center}
        for anchor in rng.sample(list(anchors), len(anchors)):
            cells.update(_random_path(center, anchor, rng))

        target_area = rng.randint(max(len(cells), target_area_low), max(len(cells), target_area_high))
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows != set(range(size)) or cols != set(range(size)):
            continue
        if not all(_edge_center((size, size), edge) in cells for edge in required_edges):
            continue
        return tuple(sorted(cells))
    raise RuntimeError("failed to sample an odd template shape")


def _assign_edges(n_templates: int, boundary_edge: str, rng: random.Random) -> list[set[str]]:
    assert n_templates in {3, 4}
    assignments = [set() for _ in range(n_templates)]
    assignments[0].add(boundary_edge)

    remaining = [edge for edge in EDGES if edge != boundary_edge]
    if n_templates == 4:
        rng.shuffle(remaining)
        for idx, edge in enumerate(remaining, start=1):
            assignments[idx].add(edge)
        if rng.random() < 0.45:
            extra_idx = rng.choice([1, 2, 3])
            current = assignments[extra_idx]
            options = [edge for edge in remaining if edge not in current]
            if "top" in current and "bottom" in options:
                options.remove("bottom")
            if "bottom" in current and "top" in options:
                options.remove("top")
            if "left" in current and "right" in options:
                options.remove("right")
            if "right" in current and "left" in options:
                options.remove("left")
            if options:
                assignments[extra_idx].add(rng.choice(options))
        return assignments

    pair_options = []
    for i, edge_a in enumerate(remaining):
        for edge_b in remaining[i + 1 :]:
            if {edge_a, edge_b} in ({"top", "bottom"}, {"left", "right"}):
                continue
            pair_options.append((edge_a, edge_b))
    edge_pair = rng.choice(pair_options)
    lone = next(edge for edge in remaining if edge not in edge_pair)
    if rng.random() < 0.5:
        assignments[1].update(edge_pair)
        assignments[2].add(lone)
    else:
        assignments[2].update(edge_pair)
        assignments[1].add(lone)
    return assignments


def _make_specs(assignments: list[set[str]], boundary_edge: str, rng: random.Random) -> list[dict[str, object]]:
    colors = rng.sample(list(range(1, 10)), len(assignments))
    specs: list[dict[str, object]] = []
    single_generic = None
    for idx in range(1, len(assignments)):
        if len(assignments[idx]) == 1:
            single_generic = idx
            break

    for idx, edges in enumerate(assignments):
        if idx == 0:
            rel_cells = _boundary_shape(boundary_edge)
            specs.append(
                {
                    "color": colors[idx],
                    "edges": set(edges),
                    "shape": (4, 4),
                    "center_rel": (2, 2),
                    "rel_cells": rel_cells,
                }
            )
            continue

        if idx == single_generic:
            size = 3
        elif len(edges) >= 2:
            size = 5
        else:
            size = 5 if rng.random() < 0.55 else 3
        rel_cells = _random_odd_shape(size, edges, rng)
        specs.append(
            {
                "color": colors[idx],
                "edges": set(edges),
                "shape": (size, size),
                "center_rel": (size // 2, size // 2),
                "rel_cells": rel_cells,
            }
        )
    return specs


def _valid_centers(dim: int, size: int) -> list[int]:
    center_rel = size // 2
    low = max(center_rel, dim // 3)
    high = min(dim - (size - center_rel), dim - dim // 3 - 1)
    return list(range(low, high + 1))


def _bbox_from_center(shape: tuple[int, int], center: tuple[int, int], center_rel: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    cr, cc = center
    rr, rc = center_rel
    r0 = cr - rr
    c0 = cc - rc
    return r0, c0, r0 + h - 1, c0 + w - 1


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _place_specs(specs: list[dict[str, object]], grid_shape: tuple[int, int], boundary_edge: str, rng: random.Random) -> list[dict[str, object]] | None:
    h, w = grid_shape
    for _ in range(300):
        placed: list[dict[str, object]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        order = [0] + rng.sample(list(range(1, len(specs))), len(specs) - 1)
        ok = True
        for idx in order:
            spec = specs[idx]
            th, tw = spec["shape"]
            rows = [r for r in _valid_centers(h, th) if r not in used_rows]
            cols = [c for c in _valid_centers(w, tw) if c not in used_cols]
            if idx == 0 and boundary_edge in {"top", "bottom"}:
                rows = [r for r in rows if (r - spec["center_rel"][0]) % th == 2]
            if idx == 0 and boundary_edge in {"left", "right"}:
                cols = [c for c in cols if (c - spec["center_rel"][1]) % tw == 2]
            rng.shuffle(rows)
            rng.shuffle(cols)
            found = None
            for cr in rows:
                for cc in cols:
                    bbox = _bbox_from_center(spec["shape"], (cr, cc), spec["center_rel"])
                    if any(_boxes_conflict(bbox, prev["bbox"]) for prev in placed):
                        continue
                    found = (cr, cc, bbox)
                    break
                if found is not None:
                    break
            if found is None:
                ok = False
                break

            cr, cc, bbox = found
            abs_cells = {(bbox[0] + rr, bbox[1] + cc0) for rr, cc0 in spec["rel_cells"]}
            placed.append(
                {
                    **spec,
                    "center": (cr, cc),
                    "bbox": bbox,
                    "abs_cells": abs_cells,
                    "top_left": (bbox[0], bbox[1]),
                }
            )
            used_rows.add(cr)
            used_cols.add(cc)
        if ok and len(placed) == len(specs):
            return sorted(placed, key=lambda spec: int(spec["color"]))
    return None


def _seed_from_template(template: dict[str, object], edge: str, grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    cr, cc = template["center"]
    if edge == "top":
        return 0, int(cc)
    if edge == "bottom":
        return h - 1, int(cc)
    if edge == "left":
        return int(cr), 0
    return int(cr), w - 1


def _build_grid(specs: list[dict[str, object]], grid_shape: tuple[int, int]) -> tuple[np.ndarray, list[dict[str, int]], set[tuple[int, int]]]:
    grid = np.zeros(grid_shape, dtype=int)
    obstacle: set[tuple[int, int]] = set()
    for spec in specs:
        for r, c in spec["abs_cells"]:
            assert int(grid[r, c]) == BG
            grid[r, c] = int(spec["color"])
            obstacle.add((int(r), int(c)))

    seeds: list[dict[str, int]] = []
    for spec in specs:
        for edge in sorted(spec["edges"]):
            r, c = _seed_from_template(spec, edge, grid_shape)
            assert int(grid[r, c]) == BG
            grid[r, c] = int(spec["color"])
            seeds.append({"row": int(r), "col": int(c), "color": int(spec["color"])})
    return grid, seeds, obstacle


def _specs_by_color(specs: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(spec["color"]): spec for spec in specs}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    boundary_edge = rng.choice(list(EDGES))
    n_templates = 3 if rng.random() < 0.72 else 4

    if boundary_edge in {"top", "bottom"}:
        h = rng.choice([20, 24, 28])
        w = rng.randint(20, 29)
    else:
        h = rng.randint(20, 29)
        w = rng.choice([20, 24, 28])

    assignments = _assign_edges(n_templates, boundary_edge, rng)
    specs = _make_specs(assignments, boundary_edge, rng)
    boundary_color = int(specs[0]["color"])
    placed = _place_specs(specs, (h, w), boundary_edge, rng)
    if placed is None:
        return None

    grid, seeds, obstacle = _build_grid(placed, (h, w))
    by_color = _specs_by_color(placed)

    dist_keys = [_center_distance_sq(grid.shape, seed["row"], seed["col"]) for seed in seeds]
    if len(dist_keys) != len(set(dist_keys)):
        return None

    branch_counts = {"boundary": 0, "template": 0}
    boundary_ok = False
    total_stamp_cells = 0
    for seed in seeds:
        template = by_color[int(seed["color"])]
        stamps, reason, touches = _stamp_positions(template, (seed["row"], seed["col"]), grid.shape, obstacle)
        if not stamps:
            return None
        total_stamp_cells += sum(len(stamp) for stamp in stamps)
        branch_counts[reason] += 1
        if int(template["color"]) == boundary_color and _edge_of_seed(grid.shape, seed["row"], seed["col"]) == boundary_edge:
            boundary_ok = reason == "boundary" and touches
    if not boundary_ok:
        return None
    if branch_counts["template"] == 0 or branch_counts["boundary"] == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if total_stamp_cells <= len(seeds):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except (AssertionError, RuntimeError, ValueError):
            continue
        if grid is not None:
            try:
                out = main(grid)
            except Exception:
                continue
            if out.ndim == 2 and max(out.shape) <= 30:
                return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

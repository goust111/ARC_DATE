from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

BG = 0
WALL = 8
WATER = 2

FRAME_MIN = 4
FRAME_MAX = 7
SEGMENT_MIN = 5
SEGMENT_MAX = 9
CHAIN_LENGTHS = (2, 3, 3, 4)
GEN_ATTEMPTS = 400
CHAIN_TRIES = 180
ISOLATED_TRIES = 180
COUNT_MAX_PER_FRAME = 4

CORNER_RAYS = {
    frozenset((UP, LEFT)): "TL",
    frozenset((UP, RIGHT)): "TR",
    frozenset((DOWN, LEFT)): "BL",
    frozenset((DOWN, RIGHT)): "BR",
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _opp(direction: tuple[int, int]) -> tuple[int, int]:
    return (-direction[0], -direction[1])


def _perpendiculars(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction in (UP, DOWN):
        return LEFT, RIGHT
    return UP, DOWN


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _gap_rays(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> tuple[tuple[int, int], ...] | None:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gap == (r0, c0):
        return (UP, LEFT)
    if gap == (r0, c1):
        return (UP, RIGHT)
    if gap == (r1, c0):
        return (DOWN, LEFT)
    if gap == (r1, c1):
        return (DOWN, RIGHT)
    if gr == r0 and c0 < gc < c1:
        return (UP,)
    if gr == r1 and c0 < gc < c1:
        return (DOWN,)
    if gc == c0 and r0 < gr < r1:
        return (LEFT,)
    if gc == c1 and r0 < gr < r1:
        return (RIGHT,)
    return None


def _frame_from_component(
    grid: np.ndarray, comp: list[tuple[int, int]], wall: int, water: int, bg: int
) -> dict | None:
    bbox = _bbox_coords(comp)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < FRAME_MIN or w < FRAME_MIN:
        return None

    border = _border_cells(bbox)
    comp_set = set(comp)
    if len(comp_set) != 2 * (h + w) - 5:
        return None
    if any(cell not in border for cell in comp_set):
        return None

    missing = [cell for cell in border if cell not in comp_set]
    if len(missing) != 1:
        return None
    gap = missing[0]
    rays = _gap_rays(bbox, gap)
    if rays is None:
        return None

    for cell in border:
        value = int(grid[cell])
        if cell == gap:
            if value != bg:
                return None
        elif value != wall:
            return None

    water_cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value == wall:
                return None
            if value == water:
                water_cells.append((r, c))

    if not water_cells:
        return None

    return {
        "bbox": bbox,
        "gap": gap,
        "rays": tuple(rays),
        "water_cells": water_cells,
        "water_count": len(water_cells),
    }


def _try_parse_with_colors(grid: np.ndarray, wall: int, water: int, bg: int) -> dict | None:
    wall_mask = grid == wall
    comps = _components(wall_mask)
    if not comps:
        return None

    frames: list[dict] = []
    owner = np.full(grid.shape, -1, dtype=int)
    for comp in comps:
        frame = _frame_from_component(grid, comp, wall, water, bg)
        if frame is None:
            return None
        idx = len(frames)
        for cell in frame["water_cells"]:
            if owner[cell] != -1:
                return None
            owner[cell] = idx
        frames.append(frame)

    water_mask = grid == water
    if np.any(water_mask & (owner == -1)):
        return None

    gap_map = {frame["gap"]: i for i, frame in enumerate(frames)}
    port_targets: dict[int, dict[tuple[int, int], int]] = {i: {} for i in range(len(frames))}
    port_paths: dict[tuple[int, tuple[int, int]], list[tuple[int, int]]] = {}
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(frames))}

    for i, frame in enumerate(frames):
        gr, gc = frame["gap"]
        for direction in frame["rays"]:
            r = gr + direction[0]
            c = gc + direction[1]
            path: list[tuple[int, int]] = []
            while _in_bounds(grid.shape, r, c):
                if (r, c) in gap_map:
                    j = gap_map[(r, c)]
                    if j != i and _opp(direction) in frames[j]["rays"]:
                        port_targets[i][direction] = j
                        port_paths[(i, direction)] = path
                    break
                if int(grid[r, c]) != bg:
                    break
                path.append((r, c))
                r += direction[0]
                c += direction[1]

    for i, targets in port_targets.items():
        for direction, j in targets.items():
            if port_targets[j].get(_opp(direction)) != i:
                return None
            adjacency[i].add(j)

    chains: list[list[int]] = []
    isolated: list[int] = []
    seen: set[int] = set()

    for start in range(len(frames)):
        if start in seen:
            continue
        queue = deque([start])
        component: list[int] = []
        seen.add(start)
        while queue:
            cur = queue.popleft()
            component.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)

        degrees = {idx: len(adjacency[idx]) for idx in component}
        if len(component) == 1 and degrees[start] == 0:
            if len(frames[start]["rays"]) != 1:
                return None
            isolated.append(start)
            continue

        if any(deg > 2 for deg in degrees.values()):
            return None

        chain_starts = [idx for idx in component if len(frames[idx]["rays"]) == 1 and degrees[idx] == 1]
        chain_ends = [idx for idx in component if len(frames[idx]["rays"]) == 2 and degrees[idx] == 1]
        internal = [idx for idx in component if degrees[idx] == 2]
        if len(chain_starts) != 1 or len(chain_ends) != 1:
            return None
        if any(len(frames[idx]["rays"]) != 2 for idx in internal):
            return None
        if len(chain_starts) + len(chain_ends) + len(internal) != len(component):
            return None

        order = [chain_starts[0]]
        prev = None
        cur = chain_starts[0]
        while True:
            nxts = [idx for idx in adjacency[cur] if idx != prev]
            if not nxts:
                break
            if len(nxts) != 1:
                return None
            prev, cur = cur, nxts[0]
            order.append(cur)

        if order[-1] != chain_ends[0]:
            return None
        chains.append(order)

    return {
        "bg": bg,
        "wall": wall,
        "water": water,
        "frames": frames,
        "port_targets": port_targets,
        "port_paths": port_paths,
        "adjacency": adjacency,
        "chains": chains,
        "isolated": isolated,
    }


def _analyze_scene(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    colors = _colors_present(grid, exclude={bg})
    assert len(colors) == 2, "expected background plus exactly two foreground colors"

    candidates = []
    for wall, water in ((colors[0], colors[1]), (colors[1], colors[0])):
        parsed = _try_parse_with_colors(grid, int(wall), int(water), bg)
        if parsed is not None:
            candidates.append(parsed)

    assert len(candidates) == 1, "scene must have a unique wall/water interpretation"
    return candidates[0]


def _paint_line(out: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = color


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    dr = 0 if ar == br else (1 if br > ar else -1)
    dc = 0 if ac == bc else (1 if bc > ac else -1)
    assert dr == 0 or dc == 0
    dist = abs(br - ar) + abs(bc - ac)
    return [(ar + k * dr, ac + k * dc) for k in range(1, dist)]


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    return [(sr + dr * k, sc + dc * k) for k in range(1, length + 1)]


def _expected_output_from_scene(grid: np.ndarray, scene: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == scene["water"]] = scene["bg"]

    def link_direction(src: int, dst: int) -> tuple[int, int]:
        for direction, other in scene["port_targets"][src].items():
            if other == dst:
                return direction
        raise AssertionError("missing chain link")

    for order in scene["chains"]:
        for a, b in zip(order, order[1:]):
            segment = _cells_between(scene["frames"][a]["gap"], scene["frames"][b]["gap"])
            _paint_line(out, segment, scene["water"])

        terminal = order[-1]
        prev = order[-2]
        used = link_direction(terminal, prev)
        unused = [direction for direction in scene["frames"][terminal]["rays"] if direction != used]
        assert len(unused) == 1
        total = sum(scene["frames"][idx]["water_count"] for idx in order)
        _paint_line(out, _ray_cells(scene["frames"][terminal]["gap"], unused[0], total), scene["water"])

    for idx in scene["isolated"]:
        direction = scene["frames"][idx]["rays"][0]
        _paint_line(
            out,
            _ray_cells(scene["frames"][idx]["gap"], direction, scene["frames"][idx]["water_count"]),
            scene["water"],
        )

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear all framed droplets, route them through uniquely aligned gap pipes, and drain each chain total from its terminal opening.

    Input structure:
    - A dominant black background contains several disjoint hollow rectangular frames of one wall color.
    - Each frame is missing exactly one border cell; a side gap exposes one outward ray, while a corner gap exposes the two orthogonal outward rays of the missing corner.
    - The interior of every frame contains one or more same-colored droplet pixels, and some exposed rays line up through empty background to another frame gap.

    Rule:
    - Treat every exposed ray as a pipe direction through which droplets can travel.
    - A connected path must start at a one-ray side gap, optionally turn through two-ray corner gaps, and end at a two-ray corner gap whose unused ray points outward.
    - Paint every inter-gap pipe segment with the droplet color, erase all interior droplets to background, and from the terminal gap extend a straight droplet strip whose length equals the total droplet count of that whole path; an isolated one-ray frame instead emits a strip of its own count.

    Key insight:
    - The missing border cell is both a geometric port and a topological hint: side gaps can only start or stand alone, while corner gaps let a chain turn and reveal which terminal must discharge the conserved total.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)
    return _expected_output_from_scene(g, scene)


def _make_side_spec(
    gap: tuple[int, int], direction: tuple[int, int], h: int, w: int, offset: int
) -> dict:
    gr, gc = gap
    if direction == RIGHT:
        bbox = (gr - offset, gc - w + 1, gr - offset + h - 1, gc)
    elif direction == LEFT:
        bbox = (gr - offset, gc, gr - offset + h - 1, gc + w - 1)
    elif direction == DOWN:
        bbox = (gr - h + 1, gc - offset, gr, gc - offset + w - 1)
    elif direction == UP:
        bbox = (gr, gc - offset, gr + h - 1, gc - offset + w - 1)
    else:
        raise ValueError("invalid side direction")
    return {"bbox": bbox, "gap": gap, "rays": (direction,), "water_count": 0, "water_cells": []}


def _make_corner_spec(
    gap: tuple[int, int], rays: tuple[tuple[int, int], tuple[int, int]], h: int, w: int
) -> dict:
    kind = CORNER_RAYS[frozenset(rays)]
    gr, gc = gap
    if kind == "TL":
        bbox = (gr, gc, gr + h - 1, gc + w - 1)
    elif kind == "TR":
        bbox = (gr, gc - w + 1, gr + h - 1, gc)
    elif kind == "BL":
        bbox = (gr - h + 1, gc, gr, gc + w - 1)
    else:
        bbox = (gr - h + 1, gc - w + 1, gr, gc)
    return {"bbox": bbox, "gap": gap, "rays": tuple(rays), "water_count": 0, "water_cells": []}


def _bbox_in_bounds(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _cells_clear_of_bboxes(cells: list[tuple[int, int]], specs: list[dict]) -> bool:
    for cell in cells:
        if any(_inside_bbox(cell, spec["bbox"]) for spec in specs):
            return False
    return True


def _available_run(
    shape: tuple[int, int], gap: tuple[int, int], direction: tuple[int, int], specs: list[dict], path_cells: set[tuple[int, int]]
) -> int:
    count = 0
    r, c = gap
    dr, dc = direction
    r += dr
    c += dc
    while _in_bounds(shape, r, c) and (r, c) not in path_cells and _cells_clear_of_bboxes([(r, c)], specs):
        count += 1
        r += dr
        c += dc
    return count


def _choose_water_cells(spec: dict, count: int) -> list[tuple[int, int]]:
    interior = _interior_cells(spec["bbox"])
    assert 0 < count <= len(interior)
    return random.sample(interior, count)


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[spec["gap"]] = BG
        for cell in spec["water_cells"]:
            grid[cell] = WATER
    return grid


def _spec_fits(
    spec: dict, shape: tuple[int, int], existing_specs: list[dict], path_cells: set[tuple[int, int]]
) -> bool:
    if not _bbox_in_bounds(spec["bbox"], shape):
        return False
    if any(not _bbox_far(spec["bbox"], other["bbox"], pad=1) for other in existing_specs):
        return False
    if any(cell in path_cells for cell in _bbox_cells(spec["bbox"])):
        return False
    return True


def _try_add_chain(shape: tuple[int, int], specs: list[dict], items: list[dict], path_cells: set[tuple[int, int]]) -> bool:
    for _ in range(CHAIN_TRIES):
        length = random.choice(CHAIN_LENGTHS)
        edge_dirs = [random.choice(DIRS)]
        for _step in range(length - 2):
            edge_dirs.append(random.choice(_perpendiculars(edge_dirs[-1])))
        drain_dir = random.choice(_perpendiculars(edge_dirs[-1]))

        local_specs: list[dict] = []
        local_paths: set[tuple[int, int]] = set()

        h = random.randint(FRAME_MIN, FRAME_MAX)
        w = random.randint(FRAME_MIN, FRAME_MAX)
        offset = random.randint(1, h - 2) if edge_dirs[0] in (LEFT, RIGHT) else random.randint(1, w - 2)
        start_gap = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1))
        start_spec = _make_side_spec(start_gap, edge_dirs[0], h, w, offset)
        if not _spec_fits(start_spec, shape, specs, path_cells):
            continue
        local_specs.append(start_spec)

        prev_gap = start_gap
        ok = True
        for step in range(1, length):
            segment_len = random.randint(SEGMENT_MIN, SEGMENT_MAX)
            next_gap = (
                prev_gap[0] + edge_dirs[step - 1][0] * segment_len,
                prev_gap[1] + edge_dirs[step - 1][1] * segment_len,
            )
            next_out = drain_dir if step == length - 1 else edge_dirs[step]
            rays = (_opp(edge_dirs[step - 1]), next_out)
            h = random.randint(FRAME_MIN, FRAME_MAX)
            w = random.randint(FRAME_MIN, FRAME_MAX)
            spec = _make_corner_spec(next_gap, rays, h, w)

            segment = _cells_between(prev_gap, next_gap)
            if not segment or not _cells_clear_of_bboxes(segment, specs + local_specs):
                ok = False
                break
            if any(cell in path_cells or cell in local_paths for cell in segment):
                ok = False
                break
            if not _spec_fits(spec, shape, specs + local_specs, path_cells | local_paths):
                ok = False
                break

            local_paths.update(segment)
            local_specs.append(spec)
            prev_gap = next_gap

        if not ok:
            continue

        counts = []
        for spec in local_specs:
            cap = min(COUNT_MAX_PER_FRAME, len(_interior_cells(spec["bbox"])))
            counts.append(random.randint(1, cap))
        total = sum(counts)

        terminal = local_specs[-1]
        max_run = _available_run(shape, terminal["gap"], drain_dir, specs + local_specs, path_cells | local_paths)
        if total > max_run:
            continue
        drain = _ray_cells(terminal["gap"], drain_dir, total)
        if any(cell in path_cells or cell in local_paths for cell in drain):
            continue
        if not _cells_clear_of_bboxes(drain, specs + local_specs):
            continue

        base = len(specs)
        indices = []
        for spec, count in zip(local_specs, counts):
            spec["water_count"] = count
            spec["water_cells"] = _choose_water_cells(spec, count)
            specs.append(spec)
            indices.append(base)
            base += 1

        path_cells.update(local_paths)
        path_cells.update(drain)
        items.append({"kind": "chain", "frames": indices, "drain_dir": drain_dir})
        return True

    return False


def _try_add_isolated(shape: tuple[int, int], specs: list[dict], items: list[dict], path_cells: set[tuple[int, int]]) -> bool:
    for _ in range(ISOLATED_TRIES):
        direction = random.choice(DIRS)
        h = random.randint(FRAME_MIN, FRAME_MAX)
        w = random.randint(FRAME_MIN, FRAME_MAX)
        offset = random.randint(1, h - 2) if direction in (LEFT, RIGHT) else random.randint(1, w - 2)
        gap = (random.randint(0, shape[0] - 1), random.randint(0, shape[1] - 1))
        spec = _make_side_spec(gap, direction, h, w, offset)
        if not _spec_fits(spec, shape, specs, path_cells):
            continue

        cap = min(COUNT_MAX_PER_FRAME, len(_interior_cells(spec["bbox"])))
        max_run = _available_run(shape, spec["gap"], direction, specs + [spec], path_cells)
        if max_run <= 0:
            continue
        count = random.randint(1, min(cap, max_run))
        drain = _ray_cells(spec["gap"], direction, count)
        if any(cell in path_cells for cell in drain):
            continue
        if not _cells_clear_of_bboxes(drain, specs + [spec]):
            continue

        spec["water_count"] = count
        spec["water_cells"] = _choose_water_cells(spec, count)
        idx = len(specs)
        specs.append(spec)
        path_cells.update(drain)
        items.append({"kind": "isolated", "frame": idx})
        return True

    return False


def _expected_from_items(shape: tuple[int, int], specs: list[dict], items: list[dict]) -> np.ndarray:
    inp = _render_input(shape, specs)
    out = inp.copy()
    out[out == WATER] = BG
    for item in items:
        if item["kind"] == "chain":
            frames = item["frames"]
            for a, b in zip(frames, frames[1:]):
                _paint_line(out, _cells_between(specs[a]["gap"], specs[b]["gap"]), WATER)
            total = sum(specs[idx]["water_count"] for idx in frames)
            _paint_line(out, _ray_cells(specs[frames[-1]]["gap"], item["drain_dir"], total), WATER)
        else:
            idx = item["frame"]
            direction = specs[idx]["rays"][0]
            _paint_line(out, _ray_cells(specs[idx]["gap"], direction, specs[idx]["water_count"]), WATER)
    return out


def _semantic_signature(items: list[dict]) -> tuple[tuple[int, ...], int]:
    chain_lengths = sorted(len(item["frames"]) for item in items if item["kind"] == "chain")
    isolated = sum(item["kind"] == "isolated" for item in items)
    return tuple(chain_lengths), isolated


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(19, 28), random.randint(19, 28))
    specs: list[dict] = []
    items: list[dict] = []
    path_cells: set[tuple[int, int]] = set()

    chain_target = random.choice((1, 1, 2))
    isolated_target = random.choice((0, 1, 1, 2))

    for _ in range(chain_target):
        if not _try_add_chain(shape, specs, items, path_cells):
            return None
    for _ in range(isolated_target):
        if not _try_add_isolated(shape, specs, items, path_cells):
            return None

    if len(specs) < 2:
        return None

    inp = _render_input(shape, specs)
    expected = _expected_from_items(shape, specs, items)

    try:
        scene = _analyze_scene(inp)
    except AssertionError:
        return None

    if len(scene["frames"]) != len(specs):
        return None
    if _semantic_signature(items) != (tuple(sorted(len(chain) for chain in scene["chains"])), len(scene["isolated"])):
        return None
    if not np.array_equal(_expected_output_from_scene(inp, scene), expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-chain drainage scene")

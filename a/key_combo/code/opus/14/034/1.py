from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SHAPE_COLORS = (1, 5, 6, 7, 8, 9)
SIDES = ("N", "E", "S", "W")


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def rotate_orbit(side: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = side - 1
    return (
        (r, c),
        (c, last - r),
        (last - r, last - c),
        (last - c, r),
    )


def orbit_by_side(side: int, rep: tuple[int, int]) -> dict[str, tuple[int, int]]:
    pts = rotate_orbit(side, rep[0], rep[1])
    return dict(zip(SIDES, pts))


def is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nb in neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def complete_visible_shape(r0: int, c0: int, side: int, visible: set[tuple[int, int]]) -> set[tuple[int, int]]:
    full: set[tuple[int, int]] = set()
    for r, c in visible:
        rr = r - r0
        cc = c - c0
        for ur, uc in rotate_orbit(side, rr, cc):
            full.add((r0 + ur, c0 + uc))
    return full


def shape_sort_key(shape: dict) -> tuple[int, int]:
    return int(shape["r0"]), int(shape["c0"])


def parse_instance(grid: np.ndarray) -> tuple[list[dict], int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    shape_mask = (g != BACKGROUND) & (g != GREEN)
    comps = components_mask(shape_mask)
    assert 4 <= len(comps) <= 6, "expected 4-6 shape components"

    shapes: list[dict] = []
    seed_id: int | None = None
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        has_seed = YELLOW in colors
        base_colors = colors - {YELLOW}
        assert len(base_colors) == 1, "each shape must have a unique body color"
        color = int(next(iter(base_colors)))
        if has_seed:
            assert seed_id is None, "expected exactly one yellow seed marker"
        r0, c0, r1, c1 = bbox_coords(coords)
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "shape bbox must already reveal its rotation square"

        visible = {(int(r), int(c)) for r, c in coords}
        full = complete_visible_shape(r0, c0, side, visible)
        assert all(0 <= r < h and 0 <= c < w for r, c in full), "completed shape left the grid"
        shapes.append(
            {
                "r0": r0,
                "c0": c0,
                "side": side,
                "color": color,
                "visible": visible,
                "full": full,
                "has_seed": has_seed,
            }
        )
        if has_seed:
            seed_id = len(shapes) - 1

    assert seed_id is not None, "expected one seed shape"

    for i, a in enumerate(shapes):
        for j in range(i + 1, len(shapes)):
            b = shapes[j]
            assert a["full"].isdisjoint(b["full"]), "completed shapes must not overlap"

    bridges: list[dict] = []
    bridge_coords = [(int(r), int(c)) for r, c in np.argwhere(g == GREEN)]
    assert 2 <= len(bridge_coords) <= 5, "expected 2-5 bridge pixels"

    for br, bc in bridge_coords:
        visible_neighbors = []
        full_neighbors = []
        for idx, shape in enumerate(shapes):
            if any(nb in shape["visible"] for nb in neighbors4(br, bc)):
                visible_neighbors.append(idx)
            if any(nb in shape["full"] for nb in neighbors4(br, bc)):
                full_neighbors.append(idx)
        assert not visible_neighbors, "bridge pixels may not touch visible shape pixels"
        assert len(full_neighbors) == 2, "each bridge must connect exactly two shapes after completion"
        bridges.append({"coord": (br, bc), "shapes": tuple(sorted(full_neighbors))})

    return shapes, seed_id, bridges


def simulate_cascade(shapes: list[dict], seed_id: int, bridges: list[dict]) -> tuple[set[int], dict[tuple[int, int], int]]:
    completed: set[int] = set()
    pending: set[int] = {seed_id}
    activated_bridges: dict[tuple[int, int], int] = {}

    while pending:
        sid = min(pending, key=lambda idx: shape_sort_key(shapes[idx]))
        pending.remove(sid)
        if sid in completed:
            continue
        completed.add(sid)

        changed = True
        while changed:
            changed = False
            for bridge in bridges:
                coord = bridge["coord"]
                if coord in activated_bridges:
                    continue
                a, b = bridge["shapes"]
                if a in completed and b not in completed:
                    activated_bridges[coord] = int(shapes[a]["color"])
                    pending.add(b)
                    changed = True
                elif b in completed and a not in completed:
                    activated_bridges[coord] = int(shapes[b]["color"])
                    pending.add(a)
                    changed = True

    return completed, activated_bridges


def render_output(shape_count: tuple[int, int], shapes: list[dict], completed: set[int], bridges: list[dict], activated_bridges: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.full(shape_count, BACKGROUND, dtype=int)
    for idx, shape in enumerate(shapes):
        if idx in completed:
            for r, c in shape["full"]:
                out[r, c] = RED
        else:
            for r, c in shape["visible"]:
                out[r, c] = int(shape["color"])

    for bridge in bridges:
        br, bc = bridge["coord"]
        out[br, bc] = int(activated_bridges.get((br, bc), GREEN))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the seeded symmetry fragments, let hidden bridge contacts cascade, then repaint the reached network.

    Input structure:
    - A black background contains 4-6 separated shape fragments plus 2-5 isolated green bridge pixels.
    - Each shape fragment is monochrome except for one yellow marker that replaces a pixel on the seed shape.
    - Every fragment fits inside its own visible square bbox and is part of a larger 90-degree rotationally symmetric shape.

    Rule:
    - Find the unique shape containing the yellow marker and restore that marker to the shape's body color.
    - For any shape that becomes active, complete it by copying every visible pixel through its 0, 90, 180, and 270 degree positions inside that square bbox.
    - After each completion, activate every green bridge pixel that now touches one completed shape and one not-yet-completed shape when adjacency is tested against the shapes' completed footprints; the bridge takes the completed shape's original color.
    - Repeat in bbox top-left order for all newly activated shapes, then repaint every completed shape red while leaving untouched fragments and unactivated bridges unchanged.

    Key insight:
    The bridges are intentionally one step away from the visible fragments, so the cascade can only be discovered by first reconstructing each shape's hidden rotational closure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    shapes, seed_id, bridges = parse_instance(g)
    completed, activated_bridges = simulate_cascade(shapes, seed_id, bridges)
    return render_output(g.shape, shapes, completed, bridges, activated_bridges)


def full_local_shape(side: int) -> dict[str, dict[str, tuple[int, int]] | tuple[int, int]]:
    mid = side // 2
    return {
        "center": (mid, mid),
        "inner": orbit_by_side(side, (mid - 1, mid)),
        "diag": orbit_by_side(side, (mid - 1, mid - 1)),
        "side": orbit_by_side(side, (0, mid)),
        "edge": orbit_by_side(side, (0, mid - 1)),
    }


def choose_visible_local(side: int, hidden_sides: set[str]) -> set[tuple[int, int]]:
    spec = full_local_shape(side)
    core = {spec["center"]}
    core.update(spec["inner"].values())
    core.update(spec["diag"].values())

    side_options = [s for s in SIDES if s not in hidden_sides]
    candidates: list[set[tuple[int, int]]] = []
    for side_count in range(1, min(3, len(side_options)) + 1):
        for side_keep in itertools.combinations(side_options, side_count):
            for edge_count in range(1, 4):
                for edge_keep in itertools.combinations(SIDES, edge_count):
                    if set(side_keep) | set(edge_keep) != set(SIDES):
                        continue
                    total = 9 + len(side_keep) + len(edge_keep)
                    if 13 <= total <= 15:
                        visible = set(core)
                        for name in side_keep:
                            visible.add(spec["side"][name])
                        for name in edge_keep:
                            visible.add(spec["edge"][name])
                        if not is_connected(visible):
                            continue
                        rs = [r for r, _ in visible]
                        cs = [c for _, c in visible]
                        if min(rs) != 0 or max(rs) != side - 1 or min(cs) != 0 or max(cs) != side - 1:
                            continue
                        candidates.append(visible)
    assert candidates, "no valid visible-orbit choices"
    return set(random.choice(candidates))


def build_layout(side: int, shape_count: int, reachable_count: int, connect_unreachable_pair: bool) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    close = side + 1
    far = side + 6
    base_positions = [
        (2, 2),
        (2, 2 + close),
        (2 + close, 2 + close),
        (2 + close, 2 + 2 * close),
    ]
    positions = base_positions[:reachable_count]
    edges = [(i, i + 1) for i in range(reachable_count - 1)]

    remaining = shape_count - reachable_count
    lower_r = 2 + 2 * close + 4
    if remaining >= 1:
        positions.append((lower_r, 2))
    if remaining >= 2:
        positions.append((lower_r, 2 + (close if connect_unreachable_pair else far)))
        if connect_unreachable_pair:
            edges.append((reachable_count, reachable_count + 1))
    return positions, edges


def edge_bridge(side: int, a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[str, str]]:
    mid = side // 2
    ar, ac = a
    br, bc = b
    if ar == br and bc == ac + side + 1:
        return (ar + mid, ac + side), ("E", "W")
    if ar == br and ac == bc + side + 1:
        return (br + mid, bc + side), ("W", "E")
    if ac == bc and br == ar + side + 1:
        return (ar + side, ac + mid), ("S", "N")
    if ac == bc and ar == br + side + 1:
        return (br + side, bc + mid), ("N", "S")
    raise AssertionError("unexpected layout edge geometry")


def visible_distance_ok(shapes: list[dict]) -> bool:
    for i, a in enumerate(shapes):
        for j in range(i + 1, len(shapes)):
            b = shapes[j]
            if min(manhattan(pa, pb) for pa in a["visible"] for pb in b["visible"]) < 3:
                return False
    return True


def full_contacts_ok(shapes: list[dict], bridges: list[dict]) -> bool:
    bridge_cells = {bridge["coord"] for bridge in bridges}
    for i, a in enumerate(shapes):
        for j in range(i + 1, len(shapes)):
            b = shapes[j]
            for ra, ca in a["full"]:
                for nb in neighbors4(ra, ca):
                    if nb in b["full"] and nb not in bridge_cells and (ra, ca) not in bridge_cells:
                        return False
    return True


def shape_graph_counts(shape_count: int) -> tuple[int, bool]:
    reachable_options = [3, 4]
    reachable_options = [v for v in reachable_options if 3 <= v < shape_count]
    reachable_count = random.choice(reachable_options)
    remaining = shape_count - reachable_count
    connect_unreachable_pair = remaining >= 2 and random.random() < 0.65
    return reachable_count, connect_unreachable_pair


def paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def make_instance() -> tuple[np.ndarray, np.ndarray]:
    side = 5
    shape_count = random.randint(4, 6)
    reachable_count, connect_unreachable_pair = shape_graph_counts(shape_count)
    positions, edges = build_layout(side, shape_count, reachable_count, connect_unreachable_pair)
    colors = random.sample(SHAPE_COLORS, shape_count)

    hidden_sides = [set() for _ in range(shape_count)]
    bridges_meta: list[dict] = []
    for a, b in edges:
        coord, (sa, sb) = edge_bridge(side, positions[a], positions[b])
        hidden_sides[a].add(sa)
        hidden_sides[b].add(sb)
        bridges_meta.append({"coord": coord, "shapes": (a, b)})

    shapes_meta: list[dict] = []
    for idx, ((r0, c0), color) in enumerate(zip(positions, colors)):
        local_visible = choose_visible_local(side, hidden_sides[idx])
        visible = {(r0 + r, c0 + c) for r, c in local_visible}
        full = complete_visible_shape(r0, c0, side, visible)
        shapes_meta.append(
            {
                "r0": r0,
                "c0": c0,
                "side": side,
                "color": color,
                "visible": visible,
                "full": full,
            }
        )

    h = max(r for shape in shapes_meta for r, _ in shape["full"]) + 3
    w = max(c for shape in shapes_meta for _, c in shape["full"]) + 3
    inp = np.full((h, w), BACKGROUND, dtype=int)

    for bridge in bridges_meta:
        br, bc = bridge["coord"]
        inp[br, bc] = GREEN

    for shape in shapes_meta:
        paint_cells(inp, shape["visible"], shape["color"])

    seed_choices = list(range(reachable_count))
    if reachable_count >= 4 and random.random() < 0.7:
        seed_choices = [1, 2]
    seed_id = random.choice(seed_choices)
    seed_cell = random.choice(tuple(shapes_meta[seed_id]["visible"]))
    inp[seed_cell] = YELLOW
    shapes_meta[seed_id]["has_seed"] = True

    for idx, shape in enumerate(shapes_meta):
        shape.setdefault("has_seed", False)

    completed, activated_bridges = simulate_cascade(shapes_meta, seed_id, bridges_meta)
    expected = render_output(inp.shape, shapes_meta, completed, bridges_meta, activated_bridges)
    return inp, expected


def transform_grid_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ops = [
        lambda g: g,
        np.fliplr,
        np.flipud,
        lambda g: np.flipud(np.fliplr(g)),
        np.transpose,
        lambda g: np.fliplr(np.transpose(g)),
        lambda g: np.flipud(np.transpose(g)),
        lambda g: np.flipud(np.fliplr(np.transpose(g))),
    ]
    op = random.choice(ops)
    return np.asarray(op(inp), dtype=int), np.asarray(op(out), dtype=int)


def semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        shapes, seed_id, bridges = parse_instance(inp)
    except AssertionError:
        return False

    if not visible_distance_ok(shapes):
        return False
    if not full_contacts_ok(shapes, bridges):
        return False
    if len({bridge["coord"] for bridge in bridges}) != len(bridges):
        return False
    if any(manhattan(a["coord"], b["coord"]) <= 1 for i, a in enumerate(bridges) for b in bridges[i + 1 :]):
        return False

    completed, activated = simulate_cascade(shapes, seed_id, bridges)
    if not (3 <= len(completed) < len(shapes)):
        return False
    if len(completed) - 1 < 2:
        return False
    if not np.array_equal(render_output(inp.shape, shapes, completed, bridges, activated), expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        inp, expected = make_instance()
        inp, expected = transform_grid_pair(inp, expected)
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid cascaded symmetry instance")

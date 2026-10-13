from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
PALETTE = (BLUE, RED, GREEN, YELLOW, ORANGE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("line_cells expects an orthogonal segment")


def path_between(
    start: tuple[int, int],
    end: tuple[int, int],
    orientation: str,
) -> list[tuple[int, int]]:
    if orientation == "horizontal":
        if start[0] == end[0] and random.random() < 0.45:
            return line_cells(start, end)
        lo = min(start[1], end[1])
        hi = max(start[1], end[1])
        mid_c = random.randint(lo, hi)
        cells: list[tuple[int, int]] = []
        for seg in (
            line_cells(start, (start[0], mid_c)),
            line_cells((start[0], mid_c), (end[0], mid_c)),
            line_cells((end[0], mid_c), end),
        ):
            if cells and seg and cells[-1] == seg[0]:
                cells.extend(seg[1:])
            else:
                cells.extend(seg)
        return cells

    if orientation == "vertical":
        if start[1] == end[1] and random.random() < 0.45:
            return line_cells(start, end)
        lo = min(start[0], end[0])
        hi = max(start[0], end[0])
        mid_r = random.randint(lo, hi)
        cells = []
        for seg in (
            line_cells(start, (mid_r, start[1])),
            line_cells((mid_r, start[1]), (mid_r, end[1])),
            line_cells((mid_r, end[1]), end),
        ):
            if cells and seg and cells[-1] == seg[0]:
                cells.extend(seg[1:])
            else:
                cells.extend(seg)
        return cells

    raise ValueError("invalid orientation")


def components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(np.asarray(coords, dtype=int))
    return out


def detect_colored_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(int(v) for v in grid.flatten()) - {BLACK, GREY}):
        for coords in components_from_mask(grid == color):
            pts = [tuple(map(int, p)) for p in coords]
            comps.append(
                {
                    "color": int(color),
                    "coords": pts,
                    "coord_set": set(pts),
                    "area": len(pts),
                }
            )
    return comps


def order_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    degrees = {
        p: sum((nr, nc) in pts for nr, nc in neighbors4(*p))
        for p in pts
    }
    endpoints = [p for p, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "path must be unbranched"

    start = min(endpoints)
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [
            p for p in neighbors4(*cur)
            if p in pts and p != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, "path cannot branch or loop"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
        assert len(ordered) <= len(pts)

    assert len(ordered) == len(pts), "path walk must cover all cells"
    return ordered


def paint_component(out: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        assert in_bounds(out.shape, r, c), "moved component out of bounds"
        assert out[r, c] == BLACK, "components must not overlap in the output"
        out[r, c] = int(color)


def shift_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    comps = detect_colored_components(g)
    comp_id_by_cell = {}
    for idx, comp in enumerate(comps):
        for cell in comp["coords"]:
            comp_id_by_cell[cell] = idx

    paths = [order_path([tuple(map(int, p)) for p in coords]) for coords in components_from_mask(g == GREY)]
    used_components: set[int] = set()
    render: dict[int, tuple[list[tuple[int, int]], int]] = {
        idx: (comp["coords"], comp["color"]) for idx, comp in enumerate(comps)
    }

    for path in paths:
        assert len(path) >= 3, "path length must be at least 3"
        end_a = path[0]
        end_b = path[-1]

        touch_a = {
            comp_id_by_cell[(nr, nc)]
            for nr, nc in neighbors4(*end_a)
            if (nr, nc) in comp_id_by_cell
        }
        touch_b = {
            comp_id_by_cell[(nr, nc)]
            for nr, nc in neighbors4(*end_b)
            if (nr, nc) in comp_id_by_cell
        }
        assert len(touch_a) == 1 and len(touch_b) == 1, "each path endpoint must touch one component"
        cid_a = next(iter(touch_a))
        cid_b = next(iter(touch_b))
        assert cid_a != cid_b, "path must connect two distinct components"
        assert cid_a not in used_components and cid_b not in used_components, "component must belong to at most one path"
        used_components.add(cid_a)
        used_components.add(cid_b)

        for mid in path[1:-1]:
            assert not any((nr, nc) in comp_id_by_cell for nr, nc in neighbors4(*mid)), "path interior cannot touch components"

        comp_a = comps[cid_a]
        comp_b = comps[cid_b]
        anchor_a = [p for p in comp_a["coords"] if abs(p[0] - end_a[0]) + abs(p[1] - end_a[1]) == 1]
        anchor_b = [p for p in comp_b["coords"] if abs(p[0] - end_b[0]) + abs(p[1] - end_b[1]) == 1]
        assert len(anchor_a) == 1 and len(anchor_b) == 1, "endpoint attachment must be unique"
        anchor_a = anchor_a[0]
        anchor_b = anchor_b[0]

        if comp_a["area"] == comp_b["area"]:
            distance = len(path) // 2
            dest_a = path[distance - 1]
            dest_b = path[-distance]
            render[cid_a] = (
                shift_coords(comp_a["coords"], dest_a[0] - anchor_a[0], dest_a[1] - anchor_a[1]),
                comp_a["color"],
            )
            render[cid_b] = (
                shift_coords(comp_b["coords"], dest_b[0] - anchor_b[0], dest_b[1] - anchor_b[1]),
                comp_b["color"],
            )
        elif comp_a["area"] < comp_b["area"]:
            dest_a = path[-1]
            render[cid_a] = (
                shift_coords(comp_a["coords"], dest_a[0] - anchor_a[0], dest_a[1] - anchor_a[1]),
                comp_b["color"],
            )
        else:
            dest_b = path[0]
            render[cid_b] = (
                shift_coords(comp_b["coords"], dest_b[0] - anchor_b[0], dest_b[1] - anchor_b[1]),
                comp_a["color"],
            )

    out = np.zeros_like(g)
    for idx in range(len(comps)):
        coords, color = render[idx]
        paint_component(out, coords, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each path-linked component according to relative size, with smaller components traveling farther and being recolored by the larger one.

    Input structure:
    - Black background contains 3-5 colored 4-connected components chosen from blue, red, green, yellow, and orange.
    - Grey 4-connected one-cell-wide paths connect components in disjoint pairs; every path is unbranched, at least length 3, and touches one component at each endpoint.
    - If the component count is odd, one colored component is not attached to any path.

    Rule:
    - Detect every colored component and every grey path, and identify the two components touched by each path's endpoints.
    - If one component in a pair is smaller, that smaller component is erased from its start, translated along the full path so it lands next to the larger component, and recolored to the larger component's color; the larger component stays put.
    - If the paired components have equal size, both translate toward each other by half the path length (floor), keeping their original colors.
    - Erase all grey path cells and leave any unpaired component unchanged.

    Key insight:
    - Component size is the single controlling signal: it decides which side moves, how far the movement goes, and whether the moving component keeps its own color or is absorbed into the larger component's color class.
    """
    return _solve(grid)


def _raw_shape(area: int, side: str) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if side == "right":
        rows = range(-2, 3)
        cols = range(-3, 1)
    elif side == "left":
        rows = range(-2, 3)
        cols = range(0, 4)
    elif side == "down":
        rows = range(-3, 1)
        cols = range(-2, 3)
    elif side == "up":
        rows = range(0, 4)
        cols = range(-2, 3)
    else:
        raise ValueError("invalid side")

    allowed = {(r, c) for r in rows for c in cols}
    coords = {(0, 0)}
    while len(coords) < area:
        frontier = sorted(
            {
                nb
                for cell in coords
                for nb in neighbors4(*cell)
                if nb in allowed and nb not in coords
            }
        )
        if not frontier:
            raise RuntimeError("shape frontier exhausted")
        weights = []
        for r, c in frontier:
            neighbors = sum((nr, nc) in coords for nr, nc in neighbors4(r, c))
            axis_bonus = 1 if r == 0 or c == 0 else 0
            weights.append(1 + neighbors + axis_bonus)
        pick = random.choices(frontier, weights=weights, k=1)[0]
        coords.add(pick)

    coords_list = list(coords)
    r0 = min(r for r, _ in coords_list)
    c0 = min(c for _, c in coords_list)
    norm = sorted((r - r0, c - c0) for r, c in coords_list)
    anchor = (-r0, -c0)
    return norm, anchor


def make_component(area: int, side: str, color: int) -> dict:
    for _ in range(120):
        coords, anchor = _raw_shape(area, side)
        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        if area >= 3 and (len(rows) == 1 or len(cols) == 1) and random.random() < 0.6:
            continue
        if side == "right":
            assert anchor[1] == max(c for _, c in coords)
        elif side == "left":
            assert anchor[1] == min(c for _, c in coords)
        elif side == "down":
            assert anchor[0] == max(r for r, _ in coords)
        else:
            assert anchor[0] == min(r for r, _ in coords)
        return {"coords": coords, "anchor": anchor, "color": int(color), "area": int(area)}
    raise RuntimeError("failed to generate component")


def absolute_coords(component: dict, anchor_abs: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = component["anchor"]
    rr, cc = anchor_abs
    return [(r - ar + rr, c - ac + cc) for r, c in component["coords"]]


def compose_grid(
    components: list[dict],
    path: list[tuple[int, int]] | None = None,
    margin: int = 3,
) -> np.ndarray:
    all_cells = []
    for comp in components:
        all_cells.extend(comp["coords"])
    if path is not None:
        all_cells.extend(path)
    min_r = min(r for r, _ in all_cells) - margin
    max_r = max(r for r, _ in all_cells) + margin
    min_c = min(c for _, c in all_cells) - margin
    max_c = max(c for _, c in all_cells) + margin
    grid = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for comp in components:
        for r, c in comp["coords"]:
            rr = r - min_r
            cc = c - min_c
            assert grid[rr, cc] == BLACK
            grid[rr, cc] = comp["color"]
    if path is not None:
        for r, c in path:
            rr = r - min_r
            cc = c - min_c
            assert grid[rr, cc] == BLACK
            grid[rr, cc] = GREY
    return grid


def pair_scene(branch: str, colors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if branch == "equal":
        area = random.randint(1, 8)
        area_a = area_b = area
    elif branch == "a_smaller":
        area_a = random.randint(1, 6)
        area_b = random.randint(area_a + 1, 8)
    elif branch == "b_smaller":
        area_b = random.randint(1, 6)
        area_a = random.randint(area_b + 1, 8)
    else:
        raise ValueError("invalid branch")

    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        comp_a = make_component(area_a, "right", colors[0])
        comp_b = make_component(area_b, "left", colors[1])
        a_anchor = (0, 0)
        b_anchor = (random.randint(-3, 3), random.randint(5, 10))
        path = path_between((a_anchor[0], a_anchor[1] + 1), (b_anchor[0], b_anchor[1] - 1), orientation)
    else:
        comp_a = make_component(area_a, "down", colors[0])
        comp_b = make_component(area_b, "up", colors[1])
        a_anchor = (0, 0)
        b_anchor = (random.randint(5, 10), random.randint(-3, 3))
        path = path_between((a_anchor[0] + 1, a_anchor[1]), (b_anchor[0] - 1, b_anchor[1]), orientation)

    abs_a = absolute_coords(comp_a, a_anchor)
    abs_b = absolute_coords(comp_b, b_anchor)
    if set(abs_a) & set(abs_b):
        raise RuntimeError("component overlap")

    all_comp_cells = set(abs_a) | set(abs_b)
    assert len(path) >= 3
    assert len(set(path)) == len(path)
    assert not (set(path) & all_comp_cells)

    for i, cell in enumerate(path):
        touching = {
            idx
            for idx, coords in enumerate((set(abs_a), set(abs_b)))
            if any(nb in coords for nb in neighbors4(*cell))
        }
        if i in (0, len(path) - 1):
            assert len(touching) == 1
        else:
            assert not touching

    scene_comps = [
        {"coords": abs_a, "color": colors[0]},
        {"coords": abs_b, "color": colors[1]},
    ]
    inp = compose_grid(scene_comps, path, margin=2)
    out = main(inp)
    return inp, out


def single_scene(color: int) -> tuple[np.ndarray, np.ndarray]:
    side = random.choice(["right", "left", "up", "down"])
    comp = make_component(random.randint(1, 8), side, color)
    coords = absolute_coords(comp, (0, 0))
    inp = compose_grid([{"coords": coords, "color": color}], None, margin=1)
    return inp, inp.copy()


def can_place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    occ = patch != BLACK
    for dr, dc in np.argwhere(occ):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        r1 = max(0, rr - padding)
        r2 = min(canvas.shape[0], rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(canvas.shape[1], cc + padding + 1)
        if np.any(canvas[r1:r2, c1:c2] != BLACK):
            return False
    return True


def paste_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    target = canvas[r0:r0 + h, c0:c0 + w]
    mask = patch != BLACK
    assert not np.any((target != BLACK) & mask)
    target[mask] = patch[mask]


def choose_scene_plan() -> tuple[list[str], bool]:
    roll = random.random()
    if roll < 0.30:
        return ["equal", random.choice(["a_smaller", "b_smaller"])], True
    if roll < 0.55:
        return ["equal", random.choice(["a_smaller", "b_smaller"])], False
    if roll < 0.78:
        return [random.choice(["a_smaller", "b_smaller"]), random.choice(["a_smaller", "b_smaller"])], True
    if roll < 0.92:
        return ["equal"], True
    return [random.choice(["a_smaller", "b_smaller"])], True


def validate_generated_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    comps = detect_colored_components(g)
    assert 3 <= len(comps) <= 5
    colors = [comp["color"] for comp in comps]
    assert len(set(colors)) == len(colors)
    paths = components_from_mask(g == GREY)
    assert 1 <= len(paths) <= 2
    cell_to_comp = {}
    for idx, comp in enumerate(comps):
        for cell in comp["coords"]:
            cell_to_comp[cell] = idx
    attached = set()
    for coords in paths:
        ordered = order_path([tuple(map(int, p)) for p in coords])
        assert len(ordered) >= 3
        end_ids = []
        for endpoint in (ordered[0], ordered[-1]):
            nbrs = {
                cell_to_comp[(nr, nc)]
                for nr, nc in neighbors4(*endpoint)
                if (nr, nc) in cell_to_comp
            }
            assert len(nbrs) == 1
            cid = next(iter(nbrs))
            assert cid not in attached
            attached.add(cid)
            end_ids.append(cid)
        assert end_ids[0] != end_ids[1]
        for mid in ordered[1:-1]:
            assert not any((nr, nc) in cell_to_comp for nr, nc in neighbors4(*mid))


def _attempt_generate() -> np.ndarray | None:
    branches, include_single = choose_scene_plan()
    scene_count = len(branches) + int(include_single)
    n_components = 2 * len(branches) + int(include_single)
    colors = random.sample(list(PALETTE), n_components)

    scene_inputs: list[np.ndarray] = []
    scene_outputs: list[np.ndarray] = []
    color_idx = 0
    try:
        for branch in branches:
            inp, out = pair_scene(branch, (colors[color_idx], colors[color_idx + 1]))
            scene_inputs.append(inp)
            scene_outputs.append(out)
            color_idx += 2
        if include_single:
            inp, out = single_scene(colors[color_idx])
            scene_inputs.append(inp)
            scene_outputs.append(out)
    except Exception:
        return None

    max_h = max(scene.shape[0] for scene in scene_inputs)
    max_w = max(scene.shape[1] for scene in scene_inputs)
    min_side = max(16, max_h + 4, max_w + 4)
    max_side = 30
    if min_side > max_side:
        return None

    for _ in range(80):
        h = random.randint(min_side, max_side)
        w = random.randint(min_side, max_side)
        canvas_in = np.zeros((h, w), dtype=int)
        canvas_out = np.zeros((h, w), dtype=int)
        order = list(range(scene_count))
        random.shuffle(order)
        placed = True
        for idx in order:
            patch_in = scene_inputs[idx]
            patch_out = scene_outputs[idx]
            ph, pw = patch_in.shape
            candidates = [
                (r0, c0)
                for r0 in range(0, h - ph + 1)
                for c0 in range(0, w - pw + 1)
                if can_place_patch(canvas_in, patch_in, r0, c0, padding=1)
                and can_place_patch(canvas_out, patch_out, r0, c0, padding=1)
            ]
            if not candidates:
                placed = False
                break
            r0, c0 = random.choice(candidates)
            paste_patch(canvas_in, patch_in, r0, c0)
            paste_patch(canvas_out, patch_out, r0, c0)
        if not placed:
            continue
        try:
            validate_generated_input(canvas_in)
            actual = main(canvas_in)
        except Exception:
            continue
        if np.array_equal(actual, canvas_out):
            return canvas_in
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

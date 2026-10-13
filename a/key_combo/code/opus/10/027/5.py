from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GUIDE = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    steps = DIR4 if connectivity == 4 else DIR8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_foreground_components(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != GUIDE)
    comps = []
    for coords in _mask_components(mask, connectivity=4):
        b = _bbox(coords)
        comps.append({"coords": coords, "bbox": b, "area": len(coords)})
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _guide_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    nodes = {tuple(map(int, rc)) for rc in np.argwhere(grid == GUIDE)}
    comps: list[set[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        comp = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nb = (r + dr, c + dc)
                if nb in nodes and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    queue.append(nb)
        comps.append(comp)
    return comps


def _legend_from_component(grid: np.ndarray, comp: dict) -> dict | None:
    coords = set(comp["coords"])
    if comp["area"] != 2:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    if r0 != r1 or c1 != c0 + 1:
        return None
    if coords != {(r0, c0), (r0, c0 + 1)}:
        return None
    return {
        "coords": sorted(coords),
        "bbox": comp["bbox"],
        "replacement": int(grid[r0, c0]),
        "source": int(grid[r0, c0 + 1]),
    }


def _touches_component(node: tuple[int, int], comp_coords: set[tuple[int, int]]) -> bool:
    r, c = node
    for dr, dc in DIR8:
        if (r + dr, c + dc) in comp_coords:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each routed target object using the connected legend and then erase the routing artifacts.

    Input structure:
    - The background is black, with gray guide pixels and several non-gray foreground components.
    - Target objects are compact 4-connected multicolor regions larger than the 1x2 horizontal legend dominos.
    - Each legend domino encodes a replacement rule as [replacement, source].
    - Gray guide pixels form one or more independent paths where consecutive path pixels are connected by two-cell horizontal or vertical jumps.

    Rule:
    - Split the gray guide pixels into path components using two-cell 4-connectivity.
    - For each path, use its endpoints to find the unique 8-adjacent legend and unique 8-adjacent target object.
    - Inside that bound target only, replace every source-colored pixel with the legend's replacement color.
    - Erase all legends and all gray guide pixels to black; any target not bound by a path stays otherwise unchanged.

    Key insight:
    - The legends do not act globally by color alone; the guide-path topology determines which legend applies to which target.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_foreground_components(g)
    legends: list[dict] = []
    targets: list[dict] = []
    for comp in comps:
        legend = _legend_from_component(g, comp)
        if legend is not None:
            legends.append(legend)
        else:
            targets.append({"coords": sorted(comp["coords"]), "bbox": comp["bbox"], "area": comp["area"]})
    assert targets, "expected at least one target object"
    assert legends, "expected at least one legend"

    target_sets = [set(target["coords"]) for target in targets]
    legend_sets = [set(legend["coords"]) for legend in legends]

    bindings: dict[int, tuple[int, int]] = {}
    for path_nodes in _guide_components(g):
        degrees = {
            node: sum((node[0] + dr, node[1] + dc) in path_nodes for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)))
            for node in path_nodes
        }
        assert max(degrees.values(), default=0) <= 2, "guide path may not branch"
        endpoints = [node for node, deg in degrees.items() if deg <= 1]
        if len(path_nodes) == 1:
            endpoints = list(path_nodes)
        assert 1 <= len(endpoints) <= 2, "guide component must be a simple chain"

        touched_legends = {
            idx
            for node in endpoints
            for idx, coords in enumerate(legend_sets)
            if _touches_component(node, coords)
        }
        touched_targets = {
            idx
            for node in endpoints
            for idx, coords in enumerate(target_sets)
            if _touches_component(node, coords)
        }
        assert len(touched_legends) == 1, "each path must point to exactly one legend"
        assert len(touched_targets) == 1, "each path must point to exactly one target"

        target_idx = next(iter(touched_targets))
        legend_idx = next(iter(touched_legends))
        assert target_idx not in bindings, "a target may be bound by only one path"
        bindings[target_idx] = (
            legends[legend_idx]["replacement"],
            legends[legend_idx]["source"],
        )

    out = g.copy()
    for target_idx, (replacement, source) in bindings.items():
        for r, c in targets[target_idx]["coords"]:
            if int(out[r, c]) == int(source):
                out[r, c] = int(replacement)

    out[out == GUIDE] = BG
    for legend in legends:
        for r, c in legend["coords"]:
            out[r, c] = BG
    return out


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _grow_polyomino(area: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.append(nb)
        frontier.sort(key=lambda rc: (abs(rc[0]) + abs(rc[1]), rc))
        pool = frontier[: max(4, min(len(frontier), 10))]
        cells.add(random.choice(pool))
    return _normalize(cells)


def _shape_patch(shape: list[tuple[int, int]], colors: dict[tuple[int, int], int]) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for rc in shape:
        patch[rc] = int(colors[rc])
    return patch


def _grow_subset(shape_cells: list[tuple[int, int]], blocked: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    free = [rc for rc in shape_cells if rc not in blocked]
    if len(free) < size:
        return None
    start = random.choice(free)
    chosen = {start}
    while len(chosen) < size:
        frontier = []
        for r, c in chosen:
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in free and nb not in chosen:
                    frontier.append(nb)
        if not frontier:
            return None
        frontier = sorted(set(frontier), key=lambda rc: (abs(rc[0] - start[0]) + abs(rc[1] - start[1]), rc))
        chosen.add(random.choice(frontier[: max(3, min(8, len(frontier)))]))
    return chosen


def _make_target_patch(source: int, base_choices: list[int]) -> np.ndarray:
    for _ in range(80):
        area = random.randint(8, 15)
        shape = _grow_polyomino(area)
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        if h > 5 or w > 5:
            continue
        if area < int(np.ceil(0.55 * h * w)):
            continue
        n_colors = random.choice([2, 2, 3])
        base = random.choice([c for c in base_choices if c != source])
        source_size = random.randint(2, min(5, area - 1))
        source_cells = _grow_subset(shape, set(), source_size)
        if source_cells is None:
            continue
        colors = {rc: base for rc in shape}
        for rc in source_cells:
            colors[rc] = source
        if n_colors == 3:
            extra_choices = [c for c in base_choices if c not in (source, base)]
            if not extra_choices:
                continue
            extra_size = random.randint(1, min(3, area - len(source_cells) - 1))
            extra_cells = _grow_subset(shape, set(source_cells), extra_size)
            if extra_cells is None:
                continue
            extra = random.choice(extra_choices)
            for rc in extra_cells:
                colors[rc] = extra
        return _shape_patch(shape, colors)
    raise RuntimeError("failed to build target patch")


def _can_place(mask: np.ndarray, patch: np.ndarray, top: int, left: int, padding: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > mask.shape[0] or left + w > mask.shape[1]:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(mask.shape[0], top + h + padding)
    c1 = min(mask.shape[1], left + w + padding)
    if np.any(mask[r0:r1, c0:c1]):
        return False
    return not np.any(mask[top : top + h, left : left + w] & (patch != BG))


def _place_patch(
    grid: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    padding: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    r_lo, r_hi = row_range
    c_lo, c_hi = col_range
    candidates = []
    for top in range(r_lo, r_hi - h + 2):
        for left in range(c_lo, c_hi - w + 2):
            if _can_place(occ, patch, top, left, padding):
                candidates.append((top, left))
    if not candidates:
        return None
    top, left = random.choice(candidates)
    mask = patch != BG
    grid[top : top + h, left : left + w][mask] = patch[mask]
    occ[top : top + h, left : left + w][mask] = True
    return top, left


def _component_ids(
    shape: tuple[int, int],
    target_coords: list[list[tuple[int, int]]],
    legend_coords: list[list[tuple[int, int]]],
    parity: tuple[int, int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[str, int]]]:
    neutral: set[tuple[int, int]] = set()
    endpoint_owner: dict[tuple[int, int], tuple[str, int]] = {}
    all_components = [("T", i, set(coords)) for i, coords in enumerate(target_coords)] + [
        ("L", i, set(coords)) for i, coords in enumerate(legend_coords)
    ]
    for r in range(parity[0], shape[0], 2):
        for c in range(parity[1], shape[1], 2):
            if (r, c) in {rc for _, _, coords in all_components for rc in coords}:
                continue
            touched = []
            for kind, idx, coords in all_components:
                if _touches_component((r, c), coords):
                    touched.append((kind, idx))
            if len(touched) == 0:
                neutral.add((r, c))
            elif len(touched) == 1:
                endpoint_owner[(r, c)] = touched[0]
    return neutral, endpoint_owner


def _lattice_neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = node
    return [(r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)]


def _bfs_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parent = {start: None}
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for nb in _lattice_neighbors(node):
            if nb not in allowed or nb in parent or not _in_bounds(shape, nb[0], nb[1]):
                continue
            parent[nb] = node
            queue.append(nb)
    if end not in parent:
        return None
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _forbidden_from_paths(paths: list[list[tuple[int, int]]]) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for path in paths:
        for node in path:
            forbidden.add(node)
            forbidden.update(_lattice_neighbors(node))
    return forbidden


def _route_pair(
    shape: tuple[int, int],
    start_candidates: list[tuple[int, int]],
    end_candidates: list[tuple[int, int]],
    neutral: set[tuple[int, int]],
    existing_paths: list[list[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    blocked = _forbidden_from_paths(existing_paths)
    starts = start_candidates[:]
    ends = end_candidates[:]
    random.shuffle(starts)
    random.shuffle(ends)
    for start in starts[:12]:
        if start in blocked:
            continue
        for end in ends[:12]:
            if end in blocked or end == start:
                continue
            direct_allowed = set(neutral) | {start, end}
            direct_allowed -= blocked
            direct_allowed |= {start, end}

            best = _bfs_path(shape, start, end, direct_allowed)
            if best is None:
                continue

            detour_pool = list(neutral - blocked - set(best))
            random.shuffle(detour_pool)
            for waypoint in detour_pool[:18]:
                allowed = set(neutral) | {start, end, waypoint}
                allowed -= blocked
                allowed |= {start, end, waypoint}
                first = _bfs_path(shape, start, waypoint, allowed)
                if first is None:
                    continue
                second_allowed = allowed - set(first[1:-1])
                second = _bfs_path(shape, waypoint, end, second_allowed)
                if second is None:
                    continue
                detour = first + second[1:]
                if len(detour) >= len(best) + 2:
                    return detour
            return best
    return None


def _pairing(bound_count: int) -> list[int]:
    order = list(range(bound_count))
    random.shuffle(order)
    if bound_count > 1 and order == list(range(bound_count)):
        order = order[1:] + order[:1]
    return order


def _global_recolor_without_paths(grid: np.ndarray) -> np.ndarray:
    comps = _find_foreground_components(grid)
    legends = [legend for comp in comps if (legend := _legend_from_component(grid, comp)) is not None]
    out = grid.copy()
    for legend in legends:
        out[out == legend["source"]] = legend["replacement"]
    out[out == GUIDE] = BG
    for legend in legends:
        for r, c in legend["coords"]:
            out[r, c] = BG
    return out


def _build_case() -> tuple[np.ndarray, dict]:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    bound_count = random.choice([2, 3])
    include_unbound = bound_count == 2 and random.random() < 0.4
    target_count = bound_count + int(include_unbound)
    force_counterexample = random.random() < 0.7

    replacement_colors = random.sample(PALETTE, bound_count)
    usable_target_colors = [c for c in PALETTE if c not in replacement_colors]

    if force_counterexample and target_count >= 2:
        shared_source = random.choice(usable_target_colors)
        source_colors = [shared_source] * target_count
        for idx in range(target_count):
            if random.random() < 0.35:
                source_colors[idx] = random.choice(usable_target_colors)
        if len(set(source_colors)) == target_count:
            source_colors[0] = source_colors[1]
    else:
        source_colors = [random.choice(usable_target_colors) for _ in range(target_count)]

    target_patches = [_make_target_patch(source_colors[i], usable_target_colors) for i in range(target_count)]

    band_height = h // max(3, target_count)
    target_coords: list[list[tuple[int, int]]] = []
    for i, patch in enumerate(target_patches):
        row_start = max(1, i * band_height + 1)
        row_end = min(h - 1, row_start + band_height + 2)
        pos = _place_patch(grid, occ, patch, (row_start, row_end), (1, max(7, w // 3)), padding=2)
        if pos is None:
            raise RuntimeError("target placement failed")
        top, left = pos
        coords = [(top + r, left + c) for r, c in zip(*np.where(patch != BG))]
        target_coords.append(coords)

    legend_specs = []
    pair_to_target = _pairing(bound_count)
    for pair_idx in range(bound_count):
        target_idx = pair_to_target[pair_idx]
        legend_patch = np.array([[replacement_colors[pair_idx], source_colors[target_idx]]], dtype=int)
        row_start = max(1, pair_idx * band_height + 1)
        row_end = min(h - 1, row_start + band_height + 2)
        pos = _place_patch(grid, occ, legend_patch, (row_start, row_end), (max(w * 2 // 3, 10), w - 1), padding=2)
        if pos is None:
            raise RuntimeError("legend placement failed")
        top, left = pos
        legend_coords = [(top, left), (top, left + 1)]
        legend_specs.append(
            {
                "coords": legend_coords,
                "replacement": replacement_colors[pair_idx],
                "source": source_colors[target_idx],
                "target_idx": target_idx,
            }
        )

    parity_choices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(parity_choices)
    paths: list[list[tuple[int, int]]] | None = None
    chosen_parity: tuple[int, int] | None = None
    for parity in parity_choices:
        neutral, endpoint_owner = _component_ids(grid.shape, target_coords, [legend["coords"] for legend in legend_specs], parity)
        path_list: list[list[tuple[int, int]]] = []
        items = list(range(bound_count))
        random.shuffle(items)
        ok = True
        for legend_idx in items:
            target_idx = legend_specs[legend_idx]["target_idx"]
            start_candidates = [cell for cell, owner in endpoint_owner.items() if owner == ("L", legend_idx)]
            end_candidates = [cell for cell, owner in endpoint_owner.items() if owner == ("T", target_idx)]
            path = _route_pair(grid.shape, start_candidates, end_candidates, neutral, path_list)
            if path is None or len(path) < 2:
                ok = False
                break
            path_list.append(path)
        if ok:
            paths = path_list
            chosen_parity = parity
            break
    if paths is None or chosen_parity is None:
        raise RuntimeError("path routing failed")

    for path in paths:
        for r, c in path:
            grid[r, c] = GUIDE

    expected = grid.copy()
    for legend_idx in range(len(paths)):
        target_idx = legend_specs[legend_idx]["target_idx"]
        replacement = legend_specs[legend_idx]["replacement"]
        source = legend_specs[legend_idx]["source"]
        for r, c in target_coords[target_idx]:
            if int(expected[r, c]) == int(source):
                expected[r, c] = replacement
    expected[expected == GUIDE] = BG
    for legend in legend_specs:
        for r, c in legend["coords"]:
            expected[r, c] = BG

    meta = {
        "expected": expected,
        "target_coords": target_coords,
        "legend_specs": legend_specs,
        "bound_count": bound_count,
        "target_count": target_count,
        "force_counterexample": force_counterexample,
        "parity": chosen_parity,
    }
    return grid, meta


def _validate_case(inp: np.ndarray, meta: dict) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, meta["expected"]):
        return False

    for legend in meta["legend_specs"]:
        for r, c in legend["coords"]:
            if int(out[r, c]) != BG:
                return False
    if np.any(out == GUIDE):
        return False

    bound_targets = {legend["target_idx"] for legend in meta["legend_specs"]}
    for idx, coords in enumerate(meta["target_coords"]):
        before = np.array([int(inp[r, c]) for r, c in coords])
        after = np.array([int(out[r, c]) for r, c in coords])
        if idx in bound_targets:
            legend = next(item for item in meta["legend_specs"] if item["target_idx"] == idx)
            expected = before.copy()
            expected[expected == legend["source"]] = legend["replacement"]
            if not np.array_equal(after, expected):
                return False
        else:
            if not np.array_equal(before, after):
                return False

    if meta["force_counterexample"]:
        naive = _global_recolor_without_paths(inp)
        if np.array_equal(naive, out):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, meta = _build_case()
        except RuntimeError:
            continue
        if _validate_case(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid routed-legend recolor input")

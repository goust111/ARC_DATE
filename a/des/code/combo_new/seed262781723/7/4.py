import random

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

# --------------- solver helpers ---------------


def _parse_shape(lines: list[str]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r, c)
        for r, row in enumerate(lines)
        for c, ch in enumerate(row)
        if ch == "#"
    )


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _shift(
    coords: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in coords))


def _bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = tuple(sorted(cells))
            comp_h, comp_w = _dims(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "height": comp_h,
                    "width": comp_w,
                    "horizontal": comp_w >= comp_h,
                }
            )

    return comps


def _adjacency_sets(grid: np.ndarray, comps: list[dict]) -> list[set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels = np.full((h, w), -1, dtype=int)
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            labels[r, c] = idx

    adj = [set() for _ in comps]
    for idx, comp in enumerate(comps):
        for r, c in comp["coords"]:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(labels[nr, nc])
                    if other != -1 and other != idx:
                        adj[idx].add(other)
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each colored object along its longer axis once per 4-neighboring object.

    Input structure:
    - The background is black.
    - The foreground consists of several small 4-connected monochrome objects, each in a unique color.
    - Different objects may or may not touch across 4-neighbor edges, and enough empty space remains to the right or below for all replicas.

    Rule:
    - Identify each monochrome 4-connected object and count how many other objects are 4-neighbor adjacent to it.
    - If an object's bounding-box width is at least its height, treat it as horizontal and copy it to the right; otherwise treat it as vertical and copy it downward.
    - Use the object's own bounding-box width or height as the translation step, and place exactly that many full replicas with no gap between consecutive bounding boxes.
    - Keep the original objects unchanged; objects with zero neighbors produce no replicas.

    Key insight:
    - The copy count comes from the global object-adjacency graph, while the copy direction comes from each object's local geometry, so both relational and geometric reasoning are required.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g)
    adj = _adjacency_sets(g, comps)
    out = g.copy()

    for idx, comp in enumerate(comps):
        step_r, step_c = (0, comp["width"]) if comp["horizontal"] else (comp["height"], 0)
        color = int(comp["color"])

        for k in range(1, len(adj[idx]) + 1):
            for r, c in comp["coords"]:
                nr = r + k * step_r
                nc = c + k * step_c
                assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1], "replica exceeds grid bounds"
                assert int(out[nr, nc]) in (BG, color), "replicas must land in reserved empty space"
                out[nr, nc] = color

    return out


# --------------- generator ---------------

HORIZONTAL_SHAPES = tuple(
    _parse_shape(lines)
    for lines in (
        ["##"],
        ["###"],
        ["####"],
        ["#.", "##"],
        [".#.", "###"],
        ["##.", ".##"],
        ["##", "##"],
        ["###", "##."],
        ["###", "###"],
    )
)
VERTICAL_SHAPES = tuple(
    _parse_shape(lines)
    for lines in (
        ["#", "#"],
        ["#", "#", "#"],
        ["#", "#", "#", "#"],
        ["#.", "##", "#."],
        [".#", "##", ".#"],
        ["##", "##", "#."],
        ["##", "##", "##"],
    )
)
HORIZONTAL_BARS = tuple(_parse_shape(lines) for lines in (["##"], ["###"], ["####"]))
VERTICAL_BARS = tuple(_parse_shape(lines) for lines in (["#", "#"], ["#", "#", "#"], ["#", "#", "#", "#"]))

MODULE_OBJECT_COUNT = {
    "isolated_h": 1,
    "isolated_v": 1,
    "pair_h": 2,
    "pair_v": 2,
    "mixed_pair": 2,
    "chain_h": 3,
    "chain_v": 3,
}

PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _random_flip(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    if random.random() < 0.5:
        pts = [(max_r - r, c) for r, c in pts]
    if random.random() < 0.5:
        pts = [(r, max_c - c) for r, c in pts]
    return _normalize(tuple(pts))


def _pick_shape(bank: tuple[tuple[tuple[int, int], ...], ...]) -> tuple[tuple[int, int], ...]:
    return _random_flip(random.choice(bank))


def _module_output_cells(
    objects: list[tuple[tuple[int, int], ...]],
) -> tuple[set[tuple[int, int]], list[dict]]:
    analyses: list[dict] = []
    object_sets = [set(obj) for obj in objects]

    for idx, coords in enumerate(objects):
        comp_h, comp_w = _dims(coords)
        adjacent = set()
        for other_idx, other in enumerate(object_sets):
            if idx == other_idx:
                continue
            if any((r + dr, c + dc) in other for r, c in coords for dr, dc in NEIGHBORS4):
                adjacent.add(other_idx)
        analyses.append(
            {
                "coords": coords,
                "height": comp_h,
                "width": comp_w,
                "horizontal": comp_w >= comp_h,
                "count": len(adjacent),
            }
        )

    cells: set[tuple[int, int]] = set()
    painted_by: dict[tuple[int, int], int] = {}
    for idx, info in enumerate(analyses):
        step_r, step_c = (0, info["width"]) if info["horizontal"] else (info["height"], 0)
        for k in range(info["count"] + 1):
            for r, c in info["coords"]:
                cell = (r + k * step_r, c + k * step_c)
                if cell in painted_by and painted_by[cell] != idx:
                    raise RuntimeError("module output collision")
                painted_by[cell] = idx
                cells.add(cell)

    return cells, analyses


def _build_module(objects: list[tuple[tuple[int, int], ...]], kind: str) -> dict:
    all_cells = [cell for obj in objects for cell in obj]
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    normalized = [
        tuple(sorted((r - min_r, c - min_c) for r, c in obj))
        for obj in objects
    ]
    input_cells = {cell for obj in normalized for cell in obj}
    output_cells, analyses = _module_output_cells(normalized)
    max_r = max(r for r, _ in output_cells)
    max_c = max(c for _, c in output_cells)

    return {
        "kind": kind,
        "objects": normalized,
        "input_cells": input_cells,
        "output_cells": output_cells,
        "height": max_r + 1,
        "width": max_c + 1,
        "analyses": analyses,
    }


def _make_module(kind: str) -> dict:
    if kind == "isolated_h":
        return _build_module([_pick_shape(HORIZONTAL_SHAPES)], kind)
    if kind == "isolated_v":
        return _build_module([_pick_shape(VERTICAL_SHAPES)], kind)
    if kind == "pair_h":
        a = _pick_shape(HORIZONTAL_BARS)
        b = _pick_shape(HORIZONTAL_BARS)
        w_a = _dims(a)[1]
        w_b = _dims(b)[1]
        col_b = random.randint(-(w_b - 1), w_a - 1)
        return _build_module([_shift(a, 0, 0), _shift(b, 1, col_b)], kind)
    if kind == "chain_h":
        a = _pick_shape(HORIZONTAL_BARS)
        b = _pick_shape(HORIZONTAL_BARS)
        c = _pick_shape(HORIZONTAL_BARS)
        w_a = _dims(a)[1]
        w_b = _dims(b)[1]
        w_c = _dims(c)[1]
        col_b = random.randint(-(w_b - 1), w_a - 1)
        col_c = col_b + random.randint(-(w_c - 1), w_b - 1)
        return _build_module(
            [_shift(a, 0, 0), _shift(b, 1, col_b), _shift(c, 2, col_c)],
            kind,
        )
    if kind == "pair_v":
        a = _pick_shape(VERTICAL_BARS)
        b = _pick_shape(VERTICAL_BARS)
        h_a = _dims(a)[0]
        h_b = _dims(b)[0]
        row_b = random.randint(-(h_b - 1), h_a - 1)
        return _build_module([_shift(a, 0, 0), _shift(b, row_b, 1)], kind)
    if kind == "chain_v":
        a = _pick_shape(VERTICAL_BARS)
        b = _pick_shape(VERTICAL_BARS)
        c = _pick_shape(VERTICAL_BARS)
        h_a = _dims(a)[0]
        h_b = _dims(b)[0]
        h_c = _dims(c)[0]
        row_b = random.randint(-(h_b - 1), h_a - 1)
        row_c = row_b + random.randint(-(h_c - 1), h_b - 1)
        return _build_module(
            [_shift(a, 0, 0), _shift(b, row_b, 1), _shift(c, row_c, 2)],
            kind,
        )
    if kind == "mixed_pair":
        h_obj = _pick_shape(HORIZONTAL_SHAPES)
        v_obj = _pick_shape(VERTICAL_SHAPES)
        h_height = _dims(h_obj)[0]
        h_cols = [c for r, c in h_obj if r == h_height - 1]
        v_cols = [c for r, c in v_obj if r == 0]
        h_col = random.choice(h_cols)
        v_col = random.choice(v_cols)
        return _build_module([_shift(h_obj, 0, 0), _shift(v_obj, h_height, h_col - v_col)], kind)
    raise ValueError(f"unknown module kind: {kind}")


def _module_orientations(module: dict) -> set[str]:
    return {"h" if info["horizontal"] else "v" for info in module["analyses"]}


def _module_has_degree(module: dict, degree: int) -> bool:
    return any(info["count"] == degree for info in module["analyses"])


def _choose_modules() -> list[dict]:
    adjacency_kind = random.choice(["pair_h", "pair_v", "mixed_pair", "chain_h", "chain_v"])
    modules = [_make_module(adjacency_kind)]

    seen_orientations = set().union(*(_module_orientations(module) for module in modules))
    if "h" not in seen_orientations:
        modules.append(_make_module("isolated_h"))
    elif "v" not in seen_orientations:
        modules.append(_make_module("isolated_v"))
    else:
        modules.append(_make_module(random.choice(["isolated_h", "isolated_v"])))

    total_objects = sum(len(module["objects"]) for module in modules)
    if random.random() < 0.7:
        options: list[str] = []
        if total_objects + 1 <= 6:
            options += ["isolated_h", "isolated_v"]
        if total_objects + 2 <= 6:
            options += ["pair_h", "pair_v", "mixed_pair"]
        if total_objects + 3 <= 6:
            options += ["chain_h", "chain_v"]

        if options:
            if not any(_module_has_degree(module, 2) for module in modules):
                chain_options = [kind for kind in options if kind.startswith("chain")]
                if chain_options and random.random() < 0.75:
                    options = chain_options
            modules.append(_make_module(random.choice(options)))

    return modules


def _expand(
    cells: set[tuple[int, int]],
    deltas: tuple[tuple[int, int], ...],
) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr, dc in deltas:
            out.add((r + dr, c + dc))
    return out


def _place_modules(modules: list[dict]) -> np.ndarray | None:
    modules = sorted(
        modules,
        key=lambda module: (len(module["output_cells"]), module["height"] * module["width"]),
        reverse=True,
    )
    max_h = max(module["height"] for module in modules)
    max_w = max(module["width"] for module in modules)
    height = random.randint(max(max_h + 2, 12), 30)
    width = random.randint(max(max_w + 2, 12), 30)

    input_buffer: set[tuple[int, int]] = set()
    output_buffer: set[tuple[int, int]] = set()
    placements: list[tuple[dict, int, int]] = []

    for module in modules:
        row_choices = list(range(0, height - module["height"] + 1))
        col_choices = list(range(0, width - module["width"] + 1))
        placed = False

        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.choice(row_choices)
            c0 = random.choice(col_choices)
            input_cells = {(r0 + r, c0 + c) for r, c in module["input_cells"]}
            output_cells = {(r0 + r, c0 + c) for r, c in module["output_cells"]}

            if any(cell in input_buffer for cell in input_cells):
                continue
            if any(cell in output_buffer for cell in output_cells):
                continue

            placements.append((module, r0, c0))
            input_buffer |= _expand(input_cells, NEIGHBORS8)
            output_buffer |= _expand(output_cells, NEIGHBORS8)
            placed = True
            break

        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    object_count = sum(len(module["objects"]) for module, _, _ in placements)
    colors = random.sample(range(1, 10), object_count)

    color_idx = 0
    for module, r0, c0 in placements:
        for obj in module["objects"]:
            color = colors[color_idx]
            color_idx += 1
            for r, c in obj:
                grid[r0 + r, c0 + c] = color

    return grid


def _input_features(grid: np.ndarray) -> list[tuple[bool, int]]:
    comps = _components(grid)
    adj = _adjacency_sets(grid, comps)
    return [(bool(comp["horizontal"]), len(adj[idx])) for idx, comp in enumerate(comps)]


def _is_valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    features = _input_features(grid)
    has_horizontal = any(horizontal for horizontal, _ in features)
    has_vertical = any(not horizontal for horizontal, _ in features)
    has_zero = any(count == 0 for _, count in features)
    has_positive = any(count > 0 for _, count in features)
    return has_horizontal and has_vertical and has_zero and has_positive


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        modules = _choose_modules()
        grid = _place_modules(modules)
        if grid is None:
            continue
        if _is_valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")

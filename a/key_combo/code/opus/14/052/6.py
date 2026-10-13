import numpy as np


BG = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3, 4)
MAX_GEN_ATTEMPTS = 500

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

MERGE_PATTERNS = [
    ["S.XX", "..X.", "XXX."],
    ["S.X.", "..XX", "XXX."],
    ["S.X.", "..X.", "XXXX"],
    ["S.X", "..X", "XXX", "X.."],
    ["S.X", "..X", "XXX", ".X."],
    ["S.X", "..X", "XXX", "..X"],
    ["..X", "S.X", "..X", "XXX"],
    [".S.X", "...X", "XXXX"],
]

EDGE_MERGE_PATTERNS = [
    ["S.XX", "..XX", ".XXX"],
]

STATIC_PATTERNS = [
    ["XX", "X."],
    ["XXX", ".X."],
    ["XX", ".X", ".X"],
    ["XXX", "X.."],
    ["XX", "XX"],
    ["XX.", ".XX"],
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in N8:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_mask(shape: tuple[int, int], comp: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return mask


def _full_neighborhood_inside(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    if r == 0 or c == 0 or r == h - 1 or c == w - 1:
        return False
    return bool(mask[r - 1 : r + 2, c - 1 : c + 2].all())


def _interior_pixels(shape: tuple[int, int], comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mask = _component_mask(shape, comp)
    return [(r, c) for r, c in comp if _full_neighborhood_inside(mask, r, c)]


def _find_isolated_gray_singletons(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GRAY:
                continue
            if all(
                not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == BG
                for nr, nc in _neighbors8(r, c)
            ):
                out.append((r, c))
    return out


def _expand_isolated_singletons(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in _find_isolated_gray_singletons(out):
        for dr in (0, 1):
            for dc in (0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
                    out[nr, nc] = GRAY
    return out


def _marker_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, tuple[int, int]] | None:
    marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) not in (BG, GRAY)]
    if not marker_cells:
        return None
    colors = {int(grid[r, c]) for r, c in marker_cells}
    assert len(colors) == 1, "each marked component must use exactly one marker color"
    assert len(marker_cells) == 1, "each marked component must contain exactly one marker pixel"
    return int(grid[marker_cells[0]]), marker_cells[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    expanded = _expand_isolated_singletons(grid)
    out = expanded.copy()
    for comp in _components(expanded != BG):
        marker = _marker_info(expanded, comp)
        if marker is None:
            continue
        marker_color, _marker_pos = marker
        for r, c in _interior_pixels(expanded.shape, comp):
            out[r, c] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand isolated gray pixels, then recolor the new component interiors by their marker color.

    Input structure:
    - Black background contains gray isolated single pixels and several small gray 4-connected clusters.
    - Each non-singleton cluster contains exactly one non-gray marker pixel, and different clusters use different marker colors.
    - Some isolated gray pixels are far from every marker cluster, while others are positioned so their 2x2 expansion changes later connectivity.

    Rule:
    - Find every gray pixel whose 8 neighbors are all background.
    - Expand each such isolated pixel into a 2x2 gray block anchored at that pixel, only writing onto background cells.
    - Recompute 4-connected non-background components after all expansions.
    - In every component that contains a unique marker, recolor each pixel whose full 3x3 neighborhood stays inside that component to the marker color.
    - Keep marker pixels unchanged, keep all other component pixels gray, and leave standalone expanded 2x2 gray blocks gray.

    Key insight:
    - The isolated-pixel expansion matters because it can merge shapes that were previously separate; only after that topology change do solid 3x3-supported interior pixels appear and inherit the component's marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _parse_pattern(rows: list[str]) -> dict:
    singleton = None
    cluster: list[tuple[int, int]] = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "S":
                singleton = (r, c)
            elif ch == "X":
                cluster.append((r, c))
    assert singleton is not None
    footprint = set(cluster)
    sr, sc = singleton
    for dr in (0, 1):
        for dc in (0, 1):
            footprint.add((sr + dr, sc + dc))
    h = max(r for r, _ in footprint | set(cluster) | {singleton}) + 1
    w = max(c for _, c in footprint | set(cluster) | {singleton}) + 1
    interior = _interior_pixels((h, w), sorted(footprint))
    assert interior, "merge prototype must create an interior after expansion"
    return {
        "kind": "merge",
        "singleton": singleton,
        "cluster": cluster,
        "height": h,
        "width": w,
        "future": sorted(footprint),
        "interior_after": interior,
    }


def _parse_static(rows: list[str]) -> dict:
    cells: list[tuple[int, int]] = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "X":
                cells.append((r, c))
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    assert not _interior_pixels((h, w), cells), "static prototype should not have an interior"
    return {
        "kind": "static",
        "cluster": cells,
        "height": h,
        "width": w,
        "future": list(cells),
    }


MERGE_OBJECTS = [_parse_pattern(rows) for rows in MERGE_PATTERNS]
EDGE_MERGE_OBJECTS = [_parse_pattern(rows) for rows in EDGE_MERGE_PATTERNS]
STATIC_OBJECTS = [_parse_static(rows) for rows in STATIC_PATTERNS]


def _degree_within(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in cells for dr, dc in N4)


def _choose_marker_cell(
    cells: list[tuple[int, int]],
    rng: np.random.Generator,
    banned: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    cell_set = set(cells)
    banned = banned or set()
    eligible = [cell for cell in cells if cell not in banned]
    robust = [cell for cell in eligible if _degree_within(cell_set, cell) >= 2]
    if not eligible:
        eligible = list(cells)
        robust = [cell for cell in eligible if _degree_within(cell_set, cell) >= 2]
    pool = robust or eligible
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]


def _make_merge_object(proto: dict, marker_color: int, rng: np.random.Generator) -> dict:
    marker = _choose_marker_cell(proto["cluster"], rng, set(proto["interior_after"]))
    colors = {cell: GRAY for cell in proto["cluster"]}
    colors[marker] = marker_color
    colors[proto["singleton"]] = GRAY
    return {
        "kind": "merge",
        "height": proto["height"],
        "width": proto["width"],
        "input_cells": colors,
        "future_cells": set(proto["future"]),
        "singleton": proto["singleton"],
        "marker_local": marker,
        "future_interiors": list(proto["interior_after"]),
    }


def _make_static_object(proto: dict, marker_color: int, rng: np.random.Generator) -> dict:
    marker = _choose_marker_cell(proto["cluster"], rng)
    colors = {cell: GRAY for cell in proto["cluster"]}
    colors[marker] = marker_color
    return {
        "kind": "static",
        "height": proto["height"],
        "width": proto["width"],
        "input_cells": colors,
        "future_cells": set(proto["future"]),
        "marker_local": marker,
    }


def _make_far_singleton() -> dict:
    return {
        "kind": "far_singleton",
        "height": 2,
        "width": 2,
        "input_cells": {(0, 0): GRAY},
        "future_cells": {(0, 0), (0, 1), (1, 0), (1, 1)},
        "singleton": (0, 0),
    }


def _reserve_halo(reserved: np.ndarray, cells: set[tuple[int, int]], top: int, left: int) -> None:
    for r, c in cells:
        gr = top + r
        gc = left + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = gr + dr
                nc = gc + dc
                if _in_bounds(reserved.shape, nr, nc):
                    reserved[nr, nc] = True


def _place_objects(
    shape: tuple[int, int],
    objects: list[dict],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict] | None:
    h, w = shape
    grid = np.zeros(shape, dtype=int)
    reserved = np.zeros(shape, dtype=bool)
    placed_meta = {
        "merge_singletons": [],
        "far_singletons": [],
        "markers": [],
        "merge_markers": [],
        "static_markers": [],
    }

    ordering = sorted(
        range(len(objects)),
        key=lambda i: (
            -len(objects[i]["future_cells"]),
            -(objects[i]["height"] * objects[i]["width"]),
            objects[i]["kind"],
        ),
    )

    for idx in ordering:
        obj = objects[idx]
        candidates: list[tuple[int, int]] = []
        for top in range(h - obj["height"] + 1):
            for left in range(w - obj["width"] + 1):
                if any(reserved[top + r, left + c] for r, c in obj["future_cells"]):
                    continue
                if any(grid[top + r, left + c] != BG for r, c in obj["input_cells"]):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = candidates[int(rng.integers(0, len(candidates)))]
        for (r, c), color in obj["input_cells"].items():
            grid[top + r, left + c] = color
        _reserve_halo(reserved, obj["future_cells"], top, left)

        if obj["kind"] == "merge":
            sr, sc = obj["singleton"]
            mr, mc = obj["marker_local"]
            placed_meta["merge_singletons"].append((top + sr, left + sc))
            placed_meta["merge_markers"].append((top + mr, left + mc))
            placed_meta["markers"].append((top + mr, left + mc))
        elif obj["kind"] == "static":
            mr, mc = obj["marker_local"]
            placed_meta["static_markers"].append((top + mr, left + mc))
            placed_meta["markers"].append((top + mr, left + mc))
        else:
            sr, sc = obj["singleton"]
            placed_meta["far_singletons"].append((top + sr, left + sc))

    return grid, placed_meta


def _marker_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for comp in _components(grid != BG):
        marker = _marker_info(grid, comp)
        if marker is None:
            continue
        comps.append({"coords": comp, "color": marker[0], "marker_pos": marker[1]})
    return comps


def _bbox_distance_to_component(cell: tuple[int, int], comp: list[tuple[int, int]]) -> int:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    r, c = cell
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(dr, dc)


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(int(v) for v in np.unique(grid)) - {BG, GRAY, *MARKER_COLORS}:
        return False

    isolated = set(_find_isolated_gray_singletons(grid))
    expected_isolated = set(meta["merge_singletons"]) | set(meta["far_singletons"])
    if isolated != expected_isolated:
        return False
    if not meta["merge_singletons"] or not meta["far_singletons"]:
        return False

    marker_comps = _marker_components(grid)
    if len(marker_comps) < 2:
        return False
    marker_colors = [entry["color"] for entry in marker_comps]
    if len(set(marker_colors)) != len(marker_colors):
        return False
    if any(_interior_pixels(grid.shape, entry["coords"]) for entry in marker_comps):
        return False

    # At least one isolated singleton starts one cell from some cluster bbox, even if
    # the recolor-causing prototype itself may sit inside a looser bbox gap.
    near_bbox = False
    for single in meta["far_singletons"] + meta["merge_singletons"]:
        for entry in marker_comps:
            if _bbox_distance_to_component(single, entry["coords"]) == 1:
                near_bbox = True
                break
        if near_bbox:
            break
    if not near_bbox:
        return False

    expanded = _expand_isolated_singletons(grid)
    expanded_marker_comps = _marker_components(expanded)
    if len(expanded_marker_comps) != len(marker_comps):
        return False

    comps_by_marker = {entry["marker_pos"]: entry for entry in expanded_marker_comps}
    merged_with_interior = 0
    untouched_without_interior = 0

    for marker_pos in meta["merge_markers"]:
        entry = comps_by_marker.get(marker_pos)
        if entry is None:
            return False
        interiors = _interior_pixels(expanded.shape, entry["coords"])
        if not interiors:
            return False
        merged_with_interior += 1

    for marker_pos in meta["static_markers"]:
        entry = comps_by_marker.get(marker_pos)
        if entry is None:
            return False
        interiors = _interior_pixels(expanded.shape, entry["coords"])
        if interiors:
            return False
        untouched_without_interior += 1

    for single in meta["far_singletons"]:
        block = {
            single,
            (single[0], single[1] + 1),
            (single[0] + 1, single[1]),
            (single[0] + 1, single[1] + 1),
        }
        match = None
        for comp in _components(expanded != BG):
            comp_set = set(comp)
            if single in comp_set:
                match = comp_set
                break
        if match != block:
            return False

    out = _solve(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out != expanded) < merged_with_interior:
        return False
    if not merged_with_interior or not untouched_without_interior:
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(12, 19))
    w = int(rng.integers(12, 19))

    cluster_count = int(rng.integers(2, 4))
    merge_count = int(rng.integers(1, cluster_count))
    static_count = cluster_count - merge_count
    far_singletons = int(rng.integers(1, 4))

    marker_colors = list(rng.choice(np.array(MARKER_COLORS), size=cluster_count, replace=False))

    objects: list[dict] = []
    for i in range(merge_count):
        if i == 0:
            proto = EDGE_MERGE_OBJECTS[int(rng.integers(0, len(EDGE_MERGE_OBJECTS)))]
        else:
            proto = MERGE_OBJECTS[int(rng.integers(0, len(MERGE_OBJECTS)))]
        objects.append(_make_merge_object(proto, int(marker_colors[i]), rng))
    for i in range(static_count):
        proto = STATIC_OBJECTS[int(rng.integers(0, len(STATIC_OBJECTS)))]
        objects.append(_make_static_object(proto, int(marker_colors[merge_count + i]), rng))
    for _ in range(far_singletons):
        objects.append(_make_far_singleton())

    placed = _place_objects((h, w), objects, rng)
    if placed is None:
        return None
    grid, meta = placed
    if not _validate_generated(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

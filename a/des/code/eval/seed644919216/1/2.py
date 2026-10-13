from __future__ import annotations

import numpy as np
from numpy.random import default_rng


# Canonical arrow templates. The right-facing sprite uses this visible shape:
# rows 1/4/1 with the tip on the right and a two-pixel tail on the middle row.
ARROW_SPECS = {
    "R": {
        "coords": ((0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)),
        "tip": (1, 3),
        "step": (0, 1),
        "size": (3, 4),
    },
    "L": {
        "coords": ((0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1)),
        "tip": (1, 0),
        "step": (0, -1),
        "size": (3, 4),
    },
    "U": {
        "coords": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)),
        "tip": (0, 1),
        "step": (-1, 0),
        "size": (4, 3),
    },
    "D": {
        "coords": ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)),
        "tip": (3, 1),
        "step": (1, 0),
        "size": (4, 3),
    },
}
ARROW_COORD_SETS = {name: frozenset(spec["coords"]) for name, spec in ARROW_SPECS.items()}


GEN_ARROW_COLOR = 5
GEN_SIZE_MIN = 16
GEN_SIZE_MAX = 22
GEN_STRIPE_COUNT_MIN = 2
GEN_STRIPE_COUNT_MAX = 3
GEN_FILLER_MAX = 2
GEN_RETRIES = 500
GEN_MIN_CHANGED = 4
GEN_MIN_LINE_GAP = 3


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _build_arrow(direction: str, top: int, left: int) -> dict[str, object]:
    spec = ARROW_SPECS[direction]
    coords = tuple((top + r, left + c) for r, c in spec["coords"])
    tip = (top + spec["tip"][0], left + spec["tip"][1])
    r0, c0 = top, left
    h, w = spec["size"]
    return {
        "direction": direction,
        "coords": coords,
        "coord_set": frozenset(coords),
        "tip": tip,
        "step": spec["step"],
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
    }


def _identify_arrow(coords: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    normalized = frozenset((r - r0, c - c0) for r, c in coords)

    for direction, coord_set in ARROW_COORD_SETS.items():
        if normalized == coord_set:
            return _build_arrow(direction, r0, c0)

    raise AssertionError("unrecognized arrow shape")


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _most_common_color(g)
    row_stripes = {r: int(g[r, 0]) for r in range(h) if int(g[r, 0]) != bg and np.all(g[r, :] == g[r, 0])}
    col_stripes = {c: int(g[0, c]) for c in range(w) if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])}
    assert row_stripes or col_stripes, "expected at least one full stripe"
    assert not (row_stripes and col_stripes), "row and column stripes would cross"

    stripe_mask = np.zeros_like(g, dtype=bool)
    stripe_axis = "horizontal" if row_stripes else "vertical"
    if row_stripes:
        for r in row_stripes:
            stripe_mask[r, :] = True
        stripe_colors = list(row_stripes.values())
        stripe_rows = set(row_stripes)
        stripe_cols: set[int] = set()
    else:
        for c in col_stripes:
            stripe_mask[:, c] = True
        stripe_colors = list(col_stripes.values())
        stripe_rows = set()
        stripe_cols = set(col_stripes)

    assert len(set(stripe_colors)) == len(stripe_colors), "stripe colors must be distinct"

    arrow_mask = (g != bg) & ~stripe_mask
    assert np.any(arrow_mask), "expected at least one arrow"
    arrow_colors = np.unique(g[arrow_mask])
    assert len(arrow_colors) == 1, "all arrows must share one color"
    arrow_color = int(arrow_colors[0])
    assert arrow_color not in stripe_colors, "arrow color must differ from stripe colors"

    components = _find_components(arrow_mask)
    arrows = [_identify_arrow(component) for component in components]
    assert len(arrows) == len(components), "each arrow component must match a template"

    return {
        "grid": g,
        "bg": bg,
        "arrow_color": arrow_color,
        "stripe_axis": stripe_axis,
        "stripe_mask": stripe_mask,
        "stripe_rows": stripe_rows,
        "stripe_cols": stripe_cols,
        "arrow_mask": arrow_mask,
        "arrows": arrows,
    }


def _trace_arrow(
    scene: dict[str, object],
    arrow: dict[str, object],
    occupied_mask: np.ndarray | None = None,
) -> dict[str, object]:
    g = np.asarray(scene["grid"], dtype=int)
    arrow_mask = np.asarray(scene["arrow_mask"], dtype=bool) if occupied_mask is None else np.asarray(occupied_mask, dtype=bool)
    stripe_mask = np.asarray(scene["stripe_mask"], dtype=bool)
    current_color = int(scene["arrow_color"])
    r, c = arrow["tip"]
    dr, dc = arrow["step"]
    path_cells: list[tuple[int, int]] = []
    crossings = 0
    painted = 0

    while True:
        r += dr
        c += dc
        if not _in_bounds(g.shape, r, c):
            return {
                "stop": "boundary",
                "crossings": crossings,
                "painted": painted,
                "path_cells": tuple(path_cells),
                "final_color": current_color,
            }
        if arrow_mask[r, c]:
            return {
                "stop": "arrow",
                "crossings": crossings,
                "painted": painted,
                "path_cells": tuple(path_cells),
                "final_color": current_color,
            }

        path_cells.append((r, c))
        if stripe_mask[r, c]:
            current_color = int(g[r, c])
            crossings += 1
        else:
            assert int(g[r, c]) == int(scene["bg"]), "ray may only pass through background or stripes"
            painted += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a ray from each grey arrow tip, recoloring background cells with the
    latest stripe color seen along that ray.
    """
    scene = _parse_scene(grid)
    g = np.asarray(scene["grid"], dtype=int)
    arrow_mask = np.asarray(scene["arrow_mask"], dtype=bool)
    stripe_mask = np.asarray(scene["stripe_mask"], dtype=bool)
    current_arrow_color = int(scene["arrow_color"])
    out = g.copy()

    for arrow in scene["arrows"]:
        current_color = current_arrow_color
        r, c = arrow["tip"]
        dr, dc = arrow["step"]
        while True:
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c) or arrow_mask[r, c]:
                break
            if stripe_mask[r, c]:
                current_color = int(g[r, c])
                continue
            assert int(g[r, c]) == int(scene["bg"]), "ray may only paint background cells"
            out[r, c] = current_color

    return out


def _sample_line_positions(length: int, count: int, rng: np.random.Generator) -> list[int] | None:
    candidates = list(range(1, length - 1))
    rng.shuffle(candidates)
    selected: list[int] = []
    for pos in candidates:
        if all(abs(pos - other) >= GEN_MIN_LINE_GAP for other in selected):
            selected.append(pos)
            if len(selected) == count:
                return sorted(selected)
    return None


def _mark_reserved(reserved: np.ndarray, arrow: dict[str, object]) -> None:
    r0, c0, r1, c1 = arrow["bbox"]
    reserved[max(0, r0 - 1) : min(reserved.shape[0], r1 + 2), max(0, c0 - 1) : min(reserved.shape[1], c1 + 2)] = True


def _candidate_positions(
    grid: np.ndarray,
    reserved: np.ndarray,
    protected: np.ndarray,
    bg: int,
    direction: str,
) -> list[tuple[int, int]]:
    h, w = ARROW_SPECS[direction]["size"]
    out: list[tuple[int, int]] = []
    for top in range(grid.shape[0] - h + 1):
        for left in range(grid.shape[1] - w + 1):
            arrow = _build_arrow(direction, top, left)
            cells = arrow["coords"]
            if any(grid[r, c] != bg or reserved[r, c] or protected[r, c] for r, c in cells):
                continue
            tip_r, tip_c = arrow["tip"]
            dr, dc = arrow["step"]
            if not _in_bounds(grid.shape, tip_r + dr, tip_c + dc):
                continue
            out.append((top, left))
    return out


def _place_arrow(
    grid: np.ndarray,
    reserved: np.ndarray,
    protected: np.ndarray,
    bg: int,
    direction: str,
    top: int,
    left: int,
) -> dict[str, object]:
    arrow = _build_arrow(direction, top, left)
    for r, c in arrow["coords"]:
        assert grid[r, c] == bg and not reserved[r, c] and not protected[r, c]
        grid[r, c] = GEN_ARROW_COLOR
    _mark_reserved(reserved, arrow)
    return arrow


def _trace_generated_arrow(
    arrow: dict[str, object],
    occupied_mask: np.ndarray,
    stripe_axis: str,
    stripe_positions: set[int],
    shape: tuple[int, int],
) -> dict[str, object]:
    r, c = arrow["tip"]
    dr, dc = arrow["step"]
    path_cells: list[tuple[int, int]] = []
    crossings = 0
    painted = 0

    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return {"stop": "boundary", "crossings": crossings, "painted": painted, "path_cells": tuple(path_cells)}
        if occupied_mask[r, c]:
            return {"stop": "arrow", "crossings": crossings, "painted": painted, "path_cells": tuple(path_cells)}

        path_cells.append((r, c))
        if (stripe_axis == "horizontal" and r in stripe_positions) or (stripe_axis == "vertical" and c in stripe_positions):
            crossings += 1
        else:
            painted += 1


def _place_with_predicate(
    grid: np.ndarray,
    reserved: np.ndarray,
    protected: np.ndarray,
    bg: int,
    stripe_axis: str,
    stripe_positions: set[int],
    directions: list[str],
    predicate,
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object]] | None:
    choices: list[tuple[str, int, int]] = []
    shuffled_dirs = list(directions)
    rng.shuffle(shuffled_dirs)
    for direction in shuffled_dirs:
        for top, left in _candidate_positions(grid, reserved, protected, bg, direction):
            choices.append((direction, top, left))
    if not choices:
        return None

    order = rng.permutation(len(choices))
    for idx in order:
        direction, top, left = choices[int(idx)]
        arrow = _build_arrow(direction, top, left)
        occupied = grid == GEN_ARROW_COLOR
        temp_occupied = occupied.copy()
        for r, c in arrow["coords"]:
            temp_occupied[r, c] = True
        info = _trace_generated_arrow(arrow, temp_occupied, stripe_axis, stripe_positions, grid.shape)
        if predicate(info):
            placed = _place_arrow(grid, reserved, protected, bg, direction, top, left)
            final_occupied = grid == GEN_ARROW_COLOR
            final_info = _trace_generated_arrow(placed, final_occupied, stripe_axis, stripe_positions, grid.shape)
            assert predicate(final_info), "placed arrow must preserve its intended behavior"
            return placed, final_info
    return None


def _place_blocker_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    protected: np.ndarray,
    bg: int,
    stripe_axis: str,
    stripe_positions: set[int],
    rng: np.random.Generator,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]] | None:
    pair_options: list[tuple[str, int, int, str, int, int, dict[str, object]]] = []
    shooter_dirs = ["L", "R"]
    blocker_dirs = ["L", "R"]

    for shooter_dir in shooter_dirs:
        shooter_positions = _candidate_positions(grid, reserved, protected, bg, shooter_dir)
        for shooter_top, shooter_left in shooter_positions:
            shooter = _build_arrow(shooter_dir, shooter_top, shooter_left)
            tmp_grid = grid.copy()
            for r, c in shooter["coords"]:
                tmp_grid[r, c] = GEN_ARROW_COLOR
            tmp_reserved = reserved.copy()
            _mark_reserved(tmp_reserved, shooter)
            tmp_occupied = tmp_grid == GEN_ARROW_COLOR

            for blocker_dir in blocker_dirs:
                blocker_positions = _candidate_positions(tmp_grid, tmp_reserved, protected, bg, blocker_dir)
                for blocker_top, blocker_left in blocker_positions:
                    blocker = _build_arrow(blocker_dir, blocker_top, blocker_left)
                    pair_occupied = tmp_occupied.copy()
                    for r, c in blocker["coords"]:
                        pair_occupied[r, c] = True
                    info = _trace_generated_arrow(shooter, pair_occupied, stripe_axis, stripe_positions, grid.shape)
                    if info["stop"] == "arrow" and info["painted"] >= 1:
                        pair_options.append(
                            (
                                shooter_dir,
                                shooter_top,
                                shooter_left,
                                blocker_dir,
                                blocker_top,
                                blocker_left,
                                info,
                            )
                        )

    if not pair_options:
        return None

    choice = pair_options[int(rng.integers(len(pair_options)))]
    shooter_dir, shooter_top, shooter_left, blocker_dir, blocker_top, blocker_left, _ = choice
    shooter = _place_arrow(grid, reserved, protected, bg, shooter_dir, shooter_top, shooter_left)
    blocker = _place_arrow(grid, reserved, protected, bg, blocker_dir, blocker_top, blocker_left)
    occupied = grid == GEN_ARROW_COLOR
    info = _trace_generated_arrow(shooter, occupied, stripe_axis, stripe_positions, grid.shape)
    assert info["stop"] == "arrow" and info["painted"] >= 1
    return shooter, blocker, info


def _place_filler(
    grid: np.ndarray,
    reserved: np.ndarray,
    protected: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> bool:
    directions = list(ARROW_SPECS)
    rng.shuffle(directions)
    all_choices: list[tuple[str, int, int]] = []
    for direction in directions:
        for top, left in _candidate_positions(grid, reserved, protected, bg, direction):
            all_choices.append((direction, top, left))
    if not all_choices:
        return False
    direction, top, left = all_choices[int(rng.integers(len(all_choices)))]
    _place_arrow(grid, reserved, protected, bg, direction, top, left)
    return True


def _unique_bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(np.max(counts))
    winners = [int(v) for v, cnt in zip(values, counts) if int(cnt) == max_count]
    return winners == [bg]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GEN_SIZE_MIN, GEN_SIZE_MAX + 1))
    w = int(rng.integers(GEN_SIZE_MIN, GEN_SIZE_MAX + 1))
    stripe_axis = "horizontal" if rng.random() < 0.5 else "vertical"
    stripe_count = int(rng.integers(GEN_STRIPE_COUNT_MIN, GEN_STRIPE_COUNT_MAX + 1))
    stripe_positions = _sample_line_positions(h if stripe_axis == "horizontal" else w, stripe_count, rng)
    if stripe_positions is None:
        return None
    stripe_set = set(stripe_positions)

    bg_choices = [color for color in range(10) if color != GEN_ARROW_COLOR]
    bg = int(rng.choice(bg_choices))
    stripe_choices = [color for color in range(10) if color not in {bg, GEN_ARROW_COLOR}]
    stripe_colors = list(map(int, rng.choice(stripe_choices, size=stripe_count, replace=False)))

    grid = np.full((h, w), bg, dtype=int)
    for pos, color in zip(stripe_positions, stripe_colors):
        if stripe_axis == "horizontal":
            grid[pos, :] = color
        else:
            grid[:, pos] = color

    reserved = np.zeros((h, w), dtype=bool)
    protected = np.zeros((h, w), dtype=bool)
    want_blocker = bool(rng.integers(0, 2))

    if want_blocker:
        pair = _place_blocker_pair(grid, reserved, protected, bg, stripe_axis, stripe_set, rng)
        if pair is None:
            return None
        shooter, _blocker, shooter_info = pair
        for r, c in shooter_info["path_cells"]:
            protected[r, c] = True

    if stripe_axis == "horizontal":
        crossing_dirs = ["U", "D"]
        parallel_dirs = ["L", "R"]
    else:
        crossing_dirs = ["L", "R"]
        parallel_dirs = ["U", "D"]

    crossing = _place_with_predicate(
        grid,
        reserved,
        protected,
        bg,
        stripe_axis,
        stripe_set,
        crossing_dirs,
        lambda info: info["stop"] == "boundary" and info["crossings"] >= 1 and info["painted"] >= 2,
        rng,
    )
    if crossing is None:
        return None
    _crossing_arrow, crossing_info = crossing
    for r, c in crossing_info["path_cells"]:
        protected[r, c] = True

    parallel = _place_with_predicate(
        grid,
        reserved,
        protected,
        bg,
        stripe_axis,
        stripe_set,
        parallel_dirs,
        lambda info: info["stop"] == "boundary" and info["crossings"] == 0 and info["painted"] >= 2,
        rng,
    )
    if parallel is None:
        return None
    _parallel_arrow, parallel_info = parallel
    for r, c in parallel_info["path_cells"]:
        protected[r, c] = True

    filler_count = int(rng.integers(1, GEN_FILLER_MAX + 1))
    for _ in range(filler_count):
        if not _place_filler(grid, reserved, protected, bg, rng):
            break

    if not _unique_bg_is_dominant(grid, bg):
        return None

    try:
        scene = _parse_scene(grid)
        infos = [_trace_arrow(scene, arrow) for arrow in scene["arrows"]]
        out = main(grid)
    except AssertionError:
        return None

    if len(scene["arrows"]) < 3:
        return None
    if not any(info["crossings"] >= 1 for info in infos):
        return None
    if not any(info["crossings"] == 0 for info in infos):
        return None
    if not any(info["stop"] == "boundary" for info in infos):
        return None
    if want_blocker and not any(info["stop"] == "arrow" for info in infos):
        return None
    if len({arrow["direction"] for arrow in scene["arrows"]}) < 2:
        return None
    if int(np.sum(out != grid)) < GEN_MIN_CHANGED:
        return None

    return grid


def generate_input() -> np.ndarray:
    """
    Generate varied scenes with horizontal or vertical color-switch stripes and
    several arrows whose rays show both boundary and arrow-stop behaviors.
    """
    rng = default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-and-stripes scene")

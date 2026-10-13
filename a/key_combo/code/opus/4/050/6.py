from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import itertools
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_shared_arc_utils_for_050_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG = 0
BLUE = 1
GREEN = 3
GREY = 5
ORANGE = 7
MAROON = 9

GRID_SHAPE = (14, 14)
CROSS_ROW = 7
CROSS_COL = 7
SPRITE_COLORS = [2, 4, 6, 8]
QUADS = ("TL", "TR", "BL", "BR")
CROSS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
SLIDE_DIR = {
    "TL": (1, 1),
    "TR": (1, -1),
    "BL": (-1, 1),
    "BR": (-1, -1),
}
ACTIVE_BY_AXIS = {
    "horizontal": ("TL", "BR"),
    "vertical": ("TR", "BL"),
}
MAX_GEN_ATTEMPTS = 400


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _cross_cells(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return tuple((int(r + dr), int(c + dc)) for dr, dc in CROSS_OFFSETS)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _quad_ranges(shape: tuple[int, int], cross_r: int, cross_c: int, quad: str) -> tuple[range, range]:
    rows = range(0, cross_r) if quad[0] == "T" else range(cross_r + 1, shape[0])
    cols = range(0, cross_c) if quad[1] == "L" else range(cross_c + 1, shape[1])
    return rows, cols


def _quadrant_cells(shape: tuple[int, int], cross_r: int, cross_c: int, quad: str) -> list[tuple[int, int]]:
    rows, cols = _quad_ranges(shape, cross_r, cross_c, quad)
    return [(r, c) for r in rows for c in cols]


def _quadrant_of(center: tuple[int, int], cross_r: int, cross_c: int) -> str:
    r, c = center
    assert r != cross_r and c != cross_c, "sprite center cannot lie on the grey cross"
    return ("T" if r < cross_r else "B") + ("L" if c < cross_c else "R")


def _step_targets(
    center: tuple[int, int],
    quad: str,
    cross_r: int,
    cross_c: int,
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], ...]]:
    dr, dc = SLIDE_DIR[quad]
    cur_r, cur_c = center
    steps: list[tuple[tuple[int, int], ...]] = []
    while True:
        nxt = (cur_r + dr, cur_c + dc)
        target = _cross_cells(nxt)
        if any((not _in_bounds(shape, r, c)) or r == cross_r or c == cross_c for r, c in target):
            break
        steps.append(target)
        cur_r, cur_c = nxt
    return steps


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    row_candidates = [
        r
        for r in range(grid.shape[0])
        if set(map(int, np.unique(grid[r, :]))) <= {GREY, MAROON}
    ]
    col_candidates = [
        c
        for c in range(grid.shape[1])
        if set(map(int, np.unique(grid[:, c]))) <= {GREY, MAROON}
    ]
    assert len(row_candidates) == 1 and len(col_candidates) == 1, "expected one grey separator row and column"
    cross_r, cross_c = int(row_candidates[0]), int(col_candidates[0])
    assert int(grid[cross_r, cross_c]) == GREY, "separator intersection must remain grey"
    return cross_r, cross_c


def _marker_axis(grid: np.ndarray, cross_r: int, cross_c: int) -> str:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MAROON)]
    assert len(markers) == 2, "expected exactly two maroon markers"
    if all(r == cross_r for r, _c in markers):
        assert any(c < cross_c for _r, c in markers) and any(c > cross_c for _r, c in markers)
        return "horizontal"
    if all(c == cross_c for _r, c in markers):
        assert any(r < cross_r for r, _c in markers) and any(r > cross_r for r, _c in markers)
        return "vertical"
    raise AssertionError("markers must occupy one separator axis")


def _sprite_center(coords: np.ndarray) -> tuple[int, int]:
    pts = {tuple(map(int, rc)) for rc in coords.tolist()}
    centers = []
    for r, c in pts:
        deg4 = sum((nr, nc) in pts for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if deg4 == 4:
            centers.append((r, c))
    assert len(centers) == 1, "cross sprite must have a unique center"
    return centers[0]


def _parse_sprites(grid: np.ndarray, cross_r: int, cross_c: int) -> dict[str, dict]:
    bg = int(get_bg(grid))
    sprite_colors = [color for color in sorted(map(int, np.unique(grid))) if color not in {bg, GREY, ORANGE, MAROON}]
    objects = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sprite_colors,
        return_format="dict",
    )
    assert len(objects) == 4, "expected four colored cross sprites"

    sprites: dict[str, dict] = {}
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        assert int(obj["area"]) == 5, "each sprite must contain five cells"
        center = _sprite_center(coords)
        rel = {(int(r - center[0]), int(c - center[1])) for r, c in coords.tolist()}
        assert rel == set(CROSS_OFFSETS), "each sprite must be a plus shape"
        quad = _quadrant_of(center, cross_r, cross_c)
        assert quad not in sprites, "expected one sprite per quadrant"
        sprites[quad] = {
            "color": int(obj["color"]),
            "coords": tuple(tuple(map(int, rc)) for rc in coords.tolist()),
            "center": center,
        }
    assert set(sprites) == set(QUADS), "missing quadrant sprite"
    return sprites


def _simulate_from_grid(grid: np.ndarray, axis_override: str | None = None) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    cross_r, cross_c = _find_cross_axes(g)
    axis = axis_override or _marker_axis(g, cross_r, cross_c)
    active = ACTIVE_BY_AXIS[axis]
    sprites = _parse_sprites(g, cross_r, cross_c)

    out = g.copy()
    results: dict[str, dict] = {}
    for quad in active:
        _paint_cells(out, sprites[quad]["coords"], BG)

    for quad in active:
        color = int(sprites[quad]["color"])
        cur_center = tuple(map(int, sprites[quad]["center"]))
        steps = _step_targets(cur_center, quad, cross_r, cross_c, out.shape)
        count = 0
        for target in steps:
            for r, c in target:
                if int(out[r, c]) == ORANGE:
                    out[r, c] = BG
                    count += 1
            cur_center = (cur_center[0] + SLIDE_DIR[quad][0], cur_center[1] + SLIDE_DIR[quad][1])
        results[quad] = {
            "count": count,
            "color": color,
            "final_center": cur_center,
            "steps": steps,
        }

    q0, q1 = active
    c0 = int(results[q0]["count"])
    c1 = int(results[q1]["count"])
    assert c0 != c1, "active sprites must produce different collision counts"
    winner = q0 if c0 > c1 else q1
    loser = q1 if winner == q0 else q0
    recolor = {winner: BLUE, loser: GREEN}
    for quad in active:
        _paint_cells(out, _cross_cells(results[quad]["final_center"]), recolor[quad])

    return out, {
        "axis": axis,
        "active": active,
        "cross": (cross_r, cross_c),
        "sprites": sprites,
        "results": results,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate one diagonal pair of quadrant crosses, slide them toward the centre, then recolor them by obstacle-hit count.

    Input structure:
    - The grid has black background, one grey horizontal separator and one grey vertical separator that cross near the centre.
    - Each quadrant contains exactly one monochromatic 5-cell plus sprite and one to four orange blocker pixels.
    - Two maroon marker pixels sit on the grey cross: both on the horizontal separator or both on the vertical separator.

    Rule:
    - Horizontal markers activate the main-diagonal pair (top-left and bottom-right); vertical markers activate the other diagonal pair.
    - Each activated plus translates one cell per step along its quadrant-to-centre diagonal, stopping before any sprite cell would enter the grey separator row or column.
    - Before every step, remove every orange blocker that lies in the target footprint and count those removed pixels; inactive quadrants stay unchanged.
    - Compare the two activated counts and recolor the higher-count sprite blue and the lower-count sprite green, while preserving the grey cross and maroon markers.

    Key insight:
    The markers do not name colors directly; they choose which diagonal pair competes, the blocker layout along each diagonal path determines the counts, and only that comparison decides the final recoloring.
    """
    out, _meta = _simulate_from_grid(grid)
    return out


def _candidate_centers(shape: tuple[int, int], cross_r: int, cross_c: int, quad: str) -> list[tuple[int, int]]:
    rows, cols = _quad_ranges(shape, cross_r, cross_c, quad)
    centers = []
    for r in rows:
        for c in cols:
            cells = _cross_cells((r, c))
            if any((not _in_bounds(shape, rr, cc)) or rr == cross_r or cc == cross_c for rr, cc in cells):
                continue
            if _step_targets((r, c), quad, cross_r, cross_c, shape):
                centers.append((r, c))
    assert centers, f"no valid centers for {quad}"
    return centers


def _first_hit_steps(
    steps: list[tuple[tuple[int, int], ...]],
) -> dict[tuple[int, int], int]:
    first: dict[tuple[int, int], int] = {}
    for idx, cells in enumerate(steps):
        for cell in cells:
            first.setdefault(tuple(map(int, cell)), idx)
    return first


def _step_hit_count(steps: list[tuple[tuple[int, int], ...]], obstacles: set[tuple[int, int]]) -> int:
    remaining = set(obstacles)
    hit_steps = 0
    for cells in steps:
        touched = remaining & set(cells)
        if touched:
            hit_steps += 1
            remaining.difference_update(touched)
    return hit_steps


def _path_subset_options(
    steps: list[tuple[tuple[int, int], ...]],
    allowed_cells: set[tuple[int, int]] | None = None,
    max_hits: int = 4,
) -> dict[int, list[tuple[set[tuple[int, int]], int]]]:
    union = set(_first_hit_steps(steps))
    if allowed_cells is not None:
        union &= set(allowed_cells)
    union = sorted(union)
    options: dict[int, list[tuple[set[tuple[int, int]], int]]] = {0: [(set(), 0)]}
    for hit_count in range(1, min(max_hits, len(union)) + 1):
        seen: set[tuple[tuple[int, int], ...]] = set()
        variants: list[tuple[set[tuple[int, int]], int]] = []
        for combo in itertools.combinations(union, hit_count):
            subset = set(combo)
            step_hits = _step_hit_count(steps, subset)
            key = tuple(sorted(combo))
            if key in seen:
                continue
            seen.add(key)
            variants.append((subset, step_hits))
        random.shuffle(variants)
        options[hit_count] = variants
    return options


def _random_marker_positions(axis: str, shape: tuple[int, int], cross_r: int, cross_c: int) -> list[tuple[int, int]]:
    h, w = shape
    if axis == "horizontal":
        left = random.randint(0, cross_c - 1)
        right = random.randint(cross_c + 1, w - 1)
        return [(cross_r, left), (cross_r, right)]
    top = random.randint(0, cross_r - 1)
    bottom = random.randint(cross_r + 1, h - 1)
    return [(top, cross_c), (bottom, cross_c)]


def _build_grid(meta: dict) -> np.ndarray:
    shape = tuple(map(int, meta["shape"]))
    cross_r, cross_c = map(int, meta["cross"])
    grid = np.full(shape, BG, dtype=int)
    grid[cross_r, :] = GREY
    grid[:, cross_c] = GREY
    for quad in QUADS:
        _paint_cells(grid, _cross_cells(meta["sprites"][quad]["center"]), meta["sprites"][quad]["color"])
    for quad in QUADS:
        for cell in meta["obstacles"][quad]:
            grid[cell] = ORANGE
    for cell in meta["markers"]:
        grid[cell] = MAROON
    return grid


def _expected_from_meta(meta: dict) -> tuple[np.ndarray, dict]:
    grid = _build_grid(meta)
    axis = meta["axis"]
    active = ACTIVE_BY_AXIS[axis]
    out = grid.copy()
    results: dict[str, dict] = {}
    for quad in active:
        _paint_cells(out, _cross_cells(meta["sprites"][quad]["center"]), BG)
    for quad in active:
        center = tuple(map(int, meta["sprites"][quad]["center"]))
        steps = _step_targets(center, quad, meta["cross"][0], meta["cross"][1], tuple(meta["shape"]))
        obstacles = set(meta["obstacles"][quad])
        count = 0
        cur_center = center
        for cells in steps:
            touched = obstacles & set(cells)
            count += len(touched)
            obstacles.difference_update(touched)
            cur_center = (cur_center[0] + SLIDE_DIR[quad][0], cur_center[1] + SLIDE_DIR[quad][1])
            for cell in touched:
                out[cell] = BG
        results[quad] = {"count": count, "final_center": cur_center, "steps": steps}
    q0, q1 = active
    assert results[q0]["count"] != results[q1]["count"]
    winner = q0 if results[q0]["count"] > results[q1]["count"] else q1
    loser = q1 if winner == q0 else q0
    _paint_cells(out, _cross_cells(results[winner]["final_center"]), BLUE)
    _paint_cells(out, _cross_cells(results[loser]["final_center"]), GREEN)
    return out, {"active": active, "results": results}


def _choose_active_obstacles(meta: dict) -> bool:
    axis = meta["axis"]
    active = ACTIVE_BY_AXIS[axis]
    infos = []
    for quad in active:
        center = tuple(meta["sprites"][quad]["center"])
        steps = _step_targets(center, quad, meta["cross"][0], meta["cross"][1], tuple(meta["shape"]))
        all_cells = set(_quadrant_cells(tuple(meta["shape"]), meta["cross"][0], meta["cross"][1], quad))
        body = set(_cross_cells(center))
        path_union = set(_first_hit_steps(steps)) - body
        off_path = sorted(all_cells - body - path_union)
        infos.append(
            {
                "quad": quad,
                "steps": steps,
                "path_options": _path_subset_options(steps, allowed_cells=path_union),
                "path_len": len(steps),
                "off_path": off_path,
            }
        )

    if any(not info["off_path"] for info in infos):
        return False

    candidates = []
    for h0, opts0 in infos[0]["path_options"].items():
        for subset0, step_hits0 in opts0:
            for h1, opts1 in infos[1]["path_options"].items():
                if h0 == h1:
                    continue
                if infos[0]["path_len"] != infos[1]["path_len"] and _sign(h0 - h1) == _sign(infos[0]["path_len"] - infos[1]["path_len"]):
                    continue
                for subset1, step_hits1 in opts1:
                    for total0 in range(max(1, h0), 5):
                        for total1 in range(max(1, h1), 5):
                            if total0 != total1 and _sign(h0 - h1) == _sign(total0 - total1):
                                continue
                            if total0 == h0 and total1 == h1:
                                continue
                            if total0 - h0 > len(infos[0]["off_path"]) or total1 - h1 > len(infos[1]["off_path"]):
                                continue
                            multi_hit = (step_hits0 < h0) or (step_hits1 < h1)
                            score = (
                                int(multi_hit),
                                int(h0 > 0 and h1 > 0),
                                int(total0 != total1),
                                random.random(),
                            )
                            candidates.append((score, subset0, step_hits0, total0, subset1, step_hits1, total1))

    if not candidates:
        return False

    candidates.sort(reverse=True)
    top = candidates[: min(20, len(candidates))]
    _score, subset0, step_hits0, total0, subset1, step_hits1, total1 = random.choice(top)
    _ = step_hits0, step_hits1
    chosen = [(subset0, total0), (subset1, total1)]
    for info, (path_subset, total) in zip(infos, chosen):
        extra_count = total - len(path_subset)
        extras = set(random.sample(info["off_path"], extra_count))
        meta["obstacles"][info["quad"]] = set(path_subset) | extras
    return True


def _fill_inactive_obstacles(meta: dict) -> None:
    active = set(ACTIVE_BY_AXIS[meta["axis"]])
    for quad in QUADS:
        if quad in active:
            continue
        all_cells = set(_quadrant_cells(tuple(meta["shape"]), meta["cross"][0], meta["cross"][1], quad))
        body = set(_cross_cells(meta["sprites"][quad]["center"]))
        candidates = sorted(all_cells - body)
        count = random.randint(1, 4)
        meta["obstacles"][quad] = set(random.sample(candidates, count))


def _meta_for_attempt() -> dict | None:
    shape = GRID_SHAPE
    cross = (CROSS_ROW, CROSS_COL)
    axis = random.choice(["horizontal", "vertical"])
    markers = _random_marker_positions(axis, shape, cross[0], cross[1])
    color_order = random.sample(SPRITE_COLORS, len(SPRITE_COLORS))
    sprites = {}
    for quad, color in zip(QUADS, color_order):
        sprites[quad] = {
            "color": int(color),
            "center": random.choice(_candidate_centers(shape, cross[0], cross[1], quad)),
        }
    meta = {
        "shape": shape,
        "cross": cross,
        "axis": axis,
        "markers": markers,
        "sprites": sprites,
        "obstacles": {quad: set() for quad in QUADS},
    }
    if not _choose_active_obstacles(meta):
        return None
    _fill_inactive_obstacles(meta)
    return meta


def _validate_meta(meta: dict) -> bool:
    grid = _build_grid(meta)
    expected, info = _expected_from_meta(meta)
    observed = main(grid)
    if not np.array_equal(expected, observed):
        return False

    sim_out, sim_meta = _simulate_from_grid(grid)
    if not np.array_equal(sim_out, observed):
        return False

    active = ACTIVE_BY_AXIS[meta["axis"]]
    counts = [info["results"][quad]["count"] for quad in active]
    totals = [len(meta["obstacles"][quad]) for quad in active]
    step_hits = [_step_hit_count(info["results"][quad]["steps"], meta["obstacles"][quad]) for quad in active]
    path_lengths = [len(info["results"][quad]["steps"]) for quad in active]

    if counts[0] == counts[1]:
        return False
    if path_lengths[0] != path_lengths[1] and _sign(counts[0] - counts[1]) == _sign(path_lengths[0] - path_lengths[1]):
        return False
    if totals[0] != totals[1] and _sign(counts[0] - counts[1]) == _sign(totals[0] - totals[1]):
        return False
    if not (step_hits[0] < counts[0] or step_hits[1] < counts[1]):
        return False
    if not (totals[0] > counts[0] or totals[1] > counts[1]):
        return False

    other_axis = "vertical" if meta["axis"] == "horizontal" else "horizontal"
    try:
        wrong_out, _wrong_meta = _simulate_from_grid(grid, axis_override=other_axis)
    except AssertionError:
        wrong_out = None
    if wrong_out is not None and np.array_equal(wrong_out, observed):
        return False

    for quad in QUADS:
        body = set(_cross_cells(meta["sprites"][quad]["center"]))
        if body & meta["obstacles"][quad]:
            return False
        if not 1 <= len(meta["obstacles"][quad]) <= 4:
            return False

    return sim_meta["axis"] == meta["axis"]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        meta = _meta_for_attempt()
        if meta is None:
            continue
        if _validate_meta(meta):
            return _build_grid(meta)
    raise RuntimeError("failed to generate a valid quadrant-slide collision task")

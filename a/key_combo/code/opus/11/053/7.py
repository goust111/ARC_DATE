from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"failed to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

find_components = _ARC_UTILS.find_components


BG = 0
BLUE = 1
RED = 2
GREEN = 3

HEIGHT_RANGE = (10, 14)
WIDTH_RANGE = (12, 18)
MAX_GENERATION_ATTEMPTS = 400

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
)


def _marker_columns(grid: np.ndarray) -> tuple[list[int], list[int]]:
    if grid.ndim != 2:
        raise AssertionError("expected a 2D grid")
    h, w = grid.shape
    if h < 4:
        raise AssertionError("grid is too short for the marker cycle")
    if not np.all(grid[0] == BLUE):
        raise AssertionError("top row must be the full blue edge bar")

    red_coords = np.argwhere(grid == RED)
    for r, _c in red_coords:
        if int(r) not in (1, 2):
            raise AssertionError("red markers may only appear in rows 1-2")

    singles: list[int] = []
    doubles: list[int] = []
    for c in range(w):
        top = int(grid[1, c]) == RED
        second = int(grid[2, c]) == RED
        if second and not top:
            raise AssertionError("row-2 red requires row-1 red in the same column")
        if top and second:
            doubles.append(int(c))
        elif top:
            singles.append(int(c))
    if not singles or not doubles:
        raise AssertionError("expected at least one single marker and one double marker")
    return singles, doubles


def _green_objects(grid: np.ndarray) -> list[dict]:
    return find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        colors=[GREEN],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )


def _hit_indices(objects: list[dict], ray_cols: set[int]) -> list[int]:
    hits: list[int] = []
    for idx, obj in enumerate(objects):
        cols = {int(c) for c in obj["coords"][:, 1]}
        if cols & ray_cols:
            hits.append(idx)
    return hits


def _base_after_cycle(grid: np.ndarray, singles: list[int], ray_cols: set[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for c in singles:
        out[2, int(c)] = RED
    h = out.shape[0]
    for c in ray_cols:
        out[: h - 1, int(c)] = BLUE
        out[h - 1, int(c)] = BLUE
    return out


def _render_variant(grid: np.ndarray, mode: str) -> np.ndarray:
    singles, doubles = _marker_columns(grid)
    objects = _green_objects(grid)
    double_cols = set(doubles)
    all_marker_cols = double_cols | set(singles)

    if mode == "true":
        out = _base_after_cycle(grid, singles, double_cols)
        hit = _hit_indices(objects, double_cols)
        for idx in hit:
            coords = objects[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
        return out

    if mode == "all_markers_fire":
        out = np.asarray(grid, dtype=int).copy()
        h = out.shape[0]
        for c in all_marker_cols:
            out[: h - 1, int(c)] = BLUE
            out[h - 1, int(c)] = BLUE
        hit = _hit_indices(objects, all_marker_cols)
        for idx in hit:
            coords = objects[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = BG
        return out

    if mode == "pixel_only":
        out = _base_after_cycle(grid, singles, double_cols)
        for obj in objects:
            coords = obj["coords"]
            ray_mask = np.isin(coords[:, 1], list(double_cols))
            if np.any(ray_mask):
                ray_coords = coords[ray_mask]
                out[ray_coords[:, 0], ray_coords[:, 1]] = BG
        return out

    if mode == "no_erase":
        return _base_after_cycle(grid, singles, double_cols)

    raise ValueError(f"unknown render mode: {mode}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance top-edge markers, spawn bottom blue ray sources from completed markers, and erase each green object hit by those rays.

    Input structure:
    - The background is black.
    - The entire top row is a blue full edge bar that fixes the migration direction downward.
    - Red markers appear only in rows 1-2: a single red pixel in row 1 is an early-state marker, while a red pair in rows 1-2 is a late-state marker.
    - The lower half contains several 4-connected green objects.

    Rule:
    - For each column, a single red marker grows downward by one cell to become a two-cell red marker.
    - A two-cell red marker completes its cycle, producing a two-cell blue mark in the same column at the bottom and a vertical blue ray that runs upward to the top bar.
    - Any green object touched by at least one completed-marker ray is erased as a whole object; untouched green objects remain unchanged.

    Key insight:
    - Marker state is a timer: only columns that already contain a two-cell red marker generate bottom blue sources and rays, so surviving objects under single markers disambiguate the timing rule from a simpler “every red column fires” shortcut.
    """
    return _render_variant(np.asarray(grid, dtype=int), "true")


def _candidate_object_cells(
    shape: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    category: str,
    target_col: int,
    single_cols: set[int],
    double_cols: set[int],
    require_off_ray: bool,
) -> list[list[tuple[int, int]]]:
    shape_h = max(r for r, _c in shape) + 1
    shape_w = max(c for _r, c in shape) + 1
    allowed_rows = range(5, h - 2 - shape_h + 1)
    allowed_cols = range(0, w - shape_w + 1)
    candidates: list[list[tuple[int, int]]] = []

    for r0 in allowed_rows:
        for c0 in allowed_cols:
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(cell in blocked for cell in cells):
                continue
            cols = {c for _r, c in cells}
            if target_col not in cols:
                continue
            if category == "hit":
                if cols & single_cols:
                    continue
                if cols & double_cols != {target_col}:
                    continue
                if require_off_ray and not any(c != target_col for c in cols):
                    continue
            elif category == "single":
                if cols & double_cols:
                    continue
                if cols & single_cols != {target_col}:
                    continue
            elif category == "empty":
                if cols & (single_cols | double_cols):
                    continue
            else:
                raise ValueError(f"unknown category: {category}")
            candidates.append(cells)
    return candidates


def _expand_blocked(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _build_specs(
    rng: random.Random,
    single_cols: list[int],
    double_cols: list[int],
    empty_cols: list[int],
) -> list[tuple[str, int, bool]]:
    hit_count = 1 if len(double_cols) == 1 else rng.choice([1, 2])
    hit_targets = rng.sample(double_cols, hit_count)
    single_target = rng.choice(single_cols)
    empty_target = rng.choice(empty_cols)

    specs: list[tuple[str, int, bool]] = [("hit", hit_targets[0], True)]
    for c in hit_targets[1:]:
        specs.append(("hit", c, False))
    specs.append(("single", single_target, False))
    specs.append(("empty", empty_target, False))

    remaining_slots = 5 - len(specs)
    extra_count = rng.randint(0, min(2, remaining_slots))
    used_single_targets = {single_target}
    used_empty_targets = {empty_target}
    for _ in range(extra_count):
        single_options = [c for c in single_cols if c not in used_single_targets]
        empty_options = [c for c in empty_cols if c not in used_empty_targets]
        options: list[tuple[str, int, bool]] = []
        if single_options:
            options.append(("single", rng.choice(single_options), False))
        if empty_options:
            options.append(("empty", rng.choice(empty_options), False))
        if not options:
            break
        spec = rng.choice(options)
        specs.append(spec)
        if spec[0] == "single":
            used_single_targets.add(spec[1])
        else:
            used_empty_targets.add(spec[1])
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*HEIGHT_RANGE)
    w = rng.randint(*WIDTH_RANGE)
    if h >= 15 and rng.random() < 0.35:
        h = 15
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = BLUE

    marker_slots = [c for c in range(1, w - 1, 2)]
    if len(marker_slots) < 4:
        return None

    total_markers = rng.randint(4, min(6, len(marker_slots)))
    double_min = max(1, total_markers - 4)
    double_max = min(3, total_markers - 1)
    double_count = rng.randint(double_min, double_max)
    single_count = total_markers - double_count

    chosen = rng.sample(marker_slots, double_count + single_count)
    doubles = sorted(chosen[:double_count])
    singles = sorted(chosen[double_count:])
    grid[1, singles] = RED
    grid[1, doubles] = RED
    grid[2, doubles] = RED

    all_cols = set(range(w))
    empty_cols = sorted(all_cols - set(singles) - set(doubles))
    if len(empty_cols) < 2:
        return None

    specs = _build_specs(rng, singles, doubles, empty_cols)
    single_set = set(singles)
    double_set = set(doubles)
    blocked: set[tuple[int, int]] = set()

    for category, target_col, require_off_ray in specs:
        shape_order = list(SHAPES)
        rng.shuffle(shape_order)
        candidates: list[list[tuple[int, int]]] = []
        for shape in shape_order:
            candidates.extend(
                _candidate_object_cells(
                    shape,
                    h,
                    w,
                    blocked,
                    category,
                    target_col,
                    single_set,
                    double_set,
                    require_off_ray,
                )
            )
        if not candidates:
            return None
        cells = rng.choice(candidates)
        for r, c in cells:
            grid[r, c] = GREEN
        blocked |= _expand_blocked(cells, h, w)

    return grid


def _validate_example(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h <= 30 and w <= 30
    assert HEIGHT_RANGE[0] <= h <= 15
    assert WIDTH_RANGE[0] <= w <= WIDTH_RANGE[1]
    assert np.all(g[0] == BLUE)
    assert int(np.count_nonzero(g == BLUE)) == w

    singles, doubles = _marker_columns(g)
    assert 1 <= len(doubles) <= 3
    assert 1 <= len(singles) <= 4
    assert 4 <= len(singles) + len(doubles) <= 6

    objects = _green_objects(g)
    assert 3 <= len(objects) <= 5
    assert all(2 <= int(obj["area"]) <= 5 for obj in objects)
    assert all(int(obj["coords"][:, 0].min()) >= 5 for obj in objects)
    assert all(int(obj["coords"][:, 0].max()) <= h - 3 for obj in objects)

    ray_cols = set(doubles)
    hit = _hit_indices(objects, ray_cols)
    survivors = [idx for idx in range(len(objects)) if idx not in hit]
    assert hit, "need at least one hit object"
    assert survivors, "need at least one surviving object"

    single_set = set(singles)
    survivor_single = False
    for idx in survivors:
        cols = {int(c) for c in objects[idx]["coords"][:, 1]}
        if cols & single_set:
            survivor_single = True
    assert survivor_single, "need a survivor under a single-marker column"

    whole_erase_witness = False
    for idx in hit:
        cols = {int(c) for c in objects[idx]["coords"][:, 1]}
        if any(c not in ray_cols for c in cols):
            whole_erase_witness = True
    assert whole_erase_witness, "need a hit object with off-ray pixels"

    out = main(g)
    assert not np.array_equal(out, _render_variant(g, "all_markers_fire"))
    assert not np.array_equal(out, _render_variant(g, "pixel_only"))
    assert not np.array_equal(out, _render_variant(g, "no_erase"))

    for c in singles:
        assert int(out[1, c]) == RED and int(out[2, c]) == RED
    for c in doubles:
        assert int(out[h - 2, c]) == BLUE and int(out[h - 1, c]) == BLUE
        assert int(out[3, c]) == BLUE

    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        vals = out[coords[:, 0], coords[:, 1]]
        if idx in hit:
            assert np.all(vals == BG)
        else:
            assert np.all(vals == GREEN)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        _validate_example(grid)
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

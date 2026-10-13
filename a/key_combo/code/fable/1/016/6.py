from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


Coord = tuple[int, int]

YELLOW = Color.YELLOW
BACKGROUND = Color.BLACK
OBJECT_COLORS = [c for c in Color.ALL_COLORS if c not in {BACKGROUND, YELLOW}]
HOLE_SIDES = (3, 5)
MARGIN_VALUES = (1, 2, 3, 4)
OBJECT_COUNT_CHOICES = (2, 3, 4)
GRID_MAX = 30
GRID_RETRY_LIMIT = 300
PLACEMENT_RETRY_LIMIT = 120
CANONICAL_PIPS = {
    3: [(0, 0), (0, 2), (2, 0), (2, 2)],
    5: [(0, 0), (0, 4), (4, 0), (4, 4)],
}


def _rotate_coords(coords: np.ndarray, pivot: Coord, quarter_turns: int) -> np.ndarray:
    arr = np.asarray(coords, dtype=int).copy()
    pr, pc = map(int, pivot)
    for _ in range(int(quarter_turns) % 4):
        dr = arr[:, 0] - pr
        dc = arr[:, 1] - pc
        arr = np.column_stack((pr + dc, pc - dr))
    return arr.astype(int)


def _find_square_hole(local: np.ndarray, object_color: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    h, w = map(int, local.shape)
    non_object_components = components_mask(local != int(object_color), connectivity=4)
    hole_candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in non_object_components:
        coords = coords.astype(int)
        if np.any((coords[:, 0] == 0) | (coords[:, 0] == h - 1) | (coords[:, 1] == 0) | (coords[:, 1] == w - 1)):
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        hole_h = int(r1 - r0 + 1)
        hole_w = int(c1 - c0 + 1)
        if hole_h != hole_w or hole_h % 2 == 0:
            continue
        if int(coords.shape[0]) != hole_h * hole_w:
            continue
        hole_candidates.append(((r0, c0, r1, c1), coords))
    assert len(hole_candidates) == 1, "expected one enclosed odd square hole per object"
    return hole_candidates[0]


def _analyze_object(grid: np.ndarray, component: dict) -> dict:
    color = int(component["color"])
    bbox = tuple(map(int, component["bbox"]))
    coords = np.asarray(component["coords"], dtype=int)
    local = crop_bbox(grid, bbox)
    hole_bbox, hole_coords = _find_square_hole(local, color)
    hr0, hc0, hr1, hc1 = hole_bbox
    hole_patch = local[hr0 : hr1 + 1, hc0 : hc1 + 1]
    pip_mask = hole_patch == YELLOW
    pip_count = int(np.count_nonzero(pip_mask))
    assert 1 <= pip_count <= 4, "hole must contain one to four yellow pips"
    hole_side = int(hole_patch.shape[0])
    pivot_local = (hr0 + hole_side // 2, hc0 + hole_side // 2)
    pivot_global = (bbox[0] + pivot_local[0], bbox[1] + pivot_local[1])

    pip_local_coords = np.argwhere(pip_mask).astype(int)
    pip_global_coords = pip_local_coords + np.array([bbox[0] + hr0, bbox[1] + hc0], dtype=int)
    rotated_coords = _rotate_coords(coords, pivot_global, pip_count)

    return {
        "color": color,
        "coords": coords,
        "rotated_coords": rotated_coords,
        "pip_coords": pip_global_coords,
        "pivot": pivot_global,
        "pip_count": pip_count,
        "hole_bbox": (bbox[0] + hr0, bbox[1] + hc0, bbox[0] + hr1, bbox[1] + hc1),
        "hole_coords_local": hole_coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each perforated object around its hole center by the number of yellow pips inside that hole, then erase the pips.

    Input structure:
    - The background contains several separated single-color objects plus yellow pip pixels.
    - Each object encloses exactly one odd-side square hole made of background cells, and the hole may sit off-center inside the object.
    - The yellow pixels appear only inside those holes, with one to four pips per hole.

    Rule:
    - For each object, locate its enclosed square hole and count the yellow pips inside it.
    - Use the hole's center cell as the rotation pivot, and rotate the whole object clockwise by that many quarter-turns.
    - Remove every yellow pip so the hole returns to background in the output.

    Key insight:
    - The yellow count encodes the quarter-turn count, and the pivot is the hole center rather than the object's own center, so off-center holes make the rotated object translate across the canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg != YELLOW, "yellow cannot be the background"

    object_colors = [int(c) for c in np.unique(g) if int(c) not in {bg, YELLOW}]
    components = find_components(
        g,
        bg=bg,
        colors=object_colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected at least one non-yellow object"

    analyses = [_analyze_object(g, component) for component in components]
    seen_pips = {
        (int(r), int(c))
        for info in analyses
        for r, c in np.asarray(info["pip_coords"], dtype=int)
    }
    all_pips = {tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)}
    assert seen_pips == all_pips, "all yellow pixels must belong to object holes"

    out = g.copy()
    for info in analyses:
        coords = np.asarray(info["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
    for r, c in seen_pips:
        out[r, c] = bg

    occupied_out: set[Coord] = set()
    for info in analyses:
        color = int(info["color"])
        rotated = np.asarray(info["rotated_coords"], dtype=int)
        assert np.all(rotated[:, 0] >= 0) and np.all(rotated[:, 0] < g.shape[0])
        assert np.all(rotated[:, 1] >= 0) and np.all(rotated[:, 1] < g.shape[1])
        rotated_tuples = [tuple(map(int, rc)) for rc in rotated]
        assert len(rotated_tuples) == len(set(rotated_tuples)), "rotation must preserve distinct cells"
        assert occupied_out.isdisjoint(rotated_tuples), "rotated objects must stay disjoint"
        occupied_out.update(rotated_tuples)
        out[rotated[:, 0], rotated[:, 1]] = color

    return out


def _touches(coords: np.ndarray, occupied: set[Coord]) -> bool:
    for r, c in np.asarray(coords, dtype=int):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in occupied:
                    return True
    return False


def _sample_pip_counts(n_objects: int) -> list[int]:
    assert 2 <= n_objects <= 4
    if n_objects == 4:
        counts = [1, 2, 3, 4]
        random.shuffle(counts)
        return counts
    if n_objects == 3 and random.random() < 0.7:
        counts = [4] + random.sample([1, 2, 3], 2)
        random.shuffle(counts)
        return counts
    if n_objects == 2 and random.random() < 0.55:
        counts = [4, random.choice([1, 2, 3])]
        random.shuffle(counts)
        return counts
    return random.sample([1, 2, 3, 4], n_objects)


def _sample_margins() -> tuple[int, int, int, int]:
    while True:
        top, bottom, left, right = [random.choice(MARGIN_VALUES) for _ in range(4)]
        if top == bottom and left == right:
            continue
        if abs(top - bottom) + abs(left - right) < 2:
            continue
        return top, bottom, left, right


def _make_object_spec(color: int, pip_count: int, bg: int) -> dict:
    hole_side = random.choice(HOLE_SIDES)
    top, bottom, left, right = _sample_margins()
    outer_h = top + hole_side + bottom
    outer_w = left + hole_side + right

    patch = np.full((outer_h, outer_w), color, dtype=int)
    hole_r0 = top
    hole_c0 = left
    patch[hole_r0 : hole_r0 + hole_side, hole_c0 : hole_c0 + hole_side] = bg

    pip_positions = random.sample(CANONICAL_PIPS[hole_side], pip_count)
    for dr, dc in pip_positions:
        patch[hole_r0 + dr, hole_c0 + dc] = YELLOW

    object_coords_local = np.argwhere(patch == color).astype(int)
    pivot_local = (hole_r0 + hole_side // 2, hole_c0 + hole_side // 2)
    rotated_local = _rotate_coords(object_coords_local, pivot_local, pip_count)

    min_row = int(min(object_coords_local[:, 0].min(), rotated_local[:, 0].min()))
    max_row = int(max(object_coords_local[:, 0].max(), rotated_local[:, 0].max()))
    min_col = int(min(object_coords_local[:, 1].min(), rotated_local[:, 1].min()))
    max_col = int(max(object_coords_local[:, 1].max(), rotated_local[:, 1].max()))

    return {
        "color": color,
        "pip_count": pip_count,
        "patch": patch,
        "object_coords_local": object_coords_local,
        "rotated_coords_local": rotated_local,
        "bounds": (min_row, min_col, max_row, max_col),
    }


def _candidate_positions(spec: dict, shape: tuple[int, int]) -> list[Coord]:
    h, w = map(int, shape)
    min_row, min_col, max_row, max_col = spec["bounds"]
    low_r = max(1, 1 - min_row)
    high_r = min(h - 2 - max_row, h - spec["patch"].shape[0] - 1)
    low_c = max(1, 1 - min_col)
    high_c = min(w - 2 - max_col, w - spec["patch"].shape[1] - 1)
    if low_r > high_r or low_c > high_c:
        return []
    return [(r0, c0) for r0 in range(low_r, high_r + 1) for c0 in range(low_c, high_c + 1)]


def _place_objects(specs: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    input_occupied: set[Coord] = set()
    output_occupied: set[Coord] = set()
    placed: list[dict] = []

    order = sorted(
        specs,
        key=lambda spec: (
            spec["patch"].shape[0] * spec["patch"].shape[1],
            spec["bounds"][2] - spec["bounds"][0],
            spec["bounds"][3] - spec["bounds"][1],
        ),
        reverse=True,
    )
    for spec in order:
        candidates = _candidate_positions(spec, shape)
        random.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            input_coords = spec["object_coords_local"] + np.array([r0, c0], dtype=int)
            output_coords = spec["rotated_coords_local"] + np.array([r0, c0], dtype=int)
            if _touches(input_coords, input_occupied):
                continue
            if _touches(output_coords, output_occupied):
                continue
            chosen = (r0, c0, input_coords, output_coords)
            break
        if chosen is None:
            return None
        r0, c0, input_coords, output_coords = chosen
        input_occupied.update(tuple(map(int, rc)) for rc in input_coords)
        output_occupied.update(tuple(map(int, rc)) for rc in output_coords)
        placed.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
            }
        )
    return placed


def _build_grid(shape: tuple[int, int], placed: list[dict], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for item in placed:
        spec = item["spec"]
        paste(grid, spec["patch"], item["r0"], item["c0"], transparent=bg)
    return grid


def _attempt_generate() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    pip_counts = _sample_pip_counts(n_objects)
    colors = random.sample(OBJECT_COLORS, n_objects)
    specs = [_make_object_spec(color, pip_count, BACKGROUND) for color, pip_count in zip(colors, pip_counts)]

    min_dim = max(18, 16 + 2 * n_objects)
    for _ in range(PLACEMENT_RETRY_LIMIT):
        shape = (
            random.randint(min_dim, GRID_MAX),
            random.randint(min_dim, GRID_MAX),
        )
        placed = _place_objects(specs, shape)
        if placed is None:
            continue
        grid = _build_grid(shape, placed, BACKGROUND)
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out == YELLOW, np.zeros_like(out, dtype=bool)) and not np.array_equal(out, grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-pivot rotation input")

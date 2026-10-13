from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


OBJ_SIZE = 4
INPUT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
BORDER_MASK = np.ones((OBJ_SIZE, OBJ_SIZE), dtype=bool)
BORDER_MASK[1:3, 1:3] = False
FRAME_PATTERN = np.array(
    [
        [Color.PINK, Color.PINK, Color.BLUE, Color.BLUE],
        [Color.PINK, Color.YELLOW, Color.YELLOW, Color.BLUE],
        [Color.RED, Color.YELLOW, Color.YELLOW, Color.GREEN],
        [Color.RED, Color.RED, Color.GREEN, Color.GREEN],
    ],
    dtype=int,
)


def _classify_patch(patch: np.ndarray, color: int, bg: int) -> str | None:
    if patch.shape != (OBJ_SIZE, OBJ_SIZE):
        return None
    values = set(map(int, np.unique(patch)))
    if not values.issubset({int(bg), int(color)}):
        return None
    if np.all(patch == int(color)):
        return "solid"
    if np.all(patch[BORDER_MASK] == int(color)) and np.all(patch[1:3, 1:3] == int(bg)):
        return "hollow"
    return None


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        assert color in INPUT_COLORS, "unexpected foreground color"
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert r1 - r0 + 1 == OBJ_SIZE and c1 - c0 + 1 == OBJ_SIZE, "every object must fit a 4x4 bbox"
        patch = crop_bbox(g, comp["bbox"])
        kind = _classify_patch(patch, color, bg)
        assert kind is not None, "expected only 4x4 hollow frames and solid squares"
        expected_area = 12 if kind == "hollow" else 16
        assert int(comp["area"]) == expected_area, "component area must match its 4x4 object type"
        objects.append(
            {
                "kind": kind,
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "coords": comp["coords"],
            }
        )
    return bg, objects


def _render_strip(count: int) -> np.ndarray:
    out = np.zeros((OBJ_SIZE, OBJ_SIZE * int(count)), dtype=int)
    for idx in range(int(count)):
        paste(out, FRAME_PATTERN, 0, OBJ_SIZE * idx)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the hollow 4x4 frames, sort them left-to-right, concatenate them, and recolor each frame by quadrant with a yellow center.

    Input structure:
    - The grid has a black background with 4-6 separated monochrome 4x4 objects.
    - Each object is either a hollow square frame whose 2x2 center is black, or a solid 4x4 square.
    - Foreground colors come from blue, red, green, and yellow.

    Rule:
    - Detect every 4x4 monochrome object and classify it by its internal structure.
    - Discard the solid 4x4 squares and keep only the hollow frames.
    - Sort the surviving frames by their bbox top-left column, breaking ties by row.
    - Concatenate the sorted frames horizontally with no gaps to make a 4 x (4*N) strip.
    - Recolor each 4x4 frame using frame-local coordinates: top-left quadrant pink, top-right blue, bottom-left red, bottom-right green, and the original 2x2 hollow center yellow.

    Key insight:
    - The relevant distinction is structural rather than color-based: only objects with a black 2x2 hole survive, and the final colors depend only on each frame's own local coordinates.
    """
    _bg, objects = _extract_objects(np.asarray(grid, dtype=int))
    hollows = [obj for obj in objects if obj["kind"] == "hollow"]
    assert hollows, "expected at least one hollow frame"
    hollows.sort(key=lambda obj: (int(obj["bbox"][1]), int(obj["bbox"][0])))
    return _render_strip(len(hollows))


# --------------- generator ---------------

GRID_H_RANGE = (19, 24)
GRID_W_RANGE = (21, 29)
SLOT_STEP = 5
MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 200

GEN_CASES = (
    {"n_hollow": 3, "n_solid_choices": (1, 2, 3), "require_tie": False},
    {"n_hollow": 4, "n_solid_choices": (1, 2), "require_tie": True},
    {"n_hollow": 5, "n_solid_choices": (1,), "require_tie": False},
    {"n_hollow": 3, "n_solid_choices": (2, 3), "require_tie": True},
    {"n_hollow": 4, "n_solid_choices": (1, 2), "require_tie": False},
    {"n_hollow": 5, "n_solid_choices": (1,), "require_tie": True},
)
_GEN_CALL_COUNT = 0


def _next_case() -> dict[str, object]:
    global _GEN_CALL_COUNT
    case = GEN_CASES[_GEN_CALL_COUNT % len(GEN_CASES)]
    _GEN_CALL_COUNT += 1
    return case


def _slot_values(limit: int, offset: int) -> list[int]:
    return list(range(int(offset), int(limit) - OBJ_SIZE + 1, SLOT_STEP))


def _hollow_patch(color: int) -> np.ndarray:
    patch = np.full((OBJ_SIZE, OBJ_SIZE), int(color), dtype=int)
    patch[1:3, 1:3] = Color.BLACK
    return patch


def _solid_patch(color: int) -> np.ndarray:
    return np.full((OBJ_SIZE, OBJ_SIZE), int(color), dtype=int)


def _sample_colors(rng: random.Random, n_hollow: int, n_solid: int) -> tuple[list[int], list[int]]:
    for _ in range(200):
        hollow_colors = [rng.choice(INPUT_COLORS) for _ in range(n_hollow)]
        solid_colors = [rng.choice(INPUT_COLORS) for _ in range(n_solid)]
        if len(set(hollow_colors)) < 2:
            continue
        if len(set(hollow_colors + solid_colors)) < min(3, n_hollow + n_solid):
            continue
        if set(hollow_colors).isdisjoint(solid_colors):
            continue
        if not any(hollow_colors.count(color) + solid_colors.count(color) >= 2 for color in INPUT_COLORS):
            continue
        return hollow_colors, solid_colors
    raise RuntimeError("failed to sample a non-degenerate color assignment")


def _sample_hollow_positions(
    rng: random.Random,
    row_slots: list[int],
    col_slots: list[int],
    n_hollow: int,
    require_tie: bool,
) -> list[tuple[int, int]] | None:
    if not row_slots or len(col_slots) < n_hollow - int(require_tie):
        return None

    row_pool_size = 2 if require_tie else rng.choice((2, 3))
    if row_pool_size > len(row_slots):
        return None
    row_start = rng.randint(0, len(row_slots) - row_pool_size)
    row_pool = row_slots[row_start : row_start + row_pool_size]

    need_unique_cols = n_hollow - 1 if require_tie else n_hollow
    if need_unique_cols > len(col_slots):
        return None
    base_cols = sorted(rng.sample(col_slots, need_unique_cols))
    positions: list[tuple[int, int]] = []

    if require_tie:
        tie_col = rng.choice(base_cols)
        tie_rows = rng.sample(row_pool, 2)
        positions.extend([(tie_rows[0], tie_col), (tie_rows[1], tie_col)])
        for col in base_cols:
            if col == tie_col:
                continue
            positions.append((rng.choice(row_pool), col))
    else:
        for col in base_cols:
            positions.append((rng.choice(row_pool), col))

    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    if len(positions) != n_hollow:
        return None
    if max(cols) - min(cols) <= max(rows) - min(rows):
        return None
    if require_tie and len(set(cols)) == len(cols):
        return None
    return positions


def _sample_solid_positions(
    rng: random.Random,
    all_slots: list[tuple[int, int]],
    used: set[tuple[int, int]],
    n_solid: int,
    hollow_cols: list[int],
) -> list[tuple[int, int]] | None:
    remaining = [slot for slot in all_slots if slot not in used]
    if len(remaining) < n_solid:
        return None

    min_hollow_col = min(hollow_cols)
    max_hollow_col = max(hollow_cols)
    interior = [slot for slot in remaining if min_hollow_col < slot[1] < max_hollow_col]
    before_or_inside = [slot for slot in remaining if slot[1] < max_hollow_col]
    anchor_pool = interior or before_or_inside
    if not anchor_pool:
        return None

    anchor = rng.choice(anchor_pool)
    rest_pool = [slot for slot in remaining if slot != anchor]
    if len(rest_pool) < n_solid - 1:
        return None
    rest = rng.sample(rest_pool, n_solid - 1)
    return [anchor, *rest]


def _sample_layout(
    rng: random.Random,
    n_hollow: int,
    n_solid: int,
    require_tie: bool,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        height = rng.randint(*GRID_H_RANGE)
        width = rng.randint(*GRID_W_RANGE)
        row_offset = rng.randint(0, 2)
        col_offset = rng.randint(0, 2)
        row_slots = _slot_values(height, row_offset)
        col_slots = _slot_values(width, col_offset)
        if len(row_slots) < 2 or len(col_slots) < n_hollow - int(require_tie):
            continue

        hollow_positions = _sample_hollow_positions(rng, row_slots, col_slots, n_hollow, require_tie)
        if hollow_positions is None:
            continue

        all_slots = [(r, c) for r in row_slots for c in col_slots]
        used = set(hollow_positions)
        solid_positions = _sample_solid_positions(
            rng,
            all_slots,
            used,
            n_solid,
            [c for _, c in hollow_positions],
        )
        if solid_positions is None:
            continue
        return height, width, hollow_positions, solid_positions
    return None


def _render_input(
    height: int,
    width: int,
    hollow_positions: list[tuple[int, int]],
    solid_positions: list[tuple[int, int]],
    hollow_colors: list[int],
    solid_colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (r0, c0), color in zip(hollow_positions, hollow_colors):
        paste(grid, _hollow_patch(color), r0, c0)
    for (r0, c0), color in zip(solid_positions, solid_colors):
        paste(grid, _solid_patch(color), r0, c0)
    return grid


def _bboxes_have_gap(objects: list[dict[str, object]]) -> bool:
    for i, obj_a in enumerate(objects):
        ar0, ac0, ar1, ac1 = map(int, obj_a["bbox"])
        for obj_b in objects[i + 1 :]:
            br0, bc0, br1, bc1 = map(int, obj_b["bbox"])
            separated = ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0
            if not separated:
                return False
    return True


def _validate_instance(grid: np.ndarray, case: dict[str, object]) -> np.ndarray | None:
    try:
        bg, objects = _extract_objects(grid)
        if bg != Color.BLACK:
            return None
        if not (4 <= len(objects) <= 6):
            return None

        hollows = [obj for obj in objects if obj["kind"] == "hollow"]
        solids = [obj for obj in objects if obj["kind"] == "solid"]
        if len(hollows) != int(case["n_hollow"]) or len(solids) < 1:
            return None
        if not _bboxes_have_gap(objects):
            return None

        hollow_rows = [int(obj["bbox"][0]) for obj in hollows]
        hollow_cols = [int(obj["bbox"][1]) for obj in hollows]
        if max(hollow_cols) - min(hollow_cols) <= max(hollow_rows) - min(hollow_rows):
            return None
        if bool(case["require_tie"]) and len(set(hollow_cols)) == len(hollow_cols):
            return None

        hollow_colors = [int(obj["color"]) for obj in hollows]
        solid_colors = [int(obj["color"]) for obj in solids]
        if len(set(hollow_colors)) < 2:
            return None
        if set(hollow_colors).isdisjoint(solid_colors):
            return None
        if not any(
            min(hollow_cols) < int(obj["bbox"][1]) < max(hollow_cols) or int(obj["bbox"][1]) < min(hollow_cols)
            for obj in solids
        ):
            return None

        out = main(grid)
        expected = _render_strip(len(hollows))
        if not np.array_equal(out, expected):
            return None
        if out.shape != (OBJ_SIZE, OBJ_SIZE * len(hollows)):
            return None
        if set(map(int, np.unique(out))) != {Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK}:
            return None
        return grid
    except AssertionError:
        return None


def _gen_attempt(case: dict[str, object]) -> np.ndarray | None:
    rng = random.Random()
    n_hollow = int(case["n_hollow"])
    n_solid = int(rng.choice(tuple(case["n_solid_choices"])))
    layout = _sample_layout(rng, n_hollow, n_solid, bool(case["require_tie"]))
    if layout is None:
        return None
    height, width, hollow_positions, solid_positions = layout
    hollow_colors, solid_colors = _sample_colors(rng, n_hollow, n_solid)
    grid = _render_input(
        height,
        width,
        hollow_positions,
        solid_positions,
        hollow_colors,
        solid_colors,
    )
    return _validate_instance(grid, case)


def generate_input() -> np.ndarray:
    case = _next_case()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(case)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input for case {case}")

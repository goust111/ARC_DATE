from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    assert horizontal ^ vertical, "each object must have exactly one of horizontal/vertical reflection symmetry"
    return "H" if horizontal else "V"


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    assert 3 <= len(objects) <= 4, "expected 3-4 foreground objects"

    colors = [int(obj["color"]) for obj in objects]
    assert len(colors) == len(set(colors)), "foreground objects must use distinct colors"

    parsed: list[dict] = []
    for obj in objects:
        color = int(obj["color"])
        bbox = tuple(map(int, obj["bbox"]))
        coords = np.asarray(obj["coords"], dtype=int)
        mask = crop_bbox(g == color, bbox)
        kind = _symmetry_kind(mask)
        assert 3 <= int(obj["area"]) <= 6, "object size must be within 3-6 pixels"
        parsed.append(
            {
                "color": color,
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "kind": kind,
            }
        )

    parsed.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

    for i, left in enumerate(parsed):
        lr0, lc0, lr1, lc1 = left["bbox"]
        for right in parsed[i + 1 :]:
            rr0, rc0, rr1, rc1 = right["bbox"]
            row_disjoint = lr1 < rr0 or rr1 < lr0
            col_disjoint = lc1 < rc0 or rc1 < lc0
            assert row_disjoint and col_disjoint, "object bbox row/col ranges must be pairwise disjoint"

    kinds = {obj["kind"] for obj in parsed}
    assert {"H", "V"} <= kinds, "expected at least one horizontal and one vertical object"
    return parsed


def _ray_cells(obj: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    coords = np.asarray(obj["coords"], dtype=int)
    if obj["kind"] == "H":
        for c in sorted(set(map(int, coords[:, 1]))):
            for r in range(h):
                cells.add((r, c))
    else:
        for r in sorted(set(map(int, coords[:, 0]))):
            for c in range(w):
                cells.add((r, c))
    return cells


def _spec_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects = _parse_objects(g)
    h, w = map(int, g.shape)

    out = np.full_like(g, Color.BLACK)
    coverage = np.zeros((len(objects), h, w), dtype=bool)

    for idx, obj in enumerate(objects):
        cells = _ray_cells(obj, g.shape)
        rr, cc = zip(*sorted(cells))
        coverage[idx, rr, cc] = True
        out[rr, cc] = int(obj["color"])

    out[coverage.sum(axis=0) >= 2] = Color.YELLOW

    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit axis-perpendicular full-length rays from symmetric objects and mark multi-object ray crossings yellow.

    Input structure:
    - A black background contains 3-4 separated 4-connected monochrome foreground objects in distinct colors.
    - Each object has exactly one reflection symmetry type: horizontal or vertical.
    - The objects' bounding-box row spans and column spans are pairwise disjoint, so object identity and symmetry are unambiguous.

    Rule:
    - Detect every foreground object and classify it by reflection symmetry inside its own bounding box.
    - A horizontally symmetric object emits vertical rays in its color through every one of its occupied pixel columns, reaching both grid edges.
    - A vertically symmetric object emits horizontal rays in its color through every one of its occupied pixel rows, reaching both grid edges.
    - Draw all rays in object order sorted by bounding-box top row then left column; later objects overwrite earlier ray colors on shared cells.
    - Recolor every cell visited by rays from two or more different objects to yellow, then redraw the original objects on top in their original colors.

    Key insight:
    - The symmetry axis does not choose a new color; it chooses the perpendicular ray direction, and only the cross-object ray intersections become yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= set(Color.ALL_COLORS), "unexpected color id"
    return _spec_output(g)


H_TEMPLATES = [
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
]

V_TEMPLATES = [
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

MATCHED_PAIRS = [
    (H_TEMPLATES[2], V_TEMPLATES[2]),
    (H_TEMPLATES[3], V_TEMPLATES[3]),
]

COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_DIM_RANGE = (14, 23)
EXTRA_MARGIN_RANGE = (0, 2)
INTER_GAP_RANGE = (1, 3)
MAX_GEN_ATTEMPTS = 500


def _sample_template(kind: str) -> np.ndarray:
    mask = random.choice(H_TEMPLATES if kind == "H" else V_TEMPLATES).copy()
    if kind == "H" and random.choice([False, True]):
        mask = np.fliplr(mask)
    if kind == "V" and random.choice([False, True]):
        mask = np.flipud(mask)
    return np.array(mask, dtype=bool)


def _choose_masks() -> list[np.ndarray]:
    object_count = random.randint(3, 4)
    pair_h, pair_v = random.choice(MATCHED_PAIRS)
    base_masks = [pair_h.copy(), pair_v.copy()]

    remaining = object_count - 2
    extra_kinds = random.choices(["H", "V"], k=remaining)
    if remaining == 2 and extra_kinds[0] == extra_kinds[1]:
        extra_kinds[1] = "V" if extra_kinds[0] == "H" else "H"

    for kind in extra_kinds:
        base_masks.append(_sample_template(kind))

    random.shuffle(base_masks)
    varied: list[np.ndarray] = []
    for mask in base_masks:
        if np.array_equal(mask, mask[::-1, :]) and random.choice([False, True]):
            mask = np.fliplr(mask)
        if np.array_equal(mask, mask[:, ::-1]) and random.choice([False, True]):
            mask = np.flipud(mask)
        varied.append(np.array(mask, dtype=bool))
    return varied


def _layout_positions(masks: list[np.ndarray]) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]

    top_margin = random.randint(*EXTRA_MARGIN_RANGE)
    left_margin = random.randint(*EXTRA_MARGIN_RANGE)
    row_gaps = [random.randint(*INTER_GAP_RANGE) for _ in range(len(masks) - 1)]
    col_gaps = [random.randint(*INTER_GAP_RANGE) for _ in range(len(masks) - 1)]
    bottom_margin = random.randint(*EXTRA_MARGIN_RANGE)
    right_margin = random.randint(*EXTRA_MARGIN_RANGE)

    needed_h = top_margin + sum(heights) + sum(row_gaps) + bottom_margin
    needed_w = left_margin + sum(widths) + sum(col_gaps) + right_margin
    if needed_h > GRID_DIM_RANGE[1] or needed_w > GRID_DIM_RANGE[1]:
        return None

    grid_h = random.randint(max(GRID_DIM_RANGE[0], needed_h), GRID_DIM_RANGE[1])
    grid_w = random.randint(max(GRID_DIM_RANGE[0], needed_w), GRID_DIM_RANGE[1])

    extra_h = grid_h - needed_h
    extra_w = grid_w - needed_w
    top_margin += random.randint(0, extra_h)
    left_margin += random.randint(0, extra_w)

    positions: list[tuple[int, int]] = []
    r = top_margin
    c = left_margin
    for idx, mask in enumerate(masks):
        positions.append((r, c))
        r += int(mask.shape[0])
        c += int(mask.shape[1])
        if idx < len(masks) - 1:
            r += row_gaps[idx]
            c += col_gaps[idx]

    if positions[-1][0] + heights[-1] > grid_h or positions[-1][1] + widths[-1] > grid_w:
        return None

    return (grid_h, grid_w), positions


def _render_input(masks: list[np.ndarray], positions: list[tuple[int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for mask, (r0, c0), color in zip(masks, positions, colors):
        h, w = mask.shape
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = int(color)
        grid[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
    return grid


def _input_semantics_ok(grid: np.ndarray) -> bool:
    try:
        objects = _parse_objects(grid)
    except AssertionError:
        return False

    kinds = [obj["kind"] for obj in objects]
    if "H" not in kinds or "V" not in kinds:
        return False

    same_size_mixed_pair = False
    for i, left in enumerate(objects):
        lh, lw = left["mask"].shape
        for right in objects[i + 1 :]:
            rh, rw = right["mask"].shape
            if left["kind"] != right["kind"] and (lh, lw) == (rh, rw):
                same_size_mixed_pair = True
                break
        if same_size_mixed_pair:
            break
    if not same_size_mixed_pair:
        return False

    coverage = np.zeros(grid.shape, dtype=int)
    for obj in objects:
        for r, c in _ray_cells(obj, grid.shape):
            coverage[r, c] += 1
    visible_yellow = (coverage >= 2) & (grid == Color.BLACK)
    if not visible_yellow.any():
        return False
    if int(visible_yellow.sum()) > grid.size // 3:
        return False

    out = _spec_output(grid)
    if not np.array_equal(main(grid), out):
        return False
    if not np.any(out == Color.YELLOW):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    masks = _choose_masks()
    layout = _layout_positions(masks)
    if layout is None:
        return None

    shape, positions = layout
    colors = random.sample(COLOR_POOL, len(masks))
    grid = _render_input(masks, positions, colors, shape)

    if not _input_semantics_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-ray ARC input")

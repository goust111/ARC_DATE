from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

bbox = _ARC_UTILS.bbox
colors_present = _ARC_UTILS.colors_present
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
label_mask = _ARC_UTILS.label_mask
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
OBJECT_PADDING = 1
OBJECT_BORDER = 1
OBJECT_COUNT_CHOICES = (2, 3)
ROOM_COUNTS = (1, 2, 3, 4)
CELL_SIZE_RANGE = (2, 3)
GRID_SIZE_BY_OBJECTS = {
    2: ((18, 30), (18, 30)),
    3: ((24, 30), (24, 30)),
}


MOTIF_LIBRARY = [
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=int,
    ),
]


def _rotate_clockwise(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -int(turns) % 4).copy()


def _count_rooms(walls: np.ndarray) -> int:
    labels, n_labels = label_mask(~walls.astype(bool), connectivity=4)
    h, w = walls.shape
    count = 0
    for k in range(1, n_labels + 1):
        coords = np.argwhere(labels == k)
        assert coords.size > 0
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_edge:
            count += 1
    return int(count)


def _is_frame_patch(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or min(mask.shape) < 3:
        return False
    return bool(
        np.all(mask[0, :])
        and np.all(mask[-1, :])
        and np.all(mask[:, 0])
        and np.all(mask[:, -1])
        and _count_rooms(mask) > 0
    )


def _frame_candidates_for_color(grid: np.ndarray, color: int, bg: int) -> list[tuple[dict, int]]:
    candidates: list[tuple[dict, int]] = []
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    for comp in comps:
        patch_mask = crop_bbox(grid == int(color), comp["bbox"]).astype(bool)
        if not _is_frame_patch(patch_mask):
            continue
        candidates.append((comp, _count_rooms(patch_mask)))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each same-color motif in place by the number of rooms enclosed inside its paired partition frame.

    Input structure:
    - A black background contains several colored rectangular frame components whose outer borders and internal partition walls use one uniform color each.
    - Elsewhere on the grid, each frame color appears again as one detached motif whose bounding box is square and does not touch any frame.

    Rule:
    - For each non-background color, identify the frame component as the same-color connected component whose cropped mask has a complete rectangular border and encloses one or more 4-connected background rooms.
    - Count the enclosed rooms inside that frame.
    - Take every remaining same-color pixel as the paired motif, crop its square bounding box, rotate that square clockwise by 90 degrees times the room count modulo 4, and write it back in place.
    - Keep every frame and wall pixel unchanged.

    Key insight:
    - The frame is not edited directly; it functions only as a same-color dial whose room count determines how far the detached motif must rotate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    out = g.copy()
    for color in colors_present(g, exclude={bg}):
        frame_candidates = _frame_candidates_for_color(g, int(color), bg)
        assert len(frame_candidates) == 1, "each color must have exactly one qualifying frame"
        frame_comp, room_count = frame_candidates[0]

        motif_mask = (g == int(color))
        frame_coords = frame_comp["coords"]
        motif_mask[frame_coords[:, 0], frame_coords[:, 1]] = False
        motif_bbox = bbox(motif_mask)
        assert motif_bbox is not None, "each frame color must have a detached motif"
        r0, c0, r1, c1 = motif_bbox
        assert (r1 - r0) == (c1 - c0), "motif bbox must be square"

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert set(map(int, np.unique(patch))) <= {bg, int(color)}, "motif bbox must contain only motif color and background"
        out[r0 : r1 + 1, c0 : c1 + 1] = _rotate_clockwise(patch, room_count)

    return out


def _factor_options(room_count: int) -> list[tuple[int, int]]:
    out = []
    for rows in range(1, room_count + 1):
        if room_count % rows == 0:
            out.append((rows, room_count // rows))
    return out


def _sample_room_grid(room_count: int) -> tuple[int, int]:
    options = _factor_options(room_count)
    weights = []
    for rows, cols in options:
        delta = abs(rows - cols)
        weights.append(4 - min(delta, 3))
    return random.choices(options, weights=weights, k=1)[0]


def _make_frame_patch(color: int, room_count: int) -> np.ndarray:
    room_rows, room_cols = _sample_room_grid(room_count)
    row_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(room_rows)]
    col_sizes = [random.randint(*CELL_SIZE_RANGE) for _ in range(room_cols)]

    h = sum(row_sizes) + room_rows + 1
    w = sum(col_sizes) + room_cols + 1
    patch = np.zeros((h, w), dtype=int)

    r_lines = [0]
    acc = 0
    for size in row_sizes:
        acc += size + 1
        r_lines.append(acc)

    c_lines = [0]
    acc = 0
    for size in col_sizes:
        acc += size + 1
        c_lines.append(acc)

    patch[r_lines, :] = int(color)
    patch[:, c_lines] = int(color)
    assert _count_rooms(patch == int(color)) == int(room_count)
    return patch


def _motif_templates() -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for tpl in MOTIF_LIBRARY:
        b = bbox(tpl.astype(bool))
        assert b == (0, 0, tpl.shape[0] - 1, tpl.shape[1] - 1)
        assert tpl.shape[0] == tpl.shape[1]
        assert not _is_frame_patch(tpl.astype(bool))
        rotations = {_rotate_clockwise(tpl, k).tobytes() for k in range(4)}
        assert len(rotations) == 4, "motif template must expose all rotation states"
        templates.append(tpl.copy())
    return templates


VALID_MOTIFS = _motif_templates()


def _make_motif_patch(color: int) -> np.ndarray:
    patch = random.choice(VALID_MOTIFS).copy()
    patch = _rotate_clockwise(patch, random.randrange(4))
    return np.where(patch != 0, int(color), BG).astype(int)


def _sample_room_counts(n_objects: int) -> list[int]:
    return random.sample(list(ROOM_COUNTS), k=int(n_objects))


def _reserve_and_paint(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int]:
    block = np.ones(patch.shape, dtype=int)
    r0, c0 = random_free_location(
        reserve,
        block,
        transparent=BG,
        border=OBJECT_BORDER,
        padding=OBJECT_PADDING,
        padding_connectivity=8,
    )
    paste(reserve, block, r0, c0, transparent=BG)
    paste(grid, patch, r0, c0, transparent=BG)
    return int(r0), int(c0)


def _well_formed_instance(grid: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    if bg != BG:
        return False
    for color in colors_present(grid, exclude={bg}):
        frame_candidates = _frame_candidates_for_color(grid, int(color), bg)
        if len(frame_candidates) != 1:
            return False
        frame_comp, _room_count = frame_candidates[0]
        motif_mask = (grid == int(color))
        frame_coords = frame_comp["coords"]
        motif_mask[frame_coords[:, 0], frame_coords[:, 1]] = False
        motif_bbox = bbox(motif_mask)
        if motif_bbox is None:
            return False
        r0, c0, r1, c1 = motif_bbox
        if (r1 - r0) != (c1 - c0):
            return False
        if not set(map(int, np.unique(grid[r0 : r1 + 1, c0 : c1 + 1]))) <= {bg, int(color)}:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choices(OBJECT_COUNT_CHOICES, weights=[1, 2], k=1)[0]
    colors = random.sample(PALETTE, k=n_objects)
    room_counts = _sample_room_counts(n_objects)

    pair_specs = []
    for color, room_count in zip(colors, room_counts):
        frame_patch = _make_frame_patch(int(color), int(room_count))
        motif_patch = _make_motif_patch(int(color))
        pair_specs.append(
            {
                "color": int(color),
                "room_count": int(room_count),
                "frame_patch": frame_patch,
                "motif_patch": motif_patch,
            }
        )

    max_h = max(max(spec["frame_patch"].shape[0], spec["motif_patch"].shape[0]) for spec in pair_specs) + 2
    max_w = max(max(spec["frame_patch"].shape[1], spec["motif_patch"].shape[1]) for spec in pair_specs) + 2
    row_range, col_range = GRID_SIZE_BY_OBJECTS[n_objects]
    min_h = max(row_range[0], max_h)
    min_w = max(col_range[0], max_w)
    if min_h > row_range[1] or min_w > col_range[1]:
        return None

    h = random.randint(min_h, row_range[1])
    w = random.randint(min_w, col_range[1])
    grid = np.full((h, w), BG, dtype=int)
    reserve = np.full((h, w), BG, dtype=int)

    placements: dict[tuple[int, str], tuple[int, int]] = {}
    items = []
    for idx, spec in enumerate(pair_specs):
        items.append((int(spec["frame_patch"].size), random.random(), idx, "frame", spec["frame_patch"]))
        items.append((int(spec["motif_patch"].size), random.random(), idx, "motif", spec["motif_patch"]))
    items.sort(key=lambda item: (-item[0], item[1]))

    try:
        for _area, _tie, idx, kind, patch in items:
            placements[(idx, kind)] = _reserve_and_paint(grid, reserve, patch)
    except ValueError:
        return None

    expected = grid.copy()
    changed = False
    for idx, spec in enumerate(pair_specs):
        mr, mc = placements[(idx, "motif")]
        motif_patch = spec["motif_patch"]
        rotated = _rotate_clockwise(motif_patch, spec["room_count"])
        expected[mr : mr + motif_patch.shape[0], mc : mc + motif_patch.shape[1]] = rotated
        changed |= not np.array_equal(rotated, motif_patch)

    if not changed:
        return None
    if not _well_formed_instance(grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")

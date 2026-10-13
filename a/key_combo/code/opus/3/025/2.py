from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import (  # noqa: E402
    Color,
    bbox_anchor,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)


BG = Color.BLACK
GREY = Color.GREY
ORANGE = Color.ORANGE
HOLE_TO_DIRECTION = {
    0: (0, 1),
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0),
}
HOLE_TO_COLOR = {
    0: Color.BLUE,
    1: Color.RED,
    2: Color.GREEN,
    3: Color.YELLOW,
}

GEN_ATTEMPTS = 400
GRID_MIN = 20
GRID_MAX = 30
OUTER_MARGIN = 1
GAP_RANGE = (2, 4)


def _mask(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


TEMPLATE_LIBRARY: dict[int, tuple[np.ndarray, ...]] = {
    0: (
        _mask(("11111", "01110", "00100")),
        _mask(("00100", "01110", "11111", "00100", "00100")),
        _mask(("010", "111", "010", "111", "010")),
    ),
    1: (
        _mask(("11111", "10001", "11111")),
        _mask(("111", "101", "101", "101", "111")),
        _mask(("11111", "10001", "10101", "10001", "11111")),
    ),
    2: (
        _mask(("11111", "10101", "11111")),
        _mask(("111", "101", "111", "101", "111")),
        _mask(("11111", "10001", "11111", "10001", "11111")),
    ),
    3: (
        _mask(("1111111", "1010101", "1111111")),
        _mask(("111", "101", "111", "101", "111", "101", "111")),
        _mask(("11111", "10101", "11111", "10001", "11111")),
    ),
}


LAYOUT_CASES = (
    (
        {"hole": 3, "row_rank": 0, "col_rank": 0},
        {"hole": 0, "row_rank": 1, "col_rank": 1},
        {"hole": 1, "row_rank": 2, "col_rank": 2},
        {"hole": 2, "row_rank": 3, "col_rank": 3},
    ),
    (
        {"hole": 3, "row_rank": 0, "col_rank": 1},
        {"hole": 0, "row_rank": 1, "col_rank": 0},
        {"hole": 1, "row_rank": 2, "col_rank": 2},
    ),
    (
        {"hole": 3, "row_rank": 0, "col_rank": 0},
        {"hole": 2, "row_rank": 1, "col_rank": 2},
        {"hole": 1, "row_rank": 2, "col_rank": 1},
    ),
    (
        {"hole": 0, "row_rank": 0, "col_rank": 0},
        {"hole": 2, "row_rank": 1, "col_rank": 2},
        {"hole": 1, "row_rank": 2, "col_rank": 1},
    ),
    (
        {"hole": 3, "row_rank": 0, "col_rank": 1},
        {"hole": 0, "row_rank": 1, "col_rank": 0},
        {"hole": 2, "row_rank": 2, "col_rank": 2},
    ),
)
LAYOUT_WEIGHTS = (12, 1, 1, 1, 1)


def _hole_count(mask: np.ndarray) -> int:
    """Count 4-connected enclosed holes inside a cropped sprite mask."""
    holes = 0
    h, w = mask.shape
    for comp in components_mask(~mask, connectivity=4):
        rows = comp[:, 0]
        cols = comp[:, 1]
        touches_border = bool(
            np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
        )
        if not touches_border:
            holes += 1
    return holes


def _sprite_infos(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    infos: list[dict] = []
    for comp in comps:
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(grid, bbox) == GREY
        holes = _hole_count(patch)
        infos.append(
            {
                "bbox": bbox,
                "coords": comp["coords"],
                "holes": int(holes),
                "patch": patch,
            }
        )
    return infos


def _ray_start(bbox: tuple[int, int, int, int], holes: int) -> tuple[int, int]:
    dr, dc = HOLE_TO_DIRECTION[holes]
    if (dr, dc) == (0, 1):
        r, c = bbox_anchor(bbox, anchor="right center")
        return int(r), int(c + 1)
    if (dr, dc) == (0, -1):
        r, c = bbox_anchor(bbox, anchor="left center")
        return int(r), int(c - 1)
    if (dr, dc) == (-1, 0):
        r, c = bbox_anchor(bbox, anchor="upper center")
        return int(r - 1), int(c)
    r, c = bbox_anchor(bbox, anchor="lower center")
    return int(r + 1), int(c)


def _ray_cells(
    shape: tuple[int, int],
    sprite_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    holes: int,
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = HOLE_TO_DIRECTION[holes]
    r, c = _ray_start(bbox, holes)
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if bool(sprite_mask[r, c]):
            break
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _render_output(
    shape: tuple[int, int],
    sprite_specs: list[dict],
    *,
    background: int = BG,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    out = np.full(shape, int(background), dtype=int)
    sprite_mask = np.zeros(shape, dtype=bool)

    for spec in sprite_specs:
        patch = spec["patch"]
        r0 = int(spec["bbox"][0])
        c0 = int(spec["bbox"][1])
        colored = np.where(patch, HOLE_TO_COLOR[int(spec["holes"])], background)
        paste(out, colored, r0, c0, transparent=background)
        paste(sprite_mask, patch, r0, c0, transparent=False)

    visits: dict[tuple[int, int], int] = {}
    for spec in sprite_specs:
        for cell in _ray_cells(shape, sprite_mask, spec["bbox"], int(spec["holes"])):
            visits[cell] = visits.get(cell, 0) + 1

    for (r, c), count in visits.items():
        out[r, c] = ORANGE if count >= 2 else GREY
    return out, visits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey sprite by its hole count and emit a hole-coded ray whose intersections turn orange.

    Input structure:
    - A black background contains 3-4 separated grey 4-connected sprites.
    - Each sprite has 0, 1, 2, or 3 enclosed black holes inside its own tight bounding box.
    - Sprite bounding boxes are separated so rays can leave from the bbox side midpoint without ambiguity.

    Rule:
    - Count each grey sprite's enclosed holes H using 4-connectivity on the background inside that sprite's bbox.
    - Recolor the sprite body only: H=0 -> blue, H=1 -> red, H=2 -> green, H=3 -> yellow.
    - Also emit a grey ray from the midpoint just outside the corresponding bbox side: H=0 right, H=1 up, H=2 left, H=3 down.
    - A ray advances cell by cell until the grid border or a grey sprite pixel blocks it; cells visited by two different rays become orange.

    Key insight:
    - The same topological measurement, the number of enclosed holes, simultaneously determines both the sprite's new identity color and its ray direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    infos = _sprite_infos(g)
    assert infos, "expected at least one grey sprite"
    for info in infos:
        assert int(info["holes"]) in HOLE_TO_DIRECTION, "hole count must be in 0..3"
    out, _visits = _render_output(g.shape, infos, background=bg)
    return out


def _random_patch(hole_count: int) -> np.ndarray:
    mask = TEMPLATE_LIBRARY[hole_count][random.randrange(len(TEMPLATE_LIBRARY[hole_count]))].copy()
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    assert _hole_count(mask) == hole_count
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = GREY
    return patch


def _axis_layout(lengths: list[int]) -> tuple[int, list[int]] | None:
    slots = [OUTER_MARGIN]
    for _ in range(len(lengths) - 1):
        slots.append(random.randint(*GAP_RANGE))
    slots.append(OUTER_MARGIN)
    base = sum(lengths) + sum(slots)
    min_size = max(GRID_MIN, base)
    if min_size > GRID_MAX:
        return None
    size = random.randint(min_size, GRID_MAX)
    slack = size - base
    for _ in range(slack):
        slots[random.randrange(len(slots))] += 1

    starts: list[int] = []
    cursor = slots[0]
    for i, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length + slots[i + 1]
    return size, starts


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _validate_bboxes(specs: list[dict]) -> bool:
    for i, a in enumerate(specs):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in specs[i + 1 :]:
            br0, bc0, br1, bc1 = b["bbox"]
            if _intervals_overlap(ar0, ar1, br0, br1):
                return False
            if _intervals_overlap(ac0, ac1, bc0, bc1):
                return False
    return True


def _choose_layout_case() -> tuple[dict, ...]:
    return random.choices(LAYOUT_CASES, weights=LAYOUT_WEIGHTS, k=1)[0]


def _scene_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    layout = _choose_layout_case()
    patches = [_random_patch(int(role["hole"])) for role in layout]

    row_order = sorted(range(len(layout)), key=lambda idx: int(layout[idx]["row_rank"]))
    col_order = sorted(range(len(layout)), key=lambda idx: int(layout[idx]["col_rank"]))

    row_lengths = [int(patches[idx].shape[0]) for idx in row_order]
    col_lengths = [int(patches[idx].shape[1]) for idx in col_order]
    row_layout = _axis_layout(row_lengths)
    col_layout = _axis_layout(col_lengths)
    if row_layout is None or col_layout is None:
        return None
    H, row_starts = row_layout
    W, col_starts = col_layout

    top_by_index = {idx: row_starts[pos] for pos, idx in enumerate(row_order)}
    left_by_index = {idx: col_starts[pos] for pos, idx in enumerate(col_order)}

    grid = np.full((H, W), BG, dtype=int)
    sprite_specs: list[dict] = []
    for idx, (role, patch) in enumerate(zip(layout, patches)):
        r0 = int(top_by_index[idx])
        c0 = int(left_by_index[idx])
        paste(grid, patch, r0, c0, transparent=BG)
        h, w = patch.shape
        sprite_specs.append(
            {
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "holes": int(role["hole"]),
                "patch": patch == GREY,
            }
        )

    if not _validate_bboxes(sprite_specs):
        return None

    observed = _sprite_infos(grid)
    observed_holes = sorted(int(info["holes"]) for info in observed)
    expected_holes = sorted(int(spec["holes"]) for spec in sprite_specs)
    if observed_holes != expected_holes:
        return None
    if len(set(observed_holes)) != len(observed_holes):
        return None
    if len(observed_holes) not in (3, 4):
        return None

    expected, visits = _render_output(grid.shape, sprite_specs, background=BG)
    if sum(count >= 2 for count in visits.values()) < 2:
        return None
    if not any(count >= 2 for count in visits.values()):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _scene_attempt()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid hole-ray puzzle")


for _hole_value, _templates in TEMPLATE_LIBRARY.items():
    for _template in _templates:
        assert _template.shape[0] % 2 == 1 and _template.shape[1] % 2 == 1
        assert 8 <= int(_template.sum()) <= 20
        assert _hole_count(_template) == _hole_value

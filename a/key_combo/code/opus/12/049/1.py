from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox_coords, components_mask


GRID_MIN = 16
GRID_MAX = 22
MAX_GENERATION_ATTEMPTS = 400
PALETTE = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY)

SHAPE_H1 = (
    np.array([[1, 1, 1]], dtype=int),
)
SHAPE_H3 = (
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=int),
)


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _shape_coords(shape: np.ndarray) -> np.ndarray:
    return np.argwhere(shape.astype(bool))


def _shape_center_offset(shape: np.ndarray) -> tuple[int, int]:
    h, w = shape.shape
    return (h - 1) // 2, (w - 1) // 2


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 0] += int(dr)
    moved[:, 1] += int(dc)
    return moved


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _barrier_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == Color.RED)]
    assert len(rows) == 1, "expected exactly one full-width red barrier row"
    return rows[0]


def _parse_objects(grid: np.ndarray) -> dict[int, list[np.ndarray]]:
    bar_r = _barrier_row(grid)
    allowed = {Color.BLACK, Color.RED, *PALETTE}
    assert set(map(int, np.unique(grid))).issubset(allowed), "unexpected colors in grid"

    parsed: dict[int, list[np.ndarray]] = {}
    for color in PALETTE:
        comps = components_mask(grid == int(color), connectivity=4)
        assert len(comps) == 2, f"expected exactly two objects of color {color}"
        parsed[int(color)] = comps

    for color, comps in parsed.items():
        for coords in comps:
            r0, _c0, r1, _c1 = bbox_coords(coords)
            assert r1 < bar_r or r0 > bar_r, f"object of color {color} must stay on one side of the barrier"
    return parsed


def _slide_to_barrier(coords: np.ndarray, bar_r: int) -> np.ndarray:
    r0, _c0, r1, _c1 = bbox_coords(coords)
    if r1 < bar_r:
        dr = (bar_r - 1) - r1
    else:
        assert r0 > bar_r, "object must be strictly above or below the barrier"
        dr = (bar_r + 1) - r0
    return _translate(coords, dr, 0)


def _priority_key(centers: list[tuple[int, int]], bar_r: int) -> tuple[int, int]:
    distances = [abs(r - bar_r) for r, _c in centers]
    min_dist = min(distances)
    tie_col = min(c for (r, c), d in zip(centers, distances) if d == min_dist)
    return min_dist, tie_col


def _fill_rectangles(
    base: np.ndarray,
    pair_infos: list[dict[str, object]],
) -> np.ndarray:
    out = base.copy()
    for info in sorted(pair_infos, key=lambda item: item["priority"]):
        color = int(info["color"])
        (r0, c0), (r1, c1) = info["centers"]
        rr0, rr1 = sorted((int(r0), int(r1)))
        cc0, cc1 = sorted((int(c0), int(c1)))
        patch = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        patch[patch == Color.BLACK] = color
    return out


def _scene_after_slide(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], dict[int, list[np.ndarray]], dict[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bar_r = _barrier_row(g)
    objects = _parse_objects(g)

    base = np.zeros_like(g)
    base[bar_r, :] = Color.RED
    moved_by_color: dict[int, list[np.ndarray]] = {}
    moved_centers: dict[int, list[tuple[int, int]]] = {}

    occupied = np.zeros(g.shape, dtype=bool)
    occupied[bar_r, :] = True
    for color, comps in objects.items():
        moved_list = []
        center_list = []
        for coords in comps:
            moved = _slide_to_barrier(coords, bar_r)
            assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < g.shape[0]) & (0 <= moved[:, 1]) & (moved[:, 1] < g.shape[1]))
            assert not np.any(occupied[moved[:, 0], moved[:, 1]]), "slid objects must not overlap"
            occupied[moved[:, 0], moved[:, 1]] = True
            _paint_coords(base, moved, color)
            moved_list.append(moved)
            center_list.append(_bbox_center(moved))

        assert len(moved_list) == 2
        assert center_list[0][0] != center_list[1][0], "paired centers must not share a row after sliding"
        assert center_list[0][1] != center_list[1][1], "paired centers must not share a column after sliding"
        moved_by_color[color] = moved_list
        moved_centers[color] = center_list

    pair_infos = []
    for color in PALETTE:
        centers = moved_centers[int(color)]
        pair_infos.append(
            {
                "color": int(color),
                "centers": tuple(centers),
                "priority": _priority_key(centers, bar_r),
            }
        )
    return base, pair_infos, moved_by_color, moved_centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored object onto the red barrier, then pair same-color endpoints to fill rectangles.

    Input structure:
    - The grid has a black background and one full-width horizontal red barrier row.
    - Four non-red foreground colors each appear as exactly two disconnected 4-connected objects.
    - Every colored object lies entirely above or below the barrier, with empty vertical space to slide toward it.

    Rule:
    - Keep the red barrier fixed.
    - Slide each object vertically toward the barrier until its nearest bbox row is adjacent to the red row, without overlap.
    - Erase the original object positions and draw the moved objects at their stopping positions.
    - For each color, take the bbox-center pixel of its two moved objects as diagonal rectangle corners, and fill only black cells inside that inclusive rectangle.
    - If two color rectangles overlap, the pair whose moved center is closer to the barrier fills first; ties use the leftmost such center column first.

    Key insight:
    - The rectangle corners are defined by the post-slide endpoints, so the motion step must be solved before the geometric pairing and fill step becomes knowable.
    """
    base, pair_infos, _moved_by_color, _moved_centers = _scene_after_slide(grid)
    return _fill_rectangles(base, pair_infos)


def _center_slots(width: int) -> list[int]:
    slots = [int(round((i + 1) * (width - 1) / 9.0)) for i in range(8)]
    out: list[int] = []
    last = 0
    for x in slots:
        x = min(max(1, x), width - 2)
        if out and x <= last:
            x = min(width - 2, last + 1)
        out.append(x)
        last = x
    assert len(set(out)) == 8
    return out


def _pick_shape(category: str, rng: random.Random) -> np.ndarray:
    if category == "h1":
        return rng.choice(SHAPE_H1).copy()
    assert category == "h3"
    return rng.choice(SHAPE_H3).copy()


def _place_shape(
    shape: np.ndarray,
    side: str,
    center_col: int,
    bar_r: int,
    height: int,
) -> np.ndarray:
    coords = _shape_coords(shape)
    center_r_off, center_c_off = _shape_center_offset(shape)
    h, _w = shape.shape
    assert h == int(height)
    if side == "top":
        top = bar_r - h
    else:
        assert side == "bottom"
        top = bar_r + 1
    left = int(center_col) - center_c_off
    return _translate(coords, top, left)


def _rect_bounds(centers: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    (r0, c0), (r1, c1) = centers
    return min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1)


def _cells_from_rect(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _sweep_cells(start: np.ndarray, end: np.ndarray) -> set[tuple[int, int]]:
    delta = int(end[0, 0] - start[0, 0])
    assert np.all(end[:, 1] == start[:, 1]), "movement must be vertical"
    step = 0 if delta == 0 else (1 if delta > 0 else -1)
    swept: set[tuple[int, int]] = set()
    cur = start.copy()
    for _ in range(abs(delta) + 1):
        swept |= {(int(r), int(c)) for r, c in cur}
        cur = _translate(cur, step, 0)
    return swept


def _base_from_moved(
    shape_infos: list[dict[str, object]],
    height: int,
    width: int,
    bar_r: int,
) -> np.ndarray:
    base = np.zeros((height, width), dtype=int)
    base[bar_r, :] = Color.RED
    for info in shape_infos:
        _paint_coords(base, info["final_coords"], int(info["color"]))
    return base


def _wrong_static_output(
    shape_infos: list[dict[str, object]],
    bar_r: int,
    height: int,
    width: int,
) -> np.ndarray:
    base = _base_from_moved(shape_infos, height, width, bar_r)
    by_color: dict[int, list[tuple[int, int]]] = {int(color): [] for color in PALETTE}
    for info in shape_infos:
        by_color[int(info["color"])].append(tuple(info["input_center"]))
    pair_infos = []
    for color in PALETTE:
        centers = by_color[int(color)]
        pair_infos.append(
            {
                "color": int(color),
                "centers": tuple(centers),
                "priority": _priority_key(centers, bar_r),
            }
        )
    return _fill_rectangles(base, pair_infos)


def _validate_branch_semantics(
    inp: np.ndarray,
    mode: str,
    expected_roles: dict[str, int],
) -> bool:
    try:
        bar_r = _barrier_row(inp)
        out = main(inp)
        base, pair_infos, moved_by_color, moved_centers = _scene_after_slide(inp)
    except AssertionError:
        return False

    if np.array_equal(out, _wrong_static_output(
        [
            {
                "color": color,
                "final_coords": coords,
                "input_center": _bbox_center(orig),
            }
            for color, comps in _parse_objects(inp).items()
            for coords, orig in zip(moved_by_color[color], comps)
        ],
        bar_r,
        inp.shape[0],
        inp.shape[1],
    )):
        return False

    pair_by_color = {int(info["color"]): info for info in pair_infos}

    for role, color in expected_roles.items():
        centers = list(moved_centers[int(color)])
        same_side = len({("top" if r < bar_r else "bottom") for r, _c in centers}) == 1
        if role == "same_top" and not (same_side and all(r < bar_r for r, _c in centers)):
            return False
        if role == "same_bottom" and not (same_side and all(r > bar_r for r, _c in centers)):
            return False
        if role.startswith("split") and same_side:
            return False

    winner = int(expected_roles["split_a"])
    loser = int(expected_roles["split_b"])
    win_info = pair_by_color[winner]
    lose_info = pair_by_color[loser]
    win_box = _rect_bounds(list(win_info["centers"]))
    lose_box = _rect_bounds(list(lose_info["centers"]))
    overlap = _cells_from_rect(win_box) & _cells_from_rect(lose_box)
    visible_overlap = {(r, c) for r, c in overlap if int(base[r, c]) == Color.BLACK}
    if mode in {"distance", "column"} and not visible_overlap:
        return False
    if mode == "none" and visible_overlap:
        return False

    if mode == "distance":
        if tuple(win_info["priority"])[0] >= tuple(lose_info["priority"])[0]:
            return False
    if mode == "column":
        if tuple(win_info["priority"])[0] != tuple(lose_info["priority"])[0]:
            return False
        if tuple(win_info["priority"])[1] >= tuple(lose_info["priority"])[1]:
            return False

    if visible_overlap:
        sample_cell = min(visible_overlap)
        if int(out[sample_cell]) != int(
            winner
            if tuple(win_info["priority"]) < tuple(lose_info["priority"])
            else loser
        ):
            return False

    for color in PALETTE:
        if len(moved_by_color[int(color)]) != 2:
            return False
        centers = moved_centers[int(color)]
        if centers[0][0] == centers[1][0] or centers[0][1] == centers[1][1]:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    bar_r = rng.randint(height // 2 - 1, height // 2 + 1)
    if bar_r < 5 or bar_r > height - 6:
        return None

    mode = rng.choice(("none", "distance", "column"))
    slots = _center_slots(width)
    top_slots = [slots[0], slots[2], slots[4], slots[6]]
    bottom_slots = [slots[1], slots[3], slots[5], slots[7]]

    colors = list(PALETTE)
    rng.shuffle(colors)
    role_colors = {
        "split_a": colors[0],
        "split_b": colors[1],
        "same_top": colors[2],
        "same_bottom": colors[3],
    }

    role_specs: dict[str, list[tuple[str, int, str]]] = {}
    if mode == "distance":
        role_specs["split_a"] = [("top", top_slots[0], "h1"), ("bottom", bottom_slots[2], "h1")]
        role_specs["split_b"] = [("top", top_slots[1], "h3"), ("bottom", bottom_slots[3], "h3")]
        role_specs["same_top"] = [("top", top_slots[2], "h1"), ("top", top_slots[3], "h3")]
        role_specs["same_bottom"] = [("bottom", bottom_slots[0], "h1"), ("bottom", bottom_slots[1], "h3")]
    elif mode == "column":
        role_specs["split_a"] = [("top", top_slots[0], "h1"), ("bottom", bottom_slots[2], "h1")]
        role_specs["split_b"] = [("top", top_slots[1], "h1"), ("bottom", bottom_slots[3], "h1")]
        role_specs["same_top"] = [("top", top_slots[2], "h1"), ("top", top_slots[3], "h3")]
        role_specs["same_bottom"] = [("bottom", bottom_slots[0], "h1"), ("bottom", bottom_slots[1], "h3")]
    else:
        split_a_top = rng.choice(("h1", "h3"))
        split_a_bottom = rng.choice(("h1", "h3"))
        split_b_top = rng.choice(("h1", "h3"))
        split_b_bottom = rng.choice(("h1", "h3"))
        role_specs["split_a"] = [("top", top_slots[0], split_a_top), ("bottom", bottom_slots[0], split_a_bottom)]
        role_specs["split_b"] = [("top", top_slots[1], split_b_top), ("bottom", bottom_slots[1], split_b_bottom)]
        role_specs["same_top"] = [("top", top_slots[2], "h1"), ("top", top_slots[3], "h3")]
        role_specs["same_bottom"] = [("bottom", bottom_slots[2], "h1"), ("bottom", bottom_slots[3], "h3")]

    shape_infos: list[dict[str, object]] = []
    final_occupied = np.zeros((height, width), dtype=bool)
    final_occupied[bar_r, :] = True
    sweep_occupied: set[tuple[int, int]] = {(bar_r, c) for c in range(width)}

    for role in ("split_a", "split_b", "same_top", "same_bottom"):
        color = int(role_colors[role])
        for side, center_col, category in role_specs[role]:
            shape = _pick_shape(category, rng)
            final_coords = _place_shape(shape, side, center_col, bar_r, shape.shape[0])
            if not np.all((0 <= final_coords[:, 0]) & (final_coords[:, 0] < height) & (0 <= final_coords[:, 1]) & (final_coords[:, 1] < width)):
                return None
            if np.any(final_occupied[final_coords[:, 0], final_coords[:, 1]]):
                return None

            final_occupied[final_coords[:, 0], final_coords[:, 1]] = True
            if side == "top":
                max_gap = int(final_coords[:, 0].min())
                min_gap = 2 if max_gap >= 2 else 1
                if max_gap < 1:
                    return None
                gap = rng.randint(min_gap, max_gap)
                input_coords = _translate(final_coords, -gap, 0)
            else:
                max_gap = int(height - 1 - final_coords[:, 0].max())
                min_gap = 2 if max_gap >= 2 else 1
                if max_gap < 1:
                    return None
                gap = rng.randint(min_gap, max_gap)
                input_coords = _translate(final_coords, gap, 0)

            swept = _sweep_cells(input_coords, final_coords)
            if swept & sweep_occupied:
                return None
            sweep_occupied |= swept

            shape_infos.append(
                {
                    "role": role,
                    "color": color,
                    "side": side,
                    "shape": shape,
                    "final_coords": final_coords,
                    "input_coords": input_coords,
                    "input_center": _bbox_center(input_coords),
                    "final_center": _bbox_center(final_coords),
                    "gap": gap,
                }
            )

    inp = np.zeros((height, width), dtype=int)
    inp[bar_r, :] = Color.RED
    for info in shape_infos:
        _paint_coords(inp, info["input_coords"], int(info["color"]))

    try:
        parsed = _parse_objects(inp)
    except AssertionError:
        return None
    if any(len(comps) != 2 for comps in parsed.values()):
        return None

    try:
        unchanged = np.array_equal(main(inp), inp)
    except AssertionError:
        return None
    if unchanged:
        return None

    by_color: dict[int, list[dict[str, object]]] = {int(color): [] for color in PALETTE}
    for info in shape_infos:
        by_color[int(info["color"])].append(info)

    for color, infos in by_color.items():
        if len(infos) != 2:
            return None
        centers = [tuple(info["final_center"]) for info in infos]
        if centers[0][0] == centers[1][0] or centers[0][1] == centers[1][1]:
            return None

    if not _validate_branch_semantics(inp, mode, {k: int(v) for k, v in role_colors.items()}):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-pair rectangle input")

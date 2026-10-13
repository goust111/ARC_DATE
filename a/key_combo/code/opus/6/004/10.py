from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, get_bg


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
YELLOW = Color.YELLOW
PAIR_COLORS = (Color.GREEN, Color.GREY, Color.ORANGE)
MAX_ATTEMPTS = 400

SHAPE_CATALOG = (
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
)

CORNER_POS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}


def _rot_cw(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=3)


def _mask_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(components_mask(mask.astype(bool), connectivity=4)) == 1


def _mask_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _sprite_coords(sprite: np.ndarray, bg: int = BG) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(sprite != int(bg)))


def _axis_setup(grid: np.ndarray) -> dict[str, dict[str, int | str | tuple[int, int]]]:
    h, w = grid.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }
    marked = [name for name, (r, c) in corners.items() if int(grid[r, c]) == int(YELLOW)]
    assert len(marked) == 1, "expected one yellow corner marker"
    corner_name = marked[0]
    cr, cc = corners[corner_name]
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(grid[cr, cc + hstep])
    vcol = int(grid[cr + vstep, cc])
    assert {hcol, vcol} == {int(BLUE), int(RED)}

    row_side = "top" if cr == 0 else "bottom"
    col_side = "left" if cc == 0 else "right"
    row_axis = {
        "kind": "row",
        "side": row_side,
        "line_index": int(cr),
        "adj_index": 1 if cr == 0 else h - 2,
        "color": hcol,
        "corner": (int(cr), int(cc)),
    }
    col_axis = {
        "kind": "col",
        "side": col_side,
        "line_index": int(cc),
        "adj_index": 1 if cc == 0 else w - 2,
        "color": vcol,
        "corner": (int(cr), int(cc)),
    }
    return {
        "corner_name": corner_name,
        "corner": (int(cr), int(cc)),
        "horizontal": row_axis,
        "vertical": col_axis,
        "blue": row_axis if hcol == int(BLUE) else col_axis,
        "red": row_axis if hcol == int(RED) else col_axis,
    }


def _touches_axis(coords: np.ndarray, axis: dict[str, int | str | tuple[int, int]]) -> bool:
    adj = int(axis["adj_index"])
    if axis["kind"] == "row":
        return bool(np.any(coords[:, 0] == adj))
    return bool(np.any(coords[:, 1] == adj))


def _coords_set(coords: np.ndarray | tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        return frozenset((int(r), int(c)) for r, c in coords)
    return frozenset((int(r), int(c)) for r, c in coords)


def _completion_candidates(
    sprite: np.ndarray,
    anchor_cells: frozenset[tuple[int, int]],
    blocked: np.ndarray,
    *,
    bg: int = BG,
) -> list[frozenset[tuple[int, int]]]:
    h, w = sprite.shape
    grid_h, grid_w = blocked.shape
    cells = _sprite_coords(sprite, bg=bg)
    assert cells, "sprite must contain foreground"
    candidates: list[frozenset[tuple[int, int]]] = []

    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            moved = frozenset((r0 + r, c0 + c) for r, c in cells)
            if any(blocked[r, c] for r, c in moved):
                continue
            union = moved | anchor_cells
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            if len(union) == area:
                candidates.append(moved)
    return candidates


def _identify_groups(
    grid: np.ndarray,
    *,
    bg: int,
    axes: dict[str, dict[str, int | str | tuple[int, int]]],
) -> list[dict[str, object]]:
    colors = sorted(int(c) for c in np.unique(grid) if int(c) not in {int(bg), int(BLUE), int(RED), int(YELLOW)})
    groups: list[dict[str, object]] = []

    for color in colors:
        comps = find_components(grid, bg=bg, colors=[color], connectivity=4, monochromatic=True)
        assert len(comps) == 2, f"color {color} must have exactly two components"
        comp_info = []
        for comp in comps:
            hit_blue = _touches_axis(comp["coords"], axes["blue"])
            hit_red = _touches_axis(comp["coords"], axes["red"])
            role = None
            if hit_blue ^ hit_red:
                role = "source"
                axis_name = "blue" if hit_blue else "red"
            else:
                axis_name = None
            comp_info.append((comp, role, axis_name))

        sources = [item for item in comp_info if item[1] == "source"]
        anchors = [item for item in comp_info if item[1] is None]
        assert len(sources) == 1 and len(anchors) == 1, f"color {color} must have one source and one anchor"
        source, _role, axis_name = sources[0]
        anchor, _role2, _axis_none = anchors[0]
        groups.append(
            {
                "color": int(color),
                "source": source,
                "anchor": anchor,
                "axis_name": str(axis_name),
            }
        )

    return groups


def _solve_plan(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    axes = _axis_setup(g)
    plan: list[dict[str, object]] = []

    for group in _identify_groups(g, bg=bg, axes=axes):
        color = int(group["color"])
        source = group["source"]
        anchor = group["anchor"]
        axis_name = str(group["axis_name"])

        sprite = crop_bbox(g, source["bbox"])
        prepared = _rot_cw(sprite) if axis_name == "blue" else sprite.copy()

        blocked = (g != int(bg))
        for r, c in source["coords"]:
            blocked[int(r), int(c)] = False

        anchor_cells = _coords_set(anchor["coords"])
        candidates = _completion_candidates(prepared, anchor_cells, blocked, bg=bg)
        assert len(candidates) == 1, f"color {color} must have a unique completion placement"
        plan.append(
            {
                "color": color,
                "source_cells": _coords_set(source["coords"]),
                "moved_cells": candidates[0],
                "anchor_cells": anchor_cells,
                "axis_name": axis_name,
                "source_sprite": sprite,
            }
        )

    return plan


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each edge-adjacent fragment, rotating only the blue-axis fragments, so each same-color pair completes a solid rectangle.

    Input structure:
    - Black background with one yellow corner marker and two adjacent edge axis lines in blue and red.
    - Each non-axis foreground color appears as exactly two 4-connected objects of the same color.
    - One object of each pair is a fragment adjacent to one axis line; the other is an interior anchor away from the edges.

    Rule:
    - For every foreground color other than the axis colors, identify the axis-adjacent object as the fragment and the other as the anchor.
    - If the fragment touches the blue axis edge, rotate that fragment 90 degrees clockwise inside its own tight bounding box; if it touches the red axis edge, keep it unchanged.
    - Translate the prepared fragment to the unique position where it and the anchor form one filled axis-aligned rectangle, erase the fragment's old position, and draw it at the new position.

    Key insight:
    - The fragment's shape alone does not determine the preprocessing step; the color of the axis line it touches decides whether rotation happens before rectangle completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    plan = _solve_plan(g)
    for item in plan:
        color = int(item["color"])
        for r, c in item["source_cells"]:
            out[r, c] = BG
        for r, c in item["moved_cells"]:
            assert int(out[r, c]) == int(BG), "moved fragment must land on background"
            out[r, c] = color
    return out


def _rect_corner_offset(rect_h: int, rect_w: int, shape_h: int, shape_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, rect_w - shape_w
    if corner == "bl":
        return rect_h - shape_h, 0
    return rect_h - shape_h, rect_w - shape_w


def _anchor_blueprints(base_sprite: np.ndarray, branch: str) -> list[dict[str, object]]:
    correct = _rot_cw(base_sprite) if branch == "blue" else base_sprite.copy()
    wrong = base_sprite.copy() if branch == "blue" else _rot_cw(base_sprite)
    ch, cw = correct.shape
    blueprints: list[dict[str, object]] = []

    for rect_h in range(ch + 1, ch + 4):
        for rect_w in range(cw + 1, cw + 4):
            for corner in ("tl", "tr", "bl", "br"):
                patch = np.zeros((rect_h, rect_w), dtype=int)
                rr, cc = _rect_corner_offset(rect_h, rect_w, ch, cw, corner)
                patch[rr : rr + ch, cc : cc + cw] = correct
                anchor = (patch == 0)
                if not _mask_connected(anchor):
                    continue

                local = np.zeros((rect_h + 4, rect_w + 4), dtype=int)
                local_anchor = frozenset((2 + r, 2 + c) for r, c in _mask_coords(anchor))
                blocked = np.zeros_like(local, dtype=bool)
                for r, c in local_anchor:
                    blocked[r, c] = True

                good = _completion_candidates(correct, local_anchor, blocked, bg=0)
                bad = _completion_candidates(wrong, local_anchor, blocked, bg=0)
                if len(good) != 1 or bad:
                    continue

                blueprints.append(
                    {
                        "anchor_mask": anchor,
                        "target_mask": correct.astype(bool),
                        "rect_shape": (rect_h, rect_w),
                        "target_offset": (rr, cc),
                    }
                )
    return blueprints


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> frozenset[tuple[int, int]]:
    cells = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                rr, cc = top + r, left + c
                grid[rr, cc] = int(color)
                cells.append((rr, cc))
    return frozenset(cells)


def _bbox_clear(occupied: np.ndarray, top: int, left: int, h: int, w: int, *, pad: int) -> bool:
    grid_h, grid_w = occupied.shape
    if top < 0 or left < 0 or top + h > grid_h or left + w > grid_w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(grid_h, top + h + pad)
    c1 = min(grid_w, left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _source_position_candidates(
    occupied: np.ndarray,
    sprite: np.ndarray,
    axis: dict[str, int | str | tuple[int, int]],
    other_axis: dict[str, int | str | tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = sprite.shape
    grid_h, grid_w = occupied.shape
    side = str(axis["side"])
    temp_occ = occupied.copy()
    if side == "top":
        temp_occ[0, :] = False
        top = 1
        c_min, c_max = 0, grid_w - w
        if other_axis["side"] == "left":
            c_min = max(c_min, 2)
        if other_axis["side"] == "right":
            c_max = min(c_max, grid_w - w - 2)
        coords = [(top, c) for c in range(c_min, c_max + 1)]
    elif side == "bottom":
        temp_occ[-1, :] = False
        top = grid_h - h - 1
        c_min, c_max = 0, grid_w - w
        if other_axis["side"] == "left":
            c_min = max(c_min, 2)
        if other_axis["side"] == "right":
            c_max = min(c_max, grid_w - w - 2)
        coords = [(top, c) for c in range(c_min, c_max + 1)]
    elif side == "left":
        temp_occ[:, 0] = False
        left = 1
        r_min, r_max = 0, grid_h - h
        if other_axis["side"] == "top":
            r_min = max(r_min, 2)
        if other_axis["side"] == "bottom":
            r_max = min(r_max, grid_h - h - 2)
        coords = [(r, left) for r in range(r_min, r_max + 1)]
    else:
        temp_occ[:, -1] = False
        left = grid_w - w - 1
        r_min, r_max = 0, grid_h - h
        if other_axis["side"] == "top":
            r_min = max(r_min, 2)
        if other_axis["side"] == "bottom":
            r_max = min(r_max, grid_h - h - 2)
        coords = [(r, left) for r in range(r_min, r_max + 1)]

    random.shuffle(coords)
    return [(r, c) for r, c in coords if _bbox_clear(temp_occ, r, c, h, w, pad=1)]


def _anchor_position_candidates(occupied: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    grid_h, grid_w = occupied.shape
    coords = []
    for top in range(2, grid_h - h - 1):
        for left in range(2, grid_w - w - 1):
            if _bbox_clear(occupied, top, left, h, w, pad=1):
                coords.append((top, left))
    random.shuffle(coords)
    return coords


def _expected_output(inp: np.ndarray, records: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for rec in records:
        color = int(rec["color"])
        for r, c in rec["source_cells"]:
            out[r, c] = BG
        for r, c in rec["moved_cells"]:
            out[r, c] = color
    return out


def _validate_semantics(inp: np.ndarray, records: list[dict[str, object]]) -> bool:
    try:
        solved = _solve_plan(inp)
        solved_by_color = {int(item["color"]): item for item in solved}
        if set(solved_by_color) != {int(rec["color"]) for rec in records}:
            return False
        for rec in records:
            color = int(rec["color"])
            got = solved_by_color[color]
            if got["moved_cells"] != rec["moved_cells"]:
                return False

        out = main(inp)
        if not np.array_equal(out, _expected_output(inp, records)):
            return False

        axes = _axis_setup(inp)
        by_color = {int(rec["color"]): rec for rec in records}
        groups = _identify_groups(inp, bg=BG, axes=axes)
        branch_seen = set()
        for group in groups:
            color = int(group["color"])
            rec = by_color[color]
            source = group["source"]
            anchor = group["anchor"]
            source_sprite = crop_bbox(inp, source["bbox"])
            blocked = (inp != BG)
            for r, c in source["coords"]:
                blocked[int(r), int(c)] = False
            anchor_cells = _coords_set(anchor["coords"])
            no_rot = _completion_candidates(source_sprite, anchor_cells, blocked, bg=BG)
            cw = _completion_candidates(_rot_cw(source_sprite), anchor_cells, blocked, bg=BG)
            branch = str(rec["branch"])
            branch_seen.add(branch)
            if branch == "blue":
                if len(cw) != 1 or no_rot:
                    return False
            else:
                if len(no_rot) != 1 or cw:
                    return False
        return branch_seen == {"blue", "red"}
    except AssertionError:
        return False


def _make_group_record(
    grid: np.ndarray,
    color: int,
    branch: str,
    base_sprite: np.ndarray,
    axes: dict[str, dict[str, int | str | tuple[int, int]]],
) -> dict[str, object] | None:
    blueprints = _anchor_blueprints(base_sprite, branch)
    if not blueprints:
        return None
    random.shuffle(blueprints)

    occupied = grid != BG
    source_axis = axes[branch]
    other_axis = axes["red" if branch == "blue" else "blue"]

    for blueprint in blueprints:
        anchor_mask = np.asarray(blueprint["anchor_mask"], dtype=bool)
        anchor_positions = _anchor_position_candidates(occupied, anchor_mask)
        for anchor_top, anchor_left in anchor_positions[:30]:
            occupied_after_anchor = occupied.copy()
            for r, c in _mask_coords(anchor_mask):
                occupied_after_anchor[anchor_top + r, anchor_left + c] = True
            source_positions = _source_position_candidates(occupied_after_anchor, base_sprite, source_axis, other_axis)
            if not source_positions:
                continue
            source_top, source_left = source_positions[0]

            target_mask = np.asarray(blueprint["target_mask"], dtype=bool)
            target_off_r, target_off_c = map(int, blueprint["target_offset"])
            moved_cells = frozenset(
                (anchor_top + target_off_r + r, anchor_left + target_off_c + c)
                for r, c in _mask_coords(target_mask)
            )
            source_cells = frozenset(
                (source_top + r, source_left + c)
                for r, c in _mask_coords(base_sprite.astype(bool))
            )
            if moved_cells & source_cells:
                continue

            trial = grid.copy()
            anchor_cells = _paint_mask(trial, anchor_mask, anchor_top, anchor_left, color)
            source_cells = _paint_mask(trial, base_sprite.astype(bool), source_top, source_left, color)
            record = {
                "color": int(color),
                "branch": branch,
                "source_cells": source_cells,
                "anchor_cells": anchor_cells,
                "moved_cells": moved_cells,
                "source_sprite": base_sprite.copy(),
            }
            return {"grid": trial, "record": record}
    return None


def _paint_axes(grid: np.ndarray, corner_name: str, blue_on: str) -> dict[str, dict[str, int | str | tuple[int, int]]]:
    h, w = grid.shape
    cr, cc = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }[corner_name]
    row_color = BLUE if blue_on == "h" else RED
    col_color = BLUE if blue_on == "v" else RED
    grid[cr, :] = int(row_color)
    grid[:, cc] = int(col_color)
    grid[cr, cc] = int(YELLOW)
    return _axis_setup(grid)


def _branch_schedule(n_pairs: int) -> list[str]:
    branches = ["blue", "red"]
    while len(branches) < n_pairs:
        branches.append(random.choice(["blue", "red"]))
    random.shuffle(branches)
    return branches


def _sprite_schedule(branches: list[str]) -> list[np.ndarray]:
    sprites: list[np.ndarray | None] = [None] * len(branches)
    if random.random() < 0.7:
        shared = random.choice(SHAPE_CATALOG).copy()
        blue_idx = next(i for i, b in enumerate(branches) if b == "blue")
        red_idx = next(i for i, b in enumerate(branches) if b == "red")
        sprites[blue_idx] = shared.copy()
        sprites[red_idx] = shared.copy()
    for i in range(len(branches)):
        if sprites[i] is None:
            sprites[i] = random.choice(SHAPE_CATALOG).copy()
    return [np.asarray(s, dtype=int) for s in sprites]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.zeros((h, w), dtype=int)
    axes = _paint_axes(grid, random.choice(["tl", "tr", "bl", "br"]), random.choice(["h", "v"]))

    n_pairs = random.choice([2, 3])
    colors = random.sample(list(PAIR_COLORS), n_pairs)
    branches = _branch_schedule(n_pairs)
    sprites = _sprite_schedule(branches)
    records: list[dict[str, object]] = []

    order = list(range(n_pairs))
    random.shuffle(order)
    for idx in order:
        built = _make_group_record(grid, colors[idx], branches[idx], sprites[idx], axes)
        if built is None:
            return None
        grid = built["grid"]
        records.append(built["record"])

    if not _validate_semantics(grid, records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 10")

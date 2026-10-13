from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import (  # type: ignore
    Color,
    bbox,
    bbox_coords,
    components_mask,
    find_components,
    get_bg,
    neighbors4,
    paste,
    random_free_location,
)


GEN_CANVAS_SHAPE = (30, 30)
GEN_PAIR_COUNT_CHOICES = (2, 3)
GEN_COLOR_POOL = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_CAVITY_HEIGHT_RANGE = (3, 6)
GEN_CAVITY_WIDTH_RANGE = (3, 6)
GEN_CAVITY_AREA_RANGE = (6, 16)
GEN_THICKNESS_RANGE = (2, 4)
GEN_PAIR_PADDING = 2
GEN_PIECE_PADDING = 1
GEN_CROP_MARGIN_MAX = 3
GEN_ATTEMPTS = 500
GEN_SHAPE_TRIES = 160


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask.astype(bool))
    if box is None:
        raise ValueError("cannot normalize an empty mask")
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    shifted = coords - coords.min(axis=0)
    h, w = shifted.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = _normalize_mask(np.rot90(mask, k))
        if not any(np.array_equal(rotated, old) for old in variants):
            variants.append(rotated)
    return variants


def _canonical_signature(mask: np.ndarray) -> tuple:
    sigs = []
    for rotated in _unique_rotations(mask):
        sigs.append((tuple(map(int, rotated.shape)), tuple(int(x) for x in rotated.astype(int).ravel())))
    return min(sigs)


def _shape_matches(piece_mask: np.ndarray, cavity_mask: np.ndarray) -> bool:
    return any(np.array_equal(rotated, cavity_mask) for rotated in _unique_rotations(piece_mask))


def _is_consecutive(values: list[int]) -> bool:
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _row_intervals(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, list[int]] = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr not in out:
            out[rr] = [cc, cc]
        else:
            out[rr][0] = min(out[rr][0], cc)
            out[rr][1] = max(out[rr][1], cc)
    return {r: (ab[0], ab[1]) for r, ab in out.items()}


def _col_intervals(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, list[int]] = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if cc not in out:
            out[cc] = [rr, rr]
        else:
            out[cc][0] = min(out[cc][0], rr)
            out[cc][1] = max(out[cc][1], rr)
    return {c: (ab[0], ab[1]) for c, ab in out.items()}


def _extract_horizontal_cavity(grid: np.ndarray, comp_a: dict, comp_b: dict, bg: int) -> np.ndarray | None:
    left, right = sorted((comp_a, comp_b), key=lambda item: item["bbox"][1])
    left_rows = _row_intervals(left["coords"])
    right_rows = _row_intervals(right["coords"])
    overlap_rows = sorted(set(left_rows) & set(right_rows))
    if len(overlap_rows) < 2 or not _is_consecutive(overlap_rows):
        return None

    cells: list[tuple[int, int]] = []
    for r in overlap_rows:
        left_edge = left_rows[r][1]
        right_edge = right_rows[r][0]
        if right_edge <= left_edge + 1:
            return None
        if np.any(grid[r, left_edge + 1 : right_edge] != bg):
            return None
        cells.extend((r, c) for c in range(left_edge + 1, right_edge))

    coords = np.array(cells, dtype=int)
    mask = _coords_to_mask(coords)
    if len(components_mask(mask, connectivity=4)) != 1:
        return None
    return coords


def _extract_vertical_cavity(grid: np.ndarray, comp_a: dict, comp_b: dict, bg: int) -> np.ndarray | None:
    top, bottom = sorted((comp_a, comp_b), key=lambda item: item["bbox"][0])
    top_cols = _col_intervals(top["coords"])
    bottom_cols = _col_intervals(bottom["coords"])
    overlap_cols = sorted(set(top_cols) & set(bottom_cols))
    if len(overlap_cols) < 2 or not _is_consecutive(overlap_cols):
        return None

    cells: list[tuple[int, int]] = []
    for c in overlap_cols:
        top_edge = top_cols[c][1]
        bottom_edge = bottom_cols[c][0]
        if bottom_edge <= top_edge + 1:
            return None
        if np.any(grid[top_edge + 1 : bottom_edge, c] != bg):
            return None
        cells.extend((r, c) for r in range(top_edge + 1, bottom_edge))

    coords = np.array(cells, dtype=int)
    mask = _coords_to_mask(coords)
    if len(components_mask(mask, connectivity=4)) != 1:
        return None
    return coords


def _extract_pair_cavity(grid: np.ndarray, comp_a: dict, comp_b: dict, bg: int) -> dict | None:
    horizontal = _extract_horizontal_cavity(grid, comp_a, comp_b, bg)
    vertical = _extract_vertical_cavity(grid, comp_a, comp_b, bg)
    found = [(horizontal, "H"), (vertical, "V")]
    valid = [(coords, orient) for coords, orient in found if coords is not None]
    if len(valid) != 1:
        return None
    coords, orientation = valid[0]
    return {
        "color": int(comp_a["color"]),
        "coords": coords,
        "orientation": orientation,
    }


def _extract_cavities(grid: np.ndarray, bg: int) -> list[dict]:
    objects = find_components(grid, bg=bg, connectivity=4, return_format="dict")
    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        color = int(obj["color"])
        if color in (bg, Color.GREY):
            continue
        by_color[color].append(obj)

    cavities: list[dict] = []
    for color, comps in sorted(by_color.items()):
        if len(comps) != 2:
            continue
        cavity = _extract_pair_cavity(grid, comps[0], comps[1], bg)
        if cavity is not None:
            cavities.append(cavity)
    return cavities


def _find_unique_assignment(match_sets: list[set[int]]) -> dict[int, int] | None:
    if not match_sets or any(len(matches) == 0 for matches in match_sets):
        return None

    order = sorted(range(len(match_sets)), key=lambda idx: (len(match_sets[idx]), idx))
    solutions: list[dict[int, int]] = []

    def dfs(pos: int, used: set[int], current: dict[int, int]) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(dict(current))
            return
        cavity_idx = order[pos]
        for piece_idx in sorted(match_sets[cavity_idx] - used):
            current[cavity_idx] = piece_idx
            dfs(pos + 1, used | {piece_idx}, current)
            current.pop(cavity_idx, None)

    dfs(0, set(), {})
    if len(solutions) != 1:
        return None
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray piece into the unique same-color jagged cavity that it exactly fills after rotation.

    Input structure
    - A black background contains several pairs of same-color 4-connected blocks and some scattered gray 4-connected pieces.
    - In each colored pair, the two blocks face each other horizontally or vertically, and the black cells between their opposing jagged edges form one implicit cavity.

    Rule
    - Extract every cavity induced by a facing same-color pair.
    - For each gray piece, try the four quarter-turn rotations; a piece matches a cavity iff some rotation is exactly the same shape as the cavity.
    - When all cavity-to-piece matches are uniquely determined, erase each matched gray piece from its original location and repaint the corresponding cavity with that pair's color. Leave unmatched gray pieces unchanged.

    Key insight
    - The target region is never drawn directly: it must be reconstructed from the full black gap between two facing jagged edges, and matching depends only on exact shape equality up to rotation, not on the piece's current position.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    cavities = _extract_cavities(g, bg)
    gray_objects = find_components(
        g,
        bg=bg,
        colors=[Color.GREY],
        connectivity=4,
        return_format="dict",
    )

    cavity_masks = [_coords_to_mask(cavity["coords"]) for cavity in cavities]
    piece_masks = [_coords_to_mask(obj["coords"]) for obj in gray_objects]
    match_sets = [
        {idx for idx, piece_mask in enumerate(piece_masks) if _shape_matches(piece_mask, cavity_mask)}
        for cavity_mask in cavity_masks
    ]
    assignment = _find_unique_assignment(match_sets)
    if assignment is None:
        return g.copy()

    out = g.copy()
    for cavity_idx, piece_idx in assignment.items():
        piece_coords = gray_objects[piece_idx]["coords"]
        out[piece_coords[:, 0], piece_coords[:, 1]] = bg
        cavity = cavities[cavity_idx]
        cavity_coords = cavity["coords"]
        out[cavity_coords[:, 0], cavity_coords[:, 1]] = int(cavity["color"])
    return out


def _randint(bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return random.randint(int(lo), int(hi))


def _make_row_convex_mask(height: int, width: int) -> np.ndarray | None:
    if width < 3 or height < 3:
        return None

    for _ in range(GEN_SHAPE_TRIES):
        starts = [random.randint(0, width - 2)]
        ends = [random.randint(starts[0] + 1, width - 1)]
        ok = True
        for _row in range(1, height):
            prev_start = starts[-1]
            prev_end = ends[-1]
            options: list[tuple[int, int]] = []
            for start in range(max(0, prev_start - 1), min(width - 2, prev_start + 1) + 1):
                min_end = max(start + 1, prev_end - 1)
                max_end = min(width - 1, prev_end + 1)
                for end in range(min_end, max_end + 1):
                    if max(start, prev_start) <= min(end, prev_end):
                        options.append((start, end))
            if not options:
                ok = False
                break
            start, end = random.choice(options)
            starts.append(start)
            ends.append(end)
        if not ok:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, (start, end) in enumerate(zip(starts, ends)):
            mask[r, start : end + 1] = True
        mask = _normalize_mask(mask)
        if int(mask.sum()) < GEN_CAVITY_AREA_RANGE[0] or int(mask.sum()) > GEN_CAVITY_AREA_RANGE[1]:
            continue
        if int(mask.sum()) == int(mask.size):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        row_spans = _row_intervals(np.argwhere(mask))
        if not all((row_spans[r][1] - row_spans[r][0] + 1) >= 2 for r in row_spans):
            continue
        if len(set(start for start, _ in row_spans.values())) == 1 and len(set(end for _, end in row_spans.values())) == 1:
            continue
        return mask
    return None


def _generate_cavity_mask(orientation: str) -> np.ndarray | None:
    if orientation == "H":
        height = _randint(GEN_CAVITY_HEIGHT_RANGE)
        width = _randint(GEN_CAVITY_WIDTH_RANGE)
        return _make_row_convex_mask(height, width)

    width = _randint(GEN_CAVITY_WIDTH_RANGE)
    height = _randint(GEN_CAVITY_HEIGHT_RANGE)
    base = _make_row_convex_mask(width, height)
    if base is None:
        return None
    return base.T


def _build_pair_patch(mask: np.ndarray, color: int, orientation: str) -> tuple[np.ndarray, frozenset[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    if orientation == "H":
        left_thickness = _randint(GEN_THICKNESS_RANGE)
        right_thickness = _randint(GEN_THICKNESS_RANGE)
        patch = np.zeros((h, left_thickness + w + right_thickness), dtype=int)
        for r in range(h):
            cols = np.flatnonzero(mask[r])
            left_edge = int(cols[0])
            right_edge = int(cols[-1])
            patch[r, : left_thickness + left_edge] = int(color)
            patch[r, left_thickness + right_edge + 1 :] = int(color)
        cavity = frozenset((r, left_thickness + c) for r, c in np.argwhere(mask))
        return patch, cavity

    top_thickness = _randint(GEN_THICKNESS_RANGE)
    bottom_thickness = _randint(GEN_THICKNESS_RANGE)
    patch = np.zeros((top_thickness + h + bottom_thickness, w), dtype=int)
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        top_edge = int(rows[0])
        bottom_edge = int(rows[-1])
        patch[: top_thickness + top_edge, c] = int(color)
        patch[top_thickness + bottom_edge + 1 :, c] = int(color)
    cavity = frozenset((top_thickness + r, c) for r, c in np.argwhere(mask))
    return patch, cavity


def _rect_block(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=int)


def _place_bbox(reserved: np.ndarray, shape: tuple[int, int], padding: int) -> tuple[int, int] | None:
    blocker = _rect_block(shape)
    try:
        return random_free_location(
            reserved,
            blocker,
            transparent=0,
            border=1,
            padding=padding,
            padding_connectivity=8,
        )
    except ValueError:
        return None


def _reserve_bbox(reserved: np.ndarray, top_left: tuple[int, int], shape: tuple[int, int]) -> None:
    patch = _rect_block(shape)
    paste(reserved, patch, int(top_left[0]), int(top_left[1]))


def _absolute_coords(local_coords: frozenset[tuple[int, int]], top_left: tuple[int, int]) -> frozenset[tuple[int, int]]:
    dr, dc = map(int, top_left)
    return frozenset((dr + r, dc + c) for r, c in local_coords)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    return _coords_to_mask(coords)


def _grow_polyomino(target_area: int) -> np.ndarray | None:
    for _ in range(GEN_SHAPE_TRIES):
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    grown = cells | {(int(nr), int(nc))}
                    rows = [rr for rr, _ in grown]
                    cols = [cc for _, cc in grown]
                    if max(rows) - min(rows) + 1 > 7 or max(cols) - min(cols) + 1 > 7:
                        continue
                    frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) != target_area:
            continue
        mask = _mask_from_cells(cells)
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        return mask
    return None


def _generate_unmatched_mask(
    cavity_signatures: set[tuple],
    cavity_areas: list[int],
    used_signatures: set[tuple],
    force_same_area: bool,
) -> np.ndarray | None:
    area_choices = cavity_areas if force_same_area else cavity_areas + list(range(GEN_CAVITY_AREA_RANGE[0], GEN_CAVITY_AREA_RANGE[1] + 1))
    for _ in range(GEN_SHAPE_TRIES):
        target_area = random.choice(area_choices)
        mask = _grow_polyomino(target_area)
        if mask is None:
            continue
        sig = _canonical_signature(mask)
        if sig in cavity_signatures or sig in used_signatures:
            continue
        return mask
    return None


def _crop_with_margin(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    box = bbox(grid != Color.BLACK)
    if box is None:
        raise ValueError("cannot crop an empty instance")
    r0, c0, r1, c1 = box
    top_margin = random.randint(1, min(GEN_CROP_MARGIN_MAX, r0))
    left_margin = random.randint(1, min(GEN_CROP_MARGIN_MAX, c0))
    bottom_margin = random.randint(1, min(GEN_CROP_MARGIN_MAX, grid.shape[0] - 1 - r1))
    right_margin = random.randint(1, min(GEN_CROP_MARGIN_MAX, grid.shape[1] - 1 - c1))
    rr0 = r0 - top_margin
    cc0 = c0 - left_margin
    rr1 = r1 + bottom_margin
    cc1 = c1 + right_margin
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), (int(rr0), int(cc0))


def _shift_coords(coords: frozenset[tuple[int, int]], origin: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r0, c0 = map(int, origin)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _build_oracle_output(
    inp: np.ndarray,
    pair_plans: list[dict],
    piece_plans: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for piece in piece_plans:
        if piece["matched_color"] is not None:
            for r, c in piece["coords"]:
                out[r, c] = Color.BLACK
    for pair in pair_plans:
        for r, c in pair["cavity_coords"]:
            out[r, c] = int(pair["color"])
    return out


def _validate_instance(inp: np.ndarray, pair_plans: list[dict], piece_plans: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if get_bg(inp) != Color.BLACK:
        return False
    if not pair_plans or not piece_plans:
        return False
    if not any(plan["orientation"] == "H" for plan in pair_plans):
        return False
    if not any(plan["orientation"] == "V" for plan in pair_plans):
        return False
    if not any(piece["matched_color"] is None for piece in piece_plans):
        return False
    cavity_areas = {plan["area"] for plan in pair_plans}
    if not any(piece["matched_color"] is None and piece["area"] in cavity_areas for piece in piece_plans):
        return False
    if not any(plan["rotational_variants"] > 1 for plan in pair_plans):
        return False
    if not any(piece["matched_color"] is not None and piece["rotated"] for piece in piece_plans):
        return False
    if len({plan["signature"] for plan in pair_plans}) != len(pair_plans):
        return False

    actual_cavities = _extract_cavities(inp, Color.BLACK)
    if len(actual_cavities) != len(pair_plans):
        return False

    plan_by_color = {int(plan["color"]): plan for plan in pair_plans}
    actual_by_color = {int(cavity["color"]): cavity for cavity in actual_cavities}
    if set(plan_by_color) != set(actual_by_color):
        return False

    actual_cavity_masks: dict[int, np.ndarray] = {}
    for color, plan in plan_by_color.items():
        actual = actual_by_color[color]
        actual_set = frozenset((int(r), int(c)) for r, c in actual["coords"])
        if actual_set != plan["cavity_coords"]:
            return False
        if actual["orientation"] != plan["orientation"]:
            return False
        actual_cavity_masks[color] = _coords_to_mask(actual["coords"])

    gray_objects = find_components(
        inp,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        return_format="dict",
    )
    if len(gray_objects) != len(piece_plans):
        return False

    actual_piece_masks: dict[frozenset[tuple[int, int]], np.ndarray] = {}
    for obj in gray_objects:
        coords = frozenset((int(r), int(c)) for r, c in obj["coords"])
        actual_piece_masks[coords] = _coords_to_mask(obj["coords"])

    plan_piece_sets = {piece["coords"] for piece in piece_plans}
    if set(actual_piece_masks) != plan_piece_sets:
        return False

    for piece in piece_plans:
        piece_mask = actual_piece_masks[piece["coords"]]
        matching_colors = sorted(
            color for color, cavity_mask in actual_cavity_masks.items() if _shape_matches(piece_mask, cavity_mask)
        )
        if piece["matched_color"] is None:
            if matching_colors:
                return False
        else:
            if matching_colors != [int(piece["matched_color"])]:
                return False

    match_sets = []
    ordered_cavities = sorted(actual_cavities, key=lambda item: int(item["color"]))
    ordered_pieces = sorted(actual_piece_masks)
    for cavity in ordered_cavities:
        cavity_mask = _coords_to_mask(cavity["coords"])
        matches = {
            idx
            for idx, coords in enumerate(ordered_pieces)
            if _shape_matches(actual_piece_masks[coords], cavity_mask)
        }
        if len(matches) != 1:
            return False
        match_sets.append(matches)
    if _find_unique_assignment(match_sets) is None:
        return False

    expected = _build_oracle_output(inp, pair_plans, piece_plans)
    actual_out = main(inp)
    if not np.array_equal(actual_out, expected):
        return False
    if np.array_equal(actual_out, inp):
        return False
    if not np.any(actual_out == Color.GREY):
        return False
    return True


def _orientation_plan(pair_count: int) -> list[str]:
    plan = ["H", "V"]
    while len(plan) < pair_count:
        plan.append(random.choice(["H", "V"]))
    random.shuffle(plan)
    return plan


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros(GEN_CANVAS_SHAPE, dtype=int)
    reserved = np.zeros(GEN_CANVAS_SHAPE, dtype=int)

    pair_count = random.choice(GEN_PAIR_COUNT_CHOICES)
    colors = random.sample(GEN_COLOR_POOL, pair_count)
    orientations = _orientation_plan(pair_count)

    pair_plans: list[dict] = []
    cavity_signatures: set[tuple] = set()
    cavity_areas: list[int] = []

    for color, orientation in zip(colors, orientations):
        mask = _generate_cavity_mask(orientation)
        if mask is None:
            return None
        sig = _canonical_signature(mask)
        if sig in cavity_signatures:
            return None

        patch, cavity_local = _build_pair_patch(mask, color, orientation)
        top_left = _place_bbox(reserved, patch.shape, GEN_PAIR_PADDING)
        if top_left is None:
            return None
        paste(grid, patch, top_left[0], top_left[1], transparent=Color.BLACK)
        _reserve_bbox(reserved, top_left, patch.shape)

        pair_plans.append(
            {
                "color": int(color),
                "orientation": orientation,
                "signature": sig,
                "area": int(mask.sum()),
                "rotational_variants": len(_unique_rotations(mask)),
                "cavity_coords": _absolute_coords(cavity_local, top_left),
            }
        )
        cavity_signatures.add(sig)
        cavity_areas.append(int(mask.sum()))

    rotatable_pairs = [plan for plan in pair_plans if plan["rotational_variants"] > 1]
    if not rotatable_pairs:
        return None
    forced_rotation_color = int(random.choice(rotatable_pairs)["color"])

    piece_plans: list[dict] = []
    used_piece_signatures: set[tuple] = set()

    for pair in pair_plans:
        cavity_mask = _coords_to_mask(np.array(sorted(pair["cavity_coords"]), dtype=int))
        rotations = _unique_rotations(cavity_mask)
        if int(pair["color"]) == forced_rotation_color:
            choices = [rot for rot in rotations if not np.array_equal(rot, cavity_mask)]
            if not choices:
                return None
            piece_mask = random.choice(choices)
            rotated = True
        else:
            piece_mask = random.choice(rotations)
            rotated = not np.array_equal(piece_mask, cavity_mask)

        patch = _mask_to_patch(piece_mask, Color.GREY)
        top_left = _place_bbox(reserved, patch.shape, GEN_PIECE_PADDING)
        if top_left is None:
            return None
        paste(grid, patch, top_left[0], top_left[1], transparent=Color.BLACK)
        _reserve_bbox(reserved, top_left, patch.shape)

        local_coords = frozenset((int(r), int(c)) for r, c in np.argwhere(piece_mask))
        piece_plans.append(
            {
                "coords": _absolute_coords(local_coords, top_left),
                "matched_color": int(pair["color"]),
                "area": int(piece_mask.sum()),
                "rotated": rotated,
            }
        )
        used_piece_signatures.add(_canonical_signature(piece_mask))

    unmatched_count = random.choice((1, 2))
    for idx in range(unmatched_count):
        piece_mask = _generate_unmatched_mask(
            cavity_signatures=cavity_signatures,
            cavity_areas=cavity_areas,
            used_signatures=used_piece_signatures,
            force_same_area=(idx == 0),
        )
        if piece_mask is None:
            return None
        patch = _mask_to_patch(piece_mask, Color.GREY)
        top_left = _place_bbox(reserved, patch.shape, GEN_PIECE_PADDING)
        if top_left is None:
            return None
        paste(grid, patch, top_left[0], top_left[1], transparent=Color.BLACK)
        _reserve_bbox(reserved, top_left, patch.shape)

        local_coords = frozenset((int(r), int(c)) for r, c in np.argwhere(piece_mask))
        piece_plans.append(
            {
                "coords": _absolute_coords(local_coords, top_left),
                "matched_color": None,
                "area": int(piece_mask.sum()),
                "rotated": False,
            }
        )
        used_piece_signatures.add(_canonical_signature(piece_mask))

    cropped, origin = _crop_with_margin(grid)
    shifted_pairs = []
    for pair in pair_plans:
        shifted = dict(pair)
        shifted["cavity_coords"] = _shift_coords(pair["cavity_coords"], origin)
        shifted_pairs.append(shifted)
    shifted_pieces = []
    for piece in piece_plans:
        shifted = dict(piece)
        shifted["coords"] = _shift_coords(piece["coords"], origin)
        shifted_pieces.append(shifted)

    if not _validate_instance(cropped, shifted_pairs, shifted_pieces):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jagged-cavity ARC instance")

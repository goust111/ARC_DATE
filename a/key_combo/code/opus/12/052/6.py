from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components


BACKGROUND = 0
MASK = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
GEN_ATTEMPTS = 500

_MODE_BAG: list[str] = []
_PAIR_BAG: list[int] = []


def _mirror_col(width: int, c: int) -> int:
    return int(width - 1 - c)


def _rows_from_coords(coords: np.ndarray | set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    if isinstance(coords, set):
        if not coords:
            return []
        arr = np.array(sorted(coords), dtype=int)
    else:
        arr = np.asarray(coords, dtype=int)
    rows: list[tuple[int, int, int]] = []
    for r in sorted({int(v) for v in arr[:, 0]}):
        row_cols = arr[arr[:, 0] == r, 1]
        assert row_cols.size == 2, "each occupied row must have exactly two boundary pixels"
        c0 = int(row_cols.min())
        c1 = int(row_cols.max())
        assert c0 < c1, "boundary endpoints must be distinct"
        rows.append((r, c0, c1))
    return rows


def _paint_spans(
    out: np.ndarray,
    spans: list[tuple[int, int, int]],
    color: int,
    *,
    only_on_background: bool,
) -> None:
    color = int(color)
    for r, c0, c1 in spans:
        if only_on_background:
            row = out[r, c0 : c1 + 1]
            mask = row == BACKGROUND
            row[mask] = color
        else:
            out[r, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill visible row-paired outlines, then mirror them across the global vertical axis to recolor and fill masked partner outlines.

    Input structure:
    - The grid has black background and an odd width, so the vertical center column is explicit.
    - Foreground consists of 4-6 sparse outline objects whose occupied rows each contain exactly two boundary pixels.
    - Every outline belongs to a left-right mirror pair across the global vertical axis; visible outlines use distinct non-pink colors, while their masked partners use pink(6).

    Rule:
    - Fill every visible outline row by row between its two boundary pixels.
    - Use the grid's vertical center column as the symmetry axis.
    - For each pink outline, mirror its pink boundary pixels across the axis to identify the unique visible partner.
    - Mirror that partner's full boundary back onto the pink side, recolor it with the partner color, and fill each mirrored row span.
    - If a reconstructed masked fill overlaps an already filled visible region, keep the visible color.

    Key insight:
    - Pink does not encode a usable object color; it marks an entire outline whose row-span fill color must be recovered from its mirror-symmetric partner before the same row-wise filling rule can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w % 2 == 1, "the symmetry axis requires an odd grid width"

    comps = find_components(g, bg=BACKGROUND, connectivity=8, monochromatic=True, return_format="dict")
    assert comps, "expected outline objects"

    visible: list[dict] = []
    masked: list[dict] = []
    for comp in comps:
        rows = _rows_from_coords(comp["coords"])
        comp = dict(comp)
        comp["rows"] = rows
        if int(comp["color"]) == MASK:
            masked.append(comp)
        else:
            visible.append(comp)

    assert visible and masked, "expected both visible and masked outlines"
    assert len(visible) == len(masked), "visible and masked outline counts must match"
    visible_colors = [int(comp["color"]) for comp in visible]
    assert len(set(visible_colors)) == len(visible_colors), "visible outline colors must be unique"

    coord_to_visible: dict[tuple[int, int], int] = {}
    visible_row_spans_by_idx: list[list[tuple[int, int, int]]] = []
    for idx, comp in enumerate(visible):
        coords = {tuple(map(int, p)) for p in comp["coords"]}
        for cell in coords:
            assert cell not in coord_to_visible, "visible outlines may not overlap"
            coord_to_visible[cell] = idx
        visible_row_spans_by_idx.append(list(comp["rows"]))

    out = np.zeros_like(g)
    for comp in visible:
        _paint_spans(out, list(comp["rows"]), int(comp["color"]), only_on_background=False)

    used_partner_indices: set[int] = set()
    for comp in masked:
        mirrored_partner_cells = {
            (int(r), _mirror_col(w, int(c)))
            for r, c in map(tuple, comp["coords"])
        }
        partner_indices = {coord_to_visible[cell] for cell in mirrored_partner_cells if cell in coord_to_visible}
        assert len(partner_indices) == 1, "each masked outline must mirror to exactly one visible partner"
        partner_idx = next(iter(partner_indices))
        assert partner_idx not in used_partner_indices, "masked-to-visible pairing must be one-to-one"
        used_partner_indices.add(partner_idx)

        partner = visible[partner_idx]
        reconstructed_boundary = {
            (int(r), _mirror_col(w, int(c)))
            for r, c in map(tuple, partner["coords"])
        }
        assert reconstructed_boundary == {tuple(map(int, p)) for p in comp["coords"]}, "masked boundary must match the mirrored partner boundary"
        mirrored_rows = [
            (int(r), _mirror_col(w, int(c1)), _mirror_col(w, int(c0)))
            for r, c0, c1 in partner["rows"]
        ]
        mirrored_rows = [(r, min(c0, c1), max(c0, c1)) for r, c0, c1 in mirrored_rows]
        _paint_spans(out, mirrored_rows, int(partner["color"]), only_on_background=True)

    assert len(used_partner_indices) == len(visible), "every visible outline must serve as one masked partner"
    return out


def _boundary_neighbor_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _shape_from_row_bounds(rows: list[tuple[int, int]]) -> dict | None:
    height = len(rows)
    boundary_local: list[tuple[int, int]] = []
    fill_local: set[tuple[int, int]] = set()
    max_c = -1
    prev_left = None
    prev_right = None
    asymmetric_steps = 0
    for r, (left, right) in enumerate(rows):
        if right - left < 1:
            return None
        if prev_left is not None:
            if abs(left - prev_left) > 1 or abs(right - prev_right) > 1:
                return None
            if (left - prev_left) != (prev_right - right):
                asymmetric_steps += 1
        prev_left = left
        prev_right = right
        boundary_local.append((r, left))
        boundary_local.append((r, right))
        for c in range(left, right + 1):
            fill_local.add((r, c))
        max_c = max(max_c, right)
    if height < 3 or max_c < 1:
        return None
    return {
        "height": int(height),
        "width": int(max_c + 1),
        "boundary_local": tuple(boundary_local),
        "fill_local": frozenset(fill_local),
        "asymmetric": bool(asymmetric_steps > 0),
    }


def _gen_shape(max_width: int) -> dict | None:
    for _ in range(80):
        height = random.randint(3, 6)
        final_width = random.randint(2, min(4, max_width))
        min_top = max(final_width + 1, final_width + max(1, height - 2))
        max_top = min(max_width, final_width + 2 * (height - 1))
        if min_top > max_top:
            continue
        top_width = random.randint(min_top, max_top)
        delta = top_width - final_width
        steps = [0] * (height - 1)
        available = [i for i in range(height - 1) for _ in range(2)]
        random.shuffle(available)
        for idx in available[:delta]:
            steps[idx] += 1
        if sum(step > 0 for step in steps) < max(1, height - 2):
            continue

        left = 0
        right = top_width - 1
        rows = [(left, right)]
        for step in steps:
            if step == 2:
                dl, dr = 1, 1
            elif step == 1:
                dl, dr = random.choice(((1, 0), (0, 1)))
            else:
                dl, dr = 0, 0
            left += dl
            right -= dr
            rows.append((left, right))
        shape = _shape_from_row_bounds(rows)
        if shape is None:
            continue
        if random.random() < 0.5:
            width = int(shape["width"])
            flipped_rows = [(int(width - 1 - right), int(width - 1 - left)) for left, right in rows]
            shape = _shape_from_row_bounds(flipped_rows)
            if shape is None:
                continue
        if any((right - left + 1) >= 5 for left, right in rows):
            return shape
    return None


def _place_shape(shape: dict, origin: tuple[int, int], width: int) -> dict:
    r0, c0 = map(int, origin)
    left_boundary = {(r0 + r, c0 + c) for r, c in shape["boundary_local"]}
    left_fill = {(r0 + r, c0 + c) for r, c in shape["fill_local"]}
    right_boundary = {(r, _mirror_col(width, c)) for r, c in left_boundary}
    right_fill = {(r, _mirror_col(width, c)) for r, c in left_fill}
    return {
        "origin": (r0, c0),
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "left_fill": left_fill,
        "right_fill": right_fill,
    }


def _choose_mode() -> str:
    global _MODE_BAG
    if not _MODE_BAG:
        _MODE_BAG = random.sample(["disjoint", "disjoint", "conflict"], 3)
    return _MODE_BAG.pop()


def _choose_pair_count(mode: str) -> int:
    global _PAIR_BAG
    if mode == "conflict":
        return 2 if random.random() < 0.65 else 3
    if not _PAIR_BAG:
        _PAIR_BAG = random.sample([2, 2, 3, 3], 4)
    return _PAIR_BAG.pop()


def _orientation_list(pair_count: int) -> list[bool]:
    mask_left = [True, False]
    while len(mask_left) < pair_count:
        mask_left.append(bool(random.getrandbits(1)))
    random.shuffle(mask_left)
    return mask_left


def _render_pair_input(grid: np.ndarray, rec: dict) -> None:
    left_color = MASK if rec["mask_left"] else int(rec["color"])
    right_color = int(rec["color"]) if rec["mask_left"] else MASK
    for r, c in rec["left_boundary"]:
        grid[r, c] = left_color
    for r, c in rec["right_boundary"]:
        grid[r, c] = right_color


def _render_expected(shape: tuple[int, int], placements: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    visible_spans: list[tuple[list[tuple[int, int, int]], int]] = []
    masked_spans: list[tuple[list[tuple[int, int, int]], int]] = []
    for rec in placements:
        left_rows = _rows_from_coords(rec["left_boundary"])
        right_rows = _rows_from_coords(rec["right_boundary"])
        if rec["mask_left"]:
            masked_spans.append((left_rows, int(rec["color"])))
            visible_spans.append((right_rows, int(rec["color"])))
        else:
            visible_spans.append((left_rows, int(rec["color"])))
            masked_spans.append((right_rows, int(rec["color"])))
    for rows, color in visible_spans:
        _paint_spans(out, rows, color, only_on_background=False)
    for rows, color in masked_spans:
        _paint_spans(out, rows, color, only_on_background=True)
    return out


def _analyze_fill_overlaps(placements: list[dict]) -> tuple[bool, int, int]:
    cell_entries: dict[tuple[int, int], list[str]] = defaultdict(list)
    for rec in placements:
        left_status = "masked" if rec["mask_left"] else "visible"
        right_status = "visible" if rec["mask_left"] else "masked"
        for cell in rec["left_fill"]:
            cell_entries[cell].append(left_status)
        for cell in rec["right_fill"]:
            cell_entries[cell].append(right_status)

    overlap_count = 0
    for entries in cell_entries.values():
        if len(entries) == 1:
            continue
        if len(entries) != 2 or sorted(entries) != ["masked", "visible"]:
            return False, 0, len(cell_entries)
        overlap_count += 1
    return True, overlap_count, len(cell_entries)


def _components_ok(inp: np.ndarray, pair_count: int) -> bool:
    comps = find_components(inp, bg=BACKGROUND, connectivity=8, monochromatic=True, return_format="dict")
    if len(comps) != pair_count * 2:
        return False
    masked = [comp for comp in comps if int(comp["color"]) == MASK]
    visible = [comp for comp in comps if int(comp["color"]) != MASK]
    if len(masked) != pair_count or len(visible) != pair_count:
        return False
    if len({int(comp["color"]) for comp in visible}) != pair_count:
        return False
    for comp in comps:
        try:
            _rows_from_coords(comp["coords"])
        except AssertionError:
            return False
    return True


def _pairing_ok(inp: np.ndarray) -> bool:
    h, w = inp.shape
    comps = find_components(inp, bg=BACKGROUND, connectivity=8, monochromatic=True, return_format="dict")
    masked = [comp for comp in comps if int(comp["color"]) == MASK]
    visible = [comp for comp in comps if int(comp["color"]) != MASK]
    coord_to_color: dict[tuple[int, int], int] = {}
    for comp in visible:
        for r, c in map(tuple, comp["coords"]):
            coord_to_color[(int(r), int(c))] = int(comp["color"])
    seen_colors: set[int] = set()
    for comp in masked:
        colors = {
            coord_to_color.get((int(r), _mirror_col(w, int(c))))
            for r, c in map(tuple, comp["coords"])
        }
        colors.discard(None)
        if len(colors) != 1:
            return False
        color = next(iter(colors))
        if color in seen_colors:
            return False
        seen_colors.add(color)
    return len(seen_colors) == len(visible)


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    mode = _choose_mode()
    pair_count = _choose_pair_count(mode)
    width = random.choice((11, 13, 15, 17))
    height = random.randint(11, 16)
    axis = width // 2

    shapes: list[dict] = []
    for _ in range(pair_count):
        shape = _gen_shape(axis)
        if shape is None:
            return None
        shapes.append(shape)
    colors = random.sample(PALETTE, pair_count)
    orientations = _orientation_list(pair_count)

    order = sorted(range(pair_count), key=lambda i: (len(shapes[i]["fill_local"]), shapes[i]["height"]), reverse=True)
    forbidden = np.zeros((height, width), dtype=int)
    placements: list[dict | None] = [None] * pair_count

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        shape = shapes[idx]
        origins = [
            (r0, c0)
            for r0 in range(height - int(shape["height"]) + 1)
            for c0 in range(axis - int(shape["width"]) + 1)
        ]
        random.shuffle(origins)
        for origin in origins:
            placed = _place_shape(shape, origin, width)
            boundary = placed["left_boundary"] | placed["right_boundary"]
            if any(forbidden[r, c] > 0 for r, c in boundary):
                continue
            placements[idx] = {
                "color": int(colors[idx]),
                "mask_left": bool(orientations[idx]),
                **placed,
            }
            touched = _boundary_neighbor_cells(boundary, (height, width))
            for r, c in touched:
                forbidden[r, c] += 1
            if search(pos + 1):
                return True
            for r, c in touched:
                forbidden[r, c] -= 1
            placements[idx] = None
        return False

    if not search(0):
        return None

    final_placements = [rec for rec in placements if rec is not None]
    ok, overlap_count, colored_total = _analyze_fill_overlaps(final_placements)
    if not ok:
        return None
    if mode == "disjoint" and overlap_count != 0:
        return None
    if mode == "conflict" and overlap_count == 0:
        return None
    if overlap_count * 5 > colored_total:
        return None

    inp = np.zeros((height, width), dtype=int)
    for rec in final_placements:
        _render_pair_input(inp, rec)
    expected = _render_expected(inp.shape, final_placements)

    if not _components_ok(inp, pair_count):
        return None
    if not _pairing_ok(inp):
        return None
    if np.array_equal(inp, expected):
        return None
    if len({int(v) for v in np.unique(expected) if int(v) != BACKGROUND}) < 2:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        attempt = _generate_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid symmetric-outline input")

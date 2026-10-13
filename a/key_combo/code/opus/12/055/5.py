from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BLUE_MARKER = 1
RED_MARKER = 2
VALID_BLOCK_COLORS = (3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500

SLOT_FAMILY_A = (
    {"name": "east_bottom", "dr": -1, "dc": 1, "move": "east"},
    {"name": "west_top", "dr": 0, "dc": -2, "move": "west"},
    {"name": "south_left", "dr": 1, "dc": 0, "move": "south"},
    {"name": "north_right", "dr": -2, "dc": -1, "move": "north"},
)
SLOT_FAMILY_B = (
    {"name": "east_top", "dr": 0, "dc": 1, "move": "east"},
    {"name": "west_bottom", "dr": -1, "dc": -2, "move": "west"},
    {"name": "south_right", "dr": 1, "dc": -1, "move": "south"},
    {"name": "north_left", "dr": -2, "dc": 0, "move": "north"},
)
SLOT_FAMILIES = (SLOT_FAMILY_A, SLOT_FAMILY_B)

BRANCH_COUNTS = {
    "h": ((2, 1), (3, 1), (3, 2)),
    "v": ((1, 2), (1, 3), (2, 3)),
    "t": ((1, 1), (2, 2), (3, 3)),
}

UNALIGNED_DISTRACTOR_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "rows": set(rows),
                    "cols": set(cols),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _is_solid_2x2(comp: dict) -> bool:
    if int(comp["area"]) != 4:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return set(comp["coords"]) == {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
    }


def _marker_positions(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    markers = {}
    for color in (BLUE_MARKER, RED_MARKER):
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 1, "expected exactly one blue marker and one red marker"
        markers[color] = tuple(int(x) for x in coords[0])
    br, bc = markers[BLUE_MARKER]
    rr, rc = markers[RED_MARKER]
    assert br != rr and bc != rc, "markers must not share a row or column"
    return markers


def _aligned_marker(comp: dict, markers: dict[int, tuple[int, int]]) -> tuple[int, str]:
    hits = []
    for color, (mr, mc) in markers.items():
        row_align = mr in comp["rows"]
        col_align = mc in comp["cols"]
        if row_align or col_align:
            assert not (row_align and col_align), "component must align to a marker by exactly one axis"
            hits.append((color, "row" if row_align else "col"))
    assert len(hits) == 1, "each valid block must align with exactly one marker"
    return hits[0]


def _slide_bbox(comp: dict, marker: tuple[int, int], axis: str) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    mr, mc = marker
    if axis == "row":
        assert mr in comp["rows"] and mc not in comp["cols"]
        return r0, mc + 1 if mc < c0 else mc - 2
    assert axis == "col"
    assert mc in comp["cols"] and mr not in comp["rows"]
    return (mr + 1 if mr < r0 else mr - 2), c0


def _paint_patch(out: np.ndarray, r0: int, c0: int, patch: np.ndarray, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((target != bg) & (patch != bg)), "moved objects must not overlap"
    target[patch != bg] = patch[patch != bg]


def _core_scene_and_counts(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    markers = _marker_positions(g)
    comps = _find_components(g, bg)

    out = np.full_like(g, bg)
    for color, (r, c) in markers.items():
        out[r, c] = color

    red_count = 0
    blue_count = 0
    n_valid = 0
    for comp in comps:
        if int(comp["color"]) in (BLUE_MARKER, RED_MARKER):
            assert int(comp["area"]) == 1
            continue
        if not _is_solid_2x2(comp):
            continue
        marker_color, axis = _aligned_marker(comp, markers)
        r0, c0 = _slide_bbox(comp, markers[marker_color], axis)
        patch = np.full((2, 2), int(comp["color"]), dtype=int)
        _paint_patch(out, r0, c0, patch, bg)
        if marker_color == RED_MARKER:
            red_count += 1
        else:
            blue_count += 1
        n_valid += 1

    assert n_valid > 0, "expected at least one valid 2x2 block"
    return out, red_count, blue_count


def _axis_from_counts(red_count: int, blue_count: int) -> str:
    if red_count > blue_count:
        return "h"
    if blue_count > red_count:
        return "v"
    return "t"


def _apply_axis(grid: np.ndarray, axis: str) -> np.ndarray:
    if axis == "h":
        return grid[:, ::-1].copy()
    if axis == "v":
        return grid[::-1, :].copy()
    if axis == "t":
        return grid.T.copy()
    raise ValueError(f"unknown axis {axis}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter solid 2x2 blocks, slide them next to the correct marker, then mirror the whole scene using the marker-specific counts.

    Input structure:
    - The background is black.
    - There are exactly two singleton markers: blue(1) and red(2), and they do not share a row or column.
    - The remaining foreground consists of monochrome connected components: some are solid 2x2 blocks, while others are non-square distractors.
    - Every valid 2x2 block shares rows or columns with exactly one marker.

    Rule:
    - Ignore every non-square foreground component and keep only solid monochrome 2x2 blocks.
    - For each kept block, find its unique aligned marker and slide the block along the shared row or column until its edge is adjacent to that marker.
    - Count how many valid blocks end up attached to the red marker and how many attach to the blue marker.
    - If red has more, mirror the whole post-slide scene left-to-right; if blue has more, mirror it top-to-bottom; if the counts are equal, transpose across the main diagonal.

    Key insight:
    - The distractors are not just clutter: filtering them changes the per-marker counts, and those counts dynamically decide which global mirror operation must be applied.
    """
    core, red_count, blue_count = _core_scene_and_counts(grid)
    return _apply_axis(core, _axis_from_counts(red_count, blue_count))


def _block_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _can_place_shape(grid: np.ndarray, r0: int, c0: int, shape: np.ndarray, color: int, bg: int = BG) -> bool:
    h, w = shape.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(shape)}
    for rr, cc in cells:
        if int(grid[rr, cc]) != bg:
            return False
    for rr, cc in cells:
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in cells:
                if int(grid[nr, nc]) == color:
                    return False
    return True


def _place_shape(grid: np.ndarray, r0: int, c0: int, shape: np.ndarray, color: int) -> None:
    for dr, dc in np.argwhere(shape):
        grid[r0 + int(dr), c0 + int(dc)] = int(color)


def _marker_candidates(size: int) -> list[tuple[int, int]]:
    margin = 3
    return [(r, c) for r in range(margin, size - margin) for c in range(margin, size - margin)]


def _sample_markers(size: int) -> dict[int, tuple[int, int]] | None:
    candidates = _marker_candidates(size)
    for _ in range(80):
        blue = random.choice(candidates)
        red = random.choice(candidates)
        if blue == red:
            continue
        if abs(blue[0] - red[0]) < 2 or abs(blue[1] - red[1]) < 2:
            continue
        if abs(blue[0] - red[0]) + abs(blue[1] - red[1]) < 5:
            continue
        return {BLUE_MARKER: blue, RED_MARKER: red}
    return None


def _slot_output_topleft(marker: tuple[int, int], slot: dict) -> tuple[int, int]:
    return marker[0] + int(slot["dr"]), marker[1] + int(slot["dc"])


def _slot_axis(slot: dict) -> str:
    return "row" if slot["move"] in ("east", "west") else "col"


def _candidate_uniquely_aligns(marker: tuple[int, int], other_marker: tuple[int, int], slot: dict, r0: int, c0: int) -> bool:
    rows = {r0, r0 + 1}
    cols = {c0, c0 + 1}
    axis = _slot_axis(slot)
    if axis == "row":
        return marker[0] in rows and marker[1] not in cols and other_marker[0] not in rows and other_marker[1] not in cols
    return marker[1] in cols and marker[0] not in rows and other_marker[1] not in cols and other_marker[0] not in rows


def _input_positions_for_slot(size: int, marker: tuple[int, int], slot: dict) -> list[tuple[int, int]]:
    mr, mc = marker
    if slot["move"] == "east":
        return [(mr + int(slot["dr"]), mc + 1 + gap) for gap in range(1, size - mc - 2)]
    if slot["move"] == "west":
        return [(mr + int(slot["dr"]), mc - 2 - gap) for gap in range(1, mc - 1)]
    if slot["move"] == "south":
        return [(mr + 1 + gap, mc + int(slot["dc"])) for gap in range(1, size - mr - 2)]
    if slot["move"] == "north":
        return [(mr - 2 - gap, mc + int(slot["dc"])) for gap in range(1, mr - 1)]
    raise ValueError("invalid slot move")


def _pick_block_colors(total: int) -> list[int]:
    if total <= len(VALID_BLOCK_COLORS):
        return random.sample(list(VALID_BLOCK_COLORS), total)
    return [random.choice(VALID_BLOCK_COLORS) for _ in range(total)]


def _build_core_from_plan(
    size: int,
    markers: dict[int, tuple[int, int]],
    branch: str,
    red_slots: list[dict],
    blue_slots: list[dict],
) -> tuple[np.ndarray, list[dict]] | None:
    counts = {RED_MARKER: len(red_slots), BLUE_MARKER: len(blue_slots)}
    assert _axis_from_counts(counts[RED_MARKER], counts[BLUE_MARKER]) == branch

    core = np.full((size, size), BG, dtype=int)
    for color, (r, c) in markers.items():
        core[r, c] = color

    colors = _pick_block_colors(len(red_slots) + len(blue_slots))
    color_iter = iter(colors)
    blocks = []
    for marker_color, slots in ((RED_MARKER, red_slots), (BLUE_MARKER, blue_slots)):
        marker = markers[marker_color]
        other_marker = markers[BLUE_MARKER if marker_color == RED_MARKER else RED_MARKER]
        for slot in slots:
            r0, c0 = _slot_output_topleft(marker, slot)
            if not _candidate_uniquely_aligns(marker, other_marker, slot, r0, c0):
                return None
            patch = np.full((2, 2), next(color_iter), dtype=int)
            try:
                _paint_patch(core, r0, c0, patch, BG)
            except AssertionError:
                return None
            blocks.append(
                {
                    "marker_color": marker_color,
                    "marker": marker,
                    "slot": slot,
                    "output_r0": r0,
                    "output_c0": c0,
                    "color": int(patch[0, 0]),
                }
            )
    return core, blocks


def _derive_input_from_core(size: int, markers: dict[int, tuple[int, int]], blocks: list[dict]) -> np.ndarray | None:
    grid = np.full((size, size), BG, dtype=int)
    for color, (r, c) in markers.items():
        grid[r, c] = color

    order = blocks[:]
    random.shuffle(order)
    for block in order:
        candidates = _input_positions_for_slot(size, block["marker"], block["slot"])
        random.shuffle(candidates)
        other_marker = markers[BLUE_MARKER if block["marker_color"] == RED_MARKER else RED_MARKER]
        placed = False
        for r0, c0 in candidates:
            shape = np.ones((2, 2), dtype=bool)
            if not _candidate_uniquely_aligns(block["marker"], other_marker, block["slot"], r0, c0):
                continue
            if _can_place_shape(grid, r0, c0, shape, block["color"]):
                _place_shape(grid, r0, c0, shape, block["color"])
                placed = True
                break
        if not placed:
            return None
    return grid


def _aligned_distractor_plan(branch: str, red_count: int, blue_count: int) -> list[int]:
    if branch == "h":
        needed = red_count - blue_count
        return [BLUE_MARKER] * needed
    if branch == "v":
        needed = blue_count - red_count
        return [RED_MARKER] * needed
    return [random.choice((RED_MARKER, BLUE_MARKER))]


def _place_required_aligned_distractors(grid: np.ndarray, markers: dict[int, tuple[int, int]], branch: str) -> bool:
    red_count = 0
    blue_count = 0
    for comp in _find_components(grid, BG):
        if _is_solid_2x2(comp):
            marker_color, _ = _aligned_marker(comp, markers)
            if marker_color == RED_MARKER:
                red_count += 1
            else:
                blue_count += 1

    targets = _aligned_distractor_plan(branch, red_count, blue_count)
    random.shuffle(targets)
    area4_used = False
    for idx, target_color in enumerate(targets):
        other_color = BLUE_MARKER if target_color == RED_MARKER else RED_MARKER
        target = markers[target_color]
        other = markers[other_color]

        shapes = []
        if not area4_used:
            shapes.extend(
                [
                    ("row", np.array([[1, 1, 1, 1]], dtype=bool)),
                    ("col", np.array([[1], [1], [1], [1]], dtype=bool)),
                ]
            )
        shapes.extend(
            [
                ("row", np.array([[1, 1, 1]], dtype=bool)),
                ("col", np.array([[1], [1], [1]], dtype=bool)),
                ("row", np.array([[1]], dtype=bool)),
                ("col", np.array([[1]], dtype=bool)),
            ]
        )
        random.shuffle(shapes)

        placed = False
        for mode, shape in shapes:
            color = random.choice(VALID_BLOCK_COLORS)
            if mode == "row":
                row = target[0]
                width = shape.shape[1]
                starts = list(range(0, grid.shape[1] - width + 1))
                random.shuffle(starts)
                for c0 in starts:
                    cols = set(range(c0, c0 + width))
                    if target[1] in cols or other[1] in cols:
                        continue
                    if row == other[0]:
                        continue
                    if _can_place_shape(grid, row, c0, shape, color):
                        _place_shape(grid, row, c0, shape, color)
                        area4_used |= int(shape.sum()) == 4
                        placed = True
                        break
            else:
                col = target[1]
                height = shape.shape[0]
                starts = list(range(0, grid.shape[0] - height + 1))
                random.shuffle(starts)
                for r0 in starts:
                    rows = set(range(r0, r0 + height))
                    if target[0] in rows or other[0] in rows:
                        continue
                    if col == other[1]:
                        continue
                    if _can_place_shape(grid, r0, col, shape, color):
                        _place_shape(grid, r0, col, shape, color)
                        area4_used |= int(shape.sum()) == 4
                        placed = True
                        break
            if placed:
                break
        if not placed:
            return False
        if idx == 0 and not area4_used:
            return False
    return True


def _component_aligned_to_marker(comp: dict, markers: dict[int, tuple[int, int]]) -> int | None:
    hits = []
    for color, (mr, mc) in markers.items():
        if mr in comp["rows"] or mc in comp["cols"]:
            if (mr in comp["rows"]) and (mc in comp["cols"]):
                continue
            hits.append(color)
    if len(hits) == 1:
        return int(hits[0])
    return None


def _place_extra_unaligned_distractors(grid: np.ndarray, markers: dict[int, tuple[int, int]]) -> None:
    n_extra = random.randint(0, 2)
    size = grid.shape[0]
    for _ in range(n_extra):
        shapes = list(UNALIGNED_DISTRACTOR_SHAPES)
        random.shuffle(shapes)
        placed = False
        for shape in shapes:
            starts = [
                (r0, c0)
                for r0 in range(size - shape.shape[0] + 1)
                for c0 in range(size - shape.shape[1] + 1)
            ]
            random.shuffle(starts)
            color = random.choice(VALID_BLOCK_COLORS)
            for r0, c0 in starts:
                if not _can_place_shape(grid, r0, c0, shape, color):
                    continue
                rows = {r0 + int(r) for r, _ in np.argwhere(shape)}
                cols = {c0 + int(c) for _, c in np.argwhere(shape)}
                bad = False
                for mr, mc in markers.values():
                    if (mr in rows) ^ (mc in cols):
                        bad = True
                        break
                if bad:
                    continue
                _place_shape(grid, r0, c0, shape, color)
                placed = True
                break
            if placed:
                break


def _counts_from_all_aligned_components(grid: np.ndarray, markers: dict[int, tuple[int, int]]) -> tuple[int, int]:
    red_count = 0
    blue_count = 0
    for comp in _find_components(grid, BG):
        if int(comp["color"]) in (BLUE_MARKER, RED_MARKER):
            continue
        marker_color = _component_aligned_to_marker(comp, markers)
        if marker_color == RED_MARKER:
            red_count += 1
        elif marker_color == BLUE_MARKER:
            blue_count += 1
    return red_count, blue_count


def _slide_all_aligned_components(grid: np.ndarray, markers: dict[int, tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    for color, (r, c) in markers.items():
        out[r, c] = color

    red_count = 0
    blue_count = 0
    for comp in _find_components(g, BG):
        if int(comp["color"]) in (BLUE_MARKER, RED_MARKER):
            continue
        marker_color = _component_aligned_to_marker(comp, markers)
        if marker_color is None:
            continue
        marker = markers[marker_color]
        row_align = marker[0] in comp["rows"]
        axis = "row" if row_align else "col"
        r0, c0 = _slide_bbox(comp, marker, axis)
        patch = np.full((comp["bbox"][2] - comp["bbox"][0] + 1, comp["bbox"][3] - comp["bbox"][1] + 1), BG, dtype=int)
        base_r, base_c = comp["bbox"][0], comp["bbox"][1]
        for rr, cc in comp["coords"]:
            patch[rr - base_r, cc - base_c] = int(comp["color"])
        try:
            _paint_patch(out, r0, c0, patch, BG)
        except AssertionError:
            return out, -1, -1
        if marker_color == RED_MARKER:
            red_count += 1
        else:
            blue_count += 1
    return out, red_count, blue_count


def _clean_input_without_distractors(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    markers = _marker_positions(g)
    for color, (r, c) in markers.items():
        out[r, c] = color
    for comp in _find_components(g, BG):
        if int(comp["color"]) in (BLUE_MARKER, RED_MARKER):
            continue
        if _is_solid_2x2(comp):
            r0, c0, _, _ = comp["bbox"]
            out[r0 : r0 + 2, c0 : c0 + 2] = int(comp["color"])
    return out


def _semantic_checks(inp: np.ndarray, core: np.ndarray, branch: str) -> bool:
    markers = _marker_positions(inp)
    out = main(inp)
    expected = _apply_axis(core, branch)
    if not np.array_equal(out, expected):
        return False

    variants = {_axis: _apply_axis(core, _axis) for _axis in ("h", "v", "t")}
    if np.array_equal(variants["h"], variants["v"]):
        return False
    if np.array_equal(variants["h"], variants["t"]):
        return False
    if np.array_equal(variants["v"], variants["t"]):
        return False

    naive_red, naive_blue = _counts_from_all_aligned_components(inp, markers)
    naive_axis = _axis_from_counts(naive_red, naive_blue)
    if naive_axis == branch:
        return False

    wrong_scene, wrong_red, wrong_blue = _slide_all_aligned_components(inp, markers)
    if wrong_red < 0:
        return False
    wrong_out = _apply_axis(wrong_scene, _axis_from_counts(wrong_red, wrong_blue))
    if np.array_equal(wrong_out, out):
        return False

    cleaned = _clean_input_without_distractors(inp)
    no_slide_out = _apply_axis(cleaned, branch)
    if np.array_equal(no_slide_out, out):
        return False

    return not np.array_equal(inp, out)


def _generate_attempt() -> np.ndarray | None:
    size = random.randint(11, 14)
    branch = random.choice(("h", "v", "t"))
    blue_count, red_count = None, None
    red_count, blue_count = random.choice(BRANCH_COUNTS[branch])

    markers = _sample_markers(size)
    if markers is None:
        return None

    red_family = random.choice(SLOT_FAMILIES)
    blue_family = random.choice(SLOT_FAMILIES)
    red_slots = random.sample(list(red_family), red_count)
    blue_slots = random.sample(list(blue_family), blue_count)

    built = _build_core_from_plan(size, markers, branch, red_slots, blue_slots)
    if built is None:
        return None
    core, blocks = built

    inp = _derive_input_from_core(size, markers, blocks)
    if inp is None:
        return None

    if not _place_required_aligned_distractors(inp, markers, branch):
        return None
    _place_extra_unaligned_distractors(inp, markers)

    try:
        if not _semantic_checks(inp, core, branch):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

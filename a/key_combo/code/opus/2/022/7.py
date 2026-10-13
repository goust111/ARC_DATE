import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

FRAME_SIDE_BY_CENTER = {
    RED: "top",
    BLUE: "bottom",
    GREEN: "left",
    YELLOW: "right",
}

SIDE_COLOR = {
    "top": RED,
    "bottom": BLUE,
    "left": GREEN,
    "right": YELLOW,
}

SHELL_COLORS = [5, 6, 7, 8, 9]
MASKS = [
    frozenset({(-1, 0), (0, -1), (0, 1), (1, 0)}),
    frozenset({(-1, -1), (-1, 1), (1, -1), (1, 1)}),
    frozenset({(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)}),
    frozenset(
        {
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        }
    ),
]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RING_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _largest_component(mask: np.ndarray):
    comps = _component_coords(mask)
    if not comps:
        raise AssertionError("expected at least one component")
    return max(comps, key=len)


def _find_frame(grid: np.ndarray):
    red = _largest_component(grid == RED)
    blue = _largest_component(grid == BLUE)
    green = _largest_component(grid == GREEN)
    yellow = _largest_component(grid == YELLOW)

    top_row = red[0][0]
    bottom_row = blue[0][0]
    left_col = green[0][1]
    right_col = yellow[0][1]

    r0, r1 = int(top_row), int(bottom_row)
    c0, c1 = int(left_col), int(right_col)

    assert r0 < r1 and c0 < c1
    assert np.all(grid[r0, c0 + 1 : c1] == RED)
    assert np.all(grid[r1, c0 + 1 : c1] == BLUE)
    assert np.all(grid[r0 + 1 : r1, c0] == GREEN)
    assert np.all(grid[r0 + 1 : r1, c1] == YELLOW)
    assert grid[r0, c0] == BLACK
    assert grid[r0, c1] == BLACK
    assert grid[r1, c0] == BLACK
    assert grid[r1, c1] == BLACK

    return {
        "outer": (r0, c0, r1, c1),
        "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _window_features(grid: np.ndarray, center_r: int, center_c: int):
    center_color = int(grid[center_r, center_c])
    shell_cells = []
    shell_colors = set()
    for dr, dc in RING_OFFSETS:
        rr = center_r + dr
        cc = center_c + dc
        value = int(grid[rr, cc])
        if value != BLACK:
            shell_cells.append((dr, dc))
            shell_colors.add(value)
    if len(shell_cells) < 4 or len(shell_colors) != 1:
        return None
    shell_color = next(iter(shell_colors))
    return {
        "center": (int(center_r), int(center_c)),
        "top_left": (int(center_r - 1), int(center_c - 1)),
        "center_color": center_color,
        "shell_color": int(shell_color),
        "mask": frozenset(shell_cells),
    }


def _parse_case(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    inner_top, inner_left, inner_bottom, inner_right = frame["inner"]

    candidates = []
    for r in range(inner_top + 1, inner_bottom):
        for c in range(inner_left + 1, inner_right):
            feat = _window_features(g, r, c)
            if feat is not None:
                candidates.append(feat)

    templates = {}
    complete = []
    for feat in candidates:
        if feat["center_color"] == BLACK:
            continue
        key = feat["shell_color"]
        val = (feat["center_color"], feat["mask"])
        if key in templates:
            assert templates[key] == val
        else:
            templates[key] = val
        complete.append(feat)

    incomplete = []
    for feat in candidates:
        if feat["center_color"] != BLACK:
            continue
        if feat["shell_color"] not in templates:
            continue
        center_color, mask = templates[feat["shell_color"]]
        if feat["mask"] != mask:
            continue
        incomplete.append(
            {
                "center": feat["center"],
                "top_left": feat["top_left"],
                "shell_color": feat["shell_color"],
                "shell_mask": mask,
                "center_color": center_color,
                "side": FRAME_SIDE_BY_CENTER[center_color],
            }
        )

    return {
        "grid": g,
        "frame": frame,
        "templates": templates,
        "complete": complete,
        "incomplete": incomplete,
    }


def _make_patch(shell_color: int, center_color: int, mask):
    patch = np.zeros((3, 3), dtype=int)
    for dr, dc in mask:
        patch[dr + 1, dc + 1] = int(shell_color)
    patch[1, 1] = int(center_color)
    return patch


def _clear_incomplete(out: np.ndarray, motif):
    center_r, center_c = motif["center"]
    out[center_r, center_c] = BLACK
    for dr, dc in motif["shell_mask"]:
        out[center_r + dr, center_c + dc] = BLACK


def _patch_fits(out: np.ndarray, patch: np.ndarray, top: int, left: int, inner_box):
    inner_top, inner_left, inner_bottom, inner_right = inner_box
    if top < inner_top or left < inner_left:
        return False
    if top + 2 > inner_bottom or left + 2 > inner_right:
        return False
    for pr in range(3):
        for pc in range(3):
            value = int(patch[pr, pc])
            if value != BLACK and out[top + pr, left + pc] != BLACK:
                return False
    return True


def _place_group(out: np.ndarray, motifs, side: str, inner_box):
    inner_top, inner_left, inner_bottom, inner_right = inner_box
    ordered = sorted(motifs, key=lambda m: (m["top_left"][0], m["top_left"][1]))
    next_cursor = inner_left if side in ("top", "bottom") else inner_top
    placements = []

    for motif in ordered:
        patch = _make_patch(motif["shell_color"], motif["center_color"], motif["shell_mask"])
        if side == "top":
            top = inner_top
            start = max(next_cursor, motif["top_left"][1], inner_left)
            found = None
            for left in range(start, inner_right - 1):
                if _patch_fits(out, patch, top, left, inner_box):
                    found = (top, left)
                    break
        elif side == "bottom":
            top = inner_bottom - 2
            start = max(next_cursor, motif["top_left"][1], inner_left)
            found = None
            for left in range(start, inner_right - 1):
                if _patch_fits(out, patch, top, left, inner_box):
                    found = (top, left)
                    break
        elif side == "left":
            left = inner_left
            start = max(next_cursor, motif["top_left"][0], inner_top)
            found = None
            for top in range(start, inner_bottom - 1):
                if _patch_fits(out, patch, top, left, inner_box):
                    found = (top, left)
                    break
        else:
            left = inner_right - 2
            start = max(next_cursor, motif["top_left"][0], inner_top)
            found = None
            for top in range(start, inner_bottom - 1):
                if _patch_fits(out, patch, top, left, inner_box):
                    found = (top, left)
                    break
        assert found is not None, "could not place moved motif"
        top, left = found
        for pr in range(3):
            for pc in range(3):
                if patch[pr, pc] != BLACK:
                    out[top + pr, left + pc] = patch[pr, pc]
        placements.append((motif, top, left, patch))
        next_cursor = (left if side in ("top", "bottom") else top) + 4

    return placements


def _expected_placements(parsed):
    out = parsed["grid"].copy()
    for motif in parsed["incomplete"]:
        _clear_incomplete(out, motif)

    all_placements = []
    for side in ("top", "bottom", "left", "right"):
        motifs = [m for m in parsed["incomplete"] if m["side"] == side]
        all_placements.extend(_place_group(out, motifs, side, parsed["frame"]["inner"]))
    return out, all_placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete shell-only 3x3 motifs to reveal their center color, then move each completed motif to the matching frame side.

    Input structure:
    - A black background contains a one-cell-thick rectangular frame whose top, bottom, left, and right sides use four distinct colors while the four corners stay black.
    - Inside the frame are 3x3 motifs: a few complete templates with a colored center plus a same-colored shell mask, and several incomplete copies that keep only the shell.
    - Each shell color appears in a complete template, so the complete templates reveal which hidden center color belongs to that shell mask.

    Rule:
    - Read every complete 3x3 template and build a mapping from shell color to the template's center color and shell-mask offsets.
    - For each incomplete 3x3 motif whose center is black, restore the missing center using the matching template.
    - Use that restored center color to choose the destination frame side, clear the motif from its original location, and place the completed 3x3 motif against the inside of the matching side.
    - If several motifs move to the same side, sort them by original position from top-left to bottom-right and place them along that side with one blank cell between consecutive 3x3 footprints; complete templates stay fixed.

    Key insight:
    - The shell-only motifs do not directly say where to move, but the complete templates reveal the hidden center color, and that center color is exactly the visible code that matches one frame side.
    """
    parsed = _parse_case(np.asarray(grid, dtype=int))
    assert parsed["templates"], "expected at least one complete template"
    assert parsed["incomplete"], "expected at least one incomplete motif"
    out, _ = _expected_placements(parsed)
    return out


def _draw_frame(grid: np.ndarray, outer):
    r0, c0, r1, c1 = outer
    grid[r0, c0 + 1 : c1] = RED
    grid[r1, c0 + 1 : c1] = BLUE
    grid[r0 + 1 : r1, c0] = GREEN
    grid[r0 + 1 : r1, c1] = YELLOW


def _position_lattice(inner):
    inner_top, inner_left, inner_bottom, inner_right = inner
    rows = list(range(inner_top + 3, inner_bottom - 4, 4))
    cols = list(range(inner_left + 3, inner_right - 4, 4))
    return rows, cols


def _choose_group_positions(rows, cols, count, orientation, used, rng, want_collision):
    available = {(r, c) for r in rows for c in cols if (r, c) not in used}
    if len(available) < count:
        raise ValueError("not enough lattice positions")

    chosen = []
    if want_collision and count > 1:
        if orientation == "horizontal":
            shared_cols = [c for c in cols if sum((r, c) in available for r in rows) >= 2]
            if shared_cols:
                c = rng.choice(shared_cols)
                same_col_rows = [r for r in rows if (r, c) in available]
                take = rng.randint(2, min(count, len(same_col_rows)))
                for r in rng.sample(same_col_rows, take):
                    chosen.append((r, c))
        else:
            shared_rows = [r for r in rows if sum((r, c) in available for c in cols) >= 2]
            if shared_rows:
                r = rng.choice(shared_rows)
                same_row_cols = [c for c in cols if (r, c) in available]
                take = rng.randint(2, min(count, len(same_row_cols)))
                for c in rng.sample(same_row_cols, take):
                    chosen.append((r, c))
        available -= set(chosen)

    remaining = count - len(chosen)
    if remaining > 0:
        chosen.extend(rng.sample(sorted(available), remaining))

    rng.shuffle(chosen)
    used.update(chosen)
    return chosen


def _build_specs(center_colors, rng):
    shell_colors = rng.sample(SHELL_COLORS, len(center_colors))
    masks = rng.sample(MASKS, len(center_colors))
    specs = []
    for center_color, shell_color, mask in zip(center_colors, shell_colors, masks):
        specs.append(
            {
                "center_color": center_color,
                "shell_color": shell_color,
                "mask": mask,
                "side": FRAME_SIDE_BY_CENTER[center_color],
            }
        )
    return specs


def _allocate_counts(type_count: int, total_incomplete: int, rng):
    counts = [1] * type_count
    for _ in range(total_incomplete - type_count):
        counts[rng.randrange(type_count)] += 1
    return counts


def _paint_motif(grid: np.ndarray, top: int, left: int, shell_color: int, mask, center_color: int | None):
    center_r = top + 1
    center_c = left + 1
    for dr, dc in mask:
        grid[center_r + dr, center_c + dc] = int(shell_color)
    if center_color is not None:
        grid[center_r, center_c] = int(center_color)


def _validate_case(grid: np.ndarray):
    parsed = _parse_case(grid)
    assert 1 <= len(parsed["templates"]) <= 2
    assert 2 <= len(parsed["incomplete"]) <= 4
    used_shells = {motif["shell_color"] for motif in parsed["incomplete"]}
    assert used_shells == set(parsed["templates"])

    motif_boxes = [((m["top_left"][0], m["top_left"][1], m["top_left"][0] + 2, m["top_left"][1] + 2)) for m in parsed["complete"]]
    motif_boxes.extend(
        (m["top_left"][0], m["top_left"][1], m["top_left"][0] + 2, m["top_left"][1] + 2) for m in parsed["incomplete"]
    )
    for i, box_a in enumerate(motif_boxes):
        for box_b in motif_boxes[i + 1 :]:
            assert box_a[2] + 1 < box_b[0] or box_b[2] + 1 < box_a[0] or box_a[3] + 1 < box_b[1] or box_b[3] + 1 < box_a[1]

    expected, placements = _expected_placements(parsed)
    actual = main(grid)
    assert np.array_equal(expected, actual)
    assert not np.array_equal(actual, grid)

    for motif, top, left, patch in placements:
        assert top == parsed["frame"]["inner"][0] or left == parsed["frame"]["inner"][1] or top + 2 == parsed["frame"]["inner"][2] or left + 2 == parsed["frame"]["inner"][3]
        window = crop_bbox(actual, (top, left, top + 2, left + 2))
        assert np.array_equal(window, patch)

    return True


def _gen_attempt():
    rng = random.Random()

    type_count = rng.choices([1, 2], weights=[2, 3])[0]
    incomplete_total = rng.choices([2, 3, 4], weights=[1, 2, 2])[0]
    incomplete_total = max(incomplete_total, type_count + 1 if type_count == 1 else type_count)

    interior_h = rng.choice([17, 18, 19, 20, 21, 22])
    interior_w = rng.choice([17, 18, 19, 20, 21, 22, 23])
    top_margin = rng.choice([1, 2])
    bottom_margin = rng.choice([1, 2])
    left_margin = rng.choice([1, 2])
    right_margin = rng.choice([1, 2])

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + interior_h + 1
    c1 = c0 + interior_w + 1
    h = r1 + bottom_margin + 1
    w = c1 + right_margin + 1
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    _draw_frame(grid, (r0, c0, r1, c1))

    inner = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    rows, cols = _position_lattice(inner)
    if len(rows) * len(cols) < type_count + incomplete_total:
        return None

    center_colors = rng.sample(list(FRAME_SIDE_BY_CENTER), type_count)
    specs = _build_specs(center_colors, rng)
    counts = _allocate_counts(type_count, incomplete_total, rng)

    used = set()
    incomplete_records = []
    for spec, count in zip(specs, counts):
        orientation = "horizontal" if spec["side"] in ("top", "bottom") else "vertical"
        want_collision = count > 1 and rng.random() < 0.7
        positions = _choose_group_positions(rows, cols, count, orientation, used, rng, want_collision)
        for top, left in positions:
            incomplete_records.append((spec, top, left))

    complete_positions = rng.sample(sorted(set((r, c) for r in rows for c in cols) - used), type_count)

    for spec, (top, left) in zip(specs, complete_positions):
        _paint_motif(grid, top, left, spec["shell_color"], spec["mask"], spec["center_color"])

    for spec, top, left in incomplete_records:
        _paint_motif(grid, top, left, spec["shell_color"], spec["mask"], None)

    try:
        _validate_case(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

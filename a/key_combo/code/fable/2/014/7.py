from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
FILL_COLORS = [3, 4, 5, 6, 7, 8, 9]

EDGE_ORDER = ("top", "right", "bottom", "left")
DIM_OPTIONS = [(3, 5), (5, 3), (3, 3), (5, 5)]
DIM_WEIGHTS = [4, 4, 1, 1]
MIN_START_GAP = 2
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _control_edge(r: int, c: int, h: int, w: int) -> str:
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top + on_bottom + on_left + on_right == 1, "control pixel must lie on one non-corner edge"
    if on_top:
        return "top"
    if on_right:
        return "right"
    if on_bottom:
        return "bottom"
    return "left"


def _rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, k=(4 - (int(k) % 4)) % 4)


def _gap_to_axis(axis_kind: str, side: int, axis_row: int, axis_col: int, r0: int, c0: int, patch: np.ndarray) -> int:
    h, w = patch.shape
    if axis_kind == "vertical":
        if side < 0:
            return int(axis_row - 1 - (r0 + h - 1))
        return int(r0 - (axis_row + 1))
    if side < 0:
        return int(axis_col - 1 - (c0 + w - 1))
    return int(c0 - (axis_col + 1))


def _can_place(mask: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > mask.shape[0] or c0 + w > mask.shape[1]:
        return False
    return not mask[r0 : r0 + h, c0 : c0 + w].any()


def _slide_until_rest(mask: np.ndarray, patch: np.ndarray, r0: int, c0: int, dr: int, dc: int) -> tuple[int, int]:
    assert _can_place(mask, patch, r0, c0)
    while _can_place(mask, patch, r0 + dr, c0 + dc):
        r0 += dr
        c0 += dc
    return int(r0), int(c0)


def _find_axes(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    red_rows = np.where((grid == RED).sum(axis=1) >= w - 1)[0]
    blue_cols = np.where((grid == BLUE).sum(axis=0) >= h - 1)[0]
    assert red_rows.size == 1 and blue_cols.size == 1, "expected one red row and one blue column"
    return int(red_rows[0]), int(blue_cols[0])


def _extract_objects(grid: np.ndarray, axis_row: int, axis_col: int) -> list[dict]:
    line_mask = np.zeros_like(grid, dtype=bool)
    line_mask[axis_row, :] = True
    line_mask[:, axis_col] = True
    mask = (grid != BG) & (~line_mask)
    out: list[dict] = []

    for coords in components_mask(mask):
        b = bbox_coords(coords)
        patch = crop_bbox(grid, b)
        h, w = patch.shape
        ctrl_coords = np.argwhere((patch == RED) | (patch == BLUE))
        assert ctrl_coords.shape == (1, 2), "each rectangle needs exactly one control pixel"
        mr, mc = map(int, ctrl_coords[0])
        control_color = int(patch[mr, mc])
        fill_colors = {int(v) for v in patch.ravel() if int(v) not in (control_color, BG)}
        assert len(fill_colors) == 1 and control_color in (RED, BLUE)
        fill_color = int(next(iter(fill_colors)))
        expected = np.full((h, w), fill_color, dtype=int)
        expected[mr, mc] = control_color
        assert np.array_equal(patch, expected), "component must be a solid rectangle with one edge control pixel"

        edge = _control_edge(mr, mc, h, w)
        r0, c0, r1, c1 = b
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0

        if control_color == RED:
            assert r1 < axis_row or r0 > axis_row
            axis_kind = "vertical"
            side = -1 if r1 < axis_row else 1
            desired_edge = "bottom" if side < 0 else "top"
            dr, dc = (1, 0) if side < 0 else (-1, 0)
            ortho_key = int(c0)
        else:
            assert c1 < axis_col or c0 > axis_col
            axis_kind = "horizontal"
            side = -1 if c1 < axis_col else 1
            desired_edge = "right" if side < 0 else "left"
            dr, dc = (0, 1) if side < 0 else (0, -1)
            ortho_key = int(r0)

        k_cw = (EDGE_ORDER.index(desired_edge) - EDGE_ORDER.index(edge)) % 4
        rotated = _rotate_cw(patch, k_cw)
        rh, rw = rotated.shape
        start_r0 = int(round(center_r - (rh - 1) / 2.0))
        start_c0 = int(round(center_c - (rw - 1) / 2.0))
        gap = _gap_to_axis(axis_kind, side, axis_row, axis_col, start_r0, start_c0, rotated)
        assert gap >= 0

        out.append(
            {
                "axis_kind": axis_kind,
                "side": int(side),
                "gap": int(gap),
                "ortho_key": int(ortho_key),
                "move_key": int(start_r0 if axis_kind == "vertical" else start_c0),
                "start_r0": int(start_r0),
                "start_c0": int(start_c0),
                "dr": int(dr),
                "dc": int(dc),
                "rotated": rotated,
            }
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each controlled rectangle so its marked edge faces the matched axis, then let it fall and stack against that axis.

    Input structure:
    - The background is black.
    - One red horizontal axis line and one blue vertical axis line cross inside the grid.
    - Several solid monochrome rectangles sit in the four quadrants, each with exactly one red or blue control pixel embedded in one non-corner edge.

    Rule:
    - A red control pixel sends its rectangle toward the red horizontal axis by vertical motion; a blue control pixel sends its rectangle toward the blue vertical axis by horizontal motion.
    - Before moving, rotate the whole rectangle clockwise by 0/90/180/270 degrees until the edge containing the control pixel becomes the face pointing at the target axis.
    - For each side of each axis, process rectangles from smaller starting axis-gap to larger starting axis-gap; each slides straight until the next step would hit an axis line or an earlier rectangle from the same stream.

    Key insight:
    - The control pixel encodes two independent decisions at once: its color chooses which axis acts as gravity, and its edge location chooses the pre-fall orientation that must face that axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG

    axis_row, axis_col = _find_axes(g)
    line_mask = np.zeros_like(g, dtype=bool)
    line_mask[axis_row, :] = True
    line_mask[:, axis_col] = True

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for obj in _extract_objects(g, axis_row, axis_col):
        groups[(obj["axis_kind"], obj["side"])].append(obj)

    landed: list[tuple[np.ndarray, int, int]] = []
    for key, items in groups.items():
        del key
        items.sort(key=lambda item: (item["gap"], item["ortho_key"], item["move_key"]))
        occupied = line_mask.copy()
        for item in items:
            patch = item["rotated"]
            r0 = item["start_r0"]
            c0 = item["start_c0"]
            assert _can_place(occupied, patch, r0, c0), "rotated start state must stay collision-free"
            fr, fc = _slide_until_rest(occupied, patch, r0, c0, item["dr"], item["dc"])
            occupied[fr : fr + patch.shape[0], fc : fc + patch.shape[1]] = True
            landed.append((patch, fr, fc))

    out = np.full_like(g, BG)
    out[line_mask] = g[line_mask]
    used = line_mask.copy()
    for patch, r0, c0 in landed:
        h, w = patch.shape
        assert not used[r0 : r0 + h, c0 : c0 + w].any(), "different streams may not overlap in a valid instance"
        paste(out, patch, r0, c0)
        used[r0 : r0 + h, c0 : c0 + w] = True
    return out


def _make_patch(h: int, w: int, fill_color: int, control_color: int, edge: str) -> np.ndarray:
    patch = np.full((h, w), fill_color, dtype=int)
    if edge == "top":
        rc = (0, random.randint(1, w - 2))
    elif edge == "bottom":
        rc = (h - 1, random.randint(1, w - 2))
    elif edge == "left":
        rc = (random.randint(1, h - 2), 0)
    else:
        rc = (random.randint(1, h - 2), w - 1)
    patch[rc] = control_color
    return patch


def _sample_dims() -> tuple[int, int]:
    return random.choices(DIM_OPTIONS, weights=DIM_WEIGHTS, k=1)[0]


def _sample_object(fill_color: int, control_color: int, face_edge: str) -> dict:
    h, w = _sample_dims()
    final_patch = _make_patch(h, w, fill_color, control_color, face_edge)
    k_cw = random.randint(0, 3)
    input_patch = np.rot90(final_patch, k=k_cw)
    return {
        "fill_color": int(fill_color),
        "control_color": int(control_color),
        "face_edge": str(face_edge),
        "k_cw": int(k_cw),
        "final_patch": final_patch,
        "input_patch": input_patch,
    }


def _choose_ordered_pair(
    near_lo: int,
    near_hi: int,
    far_lo: int,
    far_hi: int,
    sep: int,
    *,
    near_is_large: bool,
) -> tuple[int, int] | None:
    for _ in range(200):
        near = random.randint(near_lo, near_hi)
        if near_is_large:
            upper = min(far_hi, near - sep)
            if upper < far_lo:
                continue
            far = random.randint(far_lo, upper)
        else:
            lower = max(far_lo, near + sep)
            if lower > far_hi:
                continue
            far = random.randint(lower, far_hi)
        return int(near), int(far)
    return None


def _axis_ranges_vertical(specs: list[dict], axis_row: int, height: int, side: int) -> list[tuple[int, int, int]]:
    ranges = []
    for spec in specs:
        ih, _ = spec["input_patch"].shape
        fh, _ = spec["final_patch"].shape
        span = max(ih, fh) // 2
        if side < 0:
            lo = 1 + span
            hi = min(height - 2 - ih // 2, axis_row - 1 - MIN_START_GAP - fh // 2)
        else:
            lo = max(1 + ih // 2, axis_row + 1 + MIN_START_GAP + fh // 2)
            hi = height - 2 - span
        ranges.append((int(lo), int(hi), int(span)))
    return ranges


def _axis_ranges_horizontal(specs: list[dict], axis_col: int, width: int, side: int) -> list[tuple[int, int, int]]:
    ranges = []
    for spec in specs:
        _, iw = spec["input_patch"].shape
        _, fw = spec["final_patch"].shape
        span = max(iw, fw) // 2
        if side < 0:
            lo = 1 + span
            hi = min(width - 2 - iw // 2, axis_col - 1 - MIN_START_GAP - fw // 2)
        else:
            lo = max(1 + iw // 2, axis_col + 1 + MIN_START_GAP + fw // 2)
            hi = width - 2 - span
        ranges.append((int(lo), int(hi), int(span)))
    return ranges


def _place_quadrant_objects(quadrant: str, motion: str, specs: list[dict], h: int, w: int, axis_row: int, axis_col: int) -> list[dict] | None:
    row_side = -1 if quadrant[0] == "t" else 1
    col_side = -1 if quadrant[1] == "l" else 1

    if motion == "vertical":
        max_half_w = max(max(spec["input_patch"].shape[1], spec["final_patch"].shape[1]) // 2 for spec in specs)
        if col_side < 0:
            c_lo = 1 + max_half_w
            c_hi = axis_col - 2 - max_half_w
        else:
            c_lo = axis_col + 2 + max_half_w
            c_hi = w - 2 - max_half_w
        if c_lo > c_hi:
            return None

        ranges = _axis_ranges_vertical(specs, axis_row, h, row_side)
        if any(lo > hi for lo, hi, _ in ranges):
            return None

        center_c = random.randint(c_lo, c_hi)
        if len(specs) == 1:
            center_rs = [random.randint(ranges[0][0], ranges[0][1])]
        else:
            sep = ranges[0][2] + ranges[1][2] + 2
            pair = _choose_ordered_pair(
                ranges[0][0],
                ranges[0][1],
                ranges[1][0],
                ranges[1][1],
                sep,
                near_is_large=(row_side < 0),
            )
            if pair is None:
                return None
            center_rs = list(pair)

        placed = []
        cursor = axis_row - 1 if row_side < 0 else axis_row + 1
        for spec, center_r in zip(specs, center_rs):
            final_patch = spec["final_patch"]
            input_patch = spec["input_patch"]
            fh, fw = final_patch.shape
            ih, iw = input_patch.shape
            if row_side < 0:
                final_r1 = cursor
                final_r0 = final_r1 - fh + 1
                cursor = final_r0 - 1
            else:
                final_r0 = cursor
                final_r1 = final_r0 + fh - 1
                cursor = final_r1 + 1
            placed.append(
                {
                    **spec,
                    "quadrant": quadrant,
                    "motion": motion,
                    "input_r0": int(center_r - ih // 2),
                    "input_c0": int(center_c - iw // 2),
                    "final_r0": int(final_r0),
                    "final_c0": int(center_c - fw // 2),
                }
            )
        return placed

    max_half_h = max(max(spec["input_patch"].shape[0], spec["final_patch"].shape[0]) // 2 for spec in specs)
    if row_side < 0:
        r_lo = 1 + max_half_h
        r_hi = axis_row - 2 - max_half_h
    else:
        r_lo = axis_row + 2 + max_half_h
        r_hi = h - 2 - max_half_h
    if r_lo > r_hi:
        return None

    ranges = _axis_ranges_horizontal(specs, axis_col, w, col_side)
    if any(lo > hi for lo, hi, _ in ranges):
        return None

    center_r = random.randint(r_lo, r_hi)
    if len(specs) == 1:
        center_cs = [random.randint(ranges[0][0], ranges[0][1])]
    else:
        sep = ranges[0][2] + ranges[1][2] + 2
        pair = _choose_ordered_pair(
            ranges[0][0],
            ranges[0][1],
            ranges[1][0],
            ranges[1][1],
            sep,
            near_is_large=(col_side < 0),
        )
        if pair is None:
            return None
        center_cs = list(pair)

    placed = []
    cursor = axis_col - 1 if col_side < 0 else axis_col + 1
    for spec, center_c in zip(specs, center_cs):
        final_patch = spec["final_patch"]
        input_patch = spec["input_patch"]
        fh, fw = final_patch.shape
        ih, iw = input_patch.shape
        if col_side < 0:
            final_c1 = cursor
            final_c0 = final_c1 - fw + 1
            cursor = final_c0 - 1
        else:
            final_c0 = cursor
            final_c1 = final_c0 + fw - 1
            cursor = final_c1 + 1
        placed.append(
            {
                **spec,
                "quadrant": quadrant,
                "motion": motion,
                "input_r0": int(center_r - ih // 2),
                "input_c0": int(center_c - iw // 2),
                "final_r0": int(center_r - fh // 2),
                "final_c0": int(final_c0),
            }
        )
    return placed


def _render_axes(h: int, w: int, axis_row: int, axis_col: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[axis_row, :] = RED
    grid[:, axis_col] = BLUE
    grid[axis_row, axis_col] = random.choice([RED, BLUE])
    return grid


def _patch_bbox(r0: int, c0: int, patch: np.ndarray) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1)


def _bbox_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _validate_generated_pair(inp: np.ndarray, out: np.ndarray, placements: list[dict], axis_row: int, axis_col: int) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    h, w = inp.shape
    if ((inp == RED).sum(axis=1) >= w - 1).sum() != 1:
        return False
    if ((inp == BLUE).sum(axis=0) >= h - 1).sum() != 1:
        return False

    input_boxes = []
    output_boxes = []
    rotations = 0
    control_colors = set()
    group_gaps: dict[tuple[str, int], list[int]] = defaultdict(list)

    for item in placements:
        ir0, ic0 = item["input_r0"], item["input_c0"]
        fr0, fc0 = item["final_r0"], item["final_c0"]
        input_patch = item["input_patch"]
        final_patch = item["final_patch"]
        if not np.array_equal(inp[ir0 : ir0 + input_patch.shape[0], ic0 : ic0 + input_patch.shape[1]], input_patch):
            return False
        if not np.array_equal(out[fr0 : fr0 + final_patch.shape[0], fc0 : fc0 + final_patch.shape[1]], final_patch):
            return False

        ctrl = np.argwhere((input_patch == RED) | (input_patch == BLUE))
        if ctrl.shape != (1, 2):
            return False
        mr, mc = map(int, ctrl[0])
        try:
            _control_edge(mr, mc, *input_patch.shape)
        except AssertionError:
            return False

        if not np.array_equal(_rotate_cw(input_patch, item["k_cw"]), final_patch):
            return False

        rotations += int(item["k_cw"] != 0)
        control_colors.add(int(item["control_color"]))

        center_r = ir0 + (input_patch.shape[0] - 1) / 2.0
        center_c = ic0 + (input_patch.shape[1] - 1) / 2.0
        start_h, start_w = final_patch.shape
        start_r0 = int(round(center_r - (start_h - 1) / 2.0))
        start_c0 = int(round(center_c - (start_w - 1) / 2.0))
        if item["motion"] == "vertical":
            side = -1 if item["quadrant"][0] == "t" else 1
            gap = _gap_to_axis("vertical", side, axis_row, axis_col, start_r0, start_c0, final_patch)
            group_gaps[("vertical", side)].append(gap)
        else:
            side = -1 if item["quadrant"][1] == "l" else 1
            gap = _gap_to_axis("horizontal", side, axis_row, axis_col, start_r0, start_c0, final_patch)
            group_gaps[("horizontal", side)].append(gap)
        if gap < MIN_START_GAP:
            return False

        ibox = _patch_bbox(ir0, ic0, input_patch)
        obox = _patch_bbox(fr0, fc0, final_patch)
        if not all(0 <= v < bound for v, bound in ((ibox[0], h), (ibox[1], w), (ibox[2], h), (ibox[3], w))):
            return False
        if not all(0 <= v < bound for v, bound in ((obox[0], h), (obox[1], w), (obox[2], h), (obox[3], w))):
            return False
        input_boxes.append(ibox)
        output_boxes.append(obox)

    if control_colors != {RED, BLUE}:
        return False
    if rotations == 0:
        return False
    if not any(len([p for p in placements if p["quadrant"] == quad]) == 2 for quad in {p["quadrant"] for p in placements}):
        return False

    for i, a in enumerate(input_boxes):
        for b in input_boxes[i + 1 :]:
            if not _bbox_disjoint(a, b):
                return False
    for i, a in enumerate(output_boxes):
        for b in output_boxes[i + 1 :]:
            if not _bbox_disjoint(a, b):
                return False

    for gaps in group_gaps.values():
        if any(g < 0 for g in gaps):
            return False
    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(19, 27)
    w = random.randint(19, 27)
    axis_row = random.randint(7, h - 8)
    axis_col = random.randint(7, w - 8)

    quadrants = ["tl", "tr", "bl", "br"]
    vertical_quads = set(random.sample(quadrants, k=random.randint(1, 3)))
    stack_quads = set(random.sample(quadrants, k=random.choice([1, 2])))

    total_objects = 4 + len(stack_quads)
    fill_colors = random.sample(FILL_COLORS, total_objects)
    color_iter = iter(fill_colors)

    planned: dict[str, tuple[str, list[dict]]] = {}
    for quad in quadrants:
        motion = "vertical" if quad in vertical_quads else "horizontal"
        count = 2 if quad in stack_quads else 1
        if motion == "vertical":
            control_color = RED
            face_edge = "bottom" if quad[0] == "t" else "top"
        else:
            control_color = BLUE
            face_edge = "right" if quad[1] == "l" else "left"
        specs = [_sample_object(next(color_iter), control_color, face_edge) for _ in range(count)]
        planned[quad] = (motion, specs)

    placements: list[dict] = []
    for quad in quadrants:
        motion, specs = planned[quad]
        placed = _place_quadrant_objects(quad, motion, specs, h, w, axis_row, axis_col)
        if placed is None:
            return None
        placements.extend(placed)

    inp = _render_axes(h, w, axis_row, axis_col)
    out = inp.copy()
    for item in placements:
        ir0, ic0 = item["input_r0"], item["input_c0"]
        fr0, fc0 = item["final_r0"], item["final_c0"]
        input_patch = item["input_patch"]
        final_patch = item["final_patch"]
        if inp[ir0 : ir0 + input_patch.shape[0], ic0 : ic0 + input_patch.shape[1]].any():
            return None
        if out[fr0 : fr0 + final_patch.shape[0], fc0 : fc0 + final_patch.shape[1]].any():
            return None
        paste(inp, input_patch, ir0, ic0)
        paste(out, final_patch, fr0, fc0)

    if not _validate_generated_pair(inp, out, placements, axis_row, axis_col):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        pair = _generate_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid color-matched gravity sorting instance")

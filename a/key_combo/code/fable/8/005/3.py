from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
MAX_ATTEMPTS = 400
WORK_CANVAS = 30


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _hole_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return (~mask) & (~outside)


def _frame_perimeter_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _shift_for_side(side: str, obj_top: int, obj_left: int, obj_h: int, obj_w: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    if side == "top":
        return 1 - obj_top, 0
    if side == "bottom":
        return frame_h - obj_h - 1 - obj_top, 0
    if side == "left":
        return 0, 1 - obj_left
    assert side == "right"
    return 0, frame_w - obj_w - 1 - obj_left


def _analyze_frame(grid: np.ndarray, coords: np.ndarray) -> dict[str, object] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 6 or w < 6:
        return None

    local_mask = np.zeros((h, w), dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    perimeter = _frame_perimeter_mask((h, w))
    if not np.array_equal(local_mask, perimeter):
        return None

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    border_colors, border_counts = np.unique(region[perimeter], return_counts=True)
    if len(border_colors) != 2 or int(border_counts.min()) != 1:
        return None

    marker_color = int(border_colors[int(np.argmin(border_counts))])
    frame_color = int(border_colors[int(np.argmax(border_counts))])
    marker_pos = np.argwhere(region == marker_color)
    if marker_pos.shape != (1, 2):
        return None
    mr, mc = int(marker_pos[0, 0]), int(marker_pos[0, 1])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    if mr == 0:
        side = "top"
    elif mr == h - 1:
        side = "bottom"
    elif mc == 0:
        side = "left"
    elif mc == w - 1:
        side = "right"
    else:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    obj_comps = _components(interior != BLACK)
    if len(obj_comps) != 1:
        return None
    obj_local = obj_comps[0]
    obj_values = np.unique(interior[obj_local[:, 0], obj_local[:, 1]])
    if len(obj_values) != 1:
        return None
    obj_color = int(obj_values[0])
    if obj_color in {frame_color, marker_color, BLACK}:
        return None

    or0, oc0 = obj_local.min(axis=0)
    or1, oc1 = obj_local.max(axis=0)
    or0, oc0, or1, oc1 = int(or0), int(oc0), int(or1), int(oc1)
    obj_mask = np.zeros((or1 - or0 + 1, oc1 - oc0 + 1), dtype=bool)
    obj_mask[obj_local[:, 0] - or0, obj_local[:, 1] - oc0] = True
    holes = _hole_mask(obj_mask)
    if not holes.any():
        return None

    if side == "top" and or0 == 0:
        return None
    if side == "bottom" and or1 == interior.shape[0] - 1:
        return None
    if side == "left" and oc0 == 0:
        return None
    if side == "right" and oc1 == interior.shape[1] - 1:
        return None

    return {
        "frame_box": (r0, c0, r1, c1),
        "frame_color": frame_color,
        "marker_color": marker_color,
        "marker_local": (mr, mc),
        "side": side,
        "object_color": obj_color,
        "object_top": r0 + 1 + or0,
        "object_left": c0 + 1 + oc0,
        "object_mask": obj_mask,
        "hole_mask": holes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked frame's holed object to the marked side and fill its holes with the marker color.

    Input structure:
    - A black background contains several disconnected hollow rectangular frames.
    - Each frame border is monochrome except for exactly one non-corner marker pixel of a third color on one side.
    - Inside each frame is one 4-connected single-color object with at least one enclosed black hole, and it is not already touching the marker side.

    Rule:
    - For each frame, identify the marker side and translate the entire inner object straight toward that side until its bounding box is flush with the frame's inner edge on that side.
    - After moving it, fill every hole enclosed by the object with the marker color.
    - Restore the marker pixel to the frame color, and leave the frame, background, and all other pixels unchanged.

    Key insight:
    - The marker's position selects the slide direction via its side, while the marker's color separately specifies the hole-fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_specs: list[dict[str, object]] = []
    for comp in _components(g != BLACK):
        spec = _analyze_frame(g, comp)
        if spec is not None:
            frame_specs.append(spec)

    assert frame_specs, "expected at least one marked rectangular frame"
    out = g.copy()

    for spec in frame_specs:
        r0, c0, r1, c1 = spec["frame_box"]
        frame_h = int(r1 - r0 + 1)
        frame_w = int(c1 - c0 + 1)
        obj_mask = np.asarray(spec["object_mask"], dtype=bool)
        hole_mask = np.asarray(spec["hole_mask"], dtype=bool)
        obj_h, obj_w = obj_mask.shape
        obj_top = int(spec["object_top"]) - int(r0)
        obj_left = int(spec["object_left"]) - int(c0)
        dr, dc = _shift_for_side(str(spec["side"]), obj_top, obj_left, obj_h, obj_w, frame_h, frame_w)

        obj_cells = np.argwhere(obj_mask)
        abs_rows = int(r0) + obj_top + obj_cells[:, 0]
        abs_cols = int(c0) + obj_left + obj_cells[:, 1]
        out[abs_rows, abs_cols] = BLACK

        moved_rows = abs_rows + dr
        moved_cols = abs_cols + dc
        assert np.all((r0 + 1 <= moved_rows) & (moved_rows <= r1 - 1))
        assert np.all((c0 + 1 <= moved_cols) & (moved_cols <= c1 - 1))
        out[moved_rows, moved_cols] = int(spec["object_color"])

        hole_cells = np.argwhere(hole_mask)
        hole_rows = int(r0) + obj_top + dr + hole_cells[:, 0]
        hole_cols = int(c0) + obj_left + dc + hole_cells[:, 1]
        out[hole_rows, hole_cols] = int(spec["marker_color"])

        mr, mc = spec["marker_local"]
        out[int(r0) + int(mr), int(c0) + int(mc)] = int(spec["frame_color"])

    return out


def _make_holed_mask(obj_h: int, obj_w: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((obj_h, obj_w), dtype=bool)
    candidates: list[tuple[int, int, int, int]] = []
    for hh in range(1, min(3, obj_h - 1)):
        for hw in range(1, min(3, obj_w - 1)):
            for r0 in range(1, obj_h - hh):
                for c0 in range(1, obj_w - hw):
                    if r0 + hh < obj_h and c0 + hw < obj_w:
                        candidates.append((r0, c0, hh, hw))
    rng.shuffle(candidates)
    assert candidates

    chosen = [candidates[0]]
    if obj_h >= 7 and obj_w >= 7 and rng.random() < 0.45:
        for cand in candidates[1:]:
            r0, c0, hh, hw = cand
            ok = True
            for rr, cc, oh, ow in chosen:
                if not (r0 + hh < rr - 1 or rr + oh < r0 - 1 or c0 + hw < cc - 1 or cc + ow < c0 - 1):
                    ok = False
                    break
            if ok:
                chosen.append(cand)
                break

    for r0, c0, hh, hw in chosen:
        mask[r0 : r0 + hh, c0 : c0 + hw] = False

    holes = _hole_mask(mask)
    assert holes.any()
    return mask


def _choose_sides(count: int, rng: np.random.Generator) -> list[str]:
    sides = [str(rng.choice(["top", "bottom"])), str(rng.choice(["left", "right"]))]
    remaining = [side for side in SIDES if side not in sides]
    while len(sides) < count and remaining:
        idx = int(rng.integers(0, len(remaining)))
        sides.append(remaining.pop(idx))
    while len(sides) < count:
        sides.append(str(rng.choice(SIDES)))
    rng.shuffle(sides)
    return sides


def _build_patch(side: str, max_h: int, max_w: int, rng: np.random.Generator) -> dict[str, object]:
    for _ in range(200):
        frame_h = int(rng.integers(9, max_h + 1))
        frame_w = int(rng.integers(9, max_w + 1))
        inner_h = frame_h - 2
        inner_w = frame_w - 2
        if inner_h < 7 or inner_w < 7:
            continue

        max_obj_h = min(8, inner_h - 2)
        max_obj_w = min(8, inner_w - 2)
        if max_obj_h < 5 or max_obj_w < 5:
            continue
        obj_h = int(rng.integers(5, max_obj_h + 1))
        obj_w = int(rng.integers(5, max_obj_w + 1))

        total_v = inner_h - obj_h
        total_h = inner_w - obj_w
        if total_v < 2 or total_h < 2:
            continue

        gap_top = int(rng.integers(1, total_v))
        gap_bottom = total_v - gap_top
        gap_left = int(rng.integers(1, total_h))
        gap_right = total_h - gap_left
        if min(gap_top, gap_bottom, gap_left, gap_right) < 1:
            continue

        obj_top = 1 + gap_top
        obj_left = 1 + gap_left
        obj_mask = _make_holed_mask(obj_h, obj_w, rng)
        hole_mask = _hole_mask(obj_mask)

        colors = list(map(int, rng.choice(PALETTE, size=3, replace=False)))
        frame_color, object_color, marker_color = colors

        if side in {"top", "bottom"}:
            choices = [c for c in range(1, frame_w - 1) if c < obj_left or c > obj_left + obj_w - 1]
            if not choices:
                continue
            marker_r = 0 if side == "top" else frame_h - 1
            marker_c = int(rng.choice(choices))
            shift = gap_top if side == "top" else gap_bottom
        else:
            choices = [r for r in range(1, frame_h - 1) if r < obj_top or r > obj_top + obj_h - 1]
            if not choices:
                continue
            marker_r = int(rng.choice(choices))
            marker_c = 0 if side == "left" else frame_w - 1
            shift = gap_left if side == "left" else gap_right

        patch = np.zeros((frame_h, frame_w), dtype=int)
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color
        patch[marker_r, marker_c] = marker_color
        obj_cells = np.argwhere(obj_mask)
        patch[obj_top + obj_cells[:, 0], obj_left + obj_cells[:, 1]] = object_color

        return {
            "patch": patch,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "frame_color": frame_color,
            "object_color": object_color,
            "marker_color": marker_color,
            "side": side,
            "marker_r": marker_r,
            "marker_c": marker_c,
            "obj_top": obj_top,
            "obj_left": obj_left,
            "obj_mask": obj_mask,
            "hole_mask": hole_mask,
            "shift": shift,
        }
    raise RuntimeError("failed to build patch")


def _place_patches(specs: list[dict[str, object]], rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    canvas = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)
    placed: list[dict[str, object]] = []

    for spec in sorted(specs, key=lambda s: int(s["frame_h"]) * int(s["frame_w"]), reverse=True):
        patch = np.asarray(spec["patch"], dtype=int)
        ph, pw = patch.shape
        for _ in range(250):
            top = int(rng.integers(0, WORK_CANVAS - ph + 1))
            left = int(rng.integers(0, WORK_CANVAS - pw + 1))
            rr0 = max(0, top - 1)
            cc0 = max(0, left - 1)
            rr1 = min(WORK_CANVAS, top + ph + 1)
            cc1 = min(WORK_CANVAS, left + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            canvas[top : top + ph, left : left + pw] = patch
            blocked[rr0:rr1, cc0:cc1] = True
            new_spec = dict(spec)
            new_spec["origin"] = (top, left)
            placed.append(new_spec)
            break
        else:
            return None

    return canvas, placed


def _apply_expected(canvas: np.ndarray, specs: list[dict[str, object]], *, move: bool, fill_holes: bool) -> np.ndarray:
    out = canvas.copy()
    for spec in specs:
        top, left = spec["origin"]
        obj_mask = np.asarray(spec["obj_mask"], dtype=bool)
        hole_mask = np.asarray(spec["hole_mask"], dtype=bool)
        obj_h, obj_w = obj_mask.shape
        obj_top = int(spec["obj_top"])
        obj_left = int(spec["obj_left"])
        obj_cells = np.argwhere(obj_mask)
        abs_rows = int(top) + obj_top + obj_cells[:, 0]
        abs_cols = int(left) + obj_left + obj_cells[:, 1]

        out[abs_rows, abs_cols] = BLACK
        out[int(top) + int(spec["marker_r"]), int(left) + int(spec["marker_c"])] = int(spec["frame_color"])

        if move:
            dr, dc = _shift_for_side(
                str(spec["side"]),
                obj_top,
                obj_left,
                obj_h,
                obj_w,
                int(spec["frame_h"]),
                int(spec["frame_w"]),
            )
            dst_top = obj_top + dr
            dst_left = obj_left + dc
        else:
            dst_top = obj_top
            dst_left = obj_left

        out[int(top) + dst_top + obj_cells[:, 0], int(left) + dst_left + obj_cells[:, 1]] = int(spec["object_color"])
        if fill_holes:
            hole_cells = np.argwhere(hole_mask)
            out[int(top) + dst_top + hole_cells[:, 0], int(left) + dst_left + hole_cells[:, 1]] = int(spec["marker_color"])
    return out


def _adjust_specs(specs: list[dict[str, object]], crop: tuple[int, int, int, int]) -> list[dict[str, object]]:
    r0, c0, _, _ = crop
    adjusted = []
    for spec in specs:
        new_spec = dict(spec)
        top, left = spec["origin"]
        new_spec["origin"] = (int(top) - int(r0), int(left) - int(c0))
        adjusted.append(new_spec)
    return adjusted


def _validate_instance(inp: np.ndarray, expected: np.ndarray, specs: list[dict[str, object]]) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    wrong_no_slide = _apply_expected(inp, specs, move=False, fill_holes=True)
    wrong_no_fill = _apply_expected(inp, specs, move=True, fill_holes=False)
    if np.array_equal(wrong_no_slide, expected):
        return False
    if np.array_equal(wrong_no_fill, expected):
        return False

    sides = {str(spec["side"]) for spec in specs}
    if not sides.intersection({"top", "bottom"}):
        return False
    if not sides.intersection({"left", "right"}):
        return False
    if len({int(spec["marker_color"]) for spec in specs}) < 2:
        return False

    for spec in specs:
        if int(spec["shift"]) < 1:
            return False
        obj_mask = np.asarray(spec["obj_mask"], dtype=bool)
        if not _hole_mask(obj_mask).any():
            return False

        top, left = spec["origin"]
        obj_top = int(spec["obj_top"])
        obj_left = int(spec["obj_left"])
        obj_h, obj_w = obj_mask.shape
        marker_r = int(top) + int(spec["marker_r"])
        marker_c = int(left) + int(spec["marker_c"])
        row_lo = int(top) + obj_top
        row_hi = row_lo + obj_h - 1
        col_lo = int(left) + obj_left
        col_hi = col_lo + obj_w - 1
        side = str(spec["side"])

        if side in {"top", "bottom"} and col_lo <= marker_c <= col_hi:
            return False
        if side in {"left", "right"} and row_lo <= marker_r <= row_hi:
            return False

        dr, dc = _shift_for_side(
            side,
            obj_top,
            obj_left,
            obj_h,
            obj_w,
            int(spec["frame_h"]),
            int(spec["frame_w"]),
        )
        row_lo += dr
        row_hi += dr
        col_lo += dc
        col_hi += dc
        if side in {"top", "bottom"} and col_lo <= marker_c <= col_hi:
            return False
        if side in {"left", "right"} and row_lo <= marker_r <= row_hi:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        frame_count = 3 if rng.random() < 0.6 else 2
        max_h = 12 if frame_count == 3 else 14
        max_w = 12 if frame_count == 3 else 14

        specs = [_build_patch(side, max_h, max_w, rng) for side in _choose_sides(frame_count, rng)]
        placed = _place_patches(specs, rng)
        if placed is None:
            continue
        canvas, placed_specs = placed

        expected_canvas = _apply_expected(canvas, placed_specs, move=True, fill_holes=True)
        box = _bbox(canvas != BLACK)
        assert box is not None
        br0, bc0, br1, bc1 = box
        top_margin = int(rng.integers(0, min(br0, 2) + 1))
        left_margin = int(rng.integers(0, min(bc0, 2) + 1))
        bottom_margin = int(rng.integers(0, min(WORK_CANVAS - 1 - br1, 2) + 1))
        right_margin = int(rng.integers(0, min(WORK_CANVAS - 1 - bc1, 2) + 1))
        crop = (br0 - top_margin, bc0 - left_margin, br1 + bottom_margin, bc1 + right_margin)

        inp = _crop_bbox(canvas, crop)
        expected = _crop_bbox(expected_canvas, crop)
        cropped_specs = _adjust_specs(placed_specs, crop)

        if _validate_instance(inp, expected, cropped_specs):
            return inp

    raise RuntimeError("failed to generate a valid marked-frame sliding input")

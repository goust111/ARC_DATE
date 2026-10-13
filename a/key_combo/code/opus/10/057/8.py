from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OBJ_COLORS = (1, 2)
MARKER_TO_TURNS = {
    3: (1, 0),  # A cw90, B identity
    4: (2, 1),  # A cw180, B cw90
}
DIR_TO_SIDE = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CASE_ORDER = [
    ("up", 3),
    ("down", 3),
    ("left", 3),
    ("right", 3),
    ("up", 4),
    ("down", 4),
    ("left", 4),
    ("right", 4),
]
_CASE_INDEX = 0


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_to_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(patch: np.ndarray, turns: int) -> np.ndarray:
    t = int(turns) % 4
    if t == 0:
        return patch.copy()
    return np.rot90(patch, k=(4 - t) % 4)


def _centered_top_left(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    h, w = shape
    top_num = (r0 + r1) - (h - 1)
    left_num = (c0 + c1) - (w - 1)
    assert top_num % 2 == 0 and left_num % 2 == 0, "rotation center must land on grid cells"
    return top_num // 2, left_num // 2


def _occupied_coords(patch: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(patch != BG):
        coords.add((int(top + r), int(left + c)))
    return coords


def _shift_coords(coords: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _coord_bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _edge_pixels(coords: set[tuple[int, int]], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _coord_bbox(coords)
    if side == "top":
        return {(r, c) for r, c in coords if r == r0}
    if side == "bottom":
        return {(r, c) for r, c in coords if r == r1}
    if side == "left":
        return {(r, c) for r, c in coords if c == c0}
    if side == "right":
        return {(r, c) for r, c in coords if c == c1}
    raise ValueError(side)


def _has_diagonal_contact(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (r + dr, c + dc) in b:
                    return True
    return False


def _has_orth_contact(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
            return True
    return False


def _identify_structure(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int], int, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_coords = np.argwhere(np.isin(g, list(MARKER_TO_TURNS)))
    assert marker_coords.shape[0] == 1, "expected exactly one marker pixel"
    mr, mc = map(int, marker_coords[0])
    marker_color = int(g[mr, mc])

    obj_mask = np.isin(g, OBJ_COLORS)
    comps = _component_coords(obj_mask)
    assert len(comps) == 1, "expected one 4-connected two-color object"
    obj_coords = comps[0]
    box = _bbox(obj_coords)
    r0, c0, r1, c1 = box

    side_matches: list[str] = []
    if mr == r0 - 1 and c0 <= mc <= c1 and g[mr + 1, mc] in OBJ_COLORS:
        side_matches.append("up")
    if mr == r1 + 1 and c0 <= mc <= c1 and g[mr - 1, mc] in OBJ_COLORS:
        side_matches.append("down")
    if mc == c0 - 1 and r0 <= mr <= r1 and g[mr, mc + 1] in OBJ_COLORS:
        side_matches.append("left")
    if mc == c1 + 1 and r0 <= mr <= r1 and g[mr, mc - 1] in OBJ_COLORS:
        side_matches.append("right")
    assert len(side_matches) == 1, "marker side must be unique"

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 0 and w % 2 == 0, "object bbox must split evenly on both axes"
    return g, box, (mr, mc), marker_color, side_matches[0]


def _split_and_rotate(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    side: str,
    marker_color: int,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    patch = _crop_to_bbox(grid, box)
    h, w = patch.shape
    hh = h // 2
    hw = w // 2

    if side in ("up", "down"):
        box_a = (r0, c0, r0 + hh - 1, c1) if side == "up" else (r0 + hh, c0, r1, c1)
        box_b = (r0 + hh, c0, r1, c1) if side == "up" else (r0, c0, r0 + hh - 1, c1)
        patch_a = patch[:hh, :] if side == "up" else patch[hh:, :]
        patch_b = patch[hh:, :] if side == "up" else patch[:hh, :]
    else:
        box_a = (r0, c0, r1, c0 + hw - 1) if side == "left" else (r0, c0 + hw, r1, c1)
        box_b = (r0, c0 + hw, r1, c1) if side == "left" else (r0, c0, r1, c0 + hw - 1)
        patch_a = patch[:, :hw] if side == "left" else patch[:, hw:]
        patch_b = patch[:, hw:] if side == "left" else patch[:, :hw]

    a_turns, b_turns = MARKER_TO_TURNS[marker_color]
    rot_a = _rotate_cw(patch_a, a_turns)
    rot_b = _rotate_cw(patch_b, b_turns)
    top_a = _centered_top_left(box_a, rot_a.shape)
    top_b = _centered_top_left(box_b, rot_b.shape)
    return rot_a, top_a, rot_b, top_b, DIR_TO_SIDE[side]


def _find_slide_distance(
    grid_shape: tuple[int, int],
    patch_a: np.ndarray,
    top_a: tuple[int, int],
    patch_b: np.ndarray,
    top_b: tuple[int, int],
    move_dir: tuple[int, int],
) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid_shape
    dr, dc = move_dir
    coords_b = _occupied_coords(patch_b, *top_b)
    assert coords_b, "piece B must be non-empty"

    if move_dir == (-1, 0):
        a_side, b_side = "bottom", "top"
        max_d = top_a[0]
    elif move_dir == (1, 0):
        a_side, b_side = "top", "bottom"
        max_d = h - (top_a[0] + patch_a.shape[0])
    elif move_dir == (0, -1):
        a_side, b_side = "right", "left"
        max_d = top_a[1]
    elif move_dir == (0, 1):
        a_side, b_side = "left", "right"
        max_d = w - (top_a[1] + patch_a.shape[1])
    else:
        raise ValueError(move_dir)

    for d in range(max_d + 1):
        cur_top = (top_a[0] + dr * d, top_a[1] + dc * d)
        coords_a = _occupied_coords(patch_a, *cur_top)
        assert coords_a, "piece A must be non-empty"
        if not all(0 <= r < h and 0 <= c < w for r, c in coords_a):
            continue
        if coords_a & coords_b:
            continue

        edge_a = _edge_pixels(coords_a, a_side)
        edge_b = _edge_pixels(coords_b, b_side)
        if not edge_a or not edge_b:
            continue
        if _has_diagonal_contact(edge_a, edge_b) and not _has_orth_contact(coords_a, coords_b):
            return d, coords_a, coords_b

    raise AssertionError("no valid slide distance found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the marked two-color object by the marker's side, rotate the two halves by the marker's color, and slide the marked half away until the halves meet diagonally.

    Input structure:
    - The grid has black background, one 4-connected red/blue object with an even-height and even-width bounding box, and one singleton marker pixel.
    - The marker is green or yellow and lies just outside exactly one side of the object, 4-neighbor adjacent to that side.
    - The object's color pattern is asymmetric so different rotation plans remain visually distinguishable.

    Rule:
    - Use the marker side to choose the split axis: a marker above or below means a horizontal split, while a marker left or right means a vertical split.
    - Let half A be the side next to the marker and half B the opposite side.
    - If the marker is green, rotate A clockwise 90 degrees and keep B fixed; if the marker is yellow, rotate A clockwise 180 degrees and rotate B clockwise 90 degrees.
    - Keep each rotated half centered on its own original half-bounding-box center, then slide A one cell at a time in the marker direction until A and B become diagonally adjacent without orthogonal contact.
    - Erase the marker pixel.

    Key insight:
    - The single marker encodes two independent controls at once: its side selects the split axis and the near half, while its color selects the rotation pair applied before the final diagonal rejoin.
    """
    g, box, _marker, marker_color, side = _identify_structure(grid)
    patch_a, top_a, patch_b, top_b, move_dir = _split_and_rotate(g, box, side, marker_color)
    slide_d, _coords_a, _coords_b = _find_slide_distance(g.shape, patch_a, top_a, patch_b, top_b, move_dir)

    out = np.zeros_like(g)
    for rr, cc in np.argwhere(patch_b != BG):
        out[top_b[0] + rr, top_b[1] + cc] = int(patch_b[rr, cc])
    dr, dc = move_dir
    final_top_a = (top_a[0] + dr * slide_d, top_a[1] + dc * slide_d)
    for rr, cc in np.argwhere(patch_a != BG):
        out[final_top_a[0] + rr, final_top_a[1] + cc] = int(patch_a[rr, cc])
    return out


def _interp_interval(width: int, start: tuple[int, int], end: tuple[int, int], idx: int, count: int) -> tuple[int, int]:
    if count == 1:
        return start
    alpha_num = idx
    alpha_den = count - 1
    left = round((start[0] * (alpha_den - alpha_num) + end[0] * alpha_num) / alpha_den)
    right = round((start[1] * (alpha_den - alpha_num) + end[1] * alpha_num) / alpha_den)
    left = max(0, min(left, width - 1))
    right = max(left, min(right, width - 1))
    return left, right


def _wedge_half(height: int, width: int, near_width: int, side: str, toward_split: bool) -> np.ndarray:
    if side == "left":
        narrow = (0, near_width - 1)
    else:
        narrow = (width - near_width, width - 1)
    full = (0, width - 1)
    start, end = (full, narrow) if toward_split else (narrow, full)

    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        left, right = _interp_interval(width, start, end, r, height)
        mask[r, left : right + 1] = True
    return mask


def _anchored_interval(width: int, span: int, side: str) -> tuple[int, int]:
    if side == "left":
        return 0, span - 1
    if side == "right":
        return width - span, width - 1
    raise ValueError(side)


def _profile_half(
    height: int,
    width: int,
    top_span: int,
    bottom_span: int,
    top_side: str,
    bottom_side: str,
) -> np.ndarray:
    top = _anchored_interval(width, top_span, top_side)
    bottom = _anchored_interval(width, bottom_span, bottom_side)
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        left, right = _interp_interval(width, top, bottom, r, height)
        mask[r, left : right + 1] = True
    return mask


def _jitter_half(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for r in range(1, h - 1):
        row = np.flatnonzero(out[r])
        if row.size == 0:
            continue
        left = int(row.min())
        right = int(row.max())
        if left > 0 and random.random() < 0.35:
            out[r, left] = False
        if right < w - 1 and random.random() < 0.35:
            out[r, right] = False
        if not out[r].any():
            out[r, left:right + 1] = True
    return out


def _canonical_mask(box_h: int, box_w: int) -> np.ndarray:
    half_h = box_h // 2
    for _ in range(300):
        top = _profile_half(
            half_h,
            box_w,
            top_span=random.randint(2, box_w - 1),
            bottom_span=random.randint(1, box_w),
            top_side=random.choice(["left", "right"]),
            bottom_side=random.choice(["left", "right"]),
        )
        bottom = _profile_half(
            half_h,
            box_w,
            top_span=random.randint(1, box_w),
            bottom_span=random.randint(2, box_w - 1),
            top_side=random.choice(["left", "right"]),
            bottom_side=random.choice(["left", "right"]),
        )
        top = _jitter_half(top)
        bottom = _jitter_half(bottom)

        if not np.any(top[:, 0]):
            row = random.randrange(half_h)
            right = int(np.flatnonzero(top[row]).max())
            top[row, : right + 1] = True
        if not np.any(top[:, -1]):
            row = random.randrange(half_h)
            left = int(np.flatnonzero(top[row]).min())
            top[row, left:] = True
        if not np.any(bottom[:, 0]):
            row = random.randrange(half_h)
            right = int(np.flatnonzero(bottom[row]).max())
            bottom[row, : right + 1] = True
        if not np.any(bottom[:, -1]):
            row = random.randrange(half_h)
            left = int(np.flatnonzero(bottom[row]).min())
            bottom[row, left:] = True

        if not np.any(top[-1] & bottom[0]):
            candidates = [c for c in range(box_w) if top[-1, c] or bottom[0, c]]
            if not candidates:
                continue
            c = random.choice(candidates)
            top[-1, c] = True
            bottom[0, c] = True

        mask = np.vstack([top, bottom])
        comps = _component_coords(mask)
        if len(comps) != 1:
            continue
        if _bbox(comps[0]) != (0, 0, box_h - 1, box_w - 1):
            continue
        if len([c for c in np.flatnonzero(mask[0]) if 0 < c < box_w - 1]) == 0:
            continue
        return mask
    raise RuntimeError("failed to construct canonical object mask")


def _asymmetric_coloring(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    for _ in range(200):
        patch = np.zeros((h, w), dtype=int)
        for r, c in np.argwhere(mask):
            patch[r, c] = 1 + random.randint(0, 1)
        if not np.any(patch == 1) or not np.any(patch == 2):
            continue
        if all(not np.array_equal(patch, np.rot90(patch, k=k)) for k in (1, 2, 3)):
            return patch
    raise RuntimeError("failed to color object asymmetrically")


def _place_canonical_case(rows: int, cols: int, box_h: int, box_w: int, marker_color: int) -> np.ndarray:
    mask = _canonical_mask(box_h, box_w)
    patch = _asymmetric_coloring(mask)
    top_margin = 4
    side_margin = 4
    r0 = random.randint(top_margin, rows - box_h - 4)
    c0 = random.randint(side_margin, cols - box_w - 4)

    grid = np.zeros((rows, cols), dtype=int)
    rr, cc = np.argwhere(patch != BG).T
    grid[r0 + rr, c0 + cc] = patch[rr, cc]

    top_edge = np.flatnonzero(mask[0])
    marker_choices = [c for c in top_edge if 0 < c < box_w - 1]
    if not marker_choices:
        raise RuntimeError("no unambiguous marker position")
    mc = c0 + random.choice(marker_choices)
    grid[r0 - 1, mc] = marker_color
    return grid


def _transform_case(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "up":
        return grid
    if side == "down":
        return np.flipud(grid)
    if side == "left":
        return grid.T
    if side == "right":
        return np.fliplr(grid.T)
    raise ValueError(side)


def _semantic_ok(inp: np.ndarray, expected_side: str, expected_marker_color: int) -> bool:
    try:
        _, _box, _marker, marker_color, side = _identify_structure(inp)
        if side != expected_side or marker_color != expected_marker_color:
            return False
        out = main(inp)
    except Exception:
        return False

    if np.any(np.isin(out, list(MARKER_TO_TURNS))):
        return False

    comps = _component_coords(np.isin(out, OBJ_COLORS))
    if len(comps) != 2:
        return False

    comp_sets = [set(comp) for comp in comps]
    if _has_orth_contact(comp_sets[0], comp_sets[1]):
        return False
    if not _has_diagonal_contact(comp_sets[0], comp_sets[1]):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _attempt_case(side: str, marker_color: int) -> np.ndarray | None:
    rows = random.randint(14, 18)
    cols = random.randint(14, 18)
    box_h = random.choice((4, 8))
    box_w = random.choice((4, 6, 8))
    if side in ("left", "right"):
        rows, cols = cols, rows

    try:
        canonical = _place_canonical_case(rows, cols, box_h, box_w, marker_color)
    except Exception:
        return None
    grid = _transform_case(canonical, side)
    return grid if _semantic_ok(grid, side, marker_color) else None


def generate_input() -> np.ndarray:
    global _CASE_INDEX

    side, marker_color = CASE_ORDER[_CASE_INDEX % len(CASE_ORDER)]
    _CASE_INDEX += 1

    for _ in range(400):
        grid = _attempt_case(side, marker_color)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

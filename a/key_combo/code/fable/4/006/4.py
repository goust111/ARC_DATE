from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask, get_bg, in_bounds


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _border_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    coords: list[Coord] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        if r1 != r0:
            coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        if c1 != c0:
            coords.append((r, c1))
    return coords


def _corner_marker_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [
        (r0 - 1, c0),
        (r0, c0 - 1),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r1 + 1, c1),
        (r1, c1 + 1),
    ]


def _solid_rectangle_mask(g: np.ndarray, bg: int) -> np.ndarray:
    """Cells belonging to filled non-background rectangles of at least 2x2."""
    out = np.zeros(g.shape, dtype=bool)
    for color in sorted(set(map(int, g.ravel())) - {int(bg), int(Color.RED)}):
        for coords in components_mask(g == color, connectivity=4):
            box = bbox_coords(coords)
            r0, c0, r1, c1 = box
            h = bbox_h(box)
            w = bbox_w(box)
            if h < 2 or w < 2:
                continue
            if int(coords.shape[0]) != h * w:
                continue
            if np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
                out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _candidate_frames(g: np.ndarray, bg: int, occ_mask: np.ndarray) -> list[dict]:
    frame_mask = (g != bg) & (~occ_mask) & (g != Color.RED)
    combined = (g != bg) & (g != Color.RED)
    candidates: list[dict] = []
    seen: set[tuple[int, BBox]] = set()

    for comp in components_mask(combined, connectivity=8):
        comp_rows = comp[:, 0]
        comp_cols = comp[:, 1]
        colors = sorted(set(map(int, g[comp_rows, comp_cols])) - {int(bg), int(Color.RED)})
        for color in colors:
            pts = [(int(r), int(c)) for r, c in comp if frame_mask[int(r), int(c)] and int(g[int(r), int(c)]) == color]
            if len(pts) < 8:
                continue
            box = bbox_coords(np.array(pts, dtype=int))
            if (int(color), box) in seen:
                continue
            seen.add((int(color), box))
            if bbox_h(box) != bbox_w(box) or bbox_h(box) < 4:
                continue
            r0, c0, r1, c1 = box
            if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in pts):
                continue
            if not (
                np.any(g[r0, c0 : c1 + 1] == color)
                and np.any(g[r1, c0 : c1 + 1] == color)
                and np.any(g[r0 : r1 + 1, c0] == color)
                and np.any(g[r0 : r1 + 1, c1] == color)
            ):
                continue

            border = _border_coords(box)
            unexplained = [(r, c) for r, c in border if int(g[r, c]) != color and not bool(occ_mask[r, c])]
            candidates.append({"color": int(color), "bbox": box, "complete": not unexplained})

    candidates.sort(key=lambda x: (x["bbox"][0], x["bbox"][1], x["bbox"][2], x["bbox"][3], x["color"]))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark square frames whose missing border cells are fully explained by occluder rectangles, and erase true broken frames.

    Input structure:
    - A uniform background contains separated one-pixel square frame borders.
    - Filled rectangular blocks in another color may lie on top of frame borders and hide some border cells.
    - Other frame borders have exposed background gaps that are not covered by any filled rectangular block.

    Rule:
    - For each visible frame, use the visible frame pixels to recover its square border bbox.
    - If every non-frame cell on that bbox border is occupied by a solid occluder rectangle, keep the frame and draw red
      pixels at the two orthogonally outside cells beside each corner, only where those marker cells are background.
    - If any border gap is exposed background or another non-occluder cell, erase the frame-color cells on that square border.
    - Occluder rectangles and unrelated distractors remain unchanged.

    Key insight:
    - Completeness is counterfactual: a visibly broken border is still complete only when the hidden border cells are exactly
      the cells covered by occluder blocks; an equally small uncovered gap makes the whole residual frame invalid.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    occ_mask = _solid_rectangle_mask(g, bg)
    out = g.copy()

    candidates = _candidate_frames(g, bg, occ_mask)
    assert candidates, "expected at least one square-frame candidate"

    for cand in candidates:
        color = int(cand["color"])
        box = cand["bbox"]
        if bool(cand["complete"]):
            for r, c in _corner_marker_coords(box):
                if in_bounds(g.shape, r, c) and int(g[r, c]) == bg:
                    out[r, c] = Color.RED
        else:
            for r, c in _border_coords(box):
                if int(g[r, c]) == color:
                    out[r, c] = bg

    return out


# --------------- generator ---------------

_GEN_RETRIES = 300
_GEN_COLORS = [Color.BLACK, Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_GEN_TYPES_REQUIRED = ("occluded", "bare_defective", "mixed_defective")
_GEN_TYPES_EXTRA = ("occluded", "visible_complete", "bare_defective", "mixed_defective")


def _gen_draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    for r, c in _border_coords(box):
        grid[r, c] = color


def _gen_gap_segment(box: BBox, side: str, length: int) -> list[Coord]:
    r0, c0, r1, c1 = box
    if side == "top":
        start = random.randint(c0 + 1, c1 - length)
        return [(r0, c) for c in range(start, start + length)]
    if side == "bottom":
        start = random.randint(c0 + 1, c1 - length)
        return [(r1, c) for c in range(start, start + length)]
    if side == "left":
        start = random.randint(r0 + 1, r1 - length)
        return [(r, c0) for r in range(start, start + length)]
    start = random.randint(r0 + 1, r1 - length)
    return [(r, c1) for r in range(start, start + length)]


def _gen_occ_rect_for_segment(segment: list[Coord], side: str) -> BBox:
    rows = [r for r, _c in segment]
    cols = [c for _r, c in segment]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if side == "top":
        return r0 - 1, c0, r0, c1
    if side == "bottom":
        return r1, c0, r1 + 1, c1
    if side == "left":
        return r0, c0 - 1, r1, c0
    return r0, c1, r1, c1 + 1


def _gen_paint_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _gen_can_place(reserved: np.ndarray, box: BBox) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = box
    if r0 < 2 or c0 < 2 or r1 + 2 >= h or c1 + 2 >= w:
        return False
    return not bool(reserved[r0 - 2 : r1 + 3, c0 - 2 : c1 + 3].any())


def _gen_reserve(reserved: np.ndarray, box: BBox) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = box
    reserved[max(0, r0 - 2) : min(h, r1 + 3), max(0, c0 - 2) : min(w, c1 + 3)] = True


def _gen_place_frame(reserved: np.ndarray, side_len: int) -> BBox | None:
    h, w = reserved.shape
    choices = []
    for r0 in range(2, h - side_len - 1):
        for c0 in range(2, w - side_len - 1):
            box = (r0, c0, r0 + side_len - 1, c0 + side_len - 1)
            if _gen_can_place(reserved, box):
                choices.append(box)
    if not choices:
        return None
    box = random.choice(choices)
    _gen_reserve(reserved, box)
    return box


def _gen_apply_kind(grid: np.ndarray, box: BBox, kind: str, frame_color: int, occ_color: int) -> None:
    if kind == "visible_complete":
        return

    side = random.choice(["top", "bottom", "left", "right"])
    if kind == "occluded":
        segment = _gen_gap_segment(box, side, random.randint(2, min(3, bbox_h(box) - 2)))
        _gen_paint_rect(grid, _gen_occ_rect_for_segment(segment, side), occ_color)
        return

    if kind == "bare_defective":
        segment = _gen_gap_segment(box, side, random.randint(1, 2))
        for r, c in segment:
            if int(grid[r, c]) == frame_color:
                grid[r, c] = get_bg(grid)
        return

    segment = _gen_gap_segment(box, side, 3)
    occ_part = segment[:2]
    bare_part = segment[2:]
    _gen_paint_rect(grid, _gen_occ_rect_for_segment(occ_part, side), occ_color)
    for r, c in bare_part:
        grid[r, c] = get_bg(grid)


def _gen_marker_or_erase_expected(inp: np.ndarray, frames: list[dict], bg: int, frame_color: int, occ_color: int) -> np.ndarray | None:
    out = inp.copy()
    complete_count = 0
    defective_count = 0
    occluded_complete = 0
    mixed_defective = 0

    for frame in frames:
        box = frame["bbox"]
        border = _border_coords(box)
        border_values = [int(inp[r, c]) for r, c in border]
        missing = [p for p, value in zip(border, border_values) if value != frame_color]
        unexplained = [p for p, value in zip(border, border_values) if value != frame_color and value != occ_color]
        is_complete = not unexplained
        if is_complete:
            complete_count += 1
            if missing:
                occluded_complete += 1
            for r, c in _corner_marker_coords(box):
                if in_bounds(inp.shape, r, c) and int(inp[r, c]) == bg:
                    out[r, c] = Color.RED
        else:
            defective_count += 1
            if any(value == occ_color for value in border_values) and any(value == bg for value in border_values):
                mixed_defective += 1
            for r, c in border:
                if int(inp[r, c]) == frame_color:
                    out[r, c] = bg

    if complete_count < 1 or defective_count < 1 or occluded_complete < 1 or mixed_defective < 1:
        return None
    if not np.any(out == Color.RED):
        return None
    return out


def _gen_place_distractor_rect(grid: np.ndarray, reserved: np.ndarray, color: int) -> None:
    h, w = grid.shape
    for _ in range(80):
        dh = random.randint(2, 4)
        dw = random.randint(2, 5)
        r0 = random.randint(1, h - dh - 1)
        c0 = random.randint(1, w - dw - 1)
        r1, c1 = r0 + dh - 1, c0 + dw - 1
        if not reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            _gen_paint_rect(grid, (r0, c0, r1, c1), color)
            reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            return


def _gen_place_frame_distractor(grid: np.ndarray, reserved: np.ndarray, color: int) -> None:
    h, w = grid.shape
    for _ in range(80):
        length = random.randint(2, 4)
        horizontal = random.random() < 0.5
        dh, dw = (1, length) if horizontal else (length, 1)
        r0 = random.randint(1, h - dh - 1)
        c0 = random.randint(1, w - dw - 1)
        r1, c1 = r0 + dh - 1, c0 + dw - 1
        if not reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
            return


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    bg, frame_color, occ_color = random.sample(_GEN_COLORS, 3)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    n_frames = random.randint(3, 5)
    kinds = list(_GEN_TYPES_REQUIRED)
    while len(kinds) < n_frames:
        kinds.append(random.choice(_GEN_TYPES_EXTRA))
    random.shuffle(kinds)

    frames: list[dict] = []
    for kind in kinds:
        side_len = random.randint(5, 7)
        box = _gen_place_frame(reserved, side_len)
        if box is None:
            return None
        _gen_draw_frame(grid, box, frame_color)
        frames.append({"bbox": box, "kind": kind})

    for frame, kind in zip(frames, kinds):
        _gen_apply_kind(grid, frame["bbox"], kind, frame_color, occ_color)

    for _ in range(random.randint(0, 2)):
        _gen_place_distractor_rect(grid, reserved, occ_color)
    for _ in range(random.randint(0, 2)):
        _gen_place_frame_distractor(grid, reserved, frame_color)

    expected = _gen_marker_or_erase_expected(grid, frames, bg, frame_color, occ_color)
    if expected is None:
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.any((grid == occ_color) & (solved != occ_color)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-square-frame input")

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
FRAME = Color.GREY
SHELF = Color.ORANGE

DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _toggle(color: int) -> int:
    return BLUE if int(color) == RED else RED


def _paint_if_blank(out: np.ndarray, r: int, c: int, color: int) -> None:
    if int(out[r, c]) == BG:
        out[r, c] = int(color)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    out = []
    for comp in frames:
        r0, c0, r1, c1 = comp["bbox"]
        oh = int(r1 - r0 + 1)
        ow = int(c1 - c0 + 1)
        assert 4 <= oh <= 6 and 4 <= ow <= 6, "expected 1-cell rectangular frames with 2..4 interiors"

        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        border = _border_cells(r0, c0, r1, c1)
        gaps = sorted(border - coords)
        assert len(gaps) == 1, "each frame must have exactly one missing border pixel"
        gap_r, gap_c = gaps[0]

        if gap_r == r0:
            assert c0 < gap_c < c1
            side = "top"
        elif gap_r == r1:
            assert c0 < gap_c < c1
            side = "bottom"
        elif gap_c == c0:
            assert r0 < gap_r < r1
            side = "left"
        else:
            assert gap_c == c1 and r0 < gap_r < r1
            side = "right"

        out.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": (int(gap_r), int(gap_c)),
                "side": side,
                "dir": DIRS[side],
            }
        )
    out.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    for idx, frame in enumerate(out):
        frame["id"] = idx
    return out


def _shelf_segments(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    rows: list[list[tuple[int, int]]] = [[] for _ in range(h)]
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != SHELF:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == SHELF:
                c += 1
            rows[r].append((int(c0), int(c)))
            c += 1
    return rows


def _segment_covering(segments: list[tuple[int, int]], col: int) -> tuple[int, int] | None:
    for a, b in segments:
        if int(a) <= int(col) <= int(b):
            return (int(a), int(b))
    return None


def _fill_frame_interior(out: np.ndarray, frame: dict, color: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate a colored fill spilling out of one-gap frames, falling under gravity, and cascading across shelves.

    Input structure:
    - The grid is a black background with several grey rectangular border frames and orange horizontal shelves.
    - Every frame is a 1-cell-thick rectangle with exactly one non-corner border pixel missing as a gap.
    - Exactly one top frame has its whole interior pre-filled red; other frames have empty interiors.

    Rule:
    - Preserve every grey frame border and every orange shelf cell.
    - Start from the pre-filled top frame with red as the current fill color.
    - When a frame activates, fill its interior, paint its gap, extend a straight ray outward through the gap until the next step would hit structure or leave the grid, then switch to vertical gravity.
    - A falling branch paints background cells while descending, spreads across the full row directly above any shelf it lands on, then restarts downward from one cell beyond each shelf end.
    - If a falling branch reaches another frame's gap, toggle the fill color between red and blue, fill that frame, and continue from its own gap; earlier branches keep ownership of any shared background pixels.

    Key insight:
    - Each gap only chooses the ray's launch direction; after that the process is a gravity-driven cascade whose shelves branch the flow and whose newly entered frames alternate the active color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    assert 3 <= len(frames) <= 5, "expected 3..5 frames"

    start_ids = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size > 0 and np.all(interior == RED):
            start_ids.append(int(frame["id"]))
        else:
            assert np.all(interior == BG), "non-start frames must have empty interiors"
    assert len(start_ids) == 1, "expected exactly one pre-filled source frame"

    shelves = _shelf_segments(g)
    gap_to_frame = {tuple(frame["gap"]): int(frame["id"]) for frame in frames}

    out = g.copy()
    activated: set[int] = set()
    seen_falls: set[tuple[int, int, int, int]] = set()
    queue = deque([("frame", start_ids[0], RED)])

    def enqueue_fall(r: int, c: int, color: int, include_start: bool) -> None:
        if 0 <= int(r) < g.shape[0] and 0 <= int(c) < g.shape[1]:
            queue.append(("fall", int(r), int(c), int(color), int(include_start)))

    while queue:
        item = queue.popleft()
        kind = item[0]

        if kind == "frame":
            _, frame_id, color = item
            if int(frame_id) in activated:
                continue
            frame = frames[int(frame_id)]
            activated.add(int(frame_id))
            _fill_frame_interior(out, frame, int(color))

            r, c = map(int, frame["gap"])
            dr, dc = frame["dir"]
            out[r, c] = int(color)
            end_r, end_c = r, c
            while True:
                nr = int(end_r + dr)
                nc = int(end_c + dc)
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    break
                if int(g[nr, nc]) != BG:
                    break
                end_r, end_c = nr, nc
                _paint_if_blank(out, end_r, end_c, int(color))
            enqueue_fall(end_r, end_c, int(color), False)
            continue

        _, cur_r, cur_c, color, include_start_int = item
        state = (int(cur_r), int(cur_c), int(color), int(include_start_int))
        if state in seen_falls:
            continue
        seen_falls.add(state)

        if include_start_int:
            if int(g[cur_r, cur_c]) == BG:
                _paint_if_blank(out, int(cur_r), int(cur_c), int(color))

        r = int(cur_r)
        c = int(cur_c)
        while True:
            nr = int(r + 1)
            if nr >= g.shape[0]:
                break

            if (nr, c) in gap_to_frame:
                target = int(gap_to_frame[(nr, c)])
                if target not in activated:
                    queue.append(("frame", target, _toggle(int(color))))
                break

            below = int(g[nr, c])
            if below == BG:
                r = nr
                _paint_if_blank(out, r, c, int(color))
                continue

            if below == SHELF:
                seg = _segment_covering(shelves[nr], c)
                if seg is None:
                    break
                a, b = seg
                for x in range(a, b + 1):
                    if int(g[r, x]) == BG:
                        _paint_if_blank(out, r, x, int(color))
                enqueue_fall(r, a - 1, int(color), True)
                enqueue_fall(r, b + 1, int(color), True)
                break

            break

    return out


def _expanded_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_is_free(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    rr0, cc0, rr1, cc1 = _expanded_bbox(r0, c0, r1, c1, occ.shape[0], occ.shape[1], pad)
    return not bool(occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_bbox(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0, cc0, rr1, cc1 = _expanded_bbox(r0, c0, r1, c1, occ.shape[0], occ.shape[1], pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_frame(grid: np.ndarray, occ: np.ndarray, frame: dict, fill_red: bool) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    gap = tuple(frame["gap"])
    for r, c in _border_cells(r0, c0, r1, c1):
        if (r, c) != gap:
            grid[r, c] = FRAME
    if fill_red:
        grid[r0 + 1 : r1, c0 + 1 : c1] = RED
    _reserve_bbox(occ, r0, c0, r1, c1, pad=1)


def _place_frame_bbox(
    grid: np.ndarray,
    occ: np.ndarray,
    r0: int,
    c0: int,
    oh: int,
    ow: int,
    gap_side: str,
    gap_offset: int,
    fill_red: bool,
) -> dict | None:
    r1 = r0 + oh - 1
    c1 = c0 + ow - 1
    if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        return None
    if not _bbox_is_free(occ, r0, c0, r1, c1, pad=1):
        return None
    if gap_side in {"top", "bottom"}:
        if not (1 <= gap_offset <= ow - 2):
            return None
        gap = (r0 if gap_side == "top" else r1, c0 + gap_offset)
    else:
        if not (1 <= gap_offset <= oh - 2):
            return None
        gap = (r0 + gap_offset, c0 if gap_side == "left" else c1)
    frame = {"bbox": (int(r0), int(c0), int(r1), int(c1)), "gap": gap, "side": gap_side}
    _draw_frame(grid, occ, frame, fill_red=fill_red)
    return frame


def _draw_shelf(grid: np.ndarray, occ: np.ndarray, row: int, a: int, b: int) -> None:
    grid[row, a : b + 1] = SHELF
    _reserve_bbox(occ, row, a, row, b, pad=1)


def _line_is_clear(grid: np.ndarray, fixed: int, start: int, end: int, axis: str) -> bool:
    if start > end:
        return True
    if axis == "col":
        return bool(np.all(grid[start : end + 1, fixed] == BG))
    return bool(np.all(grid[fixed, start : end + 1] == BG))


def _place_frame_from_gap(
    grid: np.ndarray,
    occ: np.ndarray,
    gap_side: str,
    gap_r: int,
    gap_c: int,
    oh: int,
    ow: int,
    fill_red: bool,
) -> dict | None:
    if gap_side == "top":
        offset = random.randint(1, ow - 2)
        r0 = gap_r
        c0 = gap_c - offset
    elif gap_side == "bottom":
        offset = random.randint(1, ow - 2)
        r0 = gap_r - oh + 1
        c0 = gap_c - offset
    elif gap_side == "left":
        offset = random.randint(1, oh - 2)
        r0 = gap_r - offset
        c0 = gap_c
    else:
        assert gap_side == "right"
        offset = random.randint(1, oh - 2)
        r0 = gap_r - offset
        c0 = gap_c - ow + 1

    r1 = r0 + oh - 1
    c1 = c0 + ow - 1
    if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
        return None
    if not _bbox_is_free(occ, r0, c0, r1, c1, pad=1):
        return None

    frame = {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gap_r), int(gap_c)),
        "side": gap_side,
    }
    _draw_frame(grid, occ, frame, fill_red=fill_red)
    return frame


def _outgoing_anchor(frame: dict, shape: tuple[int, int]) -> tuple[int, int]:
    gap_r, gap_c = map(int, frame["gap"])
    side = frame["side"]
    h, w = shape
    if side == "left":
        return gap_r, 0
    if side == "right":
        return gap_r, w - 1
    if side == "top":
        return 0, gap_c
    assert side == "bottom"
    return gap_r, gap_c


def _place_split_shelf(
    grid: np.ndarray,
    occ: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    row_lo: int,
    row_hi: int,
) -> tuple[int, int, int] | None:
    row_lo = max(int(anchor_r) + 3, int(row_lo))
    row_hi = min(int(row_hi), grid.shape[0] - 4)
    if row_lo > row_hi:
        return None
    for _ in range(60):
        row = random.randint(row_lo, row_hi)
        if not _line_is_clear(grid, anchor_c, anchor_r + 1, row - 1, "col"):
            continue
        length = random.randint(3, 6)
        left_max = min(anchor_c, grid.shape[1] - length - 2)
        left_min = max(2, anchor_c - length + 1)
        if left_min > left_max:
            continue
        a = random.randint(left_min, left_max)
        b = a + length - 1
        if not (a <= anchor_c <= b):
            continue
        if a - 1 < 0 or b + 1 >= grid.shape[1]:
            continue
        if not _bbox_is_free(occ, row, a, row, b, pad=1):
            continue
        _draw_shelf(grid, occ, row, a, b)
        return (int(row), int(a), int(b))
    return None


def _place_edge_shelf(
    grid: np.ndarray,
    occ: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    side: str,
    row_lo: int,
    row_hi: int,
) -> tuple[int, int, int] | None:
    row_lo = max(int(anchor_r) + 3, int(row_lo))
    row_hi = min(int(row_hi), grid.shape[0] - 4)
    if row_lo > row_hi:
        return None
    assert side in {"left", "right"}
    for _ in range(60):
        row = random.randint(row_lo, row_hi)
        if not _line_is_clear(grid, anchor_c, anchor_r + 1, row - 1, "col"):
            continue
        length = random.randint(3, 6)
        if side == "left":
            a, b = 0, length - 1
        else:
            a, b = grid.shape[1] - length, grid.shape[1] - 1
        if not _bbox_is_free(occ, row, a, row, b, pad=1):
            continue
        _draw_shelf(grid, occ, row, a, b)
        return (int(row), int(a), int(b))
    return None


def _place_specific_shelf(
    grid: np.ndarray,
    occ: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    row: int,
    a: int,
    b: int,
) -> tuple[int, int, int] | None:
    if not (0 <= a <= b < grid.shape[1]):
        return None
    if not (a <= anchor_c <= b):
        return None
    if a - 1 < 0 or b + 1 >= grid.shape[1]:
        return None
    if row <= anchor_r + 2 or row >= grid.shape[0] - 3:
        return None
    if not _line_is_clear(grid, anchor_c, anchor_r + 1, row - 1, "col"):
        return None
    if not _bbox_is_free(occ, row, a, row, b, pad=1):
        return None
    _draw_shelf(grid, occ, row, a, b)
    return (int(row), int(a), int(b))


def _place_branch_frame(
    grid: np.ndarray,
    occ: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    preferred_side: str,
    internal: bool,
) -> dict | None:
    options = [preferred_side] if internal else [preferred_side, "top"]
    random.shuffle(options)
    for gap_side in options:
        for _ in range(80):
            oh = random.randint(4, 6)
            ow = random.randint(4, 6)
            if gap_side == "top":
                lo = anchor_r + 3
                hi = grid.shape[0] - oh - 2
                if lo > hi:
                    continue
                gap_r = random.randint(lo, hi)
                gap_c = anchor_c
                if not _line_is_clear(grid, anchor_c, anchor_r + 1, gap_r - 1, "col"):
                    continue
                if not _line_is_clear(grid, gap_c, 0, gap_r - 1, "col"):
                    continue
            elif gap_side == "left":
                lo = anchor_r + 3
                hi = grid.shape[0] - 3
                if lo > hi:
                    continue
                gap_r = random.randint(lo, hi)
                gap_c = anchor_c
                if not _line_is_clear(grid, anchor_c, anchor_r + 1, gap_r - 1, "col"):
                    continue
                if not _line_is_clear(grid, gap_r, 0, gap_c - 1, "row"):
                    continue
            else:
                assert gap_side == "right"
                lo = anchor_r + 3
                hi = grid.shape[0] - 3
                if lo > hi:
                    continue
                gap_r = random.randint(lo, hi)
                gap_c = anchor_c
                if not _line_is_clear(grid, anchor_c, anchor_r + 1, gap_r - 1, "col"):
                    continue
                if not _line_is_clear(grid, gap_r, gap_c + 1, grid.shape[1] - 1, "row"):
                    continue
            frame = _place_frame_from_gap(
                grid,
                occ,
                gap_side=gap_side,
                gap_r=int(gap_r),
                gap_c=int(gap_c),
                oh=int(oh),
                ow=int(ow),
                fill_red=False,
            )
            if frame is not None:
                return frame
    return None


def _validate_generated(grid: np.ndarray, frames: list[dict]) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    blue_frames = 0
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            return False
        colors = set(map(int, np.unique(interior)))
        if len(colors) != 1:
            return False
        color = next(iter(colors))
        if idx == 0 and color != RED:
            return False
        if idx > 0:
            if color == BLUE:
                blue_frames += 1
            elif color != BG:
                return False
    return bool((out == BLUE).any()) and bool((out == RED).any()) and blue_frames >= 1


def _place_leaf_top_frame(grid: np.ndarray, occ: np.ndarray, anchor_r: int, anchor_c: int) -> dict | None:
    for _ in range(120):
        oh = random.randint(4, 6)
        ow = random.randint(4, 6)
        lo = anchor_r + 3
        hi = grid.shape[0] - oh - 2
        if lo > hi:
            return None
        gap_r = random.randint(lo, hi)
        gap_c = anchor_c
        if not _line_is_clear(grid, anchor_c, anchor_r + 1, gap_r - 1, "col"):
            continue
        if not _line_is_clear(grid, gap_c, 0, gap_r - 1, "col"):
            continue
        frame = _place_frame_from_gap(
            grid,
            occ,
            gap_side="top",
            gap_r=int(gap_r),
            gap_c=int(gap_c),
            oh=int(oh),
            ow=int(ow),
            fill_red=False,
        )
        if frame is not None:
            return frame
    return None


def _place_top_bottom_frame(grid: np.ndarray, occ: np.ndarray) -> dict | None:
    w = grid.shape[1]
    top = None
    for _ in range(80):
        oh = random.randint(4, 6)
        ow = random.randint(4, 6)
        r0 = random.randint(1, 3)
        c0 = random.randint(5, w - ow - 6)
        gap_c = c0 + random.randint(1, ow - 2)
        gap_r = r0 + oh - 1
        top = _place_frame_from_gap(grid, occ, "bottom", gap_r, gap_c, oh, ow, fill_red=True)
        if top is not None:
            break
    return top


def _terminal_anchors_from_shelf(row: int, a: int, b: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (row - 1, a - 1), (row - 1, b + 1)


def _gen_down_split2(grid: np.ndarray, occ: np.ndarray) -> list[dict] | None:
    top = _place_top_bottom_frame(grid, occ)
    if top is None:
        return None

    gap_c = int(top["gap"][1])
    length = random.randint(4, 5)
    row = int(top["bbox"][2]) + random.randint(4, 5)
    a = max(4, min(gap_c - 2, grid.shape[1] - length - 5))
    shelf = _place_specific_shelf(grid, occ, top["gap"][0], gap_c, row, a, a + length - 1)
    if shelf is None:
        return None
    left_anchor, right_anchor = _terminal_anchors_from_shelf(*shelf)

    left = _place_leaf_top_frame(grid, occ, left_anchor[0], left_anchor[1])
    right = _place_leaf_top_frame(grid, occ, right_anchor[0], right_anchor[1])
    if left is None or right is None:
        return None
    return [top, left, right]


def _gen_down_split_chain(grid: np.ndarray, occ: np.ndarray) -> list[dict] | None:
    h, w = grid.shape
    oh = random.randint(4, 5)
    ow = random.randint(5, 6)
    gap_c = w // 2 + random.randint(-1, 1)
    c0 = gap_c - 2
    top = _place_frame_bbox(grid, occ, 2, c0, oh, ow, "bottom", 2, fill_red=True)
    if top is None:
        return None

    shelf1 = _place_specific_shelf(grid, occ, top["gap"][0], top["gap"][1], top["bbox"][2] + 4, gap_c - 2, gap_c + 1)
    if shelf1 is None:
        return None
    left_anchor, right_anchor = _terminal_anchors_from_shelf(*shelf1)
    direct_leaf = _place_leaf_top_frame(grid, occ, left_anchor[0], left_anchor[1])
    if direct_leaf is None:
        return None

    shelf2 = _place_specific_shelf(
        grid,
        occ,
        right_anchor[0],
        right_anchor[1],
        shelf1[0] + 5,
        right_anchor[1] - 1,
        right_anchor[1] + 2,
    )
    if shelf2 is None:
        return None
    a2, b2 = _terminal_anchors_from_shelf(*shelf2)
    leaf2 = _place_leaf_top_frame(grid, occ, a2[0], a2[1])
    leaf3 = _place_leaf_top_frame(grid, occ, b2[0], b2[1])
    if leaf2 is None or leaf3 is None:
        return None
    return [top, direct_leaf, leaf2, leaf3]


def _gen_down_split_double_chain(grid: np.ndarray, occ: np.ndarray) -> list[dict] | None:
    h, w = grid.shape
    oh = random.randint(4, 5)
    ow = random.randint(5, 6)
    gap_c = w // 2 + random.randint(-1, 1)
    c0 = gap_c - 2
    top = _place_frame_bbox(grid, occ, 2, c0, oh, ow, "bottom", 2, fill_red=True)
    if top is None:
        return None

    shelf1 = _place_specific_shelf(grid, occ, top["gap"][0], top["gap"][1], top["bbox"][2] + 4, gap_c - 2, gap_c + 1)
    if shelf1 is None:
        return None
    left_anchor, right_anchor = _terminal_anchors_from_shelf(*shelf1)
    left_shelf = _place_specific_shelf(grid, occ, left_anchor[0], left_anchor[1], shelf1[0] + 5, left_anchor[1] - 1, left_anchor[1] + 2)
    right_shelf = _place_specific_shelf(grid, occ, right_anchor[0], right_anchor[1], shelf1[0] + 5, right_anchor[1] - 2, right_anchor[1] + 1)
    if left_shelf is None or right_shelf is None:
        return None

    leaves = []
    for anchor in _terminal_anchors_from_shelf(*left_shelf) + _terminal_anchors_from_shelf(*right_shelf):
        leaf = _place_leaf_top_frame(grid, occ, anchor[0], anchor[1])
        if leaf is None:
            return None
        leaves.append(leaf)
    return [top] + leaves


def _gen_edge_chain3(grid: np.ndarray, occ: np.ndarray) -> list[dict] | None:
    h, w = grid.shape
    side = random.choice(["left", "right"])
    oh = random.randint(4, 5)
    ow = random.randint(5, 6)
    r0 = 2
    if side == "left":
        c0 = w // 2 - 4
        top = _place_frame_bbox(grid, occ, r0, c0, oh, ow, "left", 2, fill_red=True)
        if top is None:
            return None
        shelf1 = _place_specific_shelf(grid, occ, top["gap"][0], 0, top["bbox"][2] + 4, 0, 3)
    else:
        c0 = w // 2 - 1
        top = _place_frame_bbox(grid, occ, r0, c0, oh, ow, "right", 2, fill_red=True)
        if top is None:
            return None
        shelf1 = _place_specific_shelf(grid, occ, top["gap"][0], w - 1, top["bbox"][2] + 4, w - 4, w - 1)
    if shelf1 is None:
        return None

    if side == "left":
        inner_anchor = (shelf1[0] - 1, shelf1[2] + 1)
        shelf2 = _place_specific_shelf(grid, occ, inner_anchor[0], inner_anchor[1], shelf1[0] + 5, inner_anchor[1] - 1, inner_anchor[1] + 2)
    else:
        inner_anchor = (shelf1[0] - 1, shelf1[1] - 1)
        shelf2 = _place_specific_shelf(grid, occ, inner_anchor[0], inner_anchor[1], shelf1[0] + 5, inner_anchor[1] - 2, inner_anchor[1] + 1)
    if shelf2 is None:
        return None

    leaf1_anchor, leaf2_anchor = _terminal_anchors_from_shelf(*shelf2)
    leaf1 = _place_leaf_top_frame(grid, occ, leaf1_anchor[0], leaf1_anchor[1])
    leaf2 = _place_leaf_top_frame(grid, occ, leaf2_anchor[0], leaf2_anchor[1])
    if leaf1 is None or leaf2 is None:
        return None
    return [top, leaf1, leaf2]


def _gen_edge_single_active(grid: np.ndarray, occ: np.ndarray) -> list[dict] | None:
    h, w = grid.shape
    side = random.choice(["left", "right"])
    oh = random.randint(4, 5)
    ow = random.randint(5, 6)
    r0 = 2
    if side == "left":
        c0 = w // 2 - 4
        top = _place_frame_bbox(grid, occ, r0, c0, oh, ow, "left", 2, fill_red=True)
        if top is None:
            return None
        shelf = _place_specific_shelf(grid, occ, top["gap"][0], 0, top["bbox"][2] + 4, 0, 3)
        if shelf is None:
            return None
        leaf_anchor = (shelf[0] - 1, shelf[2] + 1)
    else:
        c0 = w // 2 - 1
        top = _place_frame_bbox(grid, occ, r0, c0, oh, ow, "right", 2, fill_red=True)
        if top is None:
            return None
        shelf = _place_specific_shelf(grid, occ, top["gap"][0], w - 1, top["bbox"][2] + 4, w - 4, w - 1)
        if shelf is None:
            return None
        leaf_anchor = (shelf[0] - 1, shelf[1] - 1)

    leaf = _place_leaf_top_frame(grid, occ, leaf_anchor[0], leaf_anchor[1])
    if leaf is None:
        return None
    return [top, leaf]


def _place_distractor_frame(grid: np.ndarray, occ: np.ndarray) -> dict | None:
    h, w = grid.shape
    for _ in range(150):
        oh = random.randint(4, 6)
        ow = random.randint(4, 6)
        r0 = random.randint(max(6, h // 2 - 2), h - oh - 2)
        c0 = random.randint(1, w - ow - 2)
        side = random.choice(["top", "bottom", "left", "right"])
        offset = random.randint(1, (ow - 2) if side in {"top", "bottom"} else (oh - 2))
        frame = _place_frame_bbox(grid, occ, r0, c0, oh, ow, side, offset, fill_red=False)
        if frame is not None:
            return frame
    return None


def _gen_attempt() -> np.ndarray | None:
    patterns = [_gen_down_split2, _gen_edge_single_active]
    weights = [4, 1]
    builder = random.choices(patterns, weights=weights, k=1)[0]
    if builder is _gen_edge_single_active:
        h = random.randint(24, 29)
        w = random.randint(22, 29)
    else:
        h = random.randint(22, 29)
        w = random.randint(22, 29)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    frames = builder(grid, occ)
    if frames is None:
        return None
    target_frames = random.randint(3, 5)
    while len(frames) < target_frames:
        frame = _place_distractor_frame(grid, occ)
        if frame is None:
            return None
        frames.append(frame)
    if not _validate_generated(grid, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascading-gap input")

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NON_RECT_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]

GEN_MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _true_runs_1d(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(arr.tolist()):
        if bool(value) and start is None:
            start = i
        if start is not None and (not bool(value) or i == len(arr) - 1):
            end = i if bool(value) else i - 1
            runs.append((start, end))
            start = None
    return runs


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    mask = grid == int(color)
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                crop[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": crop,
                }
            )
    return comps


def _detect_notches(
    grid: np.ndarray, rect_bbox: tuple[int, int, int, int], bg: int, rect_color: int
) -> list[dict]:
    r0, c0, r1, c1 = rect_bbox
    notches: list[dict] = []

    c = c0 + 1
    while c <= c1 - 1:
        if grid[r0, c] == bg and grid[r0 + 1, c] == rect_color:
            start = c
            while c <= c1 - 1 and grid[r0, c] == bg and grid[r0 + 1, c] == rect_color:
                c += 1
            end = c - 1
            notches.append(
                {
                    "side": "top",
                    "width": end - start + 1,
                    "axis_start": start,
                    "axis_end": end,
                    "cells": {(r0, cc) for cc in range(start, end + 1)},
                }
            )
        else:
            c += 1

    c = c0 + 1
    while c <= c1 - 1:
        if grid[r1, c] == bg and grid[r1 - 1, c] == rect_color:
            start = c
            while c <= c1 - 1 and grid[r1, c] == bg and grid[r1 - 1, c] == rect_color:
                c += 1
            end = c - 1
            notches.append(
                {
                    "side": "bottom",
                    "width": end - start + 1,
                    "axis_start": start,
                    "axis_end": end,
                    "cells": {(r1, cc) for cc in range(start, end + 1)},
                }
            )
        else:
            c += 1

    r = r0 + 1
    while r <= r1 - 1:
        if grid[r, c0] == bg and grid[r, c0 + 1] == rect_color:
            start = r
            while r <= r1 - 1 and grid[r, c0] == bg and grid[r, c0 + 1] == rect_color:
                r += 1
            end = r - 1
            notches.append(
                {
                    "side": "left",
                    "width": end - start + 1,
                    "axis_start": start,
                    "axis_end": end,
                    "cells": {(rr, c0) for rr in range(start, end + 1)},
                }
            )
        else:
            r += 1

    r = r0 + 1
    while r <= r1 - 1:
        if grid[r, c1] == bg and grid[r, c1 - 1] == rect_color:
            start = r
            while r <= r1 - 1 and grid[r, c1] == bg and grid[r, c1 - 1] == rect_color:
                r += 1
            end = r - 1
            notches.append(
                {
                    "side": "right",
                    "width": end - start + 1,
                    "axis_start": start,
                    "axis_end": end,
                    "cells": {(rr, c1) for rr in range(start, end + 1)},
                }
            )
        else:
            r += 1

    return notches


def _parse_piece(mask: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict:
    h, w = mask.shape
    candidates: list[dict] = []

    if h >= 3 and np.all(mask[1:, :]):
        runs = _true_runs_1d(mask[0, :])
        if len(runs) == 1:
            start, end = runs[0]
            width = end - start + 1
            if 1 <= width < w:
                candidates.append(
                    {
                        "side": "top",
                        "tab_width": width,
                        "tab_offset": start,
                        "body_h": h - 1,
                        "body_w": w,
                    }
                )

    if h >= 3 and np.all(mask[:-1, :]):
        runs = _true_runs_1d(mask[-1, :])
        if len(runs) == 1:
            start, end = runs[0]
            width = end - start + 1
            if 1 <= width < w:
                candidates.append(
                    {
                        "side": "bottom",
                        "tab_width": width,
                        "tab_offset": start,
                        "body_h": h - 1,
                        "body_w": w,
                    }
                )

    if w >= 3 and np.all(mask[:, 1:]):
        runs = _true_runs_1d(mask[:, 0])
        if len(runs) == 1:
            start, end = runs[0]
            width = end - start + 1
            if 1 <= width < h:
                candidates.append(
                    {
                        "side": "left",
                        "tab_width": width,
                        "tab_offset": start,
                        "body_h": h,
                        "body_w": w - 1,
                    }
                )

    if w >= 3 and np.all(mask[:, :-1]):
        runs = _true_runs_1d(mask[:, -1])
        if len(runs) == 1:
            start, end = runs[0]
            width = end - start + 1
            if 1 <= width < h:
                candidates.append(
                    {
                        "side": "right",
                        "tab_width": width,
                        "tab_offset": start,
                        "body_h": h,
                        "body_w": w - 1,
                    }
                )

    assert len(candidates) == 1, "each piece must be a solid rectangle plus one 1-deep tab"
    piece = candidates[0]
    piece["color"] = int(color)
    piece["coords"] = coords
    piece["mask"] = mask
    piece["shape"] = mask.shape
    piece["cells"] = {(r, c) for r in range(h) for c in range(w) if mask[r, c]}
    return piece


def _piece_center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _notch_center(notch: dict) -> tuple[float, float]:
    if notch["side"] in ("top", "bottom"):
        row = min(r for r, _ in notch["cells"])
        return float(row), (notch["axis_start"] + notch["axis_end"]) / 2.0
    col = min(c for _, c in notch["cells"])
    return (notch["axis_start"] + notch["axis_end"]) / 2.0, float(col)


def _piece_notch_distance(bbox: tuple[int, int, int, int], notch: dict) -> float:
    pr, pc = _piece_center_from_bbox(bbox)
    nr, nc = _notch_center(notch)
    return abs(pr - nr) + abs(pc - nc)


def _shares_target_axis(bbox: tuple[int, int, int, int], notch: dict) -> bool:
    r0, c0, r1, c1 = bbox
    if notch["side"] in ("top", "bottom"):
        notch_row = min(r for r, _ in notch["cells"])
        return r0 <= notch_row <= r1 or not (c1 < notch["axis_start"] or notch["axis_end"] < c0)
    notch_col = min(c for _, c in notch["cells"])
    return c0 <= notch_col <= c1 or not (r1 < notch["axis_start"] or notch["axis_end"] < r0)


def _overlaps_notch_axis(bbox: tuple[int, int, int, int], notch: dict) -> bool:
    r0, c0, r1, c1 = bbox
    if notch["side"] in ("top", "bottom"):
        return not (c1 < notch["axis_start"] or notch["axis_end"] < c0)
    return not (r1 < notch["axis_start"] or notch["axis_end"] < r0)


def _final_top_left(
    piece: dict, notch: dict, rect_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect_bbox
    assert piece["side"] == notch["side"], "piece orientation must match the slot side"
    if piece["side"] == "top":
        return r0, notch["axis_start"] - piece["tab_offset"]
    if piece["side"] == "bottom":
        return r1 - piece["shape"][0] + 1, notch["axis_start"] - piece["tab_offset"]
    if piece["side"] == "left":
        return notch["axis_start"] - piece["tab_offset"], c0
    return notch["axis_start"] - piece["tab_offset"], c1 - piece["shape"][1] + 1


def _absolute_cells_from_top_left(
    cells: set[tuple[int, int]], top: int, left: int
) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in cells}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each tabbed piece to the equally wide notch and sink the whole piece into the grey rectangle.

    Input structure:
    - A black background contains one grey solid rectangle whose boundary has one-cell-deep notches of distinct widths.
    - Several colored monochrome pieces are scattered outside the rectangle; each piece is a solid block with one same-width, one-cell-deep tab.
    - Exactly one piece has a tab width that matches no notch.

    Rule:
    - Detect the unique width match between each movable piece and one notch on the same side orientation.
    - Translate that whole piece so its tab fills the notch, then keep it pushed inward until the piece is fully inside the rectangle and flush with that edge.
    - Recolor every grey cell covered by the moved piece to the piece color, turn the piece's old location back to background, and leave the unmatched decoy where it was.

    Key insight:
    - The notch is only the keyhole: width decides the destination, and the final colored footprint depends on the full piece body that sinks inside, not just on the notch itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    grey_components = _components_for_color(g, GREY)
    assert len(grey_components) == 1, "expected one grey host rectangle"
    rect = grey_components[0]
    rect_bbox = rect["bbox"]

    notches = _detect_notches(g, rect_bbox, bg, GREY)
    assert len(notches) >= 2, "expected multiple notches"
    notch_by_width = {notch["width"]: notch for notch in notches}
    assert len(notch_by_width) == len(notches), "notch widths must be unique"

    colors = sorted(int(v) for v in np.unique(g) if v not in (bg, GREY))
    pieces: list[dict] = []
    for color in colors:
        comps = _components_for_color(g, color)
        assert len(comps) == 1, "each piece color should form one component"
        comp = comps[0]
        pieces.append(_parse_piece(comp["mask"], color, comp["coords"]))

    matched = [piece for piece in pieces if piece["tab_width"] in notch_by_width]
    decoys = [piece for piece in pieces if piece["tab_width"] not in notch_by_width]
    assert len(decoys) == 1, "expected exactly one unmatched decoy piece"
    assert len(matched) == len(notches), "every notch should have exactly one matching piece"
    assert len({piece["tab_width"] for piece in matched}) == len(matched)

    out = g.copy()
    final_occupied: set[tuple[int, int]] = set()
    rr0, cc0, rr1, cc1 = rect_bbox

    for piece in matched:
        for r, c in piece["coords"]:
            out[r, c] = bg

    for piece in matched:
        notch = notch_by_width[piece["tab_width"]]
        top, left = _final_top_left(piece, notch, rect_bbox)
        final_cells = _absolute_cells_from_top_left(piece["cells"], top, left)
        assert not (final_occupied & final_cells), "inserted pieces must not overlap"
        final_occupied |= final_cells
        for r, c in final_cells:
            assert rr0 <= r <= rr1 and cc0 <= c <= cc1, "piece must end inside the host rectangle"
            assert g[r, c] in (bg, GREY), "inserted piece should only cover the rectangle or its notch"
            out[r, c] = piece["color"]

    return out


def _build_piece(
    side: str, tab_width: int, color: int, body_h: int, body_w: int, tab_offset: int
) -> dict:
    if side in ("top", "bottom"):
        patch = np.zeros((body_h + 1, body_w), dtype=int)
        if side == "top":
            patch[1:, :] = color
            patch[0, tab_offset : tab_offset + tab_width] = color
        else:
            patch[:-1, :] = color
            patch[-1, tab_offset : tab_offset + tab_width] = color
    else:
        patch = np.zeros((body_h, body_w + 1), dtype=int)
        if side == "left":
            patch[:, 1:] = color
            patch[tab_offset : tab_offset + tab_width, 0] = color
        else:
            patch[:, :-1] = color
            patch[tab_offset : tab_offset + tab_width, -1] = color

    mask = patch == color
    return {
        "side": side,
        "tab_width": int(tab_width),
        "tab_offset": int(tab_offset),
        "body_h": int(body_h),
        "body_w": int(body_w),
        "color": int(color),
        "patch": patch,
        "cells": {(r, c) for r in range(patch.shape[0]) for c in range(patch.shape[1]) if patch[r, c] == color},
        "shape": patch.shape,
    }


def _piece_matches_spec(piece: dict) -> bool:
    try:
        parsed = _parse_piece(piece["patch"] == piece["color"], piece["color"], [])
    except AssertionError:
        return False
    return parsed["side"] == piece["side"] and parsed["tab_width"] == piece["tab_width"]


def _piece_input_bbox(piece: dict, top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    top, left = top_left
    h, w = piece["shape"]
    return top, left, top + h - 1, left + w - 1


def _inside_grid(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _has_padding(cells: set[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BLACK
    target[mask] = patch[mask]


def _candidate_final_placements(piece: dict, rect_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = rect_bbox
    candidates: list[dict] = []
    h, w = piece["shape"]
    if piece["side"] in ("top", "bottom"):
        row = r0 if piece["side"] == "top" else r1 - h + 1
        for start in range(c0 + 1, c1 - piece["tab_width"] + 1):
            if start + piece["tab_width"] - 1 > c1 - 1:
                continue
            left = start - piece["tab_offset"]
            if left < c0 or left + w - 1 > c1:
                continue
            cells = _absolute_cells_from_top_left(piece["cells"], row, left)
            candidates.append(
                {
                    "piece": piece,
                    "notch": {
                        "side": piece["side"],
                        "width": piece["tab_width"],
                        "axis_start": start,
                        "axis_end": start + piece["tab_width"] - 1,
                        "cells": (
                            {(r0, cc) for cc in range(start, start + piece["tab_width"])}
                            if piece["side"] == "top"
                            else {(r1, cc) for cc in range(start, start + piece["tab_width"])}
                        ),
                    },
                    "top_left": (row, left),
                    "cells": cells,
                }
            )
    else:
        col = c0 if piece["side"] == "left" else c1 - w + 1
        for start in range(r0 + 1, r1 - piece["tab_width"] + 1):
            if start + piece["tab_width"] - 1 > r1 - 1:
                continue
            top = start - piece["tab_offset"]
            if top < r0 or top + h - 1 > r1:
                continue
            cells = _absolute_cells_from_top_left(piece["cells"], top, col)
            candidates.append(
                {
                    "piece": piece,
                    "notch": {
                        "side": piece["side"],
                        "width": piece["tab_width"],
                        "axis_start": start,
                        "axis_end": start + piece["tab_width"] - 1,
                        "cells": (
                            {(rr, c0) for rr in range(start, start + piece["tab_width"])}
                            if piece["side"] == "left"
                            else {(rr, c1) for rr in range(start, start + piece["tab_width"])}
                        ),
                    },
                    "top_left": (top, col),
                    "cells": cells,
                }
            )
    return candidates


def _notches_conflict(a: dict, b: dict) -> bool:
    if a["side"] != b["side"]:
        return False
    return not (a["axis_end"] + 1 < b["axis_start"] or b["axis_end"] + 1 < a["axis_start"])


def _choose_final_layout(
    pieces: list[dict], rect_bbox: tuple[int, int, int, int]
) -> list[dict] | None:
    candidate_lists = [ _candidate_final_placements(piece, rect_bbox) for piece in pieces ]
    if any(not options for options in candidate_lists):
        return None

    order = sorted(range(len(pieces)), key=lambda i: len(candidate_lists[i]))
    chosen: list[dict | None] = [None] * len(pieces)
    used_cells: set[tuple[int, int]] = set()
    used_notches: list[dict] = []

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        options = candidate_lists[idx][:]
        random.shuffle(options)
        for option in options:
            if used_cells & option["cells"]:
                continue
            if any(_notches_conflict(option["notch"], other) for other in used_notches):
                continue
            chosen[idx] = option
            used_cells.update(option["cells"])
            used_notches.append(option["notch"])
            if backtrack(k + 1):
                return True
            used_notches.pop()
            used_cells.difference_update(option["cells"])
            chosen[idx] = None
        return False

    if not backtrack(0):
        return None
    return [option for option in chosen if option is not None]


def _preferred_band(side: str, rect_bbox: tuple[int, int, int, int], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect_bbox
    br0, bc0, br1, bc1 = bbox
    if side == "top":
        return br0 >= r1 + 3
    if side == "bottom":
        return br1 <= r0 - 3
    if side == "left":
        return bc0 >= c1 + 3
    return bc1 <= c0 - 3


def _candidate_input_positions(
    grid_shape: tuple[int, int],
    occupied: np.ndarray,
    piece: dict,
    rect_bbox: tuple[int, int, int, int],
    target_notch: dict | None,
    *,
    wrong_notch: dict | None = None,
    preferred_side: str | None = None,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    ph, pw = piece["shape"]
    candidates: list[tuple[int, int]] = []

    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            cells = _absolute_cells_from_top_left(piece["cells"], top, left)
            if not _inside_grid(grid_shape, cells):
                continue
            if not _has_padding(cells, occupied):
                continue
            bbox = _piece_input_bbox(piece, (top, left))
            if target_notch is not None and _shares_target_axis(bbox, target_notch):
                continue
            if preferred_side is not None and not _preferred_band(preferred_side, rect_bbox, bbox):
                continue
            if wrong_notch is not None:
                if not _overlaps_notch_axis(bbox, wrong_notch):
                    continue
                if _piece_notch_distance(bbox, wrong_notch) >= _piece_notch_distance(bbox, target_notch):
                    continue
            candidates.append((top, left))
    return candidates


def _semantic_checks(scene: dict) -> bool:
    inp = scene["input"]
    expected = scene["expected"]
    pieces = scene["pieces"]
    notches = scene["notches"]
    matched = [piece for piece in pieces if piece["matched"]]

    if len([piece for piece in pieces if not piece["matched"]]) != 1:
        return False
    if len({notch["width"] for notch in notches}) != len(notches):
        return False
    if any(piece["area"] <= piece["tab_width"] for piece in matched):
        return False
    if any(_shares_target_axis(piece["input_bbox"], piece["target_notch"]) for piece in matched):
        return False

    nearest_fail = any(
        min(notches, key=lambda notch: _piece_notch_distance(piece["input_bbox"], notch))["width"] != piece["tab_width"]
        for piece in matched
    )
    if not nearest_fail:
        return False

    axis_fail = any(
        piece.get("wrong_notch") is not None
        and _overlaps_notch_axis(piece["input_bbox"], piece["wrong_notch"])
        and not _overlaps_notch_axis(piece["input_bbox"], piece["target_notch"])
        for piece in matched
    )
    if not axis_fail:
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected) and not np.array_equal(inp, expected))


def _generate_scene() -> dict | None:
    height = random.randint(22, 28)
    width = random.randint(22, 28)
    max_rect_h = min(14, height - 8)
    max_rect_w = min(16, width - 8)
    if max_rect_h < 10 or max_rect_w < 10:
        return None
    rect_h = random.randint(10, max_rect_h)
    rect_w = random.randint(10, max_rect_w)
    r0 = random.randint(3, height - rect_h - 3)
    c0 = random.randint(3, width - rect_w - 3)
    rect_bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    matched_count = random.choice([2, 3])
    all_slot_widths = [1, 2, 3]
    if matched_count == 2:
        matched_widths = random.sample(all_slot_widths, 2)
        decoy_width = next(width_ for width_ in all_slot_widths if width_ not in matched_widths)
    else:
        matched_widths = all_slot_widths[:]
        random.shuffle(matched_widths)
        decoy_width = 4

    repeated_side = random.choice(["top", "bottom", "left", "right"])
    sides = [repeated_side, repeated_side]
    if matched_count == 3:
        extra_options = [side for side in ["top", "bottom", "left", "right"] if side != repeated_side]
        sides.append(random.choice(extra_options))
    random.shuffle(matched_widths)

    colors = random.sample(NON_RECT_COLORS, matched_count + 1)
    matched_pieces: list[dict] = []
    for side, tab_width, color in zip(sides, matched_widths, colors[:matched_count]):
        if side in ("top", "bottom"):
            body_h = random.randint(3, min(6, rect_h - 1))
            min_body_w = tab_width + 2
            max_body_w = min(rect_w - 1, tab_width + 5)
            if min_body_w > max_body_w:
                return None
            body_w = random.randint(min_body_w, max_body_w)
            tab_offset = random.randint(1, body_w - tab_width - 1)
        else:
            body_w = random.randint(3, min(6, rect_w - 1))
            min_body_h = tab_width + 2
            max_body_h = min(rect_h - 1, tab_width + 5)
            if min_body_h > max_body_h:
                return None
            body_h = random.randint(min_body_h, max_body_h)
            tab_offset = random.randint(1, body_h - tab_width - 1)
        piece = _build_piece(side, tab_width, color, body_h, body_w, tab_offset)
        if not _piece_matches_spec(piece):
            return None
        matched_pieces.append(piece)

    final_layout = _choose_final_layout(matched_pieces, rect_bbox)
    if final_layout is None:
        return None

    notches = [entry["notch"] for entry in final_layout]
    layout_by_width = {entry["piece"]["tab_width"]: entry for entry in final_layout}
    repeated_widths = [piece["tab_width"] for piece in matched_pieces if piece["side"] == repeated_side]
    repeated_widths = repeated_widths[:2]
    if len(repeated_widths) < 2:
        return None

    wrong_by_width = {
        repeated_widths[0]: layout_by_width[repeated_widths[1]]["notch"],
        repeated_widths[1]: layout_by_width[repeated_widths[0]]["notch"],
    }

    grid = np.full((height, width), BLACK, dtype=int)
    rr0, cc0, rr1, cc1 = rect_bbox
    grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = GREY
    for notch in notches:
        for r, c in notch["cells"]:
            grid[r, c] = BLACK

    expected = grid.copy()
    occupied = grid != BLACK
    pieces_meta: list[dict] = []

    for entry in final_layout:
        piece = entry["piece"]
        target_notch = entry["notch"]
        wrong_notch = wrong_by_width.get(piece["tab_width"])
        preferred_side = repeated_side if wrong_notch is not None else None
        candidates = _candidate_input_positions(
            grid.shape,
            occupied,
            piece,
            rect_bbox,
            target_notch,
            wrong_notch=wrong_notch,
            preferred_side=preferred_side,
        )
        if not candidates:
            return None
        top_left = random.choice(candidates)
        _place_patch(grid, piece["patch"], top_left[0], top_left[1])
        for r, c in _absolute_cells_from_top_left(piece["cells"], *top_left):
            occupied[r, c] = True

        final_cells = entry["cells"]
        for r, c in _absolute_cells_from_top_left(piece["cells"], *top_left):
            expected[r, c] = BLACK
        for r, c in final_cells:
            expected[r, c] = piece["color"]

        pieces_meta.append(
            {
                "matched": True,
                "tab_width": piece["tab_width"],
                "target_notch": target_notch,
                "wrong_notch": wrong_notch,
                "input_bbox": _piece_input_bbox(piece, top_left),
                "area": len(piece["cells"]),
            }
        )

    decoy_side = random.choice(["top", "bottom", "left", "right"])
    decoy_color = colors[-1]
    if decoy_side in ("top", "bottom"):
        decoy_body_h = random.randint(3, min(6, rect_h - 1))
        min_decoy_w = decoy_width + 2
        max_decoy_w = min(rect_w - 1, decoy_width + 5)
        if min_decoy_w > max_decoy_w:
            return None
        decoy_body_w = random.randint(min_decoy_w, max_decoy_w)
        decoy_offset = random.randint(1, decoy_body_w - decoy_width - 1)
    else:
        decoy_body_w = random.randint(3, min(6, rect_w - 1))
        min_decoy_h = decoy_width + 2
        max_decoy_h = min(rect_h - 1, decoy_width + 5)
        if min_decoy_h > max_decoy_h:
            return None
        decoy_body_h = random.randint(min_decoy_h, max_decoy_h)
        decoy_offset = random.randint(1, decoy_body_h - decoy_width - 1)
    decoy_piece = _build_piece(decoy_side, decoy_width, decoy_color, decoy_body_h, decoy_body_w, decoy_offset)
    if not _piece_matches_spec(decoy_piece):
        return None
    decoy_candidates = _candidate_input_positions(
        grid.shape,
        occupied,
        decoy_piece,
        rect_bbox,
        None,
        preferred_side=None,
    )
    if not decoy_candidates:
        return None
    decoy_top_left = random.choice(decoy_candidates)
    _place_patch(grid, decoy_piece["patch"], decoy_top_left[0], decoy_top_left[1])
    _place_patch(expected, decoy_piece["patch"], decoy_top_left[0], decoy_top_left[1])
    pieces_meta.append(
        {
            "matched": False,
            "tab_width": decoy_piece["tab_width"],
            "input_bbox": _piece_input_bbox(decoy_piece, decoy_top_left),
            "area": len(decoy_piece["cells"]),
        }
    )

    scene = {
        "input": grid,
        "expected": expected,
        "pieces": pieces_meta,
        "notches": notches,
    }
    return scene if _semantic_checks(scene) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene["input"]
    raise RuntimeError("failed to generate a valid tab-to-notch insertion puzzle")

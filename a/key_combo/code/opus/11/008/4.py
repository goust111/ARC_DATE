from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
FRAME_COLORS = (2, 3, 5, 6, 7, 8, 9)
FRAME_DIMS = ((5, 7), (7, 5), (5, 5))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                coords.add((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _gap_side(frame: dict, pos: tuple[int, int]) -> str | None:
    top, left, bottom, right = frame["bbox"]
    r, c = pos
    if pos in {(top, left), (top, right), (bottom, left), (bottom, right)}:
        return None
    if r == top:
        return "N"
    if r == bottom:
        return "S"
    if c == left:
        return "W"
    if c == right:
        return "E"
    return None


def _gap_inner(frame: dict, gap: dict) -> tuple[int, int]:
    r, c = gap["pos"]
    side = gap["side"]
    if side == "N":
        return r + 1, c
    if side == "S":
        return r - 1, c
    if side == "W":
        return r, c + 1
    return r, c - 1


def _detect_frames(grid: np.ndarray) -> list[dict]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, BLUE)]
    frames: list[dict] = []
    for color in colors:
        coords = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
        top, left, bottom, right = _bbox(coords)
        if bottom - top < 4 or right - left < 4:
            continue

        boundary: set[tuple[int, int]] = set()
        for c in range(left, right + 1):
            boundary.add((top, c))
            boundary.add((bottom, c))
        for r in range(top + 1, bottom):
            boundary.add((r, left))
            boundary.add((r, right))

        gaps: list[dict] = []
        for pos in sorted(boundary):
            if pos in coords:
                continue
            if int(grid[pos]) != BG:
                raise AssertionError("frame gap must be black")
            side = _gap_side({"bbox": (top, left, bottom, right)}, pos)
            if side is None:
                raise AssertionError("corners cannot be gaps")
            gaps.append({"pos": pos, "side": side})

        expected = boundary - {gap["pos"] for gap in gaps}
        if coords != expected:
            raise AssertionError("non-frame component structure")

        interior_blues = {
            (r, c)
            for r in range(top + 1, bottom)
            for c in range(left + 1, right)
            if int(grid[r, c]) == BLUE
        }
        frames.append(
            {
                "color": color,
                "bbox": (top, left, bottom, right),
                "gaps": gaps,
                "blue": interior_blues,
            }
        )

    if len(frames) != 3:
        raise AssertionError("expected exactly three frames")
    return frames


def _mirror_mode(frame: dict) -> str:
    top, left, bottom, right = frame["bbox"]
    h = bottom - top + 1
    w = right - left + 1
    return "LR" if w >= h else "TB"


def _reflect_point(frame: dict, coord: tuple[int, int]) -> tuple[int, int]:
    top, left, bottom, right = frame["bbox"]
    r, c = coord
    if _mirror_mode(frame) == "LR":
        return r, left + right - c
    return top + bottom - r, c


def _mirrored_blue_union(frame: dict) -> set[tuple[int, int]]:
    out = set(frame["blue"])
    for coord in frame["blue"]:
        out.add(_reflect_point(frame, coord))
    return out


def _point_in_any_frame_bbox(frames: list[dict], r: int, c: int) -> bool:
    for frame in frames:
        top, left, bottom, right = frame["bbox"]
        if top <= r <= bottom and left <= c <= right:
            return True
    return False


def _detect_connections(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    gap_records = []
    for idx, frame in enumerate(frames):
        for gap_idx, gap in enumerate(frame["gaps"]):
            gap_records.append({"frame": idx, "gap": gap_idx, "pos": gap["pos"], "side": gap["side"]})

    connections: list[dict] = []
    for i in range(len(gap_records)):
        for j in range(i + 1, len(gap_records)):
            a = gap_records[i]
            b = gap_records[j]
            if a["frame"] == b["frame"]:
                continue

            ar, ac = a["pos"]
            br, bc = b["pos"]
            cells: list[tuple[int, int]] | None = None

            if ar == br and {a["side"], b["side"]} == {"E", "W"}:
                left_gap, right_gap = (a, b) if ac < bc else (b, a)
                if left_gap["side"] != "E" or right_gap["side"] != "W":
                    continue
                if bc - ac < 2:
                    continue
                ok = True
                for c in range(ac + 1, bc):
                    if int(grid[ar, c]) != BG or _point_in_any_frame_bbox(frames, ar, c):
                        ok = False
                        break
                if ok:
                    cells = [(ar, c) for c in range(ac, bc + 1)]

            if ac == bc and {a["side"], b["side"]} == {"N", "S"}:
                top_gap, bottom_gap = (a, b) if ar < br else (b, a)
                if top_gap["side"] != "S" or bottom_gap["side"] != "N":
                    continue
                if br - ar < 2:
                    continue
                ok = True
                for r in range(ar + 1, br):
                    if int(grid[r, ac]) != BG or _point_in_any_frame_bbox(frames, r, ac):
                        ok = False
                        break
                if ok:
                    cells = [(r, ac) for r in range(ar, br + 1)]

            if cells is not None:
                connections.append(
                    {
                        "frames": (a["frame"], b["frame"]),
                        "gaps": (a["gap"], b["gap"]),
                        "cells": cells,
                    }
                )

    if len(connections) != 2:
        raise AssertionError("expected exactly two corridor links")
    degrees = [0, 0, 0]
    for conn in connections:
        degrees[conn["frames"][0]] += 1
        degrees[conn["frames"][1]] += 1
    if sorted(degrees) != [1, 1, 2]:
        raise AssertionError("connections must form a chain")
    return connections


def _analyze(grid: np.ndarray) -> dict:
    frames = _detect_frames(grid)
    connections = _detect_connections(grid, frames)

    passable = np.zeros_like(grid, dtype=bool)
    mirrored_union_by_frame: list[set[tuple[int, int]]] = []
    active_frames: list[int] = []
    seed_cells: set[tuple[int, int]] = set()

    for idx, frame in enumerate(frames):
        top, left, bottom, right = frame["bbox"]
        passable[top + 1 : bottom, left + 1 : right] = True
        union = _mirrored_blue_union(frame)
        mirrored_union_by_frame.append(union)
        touched = False
        for gap in frame["gaps"]:
            passable[gap["pos"]] = True
            inner = _gap_inner(frame, gap)
            if inner in union:
                touched = True
                seed_cells.add(inner)
        if touched:
            active_frames.append(idx)

    for conn in connections:
        for r, c in conn["cells"]:
            passable[r, c] = True

    return {
        "frames": frames,
        "connections": connections,
        "passable": passable,
        "mirrored_union_by_frame": mirrored_union_by_frame,
        "active_frames": active_frames,
        "seed_cells": sorted(seed_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the blue motif inside each colored frame, and flood the corridor chain only when a mirrored motif reaches a frame gap.

    Input structure:
    - The grid has black background, exactly three non-overlapping colored rectangular frames, and blue cells inside each frame.
    - Each frame has one or two one-pixel wall gaps; the three frames are linked as a chain by straight one-pixel black corridors between facing gaps.

    Rule:
    - For each frame independently, keep its original blue cells and add their mirror copy: wide or square frames mirror left-right, tall frames mirror top-bottom.
    - If the mirrored blue set in a frame reaches the inside neighbor of any gap, start a 4-connected flood from that blue cell through the linked corridor network.
    - The flood may travel only through frame interiors, gap cells, and corridor cells; it paints only black cells blue and never overwrites frame walls.

    Key insight:
    - The frame aspect ratio chooses the mirror axis, but the mirror matters only when it lands exactly beside a gap, because that single contact activates the whole corridor-limited flood chain.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    info = _analyze(g)
    out = g.copy()
    for union in info["mirrored_union_by_frame"]:
        for r, c in union:
            out[r, c] = BLUE

    if not info["seed_cells"]:
        return out

    visited = np.zeros_like(g, dtype=bool)
    q = deque(info["seed_cells"])
    for r, c in info["seed_cells"]:
        visited[r, c] = True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and info["passable"][nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))

    out[visited & (out == BG)] = BLUE
    return out


def _side_of_source_half(frame: dict, coord: tuple[int, int]) -> str | None:
    top, left, bottom, right = frame["bbox"]
    r, c = coord
    if _mirror_mode(frame) == "LR":
        center = (left + right) / 2.0
        if c < center:
            return "left"
        if c > center:
            return "right"
        return None
    center = (top + bottom) / 2.0
    if r < center:
        return "top"
    if r > center:
        return "bottom"
    return None


def _allowed_half_cells(frame: dict, side: str) -> list[tuple[int, int]]:
    top, left, bottom, right = frame["bbox"]
    cells = []
    if _mirror_mode(frame) == "LR":
        center = (left + right) / 2.0
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if side == "left" and c < center:
                    cells.append((r, c))
                if side == "right" and c > center:
                    cells.append((r, c))
    else:
        center = (top + bottom) / 2.0
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if side == "top" and r < center:
                    cells.append((r, c))
                if side == "bottom" and r > center:
                    cells.append((r, c))
    return cells


def _grow_connected_shape(allowed: list[tuple[int, int]], anchor: tuple[int, int], size: int) -> set[tuple[int, int]] | None:
    allowed_set = set(allowed)
    if anchor not in allowed_set:
        return None
    cells = {anchor}
    frontier = {p for p in neighbors4(*anchor) if p in allowed_set}
    while len(cells) < size:
        if not frontier:
            return None
        pick = random.choice(sorted(frontier))
        cells.add(pick)
        frontier.discard(pick)
        for nbr in neighbors4(*pick):
            if nbr in allowed_set and nbr not in cells:
                frontier.add(nbr)
    return cells


def _touched_gap_indices(frame: dict, blue_cells: set[tuple[int, int]]) -> tuple[set[int], set[int]]:
    before = set()
    after = set()
    union = set(blue_cells)
    for coord in blue_cells:
        union.add(_reflect_point(frame, coord))
    for idx, gap in enumerate(frame["gaps"]):
        inner = _gap_inner(frame, gap)
        if inner in blue_cells:
            before.add(idx)
        if inner in union:
            after.add(idx)
    return before, after


def _build_blue_shape(frame: dict, activate_gap: int | None) -> set[tuple[int, int]] | None:
    candidates = []
    if activate_gap is not None:
        target = _gap_inner(frame, frame["gaps"][activate_gap])
        anchor = _reflect_point(frame, target)
        side = _side_of_source_half(frame, anchor)
        if side is None:
            return None
        candidates.append((side, anchor, activate_gap, True))
    else:
        for side in ("left", "right", "top", "bottom"):
            allowed = _allowed_half_cells(frame, side)
            if allowed:
                for anchor in allowed:
                    candidates.append((side, anchor, None, False))
        random.shuffle(candidates)

    for side, anchor, target_gap, must_activate in candidates[:80]:
        allowed = _allowed_half_cells(frame, side)
        if anchor not in set(allowed):
            continue
        max_size = min(6, len(allowed))
        sizes = list(range(3, max_size + 1))
        random.shuffle(sizes)
        for size in sizes:
            for _ in range(20):
                shape = _grow_connected_shape(allowed, anchor, size)
                if shape is None:
                    continue
                before, after = _touched_gap_indices(frame, shape)
                if must_activate:
                    if before:
                        continue
                    if after != {target_gap}:
                        continue
                else:
                    if after:
                        continue
                return shape
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gaps: list[tuple[int, int]]) -> None:
    top, left, bottom, right = bbox
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    for r, c in gaps:
        grid[r, c] = BG


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _mirror_mode_from_dims(h: int, w: int) -> str:
    return "LR" if w >= h else "TB"


def _row_offset_choices(h: int, w: int, gap_side: str, must_activate: bool) -> list[int]:
    choices = list(range(1, h - 1))
    if must_activate and gap_side in {"E", "W"} and _mirror_mode_from_dims(h, w) == "TB" and h % 2 == 1:
        center = h // 2
        choices = [v for v in choices if v != center]
    return choices


def _col_offset_choices(h: int, w: int, gap_side: str, must_activate: bool) -> list[int]:
    choices = list(range(1, w - 1))
    if must_activate and gap_side in {"N", "S"} and _mirror_mode_from_dims(h, w) == "LR" and w % 2 == 1:
        center = w // 2
        choices = [v for v in choices if v != center]
    return choices


def _middle_gaps_keep_connected(h: int, w: int, side_h: str, side_v: str, row_off: int, col_off: int) -> bool:
    bad_row = 1 if side_v == "N" else h - 2
    bad_col = 1 if side_h == "W" else w - 2
    return not (row_off == bad_row and col_off == bad_col)


def _replace_frame_blue(grid: np.ndarray, frame: dict, blue_cells: set[tuple[int, int]]) -> None:
    top, left, bottom, right = frame["bbox"]
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if int(grid[r, c]) == BLUE:
                grid[r, c] = BG
    for r, c in blue_cells:
        grid[r, c] = BLUE


def _sample_layout() -> tuple[np.ndarray, list[dict], int] | None:
    active_idx = random.choice((0, 2))
    dims = [random.choice(FRAME_DIMS) for _ in range(3)]
    colors = random.sample(FRAME_COLORS, 3)
    hdir = random.choice((-1, 1))
    vdir = random.choice((-1, 1))
    sep_ab = random.randint(3, 5)
    sep_bc = random.randint(3, 5)

    (h_a, w_a), (h_b, w_b), (h_c, w_c) = dims
    side_a = "E" if hdir == -1 else "W"
    side_b_h = "W" if hdir == -1 else "E"
    side_b_v = "N" if vdir == -1 else "S"
    side_c = "S" if vdir == -1 else "N"

    row_a_choices = _row_offset_choices(h_a, w_a, side_a, active_idx == 0)
    row_b_choices = _row_offset_choices(h_b, w_b, side_b_h, False)
    col_b_choices = _col_offset_choices(h_b, w_b, side_b_v, False)
    col_c_choices = _col_offset_choices(h_c, w_c, side_c, active_idx == 2)
    if not row_a_choices or not row_b_choices or not col_b_choices or not col_c_choices:
        return None

    row_a = random.choice(row_a_choices)
    row_c_placeholder = random.choice(col_c_choices)
    row_b = None
    col_b = None
    for _ in range(40):
        cand_row_b = random.choice(row_b_choices)
        cand_col_b = random.choice(col_b_choices)
        if _middle_gaps_keep_connected(h_b, w_b, side_b_h, side_b_v, cand_row_b, cand_col_b):
            row_b = cand_row_b
            col_b = cand_col_b
            break
    if row_b is None or col_b is None:
        return None
    col_c = row_c_placeholder

    box_b = (0, 0, h_b - 1, w_b - 1)
    row_link = row_b
    col_link = col_b

    if hdir == -1:
        right_a = box_b[1] - sep_ab
        left_a = right_a - w_a + 1
    else:
        left_a = box_b[3] + sep_ab
        right_a = left_a + w_a - 1
    top_a = row_link - row_a
    box_a = (top_a, left_a, top_a + h_a - 1, right_a)

    if vdir == -1:
        bottom_c = box_b[0] - sep_bc
        top_c = bottom_c - h_c + 1
    else:
        top_c = box_b[2] + sep_bc
    left_c = col_link - col_c
    box_c = (top_c, left_c, top_c + h_c - 1, left_c + w_c - 1)

    if _boxes_overlap(box_a, box_b) or _boxes_overlap(box_a, box_c) or _boxes_overlap(box_b, box_c):
        return None

    min_r = min(box_a[0], box_b[0], box_c[0])
    min_c = min(box_a[1], box_b[1], box_c[1])
    max_r = max(box_a[2], box_b[2], box_c[2])
    max_c = max(box_a[3], box_b[3], box_c[3])
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    shift_r = top_margin - min_r
    shift_c = left_margin - min_c
    boxes = [
        (box_a[0] + shift_r, box_a[1] + shift_c, box_a[2] + shift_r, box_a[3] + shift_c),
        (box_b[0] + shift_r, box_b[1] + shift_c, box_b[2] + shift_r, box_b[3] + shift_c),
        (box_c[0] + shift_r, box_c[1] + shift_c, box_c[2] + shift_r, box_c[3] + shift_c),
    ]
    height = max(box[2] for box in boxes) + 1 + bottom_margin
    width = max(box[3] for box in boxes) + 1 + right_margin
    if max(height, width) > 30:
        return None

    gap_sets = [
        [((boxes[0][0] + row_a), boxes[0][3] if side_a == "E" else boxes[0][1])],
        [
            ((boxes[1][0] + row_b), boxes[1][1] if side_b_h == "W" else boxes[1][3]),
            ((boxes[1][0] if side_b_v == "N" else boxes[1][2]), boxes[1][1] + col_b),
        ],
        [((boxes[2][0] if side_c == "N" else boxes[2][2]), boxes[2][1] + col_c)],
    ]

    grid = np.zeros((height, width), dtype=int)
    frames = []
    for idx, (bbox, color, gaps) in enumerate(zip(boxes, colors, gap_sets)):
        _draw_frame(grid, bbox, color, gaps)
        frames.append({"bbox": bbox, "color": color, "gaps": [{"pos": g, "side": _gap_side({"bbox": bbox}, g)} for g in gaps]})

    return grid, frames, active_idx


def _validate_semantics(grid: np.ndarray, active_idx: int) -> bool:
    info = _analyze(grid)
    if info["active_frames"] != [active_idx]:
        return False
    if len(info["active_frames"]) != 1:
        return False
    if sum(len(frame["blue"]) for frame in info["frames"]) < 9:
        return False

    base_out = main(grid)
    if np.array_equal(base_out, grid):
        return False

    mutant = grid.copy()
    inactive_shape = _build_blue_shape(info["frames"][active_idx], activate_gap=None)
    if inactive_shape is None:
        return False
    _replace_frame_blue(mutant, info["frames"][active_idx], inactive_shape)
    try:
        mutant_info = _analyze(mutant)
    except AssertionError:
        return False
    if mutant_info["active_frames"]:
        return False
    if np.array_equal(main(mutant), base_out):
        return False

    broken = grid.copy()
    corridor_cells = []
    for conn in info["connections"]:
        corridor_cells.extend(conn["cells"][1:-1])
    if not corridor_cells:
        return False
    br, bc = random.choice(corridor_cells)
    broken[br, bc] = info["frames"][0]["color"]
    try:
        if np.array_equal(main(broken), base_out):
            return False
    except AssertionError:
        pass

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sampled = _sample_layout()
        if sampled is None:
            continue
        grid, frame_shells, active_idx = sampled

        shapes: list[set[tuple[int, int]] | None] = [None, None, None]
        ok = True
        for idx, shell in enumerate(frame_shells):
            shape = _build_blue_shape(shell, activate_gap=0 if idx == active_idx else None)
            if shape is None:
                ok = False
                break
            shapes[idx] = shape
            for r, c in shape:
                grid[r, c] = BLUE
        if not ok:
            continue

        try:
            info = _analyze(grid)
        except AssertionError:
            continue

        if info["active_frames"] != [active_idx]:
            continue
        if not _validate_semantics(grid, active_idx):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")

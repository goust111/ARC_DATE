from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
HINGE = 6
ANCHOR = 8
LINE_COLORS = (1, 2, 3, 4, 7, 9)
SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = ("top", "bottom")
VERTICAL_SIDES = ("left", "right")

MIN_GRID = 20
MAX_GRID = 30
MIN_PAIRS = 2
MAX_PAIRS = 4
FRAME_SIZES = (5, 7, 9)
TAIL_RANGE = (2, 6)
FIXED_RANGE = (1, 4)
GEN_RETRIES = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                comp.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _expand(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((int(r) + dr, int(c) + dc))
    return out


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _frame_interior_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _parse_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for idx, comp in enumerate(_components_from_mask(grid == FRAME)):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        border = _frame_border_cells(bbox)
        comp_set = set(comp)
        if not comp_set.issubset(border):
            raise AssertionError("frame contains non-border cells")
        missing = border - comp_set
        if len(missing) != 1:
            raise AssertionError("frame must have exactly one opening")
        opening = next(iter(missing))
        orow, ocol = opening
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            raise AssertionError("frame too small")
        if orow == r0:
            if w % 2 == 0 or ocol != c0 + w // 2:
                raise AssertionError("top opening must be centered")
            side = "top"
        elif orow == r1:
            if w % 2 == 0 or ocol != c0 + w // 2:
                raise AssertionError("bottom opening must be centered")
            side = "bottom"
        elif ocol == c0:
            if h % 2 == 0 or orow != r0 + h // 2:
                raise AssertionError("left opening must be centered")
            side = "left"
        elif ocol == c1:
            if h % 2 == 0 or orow != r0 + h // 2:
                raise AssertionError("right opening must be centered")
            side = "right"
        else:
            raise AssertionError("opening must lie on a side")
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.all(interior == bg):
            raise AssertionError("frame interior must be background")
        frames.append(
            {
                "id": idx,
                "bbox": bbox,
                "opening": opening,
                "side": side,
            }
        )
    if not frames:
        raise AssertionError("expected at least one frame")
    return frames


def _parse_rods(grid: np.ndarray, bg: int) -> list[dict]:
    rod_mask = ~np.isin(grid, [bg, FRAME])
    rods = []
    for idx, comp in enumerate(_components_from_mask(rod_mask)):
        colors = {int(grid[r, c]) for r, c in comp}
        if ANCHOR not in colors or HINGE not in colors:
            raise AssertionError("rod markers missing")
        if sum(grid[r, c] == ANCHOR for r, c in comp) != 1:
            raise AssertionError("rod must have one anchor")
        if sum(grid[r, c] == HINGE for r, c in comp) != 1:
            raise AssertionError("rod must have one hinge")
        line_colors = colors - {ANCHOR, HINGE}
        if len(line_colors) != 1:
            raise AssertionError("rod must have one line color")
        line_color = next(iter(line_colors))
        anchor = next((r, c) for r, c in comp if int(grid[r, c]) == ANCHOR)
        hinge = next((r, c) for r, c in comp if int(grid[r, c]) == HINGE)
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1:
            orientation = "h"
            if anchor[0] != hinge[0]:
                raise AssertionError("horizontal rod markers must share a row")
            ordered = sorted(comp, key=lambda p: p[1])
            min_c = ordered[0][1]
            max_c = ordered[-1][1]
            expected = {(anchor[0], c) for c in range(min_c, max_c + 1)}
            if expected != set(comp):
                raise AssertionError("horizontal rod must be contiguous")
            if anchor[1] not in (min_c, max_c):
                raise AssertionError("anchor must be on an endpoint")
            fixed_dir = (0, -1 if anchor[1] < hinge[1] else 1)
        elif len(cols) == 1:
            orientation = "v"
            if anchor[1] != hinge[1]:
                raise AssertionError("vertical rod markers must share a column")
            ordered = sorted(comp, key=lambda p: p[0])
            min_r = ordered[0][0]
            max_r = ordered[-1][0]
            expected = {(r, anchor[1]) for r in range(min_r, max_r + 1)}
            if expected != set(comp):
                raise AssertionError("vertical rod must be contiguous")
            if anchor[0] not in (min_r, max_r):
                raise AssertionError("anchor must be on an endpoint")
            fixed_dir = (-1 if anchor[0] < hinge[0] else 1, 0)
        else:
            raise AssertionError("rod must be straight")

        fixed_dist = abs(anchor[0] - hinge[0]) + abs(anchor[1] - hinge[1])
        if fixed_dist < 1:
            raise AssertionError("anchor and hinge must differ")
        for step in range(1, fixed_dist):
            rr = hinge[0] + fixed_dir[0] * step
            cc = hinge[1] + fixed_dir[1] * step
            if int(grid[rr, cc]) != int(line_color):
                raise AssertionError("fixed segment pixels must be line-colored")

        tail_dir = (-fixed_dir[0], -fixed_dir[1])
        tail = []
        step = 1
        while True:
            rr = hinge[0] + tail_dir[0] * step
            cc = hinge[1] + tail_dir[1] * step
            if not in_bounds(grid.shape, rr, cc) or (rr, cc) not in set(comp):
                break
            if int(grid[rr, cc]) != int(line_color):
                raise AssertionError("tail pixels must be line-colored")
            tail.append((rr, cc))
            step += 1
        if not tail:
            raise AssertionError("rod must have a movable tail")
        rods.append(
            {
                "id": idx,
                "orientation": orientation,
                "anchor": anchor,
                "hinge": hinge,
                "fixed_dir": fixed_dir,
                "tail_dir": tail_dir,
                "fixed_dist": fixed_dist,
                "tail_len": len(tail),
                "tail": tail,
                "line_color": int(line_color),
            }
        )
    if not rods:
        raise AssertionError("expected at least one rod")
    return rods


def _candidate_matches(grid: np.ndarray, rod: dict, frames: list[dict], bg: int) -> list[tuple[dict, tuple[int, int]]]:
    if rod["orientation"] == "h":
        dirs = [(-1, 0), (1, 0)]
    else:
        dirs = [(0, -1), (0, 1)]
    hinge_r, hinge_c = rod["hinge"]
    tail_len = int(rod["tail_len"])
    matches = []
    for frame in frames:
        opening = frame["opening"]
        for dr, dc in dirs:
            tip = (hinge_r + dr * tail_len, hinge_c + dc * tail_len)
            if tip != opening:
                continue
            clear = True
            for step in range(1, tail_len):
                rr = hinge_r + dr * step
                cc = hinge_c + dc * step
                if int(grid[rr, cc]) != int(bg):
                    clear = False
                    break
            if clear and int(grid[opening]) == int(bg):
                matches.append((frame, (dr, dc)))
    return matches


def _fill_frame(out: np.ndarray, frame: dict, color: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = int(color)
    out[frame["opening"]] = int(color)


def _apply_solution(input_grid: np.ndarray, assignments: list[tuple[dict, dict, tuple[int, int]]]) -> np.ndarray:
    out = np.array(input_grid, dtype=int, copy=True)
    bg = get_bg(out)
    for rod, frame, fold_dir in assignments:
        for rr, cc in rod["tail"]:
            out[rr, cc] = int(bg)
        for step in range(1, int(rod["tail_len"]) + 1):
            rr = rod["hinge"][0] + fold_dir[0] * step
            cc = rod["hinge"][1] + fold_dir[1] * step
            out[rr, cc] = int(rod["line_color"])
        _fill_frame(out, frame, int(rod["line_color"]))
        out[rod["anchor"]] = ANCHOR
        out[rod["hinge"]] = HINGE
    return out


def _more_space_dir(shape: tuple[int, int], rod: dict) -> tuple[int, int] | None:
    h, w = shape
    hr, hc = rod["hinge"]
    if rod["orientation"] == "h":
        up = hr
        down = h - 1 - hr
        if up == down:
            return None
        return (-1, 0) if up > down else (1, 0)
    left = hc
    right = w - 1 - hc
    if left == right:
        return None
    return (0, -1) if left > right else (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked straight rod so its free tip docks exactly into one frame opening, then flood that frame with the rod color.

    Input structure:
    - The background is black.
    - Several disjoint gray hollow rectangles appear; each has a one-cell centered opening on exactly one side, and each interior is black.
    - Several straight horizontal or vertical rods appear elsewhere; each rod has one cyan endpoint marker, one interior magenta hinge marker, and one unique line color.

    Rule:
    - For each rod, keep the segment from the cyan anchor through the magenta hinge fixed.
    - Remove the tail beyond the hinge and rotate that tail by ninety degrees around the hinge.
    - Choose the perpendicular direction whose folded tip lands exactly on one frame opening; every rod matches one frame and every frame matches one rod.
    - After docking, keep the folded rod and fill the matched frame interior plus its opening cell with that rod's line color, preserving the gray border and the two markers.

    Key insight:
    - The hinge does not fold toward the roomier side of the canvas; the only valid fold is the one whose tail length exactly equals the hinge-to-opening distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    frames = _parse_frames(g, bg)
    rods = _parse_rods(g, bg)
    assert len(frames) == len(rods), "rod/frame counts must match"

    assignments: list[tuple[dict, dict, tuple[int, int]]] = []
    used_frames = set()
    for rod in rods:
        matches = _candidate_matches(g, rod, frames, bg)
        assert len(matches) == 1, "each rod must have exactly one dockable frame"
        frame, fold_dir = matches[0]
        assert frame["id"] not in used_frames, "each frame must be used once"
        used_frames.add(frame["id"])
        assignments.append((rod, frame, fold_dir))
    assert len(used_frames) == len(frames), "every frame must be matched"
    return _apply_solution(g, assignments)


def _choose_sides(n_pairs: int) -> list[str]:
    if n_pairs == 2:
        sides = [random.choice(HORIZONTAL_SIDES), random.choice(VERTICAL_SIDES)]
    elif n_pairs == 3:
        sides = [random.choice(HORIZONTAL_SIDES), random.choice(VERTICAL_SIDES), random.choice(SIDES)]
    else:
        sides = list(SIDES)
    random.shuffle(sides)
    return sides


def _record_from_params(
    r0: int,
    c0: int,
    fh: int,
    fw: int,
    side: str,
    tail_len: int,
    fixed_dist: int,
    anchor_sign: int,
    line_color: int,
) -> dict:
    bbox = (int(r0), int(c0), int(r0 + fh - 1), int(c0 + fw - 1))
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "top":
        opening = (r0, mid_c)
        fold_dir = (1, 0)
        hinge = (r0 - tail_len, mid_c)
        orientation = "h"
        fixed_dir = (0, anchor_sign)
    elif side == "bottom":
        opening = (r1, mid_c)
        fold_dir = (-1, 0)
        hinge = (r1 + tail_len, mid_c)
        orientation = "h"
        fixed_dir = (0, anchor_sign)
    elif side == "left":
        opening = (mid_r, c0)
        fold_dir = (0, 1)
        hinge = (mid_r, c0 - tail_len)
        orientation = "v"
        fixed_dir = (anchor_sign, 0)
    else:
        opening = (mid_r, c1)
        fold_dir = (0, -1)
        hinge = (mid_r, c1 + tail_len)
        orientation = "v"
        fixed_dir = (anchor_sign, 0)

    anchor = (hinge[0] + fixed_dir[0] * fixed_dist, hinge[1] + fixed_dir[1] * fixed_dist)
    tail_dir = (-fixed_dir[0], -fixed_dir[1])
    fixed_cells = [
        (hinge[0] + fixed_dir[0] * step, hinge[1] + fixed_dir[1] * step)
        for step in range(1, fixed_dist)
    ]
    tail_cells = [
        (hinge[0] + tail_dir[0] * step, hinge[1] + tail_dir[1] * step)
        for step in range(1, tail_len + 1)
    ]
    rod_cells = set(fixed_cells + tail_cells + [hinge, anchor])
    frame_cells = _frame_border_cells(bbox) - {opening}
    interior_cells = _frame_interior_cells(bbox)
    fold_path = {(hinge[0] + fold_dir[0] * step, hinge[1] + fold_dir[1] * step) for step in range(1, tail_len + 1)}
    return {
        "bbox": bbox,
        "opening": opening,
        "side": side,
        "hinge": hinge,
        "anchor": anchor,
        "orientation": orientation,
        "fixed_dir": fixed_dir,
        "tail_dir": tail_dir,
        "tail_len": int(tail_len),
        "fixed_dist": int(fixed_dist),
        "line_color": int(line_color),
        "frame_cells": frame_cells,
        "interior_cells": interior_cells,
        "fixed_cells": fixed_cells,
        "tail_cells": tail_cells,
        "rod_cells": rod_cells,
        "fold_dir": fold_dir,
        "fold_path": fold_path,
    }


def _placeable(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(in_bounds(shape, r, c) for r, c in cells)


def _draw_input(shape: tuple[int, int], records: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rec in records:
        for r, c in rec["frame_cells"]:
            grid[r, c] = FRAME
        for r, c in rec["fixed_cells"]:
            grid[r, c] = rec["line_color"]
        for r, c in rec["tail_cells"]:
            grid[r, c] = rec["line_color"]
        grid[rec["anchor"]] = ANCHOR
        grid[rec["hinge"]] = HINGE
    return grid


def _expected_output_from_records(input_grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = np.array(input_grid, dtype=int, copy=True)
    for rec in records:
        for r, c in rec["tail_cells"]:
            out[r, c] = BG
        for r, c in rec["fold_path"]:
            out[r, c] = rec["line_color"]
        r0, c0, r1, c1 = rec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = rec["line_color"]
        out[rec["opening"]] = rec["line_color"]
        out[rec["anchor"]] = ANCHOR
        out[rec["hinge"]] = HINGE
    return out


def _validate_generated(grid: np.ndarray, expected: np.ndarray, require_decoy: bool) -> bool:
    try:
        bg = get_bg(grid)
        frames = _parse_frames(grid, bg)
        rods = _parse_rods(grid, bg)
        if len(frames) != len(rods) or len(frames) < MIN_PAIRS:
            return False
        seen_colors = {rod["line_color"] for rod in rods}
        if len(seen_colors) != len(rods):
            return False
        used_frames = set()
        has_decoy = False
        for rod in rods:
            matches = _candidate_matches(grid, rod, frames, bg)
            if len(matches) != 1:
                return False
            frame, fold_dir = matches[0]
            if frame["id"] in used_frames:
                return False
            used_frames.add(frame["id"])
            heuristic = _more_space_dir(grid.shape, rod)
            if heuristic is not None and heuristic != fold_dir:
                has_decoy = True
        if len(used_frames) != len(frames):
            return False
        if require_decoy and not has_decoy:
            return False
        solved = main(grid)
        return np.array_equal(solved, expected) and not np.array_equal(grid, expected)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    n_pairs = random.randint(MIN_PAIRS, MAX_PAIRS if min(h, w) >= 24 else 3)
    sides = _choose_sides(n_pairs)
    line_colors = random.sample(LINE_COLORS, k=n_pairs)
    reserved: set[tuple[int, int]] = set()
    records: list[dict] = []

    for side, line_color in zip(sides, line_colors):
        placed = False
        for _ in range(120):
            fh = random.choice(FRAME_SIZES)
            fw = random.choice(FRAME_SIZES)
            tail_len = random.randint(*TAIL_RANGE)
            fixed_dist = random.randint(*FIXED_RANGE)
            anchor_sign = random.choice((-1, 1))
            r0 = random.randint(1, h - fh - 2)
            c0 = random.randint(1, w - fw - 2)
            rec = _record_from_params(r0, c0, fh, fw, side, tail_len, fixed_dist, anchor_sign, line_color)
            all_cells = set(rec["frame_cells"]) | set(rec["rod_cells"]) | set(rec["interior_cells"]) | set(rec["fold_path"])
            if not _placeable((h, w), all_cells):
                continue
            reserve = _expand(set(rec["frame_cells"]) | set(rec["rod_cells"]), 1) | set(rec["interior_cells"]) | set(rec["fold_path"])
            if any(cell in reserved for cell in reserve):
                continue
            reserved |= reserve
            records.append(rec)
            placed = True
            break
        if not placed:
            return None

    grid = _draw_input((h, w), records)
    expected = _expected_output_from_records(grid, records)
    require_decoy = True
    if not _validate_generated(grid, expected, require_decoy=require_decoy):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fold-and-dock ARC input")

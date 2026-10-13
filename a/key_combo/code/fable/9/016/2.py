from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIR_IN = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GEN_MAX_ATTEMPTS = 400
GEN_FRAME_COUNT_OPTIONS = (1, 2, 2, 3)
GEN_FRAME_SIZE_RANGE = (5, 9)
GEN_CANVAS_MIN = 18
GEN_CANVAS_MAX = 30
GEN_FRAME_GAP = 2
GEN_MAX_MARKERS_PER_FRAME = 4
GEN_RANDOM_DECOY_RANGE = (0, 3)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_from_component(coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have a non-empty interior"

    comp_set = set(coords)
    border = _border_cells(r0, c0, r1, c1)
    assert comp_set <= border, "grey component must stay on its bbox border"
    missing = list(border - comp_set)
    assert len(missing) == 1, "frame border must have exactly one missing cell"
    gap = missing[0]

    if gap[0] == r0 and c0 < gap[1] < c1:
        side = "top"
    elif gap[0] == r1 and c0 < gap[1] < c1:
        side = "bottom"
    elif gap[1] == c0 and r0 < gap[0] < r1:
        side = "left"
    elif gap[1] == c1 and r0 < gap[0] < r1:
        side = "right"
    else:
        raise AssertionError("gap must lie on a non-corner border cell")

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = [_frame_from_component(comp) for comp in _components_of_color(grid, FRAME_COLOR)]
    assert frames, "expected at least one grey frame"
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["bbox"][2], frame["bbox"][3]))
    return frames


def _lane_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    gap_r, gap_c = frame["gap"]
    dr, dc = DIR_IN[frame["side"]]
    rr, cc = gap_r + dr, gap_c + dc
    cells: list[tuple[int, int]] = []
    while r0 < rr < r1 and c0 < cc < c1:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _ray_cells(grid: np.ndarray, frame: dict) -> list[tuple[int, int]]:
    gap_r, gap_c = frame["gap"]
    dr, dc = DIR_IN[frame["side"]]
    rr, cc = gap_r - dr, gap_c - dc
    cells: list[tuple[int, int]] = []
    while _in_bounds(grid.shape, rr, cc):
        if int(grid[rr, cc]) == FRAME_COLOR:
            break
        cells.append((rr, cc))
        rr -= dr
        cc -= dc
    return cells


def _markers_on_ray(grid: np.ndarray, frame: dict, bg: int) -> list[tuple[int, tuple[int, int], int]]:
    markers: list[tuple[int, tuple[int, int], int]] = []
    for dist, (r, c) in enumerate(_ray_cells(grid, frame), start=1):
        color = int(grid[r, c])
        if color not in (int(bg), FRAME_COLOR):
            markers.append((dist, (r, c), color))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each visible outer marker through the unique frame gap and stack it against the opposite inner wall.

    Input structure:
    - The background is uniform, and one or more grey one-cell-thick rectangular frames appear on it.
    - Each grey frame is hollow and has exactly one missing non-corner border cell, which is the gap.
    - Colored singleton markers lie outside the frames; some sit on the outward row or column of a gap, and some do not correspond to any gap.

    Rule:
    - Keep every grey frame unchanged.
    - For each frame, collect the colored pixels lying on the ray that starts at its gap and continues outward from the opening; order them by original distance to the gap from near to far.
    - Remove those collected pixels from their original cells and place them inside the frame along the straight inward lane through the gap: the nearest marker ends deepest against the opposite inner wall, and later markers stack behind earlier ones.
    - Any colored pixel not on such a gap ray stays where it is.

    Key insight:
    - The missing border cell is a door: only its outward line matters, and the final interior color order is the reverse depth stacking induced by the markers' original gap distances.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    frames = _find_frames(g)

    out = g.copy()
    ownership: dict[tuple[int, int], int] = {}
    frame_markers: list[list[tuple[int, tuple[int, int], int]]] = []

    for frame_index, frame in enumerate(frames):
        markers = _markers_on_ray(g, frame, bg)
        lane = _lane_cells(frame)
        assert len(markers) <= len(lane), "more entrants than lane cells"
        frame_markers.append(markers)
        for _dist, coord, _color in markers:
            if coord in ownership:
                raise AssertionError("a marker cannot belong to two gap rays")
            ownership[coord] = frame_index

    for frame, markers in zip(frames, frame_markers):
        lane = _lane_cells(frame)
        for _dist, (r, c), _color in markers:
            out[r, c] = bg
        for idx, (_dist, _coord, color) in enumerate(markers):
            tr, tc = lane[-1 - idx]
            assert int(out[tr, tc]) == int(bg), "frame lane must start empty"
            out[tr, tc] = int(color)

    return out


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 < br0:
        row_gap = br0 - ar1 - 1
    elif br1 < ar0:
        row_gap = ar0 - br1 - 1
    else:
        row_gap = 0
    if ac1 < bc0:
        col_gap = bc0 - ac1 - 1
    elif bc1 < ac0:
        col_gap = ac0 - bc1 - 1
    else:
        col_gap = 0
    return max(row_gap, col_gap)


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    gr, gc = frame["gap"]
    grid[gr, gc] = BG


def _make_frame(r0: int, c0: int, h: int, w: int, side: str) -> dict:
    r1, c1 = r0 + h - 1, c0 + w - 1
    if side in ("top", "bottom"):
        gap = (r0 if side == "top" else r1, random.randint(c0 + 1, c1 - 1))
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)
    return {"bbox": (r0, c0, r1, c1), "gap": gap, "side": side}


def _place_frames(height: int, width: int, frame_count: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BG, dtype=int)
    frames: list[dict] = []
    for _ in range(frame_count):
        placed = False
        for _try in range(180):
            h = random.randint(*GEN_FRAME_SIZE_RANGE)
            w = random.randint(*GEN_FRAME_SIZE_RANGE)
            if h >= height - 2 or w >= width - 2:
                continue
            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            frame = _make_frame(r0, c0, h, w, random.choice(SIDES))
            bbox = frame["bbox"]
            if any(_bbox_gap(bbox, other["bbox"]) < GEN_FRAME_GAP for other in frames):
                continue
            _draw_frame(grid, frame)
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None
    return grid, frames


def _ray_set(grid: np.ndarray, frame: dict) -> set[tuple[int, int]]:
    return set(_ray_cells(grid, frame))


def _pick_marker_count(capacity: int, require_multi: bool) -> int:
    if require_multi:
        return random.randint(2, min(capacity, GEN_MAX_MARKERS_PER_FRAME))
    return random.randint(1, min(capacity, GEN_MAX_MARKERS_PER_FRAME))


def _candidate_wrong_side_cells(grid: np.ndarray, frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    gap_r, gap_c = frame["gap"]
    candidates: list[tuple[int, int]] = []
    if frame["side"] == "top":
        for r in range(r1 + 2, grid.shape[0]):
            candidates.append((r, gap_c))
    elif frame["side"] == "bottom":
        for r in range(r0 - 2, -1, -1):
            candidates.append((r, gap_c))
    elif frame["side"] == "left":
        for c in range(c1 + 2, grid.shape[1]):
            candidates.append((gap_r, c))
    else:
        for c in range(c0 - 2, -1, -1):
            candidates.append((gap_r, c))
    return candidates


def _isolated_cell(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
        for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != BG:
                return False
    return True


def _outside_all_frames(cell: tuple[int, int], frames: list[dict]) -> bool:
    r, c = cell
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
    return True


def _place_decoys(
    inp: np.ndarray,
    frames: list[dict],
    ray_union: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    decoys: list[tuple[int, int, int]] = []

    target_lines = list(frames)
    random.shuffle(target_lines)
    for frame in target_lines:
        for r, c in _candidate_wrong_side_cells(inp, frame):
            if (r, c) in ray_union or not _outside_all_frames((r, c), frames):
                continue
            if _isolated_cell(inp, r, c):
                color = random.choice(MARKER_COLORS)
                inp[r, c] = color
                decoys.append((r, c, color))
                break
        if decoys:
            break

    random_count = random.randint(*GEN_RANDOM_DECOY_RANGE)
    candidates = [
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if (r, c) not in ray_union and _outside_all_frames((r, c), frames) and _isolated_cell(inp, r, c)
    ]
    random.shuffle(candidates)
    for r, c in candidates[:random_count]:
        color = random.choice(MARKER_COLORS)
        inp[r, c] = color
        decoys.append((r, c, color))
    return decoys


def _render_expected_output(
    frame_grid: np.ndarray,
    frame_plans: list[dict],
    decoys: list[tuple[int, int, int]],
) -> np.ndarray:
    out = frame_grid.copy()
    for r, c, color in decoys:
        out[r, c] = color
    for plan in frame_plans:
        lane = _lane_cells(plan["frame"])
        colors = plan["colors_near_to_far"]
        for idx, color in enumerate(colors):
            tr, tc = lane[-1 - idx]
            out[tr, tc] = color
    return out


def _validate_scene(
    inp: np.ndarray,
    out: np.ndarray,
    frames: list[dict],
    frame_plans: list[dict],
    decoys: list[tuple[int, int, int]],
    ray_union: set[tuple[int, int]],
) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(inp, out):
        return False
    if not any(len(plan["colors_near_to_far"]) >= 2 for plan in frame_plans):
        return False

    seen_sources: set[tuple[int, int]] = set()
    for plan in frame_plans:
        frame = plan["frame"]
        lane = _lane_cells(frame)
        colors = plan["colors_near_to_far"]
        sources = plan["source_cells"]
        if len(colors) != len(set(colors)):
            return False
        if not (1 <= len(colors) <= len(lane)):
            return False
        for cell in sources:
            if cell in seen_sources or cell not in ray_union:
                return False
            seen_sources.add(cell)

    for r, c, _color in decoys:
        if (r, c) in ray_union:
            return False

    parsed_frames = _find_frames(inp)
    if len(parsed_frames) != len(frames):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.choice(GEN_FRAME_COUNT_OPTIONS)
    min_side = GEN_CANVAS_MIN + 2 * (frame_count - 1)
    height = random.randint(min_side, GEN_CANVAS_MAX)
    width = random.randint(min_side, GEN_CANVAS_MAX)

    placed = _place_frames(height, width, frame_count)
    if placed is None:
        return None
    frame_grid, frames = placed

    ray_lists = [_ray_cells(frame_grid, frame) for frame in frames]
    capacities = [min(len(ray), len(_lane_cells(frame)), GEN_MAX_MARKERS_PER_FRAME) for ray, frame in zip(ray_lists, frames)]
    if not capacities or min(capacities) < 1 or max(capacities) < 2:
        return None

    ray_sets = [set(ray) for ray in ray_lists]
    for i in range(len(ray_sets)):
        for j in range(i + 1, len(ray_sets)):
            if ray_sets[i] & ray_sets[j]:
                return None
    ray_union = set().union(*ray_sets)

    multi_candidates = [i for i, cap in enumerate(capacities) if cap >= 2]
    if not multi_candidates:
        return None
    multi_index = random.choice(multi_candidates)

    inp = frame_grid.copy()
    frame_plans: list[dict] = []
    for idx, (frame, ray, capacity) in enumerate(zip(frames, ray_lists, capacities)):
        count = _pick_marker_count(capacity, require_multi=(idx == multi_index))
        picked_indices = sorted(random.sample(range(len(ray)), count))
        source_cells = [ray[i] for i in picked_indices]
        colors = random.sample(MARKER_COLORS, count)
        for (r, c), color in zip(source_cells, colors):
            inp[r, c] = color
        frame_plans.append(
            {
                "frame": frame,
                "source_cells": source_cells,
                "colors_near_to_far": colors,
            }
        )

    decoys = _place_decoys(inp, frames, ray_union)
    out = _render_expected_output(frame_grid, frame_plans, decoys)
    if not _validate_scene(inp, out, frames, frame_plans, decoys, ray_union):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid input for puzzle 2")

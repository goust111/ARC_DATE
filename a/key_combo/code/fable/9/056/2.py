from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

DIR_TO_SIDE = {
    UP: "top",
    DOWN: "bottom",
    LEFT: "left",
    RIGHT: "right",
}
SIDE_TO_DIR = {
    "top": UP,
    "bottom": DOWN,
    "left": LEFT,
    "right": RIGHT,
}
OPPOSITE_DIR = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}

MIN_GRID = 18
MAX_GRID = 30
MIN_RECT_SIDE = 5
MAX_RECT_SIDE = 9
MAX_USED_RECTS = 4
MAX_DISTRACTORS = 2
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], rc: tuple[int, int]) -> bool:
    r, c = rc
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _is_border(shape: tuple[int, int], rc: tuple[int, int]) -> bool:
    r, c = rc
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _ray_cells(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    return [_add(start, direction, k) for k in range(1, steps + 1)]


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _all_rect_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _gap_side(rc: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = rc
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the rectangle border")


def _contact_bbox(
    contact: tuple[int, int],
    side: str,
    height: int,
    width: int,
    offset: int,
) -> tuple[int, int, int, int]:
    r, c = contact
    if side == "left":
        r0 = r - offset
        c0 = c
    elif side == "right":
        r0 = r - offset
        c0 = c - width + 1
    elif side == "top":
        r0 = r
        c0 = c - offset
    else:
        r0 = r - height + 1
        c0 = c - offset
    return r0, c0, r0 + height - 1, c0 + width - 1


def _gap_from_bbox(bbox: tuple[int, int, int, int], side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "left":
        return r0 + offset, c0
    if side == "right":
        return r0 + offset, c1
    if side == "top":
        return r0, c0 + offset
    return r1, c0 + offset


def _side_offset_limit(side: str, height: int, width: int) -> int:
    return height - 2 if side in {"left", "right"} else width - 2


def _rectangle_from_bbox(
    bbox: tuple[int, int, int, int],
    gaps: tuple[tuple[int, int], tuple[int, int]],
    *,
    used: bool,
) -> dict:
    gap_set = {gaps[0], gaps[1]}
    green = [rc for rc in _perimeter_coords(bbox) if rc not in gap_set]
    return {"bbox": bbox, "gaps": gaps, "green": green, "used": used}


def _scan_rectangles(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) != GREEN:
                continue
            for r1 in range(r0 + 2, h):
                if int(grid[r1, c0]) != GREEN:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(grid[r0, c1]) != GREEN or int(grid[r1, c1]) != GREEN:
                        continue
                    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
                        continue

                    border = _perimeter_coords((r0, c0, r1, c1))
                    vals = [int(grid[r, c]) for r, c in border]
                    if any(v not in (BLACK, GREEN) for v in vals):
                        continue
                    gaps = [rc for rc in border if int(grid[rc]) == BLACK]
                    if len(gaps) != 2:
                        continue
                    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
                    if any(g in corners for g in gaps):
                        continue
                    if any(int(grid[r, c]) != GREEN for r, c in border if (r, c) not in gaps):
                        continue
                    candidates.append(_rectangle_from_bbox((r0, c0, r1, c1), (gaps[0], gaps[1]), used=False))

    green_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == GREEN)}
    rects: list[dict] = []
    covered: set[tuple[int, int]] = set()
    candidates.sort(
        key=lambda rect: (
            (rect["bbox"][2] - rect["bbox"][0] + 1) * (rect["bbox"][3] - rect["bbox"][1] + 1),
            len(rect["green"]),
        ),
        reverse=True,
    )
    for rect in candidates:
        green_set = set(rect["green"])
        if green_set & covered:
            continue
        rects.append(rect)
        covered.update(green_set)
    if covered != green_cells:
        raise AssertionError("every green cell must belong to one selected hollow rectangle border")
    return rects


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BLACK, RED, GREEN}), "unexpected input colors"

    red_pts = np.argwhere(g == RED)
    assert red_pts.shape[0] == 1, "expected a unique red start pixel"
    start = tuple(map(int, red_pts[0]))
    assert _is_border(g.shape, start), "red start pixel must lie on the canvas border"

    if start[0] == 0:
        direction = DOWN
    elif start[0] == g.shape[0] - 1:
        direction = UP
    elif start[1] == 0:
        direction = RIGHT
    else:
        direction = LEFT

    rects = _scan_rectangles(g)
    assert rects, "expected at least one hollow rectangle"

    gap_map = {}
    for idx, rect in enumerate(rects):
        g1, g2 = rect["gaps"]
        side1 = _gap_side(g1, rect["bbox"])
        side2 = _gap_side(g2, rect["bbox"])
        gap_map[g1] = {"rect": idx, "other": g2, "exit_dir": SIDE_TO_DIR[side2]}
        gap_map[g2] = {"rect": idx, "other": g1, "exit_dir": SIDE_TO_DIR[side1]}

    return {"grid": g, "start": start, "direction": direction, "rects": rects, "gap_map": gap_map}


def _simulate(parsed: dict, *, return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    g = parsed["grid"]
    out = g.copy()
    current = parsed["start"]
    direction = parsed["direction"]
    gap_map = parsed["gap_map"]

    used_rects: list[int] = []
    while True:
        nxt = _add(current, direction)
        if not _in_bounds(g.shape, nxt):
            stop_reason = "exit"
            break
        if int(out[nxt]) == RED:
            stop_reason = "revisit"
            break
        if nxt in gap_map:
            info = gap_map[nxt]
            out[nxt] = RED
            other = info["other"]
            out[other] = RED
            used_rects.append(int(info["rect"]))
            current = other
            direction = info["exit_dir"]
            continue
        if int(g[nxt]) == GREEN:
            stop_reason = "wall"
            break
        out[nxt] = RED
        current = nxt

    if return_meta:
        meta = {"stop_reason": stop_reason, "used_rects": used_rects}
        return out, meta
    return out


def _naive_straight_only(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_input(grid)
    g = parsed["grid"]
    out = g.copy()
    current = parsed["start"]
    direction = parsed["direction"]
    while True:
        nxt = _add(current, direction)
        if not _in_bounds(g.shape, nxt):
            break
        if int(out[nxt]) == RED or int(g[nxt]) == GREEN:
            break
        out[nxt] = RED
        current = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the red beam through paired box gaps and paint every visited background cell.

    Input structure:
    - The background is black.
    - There are several green hollow rectangles; each border has exactly two one-cell gaps.
    - A unique red pixel lies on the outer canvas boundary and marks the beam source.

    Rule:
    - The beam enters perpendicular to the boundary edge containing the red source and travels straight, painting traversed black cells red.
    - If the next cell is a rectangle gap, that gap turns red and the beam immediately teleports to the paired gap on the same rectangle; that paired gap also turns red, and the beam continues outward along that exit gap's side normal.
    - If the next cell is a non-gap green border cell or an already red cell, the beam stops; if it steps out of bounds, the beam ends.

    Key insight:
    - Each rectangle acts as a fixed two-port teleporter: only the paired gaps matter, and the exit direction is determined by the side containing the exit gap rather than by reflection or interior travel.
    """
    parsed = _parse_input(grid)
    return _simulate(parsed)


def _empty_state() -> dict:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge in {"top", "bottom"}:
        pos = random.randint(2, w - 3)
        start = (0, pos) if edge == "top" else (h - 1, pos)
    else:
        pos = random.randint(2, h - 3)
        start = (pos, 0) if edge == "left" else (pos, w - 1)

    return {
        "shape": (h, w),
        "start": start,
        "current": start,
        "direction": DOWN if edge == "top" else UP if edge == "bottom" else RIGHT if edge == "left" else LEFT,
        "blocked": set(),
        "green": set(),
        "future_red": {start},
        "rects": [],
        "used_count": 0,
        "has_turn": False,
    }


def _clone_state(state: dict) -> dict:
    cloned = {
        "shape": state["shape"],
        "start": state["start"],
        "current": state["current"],
        "direction": state["direction"],
        "blocked": set(state["blocked"]),
        "green": set(state["green"]),
        "future_red": set(state["future_red"]),
        "rects": list(state["rects"]),
        "used_count": state["used_count"],
        "has_turn": state["has_turn"],
    }
    if "direction0" in state:
        cloned["direction0"] = state["direction0"]
    if "stop_reason" in state:
        cloned["stop_reason"] = state["stop_reason"]
    return cloned


def _add_rectangle(state: dict, rect: dict) -> None:
    state["rects"].append(rect)
    state["blocked"].update(_all_rect_cells(rect["bbox"]))
    state["green"].update(rect["green"])


def _segment_is_clear(state: dict, cells: Iterable[tuple[int, int]]) -> bool:
    shape = state["shape"]
    for cell in cells:
        if not _in_bounds(shape, cell):
            return False
        if cell in state["blocked"] or cell in state["future_red"]:
            return False
    return True


def _weighted_exit_dirs(in_dir: tuple[int, int], need_turn: bool) -> list[tuple[int, int]]:
    dirs = list(DIRS)
    random.shuffle(dirs)

    def weight(out_dir: tuple[int, int]) -> int:
        if out_dir == in_dir:
            return 1 if need_turn else 2
        if out_dir == OPPOSITE_DIR[in_dir]:
            return 3
        return 6

    return sorted(dirs, key=weight, reverse=True)


def _try_place_used_rectangle(state: dict) -> bool:
    contact_side = DIR_TO_SIDE[OPPOSITE_DIR[state["direction"]]]
    need_turn = not state["has_turn"]

    for _ in range(120):
        distance = random.randint(2, 6)
        segment = _ray_cells(state["current"], state["direction"], distance)
        if not _segment_is_clear(state, segment):
            continue
        entry_gap = segment[-1]

        out_dirs = _weighted_exit_dirs(state["direction"], need_turn)
        for out_dir in out_dirs:
            exit_side = DIR_TO_SIDE[out_dir]
            height = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
            width = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
            contact_limit = _side_offset_limit(contact_side, height, width)
            exit_limit = _side_offset_limit(exit_side, height, width)
            contact_offset = random.randint(1, contact_limit)
            exit_offset = random.randint(1, exit_limit)
            bbox = _contact_bbox(entry_gap, contact_side, height, width, contact_offset)
            rect_cells = _all_rect_cells(bbox)
            if not all(_in_bounds(state["shape"], cell) for cell in rect_cells):
                continue
            if any(cell in state["blocked"] or cell in state["future_red"] for cell in rect_cells):
                continue
            if any(cell in rect_cells for cell in segment[:-1]):
                continue

            exit_gap = _gap_from_bbox(bbox, exit_side, exit_offset)
            if exit_gap == entry_gap:
                continue

            rect = _rectangle_from_bbox(bbox, (entry_gap, exit_gap), used=True)
            trial = _clone_state(state)
            _add_rectangle(trial, rect)
            trial["future_red"].update(segment)
            trial["future_red"].add(exit_gap)
            trial["current"] = exit_gap
            trial["has_turn"] = trial["has_turn"] or (out_dir != state["direction"])
            trial["direction"] = out_dir
            trial["used_count"] += 1

            state.clear()
            state.update(trial)
            return True
    return False


def _try_finish_exit(state: dict) -> bool:
    cells: list[tuple[int, int]] = []
    step = 1
    while True:
        nxt = _add(state["current"], state["direction"], step)
        if not _in_bounds(state["shape"], nxt):
            break
        if nxt in state["blocked"] or nxt in state["future_red"]:
            return False
        cells.append(nxt)
        step += 1
    if not cells:
        return False
    state["future_red"].update(cells)
    state["stop_reason"] = "exit"
    return True


def _try_finish_revisit(state: dict) -> bool:
    new_cells: list[tuple[int, int]] = []
    step = 1
    while True:
        nxt = _add(state["current"], state["direction"], step)
        if not _in_bounds(state["shape"], nxt):
            return False
        if nxt in state["blocked"]:
            return False
        if nxt in state["future_red"]:
            if not new_cells:
                return False
            state["future_red"].update(new_cells)
            state["stop_reason"] = "revisit"
            return True
        new_cells.append(nxt)
        step += 1


def _sample_gap_positions(
    bbox: tuple[int, int, int, int],
    forbid: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    choices = [cell for cell in _perimeter_coords(bbox) if cell not in forbid]
    corners = {(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3])}
    choices = [cell for cell in choices if cell not in corners]
    if len(choices) < 2:
        return None
    g1, g2 = random.sample(choices, 2)
    return g1, g2


def _try_finish_wall(state: dict) -> bool:
    contact_side = DIR_TO_SIDE[OPPOSITE_DIR[state["direction"]]]
    for _ in range(120):
        clear_len = random.randint(1, 5)
        lead = _ray_cells(state["current"], state["direction"], clear_len)
        if not _segment_is_clear(state, lead):
            continue
        contact = _add(state["current"], state["direction"], clear_len + 1)
        if not _in_bounds(state["shape"], contact):
            continue

        height = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
        width = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
        contact_limit = _side_offset_limit(contact_side, height, width)
        contact_offset = random.randint(1, contact_limit)
        bbox = _contact_bbox(contact, contact_side, height, width, contact_offset)
        rect_cells = _all_rect_cells(bbox)
        if not all(_in_bounds(state["shape"], cell) for cell in rect_cells):
            continue
        if any(cell in state["blocked"] or cell in state["future_red"] for cell in rect_cells):
            continue
        if any(cell in rect_cells for cell in lead):
            continue

        gaps = _sample_gap_positions(bbox, {contact})
        if gaps is None:
            continue
        rect = _rectangle_from_bbox(bbox, gaps, used=False)
        _add_rectangle(state, rect)
        state["future_red"].update(lead)
        state["stop_reason"] = "wall"
        return True
    return False


def _try_place_distractor(state: dict) -> bool:
    h, w = state["shape"]
    for _ in range(100):
        height = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
        width = random.randint(MIN_RECT_SIDE, MAX_RECT_SIDE)
        if height >= h or width >= w:
            continue
        r0 = random.randint(0, h - height)
        c0 = random.randint(0, w - width)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        rect_cells = _all_rect_cells(bbox)
        if any(cell in state["blocked"] or cell in state["future_red"] for cell in rect_cells):
            continue
        gaps = _sample_gap_positions(bbox, set())
        if gaps is None:
            continue
        rect = _rectangle_from_bbox(bbox, gaps, used=False)
        _add_rectangle(state, rect)
        return True
    return False


def _render_input(state: dict) -> np.ndarray:
    grid = np.zeros(state["shape"], dtype=int)
    for r, c in state["green"]:
        grid[r, c] = GREEN
    sr, sc = state["start"]
    grid[sr, sc] = RED
    return grid


def _render_output(state: dict) -> np.ndarray:
    out = _render_input(state)
    for r, c in state["future_red"]:
        out[r, c] = RED
    return out


def _meta_gap_map(rects: list[dict]) -> dict:
    gap_map = {}
    for idx, rect in enumerate(rects):
        g1, g2 = rect["gaps"]
        side1 = _gap_side(g1, rect["bbox"])
        side2 = _gap_side(g2, rect["bbox"])
        gap_map[g1] = {"rect": idx, "other": g2, "exit_dir": SIDE_TO_DIR[side2]}
        gap_map[g2] = {"rect": idx, "other": g1, "exit_dir": SIDE_TO_DIR[side1]}
    return gap_map


def _validate_generated(inp: np.ndarray, expected: np.ndarray, state: dict) -> bool:
    try:
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False

        parsed = _parse_input(inp)
        sim_out, meta = _simulate(parsed, return_meta=True)
        if not np.array_equal(sim_out, expected):
            return False
        if meta["stop_reason"] != state["stop_reason"]:
            return False
        if len(meta["used_rects"]) != state["used_count"]:
            return False
        if state["used_count"] < 1 or not state["has_turn"]:
            return False
        if np.array_equal(_naive_straight_only(inp), expected):
            return False

        manual = _simulate(
            {
                "grid": inp,
                "start": state["start"],
                "direction": state["direction0"],
                "rects": state["rects"],
                "gap_map": _meta_gap_map(state["rects"]),
            }
        )
        if not np.array_equal(manual, expected):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    state = _empty_state()
    state["direction0"] = state["direction"]

    mode = random.choices(["exit", "wall", "revisit"], weights=[5, 3, 2], k=1)[0]
    max_used = random.randint(1, MAX_USED_RECTS)
    min_used = random.randint(1, min(3, max_used))

    finished = False
    for _ in range(max_used):
        if not _try_place_used_rectangle(state):
            return None
        if state["used_count"] < min_used:
            continue
        if mode == "exit" and _try_finish_exit(state):
            finished = True
            break
        if mode == "wall" and _try_finish_wall(state):
            finished = True
            break
        if mode == "revisit" and _try_finish_revisit(state):
            finished = True
            break
    if not finished:
        return None

    min_distractors = 1 if len(state["rects"]) == 1 else 0
    distractor_count = random.randint(min_distractors, MAX_DISTRACTORS)
    for _ in range(distractor_count):
        if not _try_place_distractor(state):
            return None

    inp = _render_input(state)
    expected = _render_output(state)
    if _validate_generated(inp, expected, state):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 双口盒传送光束 input")

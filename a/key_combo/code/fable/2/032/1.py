from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask


DIRS = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

_GEN_CALLS = 0


def _edge_side(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = map(int, bbox)
    h, w = map(int, shape)
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each rectangle must touch exactly one outer edge"
    return sides[0]


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for idx, coords in enumerate(components_mask(grid != 0, connectivity=4)):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in np.unique(patch) if int(v) != 0}
        assert len(colors) == 1, "each object must be monochrome"
        color = int(next(iter(colors)))
        assert patch.shape[0] * patch.shape[1] == int(coords.shape[0]), "each object must be a solid rectangle"
        assert np.all(patch == color), "rectangles cannot contain holes or background"
        side = _edge_side((r0, c0, r1, c1), grid.shape)
        dr, dc = DIRS[side]
        rects.append(
            {
                "id": int(idx),
                "color": color,
                "side": side,
                "dr": int(dr),
                "dc": int(dc),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
            }
        )
    assert rects, "expected at least one border rectangle"
    return rects


def _copy_rect(rect: dict) -> dict:
    out = dict(rect)
    out["blockers"] = list(rect.get("blockers", []))
    return out


def _front_cells(rect: dict) -> list[tuple[int, int]]:
    if rect["dr"] == 1:
        return [(int(rect["r1"] + 1), int(c)) for c in range(int(rect["c0"]), int(rect["c1"]) + 1)]
    if rect["dr"] == -1:
        return [(int(rect["r0"] - 1), int(c)) for c in range(int(rect["c0"]), int(rect["c1"]) + 1)]
    if rect["dc"] == 1:
        return [(int(r), int(rect["c1"] + 1)) for r in range(int(rect["r0"]), int(rect["r1"]) + 1)]
    return [(int(r), int(rect["c0"] - 1)) for r in range(int(rect["r0"]), int(rect["r1"]) + 1)]


def _paint_rect(out: np.ndarray, rect: dict) -> None:
    target = out[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1]
    assert not np.any(target != 0), "rectangles cannot overlap"
    target[:] = int(rect["color"])


def _render_rects(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        _paint_rect(out, rect)
    return out


def _simulate(rects: list[dict], shape: tuple[int, int], *, check_future: bool) -> tuple[list[dict], list[dict]]:
    h, w = map(int, shape)
    states = [_copy_rect(rect) for rect in rects]
    for rect in states:
        rect["active"] = True
        rect["moves"] = 0
        rect["stop_reason"] = None
        rect["stop_step"] = None
        rect["blockers"] = []

    history: list[dict] = []
    step = 0
    while any(bool(rect["active"]) for rect in states):
        occ = np.full((h, w), -1, dtype=int)
        for rect in states:
            rs = slice(int(rect["r0"]), int(rect["r1"]) + 1)
            cs = slice(int(rect["c0"]), int(rect["c1"]) + 1)
            view = occ[rs, cs]
            assert not np.any(view != -1), "rectangles cannot overlap during simulation"
            view[:] = int(rect["id"])

        stop_info: dict[int, tuple[str, list[int]]] = {}
        move_ids: list[int] = []
        future_occ = np.full((h, w), -1, dtype=int)

        for rect in states:
            if not bool(rect["active"]):
                continue
            next_cells = _front_cells(rect)
            if any(nr < 0 or nr >= h or nc < 0 or nc >= w for nr, nc in next_cells):
                stop_info[int(rect["id"])] = ("edge", [])
                continue

            blockers = sorted(
                {
                    int(occ[nr, nc])
                    for nr, nc in next_cells
                    if int(occ[nr, nc]) not in (-1, int(rect["id"]))
                }
            )
            if blockers:
                stop_info[int(rect["id"])] = ("blocked", blockers)
                continue

            move_ids.append(int(rect["id"]))
            if check_future:
                nr0 = int(rect["r0"] + rect["dr"])
                nc0 = int(rect["c0"] + rect["dc"])
                nr1 = int(rect["r1"] + rect["dr"])
                nc1 = int(rect["c1"] + rect["dc"])
                view = future_occ[nr0 : nr1 + 1, nc0 : nc1 + 1]
                assert not np.any(view != -1), "two rectangles tried to enter the same cell"
                view[:] = int(rect["id"])

        for rect_id, (reason, blockers) in stop_info.items():
            rect = states[rect_id]
            rect["active"] = False
            rect["stop_reason"] = reason
            rect["stop_step"] = int(step)
            rect["blockers"] = list(blockers)
            history.append(
                {
                    "step": int(step),
                    "rect": int(rect_id),
                    "reason": str(reason),
                    "blockers": list(blockers),
                }
            )

        for rect_id in move_ids:
            rect = states[rect_id]
            rect["r0"] += int(rect["dr"])
            rect["r1"] += int(rect["dr"])
            rect["c0"] += int(rect["dc"])
            rect["c1"] += int(rect["dc"])
            rect["moves"] += 1

        step += 1
        assert step <= h + w + 5, "simulation exceeded the expected step budget"

    return states, history


def _independent_output(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    shifted: list[dict] = []
    for rect in rects:
        moved = _copy_rect(rect)
        rect_h = int(rect["r1"] - rect["r0"] + 1)
        rect_w = int(rect["c1"] - rect["c0"] + 1)
        if rect["side"] == "top":
            moved["r0"] = int(h - rect_h)
            moved["r1"] = int(h - 1)
        elif rect["side"] == "bottom":
            moved["r0"] = 0
            moved["r1"] = int(rect_h - 1)
        elif rect["side"] == "left":
            moved["c0"] = int(w - rect_w)
            moved["c1"] = int(w - 1)
        else:
            moved["c0"] = 0
            moved["c1"] = int(rect_w - 1)
        shifted.append(moved)
    return _render_rects(shifted, shape)


def _axis(rect: dict) -> str:
    return "v" if int(rect["dr"]) != 0 else "h"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each border-touching rectangle toward the opposite edge with synchronous one-step motion and permanent blocking.

    Input structure:
    - The background is black.
    - The foreground contains several solid monochrome rectangles, and each rectangle touches exactly one outer edge.
    - Different rectangles start separated, but some travel lanes cross or face each other.

    Rule:
    - A top or bottom rectangle moves vertically away from its touched edge; a left or right rectangle moves horizontally away from its touched edge.
    - All unfrozen rectangles evolve synchronously, one cell per step.
    - At the start of each step, any rectangle whose front-adjacent cell band is occupied by another rectangle stops permanently in its current place; the remaining rectangles advance by one cell.
    - A rectangle that is never blocked keeps moving until its front would leave the grid, so it finishes against the opposite edge.

    Key insight:
    - The final arrangement depends on the timing of simultaneous motion, so rectangles cannot be solved independently by sending each one straight to the opposite edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = _parse_rectangles(g)
    final_rects, _history = _simulate(rects, g.shape, check_future=True)
    return _render_rects(final_rects, g.shape)


def _apply_random_flips(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    return out


def _vertical_head_on_input() -> tuple[np.ndarray, str]:
    pair_width = random.randint(1, 3)
    extra_count = random.randint(1, 2)
    widths = [pair_width] + [random.randint(1, 3) for _ in range(extra_count)]
    gaps = [random.randint(1, 3) for _ in range(len(widths) - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    w = left_margin + right_margin + sum(widths) + sum(gaps)
    if w < 8:
        right_margin += 8 - w
        w = 8
    assert w <= 30

    h_top = random.randint(1, 3)
    h_bottom = random.randint(1, 3)
    gap = 2 * random.randint(1, 5)
    h = h_top + h_bottom + gap
    while h < 8:
        h += 2
    assert h <= 30

    bands = []
    c = left_margin
    for idx, width in enumerate(widths):
        bands.append((int(c), int(c + width - 1)))
        c += width
        if idx < len(gaps):
            c += gaps[idx]

    colors = random.sample(range(1, 10), 2 + extra_count)
    rects = [
        {"color": colors[0], "r0": 0, "c0": bands[0][0], "r1": h_top - 1, "c1": bands[0][1]},
        {"color": colors[1], "r0": h - h_bottom, "c0": bands[0][0], "r1": h - 1, "c1": bands[0][1]},
    ]
    for idx in range(extra_count):
        height = random.randint(1, min(4, h - 2))
        c0, c1 = bands[idx + 1]
        if random.random() < 0.5:
            rects.append({"color": colors[idx + 2], "r0": 0, "c0": c0, "r1": height - 1, "c1": c1})
        else:
            rects.append({"color": colors[idx + 2], "r0": h - height, "c0": c0, "r1": h - 1, "c1": c1})

    grid = _render_rects(rects, (h, w))
    return _apply_random_flips(grid), "head_on"


def _horizontal_head_on_input() -> tuple[np.ndarray, str]:
    pair_height = random.randint(1, 3)
    extra_count = random.randint(1, 2)
    heights = [pair_height] + [random.randint(1, 3) for _ in range(extra_count)]
    gaps = [random.randint(1, 3) for _ in range(len(heights) - 1)]
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    h = top_margin + bottom_margin + sum(heights) + sum(gaps)
    if h < 8:
        bottom_margin += 8 - h
        h = 8
    assert h <= 30

    w_left = random.randint(1, 3)
    w_right = random.randint(1, 3)
    gap = 2 * random.randint(1, 5)
    w = w_left + w_right + gap
    while w < 8:
        w += 2
    assert w <= 30

    bands = []
    r = top_margin
    for idx, height in enumerate(heights):
        bands.append((int(r), int(r + height - 1)))
        r += height
        if idx < len(gaps):
            r += gaps[idx]

    colors = random.sample(range(1, 10), 2 + extra_count)
    rects = [
        {"color": colors[0], "r0": bands[0][0], "c0": 0, "r1": bands[0][1], "c1": w_left - 1},
        {"color": colors[1], "r0": bands[0][0], "c0": w - w_right, "r1": bands[0][1], "c1": w - 1},
    ]
    for idx in range(extra_count):
        width = random.randint(1, min(4, w - 2))
        r0, r1 = bands[idx + 1]
        if random.random() < 0.5:
            rects.append({"color": colors[idx + 2], "r0": r0, "c0": 0, "r1": r1, "c1": width - 1})
        else:
            rects.append({"color": colors[idx + 2], "r0": r0, "c0": w - width, "r1": r1, "c1": w - 1})

    grid = _render_rects(rects, (h, w))
    return _apply_random_flips(grid), "head_on"


def _cross_input(blocked_axis: str) -> tuple[np.ndarray, str]:
    vert_h = random.randint(1, 3)
    horiz_w = random.randint(1, 3)

    if blocked_axis == "horizontal":
        row = random.randint(vert_h + 1, vert_h + 8)
        col = row + horiz_w
    else:
        row = random.randint(vert_h + 1, vert_h + 8)
        col = row - vert_h
        if col < 1:
            col = 1
            row = col + vert_h

    h = max(8, row + vert_h + random.randint(2, 8))
    w = max(8, max(col + 2, horiz_w + 2) + random.randint(1, 8))
    assert h <= 30 and w <= 30

    colors = random.sample(range(1, 10), 2)
    rects = [
        {"color": colors[0], "r0": 0, "c0": col, "r1": vert_h - 1, "c1": col},
        {"color": colors[1], "r0": row, "c0": 0, "r1": row, "c1": horiz_w - 1},
    ]
    grid = _render_rects(rects, (h, w))
    return _apply_random_flips(grid), "cross"


def _scenario_name() -> str:
    global _GEN_CALLS
    fixed = [
        "cross_horizontal",
        "vertical_head_on",
        "cross_vertical",
        "horizontal_head_on",
    ]
    if _GEN_CALLS < len(fixed):
        name = fixed[_GEN_CALLS]
    else:
        name = random.choices(
            ["cross_horizontal", "vertical_head_on", "cross_vertical", "horizontal_head_on"],
            weights=[3, 2, 3, 2],
            k=1,
        )[0]
    _GEN_CALLS += 1
    return name


def _build_candidate() -> tuple[np.ndarray, str]:
    name = _scenario_name()
    if name == "vertical_head_on":
        return _vertical_head_on_input()
    if name == "horizontal_head_on":
        return _horizontal_head_on_input()
    if name == "cross_vertical":
        return _cross_input("vertical")
    return _cross_input("horizontal")


def _valid_instance(grid: np.ndarray, expected_kind: str) -> bool:
    try:
        rects = _parse_rectangles(grid)
        if len(rects) < 2:
            return False
        if len({int(rect["color"]) for rect in rects}) != len(rects):
            return False

        final_rects, history = _simulate(rects, grid.shape, check_future=True)
        out = _render_rects(final_rects, grid.shape)
        if not np.array_equal(out, main(grid)):
            return False
        if np.array_equal(out, _independent_output(rects, grid.shape)):
            return False

        blocked_events = [event for event in history if event["reason"] == "blocked"]
        if not blocked_events:
            return False
        if not any(event["reason"] == "edge" for event in history):
            return False

        if expected_kind == "head_on":
            saw_expected = False
            for event in blocked_events:
                rect = final_rects[int(event["rect"])]
                for blocker_id in event["blockers"]:
                    blocker = final_rects[int(blocker_id)]
                    if _axis(rect) == _axis(blocker) and int(rect["dr"]) == -int(blocker["dr"]) and int(rect["dc"]) == -int(blocker["dc"]):
                        saw_expected = True
            if not saw_expected:
                return False

        if expected_kind == "cross":
            saw_expected = False
            for event in blocked_events:
                rect = final_rects[int(event["rect"])]
                for blocker_id in event["blockers"]:
                    blocker = final_rects[int(blocker_id)]
                    if _axis(rect) != _axis(blocker):
                        saw_expected = True
            if not saw_expected:
                return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, expected_kind = _build_candidate()
        if _valid_instance(grid, expected_kind):
            return grid
    raise RuntimeError("generate_input failed to build a valid sliding-block sample")

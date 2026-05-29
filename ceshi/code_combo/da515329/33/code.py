# Combo variant of da515329 (index 33)
# combo_spec: D7/1+D10/1+D13/2
# Name: 扩展网格 + 边缘十字 + 边缘停止

import math
import random

import numpy as np

from arc_utils import neighbors4


_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_TURN_RIGHT = {_DIRS_URDL[i]: _DIRS_URDL[(i + 1) % 4] for i in range(4)}
_TURN_LEFT = {_DIRS_URDL[i]: _DIRS_URDL[(i - 1) % 4] for i in range(4)}
_SIDE_PATTERNS = [("top",), ("left",), ("top", "left")]


# Design analysis required by the task:
# a) Concrete behavior:
#    - Start from a standard da515329-style square canvas whose size depends only
#      on the true arm length.
#    - Crop away the top edge, the left edge, or both, so the input cross sits on
#      an edge/corner and one or two arms are visibly shortened.
#    - Recover the missing rows/cols in the output, rebuild the full cross, then
#      run the original spiral turtles on the recovered canvas.
#    - The recovered crop rectangle becomes an internal "stop frame": a segment is
#      allowed to reach the old border line, but once it touches that line it may
#      not keep extending further in the same outward direction.
# b) Complexity calibration:
#    - Only three crop patterns are used: top, left, or top-left corner.
#    - Arm length stays in {2, 3}; this keeps the edge truncation obvious while
#      still producing repeated spiral structure instead of tiny degenerate cases.
# c) Broken original assumptions:
#    - Original code assumed all four arms are fully visible. Here we infer the
#      true arm length from the longest visible arm and reconstruct the full canvas.
#    - Original code assumed output size equals input size. Here we restore a
#      canonical square: size = 4*arm + 2*ceil(arm/2) + 6, center = 2*arm + 2.
#    - Original border-fix post-processing relied on the real outer border being
#      the only stopping boundary; that no longer holds once an internal stop
#      frame exists, so those branches are removed entirely.
# d) Placement legality:
#    - The generator builds the full cross on the canonical canvas first, then
#      crops whole rows/cols from chosen sides. This guarantees no overlap, no
#      malformed cross, and a valid inverse padding rule for the solver.
# e) Visual observability:
#    - The visible side(s) on the cropped edge always keep 1..arm-1 teal cells,
#      so the truncation is obvious but the row/column of the center hole remains
#      recoverable from the maximum teal counts.
# f) Visual cleanliness:
#    - Only one teal object exists in the input. The output still uses a single
#      teal wall system; the new information is geometric (missing strips and
#      internal stop lines), not extra colors or overlapping markers.
# g) New solver assumptions and how the generator satisfies them:
#    - There is exactly one teal cross with a center hole. Generator guarantees it.
#    - At least one vertical arm and one horizontal arm remain visible. Generator
#      only crops top/left sides and never deletes the opposite full arms.
#    - The crop is a whole-strip crop from the canonical canvas. Generator builds
#      inputs exactly that way, so padding back to the canonical canvas is valid.


CROSS_COLOR = 8
ARM_LEN_MIN, ARM_LEN_MAX = 2, 3
MIN_FILL = 0.13
MIN_STOP_EVENTS = 5
MIN_INTERNAL_STOP_HITS = 3
MIN_OUTSIDE_TEAL = 6
MAX_OUTER_RETRIES = 400


def _full_size(arm_len: int) -> int:
    return 4 * arm_len + 2 * math.ceil(arm_len / 2) + 6


def _full_center(arm_len: int) -> int:
    return 2 * arm_len + 2


def _infer_cross(grid: np.ndarray) -> tuple[int, int, int, dict[tuple[int, int], int]]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    h, w = grid.shape

    def ray_len(direction: tuple[int, int]) -> int:
        r, c = cr, cc
        n = 0
        dr, dc = direction
        while 0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            n += 1
        return n

    visible = {direction: ray_len(direction) for direction in _DIRS_URDL}
    arm_len = max(visible.values())
    return cr, cc, arm_len, visible


def _build_full_cross(out: np.ndarray, center: int, arm_len: int) -> None:
    for k in range(1, arm_len + 1):
        out[center - k, center] = CROSS_COLOR
        out[center + k, center] = CROSS_COLOR
        out[center, center - k] = CROSS_COLOR
        out[center, center + k] = CROSS_COLOR


def _solve_variant(grid: np.ndarray, return_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    cr, cc, arm_len, visible = _infer_cross(grid)

    full_size = _full_size(arm_len)
    center = _full_center(arm_len)

    top_crop = center - cr
    left_crop = center - cc
    bottom_crop = (full_size - 1 - center) - (h - 1 - cr)
    right_crop = (full_size - 1 - center) - (w - 1 - cc)
    assert min(top_crop, left_crop, bottom_crop, right_crop) >= 0

    out = np.zeros((full_size, full_size), dtype=int)
    out[top_crop : top_crop + h, left_crop : left_crop + w] = grid
    _build_full_cross(out, center, arm_len)

    stop_frame = {
        "top": top_crop,
        "bottom": full_size - 1 - bottom_crop,
        "left": left_crop,
        "right": full_size - 1 - right_crop,
    }

    def inb(r: int, c: int) -> bool:
        return 0 <= r < full_size and 0 <= c < full_size

    def is_outer_border(r: int, c: int) -> bool:
        return r == 0 or r == full_size - 1 or c == 0 or c == full_size - 1

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def stop_block(cur: tuple[int, int], direction: tuple[int, int]) -> bool:
        r, c = cur
        if direction == (-1, 0):
            return r <= stop_frame["top"]
        if direction == (1, 0):
            return r >= stop_frame["bottom"]
        if direction == (0, -1):
            return c <= stop_frame["left"]
        return c >= stop_frame["right"]

    def stop_touch(r: int, c: int, direction: tuple[int, int]) -> bool:
        if direction == (-1, 0):
            return r == stop_frame["top"]
        if direction == (1, 0):
            return r == stop_frame["bottom"]
        if direction == (0, -1):
            return c == stop_frame["left"]
        return c == stop_frame["right"]

    def can_move(cur: tuple[int, int], direction: tuple[int, int], length: int) -> int:
        if stop_block(cur, direction):
            return 0
        prev = cur
        moved = 0
        for t in range(1, length + 1):
            rr = cur[0] + direction[0] * t
            cc2 = cur[1] + direction[1] * t
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                break
            if not spacing_ok((rr, cc2), prev, cur):
                break
            moved = t
            prev = (rr, cc2)
            if stop_touch(rr, cc2, direction):
                break
        return moved

    def do_move(agent: dict, direction: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr = r0 + direction[0] * t
            cc2 = c0 + direction[1] * t
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + direction[0] * length, c0 + direction[1] * length)

    agents = []
    for outward in _DIRS_URDL:
        agents.append(
            {
                "cur": (center + outward[0] * arm_len, center + outward[1] * arm_len),
                "dir": _TURN_RIGHT[outward],
                "out_dir": outward,
                "phase": "out_first" if arm_len == 1 else None,
                "done": False,
                "p": 0,
                "touched_stop": False,
            }
        )

    stop_events = 0

    def step_agent(agent: dict) -> bool:
        nonlocal stop_events
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            moved = can_move(agent["cur"], agent["out_dir"], 1)
            if moved:
                do_move(agent, agent["out_dir"], moved)
                if stop_touch(agent["cur"][0], agent["cur"][1], agent["out_dir"]):
                    stop_events += 1
                    agent["touched_stop"] = True
            agent["phase"] = None
            agent["dir"] = _TURN_RIGHT[agent["out_dir"]]
            agent["p"] = 0
            return True

        dirs = [_TURN_RIGHT[agent["dir"]], agent["dir"]]
        if not is_outer_border(*agent["cur"]):
            dirs.append(_TURN_LEFT[agent["dir"]])

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            moved = can_move(agent["cur"], nd, length)
            if moved == length and moved > 0:
                do_move(agent, nd, moved)
                if stop_touch(agent["cur"][0], agent["cur"][1], nd):
                    stop_events += 1
                    agent["touched_stop"] = True
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and moved == 1:
                do_move(agent, nd, 1)
                if stop_touch(agent["cur"][0], agent["cur"][1], nd):
                    stop_events += 1
                    agent["touched_stop"] = True
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for agent in agents:
            if step_agent(agent):
                changed = True
        if all(agent["done"] for agent in agents):
            break

    if not return_stats:
        return out

    outside_mask = np.ones_like(out, dtype=bool)
    outside_mask[top_crop : top_crop + h, left_crop : left_crop + w] = False
    outside_teal = int(np.sum((out == CROSS_COLOR) & outside_mask))

    frame_cells = set()
    for c in range(stop_frame["left"], stop_frame["right"] + 1):
        frame_cells.add((stop_frame["top"], c))
        frame_cells.add((stop_frame["bottom"], c))
    for r in range(stop_frame["top"], stop_frame["bottom"] + 1):
        frame_cells.add((r, stop_frame["left"]))
        frame_cells.add((r, stop_frame["right"]))
    internal_stop_hits = sum(
        1
        for r, c in frame_cells
        if 0 < r < full_size - 1 and 0 < c < full_size - 1 and out[r, c] == CROSS_COLOR
    )

    stats = {
        "arm_len": arm_len,
        "full_size": full_size,
        "input_shape": (h, w),
        "output_shape": tuple(out.shape),
        "crop": {
            "top": top_crop,
            "right": right_crop,
            "bottom": bottom_crop,
            "left": left_crop,
        },
        "stop_frame": stop_frame,
        "visible": visible,
        "truncated_sides": sum(v < arm_len for v in visible.values()),
        "padding_total": top_crop + right_crop + bottom_crop + left_crop,
        "stop_events": stop_events,
        "agents_touching_stop": sum(int(agent["touched_stop"]) for agent in agents),
        "fill_fraction": float(np.mean(out == CROSS_COLOR)),
        "outside_teal": outside_teal,
        "internal_stop_hits": internal_stop_hits,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a cropped teal cross into its full canvas, then grow spirals that stop at the old crop edge.

    Input structure:
    - Background is black (0).
    - A single teal (8) plus-sign with a center hole appears flush with the top edge, left edge, or top-left corner.
    - One or two arms are shortened by the crop, while the opposite arms remain fully visible.

    Rule:
    - Infer the true arm length from the longest visible arm and restore the canonical square canvas.
    - Rebuild the missing arm cells, then launch the four clockwise spiral turtles from the completed arm tips.
    - Turtle priority stays turn-right > straight > turn-left, with the same 2/1 step rhythm and 1-cell spacing rule.
    - The original input rectangle becomes an internal stop frame: a segment may reach that recovered border line, but it cannot keep extending farther in the same outward direction.

    Key insight:
    - The edge-truncated cross tells you how many rows/cols were cropped off; once those strips are restored, the only new wrinkle is that the old border survives as a directional stopping line inside the larger spiral canvas.
    """
    return _solve_variant(grid, return_stats=False)


def _build_input(arm_len: int, sides: tuple[str, ...], visible_override: dict[str, int] | None = None) -> np.ndarray:
    full_size = _full_size(arm_len)
    center = _full_center(arm_len)
    visible = {"top": arm_len, "bottom": arm_len, "left": arm_len, "right": arm_len}

    for side in sides:
        visible[side] = random.randint(1, arm_len - 1)
    if visible_override is not None:
        visible.update(visible_override)

    top_crop = center - visible["top"] if "top" in sides else 0
    left_crop = center - visible["left"] if "left" in sides else 0
    bottom_crop = (full_size - 1) - (center + visible["bottom"]) if "bottom" in sides else 0
    right_crop = (full_size - 1) - (center + visible["right"]) if "right" in sides else 0

    h = full_size - top_crop - bottom_crop
    w = full_size - left_crop - right_crop
    grid = np.zeros((h, w), dtype=int)
    cr = center - top_crop
    cc = center - left_crop

    for k in range(1, arm_len + 1):
        candidates = [
            (cr - k, cc),
            (cr + k, cc),
            (cr, cc - k),
            (cr, cc + k),
        ]
        for rr, cc2 in candidates:
            if 0 <= rr < h and 0 <= cc2 < w:
                grid[rr, cc2] = CROSS_COLOR

    return grid


def _gen_attempt() -> np.ndarray | None:
    arm_len = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    sides = random.choices(_SIDE_PATTERNS, weights=[3, 3, 2], k=1)[0]
    inp = _build_input(arm_len, sides)

    try:
        out, stats = _solve_variant(inp, return_stats=True)
    except Exception:
        return None

    if stats["truncated_sides"] < len(sides):
        return None
    if stats["padding_total"] <= 0:
        return None
    if stats["stop_events"] < MIN_STOP_EVENTS:
        return None
    if stats["internal_stop_hits"] < MIN_INTERNAL_STOP_HITS:
        return None
    if stats["outside_teal"] < MIN_OUTSIDE_TEAL:
        return None
    if stats["fill_fraction"] < MIN_FILL:
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")

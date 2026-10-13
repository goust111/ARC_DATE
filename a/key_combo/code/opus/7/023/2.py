from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
ANCHOR_COLORS = (1, 2, 3, 4)
STAIR_COLOR = 5
ALL_CORNERS = ("TL", "TR", "BL", "BR")
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_COUNT = 0


def _corner_info(height: int, width: int) -> dict[str, dict[str, int]]:
    return {
        "TL": {"r": 0, "c": 0, "vr": -1, "hc": -1},
        "TR": {"r": 0, "c": width - 1, "vr": -1, "hc": 1},
        "BL": {"r": height - 1, "c": 0, "vr": 1, "hc": -1},
        "BR": {"r": height - 1, "c": width - 1, "vr": 1, "hc": 1},
    }


def _extract_anchors(grid: np.ndarray) -> list[dict[str, int]]:
    anchors: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, STAIR_COLOR)):
        pts = np.argwhere(grid == color)
        assert pts.shape[0] == 1, "each anchor color must appear exactly once"
        r, c = map(int, pts[0])
        anchors.append({"color": color, "r": r, "c": c})
    assert 2 <= len(anchors) <= 4, "expected 2 to 4 colored anchors"
    return anchors


def _nearest_corner(height: int, width: int, r: int, c: int) -> dict[str, int | bool | str]:
    best: tuple[int, str, int, int, int, int] | None = None
    dists: list[int] = []
    for name, info in _corner_info(height, width).items():
        dist = abs(r - info["r"]) + abs(c - info["c"])
        dists.append(dist)
        cand = (dist, name, info["r"], info["c"], info["vr"], info["hc"])
        if best is None or cand < best:
            best = cand
    assert best is not None
    return {
        "step": int(best[0]),
        "corner": str(best[1]),
        "corner_r": int(best[2]),
        "corner_c": int(best[3]),
        "vr": int(best[4]),
        "hc": int(best[5]),
        "unique": dists.count(int(best[0])) == 1,
    }


def _segment_cells(r: int, c: int, axis: int, step: int, hc: int, vr: int) -> list[tuple[int, int]]:
    if axis == 0:
        return [(r, c + hc * k) for k in range(step)]
    return [(r + vr * k, c) for k in range(step)]


def _advance(r: int, c: int, axis: int, step: int, hc: int, vr: int) -> tuple[int, int]:
    if axis == 0:
        return r, c + hc * step
    return r + vr * step, c


def _simulate(
    grid: np.ndarray,
    *,
    step_override: int | None = None,
    repeat: bool = True,
    block_gray: bool = True,
    sort_mode: str = "step",
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()
    gray = np.zeros((height, width), dtype=bool)
    anchors = _extract_anchors(g)
    anchor_cells = {(a["r"], a["c"]) for a in anchors}

    enriched: list[dict[str, object]] = []
    for anchor in anchors:
        info = _nearest_corner(height, width, anchor["r"], anchor["c"])
        step = int(info["step"] if step_override is None else step_override)
        enriched.append({**anchor, **info, "step": step})

    if sort_mode == "step":
        enriched.sort(key=lambda a: (int(a["step"]), int(a["r"]), int(a["c"])))
    elif sort_mode == "rowcol":
        enriched.sort(key=lambda a: (int(a["r"]), int(a["c"]), int(a["step"])))
    else:
        raise ValueError("unknown sort_mode")

    meta: list[dict[str, object]] = []
    for anchor in enriched:
        r0 = int(anchor["r"])
        c0 = int(anchor["c"])
        step = int(anchor["step"])
        own = (r0, c0)
        anchor_meta = {
            "color": int(anchor["color"]),
            "r": r0,
            "c": c0,
            "step": step,
            "corner": str(anchor["corner"]),
            "arms": [],
        }
        for arm_name, hc, vr in (
            ("toward", int(anchor["hc"]), int(anchor["vr"])),
            ("away", -int(anchor["hc"]), -int(anchor["vr"])),
        ):
            r = r0
            c = c0
            axis = 0
            segments = 0
            pixels = 0
            stop_reason = "cycle"
            limit = None if repeat else 2
            while True:
                if limit is not None and segments >= limit:
                    break
                cells = _segment_cells(r, c, axis, step, hc, vr)
                blocked = None
                for rr, cc in cells:
                    if not (0 <= rr < height and 0 <= cc < width):
                        blocked = "boundary"
                        break
                    if (rr, cc) != own and (rr, cc) in anchor_cells:
                        blocked = "anchor"
                        break
                    if block_gray and gray[rr, cc]:
                        blocked = "gray"
                        break
                if blocked is not None:
                    stop_reason = blocked
                    break
                for rr, cc in cells:
                    if (rr, cc) == own:
                        continue
                    out[rr, cc] = STAIR_COLOR
                    gray[rr, cc] = True
                    pixels += 1
                r, c = _advance(r, c, axis, step, hc, vr)
                axis = 1 - axis
                segments += 1
            anchor_meta["arms"].append(
                {
                    "name": arm_name,
                    "segments": segments,
                    "pixels": pixels,
                    "stop_reason": stop_reason,
                }
            )
        meta.append(anchor_meta)
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow gray periodic staircases whose step sizes come from anchor-to-corner distances.

    Input structure:
    - The grid is black except for 2 to 4 single-pixel colored anchors.
    - Anchor colors are distinct and remain in place in the output.
    - Each anchor lies inside the grid and is closest to one canvas corner.

    Rule:
    - For each anchor, compute S as its Manhattan distance to the nearest corner.
    - Process anchors in increasing S, then by row, then by column.
    - For the nearest-corner diagonal pair, draw two gray staircase arms: one toward that corner and one in the opposite direction.
    - Each arm repeats a cycle of an S-pixel horizontal segment followed by an S-pixel vertical segment.
    - Stop an arm before the next segment if that full segment would leave the grid or overlap gray pixels already drawn by earlier processing.

    Key insight:
    - The anchor position encodes the staircase step size through nearest-corner distance, and that step size changes where later staircases are blocked by earlier ones.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _candidate_positions(height: int, width: int, corner: str, step: int) -> list[tuple[int, int, int]]:
    info = _corner_info(height, width)[corner]
    out: list[tuple[int, int, int]] = []
    for vertical_offset in range(1, step):
        horizontal_offset = step - vertical_offset
        r = info["r"] - info["vr"] * vertical_offset
        c = info["c"] - info["hc"] * horizontal_offset
        if 0 < r < height - 1 and 0 < c < width - 1:
            out.append((r, c, vertical_offset))
    return out


def _choose_corners(mode: str) -> list[str]:
    if mode == "collision":
        pair = random.choice([("TL", "BR"), ("TR", "BL")])
        count = random.choice([3, 4])
        others = [corner for corner in ALL_CORNERS if corner not in pair]
        return list(pair) + random.sample(others, count - 2)
    if mode == "open":
        return random.choice(
            [
                ["TL", "TR"],
                ["BL", "BR"],
                ["TL", "BL"],
                ["TR", "BR"],
                ["TL", "TR", "BL"],
                ["TL", "TR", "BR"],
                ["TL", "BL", "BR"],
                ["TR", "BL", "BR"],
            ]
        )
    return random.sample(ALL_CORNERS, random.choice([2, 3, 4]))


def _pick_scene(mode: str) -> np.ndarray | None:
    if mode == "open":
        height = random.randint(20, 26)
        width = random.randint(20, 26)
    elif mode == "collision":
        height = random.randint(18, 23)
        width = random.randint(18, 23)
    else:
        height = random.randint(18, 25)
        width = random.randint(18, 25)

    corners = _choose_corners(mode)
    steps = [random.randint(2, 5) for _ in corners]
    if len(set(steps)) < 2:
        return None

    visible_corner = random.randrange(len(corners))
    placements: list[tuple[int, int, str, int, int]] = []
    used_cells: set[tuple[int, int]] = set()
    for idx, (corner, step) in enumerate(zip(corners, steps)):
        candidates = _candidate_positions(height, width, corner, step)
        if idx == visible_corner:
            candidates = [cand for cand in candidates if cand[2] == 1] or candidates
        random.shuffle(candidates)
        placed = None
        for r, c, vertical_offset in candidates:
            if (r, c) in used_cells:
                continue
            if any(abs(r - rr) + abs(c - cc) < 6 for rr, cc, *_ in placements):
                continue
            placed = (r, c, corner, step, vertical_offset)
            break
        if placed is None:
            return None
        placements.append(placed)
        used_cells.add((placed[0], placed[1]))

    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(list(ANCHOR_COLORS), len(placements))
    random.shuffle(colors)
    for color, (r, c, _corner, _step, _voff) in zip(colors, placements):
        grid[r, c] = color
    return grid


def _semantic_ok(grid: np.ndarray, mode: str) -> bool:
    anchors = _extract_anchors(grid)
    height, width = grid.shape

    corners = []
    steps = []
    for anchor in anchors:
        info = _nearest_corner(height, width, anchor["r"], anchor["c"])
        if not bool(info["unique"]):
            return False
        if not (2 <= int(info["step"]) <= 5):
            return False
        corners.append(str(info["corner"]))
        steps.append(int(info["step"]))

    if len(set(corners)) < 2:
        return False
    if len(set(steps)) < 2:
        return False

    for a, b in combinations(anchors, 2):
        if abs(int(a["r"]) - int(b["r"])) + abs(int(a["c"]) - int(b["c"])) < 6:
            return False

    out, meta = _simulate(grid)
    gray_count = int(np.count_nonzero(out == STAIR_COLOR))
    if gray_count < 14 or gray_count > grid.size // 3:
        return False

    toward_visible = sum(int(anchor["arms"][0]["segments"]) > 0 for anchor in meta)
    if toward_visible < 1:
        return False

    all_arms = [arm for anchor in meta for arm in anchor["arms"]]
    gray_stops = sum(arm["stop_reason"] == "gray" for arm in all_arms)
    boundary_stops = sum(arm["stop_reason"] == "boundary" for arm in all_arms)
    long_arms = sum(int(arm["segments"]) >= 4 for arm in all_arms)
    if boundary_stops == 0 or long_arms == 0:
        return False

    if mode == "open":
        if gray_stops == 0:
            return False
        if not np.array_equal(_simulate(grid, sort_mode="rowcol")[0], out):
            return False
    elif mode == "collision":
        if gray_stops == 0:
            return False
        if np.array_equal(_simulate(grid, sort_mode="rowcol")[0], out):
            return False
    else:
        if gray_stops < 2 or len(anchors) < 3:
            return False

    if np.array_equal(_simulate(grid, step_override=3)[0], out):
        return False
    if np.array_equal(_simulate(grid, repeat=False)[0], out):
        return False
    if mode != "open" and np.array_equal(_simulate(grid, block_gray=False)[0], out):
        return False

    if any(int(out[anchor["r"], anchor["c"]]) != int(anchor["color"]) for anchor in anchors):
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = ("collision", "open", "mixed")[_GEN_CALL_COUNT % 3]
    _GEN_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _pick_scene(mode)
        if grid is None:
            continue
        if _semantic_ok(grid, mode):
            return grid
    raise RuntimeError(f"failed to generate a valid {mode} scene")

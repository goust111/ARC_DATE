from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MARKER = 1
PLATFORM = 5
OTHER_COLORS = [2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _edge_marker_side(grid: np.ndarray) -> str:
    h, w = grid.shape
    candidates: list[str] = []
    if np.all(grid[0, :] == MARKER):
        candidates.append("top")
    if np.all(grid[:, w - 1] == MARKER):
        candidates.append("right")
    if np.all(grid[h - 1, :] == MARKER):
        candidates.append("bottom")
    if np.all(grid[:, 0] == MARKER):
        candidates.append("left")
    assert len(candidates) == 1, candidates
    return candidates[0]


def _initial_direction(side: str) -> tuple[int, int]:
    return {
        "top": DIRS["down"],
        "right": DIRS["left"],
        "bottom": DIRS["up"],
        "left": DIRS["right"],
    }[side]


def _distance_from_marker(side: str, r: int, c: int, shape: tuple[int, int]) -> int:
    h, w = shape
    if side == "top":
        return r
    if side == "right":
        return w - 1 - c
    if side == "bottom":
        return h - 1 - r
    return c


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    coords = set(map(tuple, np.argwhere(mask)))
    parts: list[np.ndarray] = []
    while coords:
        seed = coords.pop()
        queue = deque([seed])
        comp = [seed]
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) in coords:
                    coords.remove((nr, nc))
                    queue.append((nr, nc))
                    comp.append((nr, nc))
        parts.append(np.array(comp, dtype=int))
    return parts


def _door_cells(center: tuple[int, int], opening: str) -> set[tuple[int, int]]:
    r, c = center
    cells = {
        "up": {(r + 1, c), (r, c - 1), (r, c + 1)},
        "right": {(r - 1, c), (r + 1, c), (r, c - 1)},
        "down": {(r - 1, c), (r, c - 1), (r, c + 1)},
        "left": {(r - 1, c), (r + 1, c), (r, c + 1)},
    }[opening]
    return cells


def _infer_door(coords: np.ndarray) -> dict:
    pts = {tuple(map(int, p)) for p in coords}
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    found = []
    for cr in range(min(rs) - 1, max(rs) + 2):
        for cc in range(min(cs) - 1, max(cs) + 2):
            present = {name for name, (dr, dc) in DIRS.items() if (cr + dr, cc + dc) in pts}
            if len(present) != 3:
                continue
            if _door_cells((cr, cc), next(d for d in DIR_ORDER if d not in present)) == pts:
                opening = next(d for d in DIR_ORDER if d not in present)
                found.append({"center": (cr, cc), "opening": opening, "cells": pts})
    assert len(found) == 1, found
    return found[0]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BG
    side = _edge_marker_side(g)

    colors = sorted(set(map(int, g.ravel())) - {BG, MARKER, PLATFORM})
    portal_colors = [c for c in colors if int((g == c).sum()) == 6]
    stone_colors = [c for c in colors if int((g == c).sum()) == 1]
    assert portal_colors
    assert stone_colors

    portals: dict[int, list[dict]] = {}
    portal_hit: dict[tuple[int, int], tuple[int, int]] = {}
    portal_mask = np.zeros_like(g, dtype=bool)
    for color in portal_colors:
        parts = _components8(g == color)
        assert len(parts) == 2
        doors: list[dict] = []
        for comp in parts:
            assert comp.shape[0] == 3
            door = _infer_door(comp)
            door["color"] = color
            doors.append(door)
        portals[color] = doors
        for idx, door in enumerate(doors):
            for cell in door["cells"]:
                portal_hit[cell] = (color, idx)
                portal_mask[cell] = True

    stones = []
    for color in stone_colors:
        rc = np.argwhere(g == color)
        assert rc.shape == (1, 2)
        r, c = map(int, rc[0])
        stones.append({"color": color, "start": (r, c)})

    static_mask = (g != BG) & (~portal_mask)
    for stone in stones:
        r, c = stone["start"]
        static_mask[r, c] = False

    return {
        "grid": g,
        "shape": (h, w),
        "side": side,
        "init_dir": _initial_direction(side),
        "portals": portals,
        "portal_hit": portal_hit,
        "portal_mask": portal_mask,
        "stones": stones,
        "static_mask": static_mask,
    }


def _trace_stone(
    scene: dict,
    stone: dict,
    settled_mask: np.ndarray,
    *,
    use_settled: bool = True,
    rewrite_exit: bool = True,
) -> dict:
    h, w = scene["shape"]
    blocked = scene["static_mask"].copy()
    if use_settled:
        blocked |= settled_mask

    pos = stone["start"]
    direction = scene["init_dir"]
    teleports = 0
    history = [pos]
    seen: set[tuple[int, int, int, int]] = set()

    limit = h * w * 8
    for _ in range(limit):
        state = (pos[0], pos[1], direction[0], direction[1])
        assert state not in seen, "non-terminating portal loop"
        seen.add(state)

        nr = pos[0] + direction[0]
        nc = pos[1] + direction[1]
        if not _in_bounds(scene["shape"], nr, nc):
            return {"final": pos, "teleports": teleports, "history": history}
        if blocked[nr, nc]:
            return {"final": pos, "teleports": teleports, "history": history}

        hit = scene["portal_hit"].get((nr, nc))
        if hit is not None:
            color, in_idx = hit
            out_door = scene["portals"][color][1 - in_idx]
            exit_center = tuple(out_door["center"])
            assert not blocked[exit_center], "exit center must stay free"
            pos = exit_center
            if rewrite_exit:
                direction = DIRS[out_door["opening"]]
            teleports += 1
            history.append(pos)
            continue

        pos = (nr, nc)
        history.append(pos)

    raise AssertionError("trace exceeded step limit")


def _resolve_scene(
    scene: dict,
    *,
    use_settled: bool = True,
    rewrite_exit: bool = True,
    nearest_first: bool = True,
) -> list[dict]:
    stones = list(scene["stones"])
    stones.sort(
        key=lambda s: (
            _distance_from_marker(scene["side"], s["start"][0], s["start"][1], scene["shape"]),
            s["start"][0],
            s["start"][1],
        ),
        reverse=not nearest_first,
    )

    settled_mask = np.zeros(scene["shape"], dtype=bool)
    resolved = []
    for stone in stones:
        trace = _trace_stone(
            scene,
            stone,
            settled_mask,
            use_settled=use_settled,
            rewrite_exit=rewrite_exit,
        )
        fr, fc = trace["final"]
        assert not settled_mask[fr, fc], "final positions must be unique"
        settled_mask[fr, fc] = True
        resolved.append(
            {
                "color": stone["color"],
                "start": stone["start"],
                "final": trace["final"],
                "teleports": trace["teleports"],
                "history": trace["history"],
            }
        )
    return resolved


def _render_output_from_scene(scene: dict, resolved: list[dict]) -> np.ndarray:
    out = scene["grid"].copy()
    for stone in scene["stones"]:
        r, c = stone["start"]
        out[r, c] = BG
    for stone in resolved:
        r, c = stone["final"]
        out[r, c] = stone["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Trace each falling pixel through color-matched cup portals, redirecting at every exit until it settles.

    Input structure:
    - One complete outer edge is filled with a marker line that indicates the initial inward motion direction.
    - The field also contains grey platform cells, several portal-color pairs of three-cell U-shaped cup gates, and several single-cell falling objects.
    - Each portal color appears in exactly two three-cell gates; each falling object has its own color, different from the marker, platforms, and portals.

    Rule:
    - Resolve falling objects in order of increasing distance from the marked edge.
    - Each object moves one cell at a time in its current direction. If the next cell is a portal cell, the object teleports to the center of the other gate of the same color and its direction is replaced by that exit gate's opening direction.
    - If the next cell is blocked by a platform, a previously settled object, or any other fixed occupied cell, or if it would leave the grid, the object stops on its previous cell.
    - The output keeps the marker, platforms, and portals unchanged, and draws each falling object only at its final settled position.

    Key insight:
    - Portal entry only selects the color-matched exit; the new direction comes solely from the exit gate's opening, so the full answer requires ordered step-by-step tracing through chained redirects.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    resolved = _resolve_scene(scene)
    return _render_output_from_scene(scene, resolved)


def _place_marker(grid: np.ndarray) -> None:
    grid[0, :] = MARKER


def _paint_door(grid: np.ndarray, door: dict) -> None:
    for r, c in _door_cells(door["center"], door["opening"]):
        grid[r, c] = door["color"]


def _rotate_opening(opening: str, k: int) -> str:
    idx = DIR_ORDER.index(opening)
    return DIR_ORDER[(idx + k) % 4]


def _rotate_scene(grid: np.ndarray, k: int) -> np.ndarray:
    if k % 4 == 0:
        return grid
    return np.rot90(grid, -k)


def _build_candidate() -> np.ndarray | None:
    h = random.randint(23, 26)
    w = random.randint(20, 26)
    base_h = 22
    base_w = 18
    if h < base_h or w < base_w + 2:
        return None

    stone_count = random.choice([3, 4])
    pool = random.sample(OTHER_COLORS, 2 + stone_count)
    portal_colors = pool[:2]
    stone_colors = pool[2:]

    grid = np.full((h, w), BG, dtype=int)
    _place_marker(grid)

    shift_r = random.randint(0, h - base_h)
    shift_c = random.randint(1, w - base_w - 1)
    mirror = random.random() < 0.5

    def tx(r: int, c: int) -> tuple[int, int]:
        cc = base_w - 1 - c if mirror else c
        return shift_r + r, shift_c + cc

    def topen(opening: str) -> str:
        if not mirror:
            return opening
        if opening == "left":
            return "right"
        if opening == "right":
            return "left"
        return opening

    doors = [
        {"center": tx(10, 4), "opening": topen(random.choice(DIR_ORDER)), "color": portal_colors[0]},
        {"center": tx(12, 7), "opening": topen("right"), "color": portal_colors[0]},
        {"center": tx(12, 13), "opening": topen(random.choice(DIR_ORDER)), "color": portal_colors[1]},
        {"center": tx(15, 10), "opening": "down", "color": portal_colors[1]},
    ]
    platforms = [
        {tx(19, c) for c in range(9, 12)},
        {tx(15, c) for c in range(0, 3)},
    ]
    starts: list[tuple[int, int, int]] = [
        (*tx(2, 4), stone_colors[0]),
        (*tx(5, 4), stone_colors[1]),
        (*tx(8, 1), stone_colors[2]),
    ]
    if stone_count == 4:
        platforms.append({tx(17, c) for c in range(14, 17)})
        starts.append((*tx(10, 16), stone_colors[3]))

    if random.random() < 0.7:
        extra_len = random.randint(2, 4)
        extra_c0 = random.randint(8, 11)
        platforms.append({tx(6, c) for c in range(extra_c0, extra_c0 + extra_len)})

    door_cells = set()
    centers = set()
    for door in doors:
        centers.add(door["center"])
        cells = _door_cells(door["center"], door["opening"])
        if cells & door_cells:
            return None
        if door["center"] in door_cells:
            return None
        for r, c in cells:
            if not (1 <= r < h and 1 <= c < w - 1):
                return None
        door_cells |= cells
    if centers & door_cells:
        return None

    all_platforms = set()
    for platform in platforms:
        if any(not _in_bounds((h, w), r, c) for r, c in platform):
            return None
        all_platforms |= platform
    if all_platforms & door_cells:
        return None
    if all_platforms & centers:
        return None

    for r, c, _color in starts:
        if not (1 <= r < h and 1 <= c < w - 1):
            return None
        if (r, c) in door_cells or (r, c) in centers or (r, c) in all_platforms:
            return None

    for door in doors:
        _paint_door(grid, door)
    for r, c in all_platforms:
        grid[r, c] = PLATFORM
    for r, c, color in starts:
        grid[r, c] = color

    return _rotate_scene(grid, random.randint(0, 3))


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        resolved = _resolve_scene(scene)
        out = _render_output_from_scene(scene, resolved)
        if not np.array_equal(main(grid), out):
            return False

        dists = [
            _distance_from_marker(scene["side"], stone["start"][0], stone["start"][1], scene["shape"])
            for stone in scene["stones"]
        ]
        if len(set(dists)) != len(dists):
            return False

        if max(stone["teleports"] for stone in resolved) < 2:
            return False

        finals = {stone["final"] for stone in resolved}
        if len(finals) != len(resolved):
            return False

        def alt_output(**kwargs) -> np.ndarray | None:
            try:
                return _render_output_from_scene(scene, _resolve_scene(scene, **kwargs))
            except AssertionError:
                return None

        alt_no_rewrite = alt_output(rewrite_exit=False)
        alt_no_settle = alt_output(use_settled=False)
        alt_reverse = alt_output(nearest_first=False)

        if alt_no_rewrite is not None and np.array_equal(out, alt_no_rewrite):
            return False
        if alt_no_settle is not None and np.array_equal(out, alt_no_settle):
            return False
        if alt_reverse is not None and np.array_equal(out, alt_reverse):
            return False

        changed = sum(stone["start"] != stone["final"] for stone in resolved)
        if changed < 2:
            return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_candidate()
        if grid is not None and _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate valid slot-gate puzzle input")

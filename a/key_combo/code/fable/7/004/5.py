from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


GREY = 5
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ENTRY_SIDE_BY_DIR = {
    (-1, 0): "bottom",
    (1, 0): "top",
    (0, -1): "right",
    (0, 1): "left",
}
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIZE_RANGE = (22, 30)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
LINK_DIST_RANGE = (3, 8)
GEN_RETRIES = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _make_frame(r0: int, c0: int, h: int, w: int, side: str, offset: int) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side == "top":
        gap = (r0, c0 + offset)
    elif side == "bottom":
        gap = (r1, c0 + offset)
    elif side == "left":
        gap = (r0 + offset, c0)
    else:
        gap = (r0 + offset, c1)
    frame = {
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "offset": int(offset),
        "gap": gap,
        "dir": DIR_BY_SIDE[side],
    }
    frame["walls"] = set(_frame_wall_cells(frame))
    return frame


def _valid_offsets(h: int, w: int, side: str) -> list[int]:
    span = w if side in ("top", "bottom") else h
    return list(range(1, span - 1))


def _frame_gap(frame: dict) -> tuple[int, int]:
    return tuple(frame["gap"])


def _frame_wall_cells(frame: dict):
    r0, c0, r1, c1 = frame["bbox"]
    gap = _frame_gap(frame)
    for c in range(c0, c1 + 1):
        if (r0, c) != gap:
            yield (r0, c)
        if (r1, c) != gap:
            yield (r1, c)
    for r in range(r0 + 1, r1):
        if (r, c0) != gap:
            yield (r, c0)
        if (r, c1) != gap:
            yield (r, c1)


def _frame_interior(frame: dict):
    r0, c0, r1, c1 = frame["bbox"]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            yield (r, c)


def _inside_bbox(cell: tuple[int, int], frame: dict) -> bool:
    r, c = cell
    r0, c0, r1, c1 = frame["bbox"]
    return r0 <= r <= r1 and c0 <= c <= c1


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _path_cells(start: tuple[int, int], direction: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        if start[1] != end[1] or (end[0] - start[0]) * dr < 0:
            raise ValueError("end is not reachable from start along the given vertical ray")
    else:
        if start[0] != end[0] or (end[1] - start[1]) * dc < 0:
            raise ValueError("end is not reachable from start along the given horizontal ray")
    r, c = start
    out = []
    while True:
        out.append((r, c))
        if (r, c) == end:
            return out
        r += dr
        c += dc


def _ray_to_edge(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    out = []
    while in_bounds(shape, r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def _parse_frame_component(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    if len(coords) < 8:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    cells = {tuple(map(int, p)) for p in coords}
    if not cells.issubset(border):
        return None
    if len(cells) != len(border) - 1:
        return None
    gap_candidates = [cell for cell in border if cell not in cells]
    if len(gap_candidates) != 1:
        return None
    gap = gap_candidates[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if gap[0] == r0:
        side = "top"
        offset = gap[1] - c0
    elif gap[0] == r1:
        side = "bottom"
        offset = gap[1] - c0
    elif gap[1] == c0:
        side = "left"
        offset = gap[0] - r0
    else:
        side = "right"
        offset = gap[0] - r0
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) == GREY:
                return None
    return _make_frame(r0, c0, h, w, side, offset)


def _detect_case(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    other_colors = sorted(int(v) for v in np.unique(g) if int(v) not in (bg, GREY))
    assert len(other_colors) == 1, "expected exactly one seed color"
    seed_color = other_colors[0]
    seed_pos = np.argwhere(g == seed_color)
    assert len(seed_pos) == 1, "expected exactly one seed pixel"
    seed_r, seed_c = map(int, seed_pos[0])

    frames: list[dict] = []
    obstacle_cells: set[tuple[int, int]] = set()
    for comp in components_mask(g == GREY, connectivity=4):
        frame = _parse_frame_component(g, comp)
        if frame is None:
            obstacle_cells.update(tuple(map(int, p)) for p in comp)
        else:
            frames.append(frame)

    assert frames, "expected at least one frame"
    start_candidates = [
        idx
        for idx, frame in enumerate(frames)
        if frame["r0"] < seed_r < frame["bbox"][2] and frame["c0"] < seed_c < frame["bbox"][3]
    ]
    assert len(start_candidates) == 1, "expected seed inside exactly one frame"
    return frames, obstacle_cells, start_candidates[0], seed_color


def _simulate_chain(
    grid: np.ndarray,
    frames: list[dict],
    start_idx: int,
    fill_color: int,
    *,
    return_filled: bool = False,
):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    wall_owner: dict[tuple[int, int], int] = {}
    gap_owner: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["walls"]:
            wall_owner[cell] = idx
        gap_owner[_frame_gap(frame)] = idx

    filled = [False] * len(frames)

    def cast_from(frame_idx: int) -> None:
        r, c = _frame_gap(frames[frame_idx])
        dr, dc = frames[frame_idx]["dir"]
        while in_bounds(g.shape, r, c):
            gap_idx = gap_owner.get((r, c))
            if gap_idx is not None and gap_idx != frame_idx and not filled[gap_idx]:
                out[r, c] = fill_color
                trigger(gap_idx)
                return
            if int(g[r, c]) == GREY:
                hit_idx = wall_owner.get((r, c))
                if hit_idx is not None and not filled[hit_idx]:
                    trigger(hit_idx)
                return
            out[r, c] = fill_color
            r += dr
            c += dc

    def trigger(frame_idx: int) -> None:
        if filled[frame_idx]:
            return
        frame = frames[frame_idx]
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        filled[frame_idx] = True
        cast_from(frame_idx)

    trigger(start_idx)
    if return_filled:
        return out, {idx for idx, state in enumerate(filled) if state}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the seeded frame, then propagate the same color along gap-directed rays that infect downstream frames.

    Input structure:
    - The background is the dominant color.
    - Several grey one-cell-thick hollow rectangular frames are present; each frame border has exactly one one-cell gap.
    - Exactly one frame contains a single colored seed pixel in its interior, and some isolated grey obstacle pixels may also appear.

    Rule:
    - The seeded frame fills its whole interior with the seed color, then emits a ray from its gap outward along the normal of that side.
    - The ray colors every background cell it traverses until it hits a grey pixel or leaves the grid.
    - If the ray hits an unfilled frame, either by striking a wall cell or by entering through that frame's gap, that frame becomes fully filled and emits its own ray from its own gap.
    - Already filled frames never trigger again; obstacle pixels simply stop the ray.

    Key insight:
    - Each frame's unique gap is both its identity and its outgoing direction, so the solution is a chain-reaction simulation over frame geometry rather than a local fill.
    """
    g = np.asarray(grid, dtype=int)
    frames, _obstacles, start_idx, fill_color = _detect_case(g)
    return _simulate_chain(g, frames, start_idx, fill_color)


def _clear_corridor(
    start: tuple[int, int],
    direction: tuple[int, int],
    end: tuple[int, int],
    frames: list[dict],
    blocked_cells: set[tuple[int, int]] | None = None,
) -> bool:
    blocked_cells = blocked_cells or set()
    for cell in _path_cells(start, direction, end)[1:-1]:
        if cell in blocked_cells:
            return False
        if any(_inside_bbox(cell, frame) for frame in frames):
            return False
    return True


def _terminal_options(frame: dict, frames: list[dict], shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    out = []
    for side in SIDES:
        for offset in _valid_offsets(frame["h"], frame["w"], side):
            candidate = _make_frame(frame["r0"], frame["c0"], frame["h"], frame["w"], side, offset)
            ray = _ray_to_edge(candidate["gap"], candidate["dir"], (h, w))
            if all(not any(_inside_bbox(cell, other) for other in frames if other is not frame) for cell in ray[1:]):
                out.append(candidate)
    return out


def _place_first_frame(h: int, w: int) -> dict | None:
    for _ in range(120):
        fh = random.randint(*FRAME_H_RANGE)
        fw = random.randint(*FRAME_W_RANGE)
        if fh >= h or fw >= w:
            continue
        side = random.choice(SIDES)
        offset = random.choice(_valid_offsets(fh, fw, side))
        r0 = random.randint(0, h - fh)
        c0 = random.randint(0, w - fw)
        return _make_frame(r0, c0, fh, fw, side, offset)
    return None


def _place_next_frame(
    shape: tuple[int, int],
    prev: dict,
    frames: list[dict],
    link_mode: str,
    *,
    terminal_clear: bool = False,
) -> dict | None:
    h_grid, w_grid = shape
    gap_r, gap_c = prev["gap"]
    dr, dc = prev["dir"]
    facing_side = ENTRY_SIDE_BY_DIR[(dr, dc)]

    for _ in range(300):
        fh = random.randint(*FRAME_H_RANGE)
        fw = random.randint(*FRAME_W_RANGE)
        dist = random.randint(*LINK_DIST_RANGE)

        if (dr, dc) == (0, 1):
            c0 = gap_c + dist
            if c0 + fw > w_grid:
                continue
            lo = max(0, gap_r - (fh - 2))
            hi = min(h_grid - fh, gap_r - 1)
            if lo > hi:
                continue
            r0 = random.randint(lo, hi)
            contact = (gap_r, c0)
        elif (dr, dc) == (0, -1):
            c1 = gap_c - dist
            c0 = c1 - fw + 1
            if c0 < 0:
                continue
            lo = max(0, gap_r - (fh - 2))
            hi = min(h_grid - fh, gap_r - 1)
            if lo > hi:
                continue
            r0 = random.randint(lo, hi)
            contact = (gap_r, c0 + fw - 1)
        elif (dr, dc) == (1, 0):
            r0 = gap_r + dist
            if r0 + fh > h_grid:
                continue
            lo = max(0, gap_c - (fw - 2))
            hi = min(w_grid - fw, gap_c - 1)
            if lo > hi:
                continue
            c0 = random.randint(lo, hi)
            contact = (r0, gap_c)
        else:
            r1 = gap_r - dist
            r0 = r1 - fh + 1
            if r0 < 0:
                continue
            lo = max(0, gap_c - (fw - 2))
            hi = min(w_grid - fw, gap_c - 1)
            if lo > hi:
                continue
            c0 = random.randint(lo, hi)
            contact = (r0 + fh - 1, gap_c)

        base_bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if not all(_boxes_clear(base_bbox, frame["bbox"], pad=1) for frame in frames):
            continue
        if not _clear_corridor(prev["gap"], prev["dir"], contact, frames):
            continue

        candidates: list[dict] = []
        if link_mode == "gap":
            if facing_side in ("left", "right"):
                offset = contact[0] - r0
            else:
                offset = contact[1] - c0
            if offset in _valid_offsets(fh, fw, facing_side):
                candidates.append(_make_frame(r0, c0, fh, fw, facing_side, offset))
        else:
            for side in SIDES:
                for offset in _valid_offsets(fh, fw, side):
                    if side == facing_side:
                        gap = _make_frame(r0, c0, fh, fw, side, offset)["gap"]
                        if gap == contact:
                            continue
                    candidates.append(_make_frame(r0, c0, fh, fw, side, offset))

        if terminal_clear:
            candidates = [
                candidate
                for candidate in candidates
                if all(
                    not any(_inside_bbox(cell, other) for other in frames)
                    for cell in _ray_to_edge(candidate["gap"], candidate["dir"], shape)[1:]
                )
            ]

        if candidates:
            return random.choice(candidates)
    return None


def _place_blocking_obstacle(
    shape: tuple[int, int],
    prev: dict,
    blocked_frame: dict,
    obstacles: set[tuple[int, int]],
    frames: list[dict],
) -> tuple[int, int] | None:
    path = _path_cells(prev["gap"], prev["dir"], blocked_frame["gap"] if blocked_frame["gap"] in _path_cells(prev["gap"], prev["dir"], blocked_frame["gap"]) else next(iter(blocked_frame["walls"])))
    mids = path[1:-1]
    random.shuffle(mids)
    for cell in mids:
        r, c = cell
        if not in_bounds(shape, r, c):
            continue
        if any(_inside_bbox(cell, frame) for frame in frames):
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in obstacles):
            continue
        return cell
    return None


def _render_input(case: dict) -> np.ndarray:
    grid = np.full(case["shape"], case["bg"], dtype=int)
    for frame in case["frames"]:
        for r, c in frame["walls"]:
            grid[r, c] = GREY
    for r, c in case["obstacles"]:
        grid[r, c] = GREY
    seed_r, seed_c = case["seed_pos"]
    grid[seed_r, seed_c] = case["seed_color"]
    return grid


def _free_obstacle_cells(
    shape: tuple[int, int],
    frames: list[dict],
    obstacles: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in forbidden or any(_inside_bbox(cell, frame) for frame in frames):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in obstacles):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for frame in frames for rr, cc in frame["walls"]):
                continue
            out.append(cell)
    return out


def _generate_case() -> dict | None:
    shape = (random.randint(*GRID_SIZE_RANGE), random.randint(*GRID_SIZE_RANGE))
    seed_color = random.choice(SEED_COLORS)
    active_len = random.randint(2, 4)
    final_gap_hit = active_len >= 2 and random.random() < 0.4
    blocked_tail = (not final_gap_hit) and random.random() < 0.5

    first = _place_first_frame(*shape)
    if first is None:
        return None
    frames = [first]
    planned_filled = {0}

    for idx in range(1, active_len):
        is_last_active = idx == active_len - 1
        link_mode = "gap" if final_gap_hit and is_last_active else "wall"
        terminal_clear = is_last_active and not blocked_tail and link_mode == "wall"
        nxt = _place_next_frame(shape, frames[-1], frames, link_mode, terminal_clear=terminal_clear)
        if nxt is None:
            return None
        frames.append(nxt)
        planned_filled.add(idx)

    obstacles: set[tuple[int, int]] = set()
    if blocked_tail:
        blocked_mode = random.choice(("wall", "gap"))
        tail = _place_next_frame(shape, frames[-1], frames, blocked_mode, terminal_clear=False)
        if tail is None:
            return None
        gap_r, gap_c = frames[-1]["gap"]
        dr, dc = frames[-1]["dir"]
        if blocked_mode == "gap":
            contact = tail["gap"]
        elif (dr, dc) == (0, 1):
            contact = (gap_r, tail["c0"])
        elif (dr, dc) == (0, -1):
            contact = (gap_r, tail["bbox"][3])
        elif (dr, dc) == (1, 0):
            contact = (tail["r0"], gap_c)
        else:
            contact = (tail["bbox"][2], gap_c)
        obstacle_path = _path_cells(frames[-1]["gap"], frames[-1]["dir"], contact)
        candidates = obstacle_path[1:-1]
        random.shuffle(candidates)
        chosen = None
        for cell in candidates:
            if any(_inside_bbox(cell, frame) for frame in frames + [tail]):
                continue
            chosen = cell
            break
        if chosen is None:
            return None
        obstacles.add(chosen)
        frames.append(tail)

    seed_frame = frames[0]
    seed_choices = list(_frame_interior(seed_frame))
    if not seed_choices:
        return None
    seed_pos = random.choice(seed_choices)

    case = {
        "shape": shape,
        "bg": 0,
        "seed_color": seed_color,
        "frames": frames,
        "obstacles": obstacles,
        "seed_pos": seed_pos,
        "planned_filled": planned_filled,
        "final_gap_hit": final_gap_hit,
        "blocked_tail": blocked_tail,
    }
    inp = _render_input(case)
    expected, filled = _simulate_chain(inp, frames, 0, seed_color, return_filled=True)
    if filled != planned_filled:
        return None

    forbidden = set(zip(*np.where(expected != case["bg"])))
    extras = _free_obstacle_cells(shape, frames, obstacles, forbidden)
    random.shuffle(extras)
    for cell in extras[: random.randint(0, 2)]:
        obstacles.add(cell)

    case["obstacles"] = obstacles
    case["input"] = _render_input(case)
    case["expected"] = _simulate_chain(case["input"], frames, 0, seed_color)
    if max(case["input"].shape) > 30 or max(case["expected"].shape) > 30:
        return None
    return case


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        if not np.array_equal(main(case["input"]), case["expected"]):
            continue
        return case["input"].copy()
    raise RuntimeError("failed to generate valid input for task 5")

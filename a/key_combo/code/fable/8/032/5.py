from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

FRAME_SIDE_MIN = 4
FRAME_SIDE_MAX = 7
GRID_MIN = 20
GRID_MAX = 29
PAIR_SEP_MIN = 3
PAIR_SEP_MAX = 7
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _frame_dict(r0: int, c0: int, h: int, w: int, side: str, offset: int, color: int) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if side == "up":
        gap = (r0, c0 + offset)
    elif side == "down":
        gap = (r1, c0 + offset)
    elif side == "left":
        gap = (r0 + offset, c0)
    else:
        gap = (r0 + offset, c1)
    dr, dc = DIRS[side]
    border = set()
    for rr in range(r0, r1 + 1):
        border.add((rr, c0))
        border.add((rr, c1))
    for cc in range(c0, c1 + 1):
        border.add((r0, cc))
        border.add((r1, cc))
    border.remove(gap)
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "gap": gap,
        "dir": (dr, dc),
        "inside": (gap[0] - dr, gap[1] - dc),
        "solid": border,
    }


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    color = int(frame["color"])
    for r, c in frame["solid"]:
        grid[r, c] = color


def _expanded_boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _can_place(frame: dict, frames: list[dict], reserved: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = frame["bbox"]
    if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
        return False
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr, cc) in reserved:
                return False
    for other in frames:
        if _expanded_boxes_overlap(frame["bbox"], other["bbox"], pad=1):
            return False
    return True


def _interior_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in colors_present(grid, exclude={bg}):
        mask = grid == color
        box = bbox(mask)
        assert box is not None, "expected every foreground color to be present"
        r0, c0, r1, c1 = box
        solid = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
        assert _is_connected(solid), "each frame color must form one connected component"

        border = set()
        for rr in range(r0, r1 + 1):
            border.add((rr, c0))
            border.add((rr, c1))
        for cc in range(c0, c1 + 1):
            border.add((r0, cc))
            border.add((r1, cc))

        extra = solid - border
        missing = border - solid
        assert not extra, "foreground pixels must lie on a rectangle border"
        assert len(missing) == 1, "each hollow rectangle must have exactly one gap"
        gap = next(iter(missing))
        gr, gc = gap
        is_corner = (gr in (r0, r1)) and (gc in (c0, c1))
        assert not is_corner, "gap must be on a side, not a corner"

        if gr == r0:
            side = "up"
        elif gr == r1:
            side = "down"
        elif gc == c0:
            side = "left"
        else:
            side = "right"
        dr, dc = DIRS[side]
        inside = (gr - dr, gc - dc)
        assert r0 < inside[0] < r1 and c0 < inside[1] < c1, "gap must lead into the frame interior"
        assert all(grid[r, c] == bg for r, c in _interior_cells({"bbox": box})), "interior must be empty in the input"

        frames.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "side": side,
                "gap": gap,
                "dir": (dr, dc),
                "inside": inside,
                "solid": solid,
            }
        )
    assert frames, "expected at least one hollow frame"
    return frames


def _trace_ray(
    shooter_idx: int,
    frames: list[dict],
    shape: tuple[int, int],
    gap_owner: dict[tuple[int, int], int],
    solid_owner: dict[tuple[int, int], int],
) -> tuple[str, int | None]:
    shooter = frames[shooter_idx]
    dr, dc = shooter["dir"]
    r, c = shooter["gap"]
    r += dr
    c += dc
    while in_bounds(shape, r, c):
        target_idx = gap_owner.get((r, c))
        if target_idx is not None and target_idx != shooter_idx:
            if (r + dr, c + dc) == frames[target_idx]["inside"]:
                return "hit", int(target_idx)
        blocker_idx = solid_owner.get((r, c))
        if blocker_idx is not None and blocker_idx != shooter_idx:
            return "blocked", int(blocker_idx)
        r += dr
        c += dc
    return "offgrid", None


def _solve_with_frames(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    gap_owner = {frame["gap"]: idx for idx, frame in enumerate(frames)}
    solid_owner = {}
    for idx, frame in enumerate(frames):
        for cell in frame["solid"]:
            solid_owner[cell] = idx

    incoming: dict[int, int] = {}
    for shooter_idx in range(len(frames)):
        status, payload = _trace_ray(shooter_idx, frames, grid.shape, gap_owner, solid_owner)
        if status != "hit":
            continue
        target_idx = int(payload)
        assert target_idx not in incoming, "input guarantees at most one incoming ray per frame"
        incoming[target_idx] = shooter_idx

    for target_idx, shooter_idx in incoming.items():
        target = frames[target_idx]
        r0, c0, r1, c1 = target["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = frames[shooter_idx]["color"]
        gr, gc = target["gap"]
        out[gr, gc] = get_bg(grid)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the inside of each hollow rectangle that is entered exactly through its gap by another frame's ray.

    Input structure:
    - The background is black, and every foreground object is a one-pixel-thick hollow rectangle of a unique color.
    - Each rectangle border is missing exactly one non-corner cell, forming a gap on one side.
    - Rectangle interiors are empty in the input, and a frame can receive at most one valid incoming ray.

    Rule:
    - Every frame fires a straight ray from its interior outward through its own gap, perpendicular to that side.
    - The ray is invisible: it advances cell by cell, stops immediately if it first reaches any solid border cell, and does nothing if it leaves the grid.
    - If the ray reaches another frame's gap and the next cell in the same direction is that frame's interior, the ray enters that frame and recolors the whole interior to the shooter's color while leaving the gap cell black.

    Key insight:
    - The missing border cell is both cannon muzzle and lockhole, so only door-to-door alignment with a clear corridor causes a fill; hitting any wall cell invalidates the shot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    frames = _detect_frames(g, bg)
    return _solve_with_frames(g, frames)


def _reserve_segment(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    reserved = set()
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        for c in range(a[1] + step, b[1], step):
            reserved.add((a[0], c))
    else:
        step = 1 if b[0] > a[0] else -1
        for r in range(a[0] + step, b[0], step):
            reserved.add((r, a[1]))
    return reserved


def _try_hit_pair(
    rng: random.Random,
    shape: tuple[int, int],
    frames: list[dict],
    reserved: set[tuple[int, int]],
    colors: list[int],
    orientation: str,
) -> tuple[dict, dict, set[tuple[int, int]]] | None:
    h, w = shape
    color_a = colors[0]
    color_b = colors[1]
    for _ in range(120):
        ha = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
        wa = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
        hb = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
        wb = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
        sep = rng.randint(PAIR_SEP_MIN, PAIR_SEP_MAX)
        if orientation == "horizontal":
            oa = rng.randint(1, ha - 2)
            ob = rng.randint(1, hb - 2)
            low = max(oa, ob)
            high = min(h - ha + oa, h - hb + ob)
            if low > high:
                continue
            gap_r = rng.randint(low, high)
            total_w = wa + sep + wb - 1
            if total_w > w:
                continue
            c0a = rng.randint(0, w - total_w)
            c0b = c0a + wa - 1 + sep
            a = _frame_dict(gap_r - oa, c0a, ha, wa, "right", oa, color_a)
            b = _frame_dict(gap_r - ob, c0b, hb, wb, "left", ob, color_b)
        else:
            oa = rng.randint(1, wa - 2)
            ob = rng.randint(1, wb - 2)
            low = max(oa, ob)
            high = min(w - wa + oa, w - wb + ob)
            if low > high:
                continue
            gap_c = rng.randint(low, high)
            total_h = ha + sep + hb - 1
            if total_h > h:
                continue
            r0a = rng.randint(0, h - total_h)
            r0b = r0a + ha - 1 + sep
            a = _frame_dict(r0a, gap_c - oa, ha, wa, "down", oa, color_a)
            b = _frame_dict(r0b, gap_c - ob, hb, wb, "up", ob, color_b)
        if not _can_place(a, frames, reserved, shape):
            continue
        if not _can_place(b, frames + [a], reserved, shape):
            continue
        corridor = _reserve_segment(a["gap"], b["gap"])
        return a, b, corridor
    return None


def _try_offgrid_frame(
    rng: random.Random,
    shape: tuple[int, int],
    frames: list[dict],
    reserved: set[tuple[int, int]],
    color: int,
) -> dict | None:
    h, w = shape
    sides = ["up", "down", "left", "right"]
    rng.shuffle(sides)
    for side in sides:
        for _ in range(60):
            fh = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
            fw = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
            if side in ("up", "down"):
                if fw > w or fh > h:
                    continue
                c0 = rng.randint(0, w - fw)
                r0 = 0 if side == "up" else h - fh
                offset = rng.randint(1, fw - 2)
            else:
                if fh > h or fw > w:
                    continue
                r0 = rng.randint(0, h - fh)
                c0 = 0 if side == "left" else w - fw
                offset = rng.randint(1, fh - 2)
            frame = _frame_dict(r0, c0, fh, fw, side, offset, color)
            if _can_place(frame, frames, reserved, shape):
                return frame
    return None


def _impact_options(blocker: dict) -> list[tuple[str, tuple[int, int]]]:
    r0, c0, r1, c1 = blocker["bbox"]
    gap = blocker["gap"]
    options = []
    for c in range(c0 + 1, c1):
        if (r0, c) != gap:
            options.append(("top", (r0, c)))
        if (r1, c) != gap:
            options.append(("bottom", (r1, c)))
    for r in range(r0 + 1, r1):
        if (r, c0) != gap:
            options.append(("left", (r, c0)))
        if (r, c1) != gap:
            options.append(("right", (r, c1)))
    return options


def _try_blocked_frame(
    rng: random.Random,
    shape: tuple[int, int],
    frames: list[dict],
    reserved: set[tuple[int, int]],
    color: int,
) -> tuple[dict, int, set[tuple[int, int]]] | None:
    blocker_order = list(range(len(frames)))
    rng.shuffle(blocker_order)
    for blocker_idx in blocker_order:
        blocker = frames[blocker_idx]
        options = _impact_options(blocker)
        rng.shuffle(options)
        for impact_side, impact in options:
            for _ in range(40):
                fh = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
                fw = rng.randint(FRAME_SIDE_MIN, FRAME_SIDE_MAX)
                sep = rng.randint(PAIR_SEP_MIN, PAIR_SEP_MAX)
                ir, ic = impact
                if impact_side == "left":
                    off = rng.randint(1, fh - 2)
                    frame = _frame_dict(ir - off, ic - sep - fw + 1, fh, fw, "right", off, color)
                elif impact_side == "right":
                    off = rng.randint(1, fh - 2)
                    frame = _frame_dict(ir - off, ic + sep, fh, fw, "left", off, color)
                elif impact_side == "top":
                    off = rng.randint(1, fw - 2)
                    frame = _frame_dict(ir - sep - fh + 1, ic - off, fh, fw, "down", off, color)
                else:
                    off = rng.randint(1, fw - 2)
                    frame = _frame_dict(ir + sep, ic - off, fh, fw, "up", off, color)
                if not _can_place(frame, frames, reserved, shape):
                    continue
                corridor = _reserve_segment(frame["gap"], impact)
                return frame, blocker_idx, corridor
    return None


def _ray_statuses(frames: list[dict], shape: tuple[int, int]) -> list[tuple[str, int | None]]:
    gap_owner = {frame["gap"]: idx for idx, frame in enumerate(frames)}
    solid_owner = {}
    for idx, frame in enumerate(frames):
        for cell in frame["solid"]:
            solid_owner[cell] = idx
    return [_trace_ray(idx, frames, shape, gap_owner, solid_owner) for idx in range(len(frames))]


def _expected_output(grid: np.ndarray, frames: list[dict], incoming: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for target_idx, shooter_idx in incoming.items():
        target = frames[target_idx]
        r0, c0, r1, c1 = target["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = frames[shooter_idx]["color"]
        gr, gc = target["gap"]
        out[gr, gc] = 0
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))
    frame_count = 4 + 2 * int(rng.random() < 0.55)
    colors = rng.sample(list(range(1, 10)), frame_count)

    frames: list[dict] = []
    reserved: set[tuple[int, int]] = set()
    planned_hits: dict[int, int] = {}
    planned_blocked: dict[int, int] = {}
    planned_offgrid: set[int] = set()

    pair1_orientation = rng.choice(["horizontal", "vertical"])
    pair1 = _try_hit_pair(rng, shape, frames, reserved, colors[:2], pair1_orientation)
    if pair1 is None:
        return None
    a1, b1, corridor = pair1
    frames.extend([a1, b1])
    reserved |= corridor
    planned_hits[0] = 1
    planned_hits[1] = 0

    color_ptr = 2
    if frame_count == 6:
        pair2_orientation = "vertical" if pair1_orientation == "horizontal" else "horizontal"
        pair2 = _try_hit_pair(rng, shape, frames, reserved, colors[color_ptr : color_ptr + 2], pair2_orientation)
        if pair2 is None:
            return None
        a2, b2, corridor2 = pair2
        frames.extend([a2, b2])
        reserved |= corridor2
        planned_hits[2] = 3
        planned_hits[3] = 2
        color_ptr += 2

    offgrid_frame = _try_offgrid_frame(rng, shape, frames, reserved, colors[color_ptr])
    if offgrid_frame is None:
        return None
    frames.append(offgrid_frame)
    planned_offgrid.add(len(frames) - 1)
    color_ptr += 1

    blocked = _try_blocked_frame(rng, shape, frames, reserved, colors[color_ptr])
    if blocked is None:
        return None
    blocked_frame, blocker_idx, blocked_corridor = blocked
    frames.append(blocked_frame)
    planned_blocked[len(frames) - 1] = blocker_idx
    reserved |= blocked_corridor

    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame)

    statuses = _ray_statuses(frames, shape)
    incoming = {}
    for shooter_idx, (status, payload) in enumerate(statuses):
        if status == "hit":
            incoming[int(payload)] = shooter_idx

    for shooter_idx, target_idx in planned_hits.items():
        if statuses[shooter_idx] != ("hit", target_idx):
            return None
    for shooter_idx, blocker_idx in planned_blocked.items():
        if statuses[shooter_idx] != ("blocked", blocker_idx):
            return None
    for shooter_idx in planned_offgrid:
        if statuses[shooter_idx][0] != "offgrid":
            return None

    if set(planned_hits) | set(planned_blocked) | planned_offgrid != set(range(len(frames))):
        return None
    if set(incoming) != set(planned_hits):
        return None
    if len(incoming) != len(planned_hits):
        return None
    if len(set(frame["side"] for frame in frames)) < 3:
        return None

    expected = _expected_output(grid, frames, incoming)
    if np.array_equal(expected, grid):
        return None
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lockhole-and-cannon input")

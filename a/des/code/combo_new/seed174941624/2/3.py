from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np


DIR4: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
GRID_MIN = 22
GRID_MAX = 30
TEMPLATE_COUNT_RANGE = (2, 4)
ROOM_OUTER_MIN = 5
ROOM_OUTER_MAX = 7
GEN_RETRIES = 300


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _prefix(mask: np.ndarray) -> np.ndarray:
    ps = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return ps.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _component_is_full_frame(wall_mask: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = wall_mask[r0 : r1 + 1, c0 : c1 + 1]
    expect = np.zeros_like(sub, dtype=bool)
    expect[0, :] = True
    expect[-1, :] = True
    expect[:, 0] = True
    expect[:, -1] = True
    return bool(np.array_equal(sub, expect))


def _find_templates(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    wall_mask = grid == wall
    templates: list[dict] = []
    for coords in _components(wall_mask):
        if not _component_is_full_frame(wall_mask, coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        if not np.any((inner != bg) & (inner != wall)):
            continue
        templates.append({"bbox": (r0, c0, r1, c1), "pattern": inner.copy()})
    templates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return templates


def _count_side_doors(side: np.ndarray, wall: int, bg: int) -> int | None:
    if side.size == 0:
        return None
    if not np.all((side == wall) | (side == bg)):
        return None
    gaps = side == bg
    if gaps.size > 1 and np.any(gaps[:-1] & gaps[1:]):
        return None
    return int(gaps.sum())


def _find_rooms(grid: np.ndarray, bg: int, wall: int) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    non_bg_ps = _prefix(grid != bg)
    rooms: list[tuple[int, int, int, int, int]] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(grid[r0, c0]) != wall:
                continue
            for r1 in range(r0 + 2, h):
                if int(grid[r1, c0]) != wall:
                    continue
                inner_h = r1 - r0 - 1
                if inner_h <= 0:
                    continue
                for c1 in range(c0 + 2, w):
                    if int(grid[r0, c1]) != wall or int(grid[r1, c1]) != wall:
                        continue
                    inner_w = c1 - c0 - 1
                    if inner_w <= 0:
                        continue
                    if _rect_sum(non_bg_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue

                    top = _count_side_doors(grid[r0, c0 + 1 : c1], wall, bg)
                    bottom = _count_side_doors(grid[r1, c0 + 1 : c1], wall, bg)
                    left = _count_side_doors(grid[r0 + 1 : r1, c0], wall, bg)
                    right = _count_side_doors(grid[r0 + 1 : r1, c1], wall, bg)
                    if None in (top, bottom, left, right):
                        continue

                    rooms.append((r0, c0, r1, c1, int(top + bottom + left + right)))

    return rooms


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = pattern.shape
    h, w = shape
    reps = ((h + ph - 1) // ph, (w + pw - 1) // pw)
    tiled = np.tile(pattern, reps)
    return tiled[:h, :w].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room with the template pattern indexed by its door count.

    Input structure:
    - The background is the most common color.
    - One wall color forms one-pixel orthogonal rectangular room frames; some wall cells are missing as one-pixel doors.
    - Separate small framed templates sit in a corner strip, each with the same wall color border and a non-background interior pattern.

    Rule:
    - Detect the template frames and order them from left to right.
    - Detect every non-template rectangular room whose interior is entirely background, and count the one-pixel door gaps on its four sides.
    - Leave rooms with zero doors unchanged.
    - For a room with door count D >= 1, tile the interior pattern of template D from the room's top-left interior corner, cropping at the room boundary.

    Key insight:
    - The corner mini-frames are a pattern dictionary, while each ordinary room is identified by its rectangular wall geometry and indexed by the number of wall gaps on its boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected walls and templates"
    vals, cnts = np.unique(non_bg, return_counts=True)
    wall = int(vals[np.argmax(cnts)])

    templates = _find_templates(g, bg, wall)
    assert templates, "expected at least one template frame"

    out = g.copy()
    for r0, c0, r1, c1, door_count in _find_rooms(g, bg, wall):
        if door_count <= 0 or door_count > len(templates):
            continue
        pattern = templates[door_count - 1]["pattern"]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(pattern, (r1 - r0 - 1, c1 - c0 - 1))
    return out


def _candidate_pattern(kind: str, shape: tuple[int, int], bg: int, c1: int, c2: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))

    if kind == "solid":
        return np.full((h, w), c1, dtype=int)
    if kind == "checker":
        return np.where((rr + cc) % 2 == 0, c1, bg).astype(int)
    if kind == "vstripe":
        return np.where(cc % 2 == 0, c1, bg).astype(int)
    if kind == "hstripe":
        return np.where(rr % 2 == 0, c1, bg).astype(int)
    if kind == "diag":
        return np.where((rr + cc) % 3 == 0, c1, bg).astype(int)
    if kind == "two_color":
        return np.where((rr + cc) % 2 == 0, c1, c2).astype(int)
    if kind == "cross":
        out = np.full((h, w), bg, dtype=int)
        out[h // 2, :] = c1
        out[:, w // 2] = c2
        return out
    raise ValueError(kind)


def _sample_patterns(
    rng: random.Random, count: int, bg: int, wall: int, color_pool: Sequence[int]
) -> list[np.ndarray] | None:
    kinds = ("solid", "checker", "vstripe", "hstripe", "diag", "two_color", "cross")
    used: set[tuple[int, int, tuple[int, ...]]] = set()
    patterns: list[np.ndarray] = []

    for _ in range(count):
        for _attempt in range(100):
            shape = (rng.choice((2, 2, 3)), rng.choice((2, 2, 3)))
            c1, c2 = rng.sample(list(color_pool), 2)
            kind = rng.choice(kinds)
            pattern = _candidate_pattern(kind, shape, bg, c1, c2)
            if not np.any(pattern != bg):
                continue
            if np.any(pattern == wall):
                continue
            sig = (shape[0], shape[1], tuple(int(x) for x in pattern.ravel()))
            if sig in used:
                continue
            used.add(sig)
            patterns.append(pattern)
            break
        else:
            return None

    return patterns


def _draw_box_border(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_templates(
    grid: np.ndarray,
    blocked: np.ndarray,
    patterns: Sequence[np.ndarray],
    wall: int,
    side: str,
    corner: str,
    rng: random.Random,
) -> list[dict] | None:
    h, w = grid.shape
    gap = rng.choice((1, 2))
    outer_sizes = [(pat.shape[0] + 2, pat.shape[1] + 2) for pat in patterns]
    total_w = sum(ow for _, ow in outer_sizes) + gap * (len(patterns) - 1)
    max_h = max(oh for oh, _ in outer_sizes)
    if total_w > w - 2 or max_h > h - 4:
        return None

    start_c = 1 if corner == "left" else w - 1 - total_w
    band_r = 1 if side == "top" else h - 1 - max_h
    cur_c = start_c
    placed: list[dict] = []

    for pattern, (oh, ow) in zip(patterns, outer_sizes):
        r0 = band_r if side == "top" else band_r + (max_h - oh)
        c0 = cur_c
        _draw_box_border(grid, r0, c0, oh, ow, wall)
        grid[r0 + 1 : r0 + oh - 1, c0 + 1 : c0 + ow - 1] = pattern
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + oh + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + ow + 1)
        blocked[rr0:rr1, cc0:cc1] = True
        placed.append({"bbox": (r0, c0, r0 + oh - 1, c0 + ow - 1), "pattern": pattern.copy()})
        cur_c += ow + gap

    return placed


def _room_positions(blocked: np.ndarray, room_h: int, room_w: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - room_h):
        for c0 in range(1, w - room_w):
            if not bool(blocked[r0 - 1 : r0 + room_h + 1, c0 - 1 : c0 + room_w + 1].any()):
                positions.append((r0, c0))
    return positions


def _door_cells(room_h: int, room_w: int, door_count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    if door_count == 0:
        return []
    candidates = []
    for c in range(1, room_w - 1):
        candidates.append(("top", c, (0, c)))
        candidates.append(("bottom", c, (room_h - 1, c)))
    for r in range(1, room_h - 1):
        candidates.append(("left", r, (r, 0)))
        candidates.append(("right", r, (r, room_w - 1)))
    rng.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    used: dict[str, set[int]] = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    for side, idx, coord in candidates:
        if idx - 1 in used[side] or idx + 1 in used[side]:
            continue
        used[side].add(idx)
        chosen.append(coord)
        if len(chosen) == door_count:
            return chosen
    return None


def _paint_rooms(
    grid: np.ndarray,
    blocked: np.ndarray,
    wall: int,
    room_specs: Sequence[dict],
    rng: random.Random,
) -> list[dict] | None:
    ordered = sorted(room_specs, key=lambda item: item["h"] * item["w"], reverse=True)

    for _ in range(80):
        trial_grid = grid.copy()
        trial_blocked = blocked.copy()
        placed: list[dict] = []
        ok = True

        for spec in ordered:
            positions = _room_positions(trial_blocked, spec["h"], spec["w"])
            if not positions:
                ok = False
                break
            r0, c0 = rng.choice(positions)
            doors = _door_cells(spec["h"], spec["w"], spec["doors"], rng)
            if doors is None:
                ok = False
                break
            _draw_box_border(trial_grid, r0, c0, spec["h"], spec["w"], wall)
            for dr, dc in doors:
                trial_grid[r0 + dr, c0 + dc] = spec["bg"]
            trial_blocked[r0 - 1 : r0 + spec["h"] + 1, c0 - 1 : c0 + spec["w"] + 1] = True
            placed.append({"bbox": (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1), "doors": spec["doors"]})

        if ok:
            grid[:, :] = trial_grid
            blocked[:, :] = trial_blocked
            return placed

    return None


def _expected_output(
    inp: np.ndarray, patterns: Sequence[np.ndarray], rooms: Sequence[dict]
) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        doors = int(room["doors"])
        if doors == 0:
            continue
        r0, c0, r1, c1 = room["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(patterns[doors - 1], (r1 - r0 - 1, c1 - c0 - 1))
    return out


def _wall_is_unique_non_bg_mode(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, cnts = np.unique(grid[grid != bg], return_counts=True)
    if vals.size == 0:
        return False
    wall_count = int(cnts[vals == wall][0]) if np.any(vals == wall) else -1
    if wall_count < 0:
        return False
    other_counts = [int(cnt) for val, cnt in zip(vals, cnts) if int(val) != wall]
    return all(wall_count > cnt for cnt in other_counts)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    colors = rng.sample(list(range(10)), 8)
    bg = colors[0]
    wall = colors[1]
    pattern_colors = colors[2:]
    template_count = rng.randint(*TEMPLATE_COUNT_RANGE)

    patterns = _sample_patterns(rng, template_count, bg, wall, pattern_colors)
    if patterns is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    template_side = rng.choice(("top", "bottom"))
    template_corner = rng.choice(("left", "right"))
    if _paint_templates(grid, blocked, patterns, wall, template_side, template_corner, rng) is None:
        return None

    room_counts = list(range(template_count + 1))
    if rng.random() < 0.5:
        room_counts.append(rng.randint(0, template_count))
    rng.shuffle(room_counts)

    room_specs = []
    for door_count in room_counts:
        room_specs.append(
            {
                "h": rng.randint(ROOM_OUTER_MIN, ROOM_OUTER_MAX),
                "w": rng.randint(ROOM_OUTER_MIN, ROOM_OUTER_MAX),
                "doors": int(door_count),
                "bg": bg,
            }
        )

    rooms = _paint_rooms(grid, blocked, wall, room_specs, rng)
    if rooms is None:
        return None

    if _mode_color(grid) != bg:
        return None
    if not _wall_is_unique_non_bg_mode(grid, bg, wall):
        return None

    detected_templates = _find_templates(grid, bg, wall)
    if len(detected_templates) != template_count:
        return None
    detected_rooms = _find_rooms(grid, bg, wall)
    if len(detected_rooms) != len(rooms):
        return None

    expected = _expected_output(grid, patterns, rooms)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
ALL_COLORS = list(range(1, 10))
UP_MASK = np.array(
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=bool,
)
DOWN_MASK = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
    ],
    dtype=bool,
)

GEN_TRIES = 500
PLAN_TRIES = 200
LANE_STEP = 4


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _classify_bracket(component: dict) -> str | None:
    r0, c0, r1, c1 = component["bbox"]
    if component["area"] != 7 or r1 - r0 != 2 or c1 - c0 != 2:
        return None
    mask = _component_mask(component)
    if np.array_equal(mask, UP_MASK):
        return "up"
    if np.array_equal(mask, DOWN_MASK):
        return "down"
    return None


def _paint_bracket(grid: np.ndarray, center: tuple[int, int], color: int, kind: str) -> None:
    r, c = center
    top = r - 1
    left = c - 1
    mask = UP_MASK if kind == "up" else DOWN_MASK
    patch = grid[top : top + 3, left : left + 3]
    assert patch.shape == (3, 3)
    assert not np.any((patch != BG) & mask)
    patch[mask] = int(color)


def _lane_centers(width: int) -> list[int]:
    return [c for c in range(2, width - 1, LANE_STEP)]


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        kind = _classify_bracket(comp)
        if kind is None:
            continue
        r0, c0, _r1, _c1 = comp["bbox"]
        part = dict(comp)
        part["kind"] = kind
        part["center"] = (r0 + 1, c0 + 1)
        by_color[int(comp["color"])].append(part)

    pairs: dict[int, dict] = {}
    bracket_mask = np.zeros_like(g, dtype=bool)
    for color, parts in by_color.items():
        ups = [part for part in parts if part["kind"] == "up"]
        downs = [part for part in parts if part["kind"] == "down"]
        assert len(parts) == 2 and len(ups) == 1 and len(downs) == 1
        pairs[color] = {"color": color, "up": ups[0], "down": downs[0]}
        for part in parts:
            coords = part["coords"]
            bracket_mask[coords[:, 0], coords[:, 1]] = True

    balls: list[dict] = []
    for comp in components:
        color = int(comp["color"])
        if color in pairs:
            assert _classify_bracket(comp) is not None
            continue
        assert comp["area"] == 1
        rr, cc = map(int, comp["coords"][0])
        assert not bracket_mask[rr, cc]
        balls.append({"color": color, "pos": (rr, cc)})

    assert balls
    balls.sort(key=lambda item: (item["pos"][0], item["pos"][1], item["color"]))
    return {"grid": g, "bg": bg, "pairs": pairs, "balls": balls, "bracket_mask": bracket_mask}


def _simulate_ball(
    start: tuple[int, int],
    portals: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    solid_mask: np.ndarray,
    height: int,
    *,
    max_teleports: int | None = None,
    ignore_blocks: bool = False,
) -> dict:
    r, c = map(int, start)
    teleports: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    guard = 0
    while True:
        guard += 1
        assert guard <= height * 8 + len(portals) * 8 + 20
        nr = r + 1
        if nr >= height:
            return {
                "start": start,
                "final": (r, c),
                "teleports": teleports,
                "stop_reason": "bottom",
            }
        if not ignore_blocks and solid_mask[nr, c]:
            return {
                "start": start,
                "final": (r, c),
                "teleports": teleports,
                "stop_reason": "bracket",
            }
        r = nr
        if (r, c) in portals and (max_teleports is None or len(teleports) < max_teleports):
            color, dst = portals[(r, c)]
            teleports.append((color, (r, c), dst))
            r, c = dst


def _solve_details(
    grid: np.ndarray,
    *,
    max_teleports: int | None = None,
    ignore_blocks: bool = False,
) -> tuple[np.ndarray, list[dict], dict]:
    scene = _parse_scene(grid)
    g = scene["grid"]
    pairs = scene["pairs"]
    portals = {
        tuple(pair["up"]["center"]): (color, tuple(pair["down"]["center"]))
        for color, pair in pairs.items()
    }
    details = [
        _simulate_ball(
            ball["pos"],
            portals,
            scene["bracket_mask"],
            g.shape[0],
            max_teleports=max_teleports,
            ignore_blocks=ignore_blocks,
        )
        for ball in scene["balls"]
    ]
    finals = [detail["final"] for detail in details]
    assert len(finals) == len(set(finals))

    out = g.copy()
    for ball in scene["balls"]:
        rr, cc = ball["pos"]
        out[rr, cc] = scene["bg"]
    for ball, detail in zip(scene["balls"], details):
        rr, cc = detail["final"]
        out[rr, cc] = int(ball["color"])
    return out, details, scene


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each single-pixel ball until it is teleported through matching bracket wells and finally comes to rest.

    Input structure:
    - The background is black.
    - Several monochrome 3x3 bracket frames appear; each bracket color occurs exactly twice, once as an upward-opening ∪ frame and once as a downward-opening ∩ frame.
    - One or more single-pixel balls of non-bracket colors start near the top of the grid.

    Rule:
    - Each ball falls straight downward in its current column.
    - When a ball reaches the interior center cell of an upward-opening bracket, it is teleported to the interior center cell of the same-colored downward-opening bracket, then keeps falling from there.
    - Teleports may chain through multiple bracket pairs; otherwise a ball stops when the next cell is occupied by any bracket arm or back, or at the bottom edge if nothing blocks it.
    - Remove the balls from their starting cells, keep every bracket unchanged, and draw each ball only at its final resting cell.

    Key insight:
    - Opening direction assigns bracket roles: ∪ cells are entrances, ∩ cells are exits, so the final landing spot depends on the whole color-linked teleport chain rather than the ball's original column alone.
    """
    out, _details, _scene = _solve_details(grid)
    return out


def _sample_chain_lengths(n_balls: int) -> list[int]:
    if n_balls == 1:
        return [random.choice([1, 2])]

    for _ in range(PLAN_TRIES):
        lengths = random.choices([0, 1, 2], weights=[2, 5, 3], k=n_balls)
        if sum(lengths) == 0:
            lengths[random.randrange(n_balls)] = random.choice([1, 2])
        if 2 not in lengths and random.random() < 0.55:
            ones = [idx for idx, value in enumerate(lengths) if value == 1]
            if ones:
                lengths[random.choice(ones)] = 2
        if n_balls >= 2 and not any(value == 0 for value in lengths) and random.random() < 0.45:
            candidates = [idx for idx, value in enumerate(lengths) if value >= 1]
            idx = random.choice(candidates)
            lengths[idx] = 0 if lengths[idx] == 1 else 1
        if sum(lengths) > 0:
            return list(map(int, lengths))
    raise RuntimeError("failed to sample chain lengths")


def _plan_ball_rows(start_row: int, chain_len: int, has_blocker: bool, height: int) -> dict | None:
    current = int(start_row)
    entrances: list[int] = []
    exits: list[int] = []
    for _ in range(chain_len):
        current += random.randint(3, 4)
        entrance = current
        current = entrance + random.randint(2, 3)
        exit_row = current
        entrances.append(entrance)
        exits.append(exit_row)
    blocker = None
    if has_blocker:
        blocker = current + random.randint(4, 6)
    limit = blocker if blocker is not None else (exits[-1] if exits else start_row)
    if limit > height - 2:
        return None
    return {"start_row": start_row, "entrances": entrances, "exits": exits, "blocker": blocker}


def _render_instance(spec: dict) -> np.ndarray | None:
    height = spec["height"]
    width = spec["width"]
    grid = np.zeros((height, width), dtype=int)
    brackets: list[tuple[tuple[int, int], int, str]] = []

    for pair in spec["teleport_pairs"]:
        brackets.append((pair["up_center"], pair["color"], "up"))
        brackets.append((pair["down_center"], pair["color"], "down"))
    for pair in spec["blocker_pairs"]:
        brackets.append((pair["up_center"], pair["color"], "up"))
        brackets.append((pair["down_center"], pair["color"], "down"))
    for pair in spec["isolated_pairs"]:
        brackets.append((pair["up_center"], pair["color"], "up"))
        brackets.append((pair["down_center"], pair["color"], "down"))

    for center, color, kind in brackets:
        r, c = center
        if not (1 <= r <= height - 2 and 1 <= c <= width - 2):
            return None
        try:
            _paint_bracket(grid, center, color, kind)
        except AssertionError:
            return None

    for ball in spec["balls"]:
        r, c = ball["pos"]
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(ball["color"])

    return grid


def _build_spec() -> dict | None:
    n_balls = random.randint(1, 3)
    chain_lengths = _sample_chain_lengths(n_balls)
    blocker_flags = [random.random() < 0.45 for _ in range(n_balls)]
    if not any(blocker_flags) and random.random() < 0.5:
        blocker_flags[random.randrange(n_balls)] = True
    isolated_count = 1 if random.random() < 0.55 else 0

    path_lane_count = sum(length + 1 for length in chain_lengths)
    total_lanes = path_lane_count + sum(blocker_flags) + isolated_count
    pair_count = sum(chain_lengths) + sum(blocker_flags) + isolated_count
    if total_lanes > 7 or pair_count + n_balls > len(ALL_COLORS):
        return None

    min_width = max(18, total_lanes * LANE_STEP)
    if min_width > 30:
        return None
    width = random.randint(min_width, 30)
    height = random.randint(20, 30)

    centers = _lane_centers(width)
    if len(centers) < total_lanes:
        return None
    random.shuffle(centers)

    ball_lanes: list[list[int]] = []
    take = 0
    for length in chain_lengths:
        ball_lanes.append(centers[take : take + length + 1])
        take += length + 1

    blocker_dead_lanes: dict[int, int] = {}
    for idx, has_blocker in enumerate(blocker_flags):
        if has_blocker:
            blocker_dead_lanes[idx] = centers[take]
            take += 1

    isolated_lanes = centers[take : take + isolated_count]

    colors = random.sample(ALL_COLORS, pair_count + n_balls)
    pair_colors = colors[:pair_count]
    ball_colors = colors[pair_count:]
    pair_cursor = 0

    plans = []
    for idx in range(n_balls):
        start_row = random.randint(0, 3)
        plan = _plan_ball_rows(start_row, chain_lengths[idx], blocker_flags[idx], height)
        if plan is None:
            return None
        plans.append(plan)

    teleport_pairs = []
    blocker_pairs = []
    balls = []
    for idx in range(n_balls):
        lanes = ball_lanes[idx]
        plan = plans[idx]
        balls.append({"color": ball_colors[idx], "pos": (plan["start_row"], lanes[0])})
        for step, entrance_row in enumerate(plan["entrances"]):
            color = pair_colors[pair_cursor]
            pair_cursor += 1
            teleport_pairs.append(
                {
                    "color": color,
                    "up_center": (entrance_row, lanes[step]),
                    "down_center": (plan["exits"][step], lanes[step + 1]),
                }
            )
        if blocker_flags[idx]:
            color = pair_colors[pair_cursor]
            pair_cursor += 1
            blocker_pairs.append(
                {
                    "color": color,
                    "up_center": (random.randint(5, height - 6), blocker_dead_lanes[idx]),
                    "down_center": (plan["blocker"], lanes[-1]),
                }
            )

    isolated_pairs = []
    for lane in isolated_lanes:
        color = pair_colors[pair_cursor]
        pair_cursor += 1
        up_center = random.randint(6, height - 10)
        down_center = random.randint(up_center + 4, height - 3)
        isolated_pairs.append(
            {
                "color": color,
                "up_center": (up_center, lane),
                "down_center": (down_center, lane),
            }
        )

    return {
        "height": height,
        "width": width,
        "balls": balls,
        "teleport_pairs": teleport_pairs,
        "blocker_pairs": blocker_pairs,
        "isolated_pairs": isolated_pairs,
    }


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        out, details, scene = _solve_details(grid)
    except AssertionError:
        return False

    if grid.shape != out.shape or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.array_equal(grid, out):
        return False

    used_colors = {
        int(color)
        for detail in details
        for color, _src, _dst in detail["teleports"]
    }
    if not used_colors:
        return False

    finals = [detail["final"] for detail in details]
    if len(finals) != len(set(finals)):
        return False

    direct_variant = [
        detail["final"]
        for detail in _solve_details(grid, max_teleports=0)[1]
    ]
    if all(a == b for a, b in zip(finals, direct_variant)):
        return False

    if any(len(detail["teleports"]) >= 2 for detail in details):
        one_hop_variant = [
            detail["final"]
            for detail in _solve_details(grid, max_teleports=1)[1]
        ]
        if all(a == b for a, b in zip(finals, one_hop_variant)):
            return False

    if any(detail["stop_reason"] == "bracket" for detail in details):
        no_collision_variant = [
            detail["final"]
            for detail in _solve_details(grid, ignore_blocks=True)[1]
        ]
        if all(a == b for a, b in zip(finals, no_collision_variant)):
            return False

    pair_colors = set(scene["pairs"])
    ball_colors = {int(ball["color"]) for ball in scene["balls"]}
    if pair_colors & ball_colors:
        return False

    has_multi = any(len(detail["teleports"]) >= 2 for detail in details)
    has_direct = any(len(detail["teleports"]) == 0 for detail in details)
    has_block = any(detail["stop_reason"] == "bracket" for detail in details)
    has_unused = len(scene["pairs"]) > len(used_colors)
    if not (has_multi or has_direct or has_block or has_unused):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        spec = _build_spec()
        if spec is None:
            continue
        grid = _render_instance(spec)
        if grid is None:
            continue
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate input for bracket teleport well")

from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
SIDES = ("top", "bottom", "left", "right")
REFLECT = {
    "/": {
        DOWN: LEFT,
        RIGHT: UP,
        UP: RIGHT,
        LEFT: DOWN,
    },
    "\\": {
        DOWN: RIGHT,
        LEFT: UP,
        UP: LEFT,
        RIGHT: DOWN,
    },
}
PALETTE = tuple(range(10))
GRID_MIN = 10
GRID_MAX = 24
MAX_GENERATE_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    mirror_map: dict[tuple[int, int], str] = {}
    for comp in _components8(grid == GREY):
        assert len(comp) == 2, "each gray mirror must contain exactly two pixels"
        (r0, c0), (r1, c1) = sorted(comp)
        assert 0 < r0 < h - 1 and 0 < c0 < w - 1
        assert 0 < r1 < h - 1 and 0 < c1 < w - 1
        dr = r1 - r0
        dc = c1 - c0
        assert dr == 1 and abs(dc) == 1, "mirror pixels must form a diagonal pair"
        orient = "\\" if dc == 1 else "/"
        for cell in comp:
            mirror_map[cell] = orient
    return mirror_map


def _marker_info(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], str]:
    touches = int(r == 0) + int(r == h - 1) + int(c == 0) + int(c == w - 1)
    assert touches == 1, "markers must lie on exactly one border edge"
    if r == 0:
        return DOWN, "top"
    if r == h - 1:
        return UP, "bottom"
    if c == 0:
        return RIGHT, "left"
    return LEFT, "right"


def _find_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, tuple[int, int], str]]:
    h, w = grid.shape
    markers: list[tuple[int, int, int, tuple[int, int], str]] = []
    colors: set[int] = set()
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value == bg or value == GREY:
                continue
            assert r in (0, h - 1) or c in (0, w - 1), "non-gray foreground must be border markers"
            direction, side = _marker_info(r, c, h, w)
            assert value not in colors, "marker colors must be distinct"
            colors.add(value)
            markers.append((r, c, value, direction, side))
    assert markers, "expected at least one border marker"
    return markers


def _solve(grid: np.ndarray, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 3 <= h <= 30 and 3 <= w <= 30, "grid must fit ARC size limits"

    bg = _background_color(g)
    assert bg != GREY, "background color must differ from gray mirrors"
    mirror_map = _find_mirrors(g)
    markers = _find_markers(g, bg)
    marker_cells = {(r, c) for r, c, _color, _direction, _side in markers}

    out = g.copy()
    stats = {
        "painted": 0,
        "turns": 0,
        "overwrites": 0,
        "rays": [],
        "used_sides": set(),
    }

    for r, c, color, direction, side in markers:
        rr, cc = int(r), int(c)
        cur_dir = direction
        visited: set[tuple[int, int, tuple[int, int]]] = set()
        path_len = 0
        ray_turns = 0
        hit_orients: list[str] = []
        while True:
            state = (rr, cc, cur_dir)
            assert state not in visited, "ray entered a reflection loop"
            visited.add(state)

            nr = rr + cur_dir[0]
            nc = cc + cur_dir[1]
            assert 0 <= nr < h and 0 <= nc < w, "ray must move inward from the border"

            if (nr, nc) in mirror_map:
                cur_dir = REFLECT[mirror_map[(nr, nc)]][cur_dir]
                ray_turns += 1
                hit_orients.append(mirror_map[(nr, nc)])
            elif (nr, nc) not in marker_cells:
                if int(g[nr, nc]) == bg and int(out[nr, nc]) != bg:
                    stats["overwrites"] += 1
                out[nr, nc] = color
                path_len += 1

            rr, cc = nr, nc
            if rr in (0, h - 1) or cc in (0, w - 1):
                break

        stats["painted"] += path_len
        stats["turns"] += ray_turns
        stats["used_sides"].add(side)
        stats["rays"].append(
            {
                "side": side,
                "path_len": path_len,
                "turns": ray_turns,
                "hit_orients": tuple(hit_orients),
            }
        )

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each edge marker color inward, reflecting rays off gray diagonal mirrors.

    Input structure:
    - The background is the most common color.
    - Non-background border cells other than gray are single-pixel markers, each
      lying on exactly one edge.
    - Interior gray pixels form isolated 8-connected two-pixel diagonal segments,
      each sloping either `/` or `\\`.

    Rule:
    - Process the border markers in reading order: left-to-right within a row,
      then top-to-bottom across rows.
    - Each marker emits a ray perpendicular to its edge and the ray advances one
      cell at a time.
    - Entering a gray `/` mirror changes direction by down->left, right->up,
      up->right, left->down; entering a gray `\\` mirror changes direction by
      down->right, left->up, up->left, right->down.
    - Paint every traversed non-marker, non-mirror cell with the marker color,
      overwriting earlier ray colors at crossings; gray mirror cells stay gray.
    - Stop a ray as soon as it reaches the grid boundary.

    Key insight:
    - The only hard part is identifying each gray two-pixel diagonal's slope,
      because that slope completely determines how the discrete ray turns.
    """
    return _solve(grid, collect_stats=False)


def _orientation_plan(count: int, rng: random.Random) -> list[str]:
    mode = rng.choice(["balanced", "slash_only", "backslash_only"])
    if count == 1 and mode == "balanced":
        mode = rng.choice(["slash_only", "backslash_only"])
    if mode == "slash_only":
        return ["/"] * count
    if mode == "backslash_only":
        return ["\\"] * count

    slash_count = count // 2
    backslash_count = count - slash_count
    if count % 2 == 1 and rng.random() < 0.5:
        slash_count, backslash_count = backslash_count, slash_count
    plan = ["/"] * slash_count + ["\\"] * backslash_count
    rng.shuffle(plan)
    return plan


def _place_mirrors(h: int, w: int, count: int, rng: random.Random) -> list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] | None:
    orientations = _orientation_plan(count, rng)
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    mirrors: list[tuple[str, tuple[tuple[int, int], tuple[int, int]]]] = []

    for orient in orientations:
        placed = False
        for _ in range(300):
            r0 = rng.randint(1, h - 3)
            c0 = rng.randint(1, w - 3)
            if orient == "\\":
                cells = ((r0, c0), (r0 + 1, c0 + 1))
            else:
                cells = ((r0, c0 + 1), (r0 + 1, c0))

            if any(cell in occupied or cell in forbidden for cell in cells):
                continue

            occupied.update(cells)
            for r, c in cells:
                for dr, dc in DIRS8 + ((0, 0),):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        forbidden.add((nr, nc))
            mirrors.append((orient, tuple(sorted(cells))))
            placed = True
            break

        if not placed:
            return None

    return mirrors


def _sample_spaced(values: list[int], count: int, rng: random.Random, gap: int = 2) -> list[int] | None:
    if count == 0:
        return []
    if count > len(values):
        return None
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _coord_for_side(side: str, pos: int, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, pos
    if side == "bottom":
        return h - 1, pos
    if side == "left":
        return pos, 0
    return pos, w - 1


def _place_markers(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    side_count = min(count, rng.choices([2, 3, 4], weights=[2, 3, 2], k=1)[0])
    chosen_sides = rng.sample(SIDES, side_count)

    side_lengths = {
        "top": w - 2,
        "bottom": w - 2,
        "left": h - 2,
        "right": h - 2,
    }
    capacities = {side: (side_lengths[side] + 1) // 2 for side in chosen_sides}
    counts = {side: 1 for side in chosen_sides}

    for _ in range(count - side_count):
        eligible = [side for side in chosen_sides if counts[side] < capacities[side]]
        if not eligible:
            return None
        counts[rng.choice(eligible)] += 1

    markers: list[tuple[int, int]] = []
    for side in chosen_sides:
        limit = w if side in ("top", "bottom") else h
        picks = _sample_spaced(list(range(1, limit - 1)), counts[side], rng, gap=2)
        if picks is None:
            return None
        markers.extend(_coord_for_side(side, pos, h, w) for pos in picks)

    return markers


def _interesting_enough(grid: np.ndarray, out: np.ndarray, stats: dict, rng: random.Random) -> bool:
    if np.array_equal(grid, out):
        return False
    if stats["painted"] < max(6, 2 * len(stats["rays"])):
        return False
    if max(ray["path_len"] for ray in stats["rays"]) < 3:
        return False

    need_turns = rng.random() < 0.7
    need_straight = rng.random() < 0.6
    need_crossing = rng.random() < 0.2

    if need_turns and stats["turns"] == 0:
        return False
    if need_straight and not any(ray["turns"] == 0 for ray in stats["rays"]):
        return False
    if need_crossing and stats["overwrites"] == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([color for color in PALETTE if color != GREY])

    grid = np.full((h, w), bg, dtype=int)

    max_mirrors = min(8, max(1, ((h - 2) * (w - 2)) // 18 + 1))
    mirror_count = rng.randint(1, max_mirrors)
    mirrors = _place_mirrors(h, w, mirror_count, rng)
    if mirrors is None:
        return None
    for _orient, cells in mirrors:
        for r, c in cells:
            grid[r, c] = GREY

    color_pool = [color for color in PALETTE if color not in (bg, GREY)]
    max_markers = min(6, len(color_pool))
    marker_count = rng.randint(2, max_markers)
    markers = _place_markers(h, w, marker_count, rng)
    if markers is None:
        return None

    for color, (r, c) in zip(rng.sample(color_pool, marker_count), markers):
        grid[r, c] = color

    try:
        out, stats = _solve(grid, collect_stats=True)
    except AssertionError:
        return None

    if not _interesting_enough(grid, out, stats, rng):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-ray ARC input")

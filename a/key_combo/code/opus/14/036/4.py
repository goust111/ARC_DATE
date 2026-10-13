from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
ANCHOR_COLORS = (2, 3, 4)

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            out.append(comp)
    return out


def flood_reachable_black(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    reach = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not reach[r, c]:
                reach[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not reach[r, c]:
                reach[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def find_anchor(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    anchors: list[tuple[int, int, int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if mask[r : r + 3, c : c + 3].all():
                anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1, "each foreground color must contain exactly one 3x3 anchor"
    return anchors[0]


def ray_info(
    coord: tuple[int, int], anchor: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = coord
    r0, c0, r1, c1 = anchor

    if r0 <= r <= r1 and c < c0:
        return (r, c0 - 1), (0, 1)
    if r0 <= r <= r1 and c > c1:
        return (r, c1 + 1), (0, -1)
    if c0 <= c <= c1 and r < r0:
        return (r0 - 1, c), (1, 0)
    if c0 <= c <= c1 and r > r1:
        return (r1 + 1, c), (-1, 0)

    if r < r0 and c < c0 and (r0 - r) == (c0 - c):
        return (r0 - 1, c0 - 1), (1, 1)
    if r < r0 and c > c1 and (r0 - r) == (c - c1):
        return (r0 - 1, c1 + 1), (1, -1)
    if r > r1 and c < c0 and (r - r1) == (c0 - c):
        return (r1 + 1, c0 - 1), (-1, 1)
    if r > r1 and c > c1 and (r - r1) == (c - c1):
        return (r1 + 1, c1 + 1), (-1, -1)

    raise AssertionError("ray pixel is not aligned to its anchor by row, column, or 45-degree diagonal")


def compact_foregrounds(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    out[g == BLUE] = BLUE

    for color in ANCHOR_COLORS:
        mask = g == color
        if not mask.any():
            continue
        anchor = find_anchor(mask)
        r0, c0, r1, c1 = anchor
        out[r0 : r1 + 1, c0 : c1 + 1] = color

        rays: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = defaultdict(list)
        for rr, cc in np.argwhere(mask):
            r = int(rr)
            c = int(cc)
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            terminal, toward = ray_info((r, c), anchor)
            dist = max(abs(r - terminal[0]), abs(c - terminal[1]))
            rays[(terminal, toward)].append(dist)

        for (terminal, toward), dists in rays.items():
            dists = sorted(dists)
            assert len(dists) == len(set(dists)), "ray pixels on one ray must occupy distinct distances"
            tr, tc = terminal
            dr, dc = toward
            for i, _dist in enumerate(dists):
                dest = (tr - i * dr, tc - i * dc)
                assert in_bounds(g.shape, *dest)
                assert out[dest] == BLACK, "compacted foreground pixels must not overlap"
                out[dest] = color

    return out


def keep_blue_component(comp: list[tuple[int, int]], reach: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
    if touches_border:
        return True
    for r, c in comp:
        for nr, nc in neighbors4(r, c):
            if in_bounds(shape, nr, nc) and reach[nr, nc]:
                return True
    return False


def apply_barrier_logic(compacted: np.ndarray) -> np.ndarray:
    reach = flood_reachable_black(compacted)
    out = compacted.copy()

    for comp in components(compacted == BLUE):
        if keep_blue_component(comp, reach, compacted.shape):
            continue
        for r, c in comp:
            out[r, c] = BLACK

    for r, c in np.argwhere(reach):
        rr = int(r)
        cc = int(c)
        if rr == 0 or cc == 0 or rr == out.shape[0] - 1 or cc == out.shape[1] - 1:
            out[rr, cc] = ORANGE
            continue
        if any(in_bounds(out.shape, nr, nc) and compacted[nr, nc] != BLACK for nr, nc in neighbors4(rr, cc)):
            out[rr, cc] = ORANGE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compact each color's scattered ray pixels back to its anchor, then use the rebuilt obstacle network to draw the reachable inner border.

    Input structure:
    - The background is black.
    - Two or three non-blue foreground colors each contain one solid 3x3 anchor block, and all other pixels of that color lie on row, column, or 45-degree diagonal rays from that anchor.
    - Blue wall segments include border-connected structural walls and interior decorative fragments.

    Rule:
    - For each non-black non-blue color, locate its unique 3x3 anchor block and group every other same-colored pixel by its row-, column-, or diagonal-aligned ray into that anchor.
    - Compact each ray toward the anchor so its pixels occupy consecutive cells starting from the anchor-adjacent terminal, while keeping anchor blocks fixed.
    - Treat all non-black pixels in this compacted grid as obstacles, flood-fill all boundary-connected black cells with 4-connectivity, paint orange on each reachable black cell that lies on the grid boundary or has a 4-neighbor obstacle, and erase blue components that are trapped entirely behind the obstacle network.

    Key insight:
    - The scattered colored pixels are not decorative: only after they compress into dense anchor-attached wall pieces do they close the blue barrier network and reveal which regions are reachable from the outer boundary.
    """
    compacted = compact_foregrounds(np.asarray(grid, dtype=int))
    return apply_barrier_logic(compacted)


def fill_line(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        assert grid[r, c] == BLACK
        grid[r, c] = color


def side_cells(side: str, anchor: tuple[int, int, int, int], length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    cells: list[tuple[int, int]] = []
    if side == "right":
        for r in range(r0, r1 + 1):
            for k in range(length):
                cells.append((r, c1 + 1 + k))
    elif side == "left":
        for r in range(r0, r1 + 1):
            for k in range(length):
                cells.append((r, c0 - 1 - k))
    elif side == "up":
        for c in range(c0, c1 + 1):
            for k in range(length):
                cells.append((r0 - 1 - k, c))
    elif side == "down":
        for c in range(c0, c1 + 1):
            for k in range(length):
                cells.append((r1 + 1 + k, c))
    else:
        raise AssertionError(side)
    return cells


def diag_cells(corner: str, anchor: tuple[int, int, int, int], length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    if corner == "ul":
        start = (r0 - 1, c0 - 1)
        step = (-1, -1)
    elif corner == "ur":
        start = (r0 - 1, c1 + 1)
        step = (-1, 1)
    elif corner == "dl":
        start = (r1 + 1, c0 - 1)
        step = (1, -1)
    elif corner == "dr":
        start = (r1 + 1, c1 + 1)
        step = (1, 1)
    else:
        raise AssertionError(corner)
    return [(start[0] + i * step[0], start[1] + i * step[1]) for i in range(length)]


def scatter_from_output(
    shape: tuple[int, int],
    terminal: tuple[int, int],
    away: tuple[int, int],
    count: int,
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    tr, tc = terminal
    dr, dc = away
    cells: list[tuple[int, int]] = []
    max_dist = 0
    nr, nc = tr + dr, tc + dc
    while in_bounds(shape, nr, nc):
        max_dist += 1
        nr += dr
        nc += dc
    if max_dist < count:
        return None

    choices = list(range(1, max_dist + 1))
    rng.shuffle(choices)
    choices.sort()

    used: list[int] = []
    for dist in choices:
        pos = (tr + dr * dist, tc + dc * dist)
        if pos in occupied:
            continue
        used.append(dist)
        occupied.add(pos)
        cells.append(pos)
        if len(cells) == count:
            return cells
    return None


def build_barrier_blue(
    h: int,
    w: int,
    rt: int,
    cl: int,
    rb: int,
    cr: int,
    corridor_side: str,
    rng: random.Random,
) -> np.ndarray:
    g = np.zeros((h, w), dtype=int)
    g[rt, cl : cr + 1] = BLUE
    g[rb, cl : cr + 1] = BLUE
    g[rt : rb + 1, cl] = BLUE
    g[rt : rb + 1, cr] = BLUE

    if corridor_side == "top":
        col = rng.randint(cl + 1, cr - 1)
        g[0 : rt + 1, col] = BLUE
    elif corridor_side == "bottom":
        col = rng.randint(cl + 1, cr - 1)
        g[rb:h, col] = BLUE
    elif corridor_side == "left":
        row = rng.randint(rt + 1, rb - 1)
        g[row, 0 : cl + 1] = BLUE
    else:
        row = rng.randint(rt + 1, rb - 1)
        g[row, cr:w] = BLUE
    return g


def gap_choices(
    rt: int, cl: int, rb: int, cr: int, sides: list[str], rng: random.Random
) -> dict[str, int] | None:
    pos: dict[str, int] = {}
    mid_r = (rt + rb) // 2
    mid_c = (cl + cr) // 2
    jitter_r = rng.randint(-1, 1)
    jitter_c = rng.randint(-1, 1)

    if "left" in sides:
        low = rt + 2 + (2 if "top" in sides else 0)
        high = rb - 4 - (2 if "bottom" in sides else 0)
        if low > high:
            return None
        pos["left"] = min(max(mid_r - 1 + jitter_r, low), high)
    if "right" in sides:
        low = rt + 2 + (2 if "top" in sides else 0)
        high = rb - 4 - (2 if "bottom" in sides else 0)
        if low > high:
            return None
        pos["right"] = min(max(mid_r - 1 - jitter_r, low), high)
    if "top" in sides:
        low = cl + 2 + (2 if "left" in sides else 0)
        high = cr - 4 - (2 if "right" in sides else 0)
        if low > high:
            return None
        pos["top"] = min(max(mid_c - 1 + jitter_c, low), high)
    if "bottom" in sides:
        low = cl + 2 + (2 if "left" in sides else 0)
        high = cr - 4 - (2 if "right" in sides else 0)
        if low > high:
            return None
        pos["bottom"] = min(max(mid_c - 1 - jitter_c, low), high)
    return pos


def candidate_anchor(side: str, pos: int, rt: int, cl: int, rb: int, cr: int) -> tuple[int, int, int, int]:
    if side == "right":
        return (pos, cr - 3, pos + 2, cr - 1)
    if side == "left":
        return (pos, cl + 1, pos + 2, cl + 3)
    if side == "top":
        return (rt + 1, pos, rt + 3, pos + 2)
    if side == "bottom":
        return (rb - 3, pos, rb - 1, pos + 2)
    raise AssertionError(side)


def critical_spec(side: str, anchor: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    if side == "right":
        return "right", (0, 1)
    if side == "left":
        return "left", (0, -1)
    if side == "top":
        return "up", (-1, 0)
    if side == "bottom":
        return "down", (1, 0)
    raise AssertionError(side)


def terminal_triplets(side: str, anchor: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = anchor
    if side == "right":
        return [((r, c1 + 1), (0, 1)) for r in range(r0, r1 + 1)]
    if side == "left":
        return [((r, c0 - 1), (0, -1)) for r in range(r0, r1 + 1)]
    if side in {"top", "up"}:
        return [((r0 - 1, c), (-1, 0)) for c in range(c0, c1 + 1)]
    if side in {"bottom", "down"}:
        return [((r1 + 1, c), (1, 0)) for c in range(c0, c1 + 1)]
    raise AssertionError(side)


def add_extra_rays(
    output: np.ndarray,
    side: str,
    anchor: tuple[int, int, int, int],
    ray_counts: dict[tuple[tuple[int, int], tuple[int, int]], int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
    require_diag: bool,
    require_long: bool,
) -> bool:
    r0, c0, r1, c1 = anchor
    color = int(output[r0, c0])
    outward_side = {"right": "right", "left": "left", "top": "up", "bottom": "down"}[side]

    long_ok = not require_long
    if require_long or rng.random() < 0.75:
        length = rng.randint(2, 3)
        terminals = terminal_triplets(outward_side, anchor)
        chosen = rng.sample(terminals, rng.randint(1, 2))
        for terminal, away in chosen:
            extras = [
                (terminal[0] + away[0] * step, terminal[1] + away[1] * step)
                for step in range(1, length)
            ]
            if any(not in_bounds(output.shape, rr, cc) for rr, cc in extras):
                if require_long:
                    return False
                break
            if any((rr, cc) in occupied for rr, cc in extras):
                if require_long:
                    return False
                break
        else:
            for terminal, away in chosen:
                extras = [
                    (terminal[0] + away[0] * step, terminal[1] + away[1] * step)
                    for step in range(1, length)
                ]
                for rr, cc in extras:
                    output[rr, cc] = color
                    occupied.add((rr, cc))
                ray_counts[(terminal, away)] = length
            long_ok = True

    if not long_ok:
        return False

    diag_options = {
        "right": ("ul", "dl"),
        "left": ("ur", "dr"),
        "top": ("dl", "dr"),
        "bottom": ("ul", "ur"),
    }[side]
    need_a_diag = require_diag or rng.random() < 0.7
    if not need_a_diag:
        return True

    for diag_name in rng.sample(list(diag_options), len(diag_options)):
        cells = diag_cells(diag_name, anchor, 1)
        rr, cc = cells[0]
        if not in_bounds(output.shape, rr, cc) or (rr, cc) in occupied:
            continue
        output[rr, cc] = color
        occupied.add((rr, cc))
        if diag_name == "ul":
            ray_counts[((r0 - 1, c0 - 1), (-1, -1))] = 1
        elif diag_name == "ur":
            ray_counts[((r0 - 1, c1 + 1), (-1, 1))] = 1
        elif diag_name == "dl":
            ray_counts[((r1 + 1, c0 - 1), (1, -1))] = 1
        else:
            ray_counts[((r1 + 1, c1 + 1), (1, 1))] = 1
        return True
    return False


def place_decorative_blue(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    rng: random.Random,
) -> bool:
    reach = flood_reachable_black(output_grid)
    enclosed = (output_grid == BLACK) & (~reach)
    comps = [comp for comp in components(enclosed) if len(comp) >= 8]
    rng.shuffle(comps)
    for comp in comps:
        cells = set(comp)
        for _ in range(30):
            orientation = rng.choice(("h", "v"))
            length = rng.randint(2, 4)
            base_r, base_c = rng.choice(comp)
            if orientation == "h":
                candidate = [(base_r, base_c + k) for k in range(length)]
            else:
                candidate = [(base_r + k, base_c) for k in range(length)]
            if not all((r, c) in cells and input_grid[r, c] == BLACK for r, c in candidate):
                continue
            good = True
            for r, c in candidate:
                for nr, nc in neighbors4(r, c):
                    if in_bounds(input_grid.shape, nr, nc) and output_grid[nr, nc] != BLACK and output_grid[nr, nc] != BLUE:
                        good = False
                        break
                if not good:
                    break
            if not good:
                continue
            for r, c in candidate:
                input_grid[r, c] = BLUE
            return True
    return False


def naive_without_compaction(input_grid: np.ndarray) -> np.ndarray:
    return apply_barrier_logic(np.asarray(input_grid, dtype=int))


def semantic_checks(input_grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        actual = main(input_grid)
        compacted = compact_foregrounds(input_grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected_output):
        return False

    if int((actual == ORANGE).sum()) < 8:
        return False

    reach = flood_reachable_black(compacted)
    if not ((expected_output == BLACK) & (~reach)).any():
        return False

    if not ((input_grid == BLUE) & (expected_output == BLACK)).any():
        return False

    if np.array_equal(naive_without_compaction(input_grid), expected_output):
        return False

    loose_by_color = 0
    for color in ANCHOR_COLORS:
        mask = input_grid == color
        if not mask.any():
            continue
        anchor = find_anchor(mask)
        r0, c0, r1, c1 = anchor
        for rr, cc in np.argwhere(mask):
            r = int(rr)
            c = int(cc)
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            ray_info((r, c), anchor)
            loose_by_color += 1
    if loose_by_color < 8:
        return False

    return True


def attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 25)
    w = rng.randint(18, 25)

    room_h = rng.randint(9, min(12, h - 8))
    room_w = rng.randint(9, min(12, w - 8))
    rt = rng.randint(3, h - room_h - 4)
    cl = rng.randint(3, w - room_w - 4)
    rb = rt + room_h - 1
    cr = cl + room_w - 1

    anchor_count = 3 if rng.random() < 0.3 else 2
    sides = rng.sample(["left", "right", "top", "bottom"], anchor_count)
    gap_pos = gap_choices(rt, cl, rb, cr, sides, rng)
    if gap_pos is None:
        return None

    corridor_candidates = [side for side in ["top", "bottom", "left", "right"] if side not in sides]
    if not corridor_candidates:
        corridor_candidates = ["top", "bottom", "left", "right"]
    structural = build_barrier_blue(h, w, rt, cl, rb, cr, rng.choice(corridor_candidates), rng)
    base_with_gaps = structural.copy()
    output = structural.copy()

    for side in sides:
        pos = gap_pos[side]
        if side in {"left", "right"}:
            base_with_gaps[pos : pos + 3, cl if side == "left" else cr] = BLACK
            output[pos : pos + 3, cl if side == "left" else cr] = BLACK
        else:
            base_with_gaps[rt if side == "top" else rb, pos : pos + 3] = BLACK
            output[rt if side == "top" else rb, pos : pos + 3] = BLACK

    colors = rng.sample(list(ANCHOR_COLORS), anchor_count)
    occupied = {(r, c) for r, c in zip(*np.where(output != BLACK), strict=False)}
    color_plans: list[tuple[int, tuple[int, int, int, int], dict[tuple[tuple[int, int], tuple[int, int]], int], str]] = []
    need_diagonal = True
    need_long = True

    for color, side in zip(colors, sides, strict=False):
        anchor = candidate_anchor(side, gap_pos[side], rt, cl, rb, cr)
        r0, c0, r1, c1 = anchor
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if any(not in_bounds((h, w), r, c) or (r, c) in occupied for r, c in cells):
            return None
        for r, c in cells:
            output[r, c] = color
            occupied.add((r, c))

        ray_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        crit_side, _away = critical_spec(side, anchor)
        crit_cells = side_cells(crit_side, anchor, 1)
        if any(not in_bounds((h, w), r, c) or (r, c) in occupied for r, c in crit_cells):
            return None
        for r, c in crit_cells:
            output[r, c] = color
            occupied.add((r, c))

        for terminal, away in terminal_triplets(crit_side, anchor):
            ray_counts[(terminal, away)] = 1

        if not add_extra_rays(
            output,
            side,
            anchor,
            ray_counts,
            occupied,
            rng,
            require_diag=need_diagonal,
            require_long=need_long,
        ):
            return None
        if any(abs(toward[0]) == 1 and abs(toward[1]) == 1 for (_terminal, toward) in ray_counts):
            need_diagonal = False
        if any(count > 1 for count in ray_counts.values()):
            need_long = False

        color_plans.append((color, anchor, ray_counts, side))

    if need_diagonal or need_long:
        return None

    input_grid = base_with_gaps.copy()
    loose_occupied = {(r, c) for r, c in zip(*np.where(input_grid != BLACK), strict=False)}
    for color, anchor, ray_counts, _side in color_plans:
        r0, c0, r1, c1 = anchor
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if input_grid[r, c] != BLACK:
                    return None
                input_grid[r, c] = color
                loose_occupied.add((r, c))

        for (terminal, away), count in ray_counts.items():
            cells = scatter_from_output((h, w), terminal, away, count, loose_occupied, rng)
            if cells is None:
                return None
            for r, c in cells:
                input_grid[r, c] = color

    obstacle_output = output.copy()
    if not place_decorative_blue(input_grid, obstacle_output, rng):
        return None
    expected_output = apply_barrier_logic(obstacle_output)

    if not semantic_checks(input_grid, expected_output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample for puzzle 4")

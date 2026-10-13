import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTION_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MIN = 12
GRID_MAX = 30
FRAME_OUTER_H_RANGE = (4, 9)
FRAME_OUTER_W_RANGE = (4, 10)
SEED_COUNT_MAX = 8
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 300
SEED_PLACEMENT_RETRIES = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _branch_cells(gap: tuple[int, int], direction: tuple[int, int], length: int, h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    rr, cc = gap[0] + dr, gap[1] + dc
    pts: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= rr < h and 0 <= cc < w):
            break
        pts.append((rr, cc))
        rr += dr
        cc += dc
    return pts


def _expand_cells(cells: set[tuple[int, int]] | list[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h - 1, r + radius) + 1):
            for cc in range(max(0, c - radius), min(w - 1, c + radius) + 1):
                out.add((rr, cc))
    return out


def _sample_nonadjacent_cells(
    rng: random.Random, r0: int, c0: int, r1: int, c1: int, count: int
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for _ in range(SEED_PLACEMENT_RETRIES):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        chosen: list[tuple[int, int]] = []
        blocked: set[tuple[int, int]] = set()
        for cell in shuffled:
            if cell in blocked:
                continue
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
            cr, cc = cell
            for rr in range(cr - 1, cr + 2):
                for cc2 in range(cc - 1, cc + 2):
                    blocked.add((rr, cc2))
    return None


def _frame_direction(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "up"
    if gr == r1 and c0 < gc < c1:
        return "down"
    if gc == c0 and r0 < gr < r1:
        return "left"
    if gc == c1 and r0 < gr < r1:
        return "right"
    return None


def _nonadjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict] = []
    for comp in _components(g == FRAME_COLOR):
        pts = set(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = _perimeter_cells(r0, c0, r1, c1)
        if not pts <= border:
            continue
        if len(pts) != len(border) - 1:
            continue

        missing = list(border - pts)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if int(g[gap]) != BACKGROUND:
            continue

        direction_name = _frame_direction(gap, r0, c0, r1, c1)
        if direction_name is None:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or np.any(interior == FRAME_COLOR):
            continue
        seed_mask = (interior != BACKGROUND) & (interior != FRAME_COLOR)
        if not np.any(seed_mask):
            continue
        seed_colors = {int(v) for v in interior[seed_mask]}
        if len(seed_colors) != 1:
            continue

        seed_points = [(r0 + 1 + int(rr), c0 + 1 + int(cc)) for rr, cc in np.argwhere(seed_mask)]
        if not _nonadjacent(seed_points):
            continue

        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": DIRECTION_VECTORS[direction_name],
                "direction_name": direction_name,
                "seed_color": next(iter(seed_colors)),
                "seed_count": len(seed_points),
                "seed_points": seed_points,
            }
        )

    infos.sort(key=lambda info: (info["bbox"][0], info["bbox"][1], info["bbox"][2], info["bbox"][3]))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a same-color branch outward from each grey frame's unique border gap using the number of interior seeds as the branch length.

    Input structure:
    - Background is black.
    - One or more grey one-pixel-thick rectangular frames appear on the grid.
    - Each grey rectangle has exactly one missing border pixel that forms a gap on one side.
    - Inside each frame are isolated single-pixel seeds; within one frame all seeds share the same color.

    Rule:
    - Detect every grey rectangular frame, recover its bounding box, and locate the unique missing border pixel.
    - The side containing the gap determines the outward growth direction.
    - Count the interior seeds N and read their color C, then draw a straight length-N line of color C starting just outside the gap and extending outward, clipped by the grid boundary.
    - Keep the original frame and the interior seeds unchanged.

    Key insight:
    - The gap encodes direction, while the interior seed count and seed color independently encode the branch length and branch color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for info in _find_frames(g):
        branch = _branch_cells(info["gap"], info["direction"], info["seed_count"], h, w)
        for r, c in branch:
            out[r, c] = info["seed_color"]

    return out


def _max_nonadjacent_capacity(inner_h: int, inner_w: int) -> int:
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _desired_frame_count(rng: random.Random, area: int) -> int:
    if area < 180:
        return rng.randint(1, 2)
    if area < 320:
        return rng.randint(1, 3)
    return rng.randint(1, 4)


def _placement_modes(rng: random.Random, n_frames: int) -> list[str]:
    if n_frames == 1:
        return [rng.choice(["full", "trunc"])]
    modes = ["full", "trunc"]
    while len(modes) < n_frames:
        modes.append(rng.choice(["full", "trunc"]))
    rng.shuffle(modes)
    return modes


def _frame_support_cells(
    bbox: tuple[int, int, int, int], branch: list[tuple[int, int]], h: int, w: int
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    support = {
        (r, c)
        for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1)
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1)
    }
    support |= _expand_cells(branch, h, w, radius=1)
    return support


def _place_one_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    seed_color: int,
    mode: str,
) -> dict | None:
    h, w = grid.shape
    directions = list(DIRECTION_VECTORS.items())

    for _ in range(PLACEMENT_RETRIES):
        direction_name, direction = rng.choice(directions)
        outer_h = rng.randint(*FRAME_OUTER_H_RANGE)
        outer_w = rng.randint(*FRAME_OUTER_W_RANGE)
        if outer_h > h or outer_w > w:
            continue

        r0 = rng.randint(0, h - outer_h)
        c0 = rng.randint(0, w - outer_w)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if direction_name in ("up", "down"):
            gap = (r0 if direction_name == "up" else r1, rng.randint(c0 + 1, c1 - 1))
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c0 if direction_name == "left" else c1)

        available = len(_branch_cells(gap, direction, max(h, w), h, w))
        if available < 1:
            continue

        inner_h = outer_h - 2
        inner_w = outer_w - 2
        max_seed_count = min(SEED_COUNT_MAX, _max_nonadjacent_capacity(inner_h, inner_w))
        if max_seed_count < 1:
            continue

        if mode == "full":
            max_full = min(max_seed_count, available)
            if max_full < 1:
                continue
            seed_count = rng.randint(1, max_full)
        else:
            if available >= max_seed_count:
                continue
            seed_count = rng.randint(available + 1, max_seed_count)

        branch = _branch_cells(gap, direction, seed_count, h, w)
        if not branch:
            continue

        bbox = (r0, c0, r1, c1)
        support = _frame_support_cells(bbox, branch, h, w)
        if any(reserved[r, c] for r, c in support):
            continue

        seeds = _sample_nonadjacent_cells(rng, r0, c0, r1, c1, seed_count)
        if seeds is None:
            continue

        border = _perimeter_cells(r0, c0, r1, c1)
        frame_cells = border - {gap}
        if any(int(grid[r, c]) != BACKGROUND for r, c in frame_cells):
            continue
        if any(int(grid[r, c]) != BACKGROUND for r, c in seeds):
            continue

        for r, c in frame_cells:
            grid[r, c] = FRAME_COLOR
        for r, c in seeds:
            grid[r, c] = seed_color
        for r, c in support:
            reserved[r, c] = True

        return {
            "bbox": bbox,
            "gap": gap,
            "direction": direction,
            "direction_name": direction_name,
            "seed_color": seed_color,
            "seed_count": seed_count,
            "branch_len": len(branch),
        }

    return None


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    frame_count = _desired_frame_count(rng, h * w)
    colors = rng.sample(SEED_COLORS, frame_count)
    modes = _placement_modes(rng, frame_count)

    planned: list[dict] = []
    for color, mode in zip(colors, modes):
        placed = _place_one_frame(grid, reserved, rng, color, mode)
        if placed is None:
            return None
        planned.append(placed)

    detected = _find_frames(grid)
    if len(detected) != frame_count:
        return None

    detected_by_bbox = {info["bbox"]: info for info in detected}
    saw_full = False
    saw_trunc = False
    for info in planned:
        matched = detected_by_bbox.get(info["bbox"])
        if matched is None:
            return None
        if matched["seed_color"] != info["seed_color"] or matched["seed_count"] != info["seed_count"]:
            return None
        visible = len(_branch_cells(matched["gap"], matched["direction"], matched["seed_count"], h, w))
        if visible < 1:
            return None
        if visible == matched["seed_count"]:
            saw_full = True
        else:
            saw_trunc = True

    if len(planned) == 1:
        if modes[0] == "full" and not saw_full:
            return None
        if modes[0] == "trunc" and not saw_trunc:
            return None
    else:
        if not (saw_full and saw_trunc):
            return None

    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")

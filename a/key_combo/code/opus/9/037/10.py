from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SEED_COLORS = (BLUE, GREEN, YELLOW, ORANGE)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_MIN = 13
GRID_MAX = 17
OUTER_MIN = 10
OUTER_MAX = 12
GEN_ATTEMPTS = 400
SEED_SEARCH_ATTEMPTS = 220


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_region(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    open_mask = grid != RED
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if open_mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc):
                add(nr, nc)

    enclosed = open_mask & ~seen
    comps = _components(enclosed)
    assert len(comps) == 1, "expected one enclosed region"
    return enclosed


def _find_seeds(grid: np.ndarray, region: np.ndarray) -> list[tuple[int, int, int]]:
    seed_mask = region & np.isin(grid, SEED_COLORS)
    comps = _components(seed_mask)
    assert 1 <= len(comps) <= 2, "expected 1 or 2 isolated seeds"

    seeds: list[tuple[int, int, int]] = []
    colors: set[int] = set()
    for comp in comps:
        assert len(comp) == 1, "each seed must be isolated"
        r, c = comp[0]
        color = int(grid[r, c])
        assert color not in colors, "seed colors must be distinct"
        colors.add(color)
        seeds.append((r, c, color))
    return sorted(seeds)


def _blocked(region: np.ndarray, r: int, c: int) -> bool:
    return not _in_bounds(region.shape, r, c) or not region[r, c]


def _reflect(region: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int, str]:
    row_block = _blocked(region, r + dr, c)
    col_block = _blocked(region, r, c + dc)
    if row_block and col_block:
        return -dr, -dc, "corner"
    if row_block:
        return -dr, dc, "horizontal"
    if col_block:
        return dr, -dc, "vertical"
    return -dr, -dc, "corner"


def _largest_black_rectangle(
    grid: np.ndarray, region: np.ndarray
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, int, int, int]] = []
    best_area = 0
    h, w = grid.shape
    allowed = (grid == BLACK) & region

    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if not allowed[r0, c0]:
                continue
            for r1 in range(r0 + 1, h):
                for c1 in range(c0 + 1, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if np.all(allowed[r0 : r1 + 1, c0 : c1 + 1]):
                        rect = (r0, c0, r1, c1)
                        if area > best_area:
                            best_area = area
                            candidates = [rect]
                        else:
                            candidates.append(rect)

    if not candidates:
        return None, []
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return candidates[0], candidates


def _simulate(
    grid: np.ndarray,
    *,
    seed_order: list[tuple[int, int, int]] | None = None,
    use_reflection: bool = True,
    fill_rectangle: bool = True,
) -> dict:
    region = _enclosed_region(grid)
    seeds = _find_seeds(grid, region)
    ordered = seeds if seed_order is None else seed_order
    out = grid.copy()
    seed_positions = {(r, c): color for r, c, color in seeds}
    region_area = int(np.count_nonzero(region))
    seed_infos: list[dict] = []

    for r0, c0, color in ordered:
        painted = {(r0, c0)}
        painted_cells: set[tuple[int, int]] = set()
        info = {
            "seed": (r0, c0, color),
            "painted": painted_cells,
            "steps": 0,
            "reflections": 0,
            "horizontal": 0,
            "vertical": 0,
            "corner": 0,
            "stops": {"trapped": 0, "loop": 0, "limit": 0, "blocked": 0},
        }
        for dr0, dc0 in DIAGONALS:
            r, c = r0, c0
            dr, dc = dr0, dc0
            steps = 0
            while steps <= region_area * 2:
                nr, nc = r + dr, c + dc
                if _blocked(region, nr, nc):
                    if not use_reflection:
                        info["stops"]["blocked"] += 1
                        break
                    dr2, dc2, kind = _reflect(region, r, c, dr, dc)
                    info["reflections"] += 1
                    info[kind] += 1
                    if _blocked(region, r + dr2, c + dc2):
                        info["stops"]["trapped"] += 1
                        break
                    dr, dc = dr2, dc2
                    continue
                if (nr, nc) in painted:
                    info["stops"]["loop"] += 1
                    break
                r, c = nr, nc
                steps += 1
                info["steps"] += 1
                if steps % 2 == 1:
                    painted.add((r, c))
                    painted_cells.add((r, c))
                    out[r, c] = color
            else:
                info["stops"]["limit"] += 1
        seed_infos.append(info)

    rect, maxima = _largest_black_rectangle(out, region)
    before_fill = out.copy()
    if fill_rectangle:
        assert rect is not None, "expected at least one 2x2 black rectangle"
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY

    return {
        "region": region,
        "seeds": seeds,
        "seed_infos": seed_infos,
        "seed_positions": seed_positions,
        "pre_fill": before_fill,
        "final": out,
        "largest_rect": rect,
        "max_rectangles": maxima,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Grow diagonal seed rays inside a red enclosure, reflect them off the wall, then gray-fill the largest untouched rectangle.

    Input structure:
    - The grid has a black background and one closed red one-cell-thick wall.
    - The wall encloses exactly one interior region, which contains 1-2 isolated colored seed pixels.
    - Seed colors are distinct and come from blue, green, yellow, and orange.

    Rule:
    - Identify the unique region that is enclosed by the red wall.
    - Process interior seeds in row-major order; from each seed, launch four diagonal rays.
    - Each ray advances one cell per step, paints only on odd-numbered steps, reflects from horizontal or vertical walls by flipping the matching direction component, flips both components at corners, and stops if it is trapped, loops onto a pixel already painted by the same seed, or exceeds twice the interior area in steps.
    - After all rays finish, find the largest all-black rectangle of height and width at least 2 that remains inside the enclosure; break area ties by topmost then leftmost top-left corner, and fill that rectangle gray.

    Key insight:
    - The output depends on combining enclosure detection, parity-preserving diagonal growth, wall-based ray reflection, and a final maximal-empty-rectangle search on the reflected coverage pattern.
    """
    return _simulate(np.asarray(grid, dtype=int), fill_rectangle=True)["final"]


def _boundary_of_shape(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    wall = np.zeros_like(shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not shape[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(shape.shape, nr, nc) or not shape[nr, nc]:
                    wall[r, c] = True
                    break
    return wall


def _deep_cells(region: np.ndarray) -> list[tuple[int, int]]:
    deep = []
    h, w = region.shape
    for r in range(h):
        for c in range(w):
            if not region[r, c]:
                continue
            if all(_in_bounds(region.shape, r + dr, c + dc) and region[r + dr, c + dc] for dr, dc in DIR4):
                deep.append((r, c))
    random.shuffle(deep)
    return deep


def _make_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _make_l_shape(h: int, w: int) -> np.ndarray:
    shape = np.ones((h, w), dtype=bool)
    notch_h = random.randint(3, h - 4)
    notch_w = random.randint(3, w - 4)
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        shape[:notch_h, :notch_w] = False
    elif corner == "tr":
        shape[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        shape[h - notch_h :, :notch_w] = False
    else:
        shape[h - notch_h :, w - notch_w :] = False
    return shape


def _place_shape_on_canvas(shape: np.ndarray, canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = canvas_shape
    sh, sw = shape.shape
    top = random.randint(1, h - sh - 1)
    left = random.randint(1, w - sw - 1)
    canvas = np.zeros((h, w), dtype=bool)
    canvas[top : top + sh, left : left + sw] = shape
    return canvas


def _build_input(
    wall: np.ndarray, seeds: list[tuple[int, int]], colors: list[int]
) -> np.ndarray:
    grid = np.zeros(wall.shape, dtype=int)
    grid[wall] = RED
    for (r, c), color in zip(seeds, colors):
        grid[r, c] = color
    return grid


def _reverse_seed_order(seeds: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return list(reversed(seeds))


def _validate_candidate(grid: np.ndarray, *, require_corner: bool, require_order_matter: bool) -> bool:
    raw = _simulate(grid, fill_rectangle=False)
    pre = raw["pre_fill"]
    no_reflect = _simulate(grid, use_reflection=False, fill_rectangle=False)["pre_fill"]

    if np.array_equal(pre, no_reflect):
        return False
    if raw["largest_rect"] is None:
        return False
    max_rects = raw["max_rectangles"]
    if not max_rects:
        return False
    r0, c0, r1, c1 = raw["largest_rect"]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if area < 4:
        return False
    if area > int(np.count_nonzero(raw["region"])) // 2:
        return False
    normal = _simulate(grid, fill_rectangle=True)
    final = normal["final"]
    if np.count_nonzero(final == GREY) != area:
        return False
    if np.count_nonzero(pre == BLACK) < 4:
        return False

    total_corner = 0
    total_h = 0
    total_v = 0
    for info in normal["seed_infos"]:
        if info["reflections"] < 2:
            return False
        total_corner += info["corner"]
        total_h += info["horizontal"]
        total_v += info["vertical"]
    if total_h == 0 or total_v == 0:
        return False
    if require_corner and total_corner == 0:
        return False

    for r, c, color in normal["seeds"]:
        if final[r, c] != color:
            return False

    if len(normal["seeds"]) == 2 and require_order_matter:
        reversed_order = _reverse_seed_order(normal["seeds"])
        alt = _simulate(grid, seed_order=reversed_order, fill_rectangle=True)["final"]
        if np.array_equal(final, alt):
            return False

    if len(normal["seeds"]) == 2 and require_order_matter:
        paints = [info["painted"] for info in normal["seed_infos"]]
        if not (paints[0] & paints[1]):
            return False

    return True


def _candidate_seed_sets(cells: list[tuple[int, int]], count: int) -> list[list[tuple[int, int]]]:
    if count == 1:
        return [[cell] for cell in cells[:SEED_SEARCH_ATTEMPTS]]
    pairs = list(combinations(cells, 2))
    random.shuffle(pairs)
    out: list[list[tuple[int, int]]] = []
    for a, b in pairs[:SEED_SEARCH_ATTEMPTS]:
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 4:
            out.append([a, b])
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    outer_h = random.randint(OUTER_MIN, min(OUTER_MAX, h - 2))
    outer_w = random.randint(OUTER_MIN, min(OUTER_MAX, w - 2))
    shape_type = random.choice(["rect", "l"])
    require_corner = random.random() < 0.55
    want_two_seeds = random.random() < 0.6
    require_order_matter = want_two_seeds and random.random() < 0.65

    local_shape = _make_rect_shape(outer_h, outer_w) if shape_type == "rect" else _make_l_shape(outer_h, outer_w)
    shape = _place_shape_on_canvas(local_shape, (h, w))
    wall = _boundary_of_shape(shape)
    region = shape & ~wall
    if len(_components(region)) != 1:
        return None
    if np.count_nonzero(region) < 34:
        return None

    deep = _deep_cells(region)
    seed_count = 2 if want_two_seeds and len(deep) >= 2 else 1
    if len(deep) < seed_count:
        return None

    for seeds in _candidate_seed_sets(deep, seed_count):
        colors = random.sample(SEED_COLORS, seed_count)
        ordered_seeds = sorted(seeds)
        grid = _build_input(wall, ordered_seeds, colors)
        if _validate_candidate(grid, require_corner=require_corner, require_order_matter=require_order_matter):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-growth puzzle")

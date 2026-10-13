from __future__ import annotations

from dataclasses import dataclass
import numpy as np


BG = 0
TOP = 1
BOTTOM = 2
LEFT = 3
RIGHT = 4
WALL = 5

FRAME_COLORS = (TOP, BOTTOM, LEFT, RIGHT)
GENERATION_ATTEMPTS = 400
_GEN_CALL_COUNT = 0


@dataclass(frozen=True)
class RegionRecord:
    coords: np.ndarray
    centroid_r: float
    centroid_c: float
    fill_color: int
    start: tuple[int, int]
    direction: tuple[int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _frame_bounds(grid: np.ndarray) -> tuple[int, int, int, int]:
    top_rows = np.where(np.any(grid == TOP, axis=1))[0]
    bottom_rows = np.where(np.any(grid == BOTTOM, axis=1))[0]
    left_cols = np.where(np.any(grid == LEFT, axis=0))[0]
    right_cols = np.where(np.any(grid == RIGHT, axis=0))[0]
    assert top_rows.size > 0 and bottom_rows.size > 0 and left_cols.size > 0 and right_cols.size > 0
    top_r = int(top_rows.min())
    bottom_r = int(bottom_rows.max())
    left_c = int(left_cols.min())
    right_c = int(right_cols.max())
    assert top_r < bottom_r and left_c < right_c
    return top_r, left_c, bottom_r, right_c


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _nearest_region_cell(coords: np.ndarray, centroid_r: float, centroid_c: float) -> tuple[int, int]:
    best = None
    best_key = None
    for r, c in coords:
        key = (
            abs(float(r) - centroid_r) + abs(float(c) - centroid_c),
            abs(float(r) - centroid_r),
            abs(float(c) - centroid_c),
            int(r),
            int(c),
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (int(r), int(c))
    assert best is not None
    return best


def _same(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


def _analyze_regions(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[RegionRecord]]:
    top_r, left_c, bottom_r, right_c = _frame_bounds(grid)
    inside = grid[top_r + 1 : bottom_r, left_c + 1 : right_c]
    regions = _components(inside == BG)
    assert regions, "expected at least one enclosed black region"

    records: list[RegionRecord] = []
    for coords_local in regions:
        coords = coords_local.copy()
        coords[:, 0] += top_r + 1
        coords[:, 1] += left_c + 1
        centroid_r, centroid_c = coords.mean(axis=0)

        d_top = centroid_r - top_r
        d_bottom = bottom_r - centroid_r
        d_left = centroid_c - left_c
        d_right = right_c - centroid_c
        near_tb = min(d_top, d_bottom)
        near_lr = min(d_left, d_right)
        assert not _same(near_tb, near_lr), "cross-axis branch must be unique"

        if near_tb < near_lr:
            assert not _same(d_left, d_right), "left/right choice must be unique"
            fill_color = LEFT if d_left < d_right else RIGHT
        else:
            assert not _same(d_top, d_bottom), "top/bottom choice must be unique"
            fill_color = TOP if d_top < d_bottom else BOTTOM

        start = (_round_half_up(centroid_r), _round_half_up(centroid_c))
        if not np.any(np.all(coords == start, axis=1)):
            start = _nearest_region_cell(coords, centroid_r, centroid_c)
        direction = {
            TOP: (-1, 0),
            BOTTOM: (1, 0),
            LEFT: (0, -1),
            RIGHT: (0, 1),
        }[fill_color]
        records.append(
            RegionRecord(
                coords=coords,
                centroid_r=float(centroid_r),
                centroid_c=float(centroid_c),
                fill_color=int(fill_color),
                start=start,
                direction=direction,
            )
        )

    return (top_r, left_c, bottom_r, right_c), records


def _filled_grid(grid: np.ndarray, records: list[RegionRecord]) -> tuple[np.ndarray, np.ndarray]:
    out = np.array(grid, dtype=int, copy=True)
    region_id = np.full(grid.shape, -1, dtype=int)
    for idx, record in enumerate(records):
        rr = record.coords[:, 0]
        cc = record.coords[:, 1]
        out[rr, cc] = record.fill_color
        region_id[rr, cc] = idx
    return out, region_id


def _on_target_edge(color: int, r: int, c: int, bounds: tuple[int, int, int, int]) -> bool:
    top_r, left_c, bottom_r, right_c = bounds
    if color == TOP:
        return r == top_r and left_c <= c <= right_c
    if color == BOTTOM:
        return r == bottom_r and left_c <= c <= right_c
    if color == LEFT:
        return c == left_c and top_r <= r <= bottom_r
    return c == right_c and top_r <= r <= bottom_r


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the color of the nearer cross-axis frame side, then connect it toward that matching side.

    Input structure:
    - A black background contains one colored rectangular frame whose top, bottom,
      left, and right sides are blue, red, green, and yellow respectively.
    - Inside the frame, one-pixel grey horizontal and/or vertical walls partition
      the black interior into several disjoint enclosed regions.
    - Each enclosed region is a 4-connected black component bounded by frame sides
      and grey walls.

    Rule:
    - Find every enclosed black region inside the frame and compute its centroid.
    - Compare the centroid's nearest top/bottom distance against its nearest
      left/right distance to decide the dominant axis.
    - If the region is nearer to the horizontal frame sides, fill it with the
      nearer vertical-side color; otherwise fill it with the nearer horizontal-side
      color.
    - After all fills, process regions from upper centroid to lower centroid
      (breaking ties left to right) and extend a straight line from the region's
      centroid toward the matching frame side, painting only black or grey cells
      and stopping before any other filled region.

    Key insight:
    - The correct fill color comes from the closer side on the opposite axis, not
      from the immediately nearest frame side itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bounds, records = _analyze_regions(g)
    out, region_id = _filled_grid(g, records)

    order = sorted(range(len(records)), key=lambda i: (records[i].centroid_r, records[i].centroid_c))
    h, w = out.shape
    for idx in order:
        record = records[idx]
        color = record.fill_color
        dr, dc = record.direction
        r, c = record.start
        while 0 <= r < h and 0 <= c < w:
            if region_id[r, c] == idx:
                pass
            elif _on_target_edge(color, r, c, bounds):
                out[r, c] = color
                break
            elif out[r, c] in (BG, WALL):
                out[r, c] = color
            else:
                break
            r += dr
            c += dc

    return out


def _sample_odd(rng: np.random.Generator, low: int, high: int) -> int:
    choices = [v for v in range(low, high + 1) if v % 2 == 1]
    assert choices
    return int(rng.choice(choices))


def _sample_partition(total: int, parts: int, rng: np.random.Generator, min_size: int = 3) -> list[int] | None:
    if min_size % 2 == 0 or total < parts * min_size:
        return None
    remaining = total - parts * min_size
    if remaining < 0 or remaining % 2 != 0:
        return None
    sizes = [min_size] * parts
    for _ in range(remaining // 2):
        sizes[int(rng.integers(parts))] += 2
    rng.shuffle(sizes)
    return sizes


def _build_grid(
    rng: np.random.Generator,
    horizontal_sizes: list[int],
    vertical_sizes: list[int],
) -> np.ndarray:
    h_walls = max(0, len(horizontal_sizes) - 1)
    v_walls = max(0, len(vertical_sizes) - 1)
    inner_h = sum(horizontal_sizes) + h_walls
    inner_w = sum(vertical_sizes) + v_walls
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    margin_top = int(rng.integers(1, 4))
    margin_bottom = int(rng.integers(1, 4))
    margin_left = int(rng.integers(1, 4))
    margin_right = int(rng.integers(1, 4))
    grid = np.zeros((frame_h + margin_top + margin_bottom, frame_w + margin_left + margin_right), dtype=int)

    top_r = margin_top
    left_c = margin_left
    bottom_r = top_r + frame_h - 1
    right_c = left_c + frame_w - 1

    grid[top_r, left_c : right_c + 1] = TOP
    grid[bottom_r, left_c : right_c + 1] = BOTTOM
    grid[top_r : bottom_r + 1, left_c] = LEFT
    grid[top_r : bottom_r + 1, right_c] = RIGHT

    cursor = top_r + 1
    for size in horizontal_sizes[:-1]:
        cursor += size
        grid[cursor, left_c + 1 : right_c] = WALL
        cursor += 1

    cursor = left_c + 1
    for size in vertical_sizes[:-1]:
        cursor += size
        grid[top_r + 1 : bottom_r, cursor] = WALL
        cursor += 1

    return grid


def _make_cross_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inner_h = _sample_odd(rng, 9, 15)
    inner_w = _sample_odd(rng, 9, 15)
    horizontal_sizes = _sample_partition(inner_h - 1, 2, rng, min_size=3)
    vertical_sizes = _sample_partition(inner_w - 1, 2, rng, min_size=3)
    if horizontal_sizes is None or vertical_sizes is None:
        return None
    return _build_grid(rng, horizontal_sizes, vertical_sizes)


def _make_stacked_attempt(rng: np.random.Generator) -> np.ndarray | None:
    walls = int(rng.choice([2, 3]))
    parts = walls + 1
    min_inner_h = parts * 3 + walls
    inner_h = _sample_odd(rng, min_inner_h, min(min_inner_h + 6, 19))
    inner_w = _sample_odd(rng, 5, 9)
    horizontal_sizes = _sample_partition(inner_h - walls, parts, rng, min_size=3)
    if horizontal_sizes is None:
        return None
    return _build_grid(rng, horizontal_sizes, [inner_w])


def _make_side_attempt(rng: np.random.Generator) -> np.ndarray | None:
    walls = int(rng.choice([2, 3]))
    parts = walls + 1
    min_inner_w = parts * 3 + walls
    inner_h = _sample_odd(rng, 5, 9)
    inner_w = _sample_odd(rng, min_inner_w, min(min_inner_w + 6, 19))
    vertical_sizes = _sample_partition(inner_w - walls, parts, rng, min_size=3)
    if vertical_sizes is None:
        return None
    return _build_grid(rng, [inner_h], vertical_sizes)


def _validate_attempt(grid: np.ndarray, mode: str) -> bool:
    if grid is None or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        bounds, records = _analyze_regions(grid)
        out = main(grid)
    except AssertionError:
        return False

    if not (3 <= len(records) <= 5):
        return False

    fill_colors = {record.fill_color for record in records}
    branch_counts = {
        "horizontal_target": sum(record.fill_color in (TOP, BOTTOM) for record in records),
        "vertical_target": sum(record.fill_color in (LEFT, RIGHT) for record in records),
    }
    filled_only, _ = _filled_grid(grid, records)
    wall_painted = bool(np.any((grid == WALL) & (out != filled_only)))
    if np.array_equal(out, grid) or np.array_equal(out, filled_only):
        visible_line = wall_painted
    else:
        visible_line = True

    if mode == "cross":
        return branch_counts["horizontal_target"] > 0 and branch_counts["vertical_target"] > 0 and len(fill_colors) >= 3
    if mode == "stacked":
        return fill_colors.issubset({TOP, BOTTOM}) and {TOP, BOTTOM}.issubset(fill_colors) and wall_painted and visible_line
    return fill_colors.issubset({LEFT, RIGHT}) and {LEFT, RIGHT}.issubset(fill_colors) and wall_painted and visible_line


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    rng = np.random.default_rng()
    cycle = ("cross", "stacked", "side")
    preferred = cycle[_GEN_CALL_COUNT % len(cycle)]
    _GEN_CALL_COUNT += 1

    for _ in range(GENERATION_ATTEMPTS):
        if preferred == "cross":
            mode = "cross" if rng.random() < 0.7 else str(rng.choice(["stacked", "side"]))
        elif preferred == "stacked":
            mode = "stacked" if rng.random() < 0.7 else str(rng.choice(["cross", "side"]))
        else:
            mode = "side" if rng.random() < 0.7 else str(rng.choice(["cross", "stacked"]))

        if mode == "cross":
            grid = _make_cross_attempt(rng)
        elif mode == "stacked":
            grid = _make_stacked_attempt(rng)
        else:
            grid = _make_side_attempt(rng)

        if grid is not None and _validate_attempt(grid, mode):
            return grid

    raise RuntimeError("failed to generate a valid input grid")

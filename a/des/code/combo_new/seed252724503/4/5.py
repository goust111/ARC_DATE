from __future__ import annotations

import numpy as np


RNG = np.random.default_rng()
PALETTE = tuple(range(1, 10))
MODES = ("free", "boundary", "block", "overlap")
GEN_RETRY_LIMIT = 400


def _extract_towers(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    h, w = map(int, g.shape)
    occupied_cols: list[int] = []
    towers: list[dict[str, int]] = []

    for c in range(w):
        rows = np.flatnonzero(g[:, c] != 0)
        if rows.size == 0:
            continue

        occupied_cols.append(int(c))
        if int(rows[-1]) != h - 1:
            raise AssertionError("each occupied column must reach the bottom edge")
        expected = np.arange(int(rows[0]), h)
        if rows.size != expected.size or not np.array_equal(rows, expected):
            raise AssertionError("tower columns must form contiguous suffixes")

        r = int(rows[0])
        while r < h:
            color = int(g[r, c])
            r1 = r
            while r1 + 1 < h and int(g[r1 + 1, c]) == color:
                r1 += 1
            towers.append(
                {
                    "top": int(r),
                    "bottom": int(r1),
                    "col": int(c),
                    "height": int(r1 - r + 1),
                    "color": int(color),
                }
            )
            r = int(r1 + 1)

    if not towers:
        raise AssertionError("expected at least one tower")
    if any(occupied_cols[i + 1] - occupied_cols[i] < 2 for i in range(len(occupied_cols) - 1)):
        raise AssertionError("occupied columns must be separated by at least one black column")

    return sorted(towers, key=lambda tower: (tower["col"], tower["top"]))


def _solve(grid: np.ndarray, *, collect_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    towers = _extract_towers(g)
    h, w = map(int, g.shape)
    tower_mask = g != 0
    out = g.copy()

    col_counts: dict[int, int] = {}
    for tower in towers:
        col = int(tower["col"])
        col_counts[col] = col_counts.get(col, 0) + 1

    stats = {
        "boundary": 0,
        "block": 0,
        "full": 0,
        "overlap": 0,
        "ray_pixels": 0,
        "tower_count": len(towers),
        "stacked_columns": sum(count > 1 for count in col_counts.values()),
    }

    for tower in towers:
        r0 = int(tower["top"])
        c0 = int(tower["col"])
        ray_len = int(tower["height"])
        color = int(tower["color"])

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            reason = "full"
            for step in range(1, ray_len + 1):
                r = int(r0 + dr * step)
                c = int(c0 + dc * step)
                if not (0 <= r < h and 0 <= c < w):
                    reason = "boundary"
                    break
                if bool(tower_mask[r, c]):
                    reason = "block"
                    break
                if int(out[r, c]) == 0:
                    out[r, c] = color
                    stats["ray_pixels"] += 1
                else:
                    stats["overlap"] += 1

            stats[reason] += 1

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal color rays from each vertical signal tower, using the tower height as the ray length.

    Input structure:
    - The background is black.
    - Non-black cells appear only in width-1 vertical tower columns, and any used columns are separated by at least one black column.
    - In an occupied column, colored cells form one contiguous stack from the bottom upward; color changes within that stack split it into distinct towers.

    Rule:
    - Read each monochrome vertical run as one tower with color C and height H.
    - Process towers from left to right by column; within a column, process higher towers first.
    - From the top pixel of each tower, cast four diagonal rays (up-left, up-right, down-left, down-right) for up to H steps.
    - Stop a ray immediately before the grid boundary or any tower pixel, and never overwrite tower cells.
    - When a later ray reaches a cell that an earlier tower has already colored, keep the earlier color and continue the ray past that crossing.

    Key insight:
    - A tower's visible height is also the hidden control value: it sets the maximum diagonal propagation distance of that tower's four rays.
    """
    return _solve(grid, collect_stats=False)


def _split_positive(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    if parts == 1:
        return [int(total)]
    cuts = sorted(RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    prev = 0
    values: list[int] = []
    for cut in cuts + [int(total)]:
        values.append(int(cut - prev))
        prev = int(cut)
    return values


def _segment_heights(total_height: int, top_height: int) -> list[int]:
    if not (1 <= top_height <= total_height):
        raise AssertionError("invalid top segment height")

    remaining = int(total_height - top_height)
    if remaining == 0:
        return [int(top_height)]

    extra_cap = min(2, remaining)
    extra_segments = 0
    if extra_cap > 0 and RNG.random() < 0.45:
        extra_segments = int(RNG.integers(1, extra_cap + 1))
    return [int(top_height)] + _split_positive(remaining, extra_segments) if extra_segments else [int(top_height), int(remaining)]


def _column_colors(count: int) -> list[int]:
    colors: list[int] = []
    prev = None
    for _ in range(count):
        choices = [color for color in PALETTE if color != prev]
        color = int(RNG.choice(choices))
        colors.append(color)
        prev = color
    return colors


def _paint_column(grid: np.ndarray, col: int, top_row: int, top_height: int) -> None:
    h = int(grid.shape[0])
    total_height = int(h - top_row)
    heights = _segment_heights(total_height, top_height)
    colors = _column_colors(len(heights))

    r = int(top_row)
    for seg_h, color in zip(heights, colors):
        grid[r : r + seg_h, int(col)] = int(color)
        r += int(seg_h)


def _available_columns(width: int, taken: list[int], banned: set[int] | None = None) -> list[int]:
    banned = banned or set()
    return [
        int(c)
        for c in range(width)
        if c not in banned and all(abs(int(c) - int(col)) >= 2 for col in taken)
    ]


def _sample_extra_spec(height: int, width: int, taken: list[int], banned: set[int] | None = None) -> tuple[int, int, int] | None:
    available = _available_columns(width, taken, banned)
    if not available:
        return None

    col = int(RNG.choice(available))
    total_height = int(RNG.integers(1, min(height, 10) + 1))
    top_row = int(height - total_height)
    top_height = int(RNG.integers(1, min(5, total_height) + 1))
    return col, top_row, top_height


def _build_grid(height: int, width: int, specs: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros((int(height), int(width)), dtype=int)
    for col, top_row, top_height in sorted(specs):
        _paint_column(grid, int(col), int(top_row), int(top_height))
    return grid


def _free_scene() -> np.ndarray | None:
    height = int(RNG.integers(10, 23))
    width = int(RNG.integers(10, 25))

    top_height = int(RNG.integers(1, 4))
    feasible_cols = list(range(top_height, width - top_height))
    if not feasible_cols:
        return None
    col0 = int(RNG.choice(feasible_cols))
    top_row0 = int(RNG.integers(top_height, height - top_height + 1))

    specs = [(col0, top_row0, top_height)]
    taken = [col0]
    banned = set(range(max(0, col0 - top_height), min(width, col0 + top_height + 1)))

    extra_limit = min(3, len(_available_columns(width, taken, banned)))
    if extra_limit == 0:
        return None
    extra_count = int(RNG.integers(1, extra_limit + 1))
    for _ in range(extra_count):
        spec = _sample_extra_spec(height, width, taken, banned)
        if spec is None:
            break
        specs.append(spec)
        taken.append(int(spec[0]))

    return _build_grid(height, width, specs)


def _boundary_scene() -> np.ndarray | None:
    height = int(RNG.integers(9, 23))
    width = int(RNG.integers(8, 25))

    edge_choices = [0, width - 1]
    if width >= 4:
        edge_choices.extend([1, width - 2])
    col0 = int(RNG.choice(edge_choices))
    total_height = int(RNG.integers(2, min(height, 10) + 1))
    top_row0 = int(height - total_height)
    top_height0 = int(RNG.integers(1, min(6, total_height) + 1))

    specs = [(col0, top_row0, top_height0)]
    taken = [col0]

    extra_limit = min(3, len(_available_columns(width, taken)))
    if extra_limit == 0:
        return None
    extra_count = int(RNG.integers(1, extra_limit + 1))
    for _ in range(extra_count):
        spec = _sample_extra_spec(height, width, taken)
        if spec is None:
            break
        specs.append(spec)
        taken.append(int(spec[0]))

    return _build_grid(height, width, specs)


def _block_scene() -> np.ndarray | None:
    height = int(RNG.integers(10, 24))
    width = int(RNG.integers(9, 25))
    feasible_steps = [step for step in (2, 3, 4) if step < width]
    if not feasible_steps:
        return None

    step = int(RNG.choice(feasible_steps))
    col_left = int(RNG.integers(0, width - step))
    col_right = int(col_left + step)

    top_row_left = int(RNG.integers(0, height - step))
    top_height_left = int(RNG.integers(step, min(6, height - top_row_left) + 1))
    hit_row = int(top_row_left + step)

    right_low = max(0, hit_row - 3)
    top_row_right = int(RNG.integers(right_low, hit_row + 1))
    top_height_right = int(RNG.integers(1, min(6, height - top_row_right) + 1))

    specs = [
        (col_left, top_row_left, top_height_left),
        (col_right, top_row_right, top_height_right),
    ]
    taken = [col_left, col_right]
    banned = set(range(col_left + 1, col_right))

    extra_limit = min(2, len(_available_columns(width, taken, banned)))
    if extra_limit > 0:
        extra_count = int(RNG.integers(0, extra_limit + 1))
        for _ in range(extra_count):
            spec = _sample_extra_spec(height, width, taken, banned)
            if spec is None:
                break
            specs.append(spec)
            taken.append(int(spec[0]))

    return _build_grid(height, width, specs)


def _overlap_scene() -> np.ndarray | None:
    height = int(RNG.integers(10, 24))
    width = int(RNG.integers(9, 25))
    max_half_gap = min(4, (width - 1) // 2)
    if max_half_gap < 1:
        return None

    half_gap = int(RNG.integers(1, max_half_gap + 1))
    distance = int(2 * half_gap)
    col_left = int(RNG.integers(0, width - distance))
    col_right = int(col_left + distance)

    max_top_row = int(height - half_gap)
    if max_top_row < half_gap:
        return None
    top_row = int(RNG.integers(half_gap, max_top_row + 1))
    max_top_height = min(6, height - top_row)
    if max_top_height < half_gap:
        return None

    top_height_left = int(RNG.integers(half_gap, max_top_height + 1))
    top_height_right = int(RNG.integers(half_gap, max_top_height + 1))

    specs = [
        (col_left, top_row, top_height_left),
        (col_right, top_row, top_height_right),
    ]
    taken = [col_left, col_right]
    banned = set(range(col_left + 1, col_right))

    extra_limit = min(2, len(_available_columns(width, taken, banned)))
    if extra_limit > 0:
        extra_count = int(RNG.integers(0, extra_limit + 1))
        for _ in range(extra_count):
            spec = _sample_extra_spec(height, width, taken, banned)
            if spec is None:
                break
            specs.append(spec)
            taken.append(int(spec[0]))

    return _build_grid(height, width, specs)


def _mode_scene(mode: str) -> np.ndarray | None:
    if mode == "free":
        return _free_scene()
    if mode == "boundary":
        return _boundary_scene()
    if mode == "block":
        return _block_scene()
    if mode == "overlap":
        return _overlap_scene()
    raise AssertionError(f"unknown mode: {mode}")


def _valid_mode(mode: str, stats: dict[str, int]) -> bool:
    if stats["tower_count"] < 2 or stats["ray_pixels"] < 4:
        return False
    if mode == "free":
        return stats["full"] > 0
    if mode == "boundary":
        return stats["boundary"] > 0
    if mode == "block":
        return stats["block"] > 0
    if mode == "overlap":
        return stats["overlap"] > 0
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        mode = str(RNG.choice(MODES))
        grid = _mode_scene(mode)
        if grid is None:
            continue
        try:
            out, stats = _solve(grid, collect_stats=True)
        except AssertionError:
            continue
        if not _valid_mode(mode, stats):
            continue
        if out.shape != grid.shape or max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid signal-tower sample")

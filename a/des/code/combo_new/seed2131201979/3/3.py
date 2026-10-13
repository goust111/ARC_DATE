import numpy as np
from collections import Counter


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()

# Generator configuration.
ENTRY_COUNT_RANGE = (2, 6)  # high is exclusive
GRID_H_RANGE = (12, 29)
GRID_W_RANGE = (16, 31)
LEGEND_COL_RANGE = (0, 4)
MAX_RADIUS = 4
MAX_EXTRA_SEEDS = 3
GEN_RETRIES = 400
SEED_RETRIES = 300


def _find_legend_entries(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        cols = np.flatnonzero(grid[r] != BG)
        if len(cols) == 0:
            continue
        c = int(cols[0])
        color = int(grid[r, c])
        if color == GREY:
            continue
        end = c + 1
        while end < w and int(grid[r, end]) == GREY:
            end += 1
        radius = end - (c + 1)
        if not (1 <= radius <= MAX_RADIUS):
            continue
        if end < w and int(grid[r, end]) != BG:
            continue
        candidates.append((r, c, color, radius))

    assert candidates, "legend not found"
    col_counts = Counter(c for _, c, _, _ in candidates)
    best = max(col_counts.values())
    legend_col = min(c for c, count in col_counts.items() if count == best)
    entries = [(r, c, color, radius) for r, c, color, radius in candidates if c == legend_col]
    entries.sort(key=lambda item: item[0])
    colors = [color for _, _, color, _ in entries]
    assert len(colors) == len(set(colors)), "legend colors must be unique"

    mask = np.zeros_like(grid, dtype=bool)
    for r, c, _color, radius in entries:
        mask[r, c : c + 1 + radius] = True
    return entries, mask


def _paint_diamond(
    out: np.ndarray,
    color: int,
    center_r: int,
    center_c: int,
    radius: int,
    protected: np.ndarray,
) -> None:
    h, w = out.shape
    for dr in range(-radius, radius + 1):
        rr = center_r + dr
        if not (0 <= rr < h):
            continue
        span = radius - abs(dr)
        c0 = max(0, center_c - span)
        c1 = min(w, center_c + span + 1)
        for cc in range(c0, c1):
            if not protected[rr, cc]:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-coded diamond regions from seeds using a left-side legend.

    Input structure:
    - Black background.
    - On the left, a vertical legend stacks entries made of one colored pixel followed by a same-row gray line segment of length 1-4.
    - Elsewhere, isolated single-pixel seeds reuse the legend colors.

    Rule:
    - Read each legend entry from top to bottom.
    - The colored pixel gives a seed color; the adjacent gray segment length gives that color's growth radius.
    - For every seed of that color, fill all cells whose Manhattan distance from the seed is at most the radius, using the seed color.
    - Later legend entries overwrite earlier diamond fills where they overlap.
    - Keep the legend pixels unchanged in the output.

    Key insight:
    - The gray segment length is the hidden numeric code, the grown shape is a Manhattan diamond, and legend row order resolves overlaps.
    """
    g = np.asarray(grid, dtype=int)
    entries, legend_mask = _find_legend_entries(g)

    out = g.copy()
    for _row, _col, color, radius in entries:
        seeds = np.argwhere((g == color) & (~legend_mask))
        for r, c in seeds:
            _paint_diamond(out, int(color), int(r), int(c), int(radius), legend_mask)
    return out


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _isolated_from_existing(r: int, c: int, placed: list[dict]) -> bool:
    for item in placed:
        if max(abs(r - item["r"]), abs(c - item["c"])) <= 1:
            return False
    return True


def _sample_seed_position(
    h: int,
    w: int,
    legend_right: int,
    radius: int,
    placed: list[dict],
    overlap_with: dict | None,
    forbid_overlap: bool,
) -> tuple[int, int] | None:
    row_low = radius
    row_high = h - radius
    col_low = legend_right + radius + 2
    col_high = w - radius
    if row_low >= row_high or col_low >= col_high:
        return None

    for _ in range(SEED_RETRIES):
        r = int(RNG.integers(row_low, row_high))
        c = int(RNG.integers(col_low, col_high))
        if not _isolated_from_existing(r, c, placed):
            continue
        if forbid_overlap:
            bad = False
            for item in placed:
                if _manhattan((r, c), (item["r"], item["c"])) <= radius + item["radius"]:
                    bad = True
                    break
            if bad:
                continue
        if overlap_with is not None:
            dist = _manhattan((r, c), (overlap_with["r"], overlap_with["c"]))
            if dist > radius + overlap_with["radius"]:
                continue
            if dist < 2:
                continue
        return r, c
    return None


def _has_later_overlap(placed: list[dict]) -> bool:
    for i, first in enumerate(placed):
        for second in placed[i + 1 :]:
            if first["entry_idx"] == second["entry_idx"]:
                continue
            if _manhattan((first["r"], first["c"]), (second["r"], second["c"])) <= first["radius"] + second["radius"]:
                return True
    return False


def _legend_rows(h: int, count: int) -> list[int]:
    rows = RNG.choice(np.arange(1, h - 1), size=count, replace=False)
    return sorted(int(v) for v in rows)


def _gen_attempt() -> np.ndarray | None:
    entry_count = int(RNG.integers(*ENTRY_COUNT_RANGE))
    radii = [int(RNG.integers(1, MAX_RADIUS + 1)) for _ in range(entry_count)]
    max_radius = max(radii)
    legend_col = int(RNG.integers(*LEGEND_COL_RANGE))

    min_h = max(GRID_H_RANGE[0], 2 * max_radius + 3, entry_count + 3)
    min_w = max(GRID_W_RANGE[0], legend_col + 2 * max_radius + MAX_RADIUS + 6)
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    h = int(RNG.integers(min_h, GRID_H_RANGE[1]))
    w = int(RNG.integers(min_w, GRID_W_RANGE[1]))
    if entry_count > h - 2:
        return None

    grid = np.full((h, w), BG, dtype=int)
    legend_rows = _legend_rows(h, entry_count)
    colors = list(RNG.choice(np.array(COLOR_POOL), size=entry_count, replace=False))
    entries: list[dict] = []
    legend_right = legend_col
    for row, color, radius in zip(legend_rows, colors, radii):
        color = int(color)
        radius = int(radius)
        grid[row, legend_col] = color
        grid[row, legend_col + 1 : legend_col + 1 + radius] = GREY
        entries.append({"row": int(row), "color": color, "radius": radius})
        legend_right = max(legend_right, legend_col + radius)

    overlap_mode = bool(RNG.integers(0, 2))
    seed_counts = [1] * entry_count
    for _ in range(int(RNG.integers(0, MAX_EXTRA_SEEDS + 1))):
        idx = int(RNG.integers(0, entry_count))
        if seed_counts[idx] < 3:
            seed_counts[idx] += 1

    placed: list[dict] = []

    if overlap_mode and entry_count >= 2:
        first_idx = int(RNG.integers(0, entry_count - 1))
        second_idx = int(RNG.integers(first_idx + 1, entry_count))

        first = _sample_seed_position(
            h,
            w,
            legend_right,
            entries[first_idx]["radius"],
            placed,
            overlap_with=None,
            forbid_overlap=False,
        )
        if first is None:
            return None
        placed.append(
            {
                "entry_idx": first_idx,
                "r": first[0],
                "c": first[1],
                "radius": entries[first_idx]["radius"],
                "color": entries[first_idx]["color"],
            }
        )

        second = _sample_seed_position(
            h,
            w,
            legend_right,
            entries[second_idx]["radius"],
            placed,
            overlap_with=placed[0],
            forbid_overlap=False,
        )
        if second is None:
            return None
        placed.append(
            {
                "entry_idx": second_idx,
                "r": second[0],
                "c": second[1],
                "radius": entries[second_idx]["radius"],
                "color": entries[second_idx]["color"],
            }
        )
        seed_counts[first_idx] -= 1
        seed_counts[second_idx] -= 1

    for entry_idx, count in enumerate(seed_counts):
        for _ in range(count):
            pos = _sample_seed_position(
                h,
                w,
                legend_right,
                entries[entry_idx]["radius"],
                placed,
                overlap_with=None,
                forbid_overlap=True,
            )
            if pos is None:
                return None
            placed.append(
                {
                    "entry_idx": entry_idx,
                    "r": pos[0],
                    "c": pos[1],
                    "radius": entries[entry_idx]["radius"],
                    "color": entries[entry_idx]["color"],
                }
            )

    for item in placed:
        grid[item["r"], item["c"]] = item["color"]

    if overlap_mode and not _has_later_overlap(placed):
        return None
    if not overlap_mode:
        for i, first in enumerate(placed):
            for second in placed[i + 1 :]:
                if _manhattan((first["r"], first["c"]), (second["r"], second["c"])) <= first["radius"] + second["radius"]:
                    return None

    parsed_entries, legend_mask = _find_legend_entries(grid)
    if len(parsed_entries) != entry_count:
        return None
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if not np.array_equal(out[legend_mask], grid[legend_mask]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")

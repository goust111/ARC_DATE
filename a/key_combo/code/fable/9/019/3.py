import random
from collections import Counter

import numpy as np


BG = 0
SEP = 1
MARK = 5
PALETTE = [2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _compress_runs(values: np.ndarray) -> list[int]:
    seq: list[int] = []
    for value in map(int, values):
        if not seq or seq[-1] != value:
            seq.append(value)
    return seq


def _separator_color(partition: np.ndarray) -> int:
    row_hits: Counter[int] = Counter()
    col_hits: Counter[int] = Counter()
    for r in range(partition.shape[0]):
        if np.all(partition[r, :] == partition[r, 0]):
            row_hits[int(partition[r, 0])] += 1
    for c in range(partition.shape[1]):
        if np.all(partition[:, c] == partition[0, c]):
            col_hits[int(partition[0, c])] += 1
    candidates = [
        color
        for color in sorted(set(row_hits) | set(col_hits))
        if color != BG and row_hits[color] > 0 and col_hits[color] > 0
    ]
    assert len(candidates) == 1, "expected exactly one separator color"
    return candidates[0]


def _intervals(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    points = [-1] + sorted(cuts) + [size]
    spans: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        lo = points[i] + 1
        hi = points[i + 1] - 1
        assert lo <= hi, "empty region interval"
        spans.append((lo, hi))
    return spans


def _vote_counts(grid: np.ndarray, bbox: tuple[int, int, int, int], sep_color: int) -> Counter[int]:
    r0, c0, r1, c1 = bbox
    counts: Counter[int] = Counter()
    for value in map(int, grid[r0 : r1 + 1, c0 : c1 + 1].ravel()):
        if value not in (BG, MARK, sep_color):
            counts[value] += 1
    return counts


def _major_color(grid: np.ndarray, bbox: tuple[int, int, int, int], sep_color: int) -> int:
    counts = _vote_counts(grid, bbox, sep_color)
    assert counts, "each region must contain votes"
    most_common = counts.most_common()
    assert len(most_common) == 1 or most_common[0][1] > most_common[1][1], "vote tie is not allowed"
    return int(most_common[0][0])


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 4

    partition = g[:-1, :]
    band = g[-1, :]
    band_colors = _compress_runs(band)
    assert len(band_colors) >= 1
    assert all(color not in (BG, MARK) for color in band_colors)

    sep_color = _separator_color(partition)
    sep_rows = [r for r in range(partition.shape[0]) if np.all(partition[r, :] == sep_color)]
    sep_cols = [c for c in range(partition.shape[1]) if np.all(partition[:, c] == sep_color)]
    row_spans = _intervals(partition.shape[0], sep_rows)
    col_spans = _intervals(partition.shape[1], sep_cols)
    assert len(row_spans) >= 2 and len(col_spans) >= 2

    regions: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            regions[(rr, cc)] = (r0, c0, r1, c1)

    marker_coords = np.argwhere(partition == MARK)
    assert marker_coords.shape == (1, 2), "expected exactly one marker"
    mr, mc = map(int, marker_coords[0])

    majors: dict[tuple[int, int], int] = {}
    counts_by_region: dict[tuple[int, int], Counter[int]] = {}
    start: tuple[int, int] | None = None
    for cell, bbox in regions.items():
        r0, c0, r1, c1 = bbox
        if r0 <= mr <= r1 and c0 <= mc <= c1:
            start = cell
        counts = _vote_counts(partition, bbox, sep_color)
        counts_by_region[cell] = counts
        majors[cell] = _major_color(partition, bbox, sep_color)
    assert start is not None

    path = [start]
    current = start
    for color in band_colors:
        candidates = [
            nb
            for nb in _neighbors(current, len(row_spans), len(col_spans))
            if majors[nb] == color
        ]
        assert len(candidates) == 1, "each band step must have a unique adjacent destination"
        current = candidates[0]
        path.append(current)

    return {
        "grid": g,
        "partition": partition,
        "band_colors": band_colors,
        "sep": sep_color,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "regions": regions,
        "majors": majors,
        "counts": counts_by_region,
        "start": start,
        "path": path,
    }


def _render_output(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    visited = info["path"]
    terminal = visited[-1]

    for cell in visited[:-1]:
        r0, c0, r1, c1 = info["regions"][cell]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    r0, c0, r1, c1 = info["regions"][terminal]
    out[r0 : r1 + 1, c0 : c1 + 1] = info["majors"][terminal]
    out[out == MARK] = BG
    out[-1, :] = grid[-1, :]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the bottom color-band program across voted partition regions, erase the traversed regions, and light up the terminal region.

    Input structure:
    - The top of the grid is split by full separator rows and columns into rectangular regions.
    - Each region contains scattered vote pixels of one or two non-background, non-separator colors, and exactly one region also contains a single gray marker.
    - The bottom row is a detached color band made of consecutive color segments.

    Rule:
    - Compute every region's major color by counting its scattered vote pixels, ignoring background, separator pixels, and the gray marker.
    - Starting from the marker's region, read the bottom band from left to right after run-compression; for each band color, move to the unique 4-neighbor region whose major color matches that color.
    - In the output, every visited region except the final one is filled black, the final region is filled solid with its major color, unvisited regions stay unchanged, the separator lines and bottom band remain, and the gray marker disappears.

    Key insight:
    - The band is a movement program, but each step only becomes readable after recovering each region's identity by vote counting rather than by any single visible pixel.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    return _render_output(grid, info)


def _random_sizes(count: int, lo: int, hi: int) -> list[int]:
    return [random.randint(lo, hi) for _ in range(count)]


def _random_simple_path(rows: int, cols: int, moves: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        used = {start}
        while len(path) < moves + 1:
            options = [nb for nb in _neighbors(path[-1], rows, cols) if nb not in used]
            if not options:
                break
            nxt = random.choice(options)
            used.add(nxt)
            path.append(nxt)
        if len(path) != moves + 1:
            continue
        if any(
            len([nb for nb in _neighbors(path[i], rows, cols) if nb != path[i + 1]]) > 0
            for i in range(moves)
        ):
            return path
    return None


def _choose_step_colors(moves: int) -> list[int]:
    return random.sample(PALETTE, moves)


def _assign_major_colors(
    rows: int, cols: int, path: list[tuple[int, int]], step_colors: list[int]
) -> dict[tuple[int, int], int] | None:
    forbidden: dict[tuple[int, int], set[int]] = {
        (r, c): set() for r in range(rows) for c in range(cols)
    }
    for i, color in enumerate(step_colors):
        current = path[i]
        target = path[i + 1]
        for nb in _neighbors(current, rows, cols):
            if nb != target:
                forbidden[nb].add(color)

    majors: dict[tuple[int, int], int] = {}
    fixed = {path[i + 1]: step_colors[i] for i in range(len(step_colors))}
    for cell, color in fixed.items():
        if color in forbidden[cell]:
            return None
        majors[cell] = color

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)
    for cell in cells:
        if cell in majors:
            continue
        choices = [color for color in PALETTE if color not in forbidden[cell]]
        if not choices:
            return None
        majors[cell] = random.choice(choices)

    off_path = [cell for cell in cells if cell not in path]
    duplicate_ok = any(majors[cell] in step_colors for cell in off_path)
    if not duplicate_ok and off_path:
        candidates: list[tuple[tuple[int, int], int]] = []
        for cell in off_path:
            for color in step_colors:
                if color not in forbidden[cell]:
                    candidates.append((cell, color))
        if not candidates:
            return None
        cell, color = random.choice(candidates)
        majors[cell] = color

    return majors


def _segment_lengths(total: int, parts: int, min_len: int) -> list[int]:
    lengths = [min_len] * parts
    remaining = total - min_len * parts
    for _ in range(remaining):
        lengths[random.randrange(parts)] += 1
    return lengths


def _pick_other_color(major: int) -> int:
    choices = [color for color in PALETTE if color != major]
    return random.choice(choices)


def _populate_region(
    patch: np.ndarray,
    major: int,
    reserve_empty: bool,
    force_two_colors: bool,
    force_counterexample: bool,
) -> tuple[np.ndarray, dict]:
    h, w = map(int, patch.shape)
    area = h * w
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)

    if force_counterexample:
        other = _pick_other_color(major)
        major_count = random.randint(3, min(6, area - 2))
        minor_count = random.randint(1, min(major_count - 1, max(1, area - major_count - 1)))
        total = major_count + minor_count
        if reserve_empty and total >= area:
            major_count -= 1
        if major_count <= minor_count:
            major_count = minor_count + 1
        chosen = sorted(cells)[: total]
        minor_cells = [chosen[0]]
        major_cells = chosen[1 : 1 + major_count]
        extra_minor = chosen[1 + major_count : total]
        for r, c in major_cells:
            patch[r, c] = major
        for r, c in minor_cells + extra_minor:
            patch[r, c] = other
        return patch, {"counts": Counter({major: major_count, other: minor_count}), "counterexample": True}

    use_two = force_two_colors or (area >= 12 and random.random() < 0.55) or (area < 12 and random.random() < 0.35)
    if use_two:
        other = _pick_other_color(major)
        max_total = area - (1 if reserve_empty else 0)
        major_count = random.randint(3, min(6, max_total - 1))
        minor_count = random.randint(1, min(major_count - 1, max_total - major_count))
        chosen = cells[: major_count + minor_count]
        for r, c in chosen[:major_count]:
            patch[r, c] = major
        for r, c in chosen[major_count : major_count + minor_count]:
            patch[r, c] = other
        return patch, {"counts": Counter({major: major_count, other: minor_count}), "counterexample": False}

    max_total = area - (1 if reserve_empty else 0)
    major_count = random.randint(2, min(6, max_total))
    for r, c in cells[:major_count]:
        patch[r, c] = major
    return patch, {"counts": Counter({major: major_count}), "counterexample": False}


def _build_grid(
    region_heights: list[int],
    region_widths: list[int],
    path: list[tuple[int, int]],
    majors: dict[tuple[int, int], int],
    step_colors: list[int],
) -> tuple[np.ndarray, dict]:
    rows = len(region_heights)
    cols = len(region_widths)
    top_h = sum(region_heights) + (rows - 1)
    width = sum(region_widths) + (cols - 1)
    grid = np.full((top_h + 1, width), BG, dtype=int)

    row_starts = []
    col_starts = []
    cur = 0
    for i, size in enumerate(region_heights):
        row_starts.append(cur)
        cur += size
        if i + 1 < rows:
            grid[cur, :] = SEP
            cur += 1
    cur = 0
    for j, size in enumerate(region_widths):
        col_starts.append(cur)
        cur += size
        if j + 1 < cols:
            grid[:, cur] = SEP
            cur += 1
    grid[-1, :] = BG

    special_region = random.choice(path[1:])
    two_color_regions = 0
    counterexample_regions = 0

    for rr in range(rows):
        for cc in range(cols):
            h = region_heights[rr]
            w = region_widths[cc]
            patch = np.zeros((h, w), dtype=int)
            cell = (rr, cc)
            reserve_empty = cell == path[0]
            force_counterexample = cell == special_region
            force_two = force_counterexample or (random.random() < 0.45)
            patch, meta = _populate_region(
                patch,
                majors[cell],
                reserve_empty=reserve_empty,
                force_two_colors=force_two,
                force_counterexample=force_counterexample,
            )
            if len(meta["counts"]) == 2:
                two_color_regions += 1
            if meta["counterexample"]:
                counterexample_regions += 1
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            grid[r0 : r0 + h, c0 : c0 + w] = patch

    start_rr, start_cc = path[0]
    r0 = row_starts[start_rr]
    c0 = col_starts[start_cc]
    h = region_heights[start_rr]
    w = region_widths[start_cc]
    empties = [(r0 + r, c0 + c) for r in range(h) for c in range(w) if grid[r0 + r, c0 + c] == BG]
    assert empties, "start region must keep one empty cell for the marker"
    grid[random.choice(empties)] = MARK

    min_len = 2 if width >= 2 * len(step_colors) else 1
    lengths = _segment_lengths(width, len(step_colors), min_len)
    cursor = 0
    for color, seg_len in zip(step_colors, lengths):
        grid[-1, cursor : cursor + seg_len] = color
        cursor += seg_len

    meta = {
        "two_color_regions": two_color_regions,
        "counterexample_regions": counterexample_regions,
    }
    return grid, meta


def _shortened_output(grid: np.ndarray, keep_steps: int) -> np.ndarray:
    info = _analyze(grid)
    info["path"] = info["path"][: keep_steps + 1]
    return _render_output(grid, info)


def _validate_instance(grid: np.ndarray) -> bool:
    info = _analyze(grid)
    rows = len(info["row_spans"])
    cols = len(info["col_spans"])
    path = info["path"]
    band_colors = info["band_colors"]
    majors = info["majors"]
    counts = info["counts"]

    if not (3 <= rows <= 5 and 3 <= cols <= 5):
        return False
    if not (2 <= len(band_colors) <= 6):
        return False
    if len(path) != len(band_colors) + 1 or len(set(path)) != len(path):
        return False

    if not any(len(_neighbors(path[i], rows, cols)) >= 2 for i in range(len(path) - 1)):
        return False
    if not any(len(counter) == 2 for counter in counts.values()):
        return False

    band_set = set(band_colors)
    off_path = [cell for cell in majors if cell not in set(path)]
    if off_path and not any(majors[cell] in band_set for cell in off_path):
        return False

    counterexample_ok = False
    for cell, bbox in info["regions"].items():
        counter = counts[cell]
        if len(counter) != 2:
            continue
        r0, c0, r1, c1 = bbox
        colored = [
            ((r, c), int(grid[r, c]))
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(grid[r, c]) not in (BG, MARK, info["sep"])
        ]
        if colored and colored[0][1] != majors[cell]:
            counterexample_ok = True
            break
    if not counterexample_ok:
        return False

    full = main(grid)
    for keep_steps in range(1, len(band_colors)):
        if np.array_equal(full, _shortened_output(grid, keep_steps)):
            return False

    if max(grid.shape) > 30 or max(full.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    region_heights = _random_sizes(rows, 3, 5)
    region_widths = _random_sizes(cols, 3, 5)
    top_h = sum(region_heights) + (rows - 1)
    width = sum(region_widths) + (cols - 1)
    if top_h + 1 > 30 or width > 30:
        return None

    max_moves = min(6, rows * cols - 1)
    moves = random.randint(2, max_moves)
    path = _random_simple_path(rows, cols, moves)
    if path is None:
        return None

    step_colors = _choose_step_colors(moves)
    majors = _assign_major_colors(rows, cols, path, step_colors)
    if majors is None:
        return None

    grid, _meta = _build_grid(region_heights, region_widths, path, majors, step_colors)
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-band partition instance")

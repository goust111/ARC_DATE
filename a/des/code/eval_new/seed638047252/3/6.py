from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _spans(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [size]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((left + 1, right - 1))
    return spans


def _separator_candidate_score(
    grid: np.ndarray, color: int
) -> tuple[int, int, int] | None:
    h, w = grid.shape
    sep_rows = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
    sep_cols = np.flatnonzero(np.all(grid == color, axis=0)).tolist()
    if not sep_rows and not sep_cols:
        return None

    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)
    if not row_spans or not col_spans:
        return None

    off_total = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, cnts = np.unique(patch, return_counts=True)
            dominant = int(cnts.max())
            off_total += int(patch.size - dominant)

    line_coverage = len(sep_rows) * w + len(sep_cols) * h
    return off_total, -line_coverage, color


def _find_separator_color(grid: np.ndarray) -> int:
    candidates = []
    for color in map(int, np.unique(grid)):
        score = _separator_candidate_score(grid, color)
        if score is not None:
            candidates.append(score)
    assert candidates, "expected at least one separator-line color"
    return min(candidates)[2]


def _parse_regions(grid: np.ndarray, sep: int) -> list[dict]:
    h, w = grid.shape
    sep_rows = np.flatnonzero(np.all(grid == sep, axis=1)).tolist()
    sep_cols = np.flatnonzero(np.all(grid == sep, axis=0)).tolist()
    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)
    assert row_spans and col_spans, "separator lines must define at least one region"

    regions: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals, cnts = np.unique(patch, return_counts=True)
            base = int(vals[cnts.argmax()])
            marker_coords = [
                (int(r0 + rr), int(c0 + cc))
                for rr, cc in np.argwhere(patch != base)
            ]
            regions.append(
                {
                    "row_index": ri,
                    "col_index": ci,
                    "bbox": (r0, c0, r1, c1),
                    "base": base,
                    "markers": marker_coords,
                }
            )
    return regions


def _fill_rectangle(
    out: np.ndarray,
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    color_a: int,
    color_b: int,
) -> None:
    r0, r1 = sorted((int(pos_a[0]), int(pos_b[0])))
    c0, c1 = sorted((int(pos_a[1]), int(pos_b[1])))
    if color_a == color_b:
        out[r0 : r1 + 1, c0 : c1 + 1] = color_a
        return

    if tuple(pos_b) < tuple(pos_a):
        color_a, color_b = color_b, color_a

    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    parity = (np.add.outer(np.arange(hh), np.arange(ww)) % 2) == 0
    patch = np.full((hh, ww), color_b, dtype=int)
    patch[parity] = color_a
    out[r0 : r1 + 1, c0 : c1 + 1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored markers and fill the rectangle spanned by each pair using their host regions' base colors.

    Input structure:
    - The background is black, while one-pixel-wide grey rows and columns run across
      the whole grid and partition it into rectangular regions.
    - Every region is otherwise filled by a single non-black, non-grey base color.
    - Some regions contain one-pixel marker dots whose colors differ from the region base
      color; each marker color appears exactly twice in the whole grid.

    Rule:
    - Detect the full-row/full-column separator color and use it to recover the rectangular regions.
    - In each region, treat the dominant color as the region base and collect every off-color cell
      as a marker belonging to that region.
    - Pair markers by equal color, take the two marker coordinates as opposite corners of a rectangle,
      and fill that entire rectangle.
    - If the two host regions have the same base color, fill the rectangle solidly with that color;
      otherwise fill it as a checkerboard whose top-left cell uses the base color of the lexicographically
      earlier marker, and let the fill overwrite the original contents.

    Key insight:
    Marker colors only indicate pairing. The rectangle fill comes from the dominant base colors of the
    regions that contain the two markers, so the true information source is the segmented background.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_color(g)
    regions = _parse_regions(g, sep)

    markers: dict[int, list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for region in regions:
        base = int(region["base"])
        for r, c in region["markers"]:
            markers[int(g[r, c])].append(((int(r), int(c)), base))

    assert markers, "expected at least one marker pair"

    out = g.copy()
    pair_order = sorted(
        markers.items(),
        key=lambda item: min(pos for pos, _base in item[1]),
    )
    for _marker_color, entries in pair_order:
        assert len(entries) == 2, "each marker color must appear exactly twice"
        (pos_a, base_a), (pos_b, base_b) = entries
        _fill_rectangle(out, pos_a, pos_b, int(base_a), int(base_b))
    return out


# --------------- generator ---------------

_BG = 0
_SEP = 5
_PALETTE = tuple(c for c in range(1, 10) if c != _SEP)
_MIN_REGION_SIZE = 3
_MAX_REGION_SIZE = 6
_MIN_AXIS_REGIONS = 2
_MAX_AXIS_REGIONS = 4
_MAX_RETRIES = 400


def _sample_axis_regions() -> list[int]:
    count = random.randint(_MIN_AXIS_REGIONS, _MAX_AXIS_REGIONS)
    return [random.randint(_MIN_REGION_SIZE, _MAX_REGION_SIZE) for _ in range(count)]


def _region_layout(
    row_sizes: list[int], col_sizes: list[int]
) -> tuple[int, int, list[dict]]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1

    row_spans = []
    cursor = 0
    for size in row_sizes:
        row_spans.append((cursor, cursor + size - 1))
        cursor += size + 1

    col_spans = []
    cursor = 0
    for size in col_sizes:
        col_spans.append((cursor, cursor + size - 1))
        cursor += size + 1

    regions: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            regions.append(
                {
                    "row_index": ri,
                    "col_index": ci,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return height, width, regions


def _sample_region_cell(region: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = region["bbox"]
    return random.randint(r0, r1), random.randint(c0, c1)


def _rect_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _pair_branches(num_pairs: int) -> list[str]:
    branches = ["same", "different"]
    while len(branches) < num_pairs:
        branches.append(random.choice(["same", "different"]))
    random.shuffle(branches)
    return branches


def _choose_pair_regions(
    regions: list[dict], pair_count: int
) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    rectangles: list[tuple[int, int, int, int]] = []

    for _ in range(pair_count):
        candidates = [
            (i, j)
            for i in range(len(regions))
            for j in range(i + 1, len(regions))
            if i not in used and j not in used
        ]
        random.shuffle(candidates)
        success = False

        for i, j in candidates:
            for _ in range(80):
                pos_a = _sample_region_cell(regions[i])
                pos_b = _sample_region_cell(regions[j])
                if pos_a[0] == pos_b[0] or pos_a[1] == pos_b[1]:
                    continue
                rect = (
                    min(pos_a[0], pos_b[0]),
                    min(pos_a[1], pos_b[1]),
                    max(pos_a[0], pos_b[0]),
                    max(pos_a[1], pos_b[1]),
                )
                if any(_rect_overlap(rect, other) for other in rectangles):
                    continue
                regions[i]["marker_pos"] = pos_a
                regions[j]["marker_pos"] = pos_b
                rectangles.append(rect)
                used.add(i)
                used.add(j)
                chosen.append((i, j))
                success = True
                break
            if success:
                break
        if not success:
            return None
    return chosen


def _render_expected(grid: np.ndarray, pair_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in pair_specs:
        _fill_rectangle(
            out,
            spec["pos_a"],
            spec["pos_b"],
            spec["base_a"],
            spec["base_b"],
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    row_sizes = _sample_axis_regions()
    col_sizes = _sample_axis_regions()
    height, width, regions = _region_layout(row_sizes, col_sizes)
    if max(height, width) > 30:
        return None

    region_count = len(regions)
    max_pairs = min(4, region_count // 2)
    if max_pairs < 2:
        return None
    pair_count = random.randint(2, max_pairs)

    fill_count = random.choice([3, 4])
    colors = list(_PALETTE)
    random.shuffle(colors)
    fill_palette = colors[:fill_count]
    marker_palette = colors[fill_count:]
    if len(marker_palette) < pair_count:
        return None

    pair_regions = _choose_pair_regions(regions, pair_count)
    if pair_regions is None:
        return None

    region_bases = [random.choice(fill_palette) for _ in regions]
    pair_specs: list[dict] = []
    branches = _pair_branches(pair_count)

    for marker_color, branch, (idx_a, idx_b) in zip(
        marker_palette[:pair_count], branches, pair_regions
    ):
        if branch == "same":
            base = random.choice(fill_palette)
            base_a = base_b = base
        else:
            base_a, base_b = random.sample(fill_palette, 2)

        region_bases[idx_a] = base_a
        region_bases[idx_b] = base_b
        pair_specs.append(
            {
                "marker_color": int(marker_color),
                "pos_a": tuple(regions[idx_a]["marker_pos"]),
                "pos_b": tuple(regions[idx_b]["marker_pos"]),
                "base_a": int(base_a),
                "base_b": int(base_b),
            }
        )

    grid = np.full((height, width), _SEP, dtype=int)
    for idx, region in enumerate(regions):
        r0, c0, r1, c1 = region["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(region_bases[idx])

    for spec in pair_specs:
        ra, ca = spec["pos_a"]
        rb, cb = spec["pos_b"]
        grid[ra, ca] = int(spec["marker_color"])
        grid[rb, cb] = int(spec["marker_color"])

    expected = _render_expected(grid, pair_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

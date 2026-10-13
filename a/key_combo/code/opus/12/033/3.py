from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, label_mask, neighbors4  # type: ignore  # noqa: E402


SEGMENT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _collect_segments(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    segments: list[dict[str, int | str]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 2, f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = coords.tolist()
        if r0 == r1:
            c0, c1 = sorted((int(c0), int(c1)))
            segments.append(
                {"orientation": "horizontal", "color": color, "r": int(r0), "c0": c0, "c1": c1}
            )
        else:
            assert c0 == c1, f"color {color} endpoints must share a row or a column"
            r0, r1 = sorted((int(r0), int(r1)))
            segments.append(
                {"orientation": "vertical", "color": color, "c": int(c0), "r0": r0, "r1": r1}
            )
    return segments


def _render_segments(
    shape: tuple[int, int], segments: list[dict[str, int | str]], *, vertical_priority: bool
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    first = "horizontal" if vertical_priority else "vertical"
    second = "vertical" if vertical_priority else "horizontal"
    for orientation in (first, second):
        for seg in segments:
            if seg["orientation"] != orientation:
                continue
            color = int(seg["color"])
            if orientation == "horizontal":
                out[int(seg["r"]), int(seg["c0"]) : int(seg["c1"]) + 1] = color
            else:
                out[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c"])] = color
    return out


def _edge_label_ids(labels: np.ndarray) -> set[int]:
    ids = set(map(int, labels[0]))
    ids.update(map(int, labels[-1]))
    ids.update(map(int, labels[:, 0]))
    ids.update(map(int, labels[:, -1]))
    ids.discard(0)
    return ids


def _region_vote(
    rendered: np.ndarray, region_mask: np.ndarray
) -> tuple[int, bool, Counter[int], dict[int, tuple[int, int]]]:
    counts: Counter[int] = Counter()
    first_touch: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < rendered.shape[0] and 0 <= nc < rendered.shape[1]:
                color = int(rendered[nr, nc])
                if color == Color.BLACK:
                    continue
                counts[color] += 1
                touch = (int(nr), int(nc))
                if color not in first_touch or touch < first_touch[color]:
                    first_touch[color] = touch
    assert counts, "each enclosed black region must touch at least one line pixel"
    best = max(counts.values())
    tied = [color for color, count in counts.items() if count == best]
    unique = len(tied) == 1
    winner = min(tied, key=lambda color: first_touch[color])
    return int(winner), unique, counts, first_touch


def _fill_regions(
    rendered: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels, n_labels = label_mask(rendered == Color.BLACK, connectivity=4)
    edge_ids = _edge_label_ids(labels)
    out = rendered.copy()
    regions: list[dict[str, object]] = []
    for label_id in range(1, n_labels + 1):
        if label_id in edge_ids:
            continue
        region_mask = labels == label_id
        winner, unique, counts, first_touch = _region_vote(rendered, region_mask)
        out[region_mask] = winner
        regions.append(
            {
                "label": int(label_id),
                "winner": int(winner),
                "unique": bool(unique),
                "counts": counts,
                "first_touch": first_touch,
                "area": int(np.count_nonzero(region_mask)),
                "anchor": tuple(map(int, np.argwhere(region_mask).min(axis=0))),
            }
        )
    regions.sort(key=lambda info: tuple(info["anchor"]))  # type: ignore[arg-type]
    return out, regions


def _solve_with_priority(
    grid: np.ndarray, *, vertical_priority: bool
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    segments = _collect_segments(grid)
    rendered = _render_segments(grid.shape, segments, vertical_priority=vertical_priority)
    filled, regions = _fill_regions(rendered)
    return rendered, filled, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored aligned markers, keep vertical colors at crossings, then fill enclosed black regions by boundary-adjacency vote.
    
    Input structure:
    - The grid has a black background and 3-5 non-black colors, each appearing exactly twice.
    - Each same-colored pair shares either one row or one column, so it defines one horizontal or vertical segment.
    - Different-color horizontal and vertical segments may cross; enclosed black pockets can emerge only after all segments are drawn.
    
    Rule:
    - Draw every same-colored pair as the inclusive straight segment between its two markers.
    - At each horizontal/vertical crossing, keep the vertical segment's color.
    - Find every 4-connected black region that does not connect to the grid boundary.
    - For each enclosed region, count 4-neighbor side contacts to the surrounding line colors and fill the region with the most frequent boundary color.
    - Black regions that touch the grid boundary remain black.
    
    Key insight:
    - The crossing rule changes actual boundary pixels of the enclosed region, so understanding which color survives at intersections is necessary before the adjacency vote can be predicted.
    """
    _rendered, filled, _regions = _solve_with_priority(np.asarray(grid, dtype=int), vertical_priority=True)
    return filled


def _make_template(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 18)
    width = rng.randint(12, 18)

    inner_h = rng.randint(3, 5)
    inner_w_max = min(7, width - 7)
    inner_w_min = inner_h + 1
    if inner_w_min > inner_w_max:
        return None
    inner_w = rng.randint(inner_w_min, inner_w_max)

    top = rng.randint(3, height - inner_h - 4)
    bottom = top + inner_h + 1
    left = rng.randint(3, width - inner_w - 4)
    right = left + inner_w + 1

    top_left = rng.randint(1, left - 1)
    top_right = rng.randint(right + 1, width - 2)
    bottom_left = rng.randint(1, left - 1)
    bottom_right = rng.randint(right + 1, width - 2)
    left_top = rng.randint(1, top - 1)
    left_bottom = rng.randint(bottom + 1, height - 2)
    right_top = rng.randint(1, top - 1)
    right_bottom = rng.randint(bottom + 1, height - 2)

    extra_col = rng.randint(left + 1, right - 1)
    extra_top = rng.randint(1, top - 1)

    colors = rng.sample(list(SEGMENT_COLORS), 5)
    segments = [
        {"orientation": "horizontal", "color": colors[0], "r": top, "c0": top_left, "c1": top_right},
        {
            "orientation": "horizontal",
            "color": colors[1],
            "r": bottom,
            "c0": bottom_left,
            "c1": bottom_right,
        },
        {"orientation": "vertical", "color": colors[2], "c": left, "r0": left_top, "r1": left_bottom},
        {"orientation": "vertical", "color": colors[3], "c": right, "r0": right_top, "r1": right_bottom},
        {"orientation": "vertical", "color": colors[4], "c": extra_col, "r0": extra_top, "r1": top},
    ]

    if rng.random() < 0.5:
        for seg in segments:
            if seg["orientation"] == "horizontal":
                c0 = width - 1 - int(seg["c1"])
                c1 = width - 1 - int(seg["c0"])
                seg["c0"], seg["c1"] = c0, c1
            else:
                seg["c"] = width - 1 - int(seg["c"])

    grid = np.zeros((height, width), dtype=int)
    for seg in segments:
        color = int(seg["color"])
        if seg["orientation"] == "horizontal":
            grid[int(seg["r"]), int(seg["c0"])] = color
            grid[int(seg["r"]), int(seg["c1"])] = color
        else:
            grid[int(seg["r0"]), int(seg["c"])] = color
            grid[int(seg["r1"]), int(seg["c"])] = color
    return grid


def _intersections(segments: list[dict[str, int | str]]) -> int:
    horizontals = [seg for seg in segments if seg["orientation"] == "horizontal"]
    verticals = [seg for seg in segments if seg["orientation"] == "vertical"]
    total = 0
    for h_seg in horizontals:
        r = int(h_seg["r"])
        c0 = int(h_seg["c0"])
        c1 = int(h_seg["c1"])
        for v_seg in verticals:
            c = int(v_seg["c"])
            r0 = int(v_seg["r0"])
            r1 = int(v_seg["r1"])
            if c0 <= c <= c1 and r0 <= r <= r1:
                total += 1
    return total


def _validate_generated(grid: np.ndarray) -> bool:
    segments = _collect_segments(grid)
    if not 3 <= len(segments) <= 5:
        return False
    if _intersections(segments) < 2:
        return False

    rendered_v, output_v, regions_v = _solve_with_priority(grid, vertical_priority=True)
    _rendered_h, output_h, regions_h = _solve_with_priority(grid, vertical_priority=False)

    if len(regions_v) != 1 or len(regions_h) != 1:
        return False
    if not all(bool(region["unique"]) for region in regions_v):
        return False
    if output_v.shape != grid.shape or output_h.shape != grid.shape:
        return False
    if np.array_equal(output_v, rendered_v):
        return False
    if regions_v[0]["winner"] == regions_h[0]["winner"]:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_template(rng)
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")

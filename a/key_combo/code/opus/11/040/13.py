from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox


BG = Color.BLACK
DIVIDER = Color.RED
STRIPE_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE]
MAX_GENERATION_ATTEMPTS = 400


def _divider_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]


def _region_bounds(grid: np.ndarray) -> list[tuple[int, int]]:
    dividers = _divider_columns(grid)
    bounds: list[tuple[int, int]] = []
    left = 0
    for c in dividers + [int(grid.shape[1])]:
        right = int(c) - 1
        if right >= left:
            bounds.append((int(left), int(right)))
        left = int(c) + 1
    assert 3 <= len(bounds) <= 5, "expected 3-5 regions"
    widths = {int(c1 - c0 + 1) for c0, c1 in bounds}
    assert len(widths) == 1, "regions must have equal width"
    return bounds


def _collapse(seq: list[int]) -> list[int]:
    out: list[int] = []
    for value in seq:
        value = int(value)
        if not out or value != out[-1]:
            out.append(value)
    return out


def _leftmost_occurrence(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.size > 0
    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    r, c = pts[0]
    return int(c), int(r)


def _analyze_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    H = int(g.shape[0])
    regions: list[dict[str, object]] = []
    for idx, (c0, c1) in enumerate(_region_bounds(g)):
        region = crop_bbox(g, (0, int(c0), H - 1, int(c1)))
        assert not np.any(region == DIVIDER), "divider color must stay outside region interiors"
        assert np.all(region == region[:, :1]), "each region must be filled with horizontal full-width stripes"
        raw = [int(v) for v in region[:, 0]]
        compressed = _collapse(raw)
        top_color = int(compressed[0])
        first_pos = _leftmost_occurrence(g, top_color)
        regions.append(
            {
                "index": int(idx),
                "bounds": (int(c0), int(c1)),
                "crop": region,
                "raw": raw,
                "compressed": compressed,
                "length": int(len(compressed)),
                "top_color": top_color,
                "top_first_pos": first_pos,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder red-separated stripe regions by the length of their compressed stripe sequence.
    
    Input structure:
    - The grid has black background semantics, with 3-5 equal-width rectangular regions laid out left-to-right.
    - Full-height red vertical divider columns separate the regions.
    - Each region is completely filled by horizontal full-width color stripes using non-red foreground colors.
    
    Rule:
    - Use the red divider columns to crop each region independently.
    - Read each region top-to-bottom, collapse consecutive duplicate row colors, and measure the resulting sequence length.
    - Sort the region contents by that compressed length from short to long, breaking ties by the leftmost first appearance of the region's top stripe color in the input.
    - Paste the original uncompressed region contents back into the left-to-right slots in sorted order, keeping the red divider columns unchanged.
    
    Key insight:
    - The sorting key is an abstract compressed stripe sequence, not a geometric property like width, area, or visible stripe density.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _analyze_regions(g)
    ordered = sorted(regions, key=lambda info: (int(info["length"]), tuple(info["top_first_pos"]), int(info["index"])))
    for slot, source in zip(regions, ordered):
        c0, c1 = slot["bounds"]
        out[:, int(c0) : int(c1) + 1] = np.asarray(source["crop"], dtype=int)
    return out


def _random_thicknesses(height: int, n: int) -> list[int]:
    assert n <= height <= 3 * n
    parts = [1] * int(n)
    remaining = int(height - n)
    while remaining > 0:
        choices = [i for i, value in enumerate(parts) if value < 3]
        idx = random.choice(choices)
        add = random.randint(1, min(2, remaining, 3 - parts[idx]))
        parts[idx] += add
        remaining -= add
    random.shuffle(parts)
    return parts


def _random_color_sequence(length: int, *, force_reuse: bool) -> list[int]:
    seq = [random.choice(STRIPE_COLORS)]
    reused = False
    for i in range(1, int(length)):
        reuse_candidates = [seq[j] for j in range(max(0, i - 3), i - 1) if seq[j] != seq[-1]]
        if force_reuse and not reused and reuse_candidates and random.random() < 0.65:
            color = random.choice(reuse_candidates)
            reused = True
        else:
            options = [color for color in STRIPE_COLORS if color != seq[-1]]
            color = random.choice(options)
        seq.append(int(color))
    if force_reuse and len(set(seq)) == len(seq):
        for j in range(len(seq) - 2):
            if seq[j] != seq[-2]:
                seq[-1] = int(seq[j])
                reused = True
                break
    assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
    assert not force_reuse or len(set(seq)) < len(seq)
    return seq


def _build_region(height: int, width: int, length: int, *, force_reuse: bool) -> tuple[np.ndarray, dict[str, object]]:
    compressed = _random_color_sequence(length, force_reuse=force_reuse)
    thicknesses = _random_thicknesses(height, length)
    region = np.full((int(height), int(width)), BG, dtype=int)
    r = 0
    for color, thick in zip(compressed, thicknesses):
        region[r : r + thick, :] = int(color)
        r += int(thick)
    assert r == height
    raw = [int(v) for v in region[:, 0]]
    assert _collapse(raw) == compressed
    return region, {
        "compressed": compressed,
        "length": int(length),
        "thicknesses": thicknesses,
        "unique_colors": int(len(set(compressed))),
        "max_thickness": int(max(thicknesses)),
        "top_color": int(compressed[0]),
    }


def _validate_instance(inp: np.ndarray) -> None:
    H, W = map(int, inp.shape)
    assert inp.ndim == 2 and H <= 30 and W <= 30
    regions = _analyze_regions(inp)
    lengths = [int(info["length"]) for info in regions]
    assert len(set(lengths)) == len(lengths), "compressed lengths must be unique"
    assert lengths != sorted(lengths), "input must require at least one swap"
    assert any(
        any(raw[i] == raw[i - 1] for i in range(1, len(raw)))
        for raw in (info["raw"] for info in regions)
    ), "need at least one thick stripe so compression matters"
    assert any(len(set(info["compressed"])) < int(info["length"]) for info in regions), "need nonconsecutive color reuse in at least one region"
    out = main(inp)
    out_regions = _analyze_regions(out)
    out_lengths = [int(info["length"]) for info in out_regions]
    assert out_lengths == sorted(lengths), "output regions must be sorted by compressed length"
    assert np.array_equal(out[:, _divider_columns(inp)], inp[:, _divider_columns(inp)]), "divider columns must stay fixed"
    target_order = [int(info["index"]) for info in sorted(regions, key=lambda x: int(x["length"]))]
    unique_order = [int(info["index"]) for info in sorted(regions, key=lambda x: (len(set(x["compressed"])), int(x["length"]), int(x["index"])))]
    top_order = [int(info["index"]) for info in sorted(regions, key=lambda x: (int(x["raw"].count(x["raw"][0])), int(x["length"]), int(x["index"])))]
    assert target_order != unique_order or target_order != top_order, "generator should avoid obvious shortcut orderings"
    assert not np.array_equal(inp, out), "instance must be non-trivial"


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choice([3, 4, 5])
    width = random.choice([3, 4])
    height = random.randint(9, 15)
    total_width = n_regions * width + (n_regions - 1)
    if total_width > 30 or height > 30:
        return None

    feasible_lengths = [length for length in range(2, height + 1) if length <= height <= 3 * length]
    if len(feasible_lengths) < n_regions:
        return None
    lengths = sorted(random.sample(feasible_lengths, n_regions))

    reuse_idx = random.randrange(n_regions)
    meta: list[dict[str, object]] = []
    regions: list[np.ndarray] = []
    for i, length in enumerate(lengths):
        region, info = _build_region(height, width, length, force_reuse=(i == reuse_idx))
        regions.append(region)
        info["index"] = int(i)
        meta.append(info)

    perm = list(range(n_regions))
    random.shuffle(perm)
    if perm == list(range(n_regions)):
        return None

    target_order = list(range(n_regions))
    unique_order = [idx for idx, _ in sorted(enumerate(meta), key=lambda x: (int(x[1]["unique_colors"]), int(x[1]["length"]), int(x[0])))]
    top_order = [idx for idx, _ in sorted(enumerate(meta), key=lambda x: (int(x[1]["thicknesses"][0]), int(x[1]["length"]), int(x[0])))]
    if target_order == unique_order == top_order:
        return None

    grid = np.full((height, total_width), BG, dtype=int)
    col = 0
    for slot in range(n_regions):
        grid[:, col : col + width] = regions[perm[slot]]
        col += width
        if slot != n_regions - 1:
            grid[:, col] = DIVIDER
            col += 1

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-13 input")

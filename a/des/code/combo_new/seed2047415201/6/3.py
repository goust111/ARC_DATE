from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
WIDTH_KEYS = (2, 3, 4, 5)
MAX_GENERATION_ATTEMPTS = 500


def _components(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and mask[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if comp["height"] * comp["width"] != comp["area"]:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == comp["color"]))


def _corner_distance(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return min(
        r0 + c0,
        r0 + (w - 1 - c1),
        (h - 1 - r1) + c0,
        (h - 1 - r1) + (w - 1 - c1),
    )


def _legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    comps = _components(grid, grid != bg)
    candidates_by_width: dict[int, list[dict]] = {width: [] for width in WIDTH_KEYS}
    for comp in comps:
        if not _is_solid_rectangle(grid, comp):
            continue
        if comp["width"] in candidates_by_width and comp["height"] <= 3:
            candidates_by_width[comp["width"]].append(comp)

    if any(not candidates_by_width[width] for width in WIDTH_KEYS):
        raise AssertionError("expected one legend bar of each width 2, 3, 4, and 5")

    best: tuple[tuple[int, int], tuple[dict, ...]] | None = None
    for combo in itertools.product(*(candidates_by_width[width] for width in WIDTH_KEYS)):
        if len({id(comp) for comp in combo}) != len(WIDTH_KEYS):
            continue
        r0 = min(comp["bbox"][0] for comp in combo)
        c0 = min(comp["bbox"][1] for comp in combo)
        r1 = max(comp["bbox"][2] for comp in combo)
        c1 = max(comp["bbox"][3] for comp in combo)
        dist = _corner_distance((r0, c0, r1, c1), (h, w))
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        score = (dist, area)
        if best is None or score < best[0]:
            best = (score, combo)

    assert best is not None, "expected a compact corner legend"
    mapping: dict[int, int] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for comp in best[1]:
        mapping[int(comp["width"])] = int(comp["color"])
        for r, c in comp["coords"]:
            legend_mask[int(r), int(c)] = True
    return mapping, legend_mask


def _line_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if comp["height"] == 1:
        if comp["area"] != comp["width"]:
            return None
        return {
            "orientation": "h",
            "fixed": int(r0),
            "start": int(c0),
            "end": int(c1),
            "length": int(comp["width"]),
        }
    if comp["width"] == 1:
        if comp["area"] != comp["height"]:
            return None
        return {
            "orientation": "v",
            "fixed": int(c0),
            "start": int(r0),
            "end": int(r1),
            "length": int(comp["height"]),
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the blank strips between paired line segments using a width-coded corner legend.

    Input structure:
    - The background is Black.
    - Each non-legend foreground color forms exactly two equal, parallel horizontal
      or vertical one-pixel-thick line segments.
    - A compact legend in one grid corner contains solid colored rectangle bars
      whose visible widths are 2, 3, 4, and 5 pixels.

    Rule:
    - Read the legend as a mapping from each bar width to that bar's color.
    - For every foreground line color, find its two parallel segments and count
      the number of Black pixels G between them.
    - Look up the legend bar with width G and fill only the Black pixels between
      the two segments with that legend color.
    - Keep the original line segments and the corner legend unchanged.

    Key insight:
    The distance between the paired segments is the lookup key; the fill color
    comes from the legend bar with the same visible width.
    """
    g = np.asarray(grid, dtype=int)
    bg = BACKGROUND
    out = g.copy()

    mapping, legend_mask = _legend_mapping(g, bg)
    line_mask = (g != bg) & ~legend_mask
    lines_by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in _components(g, line_mask):
        info = _line_info(comp)
        if info is not None:
            lines_by_color[int(comp["color"])].append(info)

    for color, lines in lines_by_color.items():
        if len(lines) != 2:
            continue
        first, second = lines
        if first["orientation"] != second["orientation"]:
            continue
        if first["length"] != second["length"]:
            continue
        if first["start"] != second["start"] or first["end"] != second["end"]:
            continue

        gap = abs(int(first["fixed"]) - int(second["fixed"])) - 1
        if gap not in mapping:
            continue
        fill_color = int(mapping[gap])

        lo = min(int(first["fixed"]), int(second["fixed"])) + 1
        hi = max(int(first["fixed"]), int(second["fixed"]))
        start, end = int(first["start"]), int(first["end"])
        if first["orientation"] == "h":
            region = out[lo:hi, start : end + 1]
        else:
            region = out[start : end + 1, lo:hi]
        region[region == bg] = fill_color

    return out


def _paint_legend(
    grid: np.ndarray, used: np.ndarray, colors_by_width: dict[int, int], corner: str, bar_height: int
) -> None:
    h, w = grid.shape
    gap = 1
    order = list(WIDTH_KEYS)
    random.shuffle(order)
    total_h = len(order) * bar_height + (len(order) - 1) * gap
    top = 0 if "top" in corner else h - total_h

    for i, width in enumerate(order):
        r0 = top + i * (bar_height + gap)
        c0 = 0 if "left" in corner else w - width
        grid[r0 : r0 + bar_height, c0 : c0 + width] = colors_by_width[width]

    c0 = 0 if "left" in corner else w - max(WIDTH_KEYS)
    used[max(0, top - 1) : min(h, top + total_h + 1), max(0, c0 - 1) : min(w, c0 + max(WIDTH_KEYS) + 1)] = True


def _can_reserve(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    h, w = used.shape
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(h, r0 + height + 1), min(w, c0 + width + 1)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _reserve(used: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    h, w = used.shape
    used[max(0, r0 - 1) : min(h, r0 + height + 1), max(0, c0 - 1) : min(w, c0 + width + 1)] = True


def _place_line_pair(
    grid: np.ndarray,
    expected: np.ndarray,
    used: np.ndarray,
    line_color: int,
    fill_color: int,
    orientation: str,
    gap: int,
) -> bool:
    h, w = grid.shape
    if orientation == "h":
        max_len = min(13, w)
        if max_len < 6 or h < gap + 2:
            return False
        lengths = list(range(6, max_len + 1))
        random.shuffle(lengths)
        for length in lengths:
            positions = [
                (r, c)
                for r in range(h - (gap + 2) + 1)
                for c in range(w - length + 1)
                if _can_reserve(used, r, c, gap + 2, length)
            ]
            random.shuffle(positions)
            for r0, c0 in positions[:40]:
                r1 = r0 + gap + 1
                grid[r0, c0 : c0 + length] = line_color
                grid[r1, c0 : c0 + length] = line_color
                expected[r0, c0 : c0 + length] = line_color
                expected[r1, c0 : c0 + length] = line_color
                expected[r0 + 1 : r1, c0 : c0 + length] = fill_color
                _reserve(used, r0, c0, gap + 2, length)
                return True
    else:
        max_len = min(13, h)
        if max_len < 6 or w < gap + 2:
            return False
        lengths = list(range(6, max_len + 1))
        random.shuffle(lengths)
        for length in lengths:
            positions = [
                (r, c)
                for r in range(h - length + 1)
                for c in range(w - (gap + 2) + 1)
                if _can_reserve(used, r, c, length, gap + 2)
            ]
            random.shuffle(positions)
            for r0, c0 in positions[:40]:
                c1 = c0 + gap + 1
                grid[r0 : r0 + length, c0] = line_color
                grid[r0 : r0 + length, c1] = line_color
                expected[r0 : r0 + length, c0] = line_color
                expected[r0 : r0 + length, c1] = line_color
                expected[r0 : r0 + length, c0 + 1 : c1] = fill_color
                _reserve(used, r0, c0, length, gap + 2)
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    expected = grid.copy()
    used = np.zeros((h, w), dtype=bool)

    colors = list(range(1, 10))
    random.shuffle(colors)
    legend_colors = colors[:4]
    line_colors = colors[4:]
    colors_by_width = {width: legend_colors[i] for i, width in enumerate(WIDTH_KEYS)}

    corner = random.choice(("top-left", "top-right", "bottom-left", "bottom-right"))
    bar_height = random.choice((1, 2, 2))
    _paint_legend(grid, used, colors_by_width, corner, bar_height)
    _paint_legend(expected, np.zeros_like(used), colors_by_width, corner, bar_height)

    n_pairs = random.randint(2, 4)
    selected_line_colors = line_colors[:n_pairs]
    if n_pairs == 4:
        gaps = list(WIDTH_KEYS)
        random.shuffle(gaps)
    else:
        gaps = random.sample(list(WIDTH_KEYS), n_pairs)

    orientations = ["h", "v"]
    while len(orientations) < n_pairs:
        orientations.append(random.choice(("h", "v")))
    random.shuffle(orientations)

    for line_color, gap, orientation in zip(selected_line_colors, gaps, orientations):
        placed = _place_line_pair(
            grid,
            expected,
            used,
            int(line_color),
            int(colors_by_width[gap]),
            orientation,
            int(gap),
        )
        if not placed:
            return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid line-gap legend input")

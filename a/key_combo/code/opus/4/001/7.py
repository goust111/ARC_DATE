from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
LINE_COLORS = (1, 2, 4, 7)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for inner_h in (2, 3):
        outer_h = inner_h + 2
        if outer_h > h or 4 > w:
            continue
        for r0 in range(h - outer_h + 1):
            for c0 in range(w - 4 + 1):
                sub = grid[r0 : r0 + outer_h, c0 : c0 + 4]
                border = np.zeros(sub.shape, dtype=bool)
                border[0, :] = True
                border[-1, :] = True
                border[:, 0] = True
                border[:, -1] = True
                inner = sub[1:-1, 1:-1]
                if not np.all(sub[border] == FRAME):
                    continue
                if inner.shape != (inner_h, 2):
                    continue
                if np.any((inner == BG) | (inner == FRAME)):
                    continue
                if len(set(map(int, inner.ravel()))) < 2:
                    continue
                candidates.append((r0, c0, r0 + outer_h - 1, c0 + 3))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one legend, found {len(candidates)}")
    bbox = candidates[0]
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    pairs = [(int(inner[r, 0]), int(inner[r, 1])) for r in range(inner.shape[0])]
    return bbox, _mask_from_bbox(grid.shape, bbox), pairs


def _extract_pairs(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    bg: int,
) -> list[dict[str, int | str]]:
    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere((grid != bg) & (~legend_mask)):
        color = int(grid[r, c])
        coords_by_color.setdefault(color, []).append((int(r), int(c)))

    pairs: list[dict[str, int | str]] = []
    for color, coords in sorted(coords_by_color.items()):
        if len(coords) != 2:
            raise ValueError(f"color {color} must appear exactly twice outside the legend")
        (r0, c0), (r1, c1) = coords
        if r0 == r1 and c0 != c1:
            pairs.append({"orientation": "h", "color": color, "r": r0, "c0": min(c0, c1), "c1": max(c0, c1)})
        elif c0 == c1 and r0 != r1:
            pairs.append({"orientation": "v", "color": color, "c": c0, "r0": min(r0, r1), "r1": max(r0, r1)})
        else:
            raise ValueError(f"color {color} markers must share one row or one column")
    if not pairs:
        raise ValueError("no drawable marker pairs found")
    return pairs


def _segment_cells(pair: dict[str, int | str]) -> list[tuple[int, int]]:
    if pair["orientation"] == "h":
        r = int(pair["r"])
        return [(r, c) for c in range(int(pair["c0"]), int(pair["c1"]) + 1)]
    c = int(pair["c"])
    return [(r, c) for r in range(int(pair["r0"]), int(pair["r1"]) + 1)]


def _draw_pairs(
    base: np.ndarray,
    pairs: list[dict[str, int | str]],
    legend_mask: np.ndarray,
    *,
    vertical_priority: bool,
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    order = ("h", "v") if vertical_priority else ("v", "h")
    for orientation in order:
        for pair in pairs:
            if pair["orientation"] != orientation:
                continue
            color = int(pair["color"])
            for r, c in _segment_cells(pair):
                if not legend_mask[r, c]:
                    out[r, c] = color
    return out


def _apply_chain(
    grid: np.ndarray,
    legend_pairs: list[tuple[int, int]],
    legend_mask: np.ndarray,
    bg: int,
    *,
    cascade: bool,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in np.argwhere((grid != bg) & (~legend_mask)):
        color = int(grid[r, c])
        for src, dst in legend_pairs:
            if color == int(src):
                color = int(dst)
                if not cascade:
                    break
        out[r, c] = color
    return out


def _apply_chain_color(color: int, legend_pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in legend_pairs:
        if current == int(src):
            current = int(dst)
    return current


def _chain_steps(color: int, legend_pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    steps = 0
    for src, dst in legend_pairs:
        if current == int(src):
            current = int(dst)
            steps += 1
    return steps


def _crossings(pairs: list[dict[str, int | str]]) -> list[tuple[int, int, int, int]]:
    horiz = [pair for pair in pairs if pair["orientation"] == "h"]
    vert = [pair for pair in pairs if pair["orientation"] == "v"]
    hits: list[tuple[int, int, int, int]] = []
    for h_pair in horiz:
        for v_pair in vert:
            hr = int(h_pair["r"])
            vc = int(v_pair["c"])
            if int(h_pair["c0"]) <= vc <= int(h_pair["c1"]) and int(v_pair["r0"]) <= hr <= int(v_pair["r1"]):
                hits.append((hr, vc, int(h_pair["color"]), int(v_pair["color"])))
    return hits


def _pair_signature(pair: dict[str, int | str]) -> tuple[int, str, int, int, int]:
    if pair["orientation"] == "h":
        return (
            int(pair["color"]),
            "h",
            int(pair["r"]),
            int(pair["c0"]),
            int(pair["c1"]),
        )
    return (
        int(pair["color"]),
        "v",
        int(pair["c"]),
        int(pair["r0"]),
        int(pair["r1"]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Draw same-color marker segments, keep vertical colors at crossings, then recolor them by the ordered legend.

    Input structure:
    - Black background with one grey rectangular frame enclosing a 2-column, 2-3-row legend in the upper-left area.
    - Outside the frame, each non-background color appears exactly twice as marker pixels, and each pair shares a row or a column.
    - At least one horizontal pair and one vertical pair are present so a line crossing can occur.

    Rule:
    - Read the legend rows from top to bottom as ordered substitutions from left color to right color; the grey frame is only a separator.
    - For every legend-external same-color pair, draw the full horizontal or vertical segment between the two markers, including endpoints.
    - If a horizontal and a vertical segment cross, the crossing pixel keeps the vertical segment color.
    - After all segments are drawn, recolor every non-background pixel outside the framed legend by scanning the legend rows top-to-bottom and applying the substitutions chain-style.
    - Leave the background and the full framed legend unchanged.

    Key insight:
    - A crossing pixel is determined by two coupled steps: the vertical line chooses its initial color at the intersection, and the legend order can then rewrite that color through one or more chained substitutions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    _legend_bbox, legend_mask, legend_pairs = _find_legend(g)
    pairs = _extract_pairs(g, legend_mask, bg)
    drawn = _draw_pairs(g, pairs, legend_mask, vertical_priority=True)
    return _apply_chain(drawn, legend_pairs, legend_mask, bg, cascade=True)


def _random_interval(lo: int, hi: int, min_cells: int) -> tuple[int, int] | None:
    if hi - lo + 1 < min_cells:
        return None
    start = random.randint(lo, hi - min_cells + 1)
    end = random.randint(start + min_cells - 1, hi)
    return start, end


def _place_extra_horizontal(
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    base_h: dict[str, int | str],
    base_v: dict[str, int | str],
    color: int,
) -> dict[str, int | str] | None:
    rows = [r for r in range(row_min, row_max + 1) if r != int(base_h["r"])]
    random.shuffle(rows)
    for r in rows:
        modes: list[str] = []
        if int(base_v["r0"]) <= r <= int(base_v["r1"]) and col_min <= int(base_v["c"]) - 2 and int(base_v["c"]) + 2 <= col_max:
            modes.append("cross")
        if _random_interval(col_min, int(base_v["c"]) - 1, 3) is not None:
            modes.append("left")
        if _random_interval(int(base_v["c"]) + 1, col_max, 3) is not None:
            modes.append("right")
        random.shuffle(modes)
        for mode in modes:
            if mode == "cross":
                c0 = random.randint(col_min, int(base_v["c"]) - 2)
                c1 = random.randint(int(base_v["c"]) + 2, col_max)
            elif mode == "left":
                interval = _random_interval(col_min, int(base_v["c"]) - 1, 3)
                if interval is None:
                    continue
                c0, c1 = interval
            else:
                interval = _random_interval(int(base_v["c"]) + 1, col_max, 3)
                if interval is None:
                    continue
                c0, c1 = interval
            return {"orientation": "h", "color": color, "r": r, "c0": c0, "c1": c1}
    return None


def _place_extra_vertical(
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    base_h: dict[str, int | str],
    base_v: dict[str, int | str],
    color: int,
) -> dict[str, int | str] | None:
    cols = [c for c in range(col_min, col_max + 1) if c != int(base_v["c"])]
    random.shuffle(cols)
    for c in cols:
        modes: list[str] = []
        if int(base_h["c0"]) <= c <= int(base_h["c1"]) and row_min <= int(base_h["r"]) - 2 and int(base_h["r"]) + 2 <= row_max:
            modes.append("cross")
        if _random_interval(row_min, int(base_h["r"]) - 1, 3) is not None:
            modes.append("above")
        if _random_interval(int(base_h["r"]) + 1, row_max, 3) is not None:
            modes.append("below")
        random.shuffle(modes)
        for mode in modes:
            if mode == "cross":
                r0 = random.randint(row_min, int(base_h["r"]) - 2)
                r1 = random.randint(int(base_h["r"]) + 2, row_max)
            elif mode == "above":
                interval = _random_interval(row_min, int(base_h["r"]) - 1, 3)
                if interval is None:
                    continue
                r0, r1 = interval
            else:
                interval = _random_interval(int(base_h["r"]) + 1, row_max, 3)
                if interval is None:
                    continue
                r0, r1 = interval
            return {"orientation": "v", "color": color, "c": c, "r0": r0, "r1": r1}
    return None


def _build_spec() -> dict | None:
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    legend_rows = 3 if random.random() < 0.6 else 2
    legend_top = random.randint(0, 1)
    legend_left = random.randint(0, 1)
    legend_bbox = (legend_top, legend_left, legend_top + legend_rows + 1, legend_left + 3)

    row_min = legend_bbox[2] + 2
    row_max = height - 2
    col_min = legend_bbox[3] + 2
    col_max = width - 2
    if row_max - row_min + 1 < 5 or col_max - col_min + 1 < 5:
        return None

    pair_count = 3 if random.random() < 0.6 else 2
    pair_colors = random.sample(list(LINE_COLORS), pair_count)
    h_color, v_color = random.sample(pair_colors, 2)
    chain_palette = [color for color in LINE_COLORS if color not in {h_color, v_color}]
    mid_color, final_color = random.sample(chain_palette, 2)

    h_row = random.randint(row_min + 2, row_max - 2)
    v_col = random.randint(col_min + 2, col_max - 2)
    h_c0 = random.randint(col_min, v_col - 2)
    h_c1 = random.randint(v_col + 2, col_max)
    v_r0 = random.randint(row_min, h_row - 2)
    v_r1 = random.randint(h_row + 2, row_max)

    base_h = {"orientation": "h", "color": h_color, "r": h_row, "c0": h_c0, "c1": h_c1}
    base_v = {"orientation": "v", "color": v_color, "c": v_col, "r0": v_r0, "r1": v_r1}
    pairs: list[dict[str, int | str]] = [base_h, base_v]

    if pair_count == 3:
        extra_color = next(color for color in pair_colors if color not in {h_color, v_color})
        extra_pair = None
        for _ in range(20):
            if random.random() < 0.5:
                extra_pair = _place_extra_horizontal(row_min, row_max, col_min, col_max, base_h, base_v, extra_color)
            else:
                extra_pair = _place_extra_vertical(row_min, row_max, col_min, col_max, base_h, base_v, extra_color)
            if extra_pair is not None:
                break
        if extra_pair is None:
            return None
        pairs.append(extra_pair)

    legend_pairs: list[tuple[int, int]] = [(v_color, mid_color), (mid_color, final_color)]
    if legend_rows == 3:
        extra_sources = [color for color in pair_colors if color not in {v_color, mid_color}]
        preferred = [color for color in extra_sources if color == h_color]
        src3 = random.choice(preferred or extra_sources)
        dst_choices = [color for color in LINE_COLORS if color not in {src3, final_color}]
        if not dst_choices:
            return None
        legend_pairs.append((src3, random.choice(dst_choices)))

    return {
        "shape": (height, width),
        "legend_bbox": legend_bbox,
        "legend_pairs": legend_pairs,
        "pairs": pairs,
    }


def _build_grid(spec: dict) -> np.ndarray:
    height, width = spec["shape"]
    grid = np.full((height, width), BG, dtype=int)
    r0, c0, r1, c1 = spec["legend_bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    inner = np.array(spec["legend_pairs"], dtype=int)
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner

    for pair in spec["pairs"]:
        color = int(pair["color"])
        if pair["orientation"] == "h":
            grid[int(pair["r"]), int(pair["c0"])] = color
            grid[int(pair["r"]), int(pair["c1"])] = color
        else:
            grid[int(pair["r0"]), int(pair["c"])] = color
            grid[int(pair["r1"]), int(pair["c"])] = color
    return grid


def _expected_output(grid: np.ndarray, spec: dict) -> np.ndarray:
    bg = _get_bg(grid)
    legend_mask = _mask_from_bbox(grid.shape, spec["legend_bbox"])
    drawn = _draw_pairs(grid, spec["pairs"], legend_mask, vertical_priority=True)
    return _apply_chain(drawn, spec["legend_pairs"], legend_mask, bg, cascade=True)


def _validate_spec(grid: np.ndarray, spec: dict) -> bool:
    try:
        legend_bbox, legend_mask, legend_pairs = _find_legend(grid)
        pairs = _extract_pairs(grid, legend_mask, _get_bg(grid))
    except Exception:
        return False

    if legend_bbox != tuple(spec["legend_bbox"]):
        return False
    if legend_pairs != list(spec["legend_pairs"]):
        return False

    found_pairs = sorted(_pair_signature(pair) for pair in pairs)
    expected_pairs = sorted(_pair_signature(pair) for pair in spec["pairs"])
    if found_pairs != expected_pairs:
        return False

    crossings = _crossings(pairs)
    if not crossings:
        return False
    if not any(_chain_steps(v_color, legend_pairs) >= 2 for _r, _c, _h_color, v_color in crossings):
        return False

    drawn = _draw_pairs(grid, pairs, legend_mask, vertical_priority=True)
    out = _apply_chain(drawn, legend_pairs, legend_mask, _get_bg(grid), cascade=True)
    if np.array_equal(drawn, out):
        return False

    alt_priority = _apply_chain(
        _draw_pairs(grid, pairs, legend_mask, vertical_priority=False),
        legend_pairs,
        legend_mask,
        _get_bg(grid),
        cascade=True,
    )
    if np.array_equal(out, alt_priority):
        return False

    alt_one_step = _apply_chain(drawn, legend_pairs, legend_mask, _get_bg(grid), cascade=False)
    if np.array_equal(out, alt_one_step):
        return False

    visible_cross = False
    for r, c, h_color, v_color in crossings:
        if out[r, c] != v_color and out[r, c] != h_color:
            if out[r, c] != alt_priority[r, c] and out[r, c] != alt_one_step[r, c]:
                visible_cross = True
                break
    if not visible_cross:
        return False

    if np.array_equal(grid, out):
        return False
    if not np.array_equal(out, _expected_output(grid, spec)):
        return False
    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _build_spec()
        if spec is None:
            continue
        grid = _build_grid(spec)
        if _validate_spec(grid, spec):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

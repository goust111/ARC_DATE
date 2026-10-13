from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
NON_BG_NON_GREY = [1, 2, 3, 4, 6, 7, 8, 9]

ROW_H = 5
ROW_GAP = 1
COUNT_BLOCK_W = 2
COUNT_STEP = 3
LEGEND_SEP = 2
STYLE_H = 5
STYLE_W = 7

STYLE_SOLID = "solid"
STYLE_DASHED = "dashed"
STYLE_DOUBLE = "double"
STYLES = (STYLE_SOLID, STYLE_DASHED, STYLE_DOUBLE)

GEN_RETRIES = 400


def _style_mask(style: str) -> np.ndarray:
    mask = np.zeros((STYLE_H, STYLE_W), dtype=bool)
    if style == STYLE_SOLID:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    if style == STYLE_DASHED:
        mask[0, ::2] = True
        mask[-1, ::2] = True
        mask[2, 0] = True
        mask[2, -1] = True
        return mask

    if style == STYLE_DOUBLE:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        mask[1, 1:-1] = True
        mask[-2, 1:-1] = True
        mask[1:-1, 1] = True
        mask[1:-1, -2] = True
        return mask

    raise ValueError(style)


STYLE_MASKS = {style: _style_mask(style) for style in STYLES}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _dominant_color(values: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _find_legend_mapping(g: np.ndarray) -> dict[int, str]:
    border_values = set(map(int, np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))))
    legend_colors = [value for value in border_values if value not in {BG, GREY}]

    mapping: dict[int, str] = {}
    for color in legend_colors:
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        if not pts:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == color
        if sub.shape[0] != ROW_H or sub.shape[1] < STYLE_W + LEGEND_SEP + COUNT_BLOCK_W:
            continue

        style_part = sub[:, -STYLE_W:]
        style = None
        for name, mask in STYLE_MASKS.items():
            if np.array_equal(style_part, mask):
                style = name
                break
        if style is None:
            continue

        width = sub.shape[1]
        count_zone_w = width - LEGEND_SEP - STYLE_W
        if count_zone_w < COUNT_BLOCK_W or (count_zone_w + 1) % COUNT_STEP != 0:
            continue
        count = (count_zone_w + 1) // COUNT_STEP
        if count < 1:
            continue

        expected = np.zeros_like(sub, dtype=bool)
        for i in range(count):
            c_start = i * COUNT_STEP
            expected[1:3, c_start : c_start + COUNT_BLOCK_W] = True
        expected[:, count_zone_w + LEGEND_SEP :] = STYLE_MASKS[style]
        if not np.array_equal(sub, expected):
            continue

        mapping[count] = style

    return mapping


def _find_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        sub = g[r0 : r1 + 1, c0 : c1 + 1] == GREY
        if np.array_equal(sub, expected):
            rectangles.append((r0, c0, r1, c1))

    rectangles.sort()
    return rectangles


def _draw_solid(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _draw_dashed(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    out[r0, c0 : c1 + 1] = BG
    out[r1, c0 : c1 + 1] = BG
    out[r0 : r1 + 1, c0] = BG
    out[r0 : r1 + 1, c1] = BG

    for c in range(c0, c1 + 1, 2):
        out[r0, c] = color
        out[r1, c] = color
    for r in range(r0, r1 + 1, 2):
        out[r, c0] = color
        out[r, c1] = color


def _draw_double(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    _draw_solid(out, r0, c0, r1, c1, color)
    _draw_solid(out, r0 - 1, c0 - 1, r1 + 1, c1 + 1, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each grey rectangular boundary using the style selected by its seed-count legend entry.

    Input structure:
    - The background is black.
    - Several non-overlapping grey one-pixel-thick rectangular borders enclose empty black interiors with scattered colored single-pixel seeds.
    - A corner legend contains horizontal rows whose left side shows a count with repeated colored blocks and whose right side shows a border-style sample.

    Rule:
    - For every grey rectangle, count how many seed pixels lie strictly inside it.
    - Find the legend row whose repeated block count matches that seed total and read its style: solid, dashed, or double.
    - Repaint the rectangle border with the most frequent seed color inside that rectangle using the matched style, while leaving the seeds unchanged.
    - If a rectangle has no seeds inside, keep its grey border unchanged.

    Key insight:
    - The legend maps seed quantity to border style, while a separate majority-color vote inside each rectangle chooses the repaint color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend = _find_legend_mapping(g)
    for r0, c0, r1, c1 in _find_rectangles(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_points = [(int(r), int(c)) for r, c in np.argwhere((interior != BG) & (interior != GREY))]
        if not seed_points:
            continue

        seed_count = len(seed_points)
        style = legend.get(seed_count)
        if style is None:
            continue

        seed_colors = [int(interior[r, c]) for r, c in seed_points]
        color = _dominant_color(seed_colors)
        if style == STYLE_SOLID:
            _draw_solid(out, r0, c0, r1, c1, color)
        elif style == STYLE_DASHED:
            _draw_dashed(out, r0, c0, r1, c1, color)
        else:
            _draw_double(out, r0, c0, r1, c1, color)

    return out


def _make_legend_row(count: int, style: str, color: int) -> np.ndarray:
    count_zone_w = COUNT_STEP * count - 1
    width = count_zone_w + LEGEND_SEP + STYLE_W
    row = np.full((ROW_H, width), BG, dtype=int)
    for i in range(count):
        c0 = i * COUNT_STEP
        row[1:3, c0 : c0 + COUNT_BLOCK_W] = color
    row[:, count_zone_w + LEGEND_SEP :] = np.where(STYLE_MASKS[style], color, BG)
    return row


def _legend_patch(entries: list[tuple[int, str, int]], corner: str) -> np.ndarray:
    rows = [_make_legend_row(count, style, color) for count, style, color in entries]
    patch_h = len(rows) * ROW_H + (len(rows) - 1) * ROW_GAP
    patch_w = max(row.shape[1] for row in rows)
    patch = np.full((patch_h, patch_w), BG, dtype=int)
    for idx, row in enumerate(rows):
        r0 = idx * (ROW_H + ROW_GAP)
        if corner in {"tl", "bl"}:
            c0 = 0
        else:
            c0 = patch_w - row.shape[1]
        row_mask = row != BG
        patch[r0 : r0 + ROW_H, c0 : c0 + row.shape[1]][row_mask] = row[row_mask]
    return patch


def _patch_origin(grid_h: int, grid_w: int, patch_h: int, patch_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - patch_w
    if corner == "bl":
        return grid_h - patch_h, 0
    if corner == "br":
        return grid_h - patch_h, grid_w - patch_w
    raise ValueError(corner)


def _expand_bbox(
    bbox: tuple[int, int, int, int], amount: int, grid_h: int, grid_w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - amount), max(0, c0 - amount), min(grid_h - 1, r1 + amount), min(grid_w - 1, c1 + amount)


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _draw_grey_rectangle(g: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY


def _pick_branches(region_count: int, legend_counts: list[int]) -> list[int]:
    options = [0] + legend_counts
    for _ in range(100):
        branches = [random.choice(options) for _ in range(region_count)]
        if any(branch > 0 for branch in branches) and len(set(branches)) >= 2:
            return branches
    return [legend_counts[0]] + [0] * (region_count - 1)


def _seed_colors_for_count(count: int, palette: list[int]) -> list[int]:
    dominant = random.choice(palette)
    if count == 1:
        return [dominant]

    dominant_count = random.randint(count // 2 + 1, count)
    colors = [dominant] * dominant_count
    other_palette = [color for color in palette if color != dominant]
    for _ in range(count - dominant_count):
        colors.append(random.choice(other_palette))
    random.shuffle(colors)
    return colors


def _place_seeds(
    g: np.ndarray, bbox: tuple[int, int, int, int], count: int, palette: list[int]
) -> None:
    if count == 0:
        return
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen = random.sample(cells, count)
    colors = _seed_colors_for_count(count, palette)
    for (r, c), color in zip(chosen, colors):
        g[r, c] = color


def _generate_attempt() -> np.ndarray | None:
    legend_counts = random.sample([1, 2, 3, 4], 3)
    styles = list(STYLES)
    random.shuffle(styles)
    legend_colors = random.sample(NON_BG_NON_GREY, 3)
    legend_entries = list(zip(legend_counts, styles, legend_colors))
    random.shuffle(legend_entries)

    corner = random.choice(["tl", "tr", "bl", "br"])
    legend = _legend_patch(legend_entries, corner)
    ph, pw = legend.shape

    min_h = max(18, ph + 2)
    min_w = max(18, pw + 4)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    g = np.full((grid_h, grid_w), BG, dtype=int)

    pr, pc = _patch_origin(grid_h, grid_w, ph, pw, corner)
    legend_mask = legend != BG
    g[pr : pr + ph, pc : pc + pw][legend_mask] = legend[legend_mask]

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_bbox(occupied, _expand_bbox((pr, pc, pr + ph - 1, pc + pw - 1), 1, grid_h, grid_w))

    seed_palette = [color for color in NON_BG_NON_GREY if color not in legend_colors]
    if len(seed_palette) < 2:
        return None

    region_count = random.randint(2, 4)
    branches = _pick_branches(region_count, legend_counts)
    rectangles: list[tuple[int, int, int, int]] = []
    for seed_count in branches:
        placed = False
        for _ in range(200):
            h = random.randint(5, min(9, grid_h - 2))
            w = random.randint(5, min(10, grid_w - 2))
            if (h - 2) * (w - 2) < max(seed_count, 1):
                continue
            if grid_h - h - 1 < 1 or grid_w - w - 1 < 1:
                continue

            r0 = random.randint(1, grid_h - h - 1)
            c0 = random.randint(1, grid_w - w - 1)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            safe_bbox = (r0 - 1, c0 - 1, r0 + h, c0 + w)
            sr0, sc0, sr1, sc1 = safe_bbox
            if occupied[sr0 : sr1 + 1, sc0 : sc1 + 1].any():
                continue

            rectangles.append(bbox)
            _mark_bbox(occupied, safe_bbox)
            placed = True
            break
        if not placed:
            return None

    for bbox, seed_count in zip(rectangles, branches):
        _draw_grey_rectangle(g, bbox)
        _place_seeds(g, bbox, seed_count, seed_palette)

    out = main(g)
    if out.shape != g.shape:
        return None
    if not np.any(out != g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

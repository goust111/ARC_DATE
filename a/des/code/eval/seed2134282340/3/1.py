from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 500
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
FRAME_COUNT_RANGE = (1, 3)
FRAME_H_RANGE = (7, 12)
FRAME_W_RANGE = (7, 12)
LEGEND_MAX_RANGE = (4, 7)
PLACEMENT_PADDING = 2
PLACEMENT_TRIES = 120


@dataclass
class _FrameSpec:
    top: int
    left: int
    height: int
    width: int
    color: int
    gap_side: str
    gap_index: int
    pattern_mask: np.ndarray
    fill_color: int


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    for color in colors:
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            comp_set = set(comp)
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp_set):
                continue
            border = _perimeter_cells(r0, c0, r1, c1)
            if len(comp_set) != len(border) - 1:
                continue
            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            mr, mc = missing[0]
            if (mr in (r0, r1)) and (mc in (c0, c1)):
                continue
            if mr == r0:
                side = "top"
                gap_index = mc - c0
            elif mr == r1:
                side = "bottom"
                gap_index = mc - c0
            elif mc == c0:
                side = "left"
                gap_index = mr - r0
            else:
                side = "right"
                gap_index = mr - r0
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gap_side": side,
                    "gap_index": gap_index,
                }
            )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def _legend_mapping(grid: np.ndarray, bg: int, frames: list[dict]) -> dict[int, int]:
    residual = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        residual[r0 : r1 + 1, c0 : c1 + 1] = bg

    mapping: dict[int, int] = {}
    for color in [int(c) for c in np.unique(residual) if int(c) != bg]:
        for comp in _components(residual == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r0 != r1:
                continue
            length = c1 - c0 + 1
            if len(comp) != length:
                continue
            mapping[length] = color
    return mapping


def _project_mask(mask: np.ndarray, gap_side: str) -> np.ndarray:
    h, w = mask.shape
    proj = np.zeros((h, w), dtype=bool)
    if gap_side in ("top", "bottom"):
        counts = mask.sum(axis=0)
        for c, count in enumerate(counts):
            count = int(count)
            if count == 0:
                continue
            if gap_side == "top":
                proj[h - count :, c] = True
            else:
                proj[:count, c] = True
    else:
        counts = mask.sum(axis=1)
        for r, count in enumerate(counts):
            count = int(count)
            if count == 0:
                continue
            if gap_side == "left":
                proj[r, w - count :] = True
            else:
                proj[r, :count] = True
    return proj


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _detect_frames(g, bg)
    assert frames, "expected at least one gapped rectangular frame"
    legend = _legend_mapping(g, bg, frames)
    assert legend, "expected legend bars outside frames"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_color = int(frame["color"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern = (interior != bg) & (interior != frame_color)
        pixels = int(pattern.sum())
        assert pixels > 0, "each frame must contain a non-empty inner pattern"
        assert pixels in legend, "missing legend entry for pattern area"

        projected = _project_mask(pattern, str(frame["gap_side"]))
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        out[r0 + 1 : r1, c0 + 1 : c1][projected] = int(legend[pixels])

    return out


def _random_connected_mask(height: int, width: int, area: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier: set[tuple[int, int]] = set(_neighbors4(start[0], start[1], height, width))
    while len(cells) < area:
        choices = [cell for cell in frontier if cell not in cells]
        if not choices:
            return None
        cell = rng.choice(choices)
        frontier.discard(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if nb not in cells:
                frontier.add(nb)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_pattern_mask(frame_h: int, frame_w: int, max_area: int, rng: random.Random) -> tuple[np.ndarray, int] | None:
    interior_h = frame_h - 2
    interior_w = frame_w - 2
    usable_h = interior_h - 2
    usable_w = interior_w - 2
    if usable_h <= 0 or usable_w <= 0:
        return None

    area = rng.randint(1, min(max_area, usable_h * usable_w))
    dim_options = [
        (h, w)
        for h in range(1, usable_h + 1)
        for w in range(1, usable_w + 1)
        if h * w >= area
    ]
    rng.shuffle(dim_options)
    for shape_h, shape_w in dim_options:
        patch = _random_connected_mask(shape_h, shape_w, area, rng)
        if patch is None:
            continue
        top = 1 + rng.randint(0, usable_h - shape_h)
        left = 1 + rng.randint(0, usable_w - shape_w)
        full = np.zeros((interior_h, interior_w), dtype=bool)
        full[top : top + shape_h, left : left + shape_w] = patch
        return full, area
    return None


def _frame_patch(height: int, width: int, color: int, gap_side: str, gap_index: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if gap_side == "top":
        patch[0, gap_index] = BG
    elif gap_side == "bottom":
        patch[-1, gap_index] = BG
    elif gap_side == "left":
        patch[gap_index, 0] = BG
    else:
        patch[gap_index, -1] = BG
    return patch


def _draw_legend(grid: np.ndarray, top: int, left: int, legend_colors: list[int], right_aligned: bool) -> None:
    count = len(legend_colors)
    for idx, color in enumerate(legend_colors, start=1):
        row = top + idx - 1
        col = left + (count - idx if right_aligned else 0)
        grid[row, col : col + idx] = int(color)


def _legend_location(height: int, width: int, count: int, corner: str, rng: random.Random) -> tuple[int, int, bool]:
    row_margin = 1 if height - count <= 3 else rng.randint(1, 2)
    col_margin = 1 if width - count <= 3 else rng.randint(1, 2)
    top = row_margin if corner in ("tl", "tr") else height - count - row_margin
    left = col_margin if corner in ("tl", "bl") else width - count - col_margin
    return top, left, corner in ("tr", "br")


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not occupied[r0:r1, c0:c1].any()


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    occupied[r0:r1, c0:c1] = True


def _render_expected(inp: np.ndarray, frames: list[_FrameSpec]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        r0 = frame.top
        c0 = frame.left
        r1 = r0 + frame.height
        c1 = c0 + frame.width
        projected = _project_mask(frame.pattern_mask, frame.gap_side)
        out[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = BG
        out[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1][projected] = int(frame.fill_color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    legend_size = rng.randint(*LEGEND_MAX_RANGE)

    if grid_h <= legend_size + 4 or grid_w <= legend_size + 4:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    frame_color = colors[0]
    pattern_color = colors[1]
    legend_colors = colors[2 : 2 + legend_size]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    legend_corner = rng.choice(["tl", "tr", "bl", "br"])
    legend_top, legend_left, right_aligned = _legend_location(
        grid_h, grid_w, legend_size, legend_corner, rng
    )
    _draw_legend(grid, legend_top, legend_left, legend_colors, right_aligned)
    _mark_occupied(occupied, legend_top, legend_left, legend_size, legend_size, PLACEMENT_PADDING)

    specs: list[_FrameSpec] = []
    for _ in range(frame_count):
        frame_h = rng.randint(*FRAME_H_RANGE)
        frame_w = rng.randint(*FRAME_W_RANGE)
        made = _make_pattern_mask(frame_h, frame_w, legend_size, rng)
        if made is None:
            return None
        pattern_mask, area = made
        gap_side = rng.choice(["top", "bottom", "left", "right"])
        gap_index = rng.randint(1, frame_w - 2) if gap_side in ("top", "bottom") else rng.randint(1, frame_h - 2)

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = rng.randint(0, grid_h - frame_h)
            left = rng.randint(0, grid_w - frame_w)
            if not _can_place(occupied, top, left, frame_h, frame_w, PLACEMENT_PADDING):
                continue
            patch = _frame_patch(frame_h, frame_w, frame_color, gap_side, gap_index)
            grid[top : top + frame_h, left : left + frame_w] = patch
            interior = grid[top + 1 : top + frame_h - 1, left + 1 : left + frame_w - 1]
            interior[pattern_mask] = pattern_color
            _mark_occupied(occupied, top, left, frame_h, frame_w, PLACEMENT_PADDING)
            specs.append(
                _FrameSpec(
                    top=top,
                    left=left,
                    height=frame_h,
                    width=frame_w,
                    color=frame_color,
                    gap_side=gap_side,
                    gap_index=gap_index,
                    pattern_mask=pattern_mask,
                    fill_color=legend_colors[area - 1],
                )
            )
            placed = True
            break
        if not placed:
            return None

    expected = _render_expected(grid, specs)
    if np.array_equal(expected, grid):
        return None

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(cnts))]) != BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

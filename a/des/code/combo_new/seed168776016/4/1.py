from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"
VERTICAL_AXIS = "vertical"
HORIZONTAL_AXIS = "horizontal"

GRID_MIN = 14
GRID_MAX = 30
FRAME_COUNT_CHOICES = (1, 2, 3)
FRAME_COUNT_WEIGHTS = (3, 3, 2)
FRAME_OUTER_H_RANGE = (5, 10)
FRAME_OUTER_W_RANGE = (5, 10)
FRAME_PADDING = 1
RAY_MARGIN = 2
GEN_RETRIES = 400
PLACEMENT_RETRIES = 160
SEED_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2))


@dataclass(frozen=True)
class FrameSpec:
    r0: int
    c0: int
    r1: int
    c1: int
    wall_color: int
    gap_r: int
    gap_c: int
    side: str
    seed_color: int
    seed_r0: int
    seed_c0: int
    seed_r1: int
    seed_c1: int


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r1, c) for c in range(c0, c1 + 1))
    cells.extend((r, c0) for r in range(r0 + 1, r1))
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    return ordered


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _extract_seed_info(
    grid: np.ndarray, bg: int, wall_color: int, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, tuple[int, int, int, int]] | None:
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = {int(v) for v in np.unique(interior) if int(v) not in (bg, wall_color)}
    if len(colors) != 1:
        return None
    seed_color = next(iter(colors))
    seed_mask = interior == seed_color
    if seed_mask[0, :].any() or seed_mask[-1, :].any() or seed_mask[:, 0].any() or seed_mask[:, -1].any():
        return None
    comps = _components_mask(seed_mask)
    if len(comps) != 1:
        return None
    extra = interior[(interior != bg) & (interior != wall_color) & (interior != seed_color)]
    if extra.size != 0:
        return None
    sr0, sc0, sr1, sc1 = _bbox(comps[0])
    return seed_color, (r0 + 1 + sr0, c0 + 1 + sc0, r0 + 1 + sr1, c0 + 1 + sc1)


def _frame_from_component(grid: np.ndarray, bg: int, color: int, coords: list[tuple[int, int]]) -> FrameSpec | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    perimeter = _perimeter_cells(r0, c0, r1, c1)
    coord_set = set(coords)
    if len(coord_set) != len(perimeter) - 1:
        return None
    if not coord_set.issubset(set(perimeter)):
        return None

    gaps = [cell for cell in perimeter if cell not in coord_set]
    if len(gaps) != 1:
        return None
    gap_r, gap_c = gaps[0]
    if grid[gap_r, gap_c] != bg:
        return None
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None
    if int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == color)) != len(coord_set):
        return None

    for pr, pc in perimeter:
        if (pr, pc) == (gap_r, gap_c):
            continue
        if grid[pr, pc] != color:
            return None

    seed_info = _extract_seed_info(grid, bg, color, r0, c0, r1, c1)
    if seed_info is None:
        return None
    seed_color, (sr0, sc0, sr1, sc1) = seed_info

    if gap_r == r0:
        side = TOP
    elif gap_r == r1:
        side = BOTTOM
    elif gap_c == c0:
        side = LEFT
    else:
        side = RIGHT

    return FrameSpec(
        r0=r0,
        c0=c0,
        r1=r1,
        c1=c1,
        wall_color=color,
        gap_r=gap_r,
        gap_c=gap_c,
        side=side,
        seed_color=seed_color,
        seed_r0=sr0,
        seed_c0=sc0,
        seed_r1=sr1,
        seed_c1=sc1,
    )


def _find_frames(grid: np.ndarray, bg: int) -> list[FrameSpec]:
    frames: list[FrameSpec] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components_mask(grid == color):
            frame = _frame_from_component(grid, bg, color, comp)
            if frame is not None:
                frames.append(frame)
    return frames


def _paint_frame_ray(out: np.ndarray, frame: FrameSpec) -> None:
    if frame.side == TOP:
        out[: frame.r1, frame.gap_c] = frame.seed_color
    elif frame.side == BOTTOM:
        out[frame.r0 + 1 :, frame.gap_c] = frame.seed_color
    elif frame.side == LEFT:
        out[frame.gap_r, : frame.c1] = frame.seed_color
    else:
        out[frame.gap_r, frame.c0 + 1 :] = frame.seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast a color ray from the grid edge through each frame gap and into the frame interior.

    Input structure:
    - The background is the dominant color.
    - One or more one-pixel-thick hollow rectangular frames sit on the background, and each frame has exactly one single-cell gap on one side.
    - Each frame encloses exactly one small monochrome seed block whose color differs from the frame and from the background.

    Rule:
    - For each valid frame, read the enclosed seed color and locate the frame's unique gap side.
    - Draw a one-cell-wide ray in the seed color from the corresponding grid edge toward the frame, passing through the gap.
    - Continue the same straight ray inside the frame and fill the interior line from the gap to the opposite inner wall.
    - Keep the frame pixels and the original seed block unchanged.

    Key insight:
    - The seed block encodes the ray color, while the gap position independently encodes which edge launches the ray and which interior line must be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one gapped rectangular frame"

    out = g.copy()
    for frame in frames:
        _paint_frame_ray(out, frame)
        out[frame.r0 : frame.r1 + 1, frame.c0] = np.where(
            g[frame.r0 : frame.r1 + 1, frame.c0] == frame.wall_color,
            g[frame.r0 : frame.r1 + 1, frame.c0],
            out[frame.r0 : frame.r1 + 1, frame.c0],
        )
        out[frame.r0 : frame.r1 + 1, frame.c1] = np.where(
            g[frame.r0 : frame.r1 + 1, frame.c1] == frame.wall_color,
            g[frame.r0 : frame.r1 + 1, frame.c1],
            out[frame.r0 : frame.r1 + 1, frame.c1],
        )
        out[frame.r0, frame.c0 : frame.c1 + 1] = np.where(
            g[frame.r0, frame.c0 : frame.c1 + 1] == frame.wall_color,
            g[frame.r0, frame.c0 : frame.c1 + 1],
            out[frame.r0, frame.c0 : frame.c1 + 1],
        )
        out[frame.r1, frame.c0 : frame.c1 + 1] = np.where(
            g[frame.r1, frame.c0 : frame.c1 + 1] == frame.wall_color,
            g[frame.r1, frame.c0 : frame.c1 + 1],
            out[frame.r1, frame.c0 : frame.c1 + 1],
        )
        out[frame.seed_r0 : frame.seed_r1 + 1, frame.seed_c0 : frame.seed_c1 + 1] = g[
            frame.seed_r0 : frame.seed_r1 + 1, frame.seed_c0 : frame.seed_c1 + 1
        ]
    return out


def _reserve_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = FRAME_PADDING) -> None:
    h, w = map(int, mask.shape)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _center_band(start: int, stop: int) -> tuple[int, int]:
    if stop <= start:
        return start, stop
    width = stop - start + 1
    shrink = width // 3
    band_start = start + shrink
    band_stop = stop - shrink
    if band_start <= band_stop:
        return band_start, band_stop
    return start, stop


def _sample_seed_position(
    rng: random.Random, axis: str, frame: tuple[int, int, int, int], gap_line: int, seed_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    seed_h, seed_w = seed_shape
    row_lo = r0 + 2
    row_hi = r1 - seed_h - 1
    col_lo = c0 + 2
    col_hi = c1 - seed_w - 1
    row_lo, row_hi = _center_band(row_lo, row_hi)
    col_lo, col_hi = _center_band(col_lo, col_hi)

    if axis == VERTICAL_AXIS and rng.random() < 0.5:
        aligned_lo = max(col_lo, gap_line - seed_w + 1)
        aligned_hi = min(col_hi, gap_line)
        if aligned_lo <= aligned_hi:
            seed_c0 = rng.randint(aligned_lo, aligned_hi)
            seed_r0 = rng.randint(row_lo, row_hi)
            return seed_r0, seed_c0
    if axis == HORIZONTAL_AXIS and rng.random() < 0.5:
        aligned_lo = max(row_lo, gap_line - seed_h + 1)
        aligned_hi = min(row_hi, gap_line)
        if aligned_lo <= aligned_hi:
            seed_r0 = rng.randint(aligned_lo, aligned_hi)
            seed_c0 = rng.randint(col_lo, col_hi)
            return seed_r0, seed_c0

    return rng.randint(row_lo, row_hi), rng.randint(col_lo, col_hi)


def _choose_frame_count(rng: random.Random, height: int, width: int, axis: str) -> int:
    cap_by_space = 3 if (width >= 20 if axis == VERTICAL_AXIS else height >= 20) else 2
    choices = [n for n in FRAME_COUNT_CHOICES if n <= cap_by_space]
    weights = [FRAME_COUNT_WEIGHTS[FRAME_COUNT_CHOICES.index(n)] for n in choices]
    return rng.choices(choices, weights=weights, k=1)[0]


def _sample_bbox(
    rng: random.Random, height: int, width: int, axis: str, side: str
) -> tuple[int, int, int, int] | None:
    outer_h = rng.randint(*FRAME_OUTER_H_RANGE)
    outer_w = rng.randint(*FRAME_OUTER_W_RANGE)
    outer_h = min(outer_h, height - 2)
    outer_w = min(outer_w, width - 2)
    if outer_h < 5 or outer_w < 5:
        return None

    if axis == VERTICAL_AXIS:
        if side == TOP:
            r_min, r_max = RAY_MARGIN, height - outer_h - 1
        else:
            r_min, r_max = 1, height - outer_h - 1 - RAY_MARGIN
        c_min, c_max = 1, width - outer_w - 2
    else:
        r_min, r_max = 1, height - outer_h - 2
        if side == LEFT:
            c_min, c_max = RAY_MARGIN, width - outer_w - 1
        else:
            c_min, c_max = 1, width - outer_w - 1 - RAY_MARGIN

    if r_min > r_max or c_min > c_max:
        return None
    r0 = rng.randint(r_min, r_max)
    c0 = rng.randint(c_min, c_max)
    return r0, c0, r0 + outer_h - 1, c0 + outer_w - 1


def _outside_corridor(
    shape: tuple[int, int], frame: tuple[int, int, int, int], side: str, gap_r: int, gap_c: int
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = frame
    if side == TOP:
        return [(r, gap_c) for r in range(0, r0)]
    if side == BOTTOM:
        return [(r, gap_c) for r in range(r1 + 1, h)]
    if side == LEFT:
        return [(gap_r, c) for c in range(0, c0)]
    return [(gap_r, c) for c in range(c1 + 1, w)]


def _inside_line(frame: tuple[int, int, int, int], side: str, gap_r: int, gap_c: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    if side == TOP:
        return [(r, gap_c) for r in range(r0, r1)]
    if side == BOTTOM:
        return [(r, gap_c) for r in range(r0 + 1, r1 + 1)]
    if side == LEFT:
        return [(gap_r, c) for c in range(c0, c1)]
    return [(gap_r, c) for c in range(c0 + 1, c1 + 1)]


def _lane_free(cells: list[tuple[int, int]], occupied: np.ndarray, rays: np.ndarray) -> bool:
    return all(not occupied[r, c] and not rays[r, c] for r, c in cells)


def _draw_frame(grid: np.ndarray, frame: FrameSpec) -> None:
    grid[frame.r0 : frame.r1 + 1, frame.c0] = frame.wall_color
    grid[frame.r0 : frame.r1 + 1, frame.c1] = frame.wall_color
    grid[frame.r0, frame.c0 : frame.c1 + 1] = frame.wall_color
    grid[frame.r1, frame.c0 : frame.c1 + 1] = frame.wall_color
    grid[frame.gap_r, frame.gap_c] = grid[0, 0]
    grid[frame.seed_r0 : frame.seed_r1 + 1, frame.seed_c0 : frame.seed_c1 + 1] = frame.seed_color


def _seed_shapes_for_bbox(frame: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    return [shape for shape in SEED_SHAPES if shape[0] <= outer_h - 4 and shape[1] <= outer_w - 4]


def _render_expected(inp: np.ndarray, frames: list[FrameSpec]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        _paint_frame_ray(out, frame)
        out[frame.r0 : frame.r1 + 1, frame.c0] = inp[frame.r0 : frame.r1 + 1, frame.c0]
        out[frame.r0 : frame.r1 + 1, frame.c1] = inp[frame.r0 : frame.r1 + 1, frame.c1]
        out[frame.r0, frame.c0 : frame.c1 + 1] = inp[frame.r0, frame.c0 : frame.c1 + 1]
        out[frame.r1, frame.c0 : frame.c1 + 1] = inp[frame.r1, frame.c0 : frame.c1 + 1]
        out[frame.seed_r0 : frame.seed_r1 + 1, frame.seed_c0 : frame.seed_c1 + 1] = inp[
            frame.seed_r0 : frame.seed_r1 + 1, frame.seed_c0 : frame.seed_c1 + 1
        ]
        out[frame.gap_r, frame.gap_c] = frame.seed_color
    return out


def _build_candidate(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    axis = rng.choice((VERTICAL_AXIS, HORIZONTAL_AXIS))
    sides = (TOP, BOTTOM) if axis == VERTICAL_AXIS else (LEFT, RIGHT)
    frame_count = _choose_frame_count(rng, height, width, axis)

    colors = list(range(10))
    rng.shuffle(colors)
    bg, wall_color = colors[:2]
    seed_colors = colors[2 : 2 + frame_count]

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    reserved_rays = np.zeros((height, width), dtype=bool)
    used_lines: set[int] = set()
    frames: list[FrameSpec] = []

    for frame_idx in range(frame_count):
        side = rng.choice(sides)
        seed_color = seed_colors[frame_idx]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            bbox = _sample_bbox(rng, height, width, axis, side)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            if reserved_rays[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            if axis == VERTICAL_AXIS:
                gap_c = rng.randint(c0 + 1, c1 - 1)
                if gap_c in used_lines:
                    continue
                gap_r = r0 if side == TOP else r1
            else:
                gap_r = rng.randint(r0 + 1, r1 - 1)
                if gap_r in used_lines:
                    continue
                gap_c = c0 if side == LEFT else c1

            corridor = _outside_corridor((height, width), bbox, side, gap_r, gap_c)
            line_cells = _inside_line(bbox, side, gap_r, gap_c)
            if not _lane_free(corridor + line_cells, occupied, reserved_rays):
                continue

            seed_options = _seed_shapes_for_bbox(bbox)
            if not seed_options:
                continue
            seed_shape = rng.choice(seed_options)
            seed_r0, seed_c0 = _sample_seed_position(rng, axis, bbox, gap_c if axis == VERTICAL_AXIS else gap_r, seed_shape)
            seed_h, seed_w = seed_shape
            spec = FrameSpec(
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                wall_color=wall_color,
                gap_r=gap_r,
                gap_c=gap_c,
                side=side,
                seed_color=seed_color,
                seed_r0=seed_r0,
                seed_c0=seed_c0,
                seed_r1=seed_r0 + seed_h - 1,
                seed_c1=seed_c0 + seed_w - 1,
            )

            _draw_frame(grid, spec)
            _reserve_box(occupied, r0, c0, r1, c1)
            for rr, cc in corridor + line_cells:
                reserved_rays[rr, cc] = True
            used_lines.add(gap_c if axis == VERTICAL_AXIS else gap_r)
            frames.append(spec)
            placed = True
            break
        if not placed:
            return None

    if _dominant_color(grid) != bg:
        return None

    expected = _render_expected(grid, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate(random.Random())
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input for combo_new/seed168776016/4/1")

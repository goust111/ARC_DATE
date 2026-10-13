import random
from collections import deque

import numpy as np


BG = 0
SIDE_ORDER = ("top", "right", "bottom", "left")
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
PLACE_RETRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_full_side(line: np.ndarray, color: int) -> bool:
    return bool(np.all(line == color))


def _is_gapped_side(line: np.ndarray, color: int) -> bool:
    if line.size < 4 or int(line[0]) != color or int(line[-1]) != color:
        return False
    if np.any((line != BG) & (line != color)):
        return False
    gap_idx = np.flatnonzero(line == BG)
    if gap_idx.size < 2:
        return False
    if gap_idx[0] == 0 or gap_idx[-1] == line.size - 1:
        return False
    return int(gap_idx[-1] - gap_idx[0] + 1) == int(gap_idx.size)


def _find_legend(grid: np.ndarray) -> tuple[dict[str, int], set[tuple[int, int]]]:
    candidates = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if len(coords) != 4 or (h, w) not in {(1, 4), (4, 1)}:
            continue
        if h == 1:
            line = grid[r0, c0 : c1 + 1]
        else:
            line = grid[r0 : r1 + 1, c0]
        values = [int(v) for v in line.tolist()]
        if BG in values or len(set(values)) != 4:
            continue
        candidates.append((values, {tuple(pt) for pt in coords.tolist()}))
    assert len(candidates) == 1, "expected exactly one 4-cell legend"
    values, legend_cells = candidates[0]
    return dict(zip(SIDE_ORDER, values)), legend_cells


def _gap_side(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> str | None:
    r0, c0, r1, c1 = bbox
    top = grid[r0, c0 : c1 + 1]
    right = grid[r0 : r1 + 1, c1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 : r1 + 1, c0]

    side_checks = {
        "top": (
            _is_gapped_side(top, color)
            and _is_full_side(right, color)
            and _is_full_side(bottom, color)
            and _is_full_side(left, color)
        ),
        "right": (
            _is_full_side(top, color)
            and _is_gapped_side(right, color)
            and _is_full_side(bottom, color)
            and _is_full_side(left, color)
        ),
        "bottom": (
            _is_full_side(top, color)
            and _is_full_side(right, color)
            and _is_gapped_side(bottom, color)
            and _is_full_side(left, color)
        ),
        "left": (
            _is_full_side(top, color)
            and _is_full_side(right, color)
            and _is_full_side(bottom, color)
            and _is_gapped_side(left, color)
        ),
    }
    hits = [side for side, ok in side_checks.items() if ok]
    return hits[0] if len(hits) == 1 else None


def _find_frames(
    grid: np.ndarray, legend_cells: set[tuple[int, int]]
) -> list[dict[str, tuple[int, int, int, int] | str | int]]:
    frames = []
    for coords in _components(grid != BG):
        cell_set = {tuple(pt) for pt in coords.tolist()}
        if cell_set == legend_cells:
            continue
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        if colors.size != 1:
            continue
        color = int(colors[0])
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in cell_set):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
            continue
        side = _gap_side(grid, (r0, c0, r1, c1), color)
        if side is None:
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "gap_side": side, "border_color": color})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame with the legend color for its missing side.

    Input structure:
    - The background is black.
    - A 4-cell legend appears as either a horizontal row near the top or a vertical column near the left; its cells are all different colors.
    - The remaining non-background objects are single-color rectangular 1-pixel-thick frames whose interiors are black and whose border has exactly one continuous gap on one side.

    Rule:
    - Read the legend in order: first cell means top, second means right, third means bottom, fourth means left.
    - For each rectangular frame, detect which side contains the gap.
    - Look up the corresponding legend color for that side.
    - Fill the entire interior of the frame with that color while leaving the border and the gap itself unchanged.

    Key insight:
    - The legend encodes direction-to-color rather than object identity, so the task is to infer a side label from each frame's missing border and then apply the directional lookup.
    """
    g = np.asarray(grid, dtype=int)
    legend, legend_cells = _find_legend(g)
    frames = _find_frames(g, legend_cells)
    assert frames, "expected at least one valid frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        fill_color = int(legend[frame["gap_side"]])
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _expanded_slices(
    h: int, w: int, bbox: tuple[int, int, int, int], pad: int = 1
) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    return (
        slice(max(0, r0 - pad), min(h, r1 + pad + 1)),
        slice(max(0, c0 - pad), min(w, c1 + pad + 1)),
    )


def _draw_frame(grid: np.ndarray, spec: dict[str, int], r0: int, c0: int) -> tuple[int, int, int, int]:
    height = spec["height"]
    width = spec["width"]
    color = spec["border_color"]
    side = spec["gap_side"]
    gap_start = spec["gap_start"]
    gap_len = spec["gap_len"]

    r1 = r0 + height - 1
    c1 = c0 + width - 1

    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = BG
    elif side == "right":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c1] = BG
    elif side == "bottom":
        grid[r1, c0 + gap_start : c0 + gap_start + gap_len] = BG
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = BG

    return (r0, c0, r1, c1)


def _sample_frame_specs(rng: random.Random, n_frames: int, border_colors: list[int]) -> list[dict[str, int]]:
    gap_sides = random.sample(SIDE_ORDER, k=min(n_frames, len(SIDE_ORDER)))
    while len(gap_sides) < n_frames:
        gap_sides.append(rng.choice(SIDE_ORDER))
    rng.shuffle(gap_sides)

    specs = []
    for idx, side in enumerate(gap_sides):
        if side in ("top", "bottom"):
            height = rng.randint(3, 8)
            width = rng.randint(4, 9)
            gap_len = rng.randint(2, width - 2)
            gap_start = rng.randint(1, width - gap_len - 1)
        else:
            height = rng.randint(4, 9)
            width = rng.randint(3, 8)
            gap_len = rng.randint(2, height - 2)
            gap_start = rng.randint(1, height - gap_len - 1)
        specs.append(
            {
                "height": height,
                "width": width,
                "gap_len": gap_len,
                "gap_start": gap_start,
                "gap_side": side,
                "border_color": border_colors[idx],
            }
        )
    return specs


def _place_legend(
    grid: np.ndarray, forbidden: np.ndarray, legend_colors: list[int], orientation: str, rng: random.Random
) -> None:
    h, w = grid.shape
    if orientation == "top":
        c0 = rng.randint(0, w - 4)
        grid[0, c0 : c0 + 4] = legend_colors
        rs, cs = _expanded_slices(h, w, (0, c0, 0, c0 + 3), pad=1)
    else:
        r0 = rng.randint(0, h - 4)
        grid[r0 : r0 + 4, 0] = np.array(legend_colors, dtype=int)
        rs, cs = _expanded_slices(h, w, (r0, 0, r0 + 3, 0), pad=1)
    forbidden[rs, cs] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 5)
    orientation = rng.choice(("top", "left"))
    min_side = 10 + 2 * n_frames
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    if orientation == "top" and w < 4:
        return None
    if orientation == "left" and h < 4:
        return None

    grid = np.zeros((h, w), dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    legend_colors = rng.sample(COLORS, 4)
    border_palette = [c for c in COLORS if c not in legend_colors]
    border_colors = rng.sample(border_palette, n_frames)
    specs = _sample_frame_specs(rng, n_frames, border_colors)

    _place_legend(grid, forbidden, legend_colors, orientation, rng)

    frame_bboxes = []
    min_r0 = 2 if orientation == "top" else 0
    min_c0 = 2 if orientation == "left" else 0

    for spec in specs:
        height = spec["height"]
        width = spec["width"]
        max_r0 = h - height
        max_c0 = w - width
        if min_r0 > max_r0 or min_c0 > max_c0:
            return None

        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(min_r0, max_r0)
            c0 = rng.randint(min_c0, max_c0)
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            rs, cs = _expanded_slices(h, w, bbox, pad=1)
            if forbidden[rs, cs].any():
                continue
            _draw_frame(grid, spec, r0, c0)
            forbidden[rs, cs] = True
            frame_bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    try:
        legend, legend_cells = _find_legend(grid)
        frames = _find_frames(grid, legend_cells)
        if [legend[side] for side in SIDE_ORDER] != legend_colors:
            return None
        if len(frames) != n_frames:
            return None
        if sorted(frame["gap_side"] for frame in frames) != sorted(spec["gap_side"] for spec in specs):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

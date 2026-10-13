from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, paste, random_free_location


FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL)
WEIGHT_VALUES = (3, 4, 5, 6, 7, 8)
GEN_RETRY_LIMIT = 300


def _is_rectangular_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    coords = comp["coords"]
    on_border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return int(comp["area"]) == 2 * h + 2 * w - 4 and bool(np.all(on_border))


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    items = []
    for comp in find_components(
        np.asarray(grid, dtype=int),
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
    ):
        color = int(comp["color"])
        if color in {Color.BLACK, Color.GREY, Color.MAROON}:
            continue
        if not _is_rectangular_frame(comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        weight = int(np.sum(grid[r0 + 1 : r1, c0 + 1 : c1] == Color.GREY))
        items.append(
            {
                "color": color,
                "bbox": comp["bbox"],
                "height": h,
                "width": w,
                "area": h * w,
                "weight": weight,
            }
        )

    assert 3 <= len(items) <= 4, "expected three or four rectangular frames"
    weights = [item["weight"] for item in items]
    assert len(weights) == len(set(weights)), "grey-count weights must be unique"
    items.sort(key=lambda item: (-int(item["weight"]), -int(item["area"])))
    return items


def _draw_frame(out: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> tuple[int, int, int, int]:
    out[r, c : c + w] = color
    out[r + h - 1, c : c + w] = color
    out[r : r + h, c] = color
    out[r : r + h, c + w - 1] = color
    return r + 1, c + 1, h - 2, w - 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild colored rectangular frames as a magenta lower-left nesting ordered by interior gray-pixel counts.

    Input structure:
    - The grid has a black background with three or four non-overlapping monochrome rectangular perimeters.
    - Each rectangle encloses some gray pixels, and maroon noise pixels may appear outside all rectangles.
    - Rectangle colors are distinct and only the frame pixels, not the gray interior pixels, define each rectangle.

    Rule:
    - Ignore maroon noise and detect every non-black, non-gray rectangular frame.
    - Count the gray pixels strictly inside each frame and sort frames by that count from largest to smallest.
    - Crop the output to the bbox size of the highest-count frame, then redraw only the frames, nesting each next frame inside the previous frame's interior with lower-left alignment.
    - Recolor every redrawn frame pixel to magenta and discard all original interior gray pixels.

    Key insight:
    - The gray pixels are only a hidden ordering code: their counts determine the nesting order, while the output keeps only uniformly recolored rectangular borders.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    items = _extract_rectangles(g)
    outer = items[0]
    out = np.zeros((int(outer["height"]), int(outer["width"])), dtype=int)

    region = _draw_frame(out, 0, 0, int(outer["height"]), int(outer["width"]), Color.PINK)
    for item in items[1:]:
        rr, cc, rh, rw = region
        h = int(item["height"])
        w = int(item["width"])
        assert h <= rh and w <= rw, "sorted frames must fit inside the previous interior"
        top = rr + rh - h
        left = cc
        region = _draw_frame(out, top, left, h, w, Color.PINK)

    return out


def _grow_blob(height: int, width: int, count: int, rng: np.random.Generator) -> np.ndarray:
    assert 0 < count <= height * width
    cells = [(int(rng.integers(height)), int(rng.integers(width)))]
    used = {cells[0]}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in used:
                frontier.add((nr, nc))

    add_frontier(*cells[0])
    while len(used) < count:
        if not frontier:
            seed = cells[int(rng.integers(len(cells)))]
            add_frontier(*seed)
        choices = list(frontier)
        nr, nc = choices[int(rng.integers(len(choices)))]
        frontier.remove((nr, nc))
        used.add((nr, nc))
        cells.append((nr, nc))
        add_frontier(nr, nc)

    blob = np.zeros((height, width), dtype=bool)
    for r, c in used:
        blob[r, c] = True
    return blob


def _make_frame_patch(height: int, width: int, color: int, weight: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    interior_h = height - 2
    interior_w = width - 2
    blob = _grow_blob(interior_h, interior_w, weight, rng)
    patch[1:-1, 1:-1][blob] = Color.GREY
    return patch


def _sample_chain(count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    while True:
        inner_h = int(rng.integers(5, 7))
        inner_w = int(rng.integers(5, 7))
        sizes = [(inner_h, inner_w)]
        for _ in range(count - 1):
            prev_h, prev_w = sizes[-1]
            sizes.append((prev_h + int(rng.integers(2, 4)), prev_w + int(rng.integers(2, 4))))
        sizes = list(reversed(sizes))
        if max(h for h, _ in sizes) <= 14 and max(w for _, w in sizes) <= 14:
            return sizes


def _clear_noise_candidates(occupied: np.ndarray) -> list[tuple[int, int]]:
    H, W = occupied.shape
    cells = []
    for r in range(H):
        for c in range(W):
            if occupied[r, c]:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rect_count = 3 if rng.random() < 0.55 else 4
    sizes = _sample_chain(rect_count, rng)
    colors = list(rng.choice(FRAME_COLORS, size=rect_count, replace=False))
    weights = sorted(rng.choice(WEIGHT_VALUES, size=rect_count, replace=False), reverse=True)

    patches = [
        _make_frame_patch(h, w, int(color), int(weight), rng)
        for (h, w), color, weight in zip(sizes, colors, weights)
    ]

    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    H = int(rng.integers(max(16, max_h + 4), 27))
    W = int(rng.integers(max(16, max_w + 4), 27))

    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=int)
    order = list(range(rect_count))
    rng.shuffle(order)
    for idx in order:
        patch = patches[idx]
        reserve = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(occupied, reserve, transparent=0, border=1, padding=1)
        except ValueError:
            return None
        paste(occupied, reserve, r0, c0)
        paste(grid, patch, r0, c0)

    noise_cells = _clear_noise_candidates(occupied.astype(bool))
    rng.shuffle(noise_cells)
    for r, c in noise_cells[: int(rng.integers(0, min(5, len(noise_cells) + 1)))]:
        grid[r, c] = Color.MAROON

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_extract_rectangles(grid)) != rect_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

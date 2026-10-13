from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SEED_COLOR = 7
FRAME_COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
MIN_FRAMES = 2
MAX_FRAMES = 4
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return Counter(corners).most_common(1)[0][0]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(pts)
    return components


def _monochrome_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append((color, pts))
    return components


def _frame_from_component(color: int, pts: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(pts) < 8:
        return None
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    expected = 2 * (height + width) - 4
    if len(pts) != expected:
        return None
    pt_set = set(pts)
    if len(pt_set) != expected:
        return None
    border_set = set()
    for c in range(c0, c1 + 1):
        border_set.add((r0, c))
        border_set.add((r1, c))
    for r in range(r0, r1 + 1):
        border_set.add((r, c0))
        border_set.add((r, c1))
    if pt_set != border_set:
        return None
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "coords": pts,
        "area": height * width,
    }


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color, pts in _monochrome_components(grid, bg):
        frame = _frame_from_component(color, pts)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda frame: (-int(frame["area"]), frame["bbox"][0], frame["bbox"][1]))
    return frames


def _frame_mask(shape: tuple[int, int], frames: list[dict[str, object]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:
            mask[r, c] = True
    return mask


def _enclosed_regions(frame_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = frame_mask.shape
    regions: list[list[tuple[int, int]]] = []
    for pts in _components_mask(~frame_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        regions.append(pts)
    return regions


def _seed_count_in_region(grid: np.ndarray, bg: int, frame_mask: np.ndarray, pts: list[tuple[int, int]]) -> int:
    return sum(1 for r, c in pts if int(grid[r, c]) != bg and not bool(frame_mask[r, c]))


def _analyze_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]], np.ndarray, list[list[tuple[int, int]]]]:
    bg = _background_color(grid)
    frames = _detect_frames(grid, bg)
    if not frames:
        raise ValueError("expected at least one rectangular frame")
    frame_mask = _frame_mask(grid.shape, frames)
    regions = _enclosed_regions(frame_mask)
    return bg, frames, frame_mask, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded concentric-frame region with the color of the layer indexed by its seed count.

    Input structure:
    - A black background contains one set of concentric 1-pixel-thick rectangular frames.
    - Each frame has a distinct color, and the frames share the same center.
    - The gaps between neighboring frames and the innermost interior form enclosed regions.
      Some of those regions contain a few same-colored seed pixels.

    Rule:
    - Detect the nested frames and order them from outermost to innermost.
    - Split the non-frame cells into enclosed regions. For each region, count how many
      seed pixels it contains.
    - If a region contains K seeds, repaint the whole region with the color of the K-th
      frame from the outside in, replacing both the seeds and the black cells. Unseeded
      regions stay black, and the frame pixels stay unchanged.

    Key insight:
    - The seed count is not the output color directly; it is an index into the visible
      outer-to-inner frame order, so geometry supplies the count-to-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, frame_mask, regions = _analyze_scene(g)

    out = g.copy()
    for pts in regions:
        seed_count = _seed_count_in_region(g, bg, frame_mask, pts)
        if seed_count == 0:
            continue
        if not 1 <= seed_count <= len(frames):
            raise ValueError("seed count must map to an existing frame layer")
        fill_color = int(frames[seed_count - 1]["color"])
        for r, c in pts:
            out[r, c] = fill_color
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _pick_seed_positions(region: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count > len(region):
        raise ValueError("region too small for requested seed count")
    cells = region[:]
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return rng.sample(region, count)


def _make_bboxes(
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    inset_rows: list[int],
    inset_cols: list[int],
) -> list[tuple[int, int, int, int]]:
    total_row_inset = sum(inset_rows)
    total_col_inset = sum(inset_cols)
    outer_h = inner_h + 2 * total_row_inset
    outer_w = inner_w + 2 * total_col_inset
    bboxes: list[tuple[int, int, int, int]] = []
    r0 = top
    c0 = left
    r1 = top + outer_h - 1
    c1 = left + outer_w - 1
    bboxes.append((r0, c0, r1, c1))
    for dr, dc in zip(inset_rows, inset_cols):
        r0 += dr
        c0 += dc
        r1 -= dr
        c1 -= dc
        bboxes.append((r0, c0, r1, c1))
    return bboxes


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(MIN_FRAMES, MAX_FRAMES)

    inner_h = rng.randint(4, 9)
    inner_w = rng.randint(4, 9)
    if (inner_h - 2) * (inner_w - 2) < n_frames:
        return None

    inset_rows = [rng.randint(2, 4) for _ in range(n_frames - 1)]
    inset_cols = [rng.randint(2, 4) for _ in range(n_frames - 1)]

    outer_h = inner_h + 2 * sum(inset_rows)
    outer_w = inner_w + 2 * sum(inset_cols)
    if outer_h > 26 or outer_w > 26:
        return None

    grid_h = rng.randint(outer_h + 2, min(30, outer_h + 8))
    grid_w = rng.randint(outer_w + 2, min(30, outer_w + 8))

    top = rng.randint(1, grid_h - outer_h - 1)
    left = rng.randint(1, grid_w - outer_w - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_colors = rng.sample(FRAME_COLOR_POOL, n_frames)
    bboxes = _make_bboxes(top, left, inner_h, inner_w, inset_rows, inset_cols)

    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(grid, bbox, color)

    bg, frames, frame_mask, regions = _analyze_scene(grid)
    if bg != BG or len(frames) != n_frames or len(regions) != n_frames:
        return None

    seeded_region_count = rng.randint(1, n_frames)
    seeded_indices = rng.sample(range(n_frames), seeded_region_count)
    seed_counts = rng.sample(list(range(1, n_frames + 1)), seeded_region_count)
    rng.shuffle(seed_counts)

    for region_index, seed_count in zip(seeded_indices, seed_counts):
        region = regions[region_index]
        if len(region) < seed_count:
            return None
        for r, c in _pick_seed_positions(region, seed_count, rng):
            grid[r, c] = SEED_COLOR

    bg2, frames2, frame_mask2, regions2 = _analyze_scene(grid)
    if bg2 != BG or len(frames2) != n_frames or len(regions2) != n_frames:
        return None

    nonzero_seed_regions = 0
    for region in regions2:
        count = _seed_count_in_region(grid, BG, frame_mask2, region)
        if count:
            nonzero_seed_regions += 1
        if count > n_frames:
            return None
    if nonzero_seed_regions == 0:
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame puzzle input")

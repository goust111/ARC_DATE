from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = tuple(color for color in range(10) if color not in {BG, FRAME_COLOR})
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200

PATTERN_POOL = (
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _frame_candidates(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], np.ndarray]:
    frames: list[dict[str, object]] = []
    frame_mask = np.zeros(grid.shape, dtype=bool)

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 4 or ww < 4:
                continue
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if int(coords.shape[0]) != int(border.sum()):
                continue
            local = np.zeros((hh, ww), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if not np.array_equal(local, border):
                continue
            frame_mask[coords[:, 0], coords[:, 1]] = True
            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})

    return frames, frame_mask


def _corner_score(corner: str, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if corner == "top_left":
        return int(r0 + c0)
    if corner == "top_right":
        return int(r0 + (w - 1 - c1))
    if corner == "bottom_left":
        return int((h - 1 - r1) + c0)
    return int((h - 1 - r1) + (w - 1 - c1))


def _corner_patterns(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> dict[str, np.ndarray]:
    candidates: list[dict[str, object]] = []
    remaining = ~frame_mask

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components((grid == color) & remaining):
            r0, c0, r1, c1 = _bbox(coords)
            if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((patch != bg) & (patch != color)):
                continue
            candidates.append({"bbox": (r0, c0, r1, c1), "mask": patch == color})

    assert len(candidates) == 4, f"expected 4 corner samples, found {len(candidates)}"

    corner_to_mask: dict[str, np.ndarray] = {}
    unused = set(range(len(candidates)))
    for corner in CORNERS:
        idx = min(unused, key=lambda i: _corner_score(corner, candidates[i]["bbox"], grid.shape))
        corner_to_mask[corner] = np.array(candidates[idx]["mask"], dtype=bool)
        unused.remove(idx)

    return corner_to_mask


def _seed_info(grid: np.ndarray, frame: dict[str, object], bg: int) -> tuple[str, int]:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seeds = np.argwhere(inner != bg)
    assert int(seeds.shape[0]) == 1, "each frame interior must contain exactly one seed"
    sr, sc = map(int, seeds[0])
    seed_color = int(inner[sr, sc])

    if sr == 0 and sc == 0:
        return "top_left", seed_color
    if sr == 0 and sc == inner.shape[1] - 1:
        return "top_right", seed_color
    if sr == inner.shape[0] - 1 and sc == 0:
        return "bottom_left", seed_color
    if sr == inner.shape[0] - 1 and sc == inner.shape[1] - 1:
        return "bottom_right", seed_color
    raise AssertionError("seed must sit at one inner corner of the frame")


def _tile_pattern(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0])[:, None] % 3
    cols = np.arange(shape[1])[None, :] % 3
    return mask[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region by tiling the corner sample selected by its seed position.

    Input structure:
    - The background is black.
    - Several one-pixel-thick gray rectangular frames enclose black interiors; each interior contains exactly one colored seed pixel.
    - Each seed sits at one of the four inner corners of its frame.
    - Near the four outer grid corners, there are four separate 3x3 monochrome pattern samples drawn on black.

    Rule:
    - Identify the four 3x3 samples and map them by which outer grid corner they belong to.
    - For each framed region, read which inner corner contains the seed pixel.
    - Choose the sample from the matching outer corner, keep only its shape, and recolor that shape with the seed color.
    - Tile the recolored 3x3 sample across the full frame interior, overwriting the seed and preserving the gray border.

    Key insight:
    - The seed position chooses the template, while the seed color chooses the fill color; the corner samples contribute shape only.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames, frame_mask = _frame_candidates(g, bg)
    assert frames, "at least one rectangular frame is required"
    corner_to_mask = _corner_patterns(g, bg, frame_mask)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        seed_corner, seed_color = _seed_info(g, frame, bg)
        tiled = _tile_pattern(corner_to_mask[seed_corner], (inner_h, inner_w))
        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[:] = bg
        region[tiled] = seed_color

    return out


def _mark_reserved(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _fits_reserved(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0], r0 + h + pad)
    cc1 = min(mask.shape[1], c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _render_pattern(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _corner_position(corner: str, grid_shape: tuple[int, int], offset_r: int, offset_c: int) -> tuple[int, int]:
    h, w = grid_shape
    if corner == "top_left":
        return int(offset_r), int(offset_c)
    if corner == "top_right":
        return int(offset_r), int(w - 3 - offset_c)
    if corner == "bottom_left":
        return int(h - 3 - offset_r), int(offset_c)
    return int(h - 3 - offset_r), int(w - 3 - offset_c)


def _seed_cell(inner_h: int, inner_w: int, corner: str) -> tuple[int, int]:
    if corner == "top_left":
        return 0, 0
    if corner == "top_right":
        return 0, inner_w - 1
    if corner == "bottom_left":
        return inner_h - 1, 0
    return inner_h - 1, inner_w - 1


def _draw_frame_with_seed(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    seed_corner: str,
    seed_color: int,
) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    sr, sc = _seed_cell(inner_h, inner_w, seed_corner)
    grid[top + 1 + sr, left + 1 + sc] = int(seed_color)


def _expected_output(
    grid: np.ndarray,
    frames: list[dict[str, object]],
    corner_patterns: dict[str, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        tiled = _tile_pattern(corner_patterns[frame["seed_corner"]], (inner_h, inner_w))
        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[:] = BG
        region[tiled] = int(frame["seed_color"])
    return out


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    corner_patterns = {
        corner: PATTERN_POOL[idx].copy()
        for corner, idx in zip(CORNERS, random.sample(range(len(PATTERN_POOL)), 4), strict=False)
    }
    sample_colors = random.sample(PALETTE, 4)

    for corner, color in zip(CORNERS, sample_colors, strict=False):
        offset_r = random.randint(0, min(2, h - 3))
        offset_c = random.randint(0, min(2, w - 3))
        r0, c0 = _corner_position(corner, (h, w), offset_r, offset_c)
        grid[r0 : r0 + 3, c0 : c0 + 3] = _render_pattern(corner_patterns[corner], color)
        _mark_reserved(reserved, r0, c0, 3, 3, pad=1)

    region_count = random.randint(2, 4)
    base_corners = random.sample(CORNERS, min(len(CORNERS), region_count))
    if region_count > len(base_corners):
        base_corners.extend(random.choices(CORNERS, k=region_count - len(base_corners)))
    random.shuffle(base_corners)

    frames: list[dict[str, object]] = []
    size_cap = min(8, max(4, min(h, w) // max(3, region_count)))

    for seed_corner in base_corners:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            inner_h = random.randint(3, size_cap)
            inner_w = random.randint(3, size_cap)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h >= h - 1 or outer_w >= w - 1:
                continue
            top = random.randint(1, h - outer_h - 1)
            left = random.randint(1, w - outer_w - 1)
            if not _fits_reserved(reserved, top, left, outer_h, outer_w, pad=1):
                continue
            seed_color = random.choice(PALETTE)
            _draw_frame_with_seed(grid, top, left, inner_h, inner_w, seed_corner, seed_color)
            _mark_reserved(reserved, top, left, outer_h, outer_w, pad=1)
            frames.append(
                {
                    "bbox": (top, left, top + outer_h - 1, left + outer_w - 1),
                    "seed_corner": seed_corner,
                    "seed_color": seed_color,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, frames, corner_patterns)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _sample_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid ARC sample")

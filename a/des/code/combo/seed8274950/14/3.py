from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300
GREY = 5
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 22
FRAME_THICKNESSES = (1, 2)
FRAME_MODES = ("sides", "stripes", "quadrants")


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    counts = Counter(corners)
    color, freq = counts.most_common(1)[0]
    if sum(count == freq for count in counts.values()) == 1:
        return int(color)
    values, value_counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(value_counts))])


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, int(rr))
                r1 = max(r1, int(rr))
                c0 = min(c0, int(cc))
                c1 = max(c1, int(cc))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    comps = _components(g != bg)
    assert comps, "expected foreground content"

    areas = [int(comp["area"]) for comp in comps]
    max_area = max(areas)
    ref_ids = [idx for idx, area in enumerate(areas) if area == max_area]
    assert len(ref_ids) == 1, "expected a unique largest foreground object"
    ref = comps[ref_ids[0]]

    r0, c0, r1, c1 = map(int, ref["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for rr, cc in ref["coords"]:
        patch[int(rr) - r0, int(cc) - c0] = int(g[int(rr), int(cc)])

    singleton_markers: list[tuple[int, int, int]] = []
    for idx, comp in enumerate(comps):
        if idx == ref_ids[0] or int(comp["area"]) != 1:
            continue
        r, c = map(int, comp["coords"][0])
        singleton_markers.append((int(r), int(c), int(g[r, c])))

    assert singleton_markers, "expected singleton probe markers"
    marker_color = Counter(color for _r, _c, color in singleton_markers).most_common(1)[0][0]

    out = g.copy()
    for r, c, color in singleton_markers:
        if int(color) != int(marker_color):
            continue
        rr = int(r - r0)
        cc = int(c - c0)
        if 0 <= rr < patch.shape[0] and 0 <= cc < patch.shape[1]:
            out[r, c] = int(patch[rr, cc])
    return out


def _frame_mask(height: int, width: int, thickness: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:thickness, :] = True
    mask[-thickness:, :] = True
    mask[:, :thickness] = True
    mask[:, -thickness:] = True
    return mask


def _frame_colors(height: int, width: int, thickness: int, colors: list[int], mode: str) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    mask = _frame_mask(height, width, thickness)

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]):
                continue
            if mode == "sides":
                if r < thickness:
                    idx = 0
                elif c >= width - thickness:
                    idx = 1
                elif r >= height - thickness:
                    idx = 2
                else:
                    idx = 3
            elif mode == "stripes":
                idx = (r + c) % len(colors)
            else:
                vertical = 0 if r < height / 2 else 1
                horizontal = 0 if c < width / 2 else 1
                idx = vertical * 2 + horizontal
            patch[r, c] = int(colors[idx % len(colors)])

    used = {int(value) for value in np.unique(patch[mask])}
    assert len(used) >= 2, "reference object must be multi-color"
    return patch


def _can_place_marker(grid: np.ndarray, row: int, col: int, bg: int, placed: set[tuple[int, int]]) -> bool:
    if int(grid[row, col]) != int(bg):
        return False
    h, w = map(int, grid.shape)
    for rr in range(max(0, row - 1), min(h, row + 2)):
        for cc in range(max(0, col - 1), min(w, col + 2)):
            if (rr, cc) in placed:
                return False
            if rr == row and cc == col:
                continue
            if int(grid[rr, cc]) != int(bg):
                return False
    return True


def _select_markers(
    grid: np.ndarray,
    candidates: list[tuple[int, int]],
    count: int,
    bg: int,
    rng: np.random.Generator,
    placed: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    order = list(rng.permutation(len(candidates)))
    chosen: list[tuple[int, int]] = []
    for idx in order:
        row, col = map(int, candidates[idx])
        if _can_place_marker(grid, row, col, bg, placed):
            chosen.append((int(row), int(col)))
            placed.add((int(row), int(col)))
            if len(chosen) == count:
                break
    return chosen


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    width = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))

    palette = [color for color in range(10) if color != GREY]
    bg = int(rng.choice(palette))
    ref_palette = [color for color in range(10) if color not in {bg, GREY}]
    n_ref_colors = int(rng.integers(2, 4))
    ref_colors = list(map(int, rng.choice(ref_palette, size=n_ref_colors, replace=False)))

    thickness = int(rng.choice(FRAME_THICKNESSES))
    min_dim = 2 * thickness + 5
    max_height = min(height - 4, 12)
    max_width = min(width - 4, 12)
    if max_height < min_dim or max_width < min_dim:
        return None

    frame_h = int(rng.integers(min_dim, max_height + 1))
    frame_w = int(rng.integers(min_dim, max_width + 1))
    r0 = int(rng.integers(1, height - frame_h - 1))
    c0 = int(rng.integers(1, width - frame_w - 1))
    r1 = int(r0 + frame_h - 1)
    c1 = int(c0 + frame_w - 1)
    bbox = (int(r0), int(c0), int(r1), int(c1))

    grid = np.full((height, width), bg, dtype=int)
    mode = str(rng.choice(FRAME_MODES))
    patch = _frame_colors(frame_h, frame_w, thickness, ref_colors, mode)
    frame_mask = _frame_mask(frame_h, frame_w, thickness)
    grid[r0 : r1 + 1, c0 : c1 + 1][frame_mask] = patch[frame_mask]

    inside_candidates: list[tuple[int, int]] = []
    for rr in range(r0 + thickness + 1, r1 - thickness):
        for cc in range(c0 + thickness + 1, c1 - thickness):
            if all(int(grid[rr + dr, cc + dc]) == bg for dr, dc in DIR4):
                inside_candidates.append((int(rr), int(cc)))
    if len(inside_candidates) < 1:
        return None

    outside_candidates: list[tuple[int, int]] = []
    for rr in range(height):
        for cc in range(width):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            outside_candidates.append((int(rr), int(cc)))

    placed: set[tuple[int, int]] = set()
    max_inside = min(4, len(inside_candidates))
    inside_count = int(rng.integers(1, max_inside + 1))
    outside_count = 0
    if rng.random() < 0.55:
        outside_count = int(rng.integers(1, min(4, len(outside_candidates)) + 1))

    inside_markers = _select_markers(grid, inside_candidates, inside_count, bg, rng, placed)
    outside_markers = _select_markers(grid, outside_candidates, outside_count, bg, rng, placed)
    if len(inside_markers) != inside_count or len(outside_markers) != outside_count:
        return None

    all_markers = inside_markers + outside_markers
    for row, col in all_markers:
        grid[row, col] = GREY

    if Counter(grid.ravel()).most_common(1)[0][0] != bg:
        return None

    comps = _components(grid != bg)
    areas = sorted(int(comp["area"]) for comp in comps)
    if not areas or areas[-1] <= 4:
        return None
    if len(areas) > 1 and areas[-1] == areas[-2]:
        return None

    ref_view = np.full((frame_h, frame_w), bg, dtype=int)
    ref_view[frame_mask] = patch[frame_mask]
    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1][frame_mask], grid[r0 : r1 + 1, c0 : c1 + 1][frame_mask]):
        return None
    for row, col in all_markers:
        rr = int(row - r0)
        cc = int(col - c0)
        value = GREY
        if 0 <= rr < ref_view.shape[0] and 0 <= cc < ref_view.shape[1]:
            value = int(ref_view[rr, cc])
        if int(out[row, col]) != int(value):
            return None
    if outside_markers and not all(int(out[row, col]) == GREY for row, col in outside_markers):
        return None
    if not all(int(out[row, col]) == bg for row, col in inside_markers):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

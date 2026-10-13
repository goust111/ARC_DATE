from __future__ import annotations

from collections import Counter, deque
from itertools import permutations
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
GlyphKey = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


_PATTERN_POOL = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _mask_components(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(int(r) in {0, h - 1} or int(c) in {0, w - 1} for r, c in coords)


def _pattern_key(mask: np.ndarray) -> GlyphKey:
    arr = np.asarray(mask, dtype=int)
    return tuple(tuple(int(v) for v in row) for row in arr)  # type: ignore[return-value]


def _blob_holes(comp: dict) -> list[Coord] | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = comp["coords"] - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[local[:, 0], local[:, 1]] = True

    holes: list[Coord] = []
    for bg_comp in _mask_components(~mask):
        if _touches_border(bg_comp, mask.shape):
            continue
        if int(bg_comp.shape[0]) != 1:
            return None
        holes.append((int(bg_comp[0, 0]), int(bg_comp[0, 1])))
    return holes or None


def _legend_map(grid: np.ndarray, blob_mask: np.ndarray, bg: int) -> dict[GlyphKey, int]:
    remain = np.asarray(grid, dtype=int).copy()
    remain[blob_mask] = int(bg)
    legend: dict[GlyphKey, int] = {}

    for comp in _find_components(remain, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if r1 + 1 >= remain.shape[0]:
            continue

        cols = range(c0, c1 + 1)
        below = [(r1 + 1, c) for c in cols if int(remain[r1 + 1, c]) != int(bg)]
        if len(below) != 1:
            continue

        patch = np.zeros((3, 3), dtype=int)
        local = comp["coords"] - np.array([r0, c0], dtype=int)
        patch[local[:, 0], local[:, 1]] = 1
        key = _pattern_key(patch)
        assert key not in legend, "legend glyphs must be unique"
        legend[key] = int(remain[below[0]])

    assert len(legend) >= 2, "expected at least two legend entries"
    return legend


def _hole_glyph_candidates(holes: list[Coord]) -> set[GlyphKey]:
    rows = [int(r) for r, _ in holes]
    cols = [int(c) for _, c in holes]
    max_step = max(max(rows) - min(rows), max(cols) - min(cols), 1)
    candidates: set[GlyphKey] = set()

    for step in range(1, max_step + 1):
        row_origins = {r - k * step for r in rows for k in range(3)}
        col_origins = {c - k * step for c in cols for k in range(3)}
        for r0 in row_origins:
            for c0 in col_origins:
                patch = np.zeros((3, 3), dtype=int)
                ok = True
                for r, c in holes:
                    dr = int(r) - int(r0)
                    dc = int(c) - int(c0)
                    if dr < 0 or dc < 0 or dr > 2 * step or dc > 2 * step:
                        ok = False
                        break
                    if dr % step != 0 or dc % step != 0:
                        ok = False
                        break
                    patch[dr // step, dc // step] = 1
                if ok:
                    candidates.add(_pattern_key(patch))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each holed blob by matching its normalized hole glyph to the side legend.

    Input structure:
    - The background is dominant; the main objects are several large monochromatic blobs.
    - Each blob contains fully enclosed single-cell background holes, and those hole positions lie on a scaled 3x3 lattice subset.
    - A side legend shows several 3x3 glyphs and one color pixel directly below each glyph.

    Rule:
    - Detect every large blob by finding monochromatic components that contain enclosed background holes.
    - Normalize each blob's hole pattern back to a 3x3 glyph by undoing its lattice spacing, then find the identical glyph in the legend.
    - Repaint the entire blob with the legend's color while keeping its holes as background; leave the legend and background unchanged.

    Key insight:
    - The blob's outline and original fill color are decoys; the identity lives in the negative-space hole geometry, so matching must be done after scale normalization.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    blob_mask = np.zeros(g.shape, dtype=bool)
    blobs: list[dict] = []
    for comp in _find_components(g, bg):
        holes = _blob_holes(comp)
        if holes is None:
            continue
        blobs.append({"coords": comp["coords"], "holes": holes})
        blob_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    assert len(blobs) >= 2, "expected multiple blobs"
    legend = _legend_map(g, blob_mask, bg)

    out = g.copy()
    for blob in blobs:
        matches = [key for key in _hole_glyph_candidates(blob["holes"]) if key in legend]
        assert len(matches) == 1, "hole glyph must match exactly one legend entry"
        out[blob["coords"][:, 0], blob["coords"][:, 1]] = int(legend[matches[0]])
    return out


def _make_blob_patch(pattern: np.ndarray, color: int, bg: int) -> np.ndarray:
    step = random.choice([2, 3])
    base = np.full((2 * step + 3, 2 * step + 3), int(color), dtype=int)
    for r, c in np.argwhere(pattern):
        base[1 + int(r) * step, 1 + int(c) * step] = int(bg)

    if random.random() < 0.45:
        return base

    side = random.choice(["top", "bottom", "left", "right"])
    h, w = base.shape
    if side in {"top", "bottom"}:
        patch = np.full((h + 1, w), int(bg), dtype=int)
        row_off = 1 if side == "top" else 0
        patch[row_off : row_off + h, :] = base
        seg_len = random.randint(2, w - 1)
        seg_c0 = random.randint(0, w - seg_len)
        patch[0 if side == "top" else h, seg_c0 : seg_c0 + seg_len] = int(color)
        return patch

    patch = np.full((h, w + 1), int(bg), dtype=int)
    col_off = 1 if side == "left" else 0
    patch[:, col_off : col_off + w] = base
    seg_len = random.randint(2, h - 1)
    seg_r0 = random.randint(0, h - seg_len)
    patch[seg_r0 : seg_r0 + seg_len, 0 if side == "left" else w] = int(color)
    return patch


def _make_legend_entry(pattern: np.ndarray, glyph_color: int, target_color: int, bg: int) -> np.ndarray:
    patch = np.full((4, 3), int(bg), dtype=int)
    patch[:3, :][pattern] = int(glyph_color)
    patch[3, 1] = int(target_color)
    return patch


def _layout_blobs(patches: list[np.ndarray], legend_h: int) -> tuple[list[Coord], int, int] | None:
    feasible = []
    for slot_ids in permutations(range(4), len(patches)):
        row_heights = [0, 0]
        col_widths = [0, 0]
        for patch, slot in zip(patches, slot_ids):
            row = slot // 2
            col = slot % 2
            row_heights[row] = max(row_heights[row], int(patch.shape[0]))
            col_widths[col] = max(col_widths[col], int(patch.shape[1]))

        row_gap = 2 if row_heights[0] and row_heights[1] else 0
        col_gap = 2 if col_widths[0] and col_widths[1] else 0
        row_starts = [1, 1 + row_heights[0] + row_gap]
        col_starts = [1, 1 + col_widths[0] + col_gap]

        left_h = max((row_starts[i] + row_heights[i] for i in range(2) if row_heights[i]), default=1) + 1
        left_w = max((col_starts[i] + col_widths[i] for i in range(2) if col_widths[i]), default=1) + 1
        total_h = max(left_h, legend_h + 2)
        total_w = left_w + 2 + 3 + 1
        if total_h <= 30 and total_w <= 30:
            feasible.append((slot_ids, row_heights, col_widths, left_h, left_w))

    if not feasible:
        return None

    slot_ids, row_heights, col_widths, left_h, left_w = random.choice(feasible)
    row_gap = 2 if row_heights[0] and row_heights[1] else 0
    col_gap = 2 if col_widths[0] and col_widths[1] else 0
    row_starts = [1, 1 + row_heights[0] + row_gap]
    col_starts = [1, 1 + col_widths[0] + col_gap]

    positions: list[Coord] = []
    for patch, slot in zip(patches, slot_ids):
        row = slot // 2
        col = slot % 2
        slack_r = row_heights[row] - int(patch.shape[0])
        slack_c = col_widths[col] - int(patch.shape[1])
        rr = row_starts[row] + random.randint(0, slack_r)
        cc = col_starts[col] + random.randint(0, slack_c)
        positions.append((rr, cc))
    return positions, left_h, left_w


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    legend_count = random.choice([3, 4])
    blob_count = random.randint(2, legend_count)
    patterns = random.sample(_PATTERN_POOL, legend_count)
    used = random.sample(range(legend_count), blob_count)

    palette = random.sample(range(1, 10), legend_count + blob_count + 1)
    target_colors = palette[:legend_count]
    blob_colors = palette[legend_count : legend_count + blob_count]
    glyph_color = palette[-1]

    blob_specs = []
    for i, pat_idx in enumerate(used):
        pattern = patterns[pat_idx]
        patch = _make_blob_patch(pattern, blob_colors[i], bg)
        blob_specs.append((patch, target_colors[pat_idx]))

    legend_entries = [_make_legend_entry(pattern, glyph_color, target, bg) for pattern, target in zip(patterns, target_colors)]
    legend_h = 4 * legend_count + (legend_count - 1)
    layout = _layout_blobs([patch for patch, _target in blob_specs], legend_h)
    if layout is None:
        return None
    blob_positions, left_h, left_w = layout

    total_h = max(left_h, legend_h + 2)
    total_w = left_w + 2 + 3 + 1
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), bg, dtype=int)
    expected = np.full_like(grid, bg)

    for (patch, target), (r0, c0) in zip(blob_specs, blob_positions):
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        expected[r0 : r0 + h, c0 : c0 + w] = patch
        expected[r0 : r0 + h, c0 : c0 + w][patch != bg] = int(target)

    legend_c0 = left_w + 2
    legend_r0 = random.randint(1, total_h - legend_h - 1)
    for i, entry in enumerate(random.sample(legend_entries, len(legend_entries))):
        r0 = legend_r0 + i * 5
        grid[r0 : r0 + 4, legend_c0 : legend_c0 + 3] = entry
        expected[r0 : r0 + 4, legend_c0 : legend_c0 + 3] = entry

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid holed-glyph ARC input")

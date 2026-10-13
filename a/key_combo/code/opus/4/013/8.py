from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _arc_utils = SourceFileLoader("_arc_utils_key_combo_013_8", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    bbox = _arc_utils.bbox
    crop_bbox = _arc_utils.crop_bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BG = int(Color.BLACK)
MASK_COLOR = int(Color.GREY)
EDGE_COLORS = [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW)]
MAX_GENERATION_ATTEMPTS = 300


def _column_density_bands(mask: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    kron = np.kron(mask.astype(int), mask.astype(int))
    counts = [int(v) for v in kron.sum(axis=0).tolist()]
    densities = sorted(set(counts), reverse=True)
    lanes = [next(c for c, value in enumerate(counts) if value == density) for density in densities]
    return counts, densities, lanes


def _raw_kron_rank(counts: list[int]) -> list[int]:
    return sorted(range(len(counts)), key=lambda c: (-counts[c], c))


def _expanded_mask_rank(mask: np.ndarray) -> list[int]:
    expanded = np.repeat(mask.sum(axis=0).astype(int), 3)
    return sorted(range(9), key=lambda c: (-int(expanded[c]), c))


def _extract_mask(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    grey = grid == MASK_COLOR
    box = bbox(grey)
    assert box is not None, "expected a grey control mask"
    patch = crop_bbox(grid, box)
    assert patch.shape == (3, 3), f"expected a 3x3 mask bbox, got {patch.shape}"
    mask = (patch == MASK_COLOR).astype(int)
    assert 3 <= int(mask.sum()) <= 6, "mask must have 3..6 lit cells"
    return box, mask


def _edge_pixels_and_order(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
    pixels: list[tuple[int, int]] = []
    order: list[int] = []
    seen: set[int] = set()
    for r, color in enumerate(grid[:, 0].astype(int).tolist()):
        if color == BG:
            continue
        pixels.append((int(r), int(color)))
        if color not in seen:
            seen.add(color)
            order.append(int(color))
    assert order, "expected at least one left-edge color"
    return pixels, order


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask_box, mask = _extract_mask(g)
    counts, densities, lanes = _column_density_bands(mask)
    edge_pixels, color_order = _edge_pixels_and_order(g)
    assert len(color_order) <= len(lanes), "not enough density bands for the edge colors"

    mapping = {color: int(lane) for color, lane in zip(color_order, lanes)}
    out = g.copy()

    for r, _color in edge_pixels:
        out[r, 0] = BG

    r0, c0, r1, c1 = mask_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for r, color in edge_pixels:
        out[r, mapping[color]] = int(color)

    if not return_meta:
        return out

    meta = {
        "mask_box": mask_box,
        "mask": mask,
        "column_counts": counts,
        "density_bands": densities,
        "lanes": lanes,
        "color_order": color_order,
        "mapping": mapping,
        "edge_pixels": edge_pixels,
        "raw_kron_rank": _raw_kron_rank(counts),
        "expanded_mask_rank": _expanded_mask_rank(mask),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route left-edge colored pixels into dynamic lane columns derived from a 3x3 mask's Kronecker-self column densities.

    Input structure:
    - A black background grid contains several colored pixels on the left edge in column 0.
    - A separate 3x3 control region uses grey cells as a binary mask and black cells as empty cells.
    - No other non-black objects are present.

    Rule:
    - Extract the 3x3 grey mask, convert it to a binary mask, and build its 9x9 Kronecker self-product.
    - Count the lit cells in each of the 9 Kronecker columns and sort density bands from highest to lowest; when multiple columns share one density, use the leftmost column from that band as its representative lane.
    - Read the distinct left-edge colors by their first top-to-bottom appearance, map the first color to the highest-density lane, the second color to the next lane, and so on.
    - Erase the original left-edge pixels and the 3x3 mask, then redraw every moved pixel on its original row in the mapped target column.

    Key insight:
    - The lane assignment is not a fixed color rule: it must be computed from the self-similar Kronecker expansion of the grey control mask before any pixels can be routed.
    """
    return _solve(grid)


def _build_mask_candidates() -> list[dict]:
    candidates: list[dict] = []
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
        lit = int(mask.sum())
        if lit < 3 or lit > 6:
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        counts, densities, lanes = _column_density_bands(mask)
        if len(lanes) < 4:
            continue
        if any(lane == 0 for lane in lanes[:4]):
            continue
        raw_rank = _raw_kron_rank(counts)
        expanded_rank = _expanded_mask_rank(mask)
        if lanes[:3] == raw_rank[:3]:
            continue
        if lanes[:3] == expanded_rank[:3]:
            continue
        candidates.append(
            {
                "mask": mask,
                "counts": counts,
                "densities": densities,
                "lanes": lanes,
                "raw_rank": raw_rank,
                "expanded_rank": expanded_rank,
            }
        )
    assert candidates, "expected at least one non-degenerate mask candidate"
    return candidates


MASK_CANDIDATES = _build_mask_candidates()


def _choose_color_count() -> int:
    roll = random.random()
    if roll < 0.2:
        return 2
    if roll < 0.6:
        return 3
    return 4


def _shortcut_output_without_kronecker(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask_box, mask = _extract_mask(g)
    lanes = _expanded_mask_rank(mask)
    edge_pixels, color_order = _edge_pixels_and_order(g)

    out = g.copy()
    for r, _color in edge_pixels:
        out[r, 0] = BG

    r0, c0, r1, c1 = mask_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for color, lane in zip(color_order, lanes):
        for r, pixel_color in edge_pixels:
            if pixel_color == color:
                out[r, lane] = pixel_color
    return out


def _shortcut_output_raw_kron_rank(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask_box, mask = _extract_mask(g)
    counts, _densities, _lanes = _column_density_bands(mask)
    rank = _raw_kron_rank(counts)
    edge_pixels, color_order = _edge_pixels_and_order(g)

    out = g.copy()
    for r, _color in edge_pixels:
        out[r, 0] = BG

    r0, c0, r1, c1 = mask_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG

    mapping = {color: lane for color, lane in zip(color_order, rank)}
    for r, color in edge_pixels:
        out[r, mapping[color]] = color
    return out


def _instance_is_valid(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, return_meta=True)
    except Exception:
        return False

    mask = meta["mask"]
    color_order = meta["color_order"]
    lanes = meta["lanes"]
    mapping = meta["mapping"]
    densities = meta["density_bands"]
    edge_pixels = meta["edge_pixels"]

    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not (3 <= int(mask.sum()) <= 6):
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False
    if len(color_order) < 2 or len(color_order) > 4:
        return False
    if len(densities) < len(color_order):
        return False
    used_lanes = [mapping[color] for color in color_order]
    if len(set(used_lanes)) != len(used_lanes):
        return False
    if any(lane == 0 for lane in used_lanes):
        return False
    if used_lanes != lanes[: len(color_order)]:
        return False
    if not all(densities[i] > densities[i + 1] for i in range(len(color_order) - 1)):
        return False
    if np.array_equal(out, grid):
        return False

    expected_rows = [r for r, _color in edge_pixels]
    if any(int(out[r, 0]) != BG for r in expected_rows):
        return False

    no_kron = _shortcut_output_without_kronecker(grid)
    if np.array_equal(no_kron, out):
        return False

    if len(color_order) >= 3:
        raw_kron = _shortcut_output_raw_kron_rank(grid)
        if np.array_equal(raw_kron, out):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 15)
    w = random.randint(12, 16)
    mask_info = random.choice(MASK_CANDIDATES)
    n_colors = min(_choose_color_count(), len(mask_info["lanes"]))

    grid = np.zeros((h, w), dtype=int)
    mask_r0 = random.randint(0, h - 3)
    mask_c0 = random.randint(9, w - 3)
    mask = mask_info["mask"]
    mask_view = grid[mask_r0 : mask_r0 + 3, mask_c0 : mask_c0 + 3]
    mask_view[mask.astype(bool)] = MASK_COLOR

    palette = random.sample(EDGE_COLORS, n_colors)
    extra_pixels = random.randint(0, min(4, h - n_colors))
    rows = sorted(random.sample(range(h), n_colors + extra_pixels))

    placements = [(rows[i], palette[i]) for i in range(n_colors)]
    for row in rows[n_colors:]:
        placements.append((row, random.choice(palette)))

    for row, color in sorted(placements):
        grid[row, 0] = int(color)

    if not _instance_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")

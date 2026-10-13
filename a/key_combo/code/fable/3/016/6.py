"""Swap layout and tile masks in a two-level Kronecker structure.

Input structure
- The grid has black background and exactly one non-black foreground color.
- Foreground pixels form `kron(layout, tile)` where `layout` and `tile` are two different
  binary masks (the layout is the outer factor, the tile is the inner factor).
- The two factors are uniquely recoverable from the input.

Rule:
- Recover the two binary factors from the cropped foreground bbox such that
  `foreground = kron(layout, tile)`.
- Build a new crop as `kron(tile, layout)`, then place it back at the same bbox.
- Keep the same size as the original grid and preserve background and foreground color.

Key insight:
- The pattern is a Kronecker product with unknown inner/outer order.
- Swapping hierarchy is exactly equivalent to swapping the two binary factors of that factorization.
"""

from __future__ import annotations

import sys
from typing import Iterable, List, Tuple

import numpy as np


UTILS_PATH = "/home/lixinyu/snap/new/solvers"
if UTILS_PATH not in sys.path:
    sys.path.append(UTILS_PATH)

from arc_utils import Color, bbox, get_bg


def _iter_divisors(n: int) -> List[int]:
    out: List[int] = []
    for k in range(1, int(n) + 1):
        if n % k == 0:
            out.append(k)
    return out


def _to_bool(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask, dtype=bool).copy()).astype(bool)


def _decompose_kron(mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    """Return all valid `(layout, tile, (tile_h, tile_w))` factorizations."""
    m = _to_bool(mask)
    H, W = m.shape
    if H == 0 or W == 0 or not m.any():
        return []

    candidates: List[Tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []

    for tile_h in _iter_divisors(H):
        for tile_w in _iter_divisors(W):
            layout_h = H // tile_h
            layout_w = W // tile_w

            # both factors must be non-trivial in this task
            if (tile_h, tile_w) == (H, W):
                continue
            if layout_h * layout_w == 1:
                continue
            if tile_h * tile_w < 2:
                continue

            layout = np.zeros((layout_h, layout_w), dtype=bool)
            tile: np.ndarray | None = None
            ok = True

            for r in range(layout_h):
                for c in range(layout_w):
                    r0, c0 = r * tile_h, c * tile_w
                    block = m[r0 : r0 + tile_h, c0 : c0 + tile_w]
                    if block.any():
                        if tile is None:
                            tile = block.copy()
                        elif not np.array_equal(block, tile):
                            ok = False
                            break
                        layout[r, c] = True
                if not ok:
                    break

            if not ok or tile is None:
                continue

            if int(tile.sum()) < 2:
                continue
            if int(layout.sum()) < 2:
                continue

            # Prevent the degenerate "same-shape" interpretation.
            if int(tile_h) == int(layout_h) and int(tile_w) == int(layout_w):
                if tile.shape == layout.shape and np.array_equal(tile, layout):
                    continue

            rebuilt = np.kron(layout.astype(int), tile.astype(int)).astype(bool)
            if not np.array_equal(rebuilt, m):
                continue

            candidates.append((layout, tile, (tile_h, tile_w)))

    return candidates


def _unique_decomposition(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    candidates = _decompose_kron(mask)
    if len(candidates) != 1:
        raise AssertionError(
            "Expected exactly one valid two-level binary factorization "
            f"but found {len(candidates)}."
        )
    layout, tile, tile_size = candidates[0]
    if int(layout.sum()) < 2 or int(tile.sum()) < 2:
        raise AssertionError("Expected both factors to have at least 2 foreground pixels.")
    return layout, tile, tile_size


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg = np.setdiff1d(np.unique(g), [bg], assume_unique=False)
    if fg.size != 1:
        raise AssertionError("expected exactly one foreground color")
    fg = int(fg[0])

    full_mask = (g == fg)
    assert full_mask.any(), "expected foreground"
    box = bbox(full_mask)
    assert box is not None, "failed to compute foreground bbox"
    r0, c0, r1, c1 = box
    crop = _to_bool(full_mask[r0 : r1 + 1, c0 : c1 + 1])

    layout, tile, _ = _unique_decomposition(crop)
    swapped = np.kron(tile.astype(int), layout.astype(int)).astype(bool)
    assert swapped.shape == crop.shape, "swapped factorization shape mismatch"

    out = np.full_like(g, bg)
    out[r0 : r1 + 1, c0 : c1 + 1] = np.where(swapped, fg, bg)
    return out


def _random_connected_mask(
    rng: np.random.Generator, height: int, width: int, target_pixels: int, max_steps: int = 1000
) -> np.ndarray:
    """Build a connected binary mask with approximately `target_pixels` pixels."""
    if target_pixels < 2:
        raise ValueError("target_pixels must be at least 2")

    for _ in range(40):
        h0 = int(rng.integers(0, height))
        w0 = int(rng.integers(0, width))
        mask = np.zeros((height, width), dtype=bool)
        mask[h0, w0] = True
        cells = [(h0, w0)]
        frontier = [(h0, w0)]

        while mask.sum() < target_pixels and len(frontier) > 0 and max_steps > 0:
            max_steps -= 1
            base_idx = int(rng.integers(0, len(frontier)))
            br, bc = frontier[base_idx]
            nbrs = [(br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)]
            rng.shuffle(nbrs)

            grew = False
            for nr, nc in nbrs:
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    if int(mask.sum()) < target_pixels:
                        mask[nr, nc] = True
                        cells.append((nr, nc))
                        frontier.append((nr, nc))
                        grew = True
                        break
            if not grew:
                frontier.pop(base_idx)

        if int(mask.sum()) >= target_pixels:
            b = bbox(mask)
            if b is None:
                continue
            rr0, cc0, rr1, cc1 = b
            cropped = mask[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if int(cropped.sum()) >= 2 and cropped.shape[0] >= 2 and cropped.shape[1] >= 2:
                return cropped

    raise RuntimeError("failed to generate a connected mask")


def _shape_area_range_for_factor(height: int, width: int, rng: np.random.Generator) -> int:
    min_a = 2
    max_a = max(min_a, int(0.7 * height * width))
    if max_a < min_a:
        raise ValueError("invalid factor area range")
    return int(rng.integers(min_a, max_a + 1))


def _compose(k_outer: np.ndarray, k_inner: np.ndarray) -> np.ndarray:
    return np.kron(k_outer.astype(int), k_inner.astype(int)).astype(bool)


def _generate_candidate(rng: np.random.Generator) -> np.ndarray:
    # Choose small factors and keep output dimensions within ARC bounds.
    outer_h = int(rng.integers(2, 8))
    outer_w = int(rng.integers(2, 8))
    inner_h = int(rng.integers(2, 8))
    inner_w = int(rng.integers(2, 8))

    if outer_h * inner_h > 30 or outer_w * inner_w > 30:
        raise RuntimeError("shape too large")

    outer_area = _shape_area_range_for_factor(outer_h, outer_w, rng)
    inner_area = _shape_area_range_for_factor(inner_h, inner_w, rng)
    outer = _random_connected_mask(rng, outer_h, outer_w, outer_area)
    inner = _random_connected_mask(rng, inner_h, inner_w, inner_area)

    if np.array_equal(outer, inner):
        outer = np.fliplr(outer) if not np.array_equal(np.fliplr(outer), inner) else np.flipud(outer)

    input_core = _compose(outer, inner)
    if not input_core.any() or not input_core.sum() > 1:
        raise RuntimeError("invalid core shape")

    # Ensure unique decomposition exists and is not ambiguous on the chosen core.
    decomps = _decompose_kron(input_core)
    if len(decomps) != 1:
        raise RuntimeError("candidate core is ambiguous")
    layout, tile, _ = decomps[0]
    if not np.array_equal(layout, outer) or not np.array_equal(tile, inner):
        raise RuntimeError("decomposition does not match generator mask")

    swapped_expected = _compose(inner, outer)
    if not swapped_expected.any():
        raise RuntimeError("swap produced empty output")

    return input_core


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = Color.BLACK
    palette = [c for c in Color.NOT_BLACK if c != bg]

    for _ in range(700):
        try:
            core = _generate_candidate(rng)
        except Exception:
            continue

        h_core, w_core = core.shape
        # Add a bit of canvas variability and placement variability.
        h = int(rng.integers(h_core, min(30, h_core + 8) + 1))
        w = int(rng.integers(w_core, min(30, w_core + 8) + 1))
        r0 = int(rng.integers(0, h - h_core + 1))
        c0 = int(rng.integers(0, w - w_core + 1))

        grid = np.full((h, w), int(bg), dtype=int)
        fg = int(rng.choice(palette))
        grid[r0 : r0 + h_core, c0 : c0 + w_core] = np.where(core, fg, bg)

        # Guard the semantics by recomputing expected output and comparing with main.
        try:
            out = main(grid)
        except Exception:
            continue

        # Recover decomposition from the generated core to build a direct oracle.
        crop = bbox(grid == fg)
        if crop is None:
            continue
        cr0, cc0, cr1, cc1 = crop
        core_crop = _to_bool(grid[cr0 : cr1 + 1, cc0 : cc1 + 1] == fg)
        try:
            layout, tile, _ = _unique_decomposition(core_crop)
        except Exception:
            continue

        expected = np.full_like(grid, int(bg), dtype=int)
        expected[cr0 : cr1 + 1, cc0 : cc1 + 1] = np.where(
            np.kron(tile.astype(int), layout.astype(int)).astype(bool), fg, bg
        )
        if not np.array_equal(out, expected):
            continue
        if np.array_equal(out, grid):
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")

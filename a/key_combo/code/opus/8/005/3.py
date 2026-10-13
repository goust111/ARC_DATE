from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, components_mask


_COMPONENT_SHAPES = {
    2: [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ],
}

_INTERSECTION_PATTERNS = [
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (2, 1), (2, 2)),
]

_EXTRA_PARTITIONS = {
    2: [(2,)],
    3: [(3,)],
    4: [(4,), (2, 2)],
    5: [(2, 3), (3, 2)],
    6: [(2, 4), (4, 2), (3, 3)],
}


def _normalize(coords: np.ndarray) -> np.ndarray:
    shifted = coords - coords.min(axis=0)
    order = np.lexsort((shifted[:, 1], shifted[:, 0]))
    return shifted[order]


def _transform_coords(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> np.ndarray:
    pts = np.asarray(shape, dtype=int)
    if flip:
        pts = pts.copy()
        pts[:, 1] *= -1
    for _ in range(rot):
        pts = np.column_stack((pts[:, 1], -pts[:, 0]))
    return _normalize(pts)


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for rot in range(4):
        for flip in (False, True):
            pts = _transform_coords(shape, rot, flip)
            key = tuple(map(tuple, pts.tolist()))
            variants[key] = pts
    return list(variants.values())


def _denoise(mask: np.ndarray) -> np.ndarray:
    clean = np.zeros_like(mask, dtype=bool)
    for comp in components_mask(mask.astype(bool), connectivity=8):
        if int(comp.shape[0]) >= 2:
            clean[comp[:, 0], comp[:, 1]] = True
    return clean


def _translate(coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return coords + np.array([int(r0), int(c0)], dtype=int)


def _can_place(mask: np.ndarray, coords: np.ndarray) -> bool:
    h, w = mask.shape
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if not (0 <= r < h and 0 <= c < w):
            return False
        if mask[r, c]:
            return False
        if mask[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)].any():
            return False
    return True


def _apply_coords(mask: np.ndarray, coords: np.ndarray) -> None:
    mask[coords[:, 0], coords[:, 1]] = True


def _sample_cluster(size: int) -> np.ndarray:
    base = random.choice(_COMPONENT_SHAPES[size])
    return random.choice(_shape_variants(base))


def _place_cluster(
    mask: np.ndarray,
    size: int,
    *,
    forbidden_exact: set[tuple[int, int]] | None = None,
    attempts: int = 120,
) -> np.ndarray | None:
    forbidden_exact = forbidden_exact or set()
    for _ in range(attempts):
        shape = _sample_cluster(size)
        shape_h = int(shape[:, 0].max()) + 1
        shape_w = int(shape[:, 1].max()) + 1
        r0 = random.randint(0, mask.shape[0] - shape_h)
        c0 = random.randint(0, mask.shape[1] - shape_w)
        placed = _translate(shape, r0, c0)
        if any((int(r), int(c)) in forbidden_exact for r, c in placed):
            continue
        if _can_place(mask, placed):
            _apply_coords(mask, placed)
            return placed
    return None


def _place_intersection(panel_h: int, panel_w: int) -> np.ndarray | None:
    for _ in range(120):
        base = random.choice(_INTERSECTION_PATTERNS)
        shape = random.choice(_shape_variants(base))
        shape_h = int(shape[:, 0].max()) + 1
        shape_w = int(shape[:, 1].max()) + 1
        if shape_h > panel_h or shape_w > panel_w:
            continue
        r0 = random.randint(0, panel_h - shape_h)
        c0 = random.randint(0, panel_w - shape_w)
        return _translate(shape, r0, c0)
    return None


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _build_clean_panels(panel_h: int, panel_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    shared = _place_intersection(panel_h, panel_w)
    if shared is None:
        return None

    top = _mask_from_coords((panel_h, panel_w), shared)
    bottom = _mask_from_coords((panel_h, panel_w), shared)

    extra_total = random.choice((2, 3, 4, 5, 6))
    top_parts = random.choice(_EXTRA_PARTITIONS[extra_total])
    bottom_parts = random.choice(_EXTRA_PARTITIONS[extra_total])

    top_extra_cells: set[tuple[int, int]] = set()
    for size in top_parts:
        placed = _place_cluster(top, size)
        if placed is None:
            return None
        for r, c in placed:
            top_extra_cells.add((int(r), int(c)))

    for size in bottom_parts:
        placed = _place_cluster(bottom, size, forbidden_exact=top_extra_cells)
        if placed is None:
            return None

    if np.any(top & bottom & ~_mask_from_coords((panel_h, panel_w), shared)):
        return None

    return top, bottom


def _place_noise(mask: np.ndarray, count: int) -> bool:
    h, w = mask.shape
    for _ in range(count):
        candidates = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if mask[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        mask[r0, c0] = True
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove singleton noise in two aligned panels, intersect the cleaned masks, and tile that intersection sprite by its own pixel count.

    Input structure:
    - A black grid is split by one full-width gray horizontal line into equal-sized top and bottom panels.
    - The top panel contains red foreground pixels and the bottom panel contains blue foreground pixels.
    - In each panel, some foreground pixels form 8-connected clusters while some are isolated singleton noise pixels.

    Rule:
    - In each panel, keep only 8-connected foreground components of size at least 2 and erase singleton components.
    - Remove the separator row and align the two cleaned panel masks cell by cell.
    - Take the intersection of the two masks, crop it to its minimal bounding box, and recolor intersection pixels green on black.
    - Let N be the number of intersection pixels, then tile that cropped sprite horizontally exactly N times.

    Key insight:
    - The cleaned overlap serves two roles at once: it defines the sprite content and its pixel count sets how many copies the output contains.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = np.where(np.all(g == Color.GREY, axis=1))[0]
    assert len(sep_rows) == 1, "expected exactly one full-width gray separator row"
    sep = int(sep_rows[0])

    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape and top.size > 0, "panels must be non-empty and equally sized"

    top_mask = _denoise(top != Color.BLACK)
    bottom_mask = _denoise(bottom != Color.BLACK)
    inter = top_mask & bottom_mask

    box = bbox(inter)
    assert box is not None, "cleaned intersection must be non-empty"
    r0, c0, r1, c1 = box
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    sprite[inter[r0 : r1 + 1, c0 : c1 + 1]] = Color.GREEN

    copies = int(inter.sum())
    out = np.zeros((sprite.shape[0], sprite.shape[1] * copies), dtype=int)
    for i in range(copies):
        c = i * sprite.shape[1]
        out[:, c : c + sprite.shape[1]] = sprite
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        panel_h = random.randint(3, 5)
        panel_w = random.randint(12, 18)

        panels = _build_clean_panels(panel_h, panel_w)
        if panels is None:
            continue
        top_clean, bottom_clean = panels

        noise_count = random.randint(1, 3)
        top_mask = top_clean.copy()
        bottom_mask = bottom_clean.copy()
        if not _place_noise(top_mask, noise_count):
            continue
        if not _place_noise(bottom_mask, noise_count):
            continue

        if int(top_mask.sum()) != int(bottom_mask.sum()):
            continue

        top_denoised = _denoise(top_mask)
        bottom_denoised = _denoise(bottom_mask)
        if not np.array_equal(top_denoised, top_clean):
            continue
        if not np.array_equal(bottom_denoised, bottom_clean):
            continue

        inter = top_clean & bottom_clean
        inter_count = int(inter.sum())
        if not (2 <= inter_count <= 5):
            continue
        inter_box = bbox(inter)
        if inter_box is None:
            continue
        r0, c0, r1, c1 = inter_box
        if (r1 - r0 + 1) > 3 or (c1 - c0 + 1) > 3:
            continue

        grid = np.zeros((panel_h * 2 + 1, panel_w), dtype=int)
        grid[:panel_h][top_mask] = Color.RED
        grid[panel_h, :] = Color.GREY
        grid[panel_h + 1 :][bottom_mask] = Color.BLUE

        out = main(grid)
        if out.ndim != 2 or max(out.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")

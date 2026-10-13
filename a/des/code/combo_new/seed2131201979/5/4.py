from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 0
BORDER_COLOR = 1
SEED_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)

GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0) or np.any(rows == h - 1) or np.any(cols == 0) or np.any(cols == w - 1)
    )


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        if any(len(comp) > 1 for comp in comps):
            candidates.append((color, int(np.sum(grid == color))))
    assert candidates, "expected a border color"
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _symmetry_axes(coords: np.ndarray) -> tuple[bool, bool]:
    r0 = int(np.min(coords[:, 0]))
    r1 = int(np.max(coords[:, 0]))
    c0 = int(np.min(coords[:, 1]))
    c1 = int(np.max(coords[:, 1]))
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return bool(np.array_equal(local, np.flipud(local))), bool(np.array_equal(local, np.fliplr(local)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region along the seed row and/or column according to the region's reflection symmetry.

    Input structure:
    - The background is black.
    - One non-background color forms several disjoint one-pixel-wide closed borders.
    - Each enclosed border region contains exactly one seed pixel whose color is neither background nor border.
    - Each region interior has exactly one of three symmetry types: horizontal-axis only, vertical-axis only, or both axes.

    Rule:
    - Identify the unique border color and find every non-border connected component that is enclosed by a border.
    - For each enclosed component, treat its full interior shape as the symmetry object and detect whether it is symmetric under top-bottom flip, left-right flip, or both.
    - If the interior has only a horizontal axis, fill the seed's entire row inside that region with the seed color.
    - If the interior has only a vertical axis, fill the seed's entire column inside that region with the seed color.
    - If the interior has both axes, fill both the seed row and seed column inside that region.

    Key insight:
    - The seed color tells what to paint, but the region geometry tells where to paint: the reflection axis directions map directly to row and column expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    border = _border_color(g, bg)

    out = g.copy()
    non_border_components = _components(g != border)
    for coords in non_border_components:
        if _touches_boundary(coords, g.shape):
            continue

        values = g[coords[:, 0], coords[:, 1]]
        seed_mask = values != bg
        assert int(np.sum(seed_mask)) == 1, "each enclosed region must contain exactly one seed"
        seed_r, seed_c = map(int, coords[seed_mask][0])
        seed_color = int(g[seed_r, seed_c])

        horizontal, vertical = _symmetry_axes(coords)
        assert horizontal or vertical, "region must have at least one reflection axis"

        region_mask = np.zeros(g.shape, dtype=bool)
        region_mask[coords[:, 0], coords[:, 1]] = True
        if horizontal:
            out[seed_r, region_mask[seed_r]] = seed_color
        if vertical:
            out[region_mask[:, seed_c], seed_c] = seed_color

    return out


def _boundary_mask(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    boundary = np.zeros_like(shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not shape[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not shape[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _mirror_band(shape: np.ndarray, side: str, start: int, span: int, depth: int) -> None:
    h, w = shape.shape
    top = slice(start, start + span)
    bottom = slice(h - start - span, h - start)
    if side == "left":
        shape[top, :depth] = False
        shape[bottom, :depth] = False
    else:
        shape[top, w - depth :] = False
        shape[bottom, w - depth :] = False


def _mirror_strip(shape: np.ndarray, side: str, start: int, span: int, depth: int) -> None:
    h, w = shape.shape
    left = slice(start, start + span)
    right = slice(w - start - span, w - start)
    if side == "top":
        shape[:depth, left] = False
        shape[:depth, right] = False
    else:
        shape[h - depth :, left] = False
        shape[h - depth :, right] = False


def _seed_candidates(interior: np.ndarray, kind: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where(interior)
    for r, c in zip(rows.tolist(), cols.tolist()):
        row_len = int(np.sum(interior[r]))
        col_len = int(np.sum(interior[:, c]))
        if kind == "horizontal" and row_len >= 3:
            candidates.append((r, c))
        elif kind == "vertical" and col_len >= 3:
            candidates.append((r, c))
        elif kind == "both" and row_len >= 3 and col_len >= 3:
            candidates.append((r, c))
    return candidates


def _full_shape(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(6, 10)
        w = random.randint(6, 10)
        shape = np.ones((h, w), dtype=bool)

        if kind == "both":
            corner_h = random.randint(0, max(1, h // 3))
            corner_w = random.randint(0, max(1, w // 3))
            if corner_h > 0 and corner_w > 0:
                shape[:corner_h, :corner_w] = False
                shape[:corner_h, w - corner_w :] = False
                shape[h - corner_h :, :corner_w] = False
                shape[h - corner_h :, w - corner_w :] = False

            if random.random() < 0.8:
                notch_h = random.randint(1, max(1, h // 4))
                notch_w = random.randint(1, max(1, w // 3))
                c0 = (w - notch_w) // 2
                shape[:notch_h, c0 : c0 + notch_w] = False
                shape[h - notch_h :, c0 : c0 + notch_w] = False

            if random.random() < 0.8:
                notch_w = random.randint(1, max(1, w // 4))
                notch_h = random.randint(1, max(1, h // 3))
                r0 = (h - notch_h) // 2
                shape[r0 : r0 + notch_h, :notch_w] = False
                shape[r0 : r0 + notch_h, w - notch_w :] = False

        elif kind == "horizontal":
            side = random.choice(("left", "right"))
            for _ in range(random.randint(1, 2)):
                band_h = random.randint(1, max(1, h // 3))
                start = random.randint(0, max(0, h // 2 - band_h))
                depth = random.randint(1, max(1, min(w // 3, w - 4)))
                _mirror_band(shape, side, start, band_h, depth)

            if random.random() < 0.7:
                notch_h = random.randint(1, max(1, h // 4))
                notch_w = random.randint(1, max(1, min(w // 4, w - 4)))
                c0 = random.randint(1, max(1, w - notch_w - 1))
                shape[:notch_h, c0 : c0 + notch_w] = False
                shape[h - notch_h :, c0 : c0 + notch_w] = False

        else:
            side = random.choice(("top", "bottom"))
            for _ in range(random.randint(1, 2)):
                band_w = random.randint(1, max(1, w // 3))
                start = random.randint(0, max(0, w // 2 - band_w))
                depth = random.randint(1, max(1, min(h // 3, h - 4)))
                _mirror_strip(shape, side, start, band_w, depth)

            if random.random() < 0.7:
                notch_w = random.randint(1, max(1, w // 4))
                notch_h = random.randint(1, max(1, min(h // 4, h - 4)))
                r0 = random.randint(1, max(1, h - notch_h - 1))
                shape[r0 : r0 + notch_h, :notch_w] = False
                shape[r0 : r0 + notch_h, w - notch_w :] = False

        if not np.all(np.any(shape, axis=0)) or not np.all(np.any(shape, axis=1)):
            continue

        boundary = _boundary_mask(shape)
        interior = shape & ~boundary
        if not np.any(interior):
            continue
        if len(_components(interior)) != 1 or len(_components(boundary)) != 1:
            continue

        h_sym = bool(np.array_equal(interior, np.flipud(interior)))
        v_sym = bool(np.array_equal(interior, np.fliplr(interior)))
        if kind == "horizontal" and not (h_sym and not v_sym):
            continue
        if kind == "vertical" and not (v_sym and not h_sym):
            continue
        if kind == "both" and not (h_sym and v_sym):
            continue

        seeds = _seed_candidates(interior, kind)
        if not seeds:
            continue
        return shape, boundary, interior, random.choice(seeds)

    return None


def _scene_attempt() -> np.ndarray | None:
    kinds = ["horizontal", "vertical", "both"]
    object_count = random.randint(3, 5)
    kinds.extend(random.choice(("horizontal", "vertical", "both")) for _ in range(object_count - 3))
    random.shuffle(kinds)

    objects = []
    for kind in kinds:
        built = _full_shape(kind)
        if built is None:
            return None
        shape, boundary, interior, seed = built
        objects.append(
            {
                "kind": kind,
                "shape": shape,
                "boundary": boundary,
                "interior": interior,
                "seed": seed,
                "seed_color": random.choice(SEED_COLORS),
            }
        )

    grid_h = random.randint(20, 30)
    grid_w = random.randint(20, 30)
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    random.shuffle(objects)
    for obj in objects:
        h, w = obj["shape"].shape
        if h + 2 > grid_h or w + 2 > grid_w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(1, grid_h - h - 1)
            left = random.randint(1, grid_w - w - 1)
            r0 = max(0, top - 1)
            r1 = min(grid_h, top + h + 1)
            c0 = max(0, left - 1)
            c1 = min(grid_w, left + w + 1)
            if blocked[r0:r1, c0:c1].any():
                continue

            local = grid[top : top + h, left : left + w]
            local[obj["boundary"]] = BORDER_COLOR
            sr, sc = obj["seed"]
            local[sr, sc] = int(obj["seed_color"])
            blocked[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-fill input")

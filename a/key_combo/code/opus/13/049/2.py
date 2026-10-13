from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME = 4
GREY = 5

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frame_colors = sorted(set(map(int, grid.flatten())) - {BG, GREY})
    assert frame_colors == [FRAME], "expected grey patterns inside a single frame color"

    frames: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3, "frame must have a non-empty interior"

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == FRAME), "expected a solid one-pixel-thick frame border"
        assert np.all(patch[~border] != FRAME), "frame color must not appear in the interior"
        assert len(coords) == int(border.sum()), "frame component must be exactly the rectangle border"
        frames.append((r0, c0, r1, c1))

    assert 3 <= len(frames) <= 5, "expected 3-5 frames"
    return sorted(frames)


def _is_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.fliplr(mask)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior according to whether its grey pattern is mirror-symmetric about that frame's vertical midline.

    Input structure:
    - The grid has a black background with several disjoint one-pixel-thick rectangular frames of one shared color.
    - Each frame encloses a rectangular interior containing grey pixels and black empty cells.
    - Grey pixels stay at least one cell away from the frame border, so the frame geometry cleanly defines the local analysis region.

    Rule:
    - For each frame, inspect only its interior region and treat that interior's own vertical midline as the mirror axis.
    - If the grey mask inside the frame equals its left-right reflection, fill every black interior cell with blue.
    - Otherwise fill every black interior cell with red, while preserving the frame border and every grey pixel.

    Key insight:
    - The fill color is controlled by local symmetry of the grey pattern, and the relevant symmetry axis must be recomputed separately from each frame's geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _detect_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "frame interior must be non-empty"
        assert set(map(int, np.unique(interior))) <= {BG, GREY}, "interior may only contain black and grey"

        grey_mask = interior == GREY
        fill = BLUE if _is_symmetric(grey_mask) else RED
        fill_mask = interior == BG
        out[r0 + 1 : r1, c0 + 1 : c1][fill_mask] = fill
    return out


GEN_ATTEMPTS = 400
GEN_FRAME_COUNT_RANGE = (4, 5)
GEN_GRID_H_RANGE = (16, 23)
GEN_GRID_W_RANGE = (20, 27)
GEN_INTERIOR_SIZES = ((5, 6), (6, 5))
GEN_MIN_GAP = 1
GEN_MIN_MARGIN = 1
GEN_COUNTS = (10, 11)


def _all_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w)]


def _symmetric_pattern(core_h: int, core_w: int, count: int, rng: random.Random) -> np.ndarray | None:
    orbits: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in _all_positions(core_h, core_w):
        if (r, c) in seen:
            continue
        mate = (r, core_w - 1 - c)
        orbit = sorted({(r, c), mate})
        for cell in orbit:
            seen.add(cell)
        orbits.append(orbit)

    rng.shuffle(orbits)
    chosen: list[list[tuple[int, int]]] = []
    remain = int(count)
    for orbit in sorted(orbits, key=len):
        if len(orbit) <= remain and rng.random() < 0.7:
            chosen.append(orbit)
            remain -= len(orbit)
    if remain:
        for orbit in sorted(orbits, key=len, reverse=True):
            if orbit in chosen or len(orbit) > remain:
                continue
            chosen.append(orbit)
            remain -= len(orbit)
            if remain == 0:
                break
    if remain != 0:
        return None

    mask = np.zeros((core_h, core_w), dtype=bool)
    for orbit in chosen:
        for r, c in orbit:
            mask[r, c] = True
    if int(mask.sum()) != int(count) or not _is_symmetric(mask):
        return None
    return mask


def _asymmetric_pattern(core_h: int, core_w: int, count: int, rng: random.Random) -> np.ndarray | None:
    cells = _all_positions(core_h, core_w)
    for _ in range(80):
        rng.shuffle(cells)
        mask = np.zeros((core_h, core_w), dtype=bool)
        for r, c in cells[:count]:
            mask[r, c] = True
        if not _is_symmetric(mask):
            return mask
    return None


def _build_pattern(interior_h: int, interior_w: int, symmetric: bool, count: int, rng: random.Random) -> np.ndarray | None:
    core_h = interior_h - 2
    core_w = interior_w - 2
    maker = _symmetric_pattern if symmetric else _asymmetric_pattern
    core = maker(core_h, core_w, count, rng)
    if core is None:
        return None
    pattern = np.zeros((interior_h, interior_w), dtype=int)
    pattern[1:-1, 1:-1][core] = GREY
    return pattern


def _frame_patch(interior_h: int, interior_w: int, pattern: np.ndarray) -> np.ndarray:
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    patch[1:-1, 1:-1] = pattern
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    if top < GEN_MIN_MARGIN or left < GEN_MIN_MARGIN:
        return False
    if top + h > h0 - GEN_MIN_MARGIN or left + w > w0 - GEN_MIN_MARGIN:
        return False

    r0 = max(0, top - GEN_MIN_GAP)
    c0 = max(0, left - GEN_MIN_GAP)
    r1 = min(h0, top + h + GEN_MIN_GAP)
    c1 = min(w0, left + w + GEN_MIN_GAP)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, rng: random.Random) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = [
        (r, c)
        for r in range(grid.shape[0] - h + 1)
        for c in range(grid.shape[1] - w + 1)
        if _can_place(occupied, r, c, h, w)
    ]
    if not candidates:
        return None
    top, left = rng.choice(candidates)
    grid[top : top + h, left : left + w] = np.maximum(grid[top : top + h, left : left + w], patch)
    occupied[top : top + h, left : left + w] |= patch != BG
    return int(top), int(left)


def _frame_specs(rng: random.Random, frame_count: int) -> list[dict]:
    specs = [
        {"interior": (5, 6), "symmetric": True, "count": 10},
        {"interior": (5, 6), "symmetric": False, "count": 10},
        {"interior": (6, 5), "symmetric": True, "count": 11},
        {"interior": (6, 5), "symmetric": False, "count": 11},
    ]
    while len(specs) < frame_count:
        interior = rng.choice(GEN_INTERIOR_SIZES)
        symmetric = bool(rng.randrange(2))
        count = rng.choice(GEN_COUNTS)
        if symmetric and interior[1] == 6 and count % 2 == 1:
            count = 10
        specs.append({"interior": interior, "symmetric": symmetric, "count": count})
    rng.shuffle(specs)
    return specs


def _oracle_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        fill = BLUE if spec["symmetric"] else RED
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill
    return out


def _validate_grid(grid: np.ndarray, specs: list[dict]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    frames = _detect_frames(grid)
    if len(frames) != len(specs):
        return False

    specs_sorted = sorted(specs, key=lambda s: s["bbox"])
    counts = []
    labels = []
    sizes = []
    for bbox, spec in zip(frames, specs_sorted):
        if bbox != spec["bbox"]:
            return False
        r0, c0, r1, c1 = bbox
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        grey_mask = interior == GREY
        if not grey_mask.any():
            return False
        if np.any(interior[[0, -1], :] == GREY) or np.any(interior[:, [0, -1]] == GREY):
            return False
        if set(map(int, np.unique(interior))) != {BG, GREY}:
            return False
        if _is_symmetric(grey_mask) != bool(spec["symmetric"]):
            return False
        count = int(grey_mask.sum())
        counts.append(count)
        labels.append(bool(spec["symmetric"]))
        sizes.append(interior.shape)
        area = int(interior.size)
        density = count / area
        if not (0.30 <= density <= 0.60):
            return False
        if int((interior == BG).sum()) == 0:
            return False

    if max(counts) - min(counts) > 2:
        return False
    if not any(labels) or all(labels):
        return False
    if len(set(sizes)) < 2:
        return False
    if not any(counts[i] == counts[j] and labels[i] != labels[j] for i in range(len(counts)) for j in range(i + 1, len(counts))):
        return False
    if not any(sizes[i] == sizes[j] and labels[i] != labels[j] for i in range(len(sizes)) for j in range(i + 1, len(sizes))):
        return False
    if {shape[1] for shape in sizes} != {5, 6}:
        return False

    expected = _oracle_from_specs(grid, specs_sorted)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, grid):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()
    frame_count = rng.randint(*GEN_FRAME_COUNT_RANGE)
    specs = _frame_specs(rng, frame_count)

    grid_h = rng.randint(*GEN_GRID_H_RANGE)
    grid_w = rng.randint(*GEN_GRID_W_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    placed: list[dict] = []
    for spec in sorted(specs, key=lambda s: (s["interior"][0] + 2) * (s["interior"][1] + 2), reverse=True):
        ih, iw = spec["interior"]
        pattern = _build_pattern(ih, iw, bool(spec["symmetric"]), int(spec["count"]), rng)
        if pattern is None:
            return None
        patch = _frame_patch(ih, iw, pattern)
        loc = _place_patch(grid, occupied, patch, rng)
        if loc is None:
            return None
        top, left = loc
        placed.append(
            {
                "interior": spec["interior"],
                "symmetric": bool(spec["symmetric"]),
                "count": int(spec["count"]),
                "bbox": (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1),
            }
        )

    if not _validate_grid(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")

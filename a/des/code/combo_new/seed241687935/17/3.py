from __future__ import annotations

import random

import numpy as np


BG = 0
OBJECT_COLOR = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
THICKNESS = 3
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _local_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _opening_direction(mask: np.ndarray) -> str:
    closed = {
        "up": bool(mask[0, :].all()),
        "down": bool(mask[-1, :].all()),
        "left": bool(mask[:, 0].all()),
        "right": bool(mask[:, -1].all()),
    }
    openings = [direction for direction, is_closed in closed.items() if not is_closed]
    assert len(openings) == 1, f"expected one opening side, got {openings}"
    return openings[0]


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _edge_sources(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    if direction == "up":
        return [(0, c) for c in range(mask.shape[1]) if mask[0, c]]
    if direction == "down":
        return [(mask.shape[0] - 1, c) for c in range(mask.shape[1]) if mask[-1, c]]
    if direction == "left":
        return [(r, 0) for r in range(mask.shape[0]) if mask[r, 0]]
    assert direction == "right"
    return [(r, mask.shape[1] - 1) for r in range(mask.shape[0]) if mask[r, -1]]


def _step(direction: str) -> tuple[int, int]:
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each hollow gray U/C object by its hole count and spray rays out of its opening.

    Input structure:
    - Black is the background.
    - The bottom row is a color legend made of isolated non-black pixels with one
      black cell between consecutive legend entries; the first legend color means
      0 holes, the second means 1 hole, and so on.
    - Above the legend are several 4-connected gray objects shaped like thick U
      or C forms: a rectangular frame with exactly one side opened.
    - Each gray object may contain several enclosed black single-cell holes inside
      its gray material.

    Rule:
    - For each gray object, count its enclosed black hole regions inside the
      object's bounding box.
    - Use that count N to select the (N+1)-th color from the bottom legend.
    - Find the object's unique opening direction by locating the only bounding-box
      side that is not completely gray.
    - From every gray pixel lying on that opening side, draw a ray of length N+1
      outward in the opening direction, stopping early at the legend row, at the
      grid boundary, or at any non-black pixel.

    Key insight:
    - The same hole count controls both a lookup-table choice in the bottom legend
      and the geometric ray length, while the missing bbox side identifies where
      the rays must be emitted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 3, "grid is too small"

    legend = [int(value) for value in g[-1, :].tolist() if int(value) != BG]
    assert legend, "expected a bottom legend row"

    scene = g[:-1, :]
    object_colors = [int(value) for value in np.unique(scene) if int(value) != BG]
    assert len(object_colors) == 1, "expected one object color above the legend"
    object_color = object_colors[0]

    out = g.copy()
    for coords in _components(scene == object_color):
        mask, (r0, c0, _r1, _c1) = _local_mask(coords)
        direction = _opening_direction(mask)
        hole_count = _count_holes(mask)
        assert hole_count < len(legend), "hole count exceeds legend length"
        ray_color = legend[hole_count]
        ray_len = hole_count + 1
        dr, dc = _step(direction)

        for sr, sc in _edge_sources(mask, direction):
            base_r = r0 + sr
            base_c = c0 + sc
            for step in range(1, ray_len + 1):
                nr = base_r + dr * step
                nc = base_c + dc * step
                if not (0 <= nr < height and 0 <= nc < width):
                    break
                if nr == height - 1:
                    break
                if out[nr, nc] != BG:
                    break
                out[nr, nc] = ray_color

    return out


def _build_shell(direction: str, outer_h: int, outer_w: int) -> np.ndarray:
    patch = np.full((outer_h, outer_w), OBJECT_COLOR, dtype=int)
    t = THICKNESS
    patch[t : outer_h - t, t : outer_w - t] = BG
    if direction == "up":
        patch[:t, t : outer_w - t] = BG
    elif direction == "down":
        patch[outer_h - t :, t : outer_w - t] = BG
    elif direction == "left":
        patch[t : outer_h - t, :t] = BG
    else:
        assert direction == "right"
        patch[t : outer_h - t, outer_w - t :] = BG
    return patch


def _choose_holes(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for cell in shuffled:
        if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None if count > 0 else []


def _build_object(direction: str, hole_count: int) -> np.ndarray:
    for _ in range(200):
        lo = 7 + int(hole_count >= 3)
        outer_h = random.randint(lo, 10)
        outer_w = random.randint(lo, 10)
        if direction in ("up", "down") and random.random() < 0.5:
            outer_w = random.randint(max(outer_w, lo + 1), 10)
        if direction in ("left", "right") and random.random() < 0.5:
            outer_h = random.randint(max(outer_h, lo + 1), 10)

        patch = _build_shell(direction, outer_h, outer_w)
        candidates: list[tuple[int, int]] = []
        for r in range(1, outer_h - 1):
            for c in range(1, outer_w - 1):
                if patch[r, c] != OBJECT_COLOR:
                    continue
                if (
                    patch[r - 1, c] == OBJECT_COLOR
                    and patch[r + 1, c] == OBJECT_COLOR
                    and patch[r, c - 1] == OBJECT_COLOR
                    and patch[r, c + 1] == OBJECT_COLOR
                ):
                    candidates.append((r, c))

        chosen = _choose_holes(candidates, hole_count)
        if chosen is None:
            continue
        for r, c in chosen:
            patch[r, c] = BG

        mask = patch == OBJECT_COLOR
        if len(_components(mask)) != 1:
            continue
        if _opening_direction(mask) != direction:
            continue
        if _count_holes(mask) != hole_count:
            continue
        if not _edge_sources(mask, direction):
            continue
        return patch
    raise AssertionError("failed to construct a valid object")


def _sample_hole_counts(n_objects: int, legend_len: int) -> list[int]:
    counts = list(range(legend_len))
    random.shuffle(counts)
    return counts[:n_objects]


def _boundary_distance(
    top: int,
    left: int,
    patch: np.ndarray,
    direction: str,
    scene_h: int,
    width: int,
) -> int:
    ph, pw = patch.shape
    if direction == "up":
        return top
    if direction == "down":
        return scene_h - 1 - (top + ph - 1)
    if direction == "left":
        return left
    assert direction == "right"
    return width - 1 - (left + pw - 1)


def _bbox_cells(top: int, left: int, patch: np.ndarray) -> set[tuple[int, int]]:
    ph, pw = patch.shape
    return {(top + r, left + c) for r in range(ph) for c in range(pw)}


def _ray_cells(
    top: int,
    left: int,
    patch: np.ndarray,
    direction: str,
    ray_len: int,
    scene_h: int,
    width: int,
) -> set[tuple[int, int]]:
    mask = patch == OBJECT_COLOR
    distance = _boundary_distance(top, left, patch, direction, scene_h, width)
    actual_len = min(ray_len, distance)
    if actual_len <= 0:
        return set()

    dr, dc = _step(direction)
    cells: set[tuple[int, int]] = set()
    for sr, sc in _edge_sources(mask, direction):
        base_r = top + sr
        base_c = left + sc
        for step in range(1, actual_len + 1):
            nr = base_r + dr * step
            nc = base_c + dc * step
            if 0 <= nr < scene_h and 0 <= nc < width:
                cells.add((nr, nc))
    return cells


def _mark_buffer(buffer: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = buffer.shape
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    buffer[nr, nc] = True


def _place_object(
    occupied: np.ndarray,
    buffer: np.ndarray,
    patch: np.ndarray,
    direction: str,
    ray_len: int,
    require_truncation: bool,
) -> tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]] | None:
    scene_h, width = occupied.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(scene_h - ph + 1) for c in range(width - pw + 1)]
    random.shuffle(positions)

    for top, left in positions:
        distance = _boundary_distance(top, left, patch, direction, scene_h, width)
        if require_truncation:
            if not (1 <= distance < ray_len):
                continue
        else:
            if distance < ray_len:
                continue

        bbox = _bbox_cells(top, left, patch)
        rays = _ray_cells(top, left, patch, direction, ray_len, scene_h, width)
        if not rays:
            continue
        if any(occupied[r, c] or buffer[r, c] for r, c in bbox):
            continue
        if any(occupied[r, c] or buffer[r, c] for r, c in rays):
            continue
        return top, left, bbox, rays

    return None


def _draw_legend(grid: np.ndarray, legend: list[int]) -> None:
    width = grid.shape[1]
    span = 2 * len(legend) - 1
    start = random.randint(0, width - span)
    row = grid.shape[0] - 1
    for idx, color in enumerate(legend):
        grid[row, start + 2 * idx] = int(color)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    legend_len = random.randint(max(3, n_objects), 5)
    legend = random.sample(PALETTE_COLORS, legend_len)
    hole_counts = _sample_hole_counts(n_objects, legend_len)
    directions = random.sample(DIRS, n_objects)

    min_side = max(18, 12 + 4 * n_objects, 2 * legend_len + 1)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    grid = np.zeros((height, width), dtype=int)
    scene_h = height - 1
    occupied = np.zeros((scene_h, width), dtype=bool)
    buffer = np.zeros((scene_h, width), dtype=bool)

    specs = list(zip(directions, hole_counts))
    random.shuffle(specs)
    for direction, hole_count in specs:
        patch = _build_object(direction, hole_count)
        ray_len = hole_count + 1
        prefer_truncation = random.random() < 0.35
        placement = None
        for require_truncation in (prefer_truncation, not prefer_truncation):
            placement = _place_object(occupied, buffer, patch, direction, ray_len, require_truncation)
            if placement is not None:
                break
        if placement is None:
            return None

        top, left, bbox, rays = placement
        window = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        window[patch == OBJECT_COLOR] = OBJECT_COLOR

        reserved = bbox | rays
        for r, c in reserved:
            occupied[r, c] = True
        _mark_buffer(buffer, reserved)

    _draw_legend(grid, legend)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[-1, :], grid[-1, :]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid input")

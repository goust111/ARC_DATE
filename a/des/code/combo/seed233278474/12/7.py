from __future__ import annotations

import random
from collections import Counter

import numpy as np


YELLOW = 4
BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_same_color(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _bool_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _touches_bbox_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords)


def _find_hole(component: dict, grid: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    color = int(component["color"])
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    obj_mask = patch == color

    holes = [
        comp
        for comp in _bool_components(~obj_mask)
        if not _touches_bbox_edge(comp, patch.shape[0], patch.shape[1])
    ]
    assert len(holes) == 1, "expected exactly one enclosed background component"
    assert len(holes[0]) == 1, "expected the hole to be a single pixel"
    hr, hc = holes[0][0]
    return r0 + int(hr), c0 + int(hc)


def _direction_from_hole(bbox: tuple[int, int, int, int], hole: tuple[int, int]) -> str:
    r0, c0, r1, c1 = map(int, bbox)
    hr, hc = map(int, hole)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = hr - center_r
    dc = hc - center_c

    assert dr != 0 or dc != 0, "hole cannot sit exactly at the bbox center"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    if dc == 0:
        return "up" if dr < 0 else "down"
    if dr == 0:
        return "left" if dc < 0 else "right"
    return "up" if dr < 0 else "down"


def _mirror_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        return np.fliplr(patch)
    return np.flipud(patch)


def _mirror_origin(r0: int, c0: int, h: int, w: int, direction: str) -> tuple[int, int]:
    if direction == "left":
        return r0, c0 - w
    if direction == "right":
        return r0, c0 + w
    if direction == "up":
        return r0 - h, c0
    return r0 + h, c0


def _paint_mirror(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= out.shape[0] and c0 + w <= out.shape[1]
    target = out[r0 : r0 + h, c0 : c0 + w]
    fg = patch != bg
    assert not np.any(target[fg] != bg), "mirror copy must land on empty background"
    target[fg] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each object toward the side encoded by its internal hole.

    Input structure:
    - The background is black.
    - There are 2-4 non-overlapping monochromatic 4-connected objects.
    - Each object has exactly one enclosed 1-pixel background hole.

    Rule:
    - Find the unique hole inside each object.
    - Compare the hole position with the center of the object's bounding box.
    - Use the dominant signed offset axis to choose up/down/left/right.
    - Mirror the full object along that axis and place the yellow copy flush
      against the original on the chosen side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _majority_color(g)
    components = _components_same_color(g, bg)
    assert components, "expected foreground objects"

    out = g.copy()
    for component in components:
        r0, c0, r1, c1 = map(int, component["bbox"])
        hole = _find_hole(component, g)
        direction = _direction_from_hole(component["bbox"], hole)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mirrored = _mirror_patch(patch, direction)
        mr0, mc0 = _mirror_origin(r0, c0, patch.shape[0], patch.shape[1], direction)
        _paint_mirror(out, mirrored, mr0, mc0, bg)

    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (18, 24)
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
DIRECTIONS = ["up", "down", "left", "right"]
SIZE_RANGE = range(4, 7)
PADDING = 1
MAX_EXTRA_REMOVALS = 4
PLACEMENT_TRIES = 200
GENERATION_RETRIES = 400


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = _bool_components(mask)
    return len(comps) == 1


def _hole_stays_enclosed(mask: np.ndarray, hole: tuple[int, int]) -> bool:
    hr, hc = hole
    if bool(mask[hr, hc]):
        return False
    return all(bool(mask[hr + dr, hc + dc]) for dr, dc in DIR4)


def _hole_candidates(h: int, w: int, direction: str) -> list[tuple[int, int]]:
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    candidates: list[tuple[int, int]] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            dr = r - center_r
            dc = c - center_c
            if direction == "up" and dr < 0 and abs(dr) > abs(dc):
                candidates.append((r, c))
            elif direction == "down" and dr > 0 and abs(dr) > abs(dc):
                candidates.append((r, c))
            elif direction == "left" and dc < 0 and abs(dc) > abs(dr):
                candidates.append((r, c))
            elif direction == "right" and dc > 0 and abs(dc) > abs(dr):
                candidates.append((r, c))

    return candidates


def _sample_dimensions(direction: str) -> tuple[int, int]:
    options = [
        (h, w)
        for h in SIZE_RANGE
        for w in SIZE_RANGE
        if _hole_candidates(h, w, direction)
    ]
    return random.choice(options)


def _sample_object_mask(direction: str) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = _sample_dimensions(direction)
    hole = random.choice(_hole_candidates(h, w, direction))
    mask = np.ones((h, w), dtype=bool)
    mask[hole] = False

    protected = {hole}
    hr, hc = hole
    for dr, dc in DIR4:
        protected.add((hr + dr, hc + dc))

    border_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if mask[r, c] and (r == 0 or r == h - 1 or c == 0 or c == w - 1) and (r, c) not in protected
    ]
    random.shuffle(border_cells)
    target = random.randint(0, min(MAX_EXTRA_REMOVALS, len(border_cells)))

    removed = 0
    for r, c in border_cells:
        if removed >= target:
            break
        trial = mask.copy()
        trial[r, c] = False
        if not _full_bbox(trial):
            continue
        if not _is_connected(trial):
            continue
        if not _hole_stays_enclosed(trial, hole):
            continue
        mask = trial
        removed += 1

    return mask, hole


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _union_rect(r0: int, c0: int, h: int, w: int, direction: str) -> tuple[int, int, int, int]:
    mr0, mc0 = _mirror_origin(r0, c0, h, w, direction)
    return min(r0, mr0), min(c0, mc0), max(r0 + h - 1, mr0 + h - 1), max(c0 + w - 1, mc0 + w - 1)


def _can_reserve(occupied: np.ndarray, rect: tuple[int, int, int, int], padding: int) -> bool:
    r0, c0, r1, c1 = rect
    pr0 = max(0, r0 - padding)
    pc0 = max(0, c0 - padding)
    pr1 = min(occupied.shape[0] - 1, r1 + padding)
    pc1 = min(occupied.shape[1] - 1, c1 + padding)
    return not bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _reserve(occupied: np.ndarray, rect: tuple[int, int, int, int], padding: int) -> None:
    r0, c0, r1, c1 = rect
    pr0 = max(0, r0 - padding)
    pc0 = max(0, c0 - padding)
    pr1 = min(occupied.shape[0] - 1, r1 + padding)
    pc1 = min(occupied.shape[1] - 1, c1 + padding)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _origin_ranges(h: int, w: int, H: int, W: int, direction: str) -> tuple[range, range] | None:
    if direction == "left":
        r_range = range(0, H - h + 1)
        c_range = range(w, W - w + 1)
    elif direction == "right":
        r_range = range(0, H - h + 1)
        c_range = range(0, W - 2 * w + 1)
    elif direction == "up":
        r_range = range(h, H - h + 1)
        c_range = range(0, W - w + 1)
    else:
        r_range = range(0, H - 2 * h + 1)
        c_range = range(0, W - w + 1)
    if not r_range or not c_range:
        return None
    return r_range, c_range


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    rows, cols = np.nonzero(patch != BG)
    grid[r0 + rows, c0 + cols] = patch[rows, cols]


def _expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        patch = spec["patch"]
        direction = spec["direction"]
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        mirrored = _mirror_patch(patch, direction)
        mr0, mc0 = _mirror_origin(r0, c0, patch.shape[0], patch.shape[1], direction)
        fg = mirrored != BG
        target = out[mr0 : mr0 + patch.shape[0], mc0 : mc0 + patch.shape[1]]
        if np.any(target[fg] != BG):
            raise AssertionError("expected mirror overlap")
        target[fg] = YELLOW
    return out


def _sample_input() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    base_side = max(GRID_SIDE_RANGE[0], 14 + 2 * n_objects)
    H = random.randint(base_side, GRID_SIDE_RANGE[1])
    W = random.randint(base_side, GRID_SIDE_RANGE[1])
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    colors = random.sample(OBJECT_COLORS, n_objects)
    directions = random.sample(DIRECTIONS, n_objects)
    specs: list[dict] = []

    for color, direction in zip(colors, directions):
        mask, _hole = _sample_object_mask(direction)
        patch = _patch_from_mask(mask, color)
        h, w = patch.shape
        ranges = _origin_ranges(h, w, H, W, direction)
        if ranges is None:
            return None
        r_range, c_range = ranges

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.choice(r_range)
            c0 = random.choice(c_range)
            rect = _union_rect(r0, c0, h, w, direction)
            if not _can_reserve(occupied, rect, PADDING):
                continue
            _paint_patch(grid, patch, r0, c0)
            _reserve(occupied, rect, PADDING)
            specs.append({"patch": patch, "direction": direction, "r0": r0, "c0": c0})
            placed = True
            break
        if not placed:
            return None

    expected = _expected_from_specs(grid, specs)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    components = _components_same_color(grid, BG)
    if len(components) != n_objects:
        return None
    if any(int(comp["color"]) == YELLOW for comp in components):
        return None
    if sum(int(value) == BG for value in grid.ravel()) <= grid.size // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _sample_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

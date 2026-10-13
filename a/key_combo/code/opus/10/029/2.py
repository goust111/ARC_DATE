from __future__ import annotations

import random
from math import isqrt

import numpy as np


BG = 0
FG = 1
MARKER = 2
MAX_GENERATION_ATTEMPTS = 500
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BG) -> None:
    h, w = patch.shape
    view = patch != transparent
    dst[r0 : r0 + h, c0 : c0 + w][view] = patch[view]


def _stamp_self_similar(mask: np.ndarray) -> np.ndarray:
    k = int(mask.shape[0])
    out = np.zeros((k * k, k * k), dtype=int)
    stamp = np.where(mask, FG, BG).astype(int)
    for r, c in np.argwhere(mask):
        _paste(out, stamp, int(r) * k, int(c) * k)
    return out


def _count_and_rotate(template: np.ndarray) -> tuple[int, np.ndarray]:
    marker_count = int(np.count_nonzero(template == MARKER))
    assert 1 <= marker_count <= 3, "expected 1-3 marker pixels"
    pure = np.where(template != BG, FG, BG).astype(int)
    rotated = np.rot90(pure, -marker_count)
    return marker_count, rotated


def _extract_template(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = int(g.shape[0])
    k = isqrt(n)
    assert k * k == n, "expected side length N = k^2"

    blocks = []
    for br in range(k):
        for bc in range(k):
            block = g[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
            if np.any(block != BG):
                blocks.append((br, bc))

    assert len(blocks) == 1, "expected exactly one non-empty k x k block"
    br, bc = blocks[0]
    return g[br * k : (br + 1) * k, bc * k : (bc + 1) * k].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """Count marker pixels to choose a rotation, absorb them into the shape, then self-stamp the rotated template.

    Input structure:
    - The grid is square with side length N = k^2 and black background.
    - Exactly one k x k block contains the template; all other k x k blocks are fully black.
    - Inside that template block, blue foreground cells and red marker cells form one 4-connected small shape.

    Rule:
    - Extract the unique non-empty k x k template block.
    - Count its red marker pixels C, where C is 1, 2, or 3.
    - Recolor every red marker to blue, rotate the resulting binary template clockwise by C quarter-turns,
      then treat that rotated template as a k x k block-layout mask.
    - For every foreground cell of the rotated template at local position (r, c), paste the whole rotated template
      into output block (r, c); leave all other blocks black.

    Key insight:
    - The red cells have two roles at once: their count selects the rotation angle, and those same cells become part
      of the foreground shape before stamping, so the correct template exists only after counting and recoloring.
    """
    template = _extract_template(grid)
    assert set(np.unique(template)).issubset({BG, FG, MARKER}), "unexpected colors"
    mask = template != BG
    assert 3 <= int(mask.sum()) <= 6, "expected a 3-6 cell shape"
    assert _is_connected(mask), "foreground must be 4-connected"

    _count, rotated = _count_and_rotate(template)
    return _stamp_self_similar(rotated == FG)


def _random_connected_mask(k: int, area: int) -> np.ndarray:
    mask = np.zeros((k, k), dtype=bool)
    start = (random.randrange(k), random.randrange(k))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < k and 0 <= nc < k and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return mask
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _pick_marker_coords(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    center_r = sum(r for r, _ in coords) / len(coords)
    center_c = sum(c for _, c in coords) / len(coords)
    ranked = sorted(
        coords,
        key=lambda rc: (
            -abs(rc[0] - center_r) - abs(rc[1] - center_c),
            rc[0],
            rc[1],
        ),
    )
    return ranked[:count]


def _build_template(mask: np.ndarray, marker_count: int) -> np.ndarray:
    template = np.where(mask, FG, BG).astype(int)
    for r, c in _pick_marker_coords(mask, marker_count):
        template[r, c] = MARKER
    return template


def _wrong_ignore_markers(template: np.ndarray) -> np.ndarray:
    marker_count = int(np.count_nonzero(template == MARKER))
    blue_only = np.where(template == FG, FG, BG).astype(int)
    rotated = np.rot90(blue_only, -marker_count)
    return _stamp_self_similar(rotated == FG)


def _candidate_outputs(template: np.ndarray) -> dict[str, np.ndarray]:
    marker_count, rotated = _count_and_rotate(template)
    pure = np.where(template != BG, FG, BG).astype(int)
    outputs = {
        "correct": _stamp_self_similar(rotated == FG),
        "no_rotation": _stamp_self_similar(pure == FG),
        "ignore_markers": _wrong_ignore_markers(template),
        **{
            f"alt_count_{alt}": _stamp_self_similar(np.rot90(pure, -alt) == FG)
            for alt in (1, 2, 3)
            if alt != marker_count
        },
    }
    if marker_count != 2:
        outputs["ccw_rotation"] = _stamp_self_similar(np.rot90(pure, marker_count) == FG)
    return outputs


def _valid_mask(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    if not (3 <= area <= 6):
        return False
    if not _is_connected(mask):
        return False
    rows = np.any(mask, axis=1).sum()
    cols = np.any(mask, axis=0).sum()
    if rows < 2 or cols < 2:
        return False
    return True


def _semantic_checks(template: np.ndarray) -> bool:
    outputs = _candidate_outputs(template)
    correct = outputs["correct"]
    if int(np.count_nonzero(correct)) <= int(np.count_nonzero(template != BG)):
        return False
    for name, candidate in outputs.items():
        if name == "correct":
            continue
        if np.array_equal(candidate, correct):
            return False
    return True


def _make_input_grid(template: np.ndarray) -> np.ndarray:
    k = int(template.shape[0])
    grid = np.zeros((k * k, k * k), dtype=int)
    marker_cells = [tuple(map(int, rc)) for rc in np.argwhere(template == MARKER)]
    assert marker_cells, "expected at least one marker cell"
    br, bc = min(marker_cells)
    _paste(grid, template, br * k, bc * k)
    return grid


def _generate_attempt() -> np.ndarray | None:
    k = random.choice([3, 4, 5])
    marker_count = random.choice([1, 2, 3])
    min_area = max(3, marker_count + 1, 4 if k >= 4 else 3)
    area = random.randint(min_area, 6)

    mask = _random_connected_mask(k, area)
    if not _valid_mask(mask):
        return None

    template = _build_template(mask, marker_count)
    if not _semantic_checks(template):
        return None

    inp = _make_input_grid(template)
    try:
        out = main(inp)
    except Exception:
        return None
    if not np.array_equal(out, _candidate_outputs(template)["correct"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

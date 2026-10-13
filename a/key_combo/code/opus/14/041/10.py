from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
EDGE_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _square_ring_cells(r: int, c: int) -> set[tuple[int, int]]:
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    }


def _find_square_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            if all(int(g[r + dr, c + dc]) == GREEN for dr, dc in CORNER_OFFSETS + EDGE_OFFSETS):
                centers.append((r, c))
    return centers


def _apply_square_rewrite(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    centers = _find_square_centers(g)
    meta: list[dict] = []

    for r, c in centers:
        square_ring = _square_ring_cells(r, c)
        kept = 0
        erased = 0
        for dr, dc in CORNER_OFFSETS:
            rr, cc = r + dr, c + dc
            keep_corner = False
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREEN and (nr, nc) not in square_ring:
                    keep_corner = True
                    break
            if keep_corner:
                out[rr, cc] = GREEN
                kept += 1
            else:
                out[rr, cc] = BLACK
                erased += 1

        for dr, dc in EDGE_OFFSETS:
            out[r + dr, c + dc] = GREEN
        out[r, c] = RED
        meta.append({"center": (r, c), "kept": kept, "erased": erased})

    return out, meta


def _outside_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(g[r, c]) == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(g[r, c]) == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLACK and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _enclosed_black_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    outside = _outside_black_mask(g)
    return (g == BLACK) & (~outside)


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each green hollow square with a cross mask, then fill only the background regions still enclosed afterward.

    Input structure:
    - The grid is black background with a green orthogonal network of 1-pixel-wide line segments.
    - Embedded in that network are several 3x3 hollow green squares: black center, eight surrounding green pixels.
    - Some background regions are closed by the original network, while others already reach the grid border.

    Rule:
    - Detect every 3x3 hollow green square.
    - For each one, repaint the center red, keep the four edge-neighbor pixels green, and erase each corner to black unless that corner also continues into an external green line outside the square.
    - After all squares are rewritten simultaneously, run a 4-connected flood fill from black border cells to mark the exterior.
    - Recolor every remaining black cell that is still enclosed under the new topology to yellow.

    Key insight:
    - The answer depends on the post-rewrite global 4-connectivity of the background, because deleting a single square corner can open a leak that destroys a region far larger than the local 3x3 box.
    """
    g = np.asarray(grid, dtype=int)
    rewritten, _meta = _apply_square_rewrite(g)
    enclosed = _enclosed_black_mask(rewritten)
    out = rewritten.copy()
    out[enclosed] = YELLOW
    return out


def _patch_from_ascii(rows: list[str]) -> np.ndarray:
    return np.array([[GREEN if ch == "#" else BLACK for ch in row] for row in rows], dtype=int)


LEAKY_TEMPLATES = [
    _patch_from_ascii(
        [
            ".########",
            "#.....#..",
            "#.....#..",
            "#..###.##",
            "#..#.#...",
            "#..###...",
            "######...",
        ]
    ),
    _patch_from_ascii(
        [
            "#....#.#.",
            "#....#.#.",
            "#....#.#.",
            "#..#####.",
            "#..#.#.#.",
            "...###.#.",
            "......###",
        ]
    ),
    _patch_from_ascii(
        [
            "......#.",
            "......#.",
            "..###.#.",
            "..#.#.#.",
            "..###.#.",
            "####.##.",
            "..######",
        ]
    ),
]


def _random_patch_transform(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    out = np.rot90(out, k)
    return out.copy()


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    pts = np.argwhere(patch != BLACK)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_stable_room() -> np.ndarray:
    inner_h = random.randint(3, 6)
    inner_w = random.randint(4, 7)
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = GREEN
    patch[-1, :] = GREEN
    patch[:, 0] = GREEN
    patch[:, -1] = GREEN
    return patch


def _make_neutral_box(force_preserved_corner: bool) -> np.ndarray:
    size = random.randint(7, 10)
    patch = np.zeros((size, size), dtype=int)
    r = size // 2
    c = size // 2

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) != (0, 0):
                patch[r + dr, c + dc] = GREEN

    side_anchors = {
        "N": (r - 1, c, -1, 0),
        "S": (r + 1, c, 1, 0),
        "W": (r, c - 1, 0, -1),
        "E": (r, c + 1, 0, 1),
    }
    side_count = random.randint(1, 4)
    for side in random.sample(list(side_anchors), side_count):
        rr, cc, dr, dc = side_anchors[side]
        for step in range(1, random.randint(2, 4)):
            nr = rr + dr * step
            nc = cc + dc * step
            if in_bounds(patch.shape, nr, nc):
                patch[nr, nc] = GREEN

    corner_anchors = {
        "NW": (r - 1, c - 1, [(-1, 0), (0, -1)]),
        "NE": (r - 1, c + 1, [(-1, 0), (0, 1)]),
        "SW": (r + 1, c - 1, [(1, 0), (0, -1)]),
        "SE": (r + 1, c + 1, [(1, 0), (0, 1)]),
    }
    if force_preserved_corner:
        chosen = random.sample(list(corner_anchors), random.randint(1, 2))
    else:
        chosen = random.sample(list(corner_anchors), random.randint(0, 1))

    for key in chosen:
        rr, cc, options = corner_anchors[key]
        dr, dc = random.choice(options)
        for step in range(1, random.randint(2, 4)):
            nr = rr + dr * step
            nc = cc + dc * step
            if in_bounds(patch.shape, nr, nc):
                patch[nr, nc] = GREEN

    return _crop_patch(patch)


def _can_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 2) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 0 or left < 0 or top + h > gh or left + w > gw:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    return not bool((grid[r0:r1, c0:c1] != BLACK).any())


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    view = grid[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _place_components(grid: np.ndarray, patches: list[np.ndarray]) -> bool:
    items = sorted(patches, key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    for patch in items:
        h, w = patch.shape
        candidates = [
            (r, c)
            for r in range(grid.shape[0] - h + 1)
            for c in range(grid.shape[1] - w + 1)
            if _can_place(grid, patch, r, c, pad=2)
        ]
        if not candidates:
            return False
        r, c = random.choice(candidates)
        _paste_patch(grid, patch, r, c)
    return True


def _validate_semantics(grid: np.ndarray) -> bool:
    centers = _find_square_centers(grid)
    if not (3 <= len(centers) <= 5):
        return False

    rewritten, meta = _apply_square_rewrite(grid)
    out = main(grid)
    if not np.array_equal(out[(rewritten == GREEN) | (rewritten == RED)], rewritten[(rewritten == GREEN) | (rewritten == RED)]):
        return False

    before = _enclosed_black_mask(grid)
    after = _enclosed_black_mask(rewritten)
    center_mask = np.zeros_like(grid, dtype=bool)
    for r, c in centers:
        center_mask[r, c] = True
    before_non_center = before & (~center_mask)

    leaked = before_non_center & (~after)
    kept = after

    leaked_sizes = _component_sizes(leaked)
    kept_sizes = _component_sizes(kept)
    if not leaked_sizes or max(leaked_sizes) < 2:
        return False
    if not kept_sizes or max(kept_sizes) < 4:
        return False

    kept_corner_count = sum(item["kept"] for item in meta)
    erased_corner_count = sum(item["erased"] for item in meta)
    if kept_corner_count == 0 or erased_corner_count == 0:
        return False

    if not bool((out == YELLOW).any()) or not bool((out == RED).any()):
        return False

    return True


def _build_attempt() -> np.ndarray | None:
    total_boxes = random.randint(3, 5)
    leak_count = 1 if total_boxes == 3 else random.randint(1, 2)
    neutral_count = total_boxes - leak_count

    patches: list[np.ndarray] = [_make_stable_room()]
    for _ in range(leak_count):
        patches.append(_crop_patch(_random_patch_transform(random.choice(LEAKY_TEMPLATES))))

    forced_preserve_index = random.randrange(neutral_count) if neutral_count > 0 else -1
    for idx in range(neutral_count):
        patches.append(_make_neutral_box(force_preserved_corner=(idx == forced_preserve_index)))

    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    h = random.randint(max(16, max_h + 4), 28)
    w = random.randint(max(16, max_w + 4), 28)
    grid = np.zeros((h, w), dtype=int)

    if not _place_components(grid, patches):
        return None
    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")

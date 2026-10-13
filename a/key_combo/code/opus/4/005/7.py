from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, in_bounds, neighbors4, scale_sprite
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= r < h and 0 <= c < w

    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


CHECKER_BLOCK = np.array(
    [[Color.BLUE, Color.RED], [Color.RED, Color.BLUE]],
    dtype=int,
)

GEN_MAX_ATTEMPTS = 300


def _component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        queue = deque([(r, c)])
        seen[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in neighbors4(cr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return count


def _enclosed_region(boundary_mask: np.ndarray) -> np.ndarray:
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    h, w = map(int, boundary_mask.shape)
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not boundary_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not boundary_mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds((h, w), nr, nc) and not boundary_mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    return (~boundary_mask) & (~exterior)


def _has_bbox_exterior_pocket(boundary: np.ndarray, enclosed: np.ndarray) -> bool:
    if boundary.shape[0] <= 2 or boundary.shape[1] <= 2:
        return False
    pocket = (~boundary) & (~enclosed)
    return bool(np.any(pocket[1:-1, 1:-1]))


def _make_rectangle_ring(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        boundary = np.zeros((h, w), dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
        enclosed = _enclosed_region(boundary)
        if enclosed.sum() >= 2 and _component_count(boundary) == 1:
            return boundary, enclosed
    raise RuntimeError("failed to build rectangle ring")


def _make_notch_ring(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        h = rng.randint(5, 8)
        w = rng.randint(5, 8)
        notch_top = rng.randint(1, h - 4)
        notch_bottom = rng.randint(notch_top + 2, h - 2)
        notch_depth = rng.randint(2, w - 3)

        boundary = np.zeros((h, w), dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, -1] = True
        boundary[: notch_top + 1, 0] = True
        boundary[notch_bottom:, 0] = True
        boundary[notch_top, : notch_depth + 1] = True
        boundary[notch_bottom, : notch_depth + 1] = True
        boundary[notch_top : notch_bottom + 1, notch_depth] = True
        boundary = np.rot90(boundary, rng.randrange(4))

        enclosed = _enclosed_region(boundary)
        if enclosed.sum() < 2:
            continue
        if _component_count(boundary) != 1:
            continue
        if not _has_bbox_exterior_pocket(boundary, enclosed):
            continue
        return boundary, enclosed
    raise RuntimeError("failed to build notch ring")


def _sample_ring(rng: random.Random, *, force_concave: bool) -> dict[str, np.ndarray | bool]:
    kind = "notch" if force_concave else rng.choice(["rect", "notch", "notch"])
    if kind == "rect":
        boundary, enclosed = _make_rectangle_ring(rng)
        return {"boundary": boundary, "enclosed": enclosed, "concave": False}
    boundary, enclosed = _make_notch_ring(rng)
    return {"boundary": boundary, "enclosed": enclosed, "concave": True}


def _place_ring(
    grid: np.ndarray,
    blocked: np.ndarray,
    ring: dict[str, np.ndarray | bool],
    rng: random.Random,
) -> dict[str, np.ndarray | bool | int] | None:
    boundary = np.asarray(ring["boundary"], dtype=bool)
    h, w = map(int, boundary.shape)
    gh, gw = map(int, grid.shape)
    if gh - h < 2 or gw - w < 2:
        return None

    positions = [(r0, c0) for r0 in range(1, gh - h) for c0 in range(1, gw - w)]
    rng.shuffle(positions)

    for r0, c0 in positions:
        if np.any(blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1]):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][boundary] = Color.GREEN
        blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True
        return {
            "r0": r0,
            "c0": c0,
            "boundary": boundary,
            "enclosed": np.asarray(ring["enclosed"], dtype=bool),
            "concave": bool(ring["concave"]),
        }
    return None


def _validate_candidate(grid: np.ndarray, placed: list[dict[str, np.ndarray | bool | int]]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 15:
        return False
    if set(map(int, np.unique(grid))) - {Color.BLACK, Color.GREEN}:
        return False

    green = grid == Color.GREEN
    if _component_count(green) != len(placed):
        return False

    enclosed = _enclosed_region(green) & (grid == Color.BLACK)
    expected = np.zeros_like(green, dtype=bool)
    for info in placed:
        r0 = int(info["r0"])
        c0 = int(info["c0"])
        local = np.asarray(info["enclosed"], dtype=bool)
        h, w = map(int, local.shape)
        expected[r0 : r0 + h, c0 : c0 + w] |= local
        if int(local.sum()) < 2:
            return False
        if bool(info["concave"]) and not _has_bbox_exterior_pocket(
            np.asarray(info["boundary"], dtype=bool),
            local,
        ):
            return False

    if not np.array_equal(enclosed, expected):
        return False
    if not bool(np.any(enclosed)):
        return False
    if not bool(np.any((grid == Color.BLACK) & (~enclosed))):
        return False
    return True


def _generate_candidate(rng: random.Random, ring_count: int) -> np.ndarray | None:
    if ring_count == 1:
        h = rng.randint(9, 15)
        w = rng.randint(9, 15)
    else:
        h = rng.randint(13, 15)
        w = rng.randint(13, 15)

    grid = np.full((h, w), Color.BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    need_concave = rng.random() < 0.8
    placed: list[dict[str, np.ndarray | bool | int]] = []
    used_concave = False

    for i in range(ring_count):
        ring = _sample_ring(rng, force_concave=need_concave and not used_concave and i == 0)
        placement = _place_ring(grid, blocked, ring, rng)
        if placement is None:
            return None
        placed.append(placement)
        used_concave = used_concave or bool(placement["concave"])

    if need_concave and not used_concave:
        return None
    if not _validate_candidate(grid, placed):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale the grid and checkerboard only the black pixels enclosed by green rings.

    Input structure:
    - A black background contains one or more 4-connected green boundary rings.
    - The rings are closed, non-nested, and surround black pixels that look identical to the exterior black background.
    - No other input colors are needed: the only visible cue is whether a black pixel is topologically inside or outside a green ring.

    Rule:
    - Classify each original black pixel by 4-connected reachability before scaling: if it can reach the grid edge without crossing green, it is exterior; otherwise it is enclosed.
    - Upscale the entire grid by a factor of 2 so each input pixel becomes one 2x2 output block.
    - Green pixels become solid 2x2 green blocks, exterior black pixels become solid 2x2 black blocks, and enclosed black pixels become the blue/red checkerboard block [[1,2],[2,1]].
    - The output size is exactly double the input size in both dimensions.

    Key insight:
    - Color alone is insufficient because both target groups are black; the deciding signal is the pixel's topological position relative to the green boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    enclosed_black = _enclosed_region(g == Color.GREEN) & (g == Color.BLACK)
    out = scale_sprite(g, 2)
    for r, c in np.argwhere(enclosed_black):
        r = int(r)
        c = int(c)
        out[2 * r : 2 * r + 2, 2 * c : 2 * c + 2] = CHECKER_BLOCK
    return np.asarray(out, dtype=int)


def generate_input() -> np.ndarray:
    rng = random
    ring_count = rng.choice([1, 2, 2])
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate(rng, ring_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

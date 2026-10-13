from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400
MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 22
MIN_OBJECTS = 4
MAX_OBJECTS = 6
BRANCHES = ("hflip", "vflip", "corner", "keep")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return comps


def _branch_for_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    h, w = shape
    r0, c0, r1, c1 = bbox
    touch_top = r0 == 0
    touch_bottom = r1 == h - 1
    touch_left = c0 == 0
    touch_right = c1 == w - 1
    if (touch_top or touch_bottom) and (touch_left or touch_right):
        return "corner"
    if touch_left or touch_right:
        return "hflip"
    if touch_top or touch_bottom:
        return "vflip"
    return "keep"


def _apply_rule(mask: np.ndarray, branch: str) -> np.ndarray:
    if branch == "corner":
        return np.rot90(mask, 2)
    if branch == "hflip":
        return np.fliplr(mask)
    if branch == "vflip":
        return np.flipud(mask)
    return mask.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each monochrome sprite in place according to how its bounding box touches the outer grid border.

    Input structure:
    - Background is gray.
    - Several disjoint non-gray monochrome components appear as small 4-connected sprites.
    - Each sprite is contained in a tight bounding box no larger than 4x4, and that box may or may not touch the grid's top, bottom, left, or right border.

    Rule:
    - For each sprite, inspect which outer-grid borders coincide with its bounding box.
    - If the box touches a left or right border only, flip the sprite horizontally inside its own box.
    - If the box touches a top or bottom border only, flip the sprite vertically; if it touches a horizontal and a vertical border together (a corner case), rotate it by 180 degrees.
    - If the box touches no border, leave the sprite unchanged; always redraw the result in the original box with the original color.

    Key insight:
    - The deciding signal is not color or shape identity but the sprite bbox's contact pattern with the global grid edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    h, w = g.shape

    for comp in _find_components(g, BG):
        r0, c0, r1, c1 = comp["bbox"]
        branch = _branch_for_bbox(comp["bbox"], (h, w))
        new_mask = _apply_rule(comp["mask"], branch)
        box = out[r0 : r1 + 1, c0 : c1 + 1]
        box[:] = BG
        box[new_mask] = int(comp["color"])

    return out


def _grow_mask(box_h: int, box_w: int, area: int, rng: random.Random) -> np.ndarray:
    cells = {(rng.randrange(box_h), rng.randrange(box_w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_asymmetric(mask: np.ndarray) -> bool:
    return (
        not np.array_equal(mask, np.fliplr(mask))
        and not np.array_equal(mask, np.flipud(mask))
        and not np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_dihedral(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 4)
        area = rng.randint(3, min(8, box_h * box_w))
        mask = _grow_mask(box_h, box_w, area, rng)
        if max(mask.shape) > 4 or min(mask.shape) < 2:
            continue
        mask = _random_dihedral(mask, rng)
        if _is_asymmetric(mask):
            return mask
    raise RuntimeError("failed to generate an asymmetric sprite")


def _candidate_positions(
    branch: str, box_h: int, box_w: int, height: int, width: int
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []

    if branch == "corner":
        corners = [
            (0, 0),
            (0, width - box_w),
            (height - box_h, 0),
            (height - box_h, width - box_w),
        ]
        for pos in corners:
            if pos not in candidates:
                candidates.append(pos)
        return candidates

    if branch == "hflip":
        for r0 in range(1, height - box_h):
            candidates.append((r0, 0))
            candidates.append((r0, width - box_w))
        return candidates

    if branch == "vflip":
        for c0 in range(1, width - box_w):
            candidates.append((0, c0))
            candidates.append((height - box_h, c0))
        return candidates

    for r0 in range(1, height - box_h):
        for c0 in range(1, width - box_w):
            candidates.append((r0, c0))
    return candidates


def _fits(blocked: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + box_h + 1)
    cc1 = min(blocked.shape[1], c0 + box_w + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + box_h + 1)
    cc1 = min(blocked.shape[1], c0 + box_w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_sprite(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    branch: str,
    rng: random.Random,
) -> bool:
    box_h, box_w = mask.shape
    candidates = _candidate_positions(branch, box_h, box_w, grid.shape[0], grid.shape[1])
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        if not _fits(blocked, r0, c0, box_h, box_w):
            continue
        patch = grid[r0 : r0 + box_h, c0 : c0 + box_w]
        if np.any(patch[mask] != BG):
            continue
        patch[mask] = color
        _mark_blocked(blocked, r0, c0, box_h, box_w)
        return True

    return False


def _validate_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    h, w = grid.shape
    branches_seen = set()
    for comp in _find_components(grid, BG):
        r0, c0, r1, c1 = comp["bbox"]
        mask = comp["mask"]
        if int(mask.sum()) < 3 or int(mask.sum()) > 8:
            return False
        if max(mask.shape) > 4 or min(mask.shape) < 2:
            return False
        if not _is_asymmetric(mask):
            return False
        branch = _branch_for_bbox((r0, c0, r1, c1), (h, w))
        branches_seen.add(branch)
    return branches_seen == set(BRANCHES)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    width = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    total_objects = rng.randint(MIN_OBJECTS, MAX_OBJECTS)

    branch_counts = {branch: 1 for branch in BRANCHES}
    for _ in range(total_objects - len(BRANCHES)):
        choices = [branch for branch in BRANCHES if branch != "corner" or branch_counts[branch] < 4]
        branch_counts[rng.choice(choices)] += 1

    colors = rng.sample(PALETTE, total_objects)
    objects = []
    color_idx = 0
    for branch in BRANCHES:
        for _ in range(branch_counts[branch]):
            objects.append(
                {
                    "branch": branch,
                    "mask": _random_shape(rng),
                    "color": colors[color_idx],
                }
            )
            color_idx += 1

    priority = {"corner": 0, "hflip": 1, "vflip": 1, "keep": 2}
    objects.sort(key=lambda obj: (priority[obj["branch"]], -int(obj["mask"].sum())))

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    for obj in objects:
        if not _place_sprite(grid, blocked, obj["mask"], int(obj["color"]), obj["branch"], rng):
            return None

    if not _validate_grid(grid):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

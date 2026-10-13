from __future__ import annotations

import random

import numpy as np


AZURE = 8
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _components_by_color(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr, sc in np.argwhere(g != int(bg)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        color = int(g[sr, sc])
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIR4.values():
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr < h
                    and 0 <= cc < w
                    and not seen[rr, cc]
                    and int(g[rr, cc]) == color
                ):
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        comps.append(
            {
                "color": color,
                "coords": arr,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    return comps


def _growth_direction(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    local_r = coords[:, 0] - int(r0)
    local_c = coords[:, 1] - int(c0)
    counts = {
        "U": int(np.count_nonzero(local_r * 2 < height)),
        "D": int(np.count_nonzero(local_r * 2 >= height)),
        "L": int(np.count_nonzero(local_c * 2 < width)),
        "R": int(np.count_nonzero(local_c * 2 >= width)),
    }
    fewest = min(counts.values())
    dirs = [d for d, n in counts.items() if n == fewest]
    assert len(dirs) == 1, "growth direction must be uniquely encoded by sparse half"
    return dirs[0]


def _edge_extension_cells(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    dr, dc = DIR4[direction]
    cells: list[tuple[int, int]] = []
    for r, c in coords:
        r, c = int(r), int(c)
        if (
            (direction == "U" and r == r0)
            or (direction == "D" and r == r1)
            or (direction == "L" and c == c0)
            or (direction == "R" and c == c1)
        ):
            cells.append((r + dr, c + dc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each uneven foreground object one cell toward its sparse side.

    Input structure:
    - The background is Azure.
    - The foreground consists of separated monochrome 4-connected objects.
    - Each object has an uneven pixel distribution inside its bounding box, with one of the top, bottom, left, or right halves containing uniquely fewer object pixels.

    Rule:
    - For every object, split its bounding box into top, bottom, left, and right halves and count the object's pixels in each half.
    - The half with the unique smallest count gives the growth direction.
    - On that side of the object's bounding box, add one same-colored pixel just outside each foreground edge pixel.
    - All other pixels remain unchanged.

    Key insight:
    - The object grows from dense end toward sparse end; the density imbalance is the direction code, so the object is not expanded on every side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    h, w = g.shape
    out = g.copy()
    additions: dict[tuple[int, int], int] = {}

    for comp in _components_by_color(g, AZURE):
        color = int(comp["color"])
        coords = comp["coords"]
        bbox = comp["bbox"]
        direction = _growth_direction(coords, bbox)
        for rr, cc in _edge_extension_cells(coords, bbox, direction):
            rr, cc = int(rr), int(cc)
            assert 0 <= rr < h and 0 <= cc < w, "growth target must stay inside the grid"
            assert int(g[rr, cc]) == AZURE, "growth target must be background"
            prev = additions.get((rr, cc))
            assert prev is None or prev == color, "growth targets must not conflict"
            additions[(rr, cc)] = color

    for (r, c), color in additions.items():
        out[int(r), int(c)] = int(color)
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 28)
GRID_W_RANGE = (13, 28)
OBJECT_COUNT_RANGE = (2, 5)
PATCH_SIDES = (4, 6, 8)
GEN_RETRIES = 500
PATCH_RETRIES = 80
PLACE_RETRIES = 300


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _patch_direction(mask: np.ndarray) -> str | None:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    try:
        return _growth_direction(coords.astype(int), (int(r0), int(c0), int(r1), int(c1)))
    except AssertionError:
        return None


def _thin_dense_part(
    mask: np.ndarray,
    candidates: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    rng.shuffle(candidates)
    target = rng.randint(0, max(1, len(candidates) // 5))
    removed = 0
    for r, c in candidates:
        if removed >= target:
            break
        if (r, c) in protected:
            continue
        mask[r, c] = False
        if _mask_connected(mask):
            removed += 1
        else:
            mask[r, c] = True


def _right_sparse_patch(rng: random.Random) -> np.ndarray:
    h = rng.choice(PATCH_SIDES)
    w = rng.choice(PATCH_SIDES)
    split = w // 2
    anchor = rng.randrange(h)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :split] = True
    mask[anchor, split:] = True

    span = rng.randint(1, min(4, h))
    lo = max(0, anchor - span + 1)
    hi = min(anchor, h - span)
    start = rng.randint(lo, hi)
    mask[start : start + span, w - 1] = True

    for c in range(split, w - 1):
        if rng.random() < 0.45:
            rr = rng.randrange(start, start + span)
            mask[rr, c] = True

    dense = [(r, c) for r in range(h) for c in range(split)]
    protected = {(r, split - 1) for r in range(h)}
    protected.update((anchor, c) for c in range(split))
    _thin_dense_part(mask, dense, protected, rng)
    return mask


def _down_sparse_patch(rng: random.Random) -> np.ndarray:
    h = rng.choice(PATCH_SIDES)
    w = rng.choice(PATCH_SIDES)
    split = h // 2
    anchor = rng.randrange(w)
    mask = np.zeros((h, w), dtype=bool)
    mask[:split, :] = True
    mask[split:, anchor] = True

    span = rng.randint(1, min(4, w))
    lo = max(0, anchor - span + 1)
    hi = min(anchor, w - span)
    start = rng.randint(lo, hi)
    mask[h - 1, start : start + span] = True

    for r in range(split, h - 1):
        if rng.random() < 0.45:
            cc = rng.randrange(start, start + span)
            mask[r, cc] = True

    dense = [(r, c) for r in range(split) for c in range(w)]
    protected = {(split - 1, c) for c in range(w)}
    protected.update((r, anchor) for r in range(split))
    _thin_dense_part(mask, dense, protected, rng)
    return mask


def _make_patch(direction: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(PATCH_RETRIES):
        if direction == "R":
            mask = _right_sparse_patch(rng)
        elif direction == "L":
            mask = np.fliplr(_right_sparse_patch(rng)).copy()
        elif direction == "D":
            mask = _down_sparse_patch(rng)
        else:
            mask = np.flipud(_down_sparse_patch(rng)).copy()

        if _mask_connected(mask) and _patch_direction(mask) == direction:
            return mask
    return None


def _relative_claim(mask: np.ndarray, direction: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    bbox = (int(r0), int(c0), int(r1), int(c1))
    originals = [(int(r), int(c)) for r, c in coords]
    extensions = _edge_extension_cells(coords.astype(int), bbox, direction)
    return originals, extensions


def _mark_with_margin(forbidden: np.ndarray, cells: list[tuple[int, int]], radius: int = 1) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < h and 0 <= cc < w:
                    forbidden[rr, cc] = True


def _place_object(
    grid: np.ndarray,
    forbidden: np.ndarray,
    mask: np.ndarray,
    direction: str,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    originals, extensions = _relative_claim(mask, direction)
    claim_rel = originals + extensions
    min_r = min(r for r, _ in claim_rel)
    max_r = max(r for r, _ in claim_rel)
    min_c = min(c for _, c in claim_rel)
    max_c = max(c for _, c in claim_rel)

    r_starts = list(range(-min_r, h - max_r))
    c_starts = list(range(-min_c, w - max_c))
    placements = [(r, c) for r in r_starts for c in c_starts]
    rng.shuffle(placements)

    for r0, c0 in placements[:PLACE_RETRIES]:
        claim_abs = [(r0 + r, c0 + c) for r, c in claim_rel]
        if any(forbidden[r, c] for r, c in claim_abs):
            continue
        for r, c in originals:
            grid[r0 + r, c0 + c] = int(color)
        _mark_with_margin(forbidden, claim_abs, radius=1)
        return True
    return False


def _direction_plan(n: int, rng: random.Random) -> list[str]:
    dirs = list(DIR4.keys())
    rng.shuffle(dirs)
    out = dirs[: min(n, 4)]
    while len(out) < n:
        extra = list(DIR4.keys())
        rng.shuffle(extra)
        out.extend(extra)
    return out[:n]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(list(PALETTE), n_objects)
    directions = _direction_plan(n_objects, rng)

    grid = np.full((h, w), AZURE, dtype=int)
    forbidden = np.zeros((h, w), dtype=bool)

    for color, direction in zip(colors, directions):
        mask = _make_patch(direction, rng)
        if mask is None:
            return None
        if not _place_object(grid, forbidden, mask, direction, int(color), rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if int(np.sum(grid == AZURE)) <= int(grid.size * 0.35):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

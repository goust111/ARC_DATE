from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TALLY_SLOTS = ((0, 0), (2, 0), (0, 2), (2, 2))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_patch(cells: set[tuple[int, int]], color: int = RED) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.full((h, w), BLACK, dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)
    return patch


def _neighbor_counts(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.int16), 1)
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )


def _iterative_clean(mask: np.ndarray) -> np.ndarray:
    cur = mask.astype(bool).copy()
    while True:
        nxt = cur & (_neighbor_counts(cur) >= 2)
        if np.array_equal(nxt, cur):
            return nxt
        cur = nxt


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_hw(coords: np.ndarray) -> tuple[int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _orientation(coords: np.ndarray) -> str:
    h, w = _bbox_hw(coords)
    return "H" if w > h else "V"


def _render_output(horizontal_count: int, vertical_count: int) -> np.ndarray:
    assert 0 <= horizontal_count <= 4
    assert 0 <= vertical_count <= 4
    out = np.full((3, 6), BLACK, dtype=int)
    for r, c in TALLY_SLOTS[:horizontal_count]:
        out[r, c] = GREEN
    for r, c in TALLY_SLOTS[:vertical_count]:
        out[r, c + 3] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Iteratively clean weak red pixels and tally cleaned sprites by bbox orientation into two side panels.

    Input structure:
    - A black background contains several 4-connected red sprites plus some detached red noise pixels or tiny noise clusters.
    - Some sprites include one-pixel-wide protrusions whose tip pixels have weak 4-neighbor support.
    - Sprites are separated from each other, so every cleaned survivor remains an individual object.

    Rule:
    - Repeatedly erase every red pixel that has fewer than two red 4-neighbors until the red pattern stops changing.
    - For each surviving 4-connected red sprite, compute its tight bounding box.
    - If bbox width is strictly greater than bbox height, count it as horizontal; otherwise count it as vertical.
    - Return a 3x6 black tally grid: the left 3x3 panel marks the first H corner slots in green for horizontal sprites, and the right 3x3 panel marks the first V corner slots in blue for vertical sprites, using slot order top-left, bottom-left, top-right, bottom-right.

    Key insight:
    - The classification must be done after full iterative cleaning, because deleting a thin protrusion can shrink a sprite's bbox enough to flip its horizontal/vertical category.
    """
    g = np.asarray(grid, dtype=int)
    cleaned = _iterative_clean(g == RED)
    horizontal_count = 0
    vertical_count = 0
    for comp in _components(cleaned):
        if _orientation(comp) == "H":
            horizontal_count += 1
        else:
            vertical_count += 1
    return _render_output(horizontal_count, vertical_count)


GRID_RANGE = (15, 18)
OBJECT_COUNT_RANGE = (4, 6)
NOISE_CLUSTER_RANGE = (2, 5)
MAX_ATTEMPTS = 500

HORIZONTAL_CORES = ((2, 4), (2, 5), (3, 4), (3, 5), (2, 6))
VERTICAL_CORES = ((4, 2), (5, 2), (4, 3), (5, 3), (6, 2), (3, 3), (4, 4))


def _rectangle_cells(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _attach_spur(
    cells: set[tuple[int, int]],
    h: int,
    w: int,
    side: str,
    length: int,
) -> set[tuple[int, int]]:
    cells = set(cells)
    if side in {"top", "bottom"}:
        c = random.randrange(w)
        if side == "top":
            for k in range(1, length + 1):
                cells.add((-k, c))
        else:
            for k in range(1, length + 1):
                cells.add((h - 1 + k, c))
    else:
        r = random.randrange(h)
        if side == "left":
            for k in range(1, length + 1):
                cells.add((r, -k))
        else:
            for k in range(1, length + 1):
                cells.add((r, w - 1 + k))
    return cells


def _make_object_patch(final_orientation: str, require_flip: bool) -> tuple[np.ndarray, bool]:
    pools = HORIZONTAL_CORES if final_orientation == "H" else VERTICAL_CORES
    dims_pool = list(pools)
    random.shuffle(dims_pool)

    for h, w in dims_pool:
        area = h * w
        max_spur = min(3, 20 - area)
        if max_spur < 0:
            continue

        cells = _rectangle_cells(h, w)
        choices: list[tuple[str, int] | None] = [None]
        if max_spur >= 1:
            if require_flip:
                if final_orientation == "H":
                    need = max(1, w - h)
                    sides = ("top", "bottom")
                else:
                    need = max(1, h - w + 1)
                    sides = ("left", "right")
                if need > max_spur:
                    continue
                choices = [(side, length) for side in sides for length in range(need, max_spur + 1)]
            else:
                for side in ("top", "bottom", "left", "right"):
                    for length in range(1, max_spur + 1):
                        choices.append((side, length))
                random.shuffle(choices)

        if not require_flip and random.random() < 0.25:
            choices = [None]

        for choice in choices:
            obj_cells = set(cells)
            if choice is not None:
                obj_cells = _attach_spur(obj_cells, h, w, choice[0], choice[1])
            patch = _cells_to_patch(obj_cells)
            if not (8 <= int((patch == RED).sum()) <= 20):
                continue

            cleaned = _iterative_clean(patch == RED)
            comps = _components(cleaned)
            if len(comps) != 1:
                continue
            if int(cleaned.sum()) != area:
                continue

            cleaned_orientation = _orientation(comps[0])
            original_orientation = _orientation(np.argwhere(patch == RED))
            flipped = cleaned_orientation != original_orientation
            if cleaned_orientation != final_orientation:
                continue
            if require_flip and not flipped:
                continue
            return patch, flipped

    raise RuntimeError(f"failed to build object for orientation {final_orientation}")


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if top < 1 or left < 1 or top + h > gh - 1 or left + w > gw - 1:
        return False
    red_cells = np.argwhere(patch == RED)
    for r, c in red_cells:
        rr = top + int(r)
        cc = left + int(c)
        if grid[rr, cc] != BLACK:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < gh and 0 <= nc < gw and grid[nr, nc] == RED:
                    return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    red_cells = np.argwhere(patch == RED)
    for r, c in red_cells:
        grid[top + int(r), left + int(c)] = RED


def _place_random_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    h, w = patch.shape
    positions = [
        (r, c)
        for r in range(1, gh - h)
        for c in range(1, gw - w)
        if _can_place_patch(grid, patch, r, c)
    ]
    if not positions:
        return False
    top, left = random.choice(positions)
    _paste_patch(grid, patch, top, left)
    return True


def _noise_patch() -> np.ndarray:
    if random.random() < 0.65:
        return np.asarray([[RED]], dtype=int)
    if random.random() < 0.5:
        return np.asarray([[RED, RED]], dtype=int)
    return np.asarray([[RED], [RED]], dtype=int)


def _has_required_structure(
    grid: np.ndarray,
    expected_h: int,
    expected_v: int,
    expected_objects: int,
) -> bool:
    original = grid == RED
    cleaned = _iterative_clean(original)

    original_comps = _components(original)
    sprite_comps = [comp for comp in original_comps if int(comp.shape[0]) >= 8]
    if len(sprite_comps) != expected_objects:
        return False

    cleaned_comps = _components(cleaned)
    h_count = sum(_orientation(comp) == "H" for comp in cleaned_comps)
    v_count = len(cleaned_comps) - h_count
    if (h_count, v_count) != (expected_h, expected_v):
        return False

    flip_found = False
    for comp in sprite_comps:
        sprite_mask = np.zeros_like(original, dtype=bool)
        sprite_mask[comp[:, 0], comp[:, 1]] = True
        kept = np.argwhere(cleaned & sprite_mask)
        if kept.size == 0:
            return False
        if _orientation(comp) != _orientation(kept):
            flip_found = True
    return flip_found


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_RANGE)
    width = random.randint(*GRID_RANGE)
    total_objects = random.randint(*OBJECT_COUNT_RANGE)

    h_low = max(0, total_objects - 4)
    h_high = min(4, total_objects)
    horizontal_target = random.randint(h_low, h_high)
    vertical_target = total_objects - horizontal_target

    orientations = ["H"] * horizontal_target + ["V"] * vertical_target
    random.shuffle(orientations)
    flip_orientation = random.choice(orientations)

    patches: list[np.ndarray] = []
    flip_assigned = False
    for ori in orientations:
        require_flip = (not flip_assigned) and ori == flip_orientation
        patch, flipped = _make_object_patch(ori, require_flip=require_flip)
        patches.append(patch)
        flip_assigned |= flipped and require_flip
    random.shuffle(patches)

    grid = np.full((height, width), BLACK, dtype=int)
    for patch in patches:
        if not _place_random_patch(grid, patch):
            return None

    for _ in range(random.randint(*NOISE_CLUSTER_RANGE)):
        if not _place_random_patch(grid, _noise_patch()):
            return None

    expected = _render_output(horizontal_target, vertical_target)
    if not _has_required_structure(grid, horizontal_target, vertical_target, total_objects):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

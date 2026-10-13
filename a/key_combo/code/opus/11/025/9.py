from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch
    return dst


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not cells:
        return False
    H, W = mask.shape
    seen = {cells[0]}
    queue: deque[tuple[int, int]] = deque([cells[0]])
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _grow_mask(size: int, area: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(list(frontier))
        mask[nr, nc] = True
    return mask


def _draw_manhattan_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    if random.choice([False, True]):
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _seed_full_bbox_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    hub = (random.randint(1, size - 2), random.randint(1, size - 2))
    anchors = [
        (0, random.randrange(size)),
        (size - 1, random.randrange(size)),
        (random.randrange(size), 0),
        (random.randrange(size), size - 1),
    ]
    mask[hub] = True
    for anchor in anchors:
        _draw_manhattan_path(mask, hub, anchor)
    return mask


def _quadrant_cells(size: int, quadrant: str) -> list[tuple[int, int]]:
    if size == 4:
        top_rows = [0, 1]
        bottom_rows = [2, 3]
        left_cols = [0, 1]
        right_cols = [2, 3]
    else:
        top_rows = [0, 1]
        bottom_rows = [3, 4]
        left_cols = [0, 1]
        right_cols = [3, 4]

    if quadrant == "TL":
        return [(r, c) for r in top_rows for c in left_cols]
    if quadrant == "TR":
        return [(r, c) for r in top_rows for c in right_cols]
    if quadrant == "BL":
        return [(r, c) for r in bottom_rows for c in left_cols]
    return [(r, c) for r in bottom_rows for c in right_cols]


def _build_colored_body(mask: np.ndarray) -> np.ndarray:
    body = np.zeros(mask.shape, dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(cells)
    blue_count = random.randint(1, len(cells) - 1)
    blue_cells = set(cells[:blue_count])
    for cell in cells:
        body[cell] = BLUE if cell in blue_cells else GREEN
    return body


def _scene_transform(body: np.ndarray, marker: tuple[int, int]) -> np.ndarray:
    return np.fliplr(body) if marker[1] < body.shape[1] / 2 else np.flipud(body)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source_box = _bbox((g != BG) & (g != YELLOW))
    target_box = _bbox(g == YELLOW)
    assert source_box is not None, "expected one source sprite"
    assert target_box is not None, "expected one yellow target slot"

    source = _crop(g, source_box)
    target = _crop(g, target_box)
    assert source.shape == target.shape, "source and target must share the same size"
    assert source.shape[0] in (4, 5) and source.shape[0] == source.shape[1], "sprite bbox must be 4x4 or 5x5"
    assert np.all(target == YELLOW), "target slot must be a solid yellow rectangle"

    source_colors = set(np.unique(source).tolist())
    assert source_colors <= {BG, BLUE, RED, GREEN}, "unexpected colors inside source sprite"
    marker_coords = np.argwhere(source == RED)
    assert marker_coords.shape == (1, 2), "expected exactly one red marker"
    marker = tuple(map(int, marker_coords[0]))
    body = source.copy()
    body[marker] = BG
    assert np.any(body == BLUE) and np.any(body == GREEN), "source must contain both blue and green"

    return {
        "grid": g,
        "source_box": source_box,
        "target_box": target_box,
        "source": source,
        "target": target,
        "body": body,
        "marker": marker,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the marked source sprite in the direction implied by the marker side, then stamp it into the matching slot.

    Input structure:
    - A black background contains one 4x4 or 5x5 multicolor source sprite made of blue and green cells plus one red marker cell.
    - The red marker lies inside the source sprite bbox in one quadrant, and a same-sized solid yellow rectangle appears elsewhere as a target slot.
    - The source sprite may have internal black gaps, but the yellow slot is fully filled.

    Rule:
    - Crop the source sprite to its tight bbox and locate the unique red marker.
    - Remove the red marker from the sprite, then mirror the remaining blue/green pattern horizontally if the marker is in the left half, otherwise vertically.
    - Erase the original source bbox to black and overwrite the yellow target slot with the mirrored blue/green pattern.

    Key insight:
    - The target slot does not indicate the transform; the mirror direction is encoded only by which side of the source sprite contains the red marker.
    """
    scene = _extract_scene(grid)
    out = np.array(scene["grid"], copy=True)

    sr0, sc0, sr1, sc1 = scene["source_box"]
    tr0, tc0, tr1, tc1 = scene["target_box"]
    mirrored = _scene_transform(scene["body"], scene["marker"])

    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = mirrored
    return out


def _build_sprite(size: int, quadrant: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    quad_cells = _quadrant_cells(size, quadrant)
    max_area = min(12, size * size)
    for _ in range(MASK_ATTEMPTS):
        mask = _seed_full_bbox_mask(size)
        base_area = int(mask.sum())
        if base_area > max_area:
            continue
        area = random.randint(max(6, base_area), max_area)
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        if not _is_connected(mask):
            continue
        if _bbox(mask) != (0, 0, size - 1, size - 1):
            continue
        candidates = [(r, c) for r, c in quad_cells if mask[r, c]]
        if not candidates:
            continue

        colored = _build_colored_body(mask)
        marker = random.choice(candidates)
        source = colored.copy()
        source[marker] = RED
        if not np.any(source == BLUE) or not np.any(source == GREEN):
            continue

        body = source.copy()
        body[marker] = BG
        flip_lr = np.fliplr(body)
        flip_ud = np.flipud(body)
        if np.array_equal(body, flip_lr) or np.array_equal(body, flip_ud):
            continue
        if np.array_equal(flip_lr, flip_ud):
            continue

        chosen = _scene_transform(body, marker)
        other = flip_ud if marker[1] < size / 2 else flip_lr
        if np.array_equal(chosen, other):
            continue
        return source, body, marker
    raise RuntimeError("failed to build a non-degenerate sprite")


def _random_layout(size: int) -> tuple[int, int, int, int, int, int]:
    H = random.randint(max(10, size + 5), 14)
    W = random.randint(max(11, 2 * size + 4), 16)
    src_r = random.randint(1, H - size - 1)
    src_c = random.randint(1, max(1, W - 2 * size - 3))
    min_slot_c = src_c + size + 2
    slot_c = random.randint(min_slot_c, W - size - 1)
    slot_r = random.randint(0, H - size)
    return H, W, src_r, src_c, slot_r, slot_c


def _build_expected(
    shape: tuple[int, int],
    target_pos: tuple[int, int],
    transformed: np.ndarray,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    _paste(out, transformed, target_pos[0], target_pos[1])
    return out


def _validate_generated_scene(grid: np.ndarray, expected: np.ndarray) -> None:
    scene = _extract_scene(grid)
    body = scene["body"]
    marker = scene["marker"]
    transformed = _scene_transform(body, marker)
    alternate = np.flipud(body) if marker[1] < body.shape[1] / 2 else np.fliplr(body)

    assert not np.array_equal(body, transformed), "mirror step must matter"
    assert not np.array_equal(transformed, alternate), "marker side must uniquely determine the direction"
    assert main(grid).shape == expected.shape
    assert np.array_equal(main(grid), expected), "generated input must match the intended output"


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([4, 5])
    quadrant = random.choice(["TL", "TR", "BL", "BR"])
    try:
        source, body, marker = _build_sprite(size, quadrant)
    except RuntimeError:
        return None

    H, W, src_r, src_c, slot_r, slot_c = _random_layout(size)
    transformed = _scene_transform(body, marker)
    expected = _build_expected((H, W), (slot_r, slot_c), transformed)

    grid = expected.copy()
    grid[slot_r : slot_r + size, slot_c : slot_c + size] = YELLOW
    _paste(grid, source, src_r, src_c)

    if np.any(grid[slot_r : slot_r + size, slot_c : slot_c + size] != YELLOW):
        return None
    if src_c + size + 1 >= slot_c:
        return None

    try:
        _validate_generated_scene(grid, expected)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")

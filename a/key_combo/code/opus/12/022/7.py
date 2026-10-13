from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("lu", "ru", "ld", "rd")
GEN_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _scale_patch(patch: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(patch, np.ones((factor, factor), dtype=int))


def _paste_transparent(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    target = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    target[mask] = patch[mask]


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _anchor(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return (h - 1) // 2, (w - 1) // 2


def _quadrant(marker: tuple[int, int], frame_box: tuple[int, int, int, int]) -> str:
    r, c = marker
    r0, c0, r1, c1 = frame_box
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    if c > center_c and r < center_r:
        return "ru"
    if c < center_c and r > center_r:
        return "ld"
    if c > center_c and r > center_r:
        return "rd"
    return "lu"


def _transform_for_quadrant(patch: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "ru":
        return np.fliplr(patch)
    if quadrant == "ld":
        return np.flipud(patch)
    if quadrant == "rd":
        return np.flipud(np.fliplr(patch))
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the outside pattern into the red frame and stamp mirrored original-size copies on gray markers.

    Input structure:
    - A black background contains one red 1-pixel-thick hollow rectangular frame.
    - Outside the frame there is one small non-red non-gray multicolor pattern whose bbox is 2x2 or 3x3.
    - There are 2-3 gray single-pixel markers outside the frame, each lying in a different quadrant relative to the frame center.

    Rule:
    - Detect the red frame and extract its interior size.
    - Extract the small pattern's full bbox, including black cells inside that bbox, and scale it uniformly so it exactly fills the frame interior.
    - Keep the red border, write the scaled pattern into the frame interior, and keep the original outside pattern unchanged.
    - For each gray marker, stamp one original-size copy of the pattern with black treated as transparent; align the marker to the pattern center for odd sizes or to the upper-left of the central 2x2 area for even sizes.
    - The marker's quadrant relative to the frame center chooses the stamped orientation: upper-left = original, upper-right = horizontal flip, lower-left = vertical flip, lower-right = both flips.

    Key insight:
    - The same source pattern appears twice at different scales: enlarged inside the frame, and copied near markers where the global quadrant of each marker determines which mirror transform to use.
    """
    g = np.asarray(grid, dtype=int)
    red_box = _bbox(g == RED)
    assert red_box is not None, "expected one red frame"
    r0, c0, r1, c1 = red_box
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "frame must have an interior"

    frame_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(frame_patch[0, :] == RED)
    assert np.all(frame_patch[-1, :] == RED)
    assert np.all(frame_patch[:, 0] == RED)
    assert np.all(frame_patch[:, -1] == RED)

    pattern_mask = (g != BLACK) & (g != RED) & (g != GRAY)
    pattern_box = _bbox(pattern_mask)
    assert pattern_box is not None, "expected one source pattern"
    pattern = _crop(g, pattern_box)
    ph, pw = pattern.shape
    assert ph == pw and ph in (2, 3)

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h == inner_w
    assert inner_h % ph == 0 and inner_w % pw == 0
    scale = inner_h // ph

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = _scale_patch(pattern, scale)

    ar, ac = _anchor(pattern.shape)
    for mr, mc in np.argwhere(g == GRAY):
        marker = (int(mr), int(mc))
        transformed = _transform_for_quadrant(pattern, _quadrant(marker, red_box))
        sr = int(mr) - ar
        sc = int(mc) - ac
        assert 0 <= sr and 0 <= sc and sr + ph <= g.shape[0] and sc + pw <= g.shape[1]
        out[int(mr), int(mc)] = BLACK
        _paste_transparent(out, transformed, sr, sc, transparent=BLACK)
    return out


def _full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _generate_mask(size: int) -> np.ndarray:
    anchor_r, anchor_c = _anchor((size, size))
    for _ in range(400):
        if size == 2:
            filled = 3 if random.random() < 0.8 else 4
        else:
            filled = random.randint(5, 8)
        cells = [(r, c) for r in range(size) for c in range(size)]
        chosen = set(random.sample(cells, filled))
        chosen.add((anchor_r, anchor_c))
        mask = np.zeros((size, size), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if mask.sum() != filled:
            continue
        if not _full_bbox(mask):
            continue
        if not _connected(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a connected pattern mask")


def _generate_pattern(size: int) -> np.ndarray:
    for _ in range(500):
        mask = _generate_mask(size)
        filled_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        color_count = random.randint(2, min(4, len(filled_cells)))
        palette = random.sample(PALETTE, color_count)
        random.shuffle(filled_cells)

        patch = np.zeros((size, size), dtype=int)
        for color, (r, c) in zip(palette, filled_cells):
            patch[r, c] = color
        for r, c in filled_cells[color_count:]:
            patch[r, c] = random.choice(palette)

        variants = {
            patch.tobytes(),
            np.fliplr(patch).tobytes(),
            np.flipud(patch).tobytes(),
            np.flipud(np.fliplr(patch)).tobytes(),
        }
        if len(variants) != 4:
            continue
        return patch
    raise RuntimeError("failed to generate an asymmetric pattern")


def _rect(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return box


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _candidate_stamp_boxes(
    grid_shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    pattern_shape: tuple[int, int],
    quadrant: str,
    gap: int = 2,
) -> list[tuple[int, int, tuple[int, int], tuple[int, int, int, int]]]:
    h, w = grid_shape
    ph, pw = pattern_shape
    ar, ac = _anchor(pattern_shape)
    fr0, fc0, fr1, fc1 = frame_box
    candidates = []
    if quadrant == "lu":
        row_range = range(0, fr0 - gap - ph + 1)
        col_range = range(0, fc0 - gap - pw + 1)
    elif quadrant == "ru":
        row_range = range(0, fr0 - gap - ph + 1)
        col_range = range(fc1 + gap + 1, w - pw + 1)
    elif quadrant == "ld":
        row_range = range(fr1 + gap + 1, h - ph + 1)
        col_range = range(0, fc0 - gap - pw + 1)
    else:
        row_range = range(fr1 + gap + 1, h - ph + 1)
        col_range = range(fc1 + gap + 1, w - pw + 1)

    for sr in row_range:
        for sc in col_range:
            marker = (sr + ar, sc + ac)
            box = (sr, sc, sr + ph - 1, sc + pw - 1)
            if _quadrant(marker, frame_box) == quadrant:
                candidates.append((sr, sc, marker, box))
    random.shuffle(candidates)
    return candidates


def _candidate_boxes(
    grid_shape: tuple[int, int],
    size: int,
) -> list[tuple[int, int, int, int]]:
    h, w = grid_shape
    boxes = []
    for r0 in range(0, h - size + 1):
        for c0 in range(0, w - size + 1):
            boxes.append((r0, c0, r0 + size - 1, c0 + size - 1))
    random.shuffle(boxes)
    return boxes


def _place_layout() -> tuple[np.ndarray, dict] | None:
    size = random.choice([2, 3])
    pattern = _generate_pattern(size)
    scale_choices = [2, 3, 4] if size == 3 else [2, 3, 4, 5]
    scale = random.choice(scale_choices)
    inner = size * scale
    outer = inner + 2
    margin = size + 3

    min_side = max(16, outer + 2 * margin)
    max_side = min(30, min_side + 4)
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)

    fr0 = random.randint(margin, h - outer - margin)
    fc0 = random.randint(margin, w - outer - margin)
    frame_box = (fr0, fc0, fr0 + outer - 1, fc0 + outer - 1)

    marker_count = 3 if random.random() < 0.7 else 2
    quadrants = random.sample(list(QUADRANTS), marker_count)
    occupied = [frame_box]
    markers: list[tuple[int, int]] = []
    stamp_boxes: list[tuple[int, int, int, int]] = []

    for quadrant in quadrants:
        placed = False
        for _sr, _sc, marker, box in _candidate_stamp_boxes((h, w), frame_box, pattern.shape, quadrant):
            if all(_boxes_far_enough(box, other, gap=2) for other in occupied):
                occupied.append(box)
                stamp_boxes.append(box)
                markers.append(marker)
                placed = True
                break
        if not placed:
            return None

    pattern_box = None
    for box in _candidate_boxes((h, w), size):
        if all(_boxes_far_enough(box, other, gap=2) for other in occupied):
            pattern_box = box
            break
    if pattern_box is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[fr0 : fr0 + outer, fc0 : fc0 + outer] = RED
    grid[fr0 + 1 : fr0 + outer - 1, fc0 + 1 : fc0 + outer - 1] = BLACK

    pr0, pc0, pr1, pc1 = pattern_box
    grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = pattern
    for mr, mc in markers:
        grid[mr, mc] = GRAY

    meta = {
        "pattern": pattern,
        "frame_box": frame_box,
        "pattern_box": pattern_box,
        "markers": markers,
        "scale": scale,
        "quadrants": [_quadrant(marker, frame_box) for marker in markers],
    }
    return grid, meta


def _render_expected(meta: dict, grid_shape: tuple[int, int], source_grid: np.ndarray) -> np.ndarray:
    out = np.array(source_grid, copy=True)
    fr0, fc0, fr1, fc1 = meta["frame_box"]
    pattern = meta["pattern"]
    scale = meta["scale"]
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = _scale_patch(pattern, scale)
    ar, ac = _anchor(pattern.shape)
    for marker in meta["markers"]:
        mr, mc = marker
        sr = mr - ar
        sc = mc - ac
        out[mr, mc] = BLACK
        transformed = _transform_for_quadrant(pattern, _quadrant(marker, meta["frame_box"]))
        _paste_transparent(out, transformed, sr, sc, transparent=BLACK)
    return out


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    frame_box = meta["frame_box"]
    pattern = meta["pattern"]
    markers = meta["markers"]

    if len(markers) not in (2, 3):
        return False
    if len({tuple(m) for m in markers}) != len(markers):
        return False
    if len(set(meta["quadrants"])) != len(markers):
        return False
    if not set(meta["quadrants"]).issubset(set(QUADRANTS)):
        return False
    if pattern.shape not in ((2, 2), (3, 3)):
        return False
    if np.array_equal(grid, main(grid)):
        return False

    expected = _render_expected(meta, grid.shape, grid)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    red_box = _bbox(grid == RED)
    if red_box != frame_box:
        return False

    pattern_mask = (grid != BLACK) & (grid != RED) & (grid != GRAY)
    if _bbox(pattern_mask) != meta["pattern_box"]:
        return False

    for marker in markers:
        if _quadrant(marker, frame_box) not in QUADRANTS:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _place_layout()
        if built is None:
            continue
        grid, meta = built
        if _validate_case(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid input")

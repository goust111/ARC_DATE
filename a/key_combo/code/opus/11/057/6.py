from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
CONTROL_COLORS = (3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _is_filled_rectangle(mask: np.ndarray, comp: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _score_axis(blue_mask: np.ndarray, occluder_mask: np.ndarray, axis2: int) -> tuple[int, int, int, int]:
    h, w = blue_mask.shape
    oob = 0
    added_outside = 0
    added_inside = 0
    matched = 0
    for r, c in np.argwhere(blue_mask):
        mc = int(axis2 - c)
        if mc < 0 or mc >= w:
            oob += 1
            continue
        if blue_mask[r, mc]:
            matched += 1
        elif occluder_mask[r, mc]:
            added_inside += 1
        else:
            added_outside += 1
    return oob, added_outside, -added_inside, -matched


def _best_axis(blue_mask: np.ndarray, occluder_mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], bool]:
    h, w = blue_mask.shape
    assert h > 0 and w > 0 and np.any(blue_mask)
    ranked: list[tuple[tuple[int, int, int, int], int]] = []
    for axis2 in range(0, 2 * w - 1):
        score = _score_axis(blue_mask, occluder_mask, axis2)
        ranked.append((score, axis2))
    ranked.sort()
    best_score, best_axis2 = ranked[0]
    unique = len(ranked) == 1 or ranked[1][0] != best_score
    return int(best_axis2), best_score, unique


def _restore_symmetry(blue_mask: np.ndarray, axis2: int) -> np.ndarray:
    h, w = blue_mask.shape
    restored = blue_mask.copy()
    for r, c in np.argwhere(blue_mask):
        mc = int(axis2 - c)
        if 0 <= mc < w:
            restored[r, mc] = True
    return restored


def _scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(grid, np.ones((factor, factor), dtype=grid.dtype))


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the red occluder, mirror-complete the blue object, then enlarge it by the control-row count.

    Input structure:
    - The last row is a black control row with 2-3 scattered non-black pixels.
    - Above it is a black body containing one blue object and one red filled rectangle occluding part of that object.
    - The blue object is left-right symmetric before occlusion, and the hidden cells lie under the red rectangle.

    Rule:
    - Ignore the last row except for counting its non-black pixels K.
    - In the body, erase the red filled rectangle to black.
    - Find the vertical mirror axis that best explains the remaining blue pixels, with any newly inferred mirror pixels landing inside the erased rectangle.
    - Mirror-complete the blue object, crop to its tight bounding box, and scale every cropped pixel to a K x K block.

    Key insight:
    - The red rectangle is an occluder, not part of the target; after removing it, the missing side is uniquely recovered by mirror symmetry, and only the number of control pixels determines the magnification.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    body = g[:-1].copy()
    control_row = g[-1]
    k = int(np.count_nonzero(control_row != BLACK))
    assert 2 <= k <= 3

    red_mask = body == RED
    red_components = _components4(red_mask)
    assert len(red_components) == 1 and _is_filled_rectangle(red_mask, red_components[0])

    body[red_mask] = BLACK
    blue_mask = body == BLUE
    assert np.any(blue_mask)

    axis2, _score, unique = _best_axis(blue_mask, red_mask)
    assert unique
    restored_mask = _restore_symmetry(blue_mask, axis2)
    bb = _bbox(restored_mask)
    assert bb is not None

    cropped = np.zeros((bb[2] - bb[0] + 1, bb[3] - bb[1] + 1), dtype=int)
    cropped[_crop_bbox(restored_mask, bb)] = BLUE
    out = _scale_grid(cropped, k)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _build_shape() -> np.ndarray | None:
    for _ in range(120):
        odd = bool(random.getrandbits(1))
        if odd:
            max_extent = random.choice((1, 2))
            height = random.randint(4, 6)
            extents = [random.randint(0, max_extent) for _ in range(height)]
        else:
            max_extent = random.choice((2, 3))
            height = random.randint(4, 6 if max_extent == 2 else 5)
            extents = [random.randint(1, max_extent) for _ in range(height)]

        run_len = random.randint(2, min(3, height))
        run_start = random.randint(0, height - run_len)
        for i in range(run_start, run_start + run_len):
            extents[i] = max_extent

        if min(extents) == max(extents):
            continue

        if odd:
            width = 2 * max_extent + 1
            mask = np.zeros((height, width), dtype=bool)
            center = max_extent
            for r, e in enumerate(extents):
                mask[r, center - e : center + e + 1] = True
        else:
            width = 2 * max_extent
            mask = np.zeros((height, width), dtype=bool)
            for r, e in enumerate(extents):
                left = max_extent - e
                right = max_extent + e - 1
                mask[r, left : right + 1] = True

        area = int(mask.sum())
        if not (8 <= area <= 16):
            continue
        if np.all(mask == mask[:, [0]]):
            continue
        return mask
    return None


def _occluder_candidates(full_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = full_mask.shape
    axis2 = w - 1
    candidates: list[tuple[int, int, int, int]] = []
    for rh in (2, 3):
        for rw in (2, 3):
            if rh > h or rw > w:
                continue
            for r0 in range(h - rh + 1):
                for c0 in range(w - rw + 1):
                    cells = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
                    side_marks = {np.sign(2 * c - axis2) for _, c in cells}
                    if 0 in side_marks or len(side_marks) != 1:
                        continue

                    overlap = [(r, c) for r, c in cells if full_mask[r, c]]
                    if len(overlap) < 2:
                        continue

                    good = True
                    for r, c in overlap:
                        mc = axis2 - c
                        if not full_mask[r, mc]:
                            good = False
                            break
                        if r0 <= r < r0 + rh and c0 <= mc < c0 + rw:
                            good = False
                            break
                    if not good:
                        continue

                    visible = full_mask.copy()
                    for r, c in overlap:
                        visible[r, c] = False
                    if np.array_equal(visible, full_mask):
                        continue

                    restored = _restore_symmetry(visible, axis2)
                    if not np.array_equal(restored, full_mask):
                        continue
                    if len(_components4(visible)) > 2:
                        continue
                    candidates.append((r0, c0, rh, rw))
    return candidates


def _place_shape(full_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = full_mask.shape
    body_h = random.randint(h + 1, min(h + 4, 9))
    body_w = random.randint(w + 2, min(w + 6, 10))
    r0 = random.randint(0, body_h - h - 1)
    c0 = random.randint(1, body_w - w - 1)
    body = np.zeros((body_h, body_w), dtype=int)
    body[r0 : r0 + h, c0 : c0 + w][full_mask] = BLUE
    return body, (r0, c0)


def _make_control_row(width: int, k: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    cols = sorted(random.sample(range(width), k))
    colors = random.sample(CONTROL_COLORS, k)
    for c, color in zip(cols, colors):
        row[c] = color
    return row


def _expected_output(full_mask: np.ndarray, k: int) -> np.ndarray:
    cropped = np.where(full_mask, BLUE, BLACK).astype(int)
    return _scale_grid(cropped, k)


def _validate_instance(inp: np.ndarray, full_mask: np.ndarray, local_rect: tuple[int, int, int, int], anchor: tuple[int, int], k: int) -> bool:
    body = inp[:-1]
    control_row = inp[-1]
    if int(np.count_nonzero(control_row)) != k:
        return False
    colors = [int(v) for v in control_row if int(v) != BLACK]
    if len(colors) != len(set(colors)):
        return False

    red_mask = body == RED
    red_components = _components4(red_mask)
    if len(red_components) != 1 or not _is_filled_rectangle(red_mask, red_components[0]):
        return False

    blue_visible = body == BLUE
    axis2, score, unique = _best_axis(blue_visible, red_mask)
    if not unique:
        return False

    local_h, local_w = full_mask.shape
    _, c_anchor = anchor
    expected_axis2 = 2 * c_anchor + local_w - 1
    if axis2 != expected_axis2:
        return False
    if score[0] != 0 or score[1] != 0 or score[2] >= 0:
        return False

    restored = _restore_symmetry(blue_visible, axis2)
    expected_body = np.zeros_like(body, dtype=bool)
    r_anchor, c_anchor = anchor
    expected_body[r_anchor : r_anchor + local_h, c_anchor : c_anchor + local_w] = full_mask
    if not np.array_equal(restored, expected_body):
        return False
    if np.array_equal(restored, blue_visible):
        return False

    visible_crop_bb = _bbox(blue_visible)
    restored_crop_bb = _bbox(restored)
    if visible_crop_bb is None or restored_crop_bb is None or visible_crop_bb == restored_crop_bb:
        return False

    expected = _expected_output(full_mask, k)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    red_r0, red_c0, red_r1, red_c1 = local_rect
    hidden = full_mask[red_r0 : red_r1 + 1, red_c0 : red_c1 + 1]
    if int(hidden.sum()) < 2:
        return False
    if max(actual.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    full_mask = _build_shape()
    if full_mask is None:
        return None

    candidates = _occluder_candidates(full_mask)
    if not candidates:
        return None
    rect = random.choice(candidates)
    body, anchor = _place_shape(full_mask)
    r_anchor, c_anchor = anchor
    r0, c0, rh, rw = rect

    global_r0 = r_anchor + r0
    global_c0 = c_anchor + c0
    body[global_r0 : global_r0 + rh, global_c0 : global_c0 + rw] = RED

    k = random.randint(2, 3)
    control_row = _make_control_row(body.shape[1], k)
    inp = np.vstack([body, control_row[np.newaxis, :]])

    local_rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
    if not _validate_instance(inp, full_mask, local_rect, anchor, k):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")

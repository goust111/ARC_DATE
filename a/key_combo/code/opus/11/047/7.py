from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
PINK = 6

SEPARATOR_COLUMN = 8
GRID_WIDTH = 18
LEFT_WIDTH = 8
RIGHT_WIDTH = 9

GEN_HEIGHT_RANGE = (11, 17)
GEN_ATTEMPTS = 600
RECT_ATTEMPTS = 300
PLACE_ATTEMPTS = 120


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                crop[rr - r0, cc - c0] = True
            out.append(
                {
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": crop,
                    "area": len(coords),
                }
            )
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    m = np.asarray(mask, dtype=bool)
    return m.shape, tuple(int(v) for v in m.reshape(-1))


def _interior_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4
    out = np.zeros(shape, dtype=bool)
    out[r0 + 1 : r1, c0 + 1 : c1] = True
    return out


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _validate_left_rectangles(left: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    a_mask = left == RED
    b_mask = left == BLUE
    a_box = _bbox(a_mask)
    b_box = _bbox(b_mask)
    assert a_box is not None and b_box is not None, "expected visible red and blue rectangle evidence"

    ar0, ac0, ar1, ac1 = a_box
    br0, bc0, br1, bc1 = b_box
    assert ar1 - ar0 + 1 >= 4 and ac1 - ac0 + 1 >= 4
    assert br1 - br0 + 1 >= 4 and bc1 - bc0 + 1 >= 4

    a_patch = left[ar0 : ar1 + 1, ac0 : ac1 + 1]
    b_patch = left[br0 : br1 + 1, bc0 : bc1 + 1]
    assert set(map(int, np.unique(a_patch))) <= {RED, BLUE}, "red rectangle bbox may only contain red and blue"
    assert np.all(b_patch == BLUE), "blue rectangle must be a filled visible rectangle"

    assert np.any(left[ar0, ac0 : ac1 + 1] == RED)
    assert np.any(left[ar1, ac0 : ac1 + 1] == RED)
    assert np.any(left[ar0 : ar1 + 1, ac0] == RED)
    assert np.any(left[ar0 : ar1 + 1, ac1] == RED)

    a_inner = _interior_mask(left.shape, a_box)
    b_inner = _interior_mask(left.shape, b_box)
    stamp = a_inner | b_inner
    assert np.any(a_inner & ~b_inner), "rectangle A must contribute uniquely to the union"
    assert np.any(b_inner & ~a_inner), "rectangle B must contribute uniquely to the union"
    assert _is_connected(stamp), "combined interior mask must be 4-connected"
    return a_box, b_box, a_inner, b_inner, _crop_mask(stamp)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    divider_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(divider_cols) == 1, "expected one grey divider column"
    sep = int(divider_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    a_box, b_box, a_inner, b_inner, stamp = _validate_left_rectangles(left)

    yellow_components = _components(right == YELLOW)
    transformed = [comp for comp in yellow_components if np.array_equal(comp["mask"], stamp)]
    assert 2 <= len(transformed) <= 3, "expected 2-3 matching placeholder objects"

    classes = Counter(_mask_signature(comp["mask"]) for comp in yellow_components)
    return {
        "sep": sep,
        "left": left,
        "right": right,
        "a_box": a_box,
        "b_box": b_box,
        "a_inner_crop": _crop_mask(a_inner),
        "b_inner_crop": _crop_mask(b_inner),
        "stamp": stamp,
        "yellow_components": yellow_components,
        "matching_count": len(transformed),
        "yellow_classes": classes,
    }


def _render_with_template(grid: np.ndarray, template: np.ndarray, *, recolor_all_yellow: bool = False) -> np.ndarray:
    info = _analyze(grid)
    out = np.asarray(grid, dtype=int).copy()

    ar0, ac0, ar1, ac1 = info["a_box"]
    br0, bc0, br1, bc1 = info["b_box"]
    out[ar0 : ar1 + 1, ac0 : ac1 + 1] = BLACK
    out[br0 : br1 + 1, bc0 : bc1 + 1] = BLACK

    right_offset = info["sep"] + 1
    if recolor_all_yellow:
        out[:, right_offset:][out[:, right_offset:] == YELLOW] = PINK
        return out

    for comp in info["yellow_components"]:
        r0, c0, r1, c1 = comp["bbox"]
        if np.array_equal(comp["mask"], template):
            out[r0 : r1 + 1, right_offset + c0 : right_offset + c1 + 1] = np.where(
                template, PINK, BLACK
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the overlapped rectangles' interior union and stamp that hidden mask onto the matching yellow placeholders.

    Input structure:
    - The grid has black background and one full grey divider column splitting a left rectangle zone from a right placeholder zone.
    - The left zone contains a red filled rectangle partly occluded by a blue filled rectangle, with the overlap shown in blue.
    - The right zone contains several yellow 4-connected components; exactly 2-3 of them have the same normalized mask as the hidden union of the two rectangles' interior regions.

    Rule:
    - Recover the red rectangle bbox from the visible red extent and the blue rectangle bbox from the visible blue extent.
    - Take each rectangle's strict 1-cell interior, ignoring the current colors inside those interior cells, and compute their boolean union.
    - Erase both left-zone rectangles to black, then replace every yellow component whose normalized mask matches that union with a pink copy of the union at the component's bbox location.
    - Leave the divider column and any non-matching yellow components unchanged.

    Key insight:
    - The stamp template is not directly drawn anywhere in the input; it appears only after extracting both rectangles' interiors and unioning those interior masks.
    """
    info = _analyze(grid)
    return _render_with_template(np.asarray(grid, dtype=int), info["stamp"])


def _random_connected_mask(rng: np.random.Generator, *, max_h: int = 3, max_w: int = 3) -> np.ndarray:
    candidates = [
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    ]
    usable = [mask for mask in candidates if mask.shape[0] <= max_h and mask.shape[1] <= max_w]
    return usable[int(rng.integers(len(usable)))].copy()


def _try_left_layout(
    rng: np.random.Generator, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(RECT_ATTEMPTS):
        h_a = int(rng.integers(4, min(7, height) + 1))
        w_a = int(rng.integers(4, 7))
        h_b = int(rng.integers(4, min(7, height) + 1))
        w_b = int(rng.integers(4, 7))
        if w_a > LEFT_WIDTH or w_b > LEFT_WIDTH:
            continue

        r_a = int(rng.integers(0, height - h_a + 1))
        c_a = int(rng.integers(0, LEFT_WIDTH - w_a + 1))
        r_b = int(rng.integers(0, height - h_b + 1))
        c_b = int(rng.integers(0, LEFT_WIDTH - w_b + 1))

        a_box = (r_a, c_a, r_a + h_a - 1, c_a + w_a - 1)
        b_box = (r_b, c_b, r_b + h_b - 1, c_b + w_b - 1)
        inter_r0 = max(a_box[0], b_box[0])
        inter_c0 = max(a_box[1], b_box[1])
        inter_r1 = min(a_box[2], b_box[2])
        inter_c1 = min(a_box[3], b_box[3])
        inter_h = inter_r1 - inter_r0 + 1
        inter_w = inter_c1 - inter_c0 + 1
        if inter_h < 2 or inter_w < 2:
            continue

        left = np.zeros((height, LEFT_WIDTH), dtype=int)
        left[a_box[0] : a_box[2] + 1, a_box[1] : a_box[3] + 1] = RED
        left[b_box[0] : b_box[2] + 1, b_box[1] : b_box[3] + 1] = BLUE

        try:
            _a_box, _b_box, _a_inner, _b_inner, stamp = _validate_left_rectangles(left)
        except Exception:
            continue

        if stamp.shape[0] > 5 or stamp.shape[1] > 5 or int(stamp.sum()) < 4:
            continue

        return left, stamp, _crop_mask(_a_inner), _crop_mask(_b_inner)
    return None


def _can_place(mask: np.ndarray, top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = int(top + r)
        cc = int(left + c)
        if blocked[rr, cc]:
            return False
    return True


def _mark_blocked(mask: np.ndarray, top: int, left: int, blocked: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in np.argwhere(mask):
        rr = int(top + r)
        cc = int(left + c)
        blocked[rr, cc] = True
        for nr, nc in _neighbors8(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_masks(height: int, width: int, masks: list[np.ndarray], rng: np.random.Generator) -> list[tuple[int, int]] | None:
    for _ in range(PLACE_ATTEMPTS):
        order = list(range(len(masks)))
        rng.shuffle(order)
        blocked = np.zeros((height, width), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(masks)
        ok = True
        for idx in order:
            mask = masks[idx]
            mh, mw = mask.shape
            candidates = [
                (r, c)
                for r in range(height - mh + 1)
                for c in range(width - mw + 1)
                if _can_place(mask, r, c, blocked)
            ]
            if not candidates:
                ok = False
                break
            top, left = candidates[int(rng.integers(len(candidates)))]
            positions[idx] = (top, left)
            _mark_blocked(mask, top, left, blocked)
        if ok:
            assert all(pos is not None for pos in positions)
            return [(int(r), int(c)) for r, c in positions]  # type: ignore[misc]
    return None


def _build_candidate_grid(
    left: np.ndarray,
    placeholder_mask: np.ndarray,
    distractor_mask: np.ndarray,
    placeholder_count: int,
    distractor_count: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    height = left.shape[0]
    masks = [placeholder_mask.copy() for _ in range(placeholder_count)] + [
        distractor_mask.copy() for _ in range(distractor_count)
    ]
    positions = _place_masks(height, RIGHT_WIDTH, masks, rng)
    if positions is None:
        return None

    grid = np.zeros((height, GRID_WIDTH), dtype=int)
    grid[:, SEPARATOR_COLUMN] = GREY
    grid[:, :LEFT_WIDTH] = left

    for mask, (r, c) in zip(masks, positions):
        _paint_mask(grid[:, SEPARATOR_COLUMN + 1 :], mask, r, c, YELLOW)
    return grid


def _validate_generation(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        out = main(grid)
    except Exception:
        return False

    if not (grid.ndim == 2 and out.ndim == 2 and grid.shape == out.shape and grid.shape[1] == GRID_WIDTH):
        return False

    if int(np.count_nonzero(grid[:, :LEFT_WIDTH] == RED)) == 0 or int(np.count_nonzero(grid[:, :LEFT_WIDTH] == BLUE)) == 0:
        return False
    if int(np.count_nonzero(out[:, :LEFT_WIDTH])) != 0:
        return False
    if not np.all(out[:, SEPARATOR_COLUMN] == GREY):
        return False

    classes = info["yellow_classes"]
    match_sig = _mask_signature(info["stamp"])
    match_count = classes[match_sig]
    if match_count not in {2, 3}:
        return False
    if sum(1 for sig, count in classes.items() if sig != match_sig and count == match_count) == 0:
        return False

    if np.array_equal(_render_with_template(grid, info["a_inner_crop"]), out):
        return False
    if np.array_equal(_render_with_template(grid, info["b_inner_crop"]), out):
        return False
    if np.array_equal(_render_with_template(grid, info["stamp"], recolor_all_yellow=True), out):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        height = int(rng.integers(*GEN_HEIGHT_RANGE))
        left_result = _try_left_layout(rng, height)
        if left_result is None:
            continue
        left, placeholder_mask, _a_inner, _b_inner = left_result

        distractor_mask = _random_connected_mask(rng)
        for _ in range(20):
            if _mask_signature(distractor_mask) != _mask_signature(placeholder_mask):
                break
            distractor_mask = _random_connected_mask(rng)
        if _mask_signature(distractor_mask) == _mask_signature(placeholder_mask):
            continue

        count_options = [2]
        if (
            placeholder_mask.shape[0] <= 3
            and placeholder_mask.shape[1] <= 3
            and distractor_mask.shape[0] <= 3
            and distractor_mask.shape[1] <= 3
        ):
            count_options.append(3)
        rng.shuffle(count_options)

        grid = None
        for count in count_options:
            candidate = _build_candidate_grid(
                left=left,
                placeholder_mask=placeholder_mask,
                distractor_mask=distractor_mask,
                placeholder_count=count,
                distractor_count=count,
                rng=rng,
            )
            if candidate is not None and _validate_generation(candidate):
                grid = candidate
                break
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

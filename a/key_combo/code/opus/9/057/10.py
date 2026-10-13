from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500
RECENT_OUTPUT_WINDOW = 24

_RECENT_OUTPUT_SIGNATURES: set[bytes] = set()
_RECENT_OUTPUT_ORDER: list[bytes] = []


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_to_output(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    cropped = _crop(mask, box)
    out = np.zeros(cropped.shape, dtype=int)
    out[cropped] = RED
    return out


def _grid_signature(grid: np.ndarray) -> bytes:
    rows, cols = grid.shape
    return grid.tobytes() + f":{rows}x{cols}".encode()


def _remember_output(grid: np.ndarray) -> bool:
    sig = _grid_signature(grid)
    if sig in _RECENT_OUTPUT_SIGNATURES:
        return False
    _RECENT_OUTPUT_SIGNATURES.add(sig)
    _RECENT_OUTPUT_ORDER.append(sig)
    if len(_RECENT_OUTPUT_ORDER) > RECENT_OUTPUT_WINDOW:
        old = _RECENT_OUTPUT_ORDER.pop(0)
        _RECENT_OUTPUT_SIGNATURES.discard(old)
    return True


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _separator_columns(grid: np.ndarray, frame: int) -> list[int]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == frame)]
    assert len(cols) >= 5
    assert cols[0] == 0 and cols[-1] == grid.shape[1] - 1
    steps = [b - a for a, b in zip(cols, cols[1:])]
    assert len(set(steps)) == 1
    return cols


def _panel_blue_mask(panel: np.ndarray) -> np.ndarray:
    mask = panel == BLUE
    assert mask.any()
    return mask


def _parse_sequence(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all(g[0, :] == GREY) and np.all(g[-1, :] == GREY)

    separators = _separator_columns(g, GREY)
    panels = [g[:, c0 : c1 + 1] for c0, c1 in zip(separators, separators[1:])]
    counts = [int((panel == BLUE).sum()) for panel in panels]
    count_hist = Counter(counts)
    full_count = max(count_hist.items(), key=lambda kv: kv[1])[0]
    damaged = [i for i, count in enumerate(counts) if count != full_count]
    assert len(damaged) == 1
    damaged_idx = damaged[0]
    assert 0 < damaged_idx < len(panels) - 1
    assert counts[damaged_idx] < full_count

    complete_indices = [i for i, count in enumerate(counts) if count == full_count]
    assert len(complete_indices) >= 3

    template_mask = None
    template_shape = None
    local_left = None
    top_rows: dict[int, int] = {}
    for idx in complete_indices:
        mask = _panel_blue_mask(panels[idx])
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        crop = _crop(mask, box)
        assert _is_connected(crop)
        if template_mask is None:
            template_mask = crop
            template_shape = crop.shape
            local_left = c0
        else:
            assert crop.shape == template_shape
            assert np.array_equal(crop, template_mask)
            assert c0 == local_left
        top_rows[idx] = r0

    assert template_mask is not None and local_left is not None
    deltas = set()
    sorted_indices = sorted(complete_indices)
    for pos, idx0 in enumerate(sorted_indices):
        for idx1 in sorted_indices[pos + 1 :]:
            dy = top_rows[idx1] - top_rows[idx0]
            dx = idx1 - idx0
            assert dy % dx == 0
            deltas.add(dy // dx)
    assert len(deltas) == 1
    delta = deltas.pop()
    assert abs(delta) in (1, 2)

    inferred_tops = {top_rows[idx] + delta * (damaged_idx - idx) for idx in complete_indices}
    assert len(inferred_tops) == 1
    expected_top = inferred_tops.pop()

    sprite_h, sprite_w = template_mask.shape
    panel_h, panel_w = panels[0].shape
    assert 1 <= expected_top <= panel_h - 1 - sprite_h
    assert 1 <= local_left <= panel_w - 1 - sprite_w

    return {
        "panels": panels,
        "damaged_idx": damaged_idx,
        "template_mask": template_mask,
        "expected_top": expected_top,
        "local_left": local_left,
        "delta": delta,
        "counts": counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the damaged frame from the motion pattern, then crop and recolor the missing pixels.

    Input structure:
    - The grid is a horizontal sequence of equal-width framed panels sharing grey border columns.
    - Each panel has black fill and contains one identical blue sprite at the same local columns.
    - The sprite's top row changes by a constant vertical delta across panels, and exactly one non-edge panel is damaged by replacing some sprite pixels with black.

    Rule:
    - Identify the complete sprite template from the undamaged panels and infer the constant row delta.
    - Use the complete panels to extrapolate the damaged panel's correct sprite position, rebuilding a reference copy there.
    - Compare the rebuilt reference against the damaged panel, keep cells that should be blue but are black, crop that missing-cell mask to its own bbox, and recolor it red on black.

    Key insight:
    - The target mask is not visible directly: the damaged panel must first be aligned to an inferred full reference frame obtained from the other motion-consistent panels.
    """
    info = _parse_sequence(grid)
    damaged_panel = info["panels"][info["damaged_idx"]]
    template_mask = info["template_mask"]
    top = info["expected_top"]
    left = info["local_left"]

    ref_mask = np.zeros_like(damaged_panel, dtype=bool)
    h, w = template_mask.shape
    ref_mask[top : top + h, left : left + w] = template_mask

    observed_blue = damaged_panel == BLUE
    assert not np.any(observed_blue & ~ref_mask)

    missing = ref_mask & (damaged_panel == BLACK)
    assert missing.any()
    return _mask_to_output(missing)


def _template_catalog() -> list[np.ndarray]:
    catalog = []
    for h, w in ((2, 3), (3, 2)):
        cells = [(r, c) for r in range(h) for c in range(w)]
        for area in range(4, h * w + 1):
            for combo in combinations(cells, area):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in combo:
                    mask[r, c] = True
                if not _is_connected(mask):
                    continue
                if not mask[0].any() or not mask[-1].any():
                    continue
                if not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not any(np.array_equal(mask, old) for old in catalog):
                    catalog.append(mask)
    return catalog


TEMPLATE_MASKS = _template_catalog()


def _place_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = dst[top : top + h, left : left + w]
    patch[mask] = color


def _build_expected_output(template_mask: np.ndarray, removed: np.ndarray) -> np.ndarray:
    return _mask_to_output(removed)


def _assumed_output(
    panel_shape: tuple[int, int],
    template_mask: np.ndarray,
    damaged_panel: np.ndarray,
    assumed_top: int,
    assumed_left: int,
) -> np.ndarray | None:
    h, w = template_mask.shape
    ph, pw = panel_shape
    if assumed_top < 1 or assumed_left < 1 or assumed_top + h > ph - 1 or assumed_left + w > pw - 1:
        return None
    ref = np.zeros(panel_shape, dtype=bool)
    ref[assumed_top : assumed_top + h, assumed_left : assumed_left + w] = template_mask
    missing = ref & (damaged_panel == BLACK)
    if not missing.any():
        return None
    return _mask_to_output(missing)


def _candidate_removed_masks(template_mask: np.ndarray) -> dict[str, list[np.ndarray]]:
    h, w = template_mask.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
    area = len(coords)
    options: dict[str, list[np.ndarray]] = {"motion": [], "left": [], "internal": []}
    for remove_count in range(1, area):
        frac = remove_count / area
        if not (0.30 <= frac <= 0.50):
            continue
        for subset in combinations(coords, remove_count):
            removed = np.zeros((h, w), dtype=bool)
            for r, c in subset:
                removed[r, c] = True
            remain = template_mask & ~removed
            if not remain.any():
                continue
            box = _bbox(remain)
            assert box is not None
            r0, c0, r1, c1 = box
            if r0 > 0:
                options["motion"].append(removed)
            elif c0 > 0:
                options["left"].append(removed)
            elif (r0, c0, r1, c1) == (0, 0, h - 1, w - 1):
                options["internal"].append(removed)
    return options


def _build_input(
    panel_h: int,
    interior_w: int,
    num_panels: int,
    template_mask: np.ndarray,
    top_rows: list[int],
    left: int,
    damaged_idx: int,
    removed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    panel_w = interior_w + 2
    stride = panel_w - 1
    grid_w = num_panels * stride + 1
    grid = np.zeros((panel_h, grid_w), dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    for i in range(num_panels + 1):
        grid[:, i * stride] = GREY

    damaged_panel = None
    for i in range(num_panels):
        col0 = i * stride
        panel = grid[:, col0 : col0 + panel_w]
        if i == damaged_idx:
            remain = template_mask & ~removed
            _place_mask(panel, remain, top_rows[i], left, BLUE)
            damaged_panel = panel.copy()
        else:
            _place_mask(panel, template_mask, top_rows[i], left, BLUE)
    assert damaged_panel is not None
    return grid, damaged_panel


def _attempt() -> np.ndarray | None:
    num_panels = random.choice([4, 5])
    interior_w = random.choice([4, 5]) if num_panels == 4 else 4
    weighted_options: list[tuple[np.ndarray, str, np.ndarray, np.ndarray]] = []
    branch_weights = {"motion": 2, "left": 1, "internal": 1}
    for template_mask in TEMPLATE_MASKS:
        sprite_h, sprite_w = template_mask.shape
        if sprite_w > interior_w:
            continue
        for branch, removed_list in _candidate_removed_masks(template_mask).items():
            for removed in removed_list:
                expected_output = _build_expected_output(template_mask, removed)
                if _grid_signature(expected_output) in _RECENT_OUTPUT_SIGNATURES:
                    continue
                for _ in range(branch_weights[branch]):
                    weighted_options.append((template_mask, branch, removed.copy(), expected_output))
    if not weighted_options:
        return None
    template_mask, branch, removed, expected_output = random.choice(weighted_options)
    sprite_h, sprite_w = template_mask.shape

    panel_w = interior_w + 2
    delta = random.choice([-2, -1, 1, 2])
    min_interior_h = max(7, sprite_h + (num_panels - 1) * abs(delta) + 1)
    max_interior_h = min(10, 28)
    if min_interior_h > max_interior_h:
        return None
    interior_h = random.randint(min_interior_h, max_interior_h)
    panel_h = interior_h + 2
    max_top = panel_h - 1 - sprite_h
    if delta > 0:
        start_lo, start_hi = 1, max_top - (num_panels - 1) * delta
    else:
        start_lo, start_hi = 1 + (num_panels - 1) * (-delta), max_top
    if start_lo > start_hi:
        return None
    start_top = random.randint(start_lo, start_hi)
    top_rows = [start_top + i * delta for i in range(num_panels)]

    left = random.randint(1, interior_w - sprite_w + 1)
    damaged_idx = random.randint(1, num_panels - 2)
    grid, damaged_panel = _build_input(
        panel_h,
        interior_w,
        num_panels,
        template_mask,
        top_rows,
        left,
        damaged_idx,
        removed,
    )

    remain = template_mask & ~removed
    remain_box = _bbox(remain)
    assert remain_box is not None
    wrong_top = top_rows[damaged_idx] + remain_box[0]
    wrong_left = left + remain_box[1]
    wrong_output = _assumed_output(damaged_panel.shape, template_mask, damaged_panel, wrong_top, wrong_left)

    if branch in {"motion", "left"} and wrong_output is not None and np.array_equal(wrong_output, expected_output):
        return None
    if branch == "motion" and remain_box[0] == 0:
        return None
    if branch == "left" and remain_box[1] == 0:
        return None
    if branch == "internal" and (remain_box[0] != 0 or remain_box[1] != 0):
        return None

    info = _parse_sequence(grid)
    if info["delta"] != delta:
        return None
    if info["expected_top"] != top_rows[damaged_idx]:
        return None
    if not np.array_equal(info["template_mask"], template_mask):
        return None
    if not np.array_equal(main(grid), expected_output):
        return None
    if not _remember_output(expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for ARC task 10")

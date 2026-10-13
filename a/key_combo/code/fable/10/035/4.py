from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))
_CASE_SEQUENCE = ("AR_CORE", "RA_CORE", "AR_EXT", "RA_EXT")
_CASE_CURSOR = 0
_SEEN_INPUT_SIGS: set[bytes] = set()
_SEEN_OUTPUT_SIGS: set[bytes] = set()


def _most_common_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bool_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    return np.asarray(panel != bg, dtype=bool)


def _mask_to_panel(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    panel = np.full(mask.shape, bg, dtype=int)
    panel[mask] = fg
    return panel


def _apply_trial(canvas: np.ndarray, panel: np.ndarray, bg: int) -> tuple[np.ndarray, bool]:
    trial_mask = panel != bg
    conflict = trial_mask & (canvas != bg)
    out = canvas.copy()
    if np.any(conflict):
        out[conflict] = bg
        return out, False
    out[trial_mask] = panel[trial_mask]
    return out, True


def _panel_signature(arr: np.ndarray) -> bytes:
    return arr.tobytes() + str(tuple(map(int, arr.shape))).encode()


def _touches(mask: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    if not np.any(mask):
        return False
    r0, c0, rh, rw = rect
    for r in range(r0, r0 + rh):
        for c in range(c0, c0 + rw):
            if (r > 0 and mask[r - 1, c]) or (r + 1 < mask.shape[0] and mask[r + 1, c]):
                return True
            if (c > 0 and mask[r, c - 1]) or (c + 1 < mask.shape[1] and mask[r, c + 1]):
                return True
    return False


def _rectangle_candidates(
    allowed: np.ndarray,
    *,
    min_area: int,
    max_area: int,
    max_h: int = 4,
    max_w: int = 4,
    must_touch: np.ndarray | None = None,
) -> list[tuple[int, int, int, int]]:
    h, w = map(int, allowed.shape)
    candidates: list[tuple[int, int, int, int]] = []
    for rh in range(1, min(max_h, h) + 1):
        for rw in range(1, min(max_w, w) + 1):
            area = rh * rw
            if area < min_area or area > max_area:
                continue
            for r0 in range(0, h - rh + 1):
                for c0 in range(0, w - rw + 1):
                    if not np.all(allowed[r0 : r0 + rh, c0 : c0 + rw]):
                        continue
                    rect = (r0, c0, rh, rw)
                    if must_touch is not None and np.any(must_touch) and not _touches(must_touch, rect):
                        continue
                    candidates.append(rect)
    return candidates


def _paint_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, rh, rw = rect
    mask[r0 : r0 + rh, c0 : c0 + rw] = True


def _build_mask(
    h: int,
    w: int,
    *,
    seed: np.ndarray | None,
    extra_allowed: np.ndarray,
    total_min: int,
    total_max: int,
    min_extra: int,
) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool) if seed is None else seed.copy()
    extra_allowed = np.asarray(extra_allowed, dtype=bool).copy()
    extra_allowed[mask] = False

    start = int(mask.sum())
    target_min = max(total_min, start + min_extra)
    target_max = max(target_min, total_max)
    if target_max > h * w:
        target_max = h * w
    if target_min > target_max:
        return None
    target = int(_RNG.integers(target_min, target_max + 1))

    if start == 0:
        first_candidates = _rectangle_candidates(
            extra_allowed, min_area=4, max_area=min(9, target), max_h=4, max_w=4
        )
        if not first_candidates:
            return None
        rect = first_candidates[int(_RNG.integers(len(first_candidates)))]
        _paint_rect(mask, rect)
        extra_allowed[mask] = False

    for _ in range(8):
        if int(mask.sum()) >= target:
            break
        remaining = target - int(mask.sum())
        prefer_touch = np.any(mask) and _RNG.random() < 0.7
        candidates = _rectangle_candidates(
            extra_allowed,
            min_area=1,
            max_area=min(6, remaining),
            max_h=3,
            max_w=3,
            must_touch=mask if prefer_touch else None,
        )
        if not candidates and prefer_touch:
            candidates = _rectangle_candidates(
                extra_allowed, min_area=1, max_area=min(6, remaining), max_h=3, max_w=3
            )
        if not candidates:
            break
        rect = candidates[int(_RNG.integers(len(candidates)))]
        _paint_rect(mask, rect)
        extra_allowed[mask] = False

    size = int(mask.sum())
    if target_min <= size <= total_max:
        return mask
    return None


def _subrectangles(mask: np.ndarray, *, min_area: int, max_area: int) -> list[np.ndarray]:
    rects = _rectangle_candidates(mask, min_area=min_area, max_area=max_area, max_h=3, max_w=3)
    out: list[np.ndarray] = []
    for rect in rects:
        sub = np.zeros_like(mask, dtype=bool)
        _paint_rect(sub, rect)
        out.append(sub)
    return out


def _simulate_masks(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, tuple[bool, bool]]:
    canvas = left.copy()
    statuses: list[bool] = []
    for mask in (middle, right):
        conflict = mask & canvas
        if np.any(conflict):
            canvas = canvas.copy()
            canvas[conflict] = False
            statuses.append(False)
        else:
            canvas = canvas | mask
            statuses.append(True)
    return canvas, (statuses[0], statuses[1])


def _simulate_no_knockout(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    canvas = left.copy()
    for mask in (middle, right):
        if not np.any(mask & canvas):
            canvas = canvas | mask
    return canvas


def _simulate_original_left_reference(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    canvas = left.copy()
    original = left.copy()
    for mask in (middle, right):
        conflict = mask & original
        if np.any(conflict):
            canvas = canvas.copy()
            canvas[conflict] = False
        else:
            canvas = canvas | mask
    return canvas


def _choose_colors() -> tuple[int, int, int, int, int]:
    picks = _RNG.choice(np.array(_PALETTE, dtype=int), size=5, replace=False)
    return tuple(int(x) for x in picks)


def _choose_geometry() -> tuple[int, int]:
    h = int(_RNG.integers(6, 10))
    w = int(_RNG.integers(6, 9))
    return h, w


def _build_ar_case(h: int, w: int, *, right_extra_min: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(120):
        left = _build_mask(
            h,
            w,
            seed=None,
            extra_allowed=np.ones((h, w), dtype=bool),
            total_min=7,
            total_max=14,
            min_extra=0,
        )
        if left is None:
            continue

        middle = _build_mask(
            h,
            w,
            seed=None,
            extra_allowed=~left,
            total_min=6,
            total_max=12,
            min_extra=0,
        )
        if middle is None:
            continue

        overlaps = _subrectangles(middle, min_area=2, max_area=4)
        if not overlaps:
            continue
        overlap = overlaps[int(_RNG.integers(len(overlaps)))]
        if int(middle.sum()) <= int(overlap.sum()):
            continue

        right = _build_mask(
            h,
            w,
            seed=overlap,
            extra_allowed=~left,
            total_min=max(4, int(overlap.sum()) + right_extra_min),
            total_max=10,
            min_extra=right_extra_min,
        )
        if right is None:
            continue
        if np.any(right & left):
            continue

        result, statuses = _simulate_masks(left, middle, right)
        if statuses != (True, False):
            continue
        if np.array_equal(result, _simulate_no_knockout(left, middle, right)):
            continue
        if np.array_equal(result, _simulate_original_left_reference(left, middle, right)):
            continue
        return left, middle, right
    return None


def _build_ra_case(h: int, w: int, *, right_extra_min: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(120):
        left = _build_mask(
            h,
            w,
            seed=None,
            extra_allowed=np.ones((h, w), dtype=bool),
            total_min=7,
            total_max=14,
            min_extra=0,
        )
        if left is None:
            continue

        overlaps = _subrectangles(left, min_area=2, max_area=4)
        if not overlaps:
            continue
        overlap = overlaps[int(_RNG.integers(len(overlaps)))]

        middle = _build_mask(
            h,
            w,
            seed=overlap,
            extra_allowed=~left,
            total_min=int(overlap.sum()) + 2,
            total_max=12,
            min_extra=2,
        )
        if middle is None:
            continue

        right = _build_mask(
            h,
            w,
            seed=overlap,
            extra_allowed=~left,
            total_min=max(4, int(overlap.sum()) + right_extra_min),
            total_max=10,
            min_extra=right_extra_min,
        )
        if right is None:
            continue
        if np.any(right & (left & ~overlap)):
            continue

        result, statuses = _simulate_masks(left, middle, right)
        if statuses != (False, True):
            continue
        if np.array_equal(result, _simulate_no_knockout(left, middle, right)):
            continue
        if np.array_equal(result, _simulate_original_left_reference(left, middle, right)):
            continue
        return left, middle, right
    return None


def _build_masks_for_case(case_name: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if case_name == "AR_CORE":
        return _build_ar_case(h, w, right_extra_min=0)
    if case_name == "RA_CORE":
        return _build_ra_case(h, w, right_extra_min=0)
    if case_name == "AR_EXT":
        return _build_ar_case(h, w, right_extra_min=2)
    if case_name == "RA_EXT":
        return _build_ra_case(h, w, right_extra_min=2)
    raise ValueError(f"unknown case {case_name}")


def _assemble_grid(
    left: np.ndarray, middle: np.ndarray, right: np.ndarray, *, bg: int, sep: int, colors: tuple[int, int, int]
) -> np.ndarray:
    h, w = map(int, left.shape)
    out = np.full((h, 3 * w + 2), bg, dtype=int)
    out[:, w] = sep
    out[:, 2 * w + 1] = sep
    out[:, :w] = _mask_to_panel(left, bg, colors[0])
    out[:, w + 1 : 2 * w + 1] = _mask_to_panel(middle, bg, colors[1])
    out[:, 2 * w + 2 :] = _mask_to_panel(right, bg, colors[2])
    return out


def _parse_grid(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width >= 8 and (width - 2) % 3 == 0, "expected three equal panels plus two separator columns"

    panel_w = (width - 2) // 3
    sep1 = panel_w
    sep2 = 2 * panel_w + 1
    sep_color = int(g[0, sep1])
    assert sep_color == int(g[0, sep2]), "separator columns must share a color"
    assert np.all(g[:, sep1] == sep_color) and np.all(g[:, sep2] == sep_color), "separators must be solid columns"

    left = g[:, :panel_w]
    middle = g[:, sep1 + 1 : sep2]
    right = g[:, sep2 + 1 :]
    assert left.shape == middle.shape == right.shape == (height, panel_w)

    bg = _most_common_color(np.concatenate([left.ravel(), middle.ravel(), right.ravel()]))
    assert bg != sep_color, "separator color must differ from background"

    panel_fgs = []
    for panel in (left, middle, right):
        colors = sorted(set(map(int, panel.ravel())) - {bg})
        assert len(colors) == 1, "each panel must contain exactly one foreground color"
        panel_fgs.append(colors[0])
    assert len(set(panel_fgs)) == 3, "the three panel foreground colors must be distinct"
    assert sep_color not in panel_fgs, "separator color must not appear inside panels"
    return bg, sep_color, left, middle, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Try the center and right panels onto the left canvas in order, erasing prior foreground where a trial conflicts.

    Input structure:
    - Two solid same-color separator columns split the grid into three equal-width panels with a shared background color.
    - Each panel contains exactly one foreground color, used only as a positional mask against that background.
    - The three panel foreground colors are all distinct from one another and from the separator color.

    Rule:
    - Remove the separator columns and use the left panel as the initial canvas.
    - Process the middle panel and then the right panel: if a panel's foreground mask has no overlap with the canvas's current foreground, copy the whole panel onto the canvas in its own color.
    - If a panel overlaps the current canvas, reject the whole panel and erase every conflicting canvas cell back to background; then continue with the updated canvas.

    Key insight:
    - A rejected panel still changes the state by blasting holes in the canvas, so later acceptance or rejection must be judged against the already-modified canvas rather than against the original left panel.
    """
    bg, _sep, left, middle, right = _parse_grid(np.asarray(grid, dtype=int))
    canvas = left.copy()
    for panel in (middle, right):
        canvas, _accepted = _apply_trial(canvas, panel, bg)
    return canvas


def _validate_candidate(
    grid: np.ndarray,
    left_mask: np.ndarray,
    middle_mask: np.ndarray,
    right_mask: np.ndarray,
) -> bool:
    try:
        bg, _sep, left, middle, right = _parse_grid(grid)
    except AssertionError:
        return False

    if not np.array_equal(_bool_panel(left, bg), left_mask):
        return False
    if not np.array_equal(_bool_panel(middle, bg), middle_mask):
        return False
    if not np.array_equal(_bool_panel(right, bg), right_mask):
        return False

    oracle_mask, statuses = _simulate_masks(left_mask, middle_mask, right_mask)
    actual = main(grid)
    if not np.array_equal(_bool_panel(actual, bg), oracle_mask):
        return False
    if statuses not in {(True, False), (False, True)}:
        return False

    no_knockout = _simulate_no_knockout(left_mask, middle_mask, right_mask)
    if np.array_equal(oracle_mask, no_knockout):
        return False
    original_left_only = _simulate_original_left_reference(left_mask, middle_mask, right_mask)
    if np.array_equal(oracle_mask, original_left_only):
        return False

    output_sig = _panel_signature(actual)
    input_sig = _panel_signature(grid)
    if output_sig in _SEEN_OUTPUT_SIGS or input_sig in _SEEN_INPUT_SIGS:
        return False
    return True


def generate_input() -> np.ndarray:
    global _CASE_CURSOR

    for _ in range(400):
        case_name = _CASE_SEQUENCE[_CASE_CURSOR % len(_CASE_SEQUENCE)]
        _CASE_CURSOR += 1
        h, w = _choose_geometry()
        masks = _build_masks_for_case(case_name, h, w)
        if masks is None:
            continue
        left_mask, middle_mask, right_mask = masks

        bg, sep, left_color, middle_color, right_color = _choose_colors()
        grid = _assemble_grid(
            left_mask,
            middle_mask,
            right_mask,
            bg=bg,
            sep=sep,
            colors=(left_color, middle_color, right_color),
        )
        if not _validate_candidate(grid, left_mask, middle_mask, right_mask):
            continue

        _SEEN_INPUT_SIGS.add(_panel_signature(grid))
        _SEEN_OUTPUT_SIGS.add(_panel_signature(main(grid)))
        return grid

    raise RuntimeError("failed to generate a valid conflict-overlay panel sample")

from __future__ import annotations

import hashlib
import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_for_fable_10_027", arc_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _ARC_UTILS = _load_arc_utils()
    get_bg = _ARC_UTILS.get_bg
except Exception:  # pragma: no cover - fallback only matters if the toolbox path disappears
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[np.argmax(counts)])


def _shift_mask(mask: np.ndarray, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    cols = np.flatnonzero(mask)
    dst = cols + int(dc)
    keep = (0 <= dst) & (dst < mask.size)
    out[dst[keep]] = True
    return out


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(tuple(grid.shape)).encode(), usedforsecurity=False).hexdigest()


def _extract_palette(row: np.ndarray, bg: int) -> list[int]:
    width = int(row.size)
    k = 0
    while k < width and int(row[k]) != int(bg):
        k += 1
    assert 3 <= k < width, "palette row must contain a non-empty left prefix of colors followed by background"
    assert np.all(row[k:] == bg), "palette row must be left-aligned and padded by background"
    palette = [int(v) for v in row[:k]]
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return palette


def _row_single_color(row: np.ndarray, bg: int) -> int | None:
    colors = sorted(set(map(int, row.tolist())) - {int(bg)})
    assert len(colors) <= 1, "each canvas row must use at most one non-background color"
    return colors[0] if colors else None


def _visible_period_count(lower: np.ndarray, palette: list[int]) -> int:
    present = [bool(np.any(lower == color)) for color in palette]
    count = 0
    for flag in present:
        if flag:
            count += 1
        else:
            break
    assert 2 <= count < len(palette), "the input must show a strict prefix of at least two periods"
    assert not any(present[count:]), "only a prefix of the palette colors may appear in the canvas"
    return count


def _infer_period(lower: np.ndarray, palette: list[int], bg: int, visible_periods: int) -> int:
    rows0 = np.flatnonzero(np.any(lower == palette[0], axis=1))
    assert rows0.size > 0 and int(rows0[0]) == 0, "the first visible period must start immediately below the palette"
    assert np.array_equal(rows0, np.arange(rows0.size)), "the first period rows must be contiguous"
    period = int(rows0.size)
    assert period >= 1, "period height must be positive"

    for period_idx in range(visible_periods):
        expected_rows = np.arange(period_idx * period, (period_idx + 1) * period)
        actual_rows = np.flatnonzero(np.any(lower == palette[period_idx], axis=1))
        assert np.array_equal(actual_rows, expected_rows), "each visible color must occupy one contiguous period block"
        for row_idx in expected_rows:
            assert _row_single_color(lower[row_idx], bg) == palette[period_idx], "every visible row must use its period color"

    assert np.all(lower[visible_periods * period :] == bg), "all missing periods must be blank in the input"
    assert lower.shape[0] == len(palette) * period, "canvas height must fit exactly K periods"
    return period


def _all_matching_shifts(src: np.ndarray, dst: np.ndarray) -> list[int]:
    width = int(src.size)
    matches = []
    for dc in range(-width + 1, width):
        if np.array_equal(_shift_mask(src, dc), dst):
            matches.append(dc)
    return matches


def _infer_shift(lower: np.ndarray, palette: list[int], period: int, visible_periods: int) -> int:
    shift = None
    for period_idx in range(visible_periods - 1):
        for offset in range(period):
            src = lower[period_idx * period + offset] == palette[period_idx]
            dst = lower[(period_idx + 1) * period + offset] == palette[period_idx + 1]
            assert np.any(src) and np.any(dst), "every visible row mask must be non-empty"
            candidates = _all_matching_shifts(src, dst)
            assert len(candidates) == 1, "visible periods must determine a unique horizontal shift"
            if shift is None:
                shift = candidates[0]
            else:
                assert shift == candidates[0], "all matching row pairs must agree on the same shift"
    assert shift is not None, "at least two visible periods are required to infer the shift"
    return int(shift)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[int], int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    palette = _extract_palette(g[0], bg)
    lower = g[1:]
    visible_periods = _visible_period_count(lower, palette)
    period = _infer_period(lower, palette, bg, visible_periods)
    shift = _infer_shift(lower, palette, period, visible_periods)
    return bg, palette, visible_periods, period, shift


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the translated row-period pattern to the bottom while consuming the palette colors in order.

    Input structure:
    - The top row is a left-aligned palette: K distinct colors followed by background.
    - The rows below are a canvas whose height fits exactly K vertical periods of size p.
    - The first few periods are already drawn: each period is a p-row block, every row in
      that block uses one palette color, and the next period is the previous one shifted
      horizontally by a fixed offset.

    Rule:
    - Keep the palette row and every already drawn period unchanged.
    - Infer the visible period height p, the number of shown periods, and the unique
      horizontal shift between consecutive periods from the existing canvas rows.
    - Fill each missing period by shifting the previous period's row masks by the same
      offset, clipping cells that leave the grid, and recoloring the new period with the
      next unused palette color.

    Key insight:
    - The palette is an ordered color program, not a reusable color set: geometry repeats
      by periodic translation, while color advances exactly once per period.
    """
    g = np.asarray(grid, dtype=int)
    bg, palette, visible_periods, period, shift = _parse_scene(g)
    out = g.copy()
    width = int(g.shape[1])

    for period_idx in range(visible_periods, len(palette)):
        new_color = palette[period_idx]
        prev_start = 1 + (period_idx - 1) * period
        curr_start = 1 + period_idx * period
        for offset in range(period):
            src_row = out[prev_start + offset]
            dst_mask = _shift_mask(src_row != bg, shift)
            out[curr_start + offset] = bg
            out[curr_start + offset, np.flatnonzero(dst_mask)] = new_color

    assert out.shape[1] == width
    return out


_GEN_RETRIES = 500
_GEN_BG_CHOICES = tuple(range(10))
_GEN_K_RANGE = (3, 6)
_GEN_P_RANGE = (1, 4)
_GEN_W_RANGE = (8, 18)
_GEN_SHIFT_RANGE = range(-3, 4)
_GEN_BASE_ROW_ATTEMPTS = 100
_GEN_SEEN_OUTPUT_SIGS: set[str] = set()
_GEN_SEEN_INPUT_SIGS: set[str] = set()


def _gen_pick_dimensions() -> tuple[int, int, int]:
    for _ in range(100):
        k = random.randint(*_GEN_K_RANGE)
        period = random.randint(*_GEN_P_RANGE)
        height = 1 + k * period
        if height <= 30:
            min_width = max(_GEN_W_RANGE[0], k + 2)
            width = random.randint(min_width, _GEN_W_RANGE[1])
            return k, period, width
    raise RuntimeError("failed to sample valid grid dimensions")


def _gen_pick_shift(width: int, periods: int, period: int) -> int:
    choices = [dc for dc in _GEN_SHIFT_RANGE if abs(dc) * (periods - 1) <= width - 2]
    if period == 1:
        choices = [dc for dc in choices if dc != 0]
    weights = [3 if dc != 0 else 1 for dc in choices]
    return int(random.choices(choices, weights=weights, k=1)[0])


def _gen_anchor_bounds(width: int, periods: int, shift: int) -> tuple[int, int]:
    if shift >= 0:
        return 0, width - 1 - (periods - 1) * shift
    return (periods - 1) * (-shift), width - 1


def _gen_sample_row_mask(width: int, periods: int, shift: int) -> np.ndarray:
    mask = np.zeros(width, dtype=bool)
    safe_lo, safe_hi = _gen_anchor_bounds(width, periods, shift)
    anchor = random.randint(safe_lo, safe_hi)
    mode = random.choice(("run", "scatter", "double"))

    if mode == "run":
        length = random.randint(1, min(5, width))
        start = random.randint(max(0, anchor - length + 1), min(anchor, width - length))
        mask[start : start + length] = True
    elif mode == "double" and width >= 5:
        length1 = random.randint(1, min(3, width))
        start1 = random.randint(max(0, anchor - length1 + 1), min(anchor, width - length1))
        mask[start1 : start1 + length1] = True
        gap_start = start1 + length1 + 1
        if shift < 0:
            gap_end = max(0, start1 - 1)
            side_pool = list(range(0, gap_end))
        else:
            side_pool = list(range(gap_start, width))
        if side_pool:
            start2 = random.choice(side_pool)
            length2 = random.randint(1, min(2, width - start2))
            mask[start2 : start2 + length2] = True
    else:
        mask[anchor] = True
        pool = list(range(width))
        pool.remove(anchor)
        extra = random.randint(0, min(4, len(pool)))
        for col in random.sample(pool, extra):
            mask[col] = True

    mask[anchor] = True
    return mask


def _gen_build_masks(base_rows: list[np.ndarray], periods: int, shift: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for period_idx in range(periods):
        for base in base_rows:
            masks.append(_shift_mask(base, period_idx * shift))
    return masks


def _gen_all_rows_nonempty(masks: list[np.ndarray]) -> bool:
    return all(bool(np.any(mask)) for mask in masks)


def _gen_has_clipping(base_rows: list[np.ndarray], periods: int, shift: int) -> bool:
    if shift == 0:
        return False
    for base in base_rows:
        full = int(np.count_nonzero(base))
        for period_idx in range(1, periods):
            shifted = _shift_mask(base, period_idx * shift)
            if int(np.count_nonzero(shifted)) < full:
                return True
    return False


def _gen_minimal_visible_period(visible_masks: list[np.ndarray], period: int) -> bool:
    n_rows = len(visible_masks)
    width = int(visible_masks[0].size)
    for step in range(1, period):
        for dc in range(-width + 1, width):
            if all(np.array_equal(_shift_mask(visible_masks[idx], dc), visible_masks[idx + step]) for idx in range(n_rows - step)):
                return False
    return True


def _gen_unique_shift_between_periods(visible_masks: list[np.ndarray], period: int, visible_periods: int, shift: int) -> bool:
    for period_idx in range(visible_periods - 1):
        for offset in range(period):
            src = visible_masks[period_idx * period + offset]
            dst = visible_masks[(period_idx + 1) * period + offset]
            if _all_matching_shifts(src, dst) != [shift]:
                return False
    return True


def _gen_rows_are_distinct_enough(base_rows: list[np.ndarray], shift: int) -> bool:
    if len(base_rows) == 1:
        return True
    return any(not np.array_equal(base_rows[idx], _shift_mask(base_rows[idx - 1], shift)) for idx in range(1, len(base_rows)))


def _gen_render_output(
    bg: int,
    palette: list[int],
    period: int,
    width: int,
    shift: int,
    base_rows: list[np.ndarray],
) -> np.ndarray:
    out = np.full((1 + len(palette) * period, width), bg, dtype=int)
    out[0, : len(palette)] = np.array(palette, dtype=int)
    masks = _gen_build_masks(base_rows, len(palette), shift)
    for row_idx, mask in enumerate(masks, start=1):
        period_idx = (row_idx - 1) // period
        out[row_idx, mask] = palette[period_idx]
    return out


def _gen_make_input_from_output(out: np.ndarray, bg: int, period: int, visible_periods: int) -> np.ndarray:
    inp = out.copy()
    inp[1 + visible_periods * period :] = bg
    return inp


def _gen_semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    palette: list[int],
    visible_periods: int,
    period: int,
    shift: int,
    expect_clipping: bool,
) -> bool:
    lower_out = out[1:]
    lower_in = inp[1:]
    visible_masks = [lower_out[row_idx] == palette[row_idx // period] for row_idx in range(visible_periods * period)]

    if not _gen_minimal_visible_period(visible_masks, period):
        return False
    if not _gen_unique_shift_between_periods(visible_masks, period, visible_periods, shift):
        return False
    if not np.all(lower_in[visible_periods * period :] == int(get_bg(inp))):
        return False
    if expect_clipping and shift != 0:
        output_masks = [lower_out[row_idx] == palette[row_idx // period] for row_idx in range(lower_out.shape[0])]
        original = [output_masks[idx] for idx in range(period)]
        clipped = any(int(np.count_nonzero(output_masks[idx])) < int(np.count_nonzero(original[idx % period])) for idx in range(period, len(output_masks)))
        if not clipped:
            return False

    try:
        parsed_bg, parsed_palette, parsed_visible, parsed_period, parsed_shift = _parse_scene(inp)
    except AssertionError:
        return False
    if parsed_bg != int(get_bg(inp)):
        return False
    if parsed_palette != palette or parsed_visible != visible_periods or parsed_period != period or parsed_shift != shift:
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    periods, period, width = _gen_pick_dimensions()
    shift = _gen_pick_shift(width, periods, period)
    bg = random.choice(_GEN_BG_CHOICES)
    palette = random.sample([color for color in range(10) if color != bg], periods)
    visible_periods = random.randint(2, periods - 1)

    want_clipping = shift != 0 and random.random() < 0.55
    for _ in range(_GEN_BASE_ROW_ATTEMPTS):
        base_rows = [_gen_sample_row_mask(width, periods, shift) for _ in range(period)]
        if len({tuple(mask.astype(int).tolist()) for mask in base_rows}) != len(base_rows):
            continue
        if not _gen_rows_are_distinct_enough(base_rows, shift):
            continue

        masks = _gen_build_masks(base_rows, periods, shift)
        if not _gen_all_rows_nonempty(masks):
            continue
        has_clipping = _gen_has_clipping(base_rows, periods, shift)
        if want_clipping and not has_clipping:
            continue

        out = _gen_render_output(bg, palette, period, width, shift, base_rows)
        inp = _gen_make_input_from_output(out, bg, period, visible_periods)
        if not _gen_semantic_checks(inp, out, palette, visible_periods, period, shift, want_clipping):
            continue

        input_sig = _grid_signature(inp)
        output_sig = _grid_signature(out)
        if input_sig in _GEN_SEEN_INPUT_SIGS or output_sig in _GEN_SEEN_OUTPUT_SIGS:
            continue
        _GEN_SEEN_INPUT_SIGS.add(input_sig)
        _GEN_SEEN_OUTPUT_SIGS.add(output_sig)
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-translation input")

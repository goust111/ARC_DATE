from __future__ import annotations

from collections import Counter
import hashlib
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in set(map(int, grid.ravel())) if int(v) != bg)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _axis_scores(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    pts = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    scores = [0] * w
    for axis in range(w):
        score = 0
        for r, c in pts:
            mc = 2 * axis - c
            if 0 <= mc < w and (r, mc) in pts:
                score += 1
        scores[axis] = int(score)
    return scores


def _best_axis(mask: np.ndarray) -> tuple[int, int, int]:
    scores = _axis_scores(mask)
    best = max(scores)
    axes = [axis for axis, score in enumerate(scores) if score == best]
    assert len(axes) == 1, "best symmetry axis must be unique"
    ranked = sorted(scores, reverse=True)
    second = ranked[1] if len(ranked) > 1 else -1
    return int(axes[0]), int(best), int(second)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each noisy symmetric color layer into its own full-height vertical axis line.

    Input structure:
    - The grid has a dominant background plus several non-background colors.
    - Each non-background color forms a shape that is almost mirror-symmetric about one vertical grid column, with a few same-color noise pixels breaking the symmetry.
    - Different colors use different vertical symmetry-axis columns.

    Rule:
    - Process each non-background color independently.
    - Among all grid columns, find the unique column whose vertical mirror gives that color the highest same-color mirror-pair score.
    - Erase every original pixel of that color, then draw that color as a single vertical line covering the full grid height at its winning axis column.
    - All other cells become background.

    Key insight:
    - The visible symmetric patterns are only clues for locating hidden per-color axes; the answer is the set of axes themselves, not the cleaned patterns.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    axes_by_color: dict[int, int] = {}
    for color in _colors_present(g, bg):
        mask = g == int(color)
        assert mask.any(), "each non-background color must have pixels"
        axis, _best, _second = _best_axis(mask)
        axes_by_color[int(color)] = int(axis)

    axis_values = list(axes_by_color.values())
    assert len(axis_values) == len(set(axis_values)), "different colors must have distinct axes"

    out = np.full((h, w), int(bg), dtype=int)
    for color, axis in axes_by_color.items():
        out[:, int(axis)] = int(color)
    return out


# --------------- generator ---------------

BG_COLORS = tuple(range(10))
MIN_COLORS = 2
MAX_COLORS = 4
MIN_AXIS_GAP = 3
MIN_AXIS_MARGIN = 2
MIN_BAND_HEIGHT = 4
MAX_BAND_HEIGHT = 8
MIN_TOTAL_PIXELS = 6
MAX_TOTAL_PIXELS = 28
AXIS_GAP_MIN = 2
GEN_RETRIES = 800

_SEEN_INPUTS: set[str] = set()
_SEEN_OUTPUTS: set[str] = set()


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _sample_axes(width: int, count: int) -> list[int] | None:
    candidates = list(range(MIN_AXIS_MARGIN, width - MIN_AXIS_MARGIN))
    random.shuffle(candidates)
    axes: list[int] = []
    for axis in candidates:
        if all(abs(axis - other) >= MIN_AXIS_GAP for other in axes):
            axes.append(int(axis))
            if len(axes) == count:
                return sorted(axes)
    return None


def _sample_bands(count: int) -> tuple[int, list[tuple[int, int]]] | None:
    heights = [random.randint(MIN_BAND_HEIGHT, MAX_BAND_HEIGHT) for _ in range(count)]
    gaps = [random.randint(1, 2) for _ in range(count - 1)]
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    height = top_margin + bottom_margin + sum(heights) + sum(gaps)
    if height > 30:
        return None

    bands: list[tuple[int, int]] = []
    r0 = top_margin
    for i, band_h in enumerate(heights):
        bands.append((int(r0), int(r0 + band_h - 1)))
        r0 += band_h
        if i < len(gaps):
            r0 += gaps[i]
    return int(height), bands


def _paint_mirror_pair(mask: np.ndarray, axis: int, r: int, c: int) -> None:
    mask[int(r), int(c)] = True
    mask[int(r), int(2 * axis - c)] = True


def _build_clean_mask(
    height: int,
    width: int,
    band: tuple[int, int],
    axis: int,
    want_centerless: bool,
    want_multicomponent: bool,
) -> np.ndarray | None:
    r0, r1 = band
    band_h = r1 - r0 + 1
    max_span = min(axis, width - 1 - axis, 6)
    if max_span < 1:
        return None

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        span = random.randint(1, max_span)
        style = random.choice(("dots", "bars"))

        if style == "dots":
            available = [(r, c) for r in range(r0, r1 + 1) for c in range(axis - span, axis)]
            left_count = random.randint(max(2, band_h // 2), min(len(available), band_h * span))
            for r, c in random.sample(available, left_count):
                _paint_mirror_pair(mask, axis, r, c)
        else:
            rect_count = random.randint(2, 4)
            for _ in range(rect_count):
                rect_h = random.randint(1, max(1, min(3, band_h)))
                rect_w = random.randint(1, span)
                rr0 = random.randint(r0, r1 - rect_h + 1)
                cc0 = random.randint(axis - span, axis - rect_w)
                mask[rr0 : rr0 + rect_h, cc0 : cc0 + rect_w] = True
                mirror_l = 2 * axis - (cc0 + rect_w - 1)
                mask[rr0 : rr0 + rect_h, mirror_l : mirror_l + rect_w] = True

        if not want_centerless and random.random() < 0.65:
            center_rows = random.sample(range(r0, r1 + 1), random.randint(1, max(1, band_h // 2)))
            mask[center_rows, axis] = True

        area = int(mask.sum())
        if not (MIN_TOTAL_PIXELS <= area <= MAX_TOTAL_PIXELS):
            continue
        if want_multicomponent and _components_count(mask) < 2:
            continue

        best_axis, best_score, second_score = _best_axis(mask)
        if best_axis != axis or best_score - second_score < AXIS_GAP_MIN:
            continue
        if want_centerless and np.any(mask[:, axis]):
            continue
        return mask
    return None


def _try_add_noise(
    mask: np.ndarray,
    band: tuple[int, int],
    axis: int,
    force_bbox_skew: bool,
) -> bool:
    height, width = mask.shape
    r0, r1 = band
    current_bbox = _bbox(mask)
    if current_bbox is None:
        return False
    _br0, bc0, _br1, bc1 = current_bbox

    sides = ["left", "right"]
    if force_bbox_skew:
        sides = ["left"] if axis - bc0 <= bc1 - axis else ["right"]

    candidates: list[tuple[int, int]] = []
    for side in sides:
        if side == "left":
            cols = range(0, axis)
        else:
            cols = range(axis + 1, width)
        for r in range(r0, r1 + 1):
            for c in cols:
                if mask[r, c]:
                    continue
                mc = 2 * axis - c
                if 0 <= mc < width and mask[r, mc]:
                    continue
                if force_bbox_skew:
                    if side == "left" and c >= bc0:
                        continue
                    if side == "right" and c <= bc1:
                        continue
                candidates.append((int(r), int(c)))

    random.shuffle(candidates)
    for r, c in candidates:
        trial = mask.copy()
        trial[r, c] = True
        try:
            best_axis, best_score, second_score = _best_axis(trial)
        except AssertionError:
            continue
        if best_axis == axis and best_score - second_score >= AXIS_GAP_MIN:
            mask[:, :] = trial
            return True
    return False


def _try_remove_pair_mate(mask: np.ndarray, axis: int) -> bool:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(coords)
    for r, c in coords:
        if c == axis:
            continue
        mc = 2 * axis - c
        if not (0 <= mc < mask.shape[1] and mask[r, mc]):
            continue
        trial = mask.copy()
        trial[r, c] = False
        if int(trial.sum()) < MIN_TOTAL_PIXELS:
            continue
        try:
            best_axis, best_score, second_score = _best_axis(trial)
        except AssertionError:
            continue
        if best_axis == axis and best_score - second_score >= AXIS_GAP_MIN:
            mask[:, :] = trial
            return True
    return False


def _has_unpaired_pixel(mask: np.ndarray, axis: int) -> bool:
    width = mask.shape[1]
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        mc = 2 * axis - c
        if not (0 <= mc < width and mask[r, mc]):
            return True
    return False


def _densest_column(mask: np.ndarray) -> int:
    return int(np.argmax(mask.sum(axis=0)))


def _render_expected(
    shape: tuple[int, int],
    bg: int,
    specs: list[dict[str, int]],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        out[:, int(spec["axis"])] = int(spec["color"])
    return out


def _compose_grid(shape: tuple[int, int], bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        mask = spec["mask"]
        assert isinstance(mask, np.ndarray)
        assert not np.any((grid != bg) & mask)
        grid[mask] = color
    return grid


def _build_color_mask(
    height: int,
    width: int,
    band: tuple[int, int],
    axis: int,
    require_centerless: bool,
    require_multicomponent: bool,
    require_bbox_skew: bool,
) -> np.ndarray | None:
    for _ in range(250):
        mask = _build_clean_mask(
            height,
            width,
            band,
            axis,
            want_centerless=require_centerless,
            want_multicomponent=require_multicomponent,
        )
        if mask is None:
            continue

        noise_added = _try_add_noise(mask, band, axis, force_bbox_skew=require_bbox_skew)
        if not noise_added:
            continue
        if random.random() < 0.5:
            _try_remove_pair_mate(mask, axis)
        if random.random() < 0.55:
            _try_add_noise(mask, band, axis, force_bbox_skew=False)

        try:
            best_axis, best_score, second_score = _best_axis(mask)
        except AssertionError:
            continue
        if best_axis != axis or best_score - second_score < AXIS_GAP_MIN:
            continue
        if not _has_unpaired_pixel(mask, axis):
            continue
        if require_centerless and np.any(mask[:, axis]):
            continue
        if require_multicomponent and _components_count(mask) < 2:
            continue
        if require_bbox_skew:
            box = _bbox(mask)
            assert box is not None
            if box[1] + box[3] == 2 * axis:
                continue
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(MIN_COLORS, MAX_COLORS)
    bg = random.choice(BG_COLORS)
    colors = random.sample([c for c in range(10) if c != bg], color_count)

    bands_data = _sample_bands(color_count)
    if bands_data is None:
        return None
    height, bands = bands_data

    min_width = max(10, MIN_AXIS_MARGIN * 2 + 1, color_count * MIN_AXIS_GAP + 4)
    width = random.randint(min_width, 30)
    axes = _sample_axes(width, color_count)
    if axes is None:
        return None

    required = {
        "centerless": random.randrange(color_count),
        "multicomponent": random.randrange(color_count),
        "bbox_skew": random.randrange(color_count),
    }

    specs: list[dict[str, object]] = []
    for idx, (color, axis, band) in enumerate(zip(colors, axes, bands)):
        mask = _build_color_mask(
            height,
            width,
            band,
            axis,
            require_centerless=(idx == required["centerless"]),
            require_multicomponent=(idx == required["multicomponent"]),
            require_bbox_skew=(idx == required["bbox_skew"]),
        )
        if mask is None:
            return None
        specs.append({"color": int(color), "axis": int(axis), "mask": mask})

    grid = _compose_grid((height, width), bg, specs)
    expected = _render_expected((height, width), bg, specs)

    if not np.array_equal(main(grid), expected):
        return None

    centerless_ok = any(not np.any(spec["mask"][:, int(spec["axis"])]) for spec in specs)
    bbox_skew_ok = any(
        ((box := _bbox(spec["mask"])) is not None) and (box[1] + box[3] != 2 * int(spec["axis"]))
        for spec in specs
    )
    multicomp_ok = any(_components_count(spec["mask"]) >= 2 for spec in specs)
    densest_shortcut_ok = any(_densest_column(spec["mask"]) != int(spec["axis"]) for spec in specs)
    if not (centerless_ok and bbox_skew_ok and multicomp_ok and densest_shortcut_ok):
        return None

    for spec in specs:
        axis, best_score, second_score = _best_axis(spec["mask"])
        if axis != int(spec["axis"]) or best_score - second_score < AXIS_GAP_MIN:
            return None

    input_sig = _sig(grid)
    output_sig = _sig(expected)
    if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(input_sig)
    _SEEN_OUTPUTS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")

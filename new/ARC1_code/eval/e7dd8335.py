from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, find_components, get_bg
import itertools as _gen_itertools


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the lower half of the foreground shape red.

    Input structure:
    - The grid has one background color and one non-background foreground color.
    - The foreground pixels share one bounding box whose height is even.
    - The foreground shape occupies both the upper and lower halves of that box.

    Rule:
    - Preserve the background and the foreground pixels in the upper half of the bounding box.
    - Recolor every foreground pixel in the lower half of the bounding box to red.

    Key insight:
    - The red/blue split is positional within the object's bounding box: the horizontal midpoint of the shape, not component identity, decides which pixels change.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"

    fg_mask = g != bg
    fg_bbox = bbox(fg_mask)
    assert fg_bbox is not None, "expected foreground pixels"
    r0, _c0, r1, _c1 = fg_bbox
    height = r1 - r0 + 1
    assert height % 2 == 0, "foreground bounding box height must split evenly"

    split_row = r0 + height // 2
    rows = np.arange(g.shape[0])[:, None]
    lower_half = rows >= split_row
    lower_foreground = fg_mask & lower_half
    upper_foreground = fg_mask & ~lower_half
    assert np.any(upper_foreground), "expected foreground pixels in the upper half"
    assert np.any(lower_foreground), "expected foreground pixels in the lower half"

    out = g.copy()
    out[lower_foreground] = Color.RED
    return out

# --------------- generator (o5x) ---------------

# Canvas, bbox, and pixel-count constraints.
_GEN_GRID_MIN, _GEN_GRID_MAX = 7, 11
_GEN_BBOX_HEIGHTS = (6, 8)
_GEN_BBOX_W_MIN, _GEN_BBOX_W_MAX = 4, 10
_GEN_FG_PIXELS_MIN, _GEN_FG_PIXELS_MAX = 16, 22
_GEN_TOP_PIXELS_MIN = _GEN_FG_PIXELS_MIN // 2
_GEN_TOP_PIXELS_MAX = _GEN_FG_PIXELS_MAX // 2

# Outline construction limits.
_GEN_ENDPOINT_DRIFT = 1
_GEN_MIN_BOTTOM_RUN_TOTAL, _GEN_MIN_SIDE_RUN, _GEN_MIN_CENTER_GAP = 2, 1, 1
_GEN_SIDE_POINTS_PER_MID_ROW = 2

# Sampling preferences and retry budget.
_GEN_LR_SYMMETRY_RATE = 0.7
_GEN_ATTEMPT_LIMIT = 300

_gen_MASK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _gen_has_interior_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    if not bg.any():
        return False

    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any(bg & ~seen))


def _gen_valid_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h not in _GEN_BBOX_HEIGHTS or not (_GEN_BBOX_W_MIN <= w <= _GEN_BBOX_W_MAX):
        return False
    if not np.array_equal(mask, np.flipud(mask)):
        return False
    if bbox(mask) != (0, 0, h - 1, w - 1):
        return False
    pixels = int(mask.sum())
    if not (_GEN_FG_PIXELS_MIN <= pixels <= _GEN_FG_PIXELS_MAX):
        return False
    split = h // 2
    if not mask[:split].any() or not mask[split:].any():
        return False
    comps = find_components(
        mask.astype(int), bg=0, connectivity=8, monochromatic=True
    )
    return len(comps) == 1 and _gen_has_interior_hole(mask)


def _gen_mirror_top(top: np.ndarray) -> np.ndarray:
    return np.vstack([top, top[::-1]])


def _gen_unique_masks(candidates: list[np.ndarray]) -> list[np.ndarray]:
    seen = set()
    out = []
    for mask in candidates:
        key = mask.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(mask)
    return out


def _gen_endpoint_sequences(
    n: int,
    start_lo: int,
    start_hi: int,
    w: int,
    *,
    last_max: int | None = None,
    last_min: int | None = None,
) -> list[tuple[int, ...]]:
    seqs: list[tuple[int, ...]] = []

    def rec(prefix: list[int]) -> None:
        if len(prefix) == n:
            last = prefix[-1]
            if last_max is not None and last > last_max:
                return
            if last_min is not None and last < last_min:
                return
            seqs.append(tuple(prefix))
            return
        if prefix:
            lo = max(0, prefix[-1] - _GEN_ENDPOINT_DRIFT)
            hi = min(w - 1, prefix[-1] + _GEN_ENDPOINT_DRIFT)
        else:
            lo = max(0, start_lo)
            hi = min(w - 1, start_hi)
        for c in range(lo, hi + 1):
            rec(prefix + [c])

    rec([])
    return seqs


def _gen_angular_masks(h: int, w: int) -> list[np.ndarray]:
    k = h // 2
    mid_rows = k - 2
    candidates: list[np.ndarray] = []
    for target_top in range(_GEN_TOP_PIXELS_MIN, _GEN_TOP_PIXELS_MAX + 1):
        for cap_l in range(w):
            for cap_r in range(cap_l, w):
                cap_len = cap_r - cap_l + 1
                lower_count = target_top - cap_len - _GEN_SIDE_POINTS_PER_MID_ROW * mid_rows
                if lower_count < _GEN_MIN_BOTTOM_RUN_TOTAL or lower_count > w - _GEN_MIN_CENTER_GAP:
                    continue
                for left_run in range(_GEN_MIN_SIDE_RUN, lower_count):
                    right_run = lower_count - left_run
                    if right_run < _GEN_MIN_SIDE_RUN or left_run + right_run > w - _GEN_MIN_CENTER_GAP:
                        continue
                    left_seqs = _gen_endpoint_sequences(
                        mid_rows,
                        cap_l - _GEN_ENDPOINT_DRIFT,
                        cap_l + _GEN_ENDPOINT_DRIFT,
                        w,
                        last_max=left_run,
                    )
                    right_seqs = _gen_endpoint_sequences(
                        mid_rows,
                        cap_r - _GEN_ENDPOINT_DRIFT,
                        cap_r + _GEN_ENDPOINT_DRIFT,
                        w,
                        last_min=w - right_run - _GEN_MIN_CENTER_GAP,
                    )
                    for left_seq in left_seqs:
                        for right_seq in right_seqs:
                            if any(l >= r for l, r in zip(left_seq, right_seq)):
                                continue
                            top = np.zeros((k, w), dtype=bool)
                            top[0, cap_l : cap_r + 1] = True
                            for r, (left_c, right_c) in enumerate(
                                zip(left_seq, right_seq), start=1
                            ):
                                top[r, left_c] = True
                                top[r, right_c] = True
                            top[k - 1, :left_run] = True
                            top[k - 1, w - right_run :] = True
                            mask = _gen_mirror_top(top)
                            if (
                                int(mask[:k].sum()) == target_top
                                and _gen_valid_mask(mask)
                            ):
                                candidates.append(mask)
    return _gen_unique_masks(candidates)


def _gen_frame_masks(h: int, w: int) -> list[np.ndarray]:
    k = h // 2
    base = np.zeros((h, w), dtype=bool)
    base[0, :] = True
    base[-1, :] = True
    base[:, 0] = True
    base[:, -1] = True
    base_top_count = int(base[:k].sum())
    top_choices = [(r, c) for r in range(1, k) for c in range(1, w - 1)]

    candidates: list[np.ndarray] = []
    for target_top in range(_GEN_TOP_PIXELS_MIN, _GEN_TOP_PIXELS_MAX + 1):
        extra = target_top - base_top_count
        if extra < 0 or extra > len(top_choices):
            continue
        for combo in _gen_itertools.combinations(top_choices, extra):
            top = base[:k].copy()
            for r, c in combo:
                top[r, c] = True
            mask = _gen_mirror_top(top)
            if _gen_valid_mask(mask):
                candidates.append(mask)
    return _gen_unique_masks(candidates)


def _gen_outline_masks(h: int, w: int) -> list[np.ndarray]:
    key = (h, w)
    if key not in _gen_MASK_CACHE:
        _gen_MASK_CACHE[key] = _gen_unique_masks(
            _gen_angular_masks(h, w) + _gen_frame_masks(h, w)
        )
    return _gen_MASK_CACHE[key]


def _gen_canvas_size() -> tuple[int, int]:
    return (
        int(np.random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX + 1)),
        int(np.random.randint(_GEN_GRID_MIN, _GEN_GRID_MAX + 1)),
    )


def _gen_group_masks_for_canvas(
    rows: int, cols: int
) -> dict[int, dict[tuple[int, int], list[np.ndarray]]]:
    bbox_h_options = [height for height in _GEN_BBOX_HEIGHTS if height <= rows]
    groups: dict[int, dict[tuple[int, int], list[np.ndarray]]] = {}
    for bbox_h in bbox_h_options:
        for bbox_w in range(_GEN_BBOX_W_MIN, min(_GEN_BBOX_W_MAX, cols) + 1):
            for mask in _gen_outline_masks(bbox_h, bbox_w):
                groups.setdefault(int(mask.sum()), {}).setdefault(
                    (bbox_h, bbox_w), []
                ).append(mask)
    return groups


def _gen_choose_grouped_mask(
    groups: dict[int, dict[tuple[int, int], list[np.ndarray]]]
) -> np.ndarray | None:
    if not groups:
        return None
    pixel_count = int(np.random.choice(sorted(groups)))
    dim_groups = groups[pixel_count]
    dims = list(dim_groups)
    bbox_h, bbox_w = dims[int(np.random.randint(len(dims)))]
    options = dim_groups[(bbox_h, bbox_w)]
    lr_options = [mask for mask in options if np.array_equal(mask, np.fliplr(mask))]
    if lr_options and np.random.random() < _GEN_LR_SYMMETRY_RATE:
        options = lr_options
    return options[int(np.random.randint(len(options)))]


def _gen_render_output(rows: int, cols: int, mask: np.ndarray) -> np.ndarray:
    bbox_h, bbox_w = mask.shape
    r0 = int(np.random.randint(0, rows - bbox_h + 1))
    c0 = int(np.random.randint(0, cols - bbox_w + 1))
    out = np.full((rows, cols), Color.BLACK, dtype=int)
    split = bbox_h // 2
    upper = mask.copy()
    upper[split:, :] = False
    lower = mask.copy()
    lower[:split, :] = False
    out[r0 : r0 + bbox_h, c0 : c0 + bbox_w][upper] = Color.BLUE
    out[r0 : r0 + bbox_h, c0 : c0 + bbox_w][lower] = Color.RED
    return out


def _gen_input_from_output(out: np.ndarray) -> np.ndarray:
    inp = out.copy()
    inp[inp == Color.RED] = Color.BLUE
    return inp


def _gen_attempt() -> np.ndarray | None:
    rows, cols = _gen_canvas_size()
    mask = _gen_choose_grouped_mask(_gen_group_masks_for_canvas(rows, cols))
    if mask is None:
        return None

    out = _gen_render_output(rows, cols, mask)
    inp = _gen_input_from_output(out)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e7dd8335 input")

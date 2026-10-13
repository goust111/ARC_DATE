from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BG = 8
TEMPLATE = 6
SEED = 2
H_BAR = 1
V_BAR = 3
MARKER = 4
FRAME = 5

TEMPLATE_SHAPES = ((3, 3), (3, 4))
TEMPLATE_COUNT_CHOICES = (2, 3)
SCALE_CHOICES = (2, 3)
GEN_ATTEMPTS = 400


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((int(factor), int(factor)), dtype=bool))


def _mask_touches_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(grid == FRAME, connectivity=4):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(coords.shape[0]) == int(border.sum()) and np.array_equal(sub == FRAME, border):
            candidates.append((int(coords.shape[0]), (r0, c0, r1, c1)))

    assert candidates, "expected one rectangular frame"
    max_area = max(area for area, _bbox in candidates)
    best = [bbox for area, bbox in candidates if area == max_area]
    assert len(best) == 1, "frame must be structurally unique"
    return best[0]


def _extract_templates(grid: np.ndarray, frame_r0: int) -> list[np.ndarray]:
    top = grid[:frame_r0]
    templates: list[tuple[int, np.ndarray]] = []
    for coords in _components(top == TEMPLATE, connectivity=8):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = top[r0 : r1 + 1, c0 : c1 + 1] == TEMPLATE
        if patch.shape not in TEMPLATE_SHAPES:
            continue
        if not _mask_touches_full_bbox(patch):
            continue
        templates.append((c0, patch))

    templates.sort(key=lambda item: item[0])
    out = [patch for _c0, patch in templates]
    assert 2 <= len(out) <= 3, "expected 2 or 3 upper templates"
    shape_set = {tuple(t.shape) for t in out}
    assert len(shape_set) == 1, "all templates must share one bbox size"
    return out


def _find_bar_row(grid: np.ndarray) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        non_bg = np.flatnonzero(row != BG)
        if non_bg.size == 0:
            continue
        if not np.all(np.diff(non_bg) == 1):
            continue
        colors = set(map(int, row[non_bg]))
        if not colors.issubset({H_BAR, V_BAR, MARKER}):
            continue
        marker_cols = np.flatnonzero(row == MARKER)
        if marker_cols.size != 2:
            continue
        if int(np.count_nonzero(row == H_BAR)) < 2:
            continue
        if int(np.count_nonzero(row == V_BAR)) > 1:
            continue
        candidates.append((r, marker_cols))
    assert len(candidates) == 1, "expected one horizontal marker bar"
    return int(candidates[0][0]), candidates[0][1]


def _find_bar_col(grid: np.ndarray) -> tuple[int, np.ndarray]:
    candidates: list[tuple[int, np.ndarray]] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        non_bg = np.flatnonzero(col != BG)
        if non_bg.size == 0:
            continue
        if not np.all(np.diff(non_bg) == 1):
            continue
        colors = set(map(int, col[non_bg]))
        if not colors.issubset({H_BAR, V_BAR, MARKER}):
            continue
        marker_rows = np.flatnonzero(col == MARKER)
        if marker_rows.size != 2:
            continue
        if int(np.count_nonzero(col == V_BAR)) < 2:
            continue
        if int(np.count_nonzero(col == H_BAR)) > 1:
            continue
        candidates.append((c, marker_rows))
    assert len(candidates) == 1, "expected one vertical marker bar"
    return int(candidates[0][0]), candidates[0][1]


def _weave_patch(mask: np.ndarray, dh: int, dv: int) -> np.ndarray:
    out = np.full(mask.shape, BG, dtype=int)
    for r, c in np.argwhere(mask):
        use_h = ((int(r) // int(dv)) % 2) == ((int(c) // int(dh)) % 2)
        out[int(r), int(c)] = H_BAR if use_h else V_BAR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the selected scaled template inside the frame with a two-color weave driven by marker spans.

    Input structure:
    - A teal background contains 2 or 3 small magenta template masks above a larger grey rectangular frame.
    - The frame interior contains red seed pixels that lie on a scaled copy of exactly one template mask.
    - Outside the frame, one blue horizontal bar and one green vertical bar cross, and each bar has two yellow marker pixels.

    Rule:
    - Find the unique rectangular frame and read its interior size.
    - Extract the upper templates, compute the integer scale from template bbox to frame interior, and choose the unique template whose scaled foreground covers every red seed pixel.
    - Let dh be the column distance between the two yellow markers on the horizontal bar, and let dv be the row distance between the two yellow markers on the vertical bar.
    - Clear the frame interior to teal, then for every foreground cell of the selected scaled template at interior coordinate (r, c), paint blue when the parity of r // dv matches the parity of c // dh, otherwise paint green.
    - Keep the templates, frame border, bars, and markers unchanged.

    Key insight:
    The red seeds determine where the large mask goes, while the yellow marker spans determine how that mask is woven with alternating horizontal-bar and vertical-bar colors.
    """
    g = np.asarray(grid, dtype=int)
    frame_r0, frame_c0, frame_r1, frame_c1 = _find_frame_bbox(g)
    templates = _extract_templates(g, frame_r0)

    interior = g[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1]
    seed_mask = interior == SEED
    assert seed_mask.any(), "frame must contain red seeds"

    ih, iw = interior.shape
    template_shape = templates[0].shape
    th, tw = template_shape
    assert ih % th == 0 and iw % tw == 0, "frame interior must be an integer scale of template size"
    scale_h = ih // th
    scale_w = iw // tw
    assert scale_h == scale_w, "template scaling must be isotropic"
    scale = int(scale_h)

    matches: list[np.ndarray] = []
    for template in templates:
        scaled = _scale_mask(template, scale)
        if np.all(~seed_mask | scaled):
            matches.append(scaled)
    assert len(matches) == 1, "seed pixels must select exactly one template"
    chosen = matches[0]

    _bar_r, h_markers = _find_bar_row(g)
    _bar_c, v_markers = _find_bar_col(g)
    dh = int(abs(int(h_markers[1]) - int(h_markers[0])))
    dv = int(abs(int(v_markers[1]) - int(v_markers[0])))
    assert dh > 0 and dv > 0, "marker pairs must have non-zero span"

    out = g.copy()
    out[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1] = BG
    woven = _weave_patch(chosen, dh, dv)
    fill_view = out[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1]
    fill_view[chosen] = woven[chosen]
    return out


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    return len(_components(mask, connectivity)) == 1


@lru_cache(maxsize=None)
def _template_pool(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    h, w = shape
    n = h * w
    lo = max(4, (n + 2) // 3)
    hi = min(n - 2, (3 * n) // 4 + 1)
    masks: list[np.ndarray] = []
    for bits in range(1 << n):
        area = int(bits.bit_count())
        if area < lo or area > hi:
            continue
        flat = np.array([(bits >> k) & 1 for k in range(n)], dtype=bool)
        mask = flat.reshape(h, w)
        if not _mask_touches_full_bbox(mask):
            continue
        if not _is_connected(mask, connectivity=8):
            continue
        masks.append(mask)
    return tuple(masks)


def _choose_templates(
    rng: random.Random, shape: tuple[int, int], count: int
) -> tuple[list[np.ndarray], int]:
    pool = list(_template_pool(shape))
    assert len(pool) >= count
    templates = [pool[idx].copy() for idx in rng.sample(range(len(pool)), count)]
    target_idx = rng.randrange(count)
    return templates, int(target_idx)


def _choose_seed_pixels(
    rng: random.Random, scaled_masks: list[np.ndarray], target_idx: int
) -> np.ndarray | None:
    target_pixels = np.argwhere(scaled_masks[target_idx])
    if target_pixels.shape[0] < 3:
        return None
    max_seed_count = min(6, target_pixels.shape[0] - 1)
    for _ in range(120):
        k = rng.randint(2, max_seed_count)
        pick = target_pixels[rng.sample(range(target_pixels.shape[0]), k)]
        cover = []
        for scaled in scaled_masks:
            ok = True
            for r, c in pick:
                if not scaled[int(r), int(c)]:
                    ok = False
                    break
            cover.append(ok)
        if sum(cover) == 1 and cover[target_idx]:
            return pick
    return None


def _choose_marker_pair(
    rng: random.Random, positions: list[int], blocked: set[int]
) -> tuple[int, int] | None:
    usable = [p for p in positions if p not in blocked]
    if len(usable) < 2:
        return None
    rng.shuffle(usable)
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            if abs(a - b) > 0:
                return int(a), int(b)
    return None


def _output_uses_both_colors(mask: np.ndarray, dh: int, dv: int) -> bool:
    woven = _weave_patch(mask, dh, dv)
    vals = set(map(int, woven[mask]))
    return vals == {H_BAR, V_BAR}


def _periods_are_semantically_active(mask: np.ndarray, dh: int, dv: int) -> bool:
    coords = np.argwhere(mask)
    row_bands = {int(r) // int(dv) for r, _c in coords}
    col_bands = {int(c) // int(dh) for _r, c in coords}
    if len(row_bands) < 2 or len(col_bands) < 2:
        return False

    base = _weave_patch(mask, dh, dv)
    alt_dh = 2 if dh == 1 else dh - 1
    alt_dv = 2 if dv == 1 else dv - 1
    if alt_dh > mask.shape[1]:
        alt_dh = dh + 1
    if alt_dv > mask.shape[0]:
        alt_dv = dv + 1
    if alt_dh > mask.shape[1] or alt_dv > mask.shape[0]:
        return False
    return not np.array_equal(base, _weave_patch(mask, alt_dh, dv)) and not np.array_equal(
        base, _weave_patch(mask, dh, alt_dv)
    )


def _expected_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    scaled_target: np.ndarray,
    dh: int,
    dv: int,
) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = frame_bbox
    out[r0 + 1 : r1, c0 + 1 : c1] = BG
    woven = _weave_patch(scaled_target, dh, dv)
    view = out[r0 + 1 : r1, c0 + 1 : c1]
    view[scaled_target] = woven[scaled_target]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_shape = rng.choice(TEMPLATE_SHAPES)
    n_templates = rng.choice(TEMPLATE_COUNT_CHOICES)
    scale = rng.choice(SCALE_CHOICES)

    templates, target_idx = _choose_templates(rng, template_shape, n_templates)
    th, tw = template_shape
    scaled_masks = [_scale_mask(mask, scale) for mask in templates]
    scaled_target = scaled_masks[target_idx]

    seed_pixels = _choose_seed_pixels(rng, scaled_masks, target_idx)
    if seed_pixels is None:
        return None

    ih, iw = scaled_target.shape
    frame_h = ih + 2
    frame_w = iw + 2

    template_gap = 2
    template_total_w = n_templates * tw + (n_templates - 1) * template_gap
    template_r0 = 1
    frame_r0 = template_r0 + th + rng.randint(4, 6)
    bottom_margin = rng.randint(2, 4)
    frame_c0 = rng.randint(1, 4)
    right_space = rng.randint(7, 10)

    height = frame_r0 + frame_h + bottom_margin
    width = max(template_total_w + 2, frame_c0 + frame_w + right_space)
    if height > 30 or width > 30:
        return None

    hbar_row_choices = list(range(template_r0 + th + 2, frame_r0 - 1))
    if not hbar_row_choices:
        return None
    hbar_row = rng.choice(hbar_row_choices)

    vbar_r0 = template_r0 + th + 1
    vbar_r1 = height - 2
    hbar_c0 = frame_c0 + frame_w + 1
    hbar_c1 = width - 2
    if hbar_c1 - hbar_c0 + 1 < 5 or vbar_r1 - vbar_r0 + 1 < 5:
        return None

    vbar_col = rng.randint(hbar_c0 + 1, hbar_c1 - 1)
    h_marker_pair = _choose_marker_pair(
        rng, list(range(hbar_c0, hbar_c1 + 1)), blocked={vbar_col}
    )
    v_marker_pair = _choose_marker_pair(
        rng, list(range(vbar_r0, vbar_r1 + 1)), blocked={hbar_row}
    )
    if h_marker_pair is None or v_marker_pair is None:
        return None

    dh = abs(h_marker_pair[1] - h_marker_pair[0])
    dv = abs(v_marker_pair[1] - v_marker_pair[0])
    if dh <= 0 or dv <= 0:
        return None
    if dh > iw or dv > ih:
        return None
    if not _output_uses_both_colors(scaled_target, dh, dv):
        return None
    if not _periods_are_semantically_active(scaled_target, dh, dv):
        return None

    grid = np.full((height, width), BG, dtype=int)

    template_c0 = rng.randint(1, width - template_total_w - 1)
    cursor = template_c0
    for mask in templates:
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = TEMPLATE
        grid[template_r0 : template_r0 + th, cursor : cursor + tw] = patch
        cursor += tw + template_gap

    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1
    grid[frame_r0, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r1, frame_c0 : frame_c1 + 1] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c0] = FRAME
    grid[frame_r0 : frame_r1 + 1, frame_c1] = FRAME
    for r, c in seed_pixels:
        grid[frame_r0 + 1 + int(r), frame_c0 + 1 + int(c)] = SEED

    grid[hbar_row, hbar_c0 : hbar_c1 + 1] = H_BAR
    grid[vbar_r0 : vbar_r1 + 1, vbar_col] = V_BAR
    grid[hbar_row, list(h_marker_pair)] = MARKER
    grid[list(v_marker_pair), vbar_col] = MARKER

    expected = _expected_output(grid, (frame_r0, frame_c0, frame_r1, frame_c1), scaled_target, dh, dv)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Additional semantic checks beyond round-tripping.
    seed_mask = np.zeros((ih, iw), dtype=bool)
    for r, c in seed_pixels:
        seed_mask[int(r), int(c)] = True
    if sum(np.all(~seed_mask | scaled) for scaled in scaled_masks) != 1:
        return None
    alt_dh = 2 if dh == 1 else dh - 1
    alt_dv = 2 if dv == 1 else dv - 1
    if alt_dh <= iw and np.array_equal(_weave_patch(scaled_target, dh, dv), _weave_patch(scaled_target, alt_dh, dv)):
        return None
    if alt_dv <= ih and np.array_equal(_weave_patch(scaled_target, dh, dv), _weave_patch(scaled_target, dh, alt_dv)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")

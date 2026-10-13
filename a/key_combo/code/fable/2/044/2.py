from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]
PALETTE = tuple(range(10))
GEN_RETRIES = 500
SPRITE_CATEGORY_WEIGHTS = (
    ("one_quadrant", 1),
    ("vertical_cross", 2),
    ("horizontal_cross", 2),
    ("double_cross", 2),
)


def get_bg(grid: np.ndarray) -> int:
    """Minimal self-contained copy of arc_utils.get_bg()."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> BBox | None:
    """Minimal self-contained copy of arc_utils.bbox()."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    """Minimal self-contained copy of arc_utils.crop_bbox()."""
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _is_contiguous(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _orbit_key(h: int, w: int, r: int, c: int) -> Coord:
    return min(r, h - 1 - r), min(c, w - 1 - c)


def _orbit_positions(h: int, w: int, r: int, c: int) -> list[Coord]:
    return [
        (r, c),
        (r, w - 1 - c),
        (h - 1 - r, c),
        (h - 1 - r, w - 1 - c),
    ]


def _extract_panel(grid: np.ndarray) -> tuple[int, np.ndarray, BBox]:
    bg = get_bg(grid)
    box = bbox(grid != bg)
    assert box is not None, "expected one non-background panel"
    panel = crop_bbox(grid, box)
    assert np.all(panel != bg), "panel should be a solid non-background rectangle"
    h, w = panel.shape
    assert h % 2 == 0 and w % 2 == 0, "panel must have even height and width"
    return bg, panel, box


def _recover_expected_panel(panel: np.ndarray) -> np.ndarray:
    h, w = map(int, panel.shape)
    assert h % 2 == 0 and w % 2 == 0
    expected = panel.copy()
    for r in range(h // 2):
        for c in range(w // 2):
            pos = _orbit_positions(h, w, r, c)
            values = [int(panel[rr, cc]) for rr, cc in pos]
            counts = Counter(values).most_common()
            majority_color, majority_count = counts[0]
            assert majority_count >= 3, "every mirror orbit needs a unique majority"
            for rr, cc in pos:
                expected[rr, cc] = int(majority_color)
    return expected


def _intruder_mask(panel: np.ndarray) -> np.ndarray:
    expected = _recover_expected_panel(panel)
    mask = panel != expected
    assert mask.any(), "expected a non-empty intruder"
    return mask


def _output_from_intruder(panel: np.ndarray, intruder: np.ndarray, bg: int) -> np.ndarray:
    box = bbox(intruder)
    assert box is not None
    panel_crop = crop_bbox(panel, box)
    mask_crop = crop_bbox(intruder, box)
    out = np.full(mask_crop.shape, int(bg), dtype=int)
    out[mask_crop] = panel_crop[mask_crop]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the overwritten intruder out of a four-way mirror mosaic by majority voting each mirror orbit.

    Input structure:
    - A dominant background surrounds one solid rectangular foreground panel whose height and width are both even.
    - The panel is a 2x2 mirror mosaic: one quarter pattern is reflected left-right, up-down, and by 180 degrees to fill the rectangle.
    - A monochrome intruder sprite overwrites some panel cells with a color already used elsewhere in the panel.

    Rule:
    - Crop the foreground panel from the background grid.
    - For every panel cell, compare it with its three horizontal/vertical/180-degree mirror partners and recover the intended color by majority vote on that four-cell orbit.
    - Cells that disagree with their orbit majority are intruder cells; return the tight bbox of those cells, keeping intruder cells in their sprite color and filling all other cells in the crop with the global background color.

    Key insight:
    - The intruder color is not special by frequency, because it already belongs to the mosaic palette; only the panel's four-way mirror redundancy can reveal which cells were overwritten.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, panel, _panel_box = _extract_panel(g)
    intruder = _intruder_mask(panel)
    return _output_from_intruder(panel, intruder, bg)


def _paint_feature(source: np.ndarray, color: int) -> None:
    h, w = map(int, source.shape)
    mode = random.choice(["rect", "hstripe", "vstripe", "lshape", "dot"])
    if mode == "rect":
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        r1 = random.randrange(r0, h)
        c1 = random.randrange(c0, w)
        source[r0 : r1 + 1, c0 : c1 + 1] = color
        return
    if mode == "hstripe":
        r = random.randrange(h)
        c0 = random.randrange(w)
        c1 = random.randrange(c0, w)
        source[r, c0 : c1 + 1] = color
        return
    if mode == "vstripe":
        c = random.randrange(w)
        r0 = random.randrange(h)
        r1 = random.randrange(r0, h)
        source[r0 : r1 + 1, c] = color
        return
    if mode == "lshape":
        r = random.randrange(h)
        c = random.randrange(w)
        height = random.randint(1, h - r)
        width = random.randint(1, w - c)
        source[r : r + height, c] = color
        source[r, c : c + width] = color
        return
    source[random.randrange(h), random.randrange(w)] = color


def _make_source_pattern(qh: int, qw: int, colors: list[int]) -> np.ndarray | None:
    source = np.full((qh, qw), int(colors[0]), dtype=int)
    feature_count = random.randint(4, 8)
    for _ in range(feature_count):
        _paint_feature(source, random.choice(colors))
    for color in colors:
        if not np.any(source == color):
            _paint_feature(source, color)
    if len(set(map(int, source.ravel()))) < 3:
        return None
    if np.array_equal(source, np.fliplr(source)):
        return None
    if np.array_equal(source, np.flipud(source)):
        return None
    if np.array_equal(source, np.rot90(source, 2)):
        return None
    return source


def _make_panel(source: np.ndarray) -> np.ndarray:
    top = np.hstack([source, np.fliplr(source)])
    return np.vstack([top, np.flipud(top)])


def _sprite_category(mask: np.ndarray) -> str:
    h, w = map(int, mask.shape)
    rows, cols = np.argwhere(mask).T
    top = bool(np.any(rows < h // 2))
    bottom = bool(np.any(rows >= h // 2))
    left = bool(np.any(cols < w // 2))
    right = bool(np.any(cols >= w // 2))
    if (top != bottom) and (left != right):
        return "one_quadrant"
    if (top != bottom) and left and right:
        return "vertical_cross"
    if (left != right) and top and bottom:
        return "horizontal_cross"
    return "double_cross"


def _candidate_priority(
    h: int,
    w: int,
    category: str,
    have_top: bool,
    have_bottom: bool,
    have_left: bool,
    have_right: bool,
    cell: Coord,
) -> tuple[int, int, int]:
    r, c = cell
    add_tb = int((r < h // 2 and not have_top) or (r >= h // 2 and not have_bottom))
    add_lr = int((c < w // 2 and not have_left) or (c >= w // 2 and not have_right))
    seam_bonus = int(abs(c - (w // 2 - 1)) <= 1 or abs(r - (h // 2 - 1)) <= 1)
    if category == "vertical_cross":
        return add_lr, seam_bonus, add_tb
    if category == "horizontal_cross":
        return add_tb, seam_bonus, add_lr
    if category == "double_cross":
        return add_tb + add_lr, seam_bonus, add_tb
    return seam_bonus, add_tb, add_lr


def _grow_sprite_mask(h: int, w: int, category: str, target_size: int) -> np.ndarray | None:
    qh, qw = h // 2, w // 2
    for _ in range(200):
        if category == "one_quadrant":
            qr = random.randrange(2)
            qc = random.randrange(2)
            row_range = range(qr * qh, (qr + 1) * qh)
            col_range = range(qc * qw, (qc + 1) * qw)
            start = (random.choice(tuple(row_range)), random.choice(tuple(col_range)))

            def allowed(rr: int, cc: int) -> bool:
                return rr in row_range and cc in col_range

        elif category == "vertical_cross":
            top_half = random.randrange(2)
            row_range = range(top_half * qh, (top_half + 1) * qh)
            seam_cols = [c for c in (qw - 1, qw) if 0 <= c < w]
            start = (random.choice(tuple(row_range)), random.choice(seam_cols))

            def allowed(rr: int, cc: int) -> bool:
                return rr in row_range and 0 <= cc < w

        elif category == "horizontal_cross":
            left_half = random.randrange(2)
            col_range = range(left_half * qw, (left_half + 1) * qw)
            seam_rows = [r for r in (qh - 1, qh) if 0 <= r < h]
            start = (random.choice(seam_rows), random.choice(tuple(col_range)))

            def allowed(rr: int, cc: int) -> bool:
                return 0 <= rr < h and cc in col_range

        else:
            center_rows = [r for r in (qh - 1, qh) if 0 <= r < h]
            center_cols = [c for c in (qw - 1, qw) if 0 <= c < w]
            start = (random.choice(center_rows), random.choice(center_cols))

            def allowed(rr: int, cc: int) -> bool:
                return 0 <= rr < h and 0 <= cc < w

        mask = np.zeros((h, w), dtype=bool)
        used_keys = set()
        sr, sc = start
        used_keys.add(_orbit_key(h, w, sr, sc))
        mask[sr, sc] = True
        cells = {(sr, sc)}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(sr, sc)
            if allowed(nr, nc) and _orbit_key(h, w, nr, nc) not in used_keys
        }
        have_top = sr < h // 2
        have_bottom = sr >= h // 2
        have_left = sc < w // 2
        have_right = sc >= w // 2

        while len(cells) < target_size:
            candidates = [
                cell for cell in frontier if _orbit_key(h, w, cell[0], cell[1]) not in used_keys and not mask[cell]
            ]
            if not candidates:
                break
            best = max(
                _candidate_priority(
                    h,
                    w,
                    category,
                    have_top,
                    have_bottom,
                    have_left,
                    have_right,
                    cell,
                )
                for cell in candidates
            )
            options = [
                cell
                for cell in candidates
                if _candidate_priority(
                    h,
                    w,
                    category,
                    have_top,
                    have_bottom,
                    have_left,
                    have_right,
                    cell,
                )
                == best
            ]
            nr, nc = random.choice(options)
            mask[nr, nc] = True
            cells.add((nr, nc))
            used_keys.add(_orbit_key(h, w, nr, nc))
            have_top = have_top or nr < h // 2
            have_bottom = have_bottom or nr >= h // 2
            have_left = have_left or nc < w // 2
            have_right = have_right or nc >= w // 2
            frontier.discard((nr, nc))
            for rr, cc in _neighbors4(nr, nc):
                if allowed(rr, cc) and not mask[rr, cc]:
                    frontier.add((rr, cc))

        if len(cells) != target_size:
            continue
        if not _is_contiguous(mask):
            continue
        if _sprite_category(mask) != category:
            continue
        return mask
    return None


def _choose_intruder_color(panel: np.ndarray, mask: np.ndarray) -> int | None:
    masked_colors = set(map(int, panel[mask]))
    outside_counts = Counter(map(int, panel[~mask]))
    candidates = [color for color, count in outside_counts.items() if count >= 4 and color not in masked_colors]
    if not candidates:
        return None
    candidates.sort(key=lambda color: (outside_counts[color], color), reverse=True)
    return int(random.choice(candidates[: min(3, len(candidates))]))


def _sample_canvas(panel_h: int, panel_w: int) -> tuple[int, int, int, int] | None:
    for _ in range(100):
        extra_h = random.randint(3, 30 - panel_h)
        extra_w = random.randint(3, 30 - panel_w)
        top = random.randint(0, extra_h)
        bottom = extra_h - top
        left = random.randint(0, extra_w)
        right = extra_w - left
        h = panel_h + top + bottom
        w = panel_w + left + right
        if h <= 30 and w <= 30 and h * w > 2 * panel_h * panel_w:
            return top, left, h, w
    return None


def _expected_intruder_output(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    mask_crop = crop_bbox(mask, box)
    out = np.full(mask_crop.shape, int(bg), dtype=int)
    out[mask_crop] = int(color)
    return out


def _semantic_validate(
    grid: np.ndarray,
    clean_panel: np.ndarray,
    panel_origin: Coord,
    intruder_mask: np.ndarray,
    intruder_color: int,
) -> None:
    bg = get_bg(grid)
    pr, pc = panel_origin
    ph, pw = clean_panel.shape
    observed_panel = grid[pr : pr + ph, pc : pc + pw]
    assert np.all(observed_panel[intruder_mask] == intruder_color)
    assert np.all(observed_panel[~intruder_mask] == clean_panel[~intruder_mask])
    assert _is_contiguous(intruder_mask)
    assert np.any(clean_panel[~intruder_mask] == intruder_color), "intruder color must also appear in the clean panel"

    qh, qw = ph // 2, pw // 2
    tl = clean_panel[:qh, :qw]
    assert np.array_equal(clean_panel[:qh, qw:], np.fliplr(tl))
    assert np.array_equal(clean_panel[qh:, :qw], np.flipud(tl))
    assert np.array_equal(clean_panel[qh:, qw:], np.rot90(tl, 2))

    expected = _recover_expected_panel(observed_panel)
    assert np.array_equal(expected, clean_panel)

    for r in range(qh):
        for c in range(qw):
            pos = _orbit_positions(ph, pw, r, c)
            values = [int(observed_panel[rr, cc]) for rr, cc in pos]
            counts = Counter(values)
            changed = [intruder_mask[rr, cc] for rr, cc in pos]
            if any(changed):
                assert sum(changed) == 1, "an orbit may contain at most one overwritten cell"
                assert max(counts.values()) == 3 and min(counts.values()) == 1
            else:
                assert len(counts) == 1

    expected_out = _expected_intruder_output(intruder_mask, intruder_color, bg)
    actual_out = main(grid)
    assert np.array_equal(actual_out, expected_out)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    usable = [color for color in PALETTE if color != bg]
    panel_colors = random.sample(usable, random.randint(3, 4))
    qh = random.randint(4, 7)
    qw = random.randint(4, 7)

    source = _make_source_pattern(qh, qw, panel_colors)
    if source is None:
        return None
    clean_panel = _make_panel(source)
    ph, pw = clean_panel.shape

    weighted_categories = [name for name, weight in SPRITE_CATEGORY_WEIGHTS for _ in range(weight)]
    category = random.choice(weighted_categories)
    max_size = min(12, ph + pw - 2, (ph * pw) // 4)
    min_size = 4 if category == "one_quadrant" else 5
    if max_size < min_size:
        return None
    target_size = random.randint(min_size, max_size)
    sprite_mask = _grow_sprite_mask(ph, pw, category, target_size)
    if sprite_mask is None:
        return None

    intruder_color = _choose_intruder_color(clean_panel, sprite_mask)
    if intruder_color is None:
        return None

    panel_with_intruder = clean_panel.copy()
    panel_with_intruder[sprite_mask] = intruder_color

    canvas = _sample_canvas(ph, pw)
    if canvas is None:
        return None
    top, left, h, w = canvas
    grid = np.full((h, w), int(bg), dtype=int)
    grid[top : top + ph, left : left + pw] = panel_with_intruder

    if get_bg(grid) != bg:
        return None

    try:
        _semantic_validate(grid, clean_panel, (top, left), sprite_mask, intruder_color)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-mosaic intruder puzzle")

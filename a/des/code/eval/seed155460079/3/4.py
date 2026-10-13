import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 300
TEMPLATE_COUNT_RANGE = (2, 3)
TEMPLATE_HEIGHTS = (5, 6, 7, 8)
TEMPLATE_WIDTHS = (4, 5, 6, 7)
TEMPLATE_GAP_RANGE = (2, 3)
TOP_MARGIN_RANGE = (1, 2)
SEPARATOR_GAP_RANGE = (2, 4)
SIDE_MARGIN_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (1, 3)
TARGET_EXTRA_COUNT = (0, 1, 2)
PLACEMENT_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max()) + 1
    c1 = int(coords[:, 1].max()) + 1
    return r0, c0, r1, c1


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _is_horizontally_symmetric(mask: np.ndarray) -> bool:
    return np.array_equal(mask, mask[::-1, :])


def _adjacent_background(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    out.append((r, c))
                    break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_has_fg = np.any(g != BG, axis=1)
    occupied_rows = np.flatnonzero(row_has_fg)
    assert occupied_rows.size > 0

    first_row = int(occupied_rows[0])
    split = first_row
    while split < g.shape[0] and row_has_fg[split]:
        split += 1
    assert split < g.shape[0], "expected a blank separator below the template library"
    assert np.any(g[split + 1 :, :] != BG), "expected target objects below the separator"

    templates: dict[int, np.ndarray] = {}
    targets: list[tuple[int, int, int, int]] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0:r1, c0:c1]
        non_bg = patch[patch != BG]
        colors = {int(v) for v in np.unique(non_bg)}
        if r1 <= split:
            assert len(colors) == 1
            color = next(iter(colors))
            assert color != GREY
            mask = patch == color
            assert _is_connected(mask)
            assert _is_horizontally_symmetric(mask)
            assert color not in templates
            templates[color] = mask
        else:
            assert r0 > split
            targets.append((r0, c0, r1, c1))

    assert templates
    assert targets

    out = g.copy()
    for r0, c0, r1, c1 in targets:
        patch = g[r0:r1, c0:c1]
        marker_colors = {int(v) for v in np.unique(patch) if int(v) not in {BG, GREY}}
        assert len(marker_colors) == 1
        marker_color = next(iter(marker_colors))
        assert marker_color in templates

        template = templates[marker_color]
        assert patch.shape == template.shape

        marker_mask = patch == marker_color
        grey_mask = patch == GREY
        assert 1 <= int(marker_mask.sum()) <= 2
        assert grey_mask.any()
        assert np.all(marker_mask <= template)

        repaired = np.full(patch.shape, BG, dtype=int)
        repaired[template] = GREY
        out[r0:r1, c0:c1] = repaired

    return out


def _random_interval(width: int, prev: tuple[int, int] | None) -> tuple[int, int]:
    for _ in range(60):
        length = random.randint(2, width)
        left = random.randint(0, width - length)
        right = left + length - 1
        if prev is None:
            return left, right
        prev_left, prev_right = prev
        if left <= prev_right and right >= prev_left:
            return left, right
    assert prev is not None
    prev_left, prev_right = prev
    left = max(0, prev_left - 1)
    right = min(width - 1, prev_right + 1)
    return left, right


def _make_template_mask(height: int, width: int) -> np.ndarray | None:
    top_rows = (height + 1) // 2
    intervals: list[tuple[int, int]] = []
    prev = None
    for _ in range(top_rows):
        interval = _random_interval(width, prev)
        intervals.append(interval)
        prev = interval

    mask = np.zeros((height, width), dtype=bool)
    for r, (left, right) in enumerate(intervals):
        mask[r, left : right + 1] = True
        mirror_r = height - 1 - r
        mask[mirror_r, left : right + 1] = True

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    mask = mask[r0:r1, c0:c1]
    if not _is_connected(mask):
        return None
    if not _is_horizontally_symmetric(mask):
        return None
    if not _adjacent_background(mask):
        return None
    if np.all(mask):
        return None
    return mask


def _build_target_patch(template: np.ndarray, color: int) -> np.ndarray | None:
    h, w = template.shape
    patch = np.full((h, w), BG, dtype=int)
    patch[template] = GREY

    template_cells = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    damage_count = random.choice((1, 2))
    if len(template_cells) < damage_count:
        return None
    marker_cells = random.sample(template_cells, damage_count)
    for r, c in marker_cells:
        patch[r, c] = color

    extra_candidates = _adjacent_background(template)
    extra_mode = random.choice(TARGET_EXTRA_COUNT)
    extra_count = min(extra_mode, len(extra_candidates), damage_count)
    if extra_count > 0:
        for r, c in random.sample(extra_candidates, extra_count):
            patch[r, c] = GREY

    if not _is_connected(patch != BG):
        return None
    return patch


def _place_boxes(
    grid_shape: tuple[int, int],
    start_row: int,
    boxes: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for box in boxes:
        bh, bw = box.shape
        if start_row + bh >= h or bw + 2 > w:
            return None
        found = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r = random.randint(start_row, h - bh - 1)
            c = random.randint(1, w - bw - 1)
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(h, r + bh + 1)
            cc1 = min(w, c + bw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            found = (r, c)
            blocked[rr0:rr1, cc0:cc1] = True
            placements.append(found)
            break
        if found is None:
            return None
    return placements


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    template_height = random.choice(TEMPLATE_HEIGHTS)
    template_colors = random.sample(PALETTE, template_count)

    templates: list[np.ndarray] = []
    seen_shapes: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for _color in template_colors:
        mask = None
        for _ in range(80):
            raw = _make_template_mask(template_height, random.choice(TEMPLATE_WIDTHS))
            if raw is None:
                continue
            key = (raw.shape, tuple(int(v) for v in raw.ravel()))
            if key in seen_shapes:
                continue
            seen_shapes.add(key)
            mask = raw
            break
        if mask is None:
            return None
        templates.append(mask)

    target_indices = list(range(template_count))
    extra_targets = random.randint(0, 1)
    for _ in range(extra_targets):
        target_indices.append(random.randrange(template_count))
    random.shuffle(target_indices)

    target_patches: list[np.ndarray] = []
    for idx in target_indices:
        patch = _build_target_patch(templates[idx], template_colors[idx])
        if patch is None:
            return None
        target_patches.append(patch)

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    side_margin = random.randint(*SIDE_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    template_gap = random.randint(*TEMPLATE_GAP_RANGE)
    separator_gap = random.randint(*SEPARATOR_GAP_RANGE)

    library_width = sum(t.shape[1] for t in templates) + template_gap * (len(templates) - 1)
    max_target_w = max(t.shape[1] for t in target_patches)
    grid_w = max(
        library_width + 2 * side_margin,
        max_target_w + 2 * side_margin + 4,
    ) + random.randint(0, 3)
    lower_band = random.randint(max(10, template_height + 2), 16)
    grid_h = (
        top_margin
        + template_height
        + separator_gap
        + lower_band
        + bottom_margin
    )

    lower_start = top_margin + template_height + separator_gap
    placements = _place_boxes((grid_h, grid_w), lower_start, target_patches)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    col = side_margin
    for color, mask in zip(template_colors, templates):
        h, w = mask.shape
        patch = grid[top_margin : top_margin + h, col : col + w]
        patch[mask] = color
        col += w + template_gap

    for patch, (r, c) in zip(target_patches, placements):
        h, w = patch.shape
        grid[r : r + h, c : c + w] = patch

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

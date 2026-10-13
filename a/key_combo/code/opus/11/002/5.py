from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GRAY = 5
LEGEND_COLORS = (2, 3, 4, 6)
SPRITE_COLORS = (1, 7, 8, 9)
GEN_RETRIES = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
TEMPLATE_MASKS = (
    np.array([[0, 1], [1, 1]], dtype=bool),  # missing anchor cell
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _legend_templates(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    templates: dict[int, np.ndarray] = {}
    marker_coords: set[tuple[int, int]] = set()
    for r in range(h - 2):
        for c in range(w - 2):
            color = int(g[r, c])
            if color in (bg, GRAY):
                continue
            if any(_in_bounds(g.shape, nr, nc) and g[nr, nc] != bg for nr, nc in _neighbors4(r, c)):
                continue
            block = g[r + 1 : r + 3, c + 1 : c + 3]
            if not np.isin(block, [bg, GRAY]).all() or not np.any(block == GRAY):
                continue
            window = g[r : r + 3, c : c + 3]
            allowed = np.zeros((3, 3), dtype=bool)
            allowed[0, 0] = True
            allowed[1:, 1:] = True
            if np.any((window != bg) & ~allowed):
                continue
            templates[color] = block == GRAY
            marker_coords.add((r, c))
    assert len(templates) >= 2, "expected 2-3 legend entries"
    assert len(marker_coords) == len(templates), "legend colors must be unique"
    return templates, marker_coords


def _source_component(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    matches = []
    for coords in _components(grid != bg):
        vals = grid[coords[:, 0], coords[:, 1]]
        gray_cells = coords[vals == GRAY]
        if gray_cells.shape[0] == 1 and np.any(vals != GRAY):
            matches.append((coords, tuple(map(int, gray_cells[0]))))
    assert len(matches) == 1, "expected one multicolor source sprite with one gray anchor"
    return matches[0]


def _target_anchors(grid: np.ndarray, bg: int, legend_markers: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    anchors: list[tuple[int, int, int]] = []
    for coords in _components(grid != bg):
        if coords.shape[0] != 1:
            continue
        r, c = map(int, coords[0])
        if (r, c) in legend_markers:
            continue
        color = int(grid[r, c])
        if color == GRAY:
            continue
        anchors.append((r, c, color))
    anchors.sort()
    assert 2 <= len(anchors) <= 4, "expected 2-4 target anchors"
    return anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate the source sprite to each colored anchor, then stamp the color-selected 2x2 legend mask over the aligned anchor area.

    Input structure:
    - A black background contains three separated regions: a top legend, one middle multicolor sprite, and 2-4 lower isolated colored anchor pixels.
    - Each legend entry is one isolated colored key pixel with a 2x2 grey/black template immediately down-right from it.
    - The source sprite is one 4-connected multicolor object containing exactly one grey pixel that serves as the source anchor.

    Rule:
    - For each target anchor pixel in raster order, erase that pixel, translate every non-grey pixel of the source sprite so the source grey anchor lands on the target, and keep the original sprite unchanged.
    - Use the target anchor color to select the matching legend template; overlay that 2x2 mask with the target anchor as the template's top-left corner.
    - Template cells that are grey become the target color, while template cells that are black leave the translated copy underneath unchanged.

    Key insight:
    - The same target-anchor pixel carries three meanings at once: its position gives the translation offset, its color chooses the legend mask, and that same color recolors the stamped mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    assert bg == BG, "this task uses black background"

    templates, legend_markers = _legend_templates(g, bg)
    source_coords, source_anchor = _source_component(g, bg)
    targets = _target_anchors(g, bg, legend_markers)
    sr, sc = source_anchor

    out = g.copy()
    for tr, tc, color in targets:
        assert color in templates, "every target color must appear in the legend"
        out[tr, tc] = bg
        mask = templates[color]
        for r, c in source_coords:
            rr = tr + (int(r) - sr)
            cc = tc + (int(c) - sc)
            value = int(g[int(r), int(c)])
            if value == GRAY:
                continue
            assert _in_bounds(g.shape, rr, cc), "translated copy must stay in bounds"
            out[rr, cc] = value
        for dr in range(2):
            for dc in range(2):
                if mask[dr, dc]:
                    rr, cc = tr + dr, tc + dc
                    assert _in_bounds(g.shape, rr, cc), "template stamp must stay in bounds"
                    out[rr, cc] = color

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask() -> np.ndarray:
    for _ in range(120):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        lo = max(8, int(np.ceil(0.45 * h * w)))
        hi = min(18, int(np.floor(0.75 * h * w)))
        if lo > hi:
            continue
        count = random.randint(lo, hi)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < count:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if not (4 <= mask.shape[0] <= 6 and 4 <= mask.shape[1] <= 6):
            continue
        if _anchor_candidates(mask):
            return mask
    raise RuntimeError("failed to build connected sprite mask")


def _anchor_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    total = int(mask.sum())
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r + 1 >= h or c + 1 >= w:
            continue
        if not any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
            continue
        local = int(mask[r : r + 2, c : c + 2].sum())
        if local < 2 or local >= total:
            continue
        candidates.append((r, c))
    return candidates


def _color_sprite(mask: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    sprite = np.full((h, w), BG, dtype=int)
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    colored = [rc for rc in cells if rc != anchor]
    n_colors = min(len(SPRITE_COLORS), random.randint(2, 4), len(colored))
    palette = random.sample(list(SPRITE_COLORS), n_colors)
    random.shuffle(colored)
    assignments = {}
    for i, cell in enumerate(colored):
        color = palette[i] if i < n_colors else random.choice(palette)
        assignments[cell] = color
    for (r, c), color in assignments.items():
        sprite[r, c] = color
    ar, ac = anchor
    sprite[ar, ac] = GRAY
    return sprite


def _sample_legend() -> tuple[list[int], dict[int, np.ndarray]]:
    n = random.choices([2, 3], weights=[2, 3], k=1)[0]
    colors = random.sample(list(LEGEND_COLORS), n)
    masks = [TEMPLATE_MASKS[0].copy()]
    if n == 2:
        masks.append(TEMPLATE_MASKS[4].copy() if random.random() < 0.7 else random.choice(TEMPLATE_MASKS[1:4]).copy())
    else:
        masks.append(TEMPLATE_MASKS[4].copy())
        masks.append(random.choice(TEMPLATE_MASKS[1:4]).copy())
    random.shuffle(masks)
    return colors, {color: mask for color, mask in zip(colors, masks)}


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _render_expected(
    inp: np.ndarray,
    sprite: np.ndarray,
    anchor: tuple[int, int],
    sprite_origin: tuple[int, int],
    targets: list[tuple[int, int, int]],
    templates: dict[int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    ar, ac = anchor
    src_r0, src_c0 = sprite_origin
    for tr, tc, color in sorted(targets):
        out[tr, tc] = BG
        for r in range(sprite.shape[0]):
            for c in range(sprite.shape[1]):
                value = int(sprite[r, c])
                if value in (BG, GRAY):
                    continue
                rr = tr + (r - ar)
                cc = tc + (c - ac)
                out[rr, cc] = value
        mask = templates[color]
        for dr in range(2):
            for dc in range(2):
                if mask[dr, dc]:
                    out[tr + dr, tc + dc] = color
    return out


def _plain_copy_render(
    inp: np.ndarray,
    sprite: np.ndarray,
    anchor: tuple[int, int],
    targets: list[tuple[int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    ar, ac = anchor
    for tr, tc, _color in sorted(targets):
        out[tr, tc] = BG
        for r in range(sprite.shape[0]):
            for c in range(sprite.shape[1]):
                value = int(sprite[r, c])
                if value in (BG, GRAY):
                    continue
                out[tr + (r - ar), tc + (c - ac)] = value
    return out


def _entry_columns(n_legend: int) -> list[int]:
    cols = [random.randint(1, 3)]
    for _ in range(n_legend - 1):
        cols.append(cols[-1] + random.randint(4, 6))
    return cols


def _place_targets(
    h: int,
    w: int,
    sprite_shape: tuple[int, int],
    source_rect: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]]:
    sh, sw = sprite_shape
    row_lo = source_rect[2] + 3
    row_hi = h - sh
    if row_lo > row_hi:
        return []
    rects: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int]] = []
    for _ in range(count):
        placed = False
        for _attempt in range(240):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(0, w - sw)
            rect = (r0, c0, r0 + sh - 1, c0 + sw - 1)
            if _rects_overlap(rect, source_rect, pad=1):
                continue
            if any(_rects_overlap(rect, other, pad=1) for other in rects):
                continue
            rects.append(rect)
            positions.append((r0, c0))
            placed = True
            break
        if not placed:
            return []
    return positions


def _gen_attempt() -> np.ndarray | None:
    legend_colors, templates = _sample_legend()
    mask = _random_connected_mask()
    anchor = random.choice(_anchor_candidates(mask))
    sprite = _color_sprite(mask, anchor)
    sh, sw = sprite.shape
    ar, ac = anchor

    entry_cols = _entry_columns(len(legend_colors))
    legend_width = entry_cols[-1] + 4
    legend_row = random.randint(0, 1)
    min_h = legend_row + 3 + 2 + sh + 3 + sh
    if min_h > 29:
        return None
    h = random.randint(max(20, min_h), 29)
    w = random.randint(max(20, legend_width), 29)

    max_source_top = h - sh - 8
    min_source_top = legend_row + 5
    if min_source_top > max_source_top:
        return None
    src_r0 = random.randint(min_source_top, max_source_top)
    src_c0 = random.randint(1, w - sw - 2)
    source_rect = (src_r0, src_c0, src_r0 + sh - 1, src_c0 + sw - 1)

    n_targets = random.randint(max(2, len(legend_colors)), 4)
    target_origins = _place_targets(h, w, sprite.shape, source_rect, n_targets)
    if len(target_origins) != n_targets:
        return None

    target_colors = legend_colors[:]
    while len(target_colors) < n_targets:
        target_colors.append(random.choice(legend_colors))
    random.shuffle(target_colors)
    targets = [(r0 + ar, c0 + ac, color) for (r0, c0), color in zip(target_origins, target_colors)]

    inp = np.full((h, w), BG, dtype=int)
    for color, col in zip(legend_colors, entry_cols):
        inp[legend_row, col] = color
        inp[legend_row + 1 : legend_row + 3, col + 1 : col + 3] = np.where(templates[color], GRAY, BG)
    for r in range(sh):
        for c in range(sw):
            value = int(sprite[r, c])
            if value != BG:
                inp[src_r0 + r, src_c0 + c] = value
    for tr, tc, color in targets:
        if inp[tr, tc] != BG:
            return None
        if any(inp[nr, nc] != BG for nr, nc in _neighbors4(tr, tc) if _in_bounds(inp.shape, nr, nc)):
            return None
        inp[tr, tc] = color

    expected = _render_expected(inp, sprite, anchor, (src_r0, src_c0), targets, templates)
    plain = _plain_copy_render(inp, sprite, anchor, targets)

    # Semantic checks: every legend color is demonstrated, stamps matter, and
    # at least one used template leaves the anchor cell empty so omitting the
    # copied gray anchor is observable.
    if {color for _, _, color in targets} != set(legend_colors):
        return None
    for tr, tc, color in targets:
        if np.array_equal(expected[tr : tr + 2, tc : tc + 2], plain[tr : tr + 2, tc : tc + 2]):
            return None
    if not any(not templates[color][0, 0] for _, _, color in targets):
        return None
    if int(mask.sum()) == int(mask[ar : ar + 2, ac : ac + 2].sum()):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

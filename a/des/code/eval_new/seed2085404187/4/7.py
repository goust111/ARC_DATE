import random

import numpy as np


BG = 0
SEP_COLOR = 5
FRAME_COLOR = 9
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    holes = []
    for comp in _components(~mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == mask.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == mask.shape[1] - 1):
            continue
        holes.append(comp)
    return holes


def _is_filled_square(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != c1 - c0:
        return False
    hole = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    hole[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return bool(np.all(hole))


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP_COLOR)]
    valid = [c for c in cols if 0 < c < grid.shape[1] - 1]
    assert len(valid) == 1, f"expected exactly one separator column, got {valid}"
    return int(valid[0])


def _find_frame_bbox(panel: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(panel == FRAME_COLOR)
    assert coords.size, "missing maroon template frame"
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "template frame must enclose an interior"
    assert np.all(panel[r0, c0 : c1 + 1] == FRAME_COLOR)
    assert np.all(panel[r1, c0 : c1 + 1] == FRAME_COLOR)
    assert np.all(panel[r0 : r1 + 1, c0] == FRAME_COLOR)
    assert np.all(panel[r0 : r1 + 1, c1] == FRAME_COLOR)
    return r0, c0, r1, c1


def _extract_mapping(left_panel: np.ndarray) -> dict[int, int]:
    r0, c0, r1, c1 = _find_frame_bbox(left_panel)
    inner = left_panel[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components(inner != BG)
    assert comps, "template frame must contain rule glyphs"

    mapping: dict[int, int] = {}
    for comp in comps:
        rr0, cc0, rr1, cc1 = _bbox(comp)
        patch = inner[rr0 : rr1 + 1, cc0 : cc1 + 1]
        local = np.zeros(patch.shape, dtype=bool)
        local[comp[:, 0] - rr0, comp[:, 1] - cc0] = True
        colors = [int(color) for color in np.unique(patch[local])]
        assert len(colors) == 2, f"rule glyph must contain exactly two colors, got {colors}"

        boundary_colors = []
        interior_colors = []
        for color in colors:
            color_mask = (patch == color) & local
            touches_outside = False
            for rr, cc in np.argwhere(color_mask):
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < patch.shape[0] and 0 <= nc < patch.shape[1]) or not local[nr, nc]:
                        touches_outside = True
                        break
                if touches_outside:
                    break
            if touches_outside:
                boundary_colors.append(color)
            else:
                interior_colors.append(color)

        assert len(boundary_colors) == 1 and len(interior_colors) == 1, (
            f"could not isolate outline/fill colors: {colors}"
        )
        source, target = int(boundary_colors[0]), int(interior_colors[0])
        if source in mapping:
            assert mapping[source] == target, "duplicate source color maps inconsistently"
        mapping[source] = target

    return mapping


def _has_square_hole(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return any(_is_filled_square(hole) for hole in _enclosed_holes(mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the holed right-panel regions by decoding a two-color rule template from the left panel.

    Input structure:
    - `black` is the background. A full-height `grey` column splits the grid into a left template panel and a right object panel.
    - The left panel contains one `maroon` rectangular frame whose interior holds several small two-color nested glyphs.
    - In each glyph, the outer contour color is the source color and the enclosed inner fill color is the replacement color.

    Rule:
    - Read every nested glyph inside the maroon frame to build a source→target recoloring map.
    - In the right panel, treat each 4-connected monochrome region independently and test whether it encloses a square black hole.
    - Recolor exactly those holed regions using the decoded map, keep non-holed regions unchanged, and return only the right panel.

    Key insight:
    - The left panel is a legend made of visible outline/fill pairs, while the square hole inside a right-panel object is the trigger that decides whether the legend applies.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    mapping = _extract_mapping(left)

    out = right.copy()
    for color in [int(color) for color in np.unique(right) if int(color) != BG]:
        for comp in _components(right == color):
            if not _has_square_hole(comp):
                continue
            out[comp[:, 0], comp[:, 1]] = mapping.get(color, color)
    return out


def _make_rule_glyph(rng: random.Random, source: int, target: int) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    glyph = np.full((h, w), source, dtype=int)
    glyph[1:-1, 1:-1] = target
    return glyph


def _pack_glyphs(glyphs: list[np.ndarray], rng: random.Random) -> tuple[list[tuple[np.ndarray, int, int]], int, int]:
    max_glyph_w = max(int(glyph.shape[1]) for glyph in glyphs)
    max_row_width = rng.randint(max(7, max_glyph_w + 2), 11)
    rows: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    current_width = 0
    for glyph in glyphs:
        gw = int(glyph.shape[1])
        needed = gw if not current else current_width + 1 + gw
        if current and needed > max_row_width:
            rows.append(current)
            current = [glyph]
            current_width = gw
        else:
            current.append(glyph)
            current_width = needed
    if current:
        rows.append(current)

    placements: list[tuple[np.ndarray, int, int]] = []
    y = 0
    interior_w = 0
    for row in rows:
        row_h = max(int(glyph.shape[0]) for glyph in row)
        x = 0
        for glyph in row:
            placements.append((glyph, y, x))
            x += int(glyph.shape[1]) + 1
        interior_w = max(interior_w, x - 1)
        y += row_h + 1
    interior_h = y - 1
    return placements, interior_h, interior_w


def _add_tab(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        depth = rng.randint(1, 2)
        span = rng.randint(max(2, w // 2), w)
        start = rng.randint(0, w - span)
        if side == "top":
            out = np.pad(mask, ((depth, 0), (0, 0)))
            out[:depth, start : start + span] = True
        else:
            out = np.pad(mask, ((0, depth), (0, 0)))
            out[-depth:, start : start + span] = True
    else:
        depth = rng.randint(1, 2)
        span = rng.randint(max(2, h // 2), h)
        start = rng.randint(0, h - span)
        if side == "left":
            out = np.pad(mask, ((0, 0), (depth, 0)))
            out[start : start + span, :depth] = True
        else:
            out = np.pad(mask, ((0, 0), (0, depth)))
            out[start : start + span, -depth:] = True
    return out


def _add_notch(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    side = rng.choice(("top", "bottom", "left", "right"))
    out = mask.copy()
    if side in ("top", "bottom") and h >= 4 and w >= 3:
        depth = rng.randint(1, min(2, h - 2))
        span = rng.randint(1, w - 1)
        start = rng.randint(0, w - span)
        if side == "top":
            out[:depth, start : start + span] = False
        else:
            out[-depth:, start : start + span] = False
    elif side in ("left", "right") and w >= 4 and h >= 3:
        depth = rng.randint(1, min(2, w - 2))
        span = rng.randint(1, h - 1)
        start = rng.randint(0, h - span)
        if side == "left":
            out[start : start + span, :depth] = False
        else:
            out[start : start + span, -depth:] = False
    return _trim_mask(out)


def _make_region_mask(rng: random.Random, with_hole: bool) -> np.ndarray | None:
    for _ in range(60):
        if with_hole:
            h = rng.randint(5, 6)
            w = rng.randint(5, 6)
        else:
            h = rng.randint(4, 6)
            w = rng.randint(4, 6)
        mask = np.ones((h, w), dtype=bool)
        if with_hole:
            max_hole = min(3, h - 4, w - 4)
            if max_hole < 1:
                continue
            hole_size = rng.randint(1, max_hole)
            r0 = rng.randint(1, h - hole_size - 1)
            c0 = rng.randint(1, w - hole_size - 1)
            mask[r0 : r0 + hole_size, c0 : c0 + hole_size] = False
            if rng.random() < 0.7:
                mask = _add_tab(mask, rng)
        else:
            if rng.random() < 0.8:
                mask = _add_tab(mask, rng)
            if rng.random() < 0.45:
                mask = _add_notch(mask, rng)

        mask = _trim_mask(mask)
        if len(_components(mask)) != 1:
            continue
        holes = _enclosed_holes(mask)
        if with_hole:
            square_holes = [hole for hole in holes if _is_filled_square(hole)]
            if len(square_holes) != 1:
                continue
        elif holes:
            continue
        if int(mask.sum()) < 10:
            continue
        return mask
    return None


def _place_object(canvas: np.ndarray, mask: np.ndarray, color: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    mh, mw = mask.shape
    if mh > h or mw > w:
        return False
    obj_cells = np.argwhere(mask)
    for _ in range(150):
        r0 = rng.randint(0, h - mh)
        c0 = rng.randint(0, w - mw)
        blocked = False
        for rr, cc in obj_cells:
            ar, ac = int(r0 + rr), int(c0 + cc)
            if canvas[ar, ac] != BG:
                blocked = True
                break
            for dr, dc in DIR4:
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] != BG:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            continue
        patch = canvas[r0 : r0 + mh, c0 : c0 + mw]
        patch[mask] = color
        return True
    return False


def _build_left_panel(height: int, mapping: dict[int, int], rng: random.Random) -> np.ndarray | None:
    glyphs = [_make_rule_glyph(rng, source, target) for source, target in mapping.items()]
    rng.shuffle(glyphs)
    placements, inner_h, inner_w = _pack_glyphs(glyphs, rng)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    if frame_h + 2 > height:
        return None

    left_w = frame_w + rng.randint(2, 4)
    panel = np.full((height, left_w), BG, dtype=int)
    frame_r0 = rng.randint(1, height - frame_h - 1)
    frame_c0 = rng.randint(1, left_w - frame_w - 1)
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1
    panel[frame_r0, frame_c0 : frame_c1 + 1] = FRAME_COLOR
    panel[frame_r1, frame_c0 : frame_c1 + 1] = FRAME_COLOR
    panel[frame_r0 : frame_r1 + 1, frame_c0] = FRAME_COLOR
    panel[frame_r0 : frame_r1 + 1, frame_c1] = FRAME_COLOR

    for glyph, gr, gc in placements:
        gh, gw = glyph.shape
        rr = frame_r0 + 1 + gr
        cc = frame_c0 + 1 + gc
        panel[rr : rr + gh, cc : cc + gw] = glyph
    return panel


def _expected_right(right: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = right.copy()
    for color in [int(color) for color in np.unique(right) if int(color) != BG]:
        for comp in _components(right == color):
            if _has_square_hole(comp):
                out[comp[:, 0], comp[:, 1]] = mapping[color]
    return out


def _build_right_panel(height: int, width: int, source_colors: list[int], rng: random.Random) -> np.ndarray | None:
    pivot = rng.choice(source_colors)
    specs: list[tuple[int, bool]] = [(pivot, True), (pivot, False)]
    for color in source_colors:
        if color != pivot:
            specs.append((color, rng.choice((True, False))))

    objects = []
    for color, with_hole in specs:
        mask = _make_region_mask(rng, with_hole)
        if mask is None:
            return None
        objects.append((int(mask.sum()), color, with_hole, mask))
    objects.sort(reverse=True, key=lambda item: item[0])

    panel = np.full((height, width), BG, dtype=int)
    for _area, color, _with_hole, mask in objects:
        if not _place_object(panel, mask, color, rng):
            return None
    return panel


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rule_count = rng.choice((2, 3))
    colors = list(PALETTE_COLORS)
    rng.shuffle(colors)
    source_colors = colors[:rule_count]
    target_colors = colors[rule_count : 2 * rule_count]
    mapping = dict(zip(source_colors, target_colors))

    height = rng.randint(18, 24)
    left = _build_left_panel(height, mapping, rng)
    if left is None:
        return None

    max_right_w = 30 - left.shape[1] - 1
    if max_right_w < 10:
        return None
    right_w = rng.randint(11, min(16, max_right_w))
    right = _build_right_panel(height, right_w, source_colors, rng)
    if right is None:
        return None

    grid = np.concatenate([left, np.full((height, 1), SEP_COLOR, dtype=int), right], axis=1)
    expected = _expected_right(right, mapping)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if not np.any(actual != right):
        return None
    if not any(np.any((right == color) & (actual == color)) for color in source_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty glyph mask"
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool).copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    m = _normalize_mask(mask)
    return (tuple(map(int, m.shape)), tuple(int(v) for v in m.ravel()))


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    base = _normalize_mask(mask)
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(base, k)
        for candidate in (rot, np.fliplr(rot)):
            m = _normalize_mask(candidate)
            key = _mask_key(m)
            if key not in seen:
                seen.add(key)
                out.append(m)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return min(_mask_key(t) for t in _unique_transforms(mask))


def _find_separator(g: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = g.shape
    rows = [
        r
        for r in range(h)
        if int(g[r, 0]) != bg and all(int(v) == int(g[r, 0]) for v in g[r, :])
    ]
    cols = [
        c
        for c in range(w)
        if int(g[0, c]) != bg and all(int(v) == int(g[0, c]) for v in g[:, c])
    ]
    assert len(rows) + len(cols) == 1, "expected exactly one full separator row or column"
    if rows:
        return "row", int(rows[0])
    return "col", int(cols[0])


def _split_regions(
    g: np.ndarray, separator: tuple[str, int]
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, tuple[int, int, int, int]]:
    orient, idx = separator
    h, w = g.shape
    if orient == "row":
        top = (0, idx, 0, w)
        bottom = (idx + 1, h, 0, w)
        assert top[1] > top[0] and bottom[1] > bottom[0]
        legend_box, main_box = (top, bottom) if (idx < h - idx - 1) else (bottom, top)
    else:
        left = (0, h, 0, idx)
        right = (0, h, idx + 1, w)
        assert left[3] > left[2] and right[3] > right[2]
        legend_box, main_box = (left, right) if (idx < w - idx - 1) else (right, left)

    lr0, lr1, lc0, lc1 = legend_box
    mr0, mr1, mc0, mc1 = main_box
    return (
        g[lr0:lr1, lc0:lc1],
        legend_box,
        g[mr0:mr1, mc0:mc1],
        main_box,
    )


def _parse_legend(legend: np.ndarray, bg: int) -> dict[tuple[tuple[int, int], tuple[int, ...]], int]:
    shape_to_color: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for coords in _components(legend != bg):
        colors = Counter(int(legend[int(r), int(c)]) for r, c in coords)
        assert len(colors) == 2, "each legend entry must contain one glyph color and one swatch color"
        swatches = [color for color, count in colors.items() if count == 1]
        assert len(swatches) == 1, "each legend entry must have exactly one single-cell swatch"
        swatch = int(swatches[0])
        glyph_color = next(int(color) for color in colors if int(color) != swatch)
        glyph_mask = np.zeros(legend.shape, dtype=bool)
        for r, c in coords:
            if int(legend[int(r), int(c)]) == glyph_color:
                glyph_mask[int(r), int(c)] = True
        assert int(glyph_mask.sum()) >= 3, "legend glyphs must be larger than the swatch"
        key = _canonical_key(glyph_mask)
        assert key not in shape_to_color, "legend glyph shapes must be unique under rotations/reflections"
        shape_to_color[key] = swatch

    assert len(shape_to_color) >= 2, "expected multiple glyph-to-color legend entries"
    return shape_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a separated glyph-to-swatch legend to recolor every large mask in the main region.

    Input structure:
    - A full separator row or column splits the grid into a narrow legend strip and a larger main region.
    - Each legend entry is one small monochrome glyph touching a single-cell color swatch.
    - The main region contains large monochrome masks, each with one embedded small glyph in a different color.

    Rule:
    - Match each embedded glyph to a legend glyph up to rotations and mirror reflections.
    - Recolor every cell of that mask component, including the embedded glyph, to the matched legend swatch color.
    - Leave the separator and the entire legend strip unchanged.

    Key insight:
    - The glyph shape is the lookup key; the embedded glyph's own color is a deliberate decoy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    separator = _find_separator(g, bg)
    legend, _legend_box, main_region, main_box = _split_regions(g, separator)
    shape_to_color = _parse_legend(legend, bg)

    out = g.copy()
    mr0, _mr1, mc0, _mc1 = main_box
    changed = 0
    for coords in _components(main_region != bg):
        colors = Counter(int(main_region[int(r), int(c)]) for r, c in coords)
        assert len(colors) == 2, "each main mask must contain one body color and one embedded glyph color"
        ranked = colors.most_common()
        assert ranked[0][1] > ranked[1][1], "the mask body must be larger than the embedded glyph"
        glyph_color = int(ranked[1][0])

        glyph_mask = np.zeros(main_region.shape, dtype=bool)
        for r, c in coords:
            if int(main_region[int(r), int(c)]) == glyph_color:
                glyph_mask[int(r), int(c)] = True
        key = _canonical_key(glyph_mask)
        assert key in shape_to_color, "embedded glyph must match exactly one legend glyph"
        new_color = int(shape_to_color[key])

        for r, c in coords:
            out[mr0 + int(r), mc0 + int(c)] = new_color
        changed += 1

    assert changed > 0, "expected at least one main-region mask"
    return out


# --------------- generator ---------------

MAX_RETRIES = 400
COLORS = tuple(range(10))

BASE_GLYPHS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
)


def _paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def _place_horizontal_legend(
    legend: np.ndarray,
    glyphs: list[np.ndarray],
    swatches: list[int],
    glyph_ink: int,
) -> None:
    x = random.randint(1, max(1, legend.shape[1] - sum(g.shape[1] + 3 for g in glyphs)))
    for glyph, swatch in zip(glyphs, swatches):
        gh, gw = glyph.shape
        r0 = random.randint(1, legend.shape[0] - gh - 1)
        c0 = x
        _paste_mask(legend, glyph, r0, c0, glyph_ink)
        edge_rows = [int(r) for r in np.argwhere(glyph[:, -1]).ravel()]
        rr = r0 + random.choice(edge_rows)
        legend[rr, c0 + gw] = int(swatch)
        x += gw + 1 + random.randint(1, 2)


def _place_vertical_legend(
    legend: np.ndarray,
    glyphs: list[np.ndarray],
    swatches: list[int],
    glyph_ink: int,
) -> None:
    y = random.randint(1, max(1, legend.shape[0] - sum(g.shape[0] + 3 for g in glyphs)))
    for glyph, swatch in zip(glyphs, swatches):
        gh, gw = glyph.shape
        r0 = y
        c0 = random.randint(1, legend.shape[1] - gw - 1)
        _paste_mask(legend, glyph, r0, c0, glyph_ink)
        edge_cols = [int(c) for c in np.argwhere(glyph[-1, :]).ravel()]
        cc = c0 + random.choice(edge_cols)
        legend[r0 + gh, cc] = int(swatch)
        y += gh + 1 + random.randint(1, 2)


def _free_rect_locations(canvas: np.ndarray, height: int, width: int, bg: int, padding: int) -> list[tuple[int, int]]:
    h, w = canvas.shape
    locs: list[tuple[int, int]] = []
    for r in range(0, h - height + 1):
        for c in range(0, w - width + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(h, r + height + padding)
            cc1 = min(w, c + width + padding)
            if np.all(canvas[rr0:rr1, cc0:cc1] == bg):
                locs.append((r, c))
    return locs


def _build_object_patch(
    glyph: np.ndarray,
    target_color: int,
    embedded_color: int,
    bg: int,
    sep_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    transforms = _unique_transforms(glyph)
    transformed = random.choice(transforms)
    gh, gw = transformed.shape
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)

    forbidden = {int(bg), int(sep_color), int(target_color), int(embedded_color)}
    body_choices = [c for c in COLORS if c not in forbidden]
    body_color = random.choice(body_choices)

    patch = np.full((gh + top + bottom, gw + left + right), int(body_color), dtype=int)
    glyph_area = patch[top : top + gh, left : left + gw]
    glyph_area[transformed] = int(embedded_color)
    return patch, transformed


def _assemble_grid(
    legend: np.ndarray,
    main_region: np.ndarray,
    bg: int,
    sep_color: int,
    orient: str,
    legend_first: bool,
) -> np.ndarray:
    if orient == "row":
        h = legend.shape[0] + 1 + main_region.shape[0]
        w = legend.shape[1]
        grid = np.full((h, w), int(bg), dtype=int)
        if legend_first:
            grid[: legend.shape[0], :] = legend
            grid[legend.shape[0], :] = int(sep_color)
            grid[legend.shape[0] + 1 :, :] = main_region
        else:
            grid[: main_region.shape[0], :] = main_region
            grid[main_region.shape[0], :] = int(sep_color)
            grid[main_region.shape[0] + 1 :, :] = legend
        return grid

    h = legend.shape[0]
    w = legend.shape[1] + 1 + main_region.shape[1]
    grid = np.full((h, w), int(bg), dtype=int)
    if legend_first:
        grid[:, : legend.shape[1]] = legend
        grid[:, legend.shape[1]] = int(sep_color)
        grid[:, legend.shape[1] + 1 :] = main_region
    else:
        grid[:, : main_region.shape[1]] = main_region
        grid[:, main_region.shape[1]] = int(sep_color)
        grid[:, main_region.shape[1] + 1 :] = legend
    return grid


def _expected_from_main(grid: np.ndarray) -> np.ndarray | None:
    try:
        return main(grid)
    except Exception:
        return None


def _gen_attempt() -> np.ndarray | None:
    n_entries = random.choice((3, 4))
    glyphs = [g.copy() for g in random.sample(BASE_GLYPHS, n_entries)]
    if len({_canonical_key(g) for g in glyphs}) != n_entries:
        return None

    bg = random.choice(COLORS)
    sep_color = random.choice([c for c in COLORS if c != bg])
    swatches = random.sample([c for c in COLORS if c not in (bg, sep_color)], n_entries)
    ink_choices = [c for c in COLORS if c not in {bg, sep_color, *swatches}]
    if not ink_choices:
        return None
    glyph_ink = random.choice(ink_choices)

    orient = random.choice(("row", "col"))
    legend_first = bool(random.getrandbits(1))

    max_gh = max(int(g.shape[0]) for g in glyphs)
    max_gw = max(int(g.shape[1]) for g in glyphs)
    if orient == "row":
        legend_h = max_gh + 2
        needed_w = sum(int(g.shape[1]) + 3 for g in glyphs) + 1
        if needed_w > 30:
            return None
        width = random.randint(max(needed_w, 22), 30)
        max_main_h = 30 - legend_h - 1
        if max_main_h < 14:
            return None
        main_h = random.randint(14, max_main_h)
        legend = np.full((legend_h, width), int(bg), dtype=int)
        main_region = np.full((main_h, width), int(bg), dtype=int)
        _place_horizontal_legend(legend, glyphs, swatches, glyph_ink)
    else:
        legend_w = max_gw + 2
        needed_h = sum(int(g.shape[0]) + 3 for g in glyphs) + 1
        if needed_h > 30:
            return None
        height = random.randint(max(needed_h, 22), 30)
        max_main_w = 30 - legend_w - 1
        if max_main_w < 14:
            return None
        main_w = random.randint(14, max_main_w)
        legend = np.full((height, legend_w), int(bg), dtype=int)
        main_region = np.full((height, main_w), int(bg), dtype=int)
        _place_vertical_legend(legend, glyphs, swatches, glyph_ink)

    order = list(range(n_entries))
    random.shuffle(order)
    color_shift = random.randint(1, n_entries - 1)
    expected_main = main_region.copy()
    wrong_by_glyph_color = main_region.copy()
    used_changed_transform = False

    for idx in order:
        target = int(swatches[idx])
        embedded = int(swatches[(idx + color_shift) % n_entries])
        patch, transformed = _build_object_patch(glyphs[idx], target, embedded, bg, sep_color)
        if _mask_key(transformed) != _mask_key(glyphs[idx]):
            used_changed_transform = True
        ph, pw = patch.shape
        locs = _free_rect_locations(main_region, ph, pw, bg, padding=1)
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        main_region[r0 : r0 + ph, c0 : c0 + pw] = patch
        expected_main[r0 : r0 + ph, c0 : c0 + pw] = target
        wrong_by_glyph_color[r0 : r0 + ph, c0 : c0 + pw] = embedded

    if not used_changed_transform:
        return None

    grid = _assemble_grid(legend, main_region, bg, sep_color, orient, legend_first)
    expected = _assemble_grid(legend, expected_main, bg, sep_color, orient, legend_first)
    wrong = _assemble_grid(legend, wrong_by_glyph_color, bg, sep_color, orient, legend_first)

    if max(grid.shape) > 30:
        return None
    if _get_bg(grid) != bg:
        return None
    actual = _expected_from_main(grid)
    if actual is None or not np.array_equal(actual, expected):
        return None
    if np.array_equal(wrong, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    if len({int(c) for c in swatches}) != n_entries:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shape-dictionary input")

from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    GREY = 5
    TEAL = 8


MARKER_COLORS = (1, 2, 3, 4, 6, 7, 9)
GRID_LIMIT = 30
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sprite_offsets(grid: np.ndarray, coords: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    colors = [int(grid[r, c]) for r, c in coords]
    assert colors.count(Color.GREY) == 1, "each sprite must contain exactly one grey anchor"
    palette = set(colors)
    assert len(palette) == 2 and Color.GREY in palette, "sprite must be grey anchor plus one body color"
    body_color = next(color for color in palette if color != Color.GREY)
    anchor_idx = colors.index(Color.GREY)
    ar, ac = map(int, coords[anchor_idx])
    offsets = {(int(r) - ar, int(c) - ac) for r, c in coords}
    return offsets, int(body_color)


def _template_patch(template: set[tuple[int, int]]) -> np.ndarray:
    rows = [dr for dr, _dc in template]
    cols = [dc for _dr, dc in template]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for dr, dc in template:
        patch[dr - r0, dc - c0] = True
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden anchored template and emit recolored copies ordered by the marker line.

    Input structure:
    - A black background contains several noisy sprites drawn in one body color.
    - Every sprite contains exactly one grey anchor pixel inside the connected component.
    - Elsewhere there is a row or column of isolated single-pixel markers whose colors are all distinct and are neither the body color nor grey.

    Rule:
    - Split the non-background cells into connected components.
    - Treat every multi-pixel component with one grey cell as a sprite, align all sprites on that grey anchor, and intersect their occupied offsets to recover the common template including the anchor position.
    - Sort the single-pixel markers left-to-right when they share a row, or top-to-bottom when they share a column.
    - Recolor the recovered template once per marker color and concatenate the template crops in the marker order along that axis.

    Key insight:
    - The true template is never shown by itself, so bbox alignment is unreliable; only anchor-aligned occupancy intersection removes the different extra pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    sprites: list[set[tuple[int, int]]] = []
    body_colors: list[int] = []
    markers: list[tuple[int, int, int]] = []

    for coords in _components(g != bg):
        if len(coords) == 1:
            r, c = map(int, coords[0])
            markers.append((r, c, int(g[r, c])))
            continue
        offsets, body_color = _sprite_offsets(g, coords)
        sprites.append(offsets)
        body_colors.append(body_color)

    assert len(sprites) >= 2, "expected at least two sprites"
    assert markers, "expected marker pixels"
    assert len(set(body_colors)) == 1, "all sprites must share one body color"
    body_color = body_colors[0]

    marker_colors = [color for _r, _c, color in markers]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be distinct"
    assert all(color not in {bg, Color.GREY, body_color} for color in marker_colors)

    template = set.intersection(*sprites)
    assert template and (0, 0) in template, "anchor must remain in the common template"
    assert all(len(template) < len(sprite) for sprite in sprites), "no sprite may equal the template itself"

    same_row = len({r for r, _c, _color in markers}) == 1
    same_col = len({c for _r, c, _color in markers}) == 1
    assert same_row or same_col, "markers must share a row or a column"
    axis = 1 if same_row else 0
    key = (lambda item: item[1]) if same_row else (lambda item: item[0])
    markers.sort(key=key)

    mask = _template_patch(template)
    copies = []
    for _r, _c, color in markers:
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = int(color)
        copies.append(patch)
    return np.concatenate(copies, axis=axis)


def _random_template() -> set[tuple[int, int]] | None:
    for _ in range(100):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        ar = random.randint(0, h - 1)
        ac = random.randint(0, w - 1)
        max_area = min(8, h * w - 1)
        if max_area < 4:
            continue
        target_area = random.randint(4, max_area)

        cells = {(ar, ac)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        rel = {(r - ar, c - ac) for r, c in cells}
        rows = {dr for dr, _dc in rel}
        cols = {dc for _dr, dc in rel}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(rel) == h * w:
            continue
        return rel
    return None


def _boundary_candidates(template: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [dr for dr, _dc in template]
    cols = [dc for _dr, dc in template]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    candidates = set()
    for c in range(c0, c1 + 1):
        if (r0, c) in template:
            candidates.add((r0 - 1, c))
        if (r1, c) in template:
            candidates.add((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if (r, c0) in template:
            candidates.add((r, c0 - 1))
        if (r, c1) in template:
            candidates.add((r, c1 + 1))
    return [coord for coord in candidates if coord not in template]


def _secondary_candidates(template: set[tuple[int, int]], extra: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for cand in _neighbors4(*extra):
        if cand in template or cand == (0, 0):
            continue
        if cand not in out:
            out.append(cand)
    return out


def _offsets_to_patch(offsets: set[tuple[int, int]], body_color: int) -> np.ndarray:
    rows = [dr for dr, _dc in offsets]
    cols = [dc for _dr, dc in offsets]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for dr, dc in offsets:
        patch[dr - r0, dc - c0] = body_color
    patch[-r0, -c0] = Color.GREY
    return patch


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_sprites(grid: np.ndarray, patches: list[np.ndarray], region: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = region
    placed_rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].size, reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph > r1 - r0 + 1 or pw > c1 - c0 + 1:
            return False
        ok = False
        for _ in range(200):
            top = random.randint(r0, r1 - ph + 1)
            left = random.randint(c0, c1 - pw + 1)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(_rects_too_close(rect, other, gap=1) for other in placed_rects):
                continue
            target = grid[top : top + ph, left : left + pw]
            if np.any((patch != 0) & (target != 0)):
                continue
            target[patch != 0] = patch[patch != 0]
            placed_rects.append(rect)
            ok = True
            break
        if not ok:
            return False
    return True


def _crop_with_padding(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != Color.BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pad_top = random.randint(0, min(2, int(r0)))
    pad_left = random.randint(0, min(2, int(c0)))
    pad_bottom = random.randint(0, min(2, GRID_LIMIT - 1 - int(r1)))
    pad_right = random.randint(0, min(2, GRID_LIMIT - 1 - int(c1)))
    return grid[
        int(r0) - pad_top : int(r1) + pad_bottom + 1,
        int(c0) - pad_left : int(c1) + pad_right + 1,
    ]


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        template = _random_template()
        if template is None:
            continue

        orientation = random.choice(("horizontal", "vertical"))
        mask = _template_patch(template)
        unit = mask.shape[1] if orientation == "horizontal" else mask.shape[0]
        max_markers = min(len(MARKER_COLORS), 5, GRID_LIMIT // unit)
        if max_markers < 2:
            continue

        sprite_count = random.randint(3, 4)
        boundary = _boundary_candidates(template)
        if len(boundary) < sprite_count:
            continue

        marker_count = random.randint(2, max_markers)
        chosen = random.sample(boundary, sprite_count)
        used_extras = set()
        patches = []
        for extra in chosen:
            extras = {extra}
            used_extras.add(extra)
            if random.random() < 0.6:
                secondary_options = [
                    cand
                    for cand in _secondary_candidates(template, extra)
                    if cand not in used_extras
                ]
                if secondary_options:
                    second = random.choice(secondary_options)
                    extras.add(second)
                    used_extras.add(second)
            patches.append(_offsets_to_patch(template | extras, Color.TEAL))

        grid = np.zeros((GRID_LIMIT, GRID_LIMIT), dtype=int)
        marker_colors = random.sample(MARKER_COLORS, marker_count)

        if orientation == "horizontal":
            top_band = random.choice((True, False))
            marker_row = random.randint(1, 3) if top_band else random.randint(26, 28)
            step = random.choice((2, 3))
            span = 1 + step * (marker_count - 1)
            start_c = random.randint(1, GRID_LIMIT - span - 1)
            for idx, color in enumerate(marker_colors):
                grid[marker_row, start_c + idx * step] = color
            region = (7, 1, 28, 28) if top_band else (1, 1, 22, 28)
        else:
            left_band = random.choice((True, False))
            marker_col = random.randint(1, 3) if left_band else random.randint(26, 28)
            step = random.choice((2, 3))
            span = 1 + step * (marker_count - 1)
            start_r = random.randint(1, GRID_LIMIT - span - 1)
            for idx, color in enumerate(marker_colors):
                grid[start_r + idx * step, marker_col] = color
            region = (1, 7, 28, 28) if left_band else (1, 1, 28, 22)

        if not _place_sprites(grid, patches, region):
            continue

        final = _crop_with_padding(grid)
        try:
            out = main(final)
        except AssertionError:
            continue
        if final.ndim != 2 or out.ndim != 2:
            continue
        if max(final.shape) > GRID_LIMIT or max(out.shape) > GRID_LIMIT:
            continue
        return final

    raise RuntimeError("failed to generate a valid ghost-template input")

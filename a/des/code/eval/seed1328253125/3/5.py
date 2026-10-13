from __future__ import annotations

from collections import deque

import numpy as np


GENERATION_ATTEMPTS = 300
SEPARATOR_COLOR = 5


def _boundary_background(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    if g.shape[0] == 1:
        border = g[0]
    elif g.shape[1] == 1:
        border = g[:, 0]
    else:
        border = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _rectangles_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_sample_pair(
    grid: np.ndarray, bg: int
) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], int]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        infos = []
        for coords in _components(g == int(color)):
            bbox = _bbox(coords)
            crop = _crop(g, bbox)
            infos.append(
                {
                    "bbox": bbox,
                    "crop": crop,
                    "shape": crop.shape,
                    "area": int(crop.size),
                    "size": int(coords.shape[0]),
                }
            )

        solids = [info for info in infos if info["size"] == info["area"]]
        partials = [info for info in infos if info["size"] < info["area"]]
        for top in solids:
            tr0, tr1, tc0, tc1 = top["bbox"]
            for bottom in partials:
                br0, br1, bc0, bc1 = bottom["bbox"]
                if top["shape"] != bottom["shape"]:
                    continue
                if (tc0, tc1) != (bc0, bc1):
                    continue
                if br0 != tr1 + 2:
                    continue
                crop = bottom["crop"]
                if set(map(int, np.unique(crop))) - {int(bg), int(color)}:
                    continue
                if crop.shape[0] < 3 or crop.shape[1] < 3:
                    continue
                if not np.any(crop[1:-1, 1:-1] == int(bg)):
                    continue
                if not (
                    np.all(crop[0, :] == int(color))
                    and np.all(crop[-1, :] == int(color))
                    and np.all(crop[:, 0] == int(color))
                    and np.all(crop[:, -1] == int(color))
                ):
                    continue

                sep_row = g[tr1 + 1, tc0 : tc1 + 1]
                if not np.all(sep_row == sep_row[0]):
                    continue
                sep_color = int(sep_row[0])
                if sep_color in (int(bg), int(color)):
                    continue

                candidates.append((int(color), top["bbox"], bottom["bbox"], sep_color))

    assert len(candidates) == 1, f"expected one sample pair, found {len(candidates)}"
    return candidates[0]


def _find_palette(
    grid: np.ndarray,
    bg: int,
    shape: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = shape
    gh, gw = g.shape
    candidates = []
    for r0 in range(gh - h + 1):
        for c0 in range(gw - w + 1):
            bbox = (r0, r0 + h - 1, c0, c0 + w - 1)
            if any(_rectangles_intersect(bbox, other) for other in forbidden):
                continue
            patch = g[r0 : r0 + h, c0 : c0 + w]
            if np.any(patch == int(bg)):
                continue
            if len(np.unique(patch)) < 2:
                continue
            candidates.append(bbox)
    assert len(candidates) == 1, f"expected one palette, found {len(candidates)}"
    return candidates[0]


def _masked_palette_crop(bg: int, palette: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(hole_mask)
    assert rows.size > 0, "hole mask must be non-empty"
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    submask = hole_mask[r0 : r1 + 1, c0 : c1 + 1]
    out = np.full(submask.shape, int(bg), dtype=int)
    patch = palette[r0 : r1 + 1, c0 : c1 + 1]
    out[submask] = patch[submask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Derive the missing-hole mask from a complete sample and apply it to a palette strip.

    Input structure:
    - Background fills most of the grid.
    - Two equal-size sample rectangles appear on the left, separated by a single grey row.
    - The upper sample is a solid monochrome rectangle.
    - The lower sample uses the same color and bbox, but some interior cells are replaced by bg.
    - A same-size multicolor rectangle appears elsewhere as the palette.

    Rule:
    - Compare the complete and incomplete samples to recover the missing-cell mask.
    - Keep only palette cells under that mask.
    - Crop the result to the mask's tight bounding box.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_background(g)

    _sample_color, full_bbox, broken_bbox, _sep_color = _find_sample_pair(g, bg)
    full = _crop(g, full_bbox)
    broken = _crop(g, broken_bbox)
    palette_bbox = _find_palette(g, bg, full.shape, [full_bbox, broken_bbox])
    palette = _crop(g, palette_bbox)

    hole_mask = (full != int(bg)) & (broken == int(bg))
    return _masked_palette_crop(bg, palette, hole_mask)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _random_walk_shape(height: int, width: int, area: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(rng.integers(height)), int(rng.integers(width)))
    mask[start] = True
    frontier = [start]
    while int(mask.sum()) < area:
        r, c = frontier[int(rng.integers(len(frontier)))]
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            filled = np.argwhere(mask)
            if len(filled) == mask.size:
                break
            frontier = [tuple(cell) for cell in filled]
            continue
        nr, nc = neighbors[int(rng.integers(len(neighbors)))]
        mask[nr, nc] = True
        frontier.append((nr, nc))
    return mask


def _random_component_shape(
    max_h: int, max_w: int, rng: np.random.Generator
) -> np.ndarray | None:
    kinds = ("rectangle", "el", "plus", "walk")
    kind = kinds[int(rng.integers(len(kinds)))]

    if kind == "rectangle":
        h = int(rng.integers(1, min(max_h, 4) + 1))
        w = int(rng.integers(1, min(max_w, 4) + 1))
        return np.ones((h, w), dtype=bool)

    if kind == "el":
        if max_h < 2 or max_w < 2:
            return None
        h = int(rng.integers(2, min(max_h, 5) + 1))
        w = int(rng.integers(2, min(max_w, 5) + 1))
        mask = np.zeros((h, w), dtype=bool)
        orientation = int(rng.integers(4))
        if orientation == 0:
            mask[:, 0] = True
            mask[-1, :] = True
        elif orientation == 1:
            mask[:, -1] = True
            mask[-1, :] = True
        elif orientation == 2:
            mask[:, 0] = True
            mask[0, :] = True
        else:
            mask[:, -1] = True
            mask[0, :] = True
        return mask

    if kind == "plus":
        if max_h < 3 or max_w < 3:
            return None
        h_choices = [value for value in range(3, min(max_h, 7) + 1) if value % 2 == 1]
        w_choices = [value for value in range(3, min(max_w, 7) + 1) if value % 2 == 1]
        if not h_choices or not w_choices:
            return None
        h = int(h_choices[int(rng.integers(len(h_choices)))])
        w = int(w_choices[int(rng.integers(len(w_choices)))])
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
        return mask

    if max_h < 2 or max_w < 2:
        return None
    h = int(rng.integers(2, min(max_h, 5) + 1))
    w = int(rng.integers(2, min(max_w, 5) + 1))
    max_area = min(h * w, max(3, (h * w + 1) // 2))
    area = int(rng.integers(2, max_area + 1))
    return _random_walk_shape(h, w, area, rng)


def _make_hole_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    inner_h = height - 2
    inner_w = width - 2
    if inner_h <= 0 or inner_w <= 0:
        return None

    inner = np.zeros((inner_h, inner_w), dtype=bool)
    target_parts = int(rng.integers(1, 4))
    for _ in range(target_parts):
        placed = False
        forbidden = _dilate8(inner)
        for _ in range(80):
            shape = _random_component_shape(inner_h, inner_w, rng)
            if shape is None:
                continue
            sh, sw = shape.shape
            if sh > inner_h or sw > inner_w:
                continue
            positions = []
            for r0 in range(inner_h - sh + 1):
                for c0 in range(inner_w - sw + 1):
                    if np.any(forbidden[r0 : r0 + sh, c0 : c0 + sw] & shape):
                        continue
                    positions.append((r0, c0))
            if not positions:
                continue
            r0, c0 = positions[int(rng.integers(len(positions)))]
            inner[r0 : r0 + sh, c0 : c0 + sw] |= shape
            placed = True
            break
        if not placed:
            return None

    if not inner.any():
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[1:-1, 1:-1] = inner
    return mask


def _ensure_multicolor(palette: np.ndarray, colors: np.ndarray) -> np.ndarray:
    if len(np.unique(palette)) >= 2:
        return palette
    palette = palette.copy()
    alt = int(colors[1 % len(colors)])
    palette[0, 0] = alt
    return palette


def _make_palette(height: int, width: int, colors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(colors)
    rr, cc = np.indices((height, width))
    mode = int(rng.integers(4))

    if mode == 0:
        palette = rng.choice(colors, size=(height, width), replace=True)
        return _ensure_multicolor(palette.astype(int), colors)

    if mode == 1:
        stripe = rng.choice(colors, size=width, replace=True)
        palette = np.tile(stripe, (height, 1))
        return _ensure_multicolor(palette.astype(int), colors)

    if mode == 2:
        stripe = rng.choice(colors, size=height, replace=True)
        palette = np.tile(stripe[:, None], (1, width))
        return _ensure_multicolor(palette.astype(int), colors)

    step_r = int(rng.integers(1, min(height, 3) + 1))
    step_c = int(rng.integers(1, min(width, 3) + 1))
    palette = colors[((rr // step_r) + 2 * (cc // step_c) + rr + cc) % n]
    return _ensure_multicolor(palette.astype(int), colors)


def _choose_colors(rng: np.random.Generator) -> tuple[int, int, int, np.ndarray]:
    all_colors = [c for c in range(10) if c != SEPARATOR_COLOR]
    bg = int(rng.choice(all_colors))
    remaining = [c for c in all_colors if c != bg]
    sample = int(rng.choice(remaining))
    remaining = [c for c in remaining if c != sample]
    palette_count = int(rng.integers(2, min(5, len(remaining)) + 1))
    palette_colors = rng.choice(remaining, size=palette_count, replace=False).astype(int)
    return bg, int(SEPARATOR_COLOR), sample, palette_colors


def _gen_attempt(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    bg, sep_color, sample_color, palette_colors = _choose_colors(rng)
    sample_h = int(rng.integers(5, 11))
    sample_w = int(rng.integers(5, 12))
    holes = _make_hole_mask(sample_h, sample_w, rng)
    if holes is None:
        return None

    palette = _make_palette(sample_h, sample_w, palette_colors, rng)
    expected = _masked_palette_crop(bg, palette, holes)

    top_margin = int(rng.integers(1, 5))
    left_margin = int(rng.integers(1, 5))
    right_gap = int(rng.integers(2, 6))
    right_margin = int(rng.integers(1, 5))
    bottom_margin = int(rng.integers(1, 5))
    palette_offset = int(rng.integers(0, sample_h + 2))

    stack_h = 2 * sample_h + 1
    content_h = max(stack_h, palette_offset + sample_h)
    grid_h = top_margin + content_h + bottom_margin
    palette_left = left_margin + sample_w + right_gap
    grid_w = palette_left + sample_w + right_margin

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    top_r = top_margin
    left_c = left_margin
    bottom_r = top_r + sample_h + 1
    palette_r = top_margin + palette_offset

    grid[top_r : top_r + sample_h, left_c : left_c + sample_w] = sample_color
    grid[bottom_r : bottom_r + sample_h, left_c : left_c + sample_w] = sample_color
    broken = grid[bottom_r : bottom_r + sample_h, left_c : left_c + sample_w]
    broken[holes] = bg
    grid[top_r + sample_h, left_c : left_c + sample_w] = sep_color
    grid[palette_r : palette_r + sample_h, palette_left : palette_left + sample_w] = palette

    assert np.array_equal(main(grid), expected)
    return grid, expected


def _generate_example() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        example = _gen_attempt(rng)
        if example is not None:
            return example
    raise RuntimeError("failed to generate valid input")


def generate_input() -> np.ndarray:
    return _generate_example()[0]

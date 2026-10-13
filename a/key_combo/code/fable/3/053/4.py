from __future__ import annotations

import random

import numpy as np


BG = 0
NEUTRAL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
MASK_RETRIES = 400
LAYOUT_RETRIES = 120


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(grid != bg)
    assert box is not None, "expected a non-empty foreground region"
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _crop_true(mask)
    variants = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for candidate in (rot, np.fliplr(rot)):
            cropped = _crop_true(candidate)
            key = (tuple(map(int, cropped.shape)), cropped.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cropped)
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, v.shape)), v.tobytes()) for v in _dihedral_variants(mask))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_sections(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int]]:
    nonempty_rows = np.any(grid != bg, axis=1)
    rows = np.flatnonzero(nonempty_rows)
    assert rows.size > 0

    top_r0 = int(rows[0])
    top_r1 = top_r0
    while top_r1 + 1 < grid.shape[0] and bool(nonempty_rows[top_r1 + 1]):
        top_r1 += 1

    lower_rows = rows[rows > top_r1]
    assert lower_rows.size > 0, "expected a lower index grid beneath the templates"
    lower_r0 = int(lower_rows[0])
    lower_r1 = int(lower_rows[-1])
    return (top_r0, top_r1), (lower_r0, lower_r1)


def _extract_templates(top: np.ndarray, bg: int) -> list[np.ndarray]:
    colors = {int(x) for x in np.unique(top)} - {int(bg)}
    assert len(colors) == 1, "top legend must use one neutral template color"

    k = int(top.shape[0])
    assert k >= 2
    non_bg_cols = np.any(top != bg, axis=0)

    templates = []
    c = 0
    while c < top.shape[1]:
        if not bool(non_bg_cols[c]):
            c += 1
            continue
        assert c + k <= top.shape[1], "template patch must be square"
        patch = top[:, c : c + k]
        assert np.any(patch != bg)
        assert set(map(int, np.unique(patch))) <= {int(bg), *colors}
        templates.append(patch.copy())
        c += k
    assert len(templates) >= 2
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the lower index grid by stamping each pixel with the stencil identified by its color's distribution shape.

    Input structure:
    - The top band contains several same-sized neutral square templates, separated by background gaps.
    - Each template is a solid neutral square with background-colored holes.
    - Below a blank gap sits a lower color index grid whose repeated colors form shape clues.

    Rule:
    - For each lower-grid color, collect all of its cells, normalize them to their tight bbox, and match that shape to exactly one template's hole pattern up to rotation or reflection.
    - Keep the matched template in its original orientation.
    - Output the lower grid scaled by the template size: every non-background cell becomes one recolored template block whose solid cells take that cell's color and whose holes stay background.

    Key insight:
    - The lower color layout is both the picture to enlarge and the legend that identifies which stencil each color uses; counts alone are insufficient because different templates can share the same number of holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    (top_r0, top_r1), (lower_r0, lower_r1) = _find_sections(g, bg)
    top = g[top_r0 : top_r1 + 1, :]
    lower = _crop_non_bg(g[lower_r0 : lower_r1 + 1, :], bg)

    templates = _extract_templates(top, bg)
    k = int(templates[0].shape[0])
    assert all(t.shape == (k, k) for t in templates)

    template_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    solid_masks: list[np.ndarray] = []
    for patch in templates:
        solid = patch != bg
        holes = patch == bg
        key = _canonical_key(holes)
        assert key not in template_by_key, "template hole shapes must be unique up to D4"
        template_by_key[key] = len(solid_masks)
        solid_masks.append(solid)

    colors = [int(x) for x in np.unique(lower) if int(x) != int(bg)]
    assert len(colors) == len(templates), "expected one lower-grid color per template"

    color_to_template: dict[int, int] = {}
    used_templates: set[int] = set()
    for color in colors:
        color_mask = lower == color
        key = _canonical_key(color_mask)
        assert key in template_by_key, "every color distribution must match a template hole shape"
        idx = int(template_by_key[key])
        assert idx not in used_templates, "template matches must be one-to-one"
        used_templates.add(idx)
        color_to_template[color] = idx

    out = np.full((lower.shape[0] * k, lower.shape[1] * k), int(bg), dtype=int)
    for r in range(lower.shape[0]):
        for c in range(lower.shape[1]):
            color = int(lower[r, c])
            if color == int(bg):
                continue
            solid = solid_masks[color_to_template[color]]
            block = out[r * k : (r + 1) * k, c * k : (c + 1) * k]
            block[solid] = color
    return out


def _random_hole_mask(k: int, area: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(k) for c in range(k)]
    for _ in range(MASK_RETRIES):
        chosen = random.sample(cells, area)
        mask = np.zeros((k, k), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        if np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)):
            continue
        cropped = _crop_true(mask)
        if min(cropped.shape) < 2:
            continue
        return mask
    return None


def _make_template_set(k: int, count: int) -> list[np.ndarray] | None:
    max_area = min(6, k * k - 2)
    area_choices = list(range(3, max_area + 1))
    if not area_choices:
        return None

    duplicate_area = random.choice(area_choices)
    specs = [duplicate_area, duplicate_area]
    for _ in range(count - 2):
        specs.append(random.choice(area_choices))
    random.shuffle(specs)

    masks = []
    keys: set[tuple[tuple[int, int], bytes]] = set()
    same_area_keys: dict[int, list[tuple[tuple[int, int], bytes]]] = {}
    for area in specs:
        placed = False
        for _ in range(MASK_RETRIES):
            mask = _random_hole_mask(k, area)
            if mask is None:
                break
            key = _canonical_key(mask)
            if key in keys:
                continue
            keys.add(key)
            masks.append(mask)
            same_area_keys.setdefault(area, []).append(key)
            placed = True
            break
        if not placed:
            return None

    assert any(len(v) >= 2 for v in same_area_keys.values())
    return masks


def _random_oriented(mask: np.ndarray) -> np.ndarray:
    variant = _crop_true(mask)
    variant = np.rot90(variant, random.randrange(4))
    if random.random() < 0.5:
        variant = np.fliplr(variant)
    return _crop_true(variant)


def _layout_index_shapes(items: list[dict], max_side: int) -> np.ndarray | None:
    for _ in range(LAYOUT_RETRIES):
        shuffled = items[:]
        random.shuffle(shuffled)
        row_count = random.randint(1, min(2, len(shuffled)))
        rows = [[] for _ in range(row_count)]
        for item in shuffled:
            rows[random.randrange(row_count)].append(item)
        if any(not row for row in rows):
            continue

        row_gap = random.randint(1, 2)
        row_specs = []
        total_h = 0
        total_w = 0
        ok = True
        for row in rows:
            random.shuffle(row)
            gaps = [random.randint(1, 2) for _ in range(max(0, len(row) - 1))]
            width = sum(int(item["shape"].shape[1]) for item in row) + sum(gaps)
            height = max(int(item["shape"].shape[0]) for item in row)
            if width > max_side or height > max_side:
                ok = False
                break
            row_specs.append((row, gaps, height, width))
            total_h += height
            total_w = max(total_w, width)
        total_h += row_gap * (row_count - 1)
        if not ok or total_h > max_side or total_w > max_side:
            continue

        grid = np.full((total_h, total_w), BG, dtype=int)
        y = 0
        for row, gaps, row_h, row_w in row_specs:
            x = random.randint(0, total_w - row_w)
            for idx, item in enumerate(row):
                shape = item["shape"]
                h, w = map(int, shape.shape)
                y0 = y + random.randint(0, row_h - h)
                patch = grid[y0 : y0 + h, x : x + w]
                patch[shape] = int(item["color"])
                if idx < len(gaps):
                    x += w + gaps[idx]
                else:
                    x += w
            y += row_h + row_gap
        return _crop_non_bg(grid, BG)
    return None


def _build_template_band(hole_masks: list[np.ndarray]) -> np.ndarray:
    k = int(hole_masks[0].shape[0])
    gaps = [random.randint(1, 2) for _ in range(len(hole_masks) - 1)]
    width = len(hole_masks) * k + sum(gaps)
    top = np.full((k, width), BG, dtype=int)
    x = 0
    for idx, mask in enumerate(hole_masks):
        patch = np.full((k, k), NEUTRAL, dtype=int)
        patch[mask] = BG
        top[:, x : x + k] = patch
        x += k
        if idx < len(gaps):
            x += gaps[idx]
    return top


def _expected_output(lower: np.ndarray, solids_by_color: dict[int, np.ndarray], bg: int) -> np.ndarray:
    k = int(next(iter(solids_by_color.values())).shape[0])
    out = np.full((lower.shape[0] * k, lower.shape[1] * k), int(bg), dtype=int)
    for r in range(lower.shape[0]):
        for c in range(lower.shape[1]):
            color = int(lower[r, c])
            if color == int(bg):
                continue
            solid = solids_by_color[color]
            block = out[r * k : (r + 1) * k, c * k : (c + 1) * k]
            block[solid] = color
    return out


def _compose_input(top: np.ndarray, lower: np.ndarray) -> np.ndarray:
    gap_rows = random.randint(1, 3)
    bottom_margin = random.randint(0, 2)
    top_left = random.randint(0, 2)
    lower_left = random.randint(0, 2)
    width = max(top_left + top.shape[1], lower_left + lower.shape[1]) + random.randint(0, 2)
    height = top.shape[0] + gap_rows + lower.shape[0] + bottom_margin
    grid = np.full((height, width), BG, dtype=int)
    grid[: top.shape[0], top_left : top_left + top.shape[1]] = top
    lower_r0 = top.shape[0] + gap_rows
    grid[lower_r0 : lower_r0 + lower.shape[0], lower_left : lower_left + lower.shape[1]] = lower
    return grid


def _gen_attempt() -> np.ndarray | None:
    k = random.choice([3, 4])
    count = random.randint(2, 4)
    hole_masks = _make_template_set(k, count)
    if hole_masks is None:
        return None

    colors = random.sample(PALETTE, count)
    items = []
    for color, mask in zip(colors, hole_masks):
        items.append({"color": color, "shape": _random_oriented(mask)})

    lower = _layout_index_shapes(items, max_side=30 // k)
    if lower is None:
        return None

    top = _build_template_band(hole_masks)
    grid = _compose_input(top, lower)
    expected = _expected_output(lower, {color: ~mask for color, mask in zip(colors, hole_masks)}, BG)

    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if max(out.shape) > 30 or not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")

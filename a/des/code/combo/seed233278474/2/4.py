from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(grid: np.ndarray, ignored_colors: set[int]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(g[r, c])
            if color in ignored_colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for c in range(g.shape[1]):
        col = g[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    sep_col, sep_color = candidates[0]
    assert 0 < sep_col < g.shape[1] - 1, "separator must split the grid into two non-empty panels"
    return sep_col, sep_color


def _is_frame_component(comp: dict) -> bool:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.ones((h, w), dtype=bool)
    border[1:-1, 1:-1] = False
    return bool(np.array_equal(mask, border))


def _tile_pattern(tile_patch: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    th, tw = tile_patch.shape
    hh, ww = shape
    rows = np.arange(hh)[:, None] % th
    cols = np.arange(ww)[None, :] % tw
    return tile_patch[rows, cols]


def _aligned_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    hh, ww = shape
    out = np.zeros((hh, ww), dtype=bool)
    mh, mw = mask.shape
    use_h = min(hh, mh)
    use_w = min(ww, mw)
    out[:use_h, :use_w] = mask[:use_h, :use_w]
    return out


def _render_masked_texture(template_mask: np.ndarray, tile_patch: np.ndarray, bg: int, shape: tuple[int, int]) -> np.ndarray:
    tiled = _tile_pattern(tile_patch, shape)
    keep = _aligned_mask(template_mask, shape)
    out = np.full(shape, bg, dtype=int)
    out[keep] = tiled[keep]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each right-panel frame by combining a left-panel template mask with a right-panel texture tile.

    Input structure:
    - Black background with a full-height gray vertical separator.
    - Left panel: monochrome solid template objects, one color per template.
    - Right panel: color-matched hollow frames, each with a single interior seed pixel.
    - Right panel also contains small 2x2 or 3x3 monochrome texture tiles keyed by seed color.

    Rule:
    - For each frame, use its border color to find the left template mask.
    - Use the frame's seed color to find the matching texture tile on the right.
    - Tile the texture across the frame interior and keep only the cells covered by the aligned template mask.
    - Overwrite the seed with the resulting texture. Leave templates, tiles, and frame borders unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _most_common_color(g)
    sep_col, sep_color = _find_separator(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    ignored = {bg, sep_color}

    template_by_color: dict[int, np.ndarray] = {}
    for comp in _components(left, ignored):
        color = int(comp["color"])
        assert color not in template_by_color, "template colors must be unique on the left panel"
        template_by_color[color] = comp["mask"].copy()
    assert template_by_color, "expected at least one left-panel template"

    frame_components: list[dict] = []
    tile_by_color: dict[int, np.ndarray] = {}
    for comp in _components(right, ignored):
        color = int(comp["color"])
        if _is_frame_component(comp):
            frame_components.append(comp)
            continue
        if int(comp["area"]) == 1:
            continue
        tile = np.full(comp["mask"].shape, bg, dtype=int)
        tile[comp["mask"]] = color
        assert tile.shape[0] == tile.shape[1] and tile.shape[0] in (2, 3), "texture tiles must be 2x2 or 3x3"
        assert color not in tile_by_color, "texture colors must identify a unique tile"
        tile_by_color[color] = tile
    assert frame_components, "expected at least one right-panel frame"

    out = g.copy()
    right_out = out[:, sep_col + 1 :]
    for frame in frame_components:
        frame_color = int(frame["color"])
        assert frame_color in template_by_color, "every frame color must match a left template"
        r0, c0, r1, c1 = frame["bbox"]
        interior = right[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere(interior != bg)
        assert seed_coords.shape[0] == 1, "each frame interior must contain exactly one seed pixel"
        seed_r, seed_c = map(int, seed_coords[0])
        seed_color = int(interior[seed_r, seed_c])
        assert seed_color in tile_by_color, "each seed color must match a texture tile"

        filled = _render_masked_texture(template_by_color[frame_color], tile_by_color[seed_color], bg, interior.shape)
        right_out[r0 + 1 : r1, c0 + 1 : c1] = filled
    return out


BACKGROUND = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
OBJECT_COUNT_RANGE = (2, 4)
TEMPLATE_SIZE_RANGE = (2, 4)
FRAME_EXTRA_RANGE = (0, 2)
TEXTURE_SIZE_CHOICES = (2, 3)
PLACEMENT_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 400


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sr, sc = map(int, points[0])
    stack = [(sr, sc)]
    seen[sr, sc] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if seen[nr, nc] or not bool(mask[nr, nc]):
                continue
            seen[nr, nc] = True
            stack.append((nr, nc))
    return count == int(mask.sum())


def _has_hole(mask: np.ndarray) -> bool:
    bg = np.pad(~mask, 1, constant_values=True)
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if seen[nr, nc] or not bool(bg[nr, nc]):
                continue
            seen[nr, nc] = True
            stack.append((nr, nc))
    return bool(np.any(bg & ~seen))


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, copy=True)


def _carve_mask(h: int, w: int, target_area: int) -> np.ndarray | None:
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        changed = True
        while int(mask.sum()) > target_area and changed:
            changed = False
            random.shuffle(cells)
            for r, c in cells:
                if int(mask.sum()) <= target_area:
                    break
                if not bool(mask[r, c]):
                    continue
                test = mask.copy()
                test[r, c] = False
                if not _touches_all_edges(test):
                    continue
                if not _is_connected(test):
                    continue
                if _has_hole(test):
                    continue
                mask = test
                changed = True
        if int(mask.sum()) == target_area and _touches_all_edges(mask) and _is_connected(mask) and not _has_hole(mask):
            return _random_transform(mask)
    return None


def _sample_template_masks(count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(count):
        mask = None
        for _ in range(300):
            h = random.randint(*TEMPLATE_SIZE_RANGE)
            w = random.randint(*TEMPLATE_SIZE_RANGE)
            lo = max(3, (h * w + 1) // 2)
            hi = h * w - 1
            if lo > hi:
                continue
            target_area = random.randint(lo, hi)
            candidate = _carve_mask(h, w, target_area)
            if candidate is None:
                continue
            sig = (candidate.shape, candidate.tobytes())
            if sig in signatures:
                continue
            mask = candidate
            signatures.add(sig)
            break
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _sample_texture_mask() -> np.ndarray | None:
    size = random.choice(TEXTURE_SIZE_CHOICES)
    if size == 2:
        target_area = 3
    else:
        target_area = random.randint(5, 8)
    return _carve_mask(size, size, target_area)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BACKGROUND, dtype=int)
    patch[mask] = color
    return patch


def _make_frame_patch(template_mask: np.ndarray, frame_color: int, seed_color: int) -> tuple[np.ndarray, dict]:
    th, tw = template_mask.shape
    inner_h = th + random.randint(*FRAME_EXTRA_RANGE)
    inner_w = tw + random.randint(*FRAME_EXTRA_RANGE)
    patch = np.full((inner_h + 2, inner_w + 2), BACKGROUND, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    seed_r = random.randrange(inner_h)
    seed_c = random.randrange(inner_w)
    patch[1 + seed_r, 1 + seed_c] = seed_color
    return patch, {"frame_color": frame_color, "seed_color": seed_color}


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _place_patch(panel: np.ndarray, patch: np.ndarray, boxes: list[tuple[int, int, int, int]], margin: int) -> tuple[int, int] | None:
    h, w = panel.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    mask = patch != BACKGROUND
    for top, left in positions:
        bbox = (top, left, top + ph - 1, left + pw - 1)
        if any(_boxes_too_close(bbox, other, margin) for other in boxes):
            continue
        region = panel[top : top + ph, left : left + pw]
        if np.any(region[mask] != BACKGROUND):
            continue
        region[mask] = patch[mask]
        boxes.append(bbox)
        return top, left
    return None


def _build_expected(
    grid: np.ndarray,
    sep_col: int,
    frame_records: list[dict],
    template_masks: dict[int, np.ndarray],
    tile_patches: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    right = out[:, sep_col + 1 :]
    for record in frame_records:
        r0, c0, r1, c1 = record["bbox"]
        frame_color = int(record["frame_color"])
        seed_color = int(record["seed_color"])
        shape = (int(r1 - r0 - 1), int(c1 - c0 - 1))
        filled = _render_masked_texture(template_masks[frame_color], tile_patches[seed_color], BACKGROUND, shape)
        right[r0 + 1 : r1, c0 + 1 : c1] = filled
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    if 2 * count > len(PALETTE):
        return None

    masks = _sample_template_masks(count)
    if masks is None:
        return None

    texture_masks = []
    for _ in range(count):
        tex = _sample_texture_mask()
        if tex is None:
            return None
        texture_masks.append(tex)

    colors = random.sample(PALETTE, 2 * count)
    frame_colors = colors[:count]
    seed_colors = colors[count:]

    left_patches = [_mask_to_patch(mask, color) for mask, color in zip(masks, frame_colors)]
    tile_patches = {color: _mask_to_patch(mask, color) for mask, color in zip(texture_masks, seed_colors)}

    frame_patches: list[np.ndarray] = []
    frame_meta: list[dict] = []
    for mask, frame_color, seed_color in zip(masks, frame_colors, seed_colors):
        patch, meta = _make_frame_patch(mask, frame_color, seed_color)
        frame_patches.append(patch)
        frame_meta.append(meta)

    max_left_h = max(p.shape[0] for p in left_patches)
    max_left_w = max(p.shape[1] for p in left_patches)
    max_right_h = max(max(p.shape[0] for p in frame_patches), max(p.shape[0] for p in tile_patches.values()))
    max_right_w = max(max(p.shape[1] for p in frame_patches), max(p.shape[1] for p in tile_patches.values()))

    panel_h_min = max(max_left_h, max_right_h) + 2
    panel_h_min = max(panel_h_min, 14 + 2 * max(0, count - 2))
    left_w_min = max(max_left_w + 2, 8 + max(0, count - 2))
    right_w_min = max(max_right_w + 2, 15 + 2 * max(0, count - 2))

    panel_h = random.randint(panel_h_min, panel_h_min + 6)
    left_w = random.randint(left_w_min, left_w_min + 4)
    right_w = random.randint(right_w_min, right_w_min + 6)

    left_panel = np.full((panel_h, left_w), BACKGROUND, dtype=int)
    right_panel = np.full((panel_h, right_w), BACKGROUND, dtype=int)

    left_boxes: list[tuple[int, int, int, int]] = []
    for patch in sorted(left_patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        if _place_patch(left_panel, patch, left_boxes, PLACEMENT_MARGIN) is None:
            return None

    right_boxes: list[tuple[int, int, int, int]] = []
    frame_records: list[dict] = []
    frame_order = list(range(count))
    frame_order.sort(key=lambda idx: frame_patches[idx].shape[0] * frame_patches[idx].shape[1], reverse=True)
    for idx in frame_order:
        top_left = _place_patch(right_panel, frame_patches[idx], right_boxes, PLACEMENT_MARGIN)
        if top_left is None:
            return None
        top, left = top_left
        ph, pw = frame_patches[idx].shape
        record = dict(frame_meta[idx])
        record["bbox"] = (top, left, top + ph - 1, left + pw - 1)
        frame_records.append(record)

    tile_items = list(tile_patches.items())
    random.shuffle(tile_items)
    for _color, patch in tile_items:
        if _place_patch(right_panel, patch, right_boxes, PLACEMENT_MARGIN) is None:
            return None

    sep_col = left_w
    grid = np.full((panel_h, left_w + 1 + right_w), BACKGROUND, dtype=int)
    grid[:, :left_w] = left_panel
    grid[:, sep_col] = SEPARATOR
    grid[:, sep_col + 1 :] = right_panel

    template_masks = {color: mask for color, mask in zip(frame_colors, masks)}
    expected = _build_expected(grid, sep_col, frame_records, template_masks, tile_patches)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")

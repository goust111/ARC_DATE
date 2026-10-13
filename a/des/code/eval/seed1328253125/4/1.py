import numpy as np


GREY = 5
_GEN_RETRIES = 300
_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = exclude or set()
    return [int(color) for color in np.unique(grid) if int(color) not in banned]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _extract_templates(
    grid: np.ndarray, bg: int, grey: int
) -> tuple[tuple[int, int, int, int], int, list[np.ndarray], list[np.ndarray]]:
    grey_comps = _components(grid == grey)
    assert grey_comps
    h = grid.shape[0]
    bank_candidates = [i for i, comp in enumerate(grey_comps) if int(comp[:, 0].max()) == h - 1]
    assert len(bank_candidates) == 1
    bank_index = bank_candidates[0]
    bank_coords = grey_comps[bank_index]
    bank_bbox = _bbox(bank_coords)
    r0, c0, r1, c1 = bank_bbox

    full_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == grey)]
    templates: list[np.ndarray] = []
    for left, right in zip(full_cols, full_cols[1:]):
        if right - left <= 1:
            continue
        window = grid[r0 + 1 : r1, left + 1 : right]
        mask = (window != bg) & (window != grey)
        coords = np.argwhere(mask)
        assert coords.size > 0
        colors = np.unique(window[mask])
        assert len(colors) == 1
        wr0, wc0 = coords.min(axis=0)
        wr1, wc1 = coords.max(axis=0)
        templates.append(window[wr0 : wr1 + 1, wc0 : wc1 + 1].copy())
    assert templates
    return bank_bbox, bank_index, templates, grey_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose a stamp by the number of distinct interior colors in each framed region,
    then stamp it at the centered position without overwriting colored dots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0
    values, counts = np.unique(non_bg, return_counts=True)
    grey = int(values[np.argmax(counts)])

    _bank_bbox, bank_index, templates, grey_comps = _extract_templates(g, bg, grey)

    out = g.copy()
    frame_count = 0
    for idx, coords in enumerate(grey_comps):
        if idx == bank_index:
            continue
        if not _is_rectangular_frame(coords):
            continue
        frame_count += 1
        r0, c0, r1, c1 = _bbox(coords)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = _colors_present(inner, exclude={bg, grey})
        assert colors
        assert len(colors) <= len(templates)
        stamp = templates[len(colors) - 1]
        sh, sw = stamp.shape
        ih, iw = inner.shape
        assert ih >= sh and iw >= sw
        top = r0 + 1 + (ih - sh) // 2
        left = c0 + 1 + (iw - sw) // 2
        for dr in range(sh):
            for dc in range(sw):
                if stamp[dr, dc] != bg and out[top + dr, left + dc] == bg:
                    out[top + dr, left + dc] = int(stamp[dr, dc])
    assert frame_count > 0
    return out


def _make_stamp_mask(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    kind = int(rng.integers(6))
    mask = np.zeros((h, w), dtype=bool)

    if kind == 0:
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
    elif kind == 1:
        mask[:, 0] = True
        mask[:, -1] = True
        mask[int(rng.integers(h)), :] = True
    elif kind == 2:
        mask[:, 0] = True
        mask[:, -1] = True
        if rng.random() < 0.5:
            mask[0, :] = True
        else:
            mask[-1, :] = True
    elif kind == 3:
        corner = int(rng.integers(4))
        if corner == 0:
            mask[0, :] = True
            mask[:, 0] = True
        elif corner == 1:
            mask[0, :] = True
            mask[:, -1] = True
        elif corner == 2:
            mask[-1, :] = True
            mask[:, 0] = True
        else:
            mask[-1, :] = True
            mask[:, -1] = True
    elif kind == 4:
        if rng.random() < 0.5:
            mask[0, :] = True
        else:
            mask[-1, :] = True
        mask[:, w // 2] = True
    else:
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        if h > 2 and w > 2:
            side = int(rng.integers(4))
            if side == 0:
                mask[0, int(rng.integers(1, w - 1))] = False
            elif side == 1:
                mask[-1, int(rng.integers(1, w - 1))] = False
            elif side == 2:
                mask[int(rng.integers(1, h - 1)), 0] = False
            else:
                mask[int(rng.integers(1, h - 1)), -1] = False

    extra = int(rng.integers(0, 4))
    for _ in range(extra):
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            break
        rr, cc = frontier[int(rng.integers(len(frontier)))]
        mask[rr, cc] = True

    assert mask.any()
    return mask


def _far_enough(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = bbox_a
    br0, bc0, br1, bc1 = bbox_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_templates(
    grid: np.ndarray, rng: np.random.Generator, bg: int
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    template_colors = rng.permutation(_PALETTE)[:4]
    masks: list[np.ndarray] = []
    seen = set()
    while len(masks) < 4:
        mask = _make_stamp_mask(rng)
        key = (mask.shape, mask.tobytes())
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)

    slot_widths = [mask.shape[1] + 2 * int(rng.integers(1, 3)) for mask in masks]
    inner_h = max(mask.shape[0] for mask in masks) + 2 * int(rng.integers(1, 3))
    bank_h = inner_h + 2
    bank_w = 2 + sum(slot_widths) + len(masks) - 1

    h, w = grid.shape
    bank_top = h - bank_h
    bank_left = int(rng.integers(1, w - bank_w))
    bank_bbox = (bank_top, bank_left, h - 1, bank_left + bank_w - 1)
    r0, c0, r1, c1 = bank_bbox
    _paint_frame(grid, bank_bbox, GREY)

    colored_templates: list[np.ndarray] = []
    cursor = c0 + 1
    for idx, (mask, color, slot_w) in enumerate(zip(masks, template_colors, slot_widths)):
        slot_left = cursor
        slot_right = slot_left + slot_w - 1
        stamp = np.full(mask.shape, bg, dtype=int)
        stamp[mask] = int(color)
        colored_templates.append(stamp)

        top = r0 + 1 + (inner_h - mask.shape[0]) // 2
        left = slot_left + (slot_w - mask.shape[1]) // 2
        grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = int(color)

        if idx < len(masks) - 1:
            sep = slot_right + 1
            grid[r0 : r1 + 1, sep] = GREY
            cursor = sep + 1
        else:
            cursor = slot_right + 1
    assert cursor == c1
    return colored_templates, bank_bbox


def _frame_targets(rng: np.random.Generator) -> list[int]:
    targets = list(rng.permutation(np.arange(1, 5)))
    if rng.random() < 0.5:
        targets.append(int(rng.integers(1, 5)))
        rng.shuffle(targets)
    return [int(x) for x in targets]


def _scatter_dots(
    grid: np.ndarray,
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    stamp: np.ndarray,
    dot_colors: np.ndarray,
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    sh, sw = stamp.shape
    top = r0 + 1 + (inner_h - sh) // 2
    left = c0 + 1 + (inner_w - sw) // 2

    all_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    stamp_cells = [
        (top + dr, left + dc)
        for dr, dc in np.argwhere(stamp != bg)
        if r0 + 1 <= top + dr < r1 and c0 + 1 <= left + dc < c1
    ]
    used: set[tuple[int, int]] = set()
    chosen_colors = rng.choice(dot_colors, size=len(dot_colors), replace=False)

    if stamp_cells and rng.random() < 0.6:
        overlap_pos = stamp_cells[int(rng.integers(len(stamp_cells)))]
        overlap_color = int(chosen_colors[int(rng.integers(len(chosen_colors)))])
        grid[overlap_pos] = overlap_color
        used.add(overlap_pos)

    for color in chosen_colors:
        available = [cell for cell in all_cells if cell not in used]
        assert available
        pos = available[int(rng.integers(len(available)))]
        grid[pos] = int(color)
        used.add(pos)

    extra = int(rng.integers(0, len(chosen_colors) + 1))
    for _ in range(extra):
        available = [cell for cell in all_cells if cell not in used]
        if not available:
            break
        pos = available[int(rng.integers(len(available)))]
        color = int(chosen_colors[int(rng.integers(len(chosen_colors)))])
        grid[pos] = color
        used.add(pos)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = 0
    width = int(rng.integers(43, 53))
    height = int(rng.integers(28, 37))
    grid = np.full((height, width), bg, dtype=int)

    templates, bank_bbox = _place_templates(grid, rng, bg)
    bank_top = bank_bbox[0]
    reserved = [bank_bbox]

    color_pool = rng.permutation(_PALETTE)
    template_colors = {int(np.unique(stamp[stamp != bg])[0]) for stamp in templates}
    dot_palette = np.array([c for c in color_pool if int(c) not in template_colors], dtype=int)
    assert len(dot_palette) >= 4

    frames = _frame_targets(rng)
    for target in frames:
        stamp = templates[target - 1]
        sh, sw = stamp.shape
        inner_h = sh + 2 * int(rng.integers(1, 4))
        inner_w = sw + 2 * int(rng.integers(1, 4))
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        max_r0 = bank_top - outer_h - 1
        max_c0 = width - outer_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        placed = False
        for _ in range(200):
            r0 = int(rng.integers(1, max_r0 + 1))
            c0 = int(rng.integers(1, max_c0 + 1))
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if all(_far_enough(bbox, other, gap=1) for other in reserved):
                _paint_frame(grid, bbox, GREY)
                dot_colors = rng.choice(dot_palette, size=target, replace=False)
                _scatter_dots(grid, rng, bbox, stamp, dot_colors, bg)
                reserved.append(bbox)
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

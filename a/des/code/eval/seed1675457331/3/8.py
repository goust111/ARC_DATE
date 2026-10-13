from __future__ import annotations

import random

import numpy as np


SLOT_SIZE = 5
FRAME_HEIGHT = 2 * SLOT_SIZE + 3
RING_STEP = SLOT_SIZE + 1
COLOR_POOL = tuple(range(10))
GEN_RETRIES = 400
PLACE_RETRIES = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    parts: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _submask(coords: list[tuple[int, int]], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _template_mask(slot_count: int) -> np.ndarray:
    width = slot_count * (SLOT_SIZE + 1) + 1
    out = np.zeros((FRAME_HEIGHT, width), dtype=bool)
    out[0, :] = True
    out[SLOT_SIZE + 1, :] = True
    out[-1, :] = True
    out[:, :: (SLOT_SIZE + 1)] = True
    return out


def _find_template(grid: np.ndarray, bg: int) -> dict[str, object]:
    height, width = grid.shape
    best: dict[str, object] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color, connectivity=4):
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            sub = _submask(coords, box)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            if box_h != FRAME_HEIGHT or box_w < 7 or (box_w - 1) % (SLOT_SIZE + 1) != 0:
                continue
            slot_count = (box_w - 1) // (SLOT_SIZE + 1)
            if not np.array_equal(sub, _template_mask(slot_count)):
                continue
            mask = np.zeros((height, width), dtype=bool)
            for r, c in coords:
                mask[r, c] = True
            cand = {
                "color": color,
                "bbox": box,
                "mask": mask,
                "slots": slot_count,
            }
            if best is None or slot_count > int(best["slots"]):
                best = cand
    assert best is not None, "template frame not found"
    return best


def _find_legend(grid: np.ndarray, bg: int, template: dict[str, object]) -> dict[str, object]:
    frame_mask = np.asarray(template["mask"], dtype=bool)
    frame_box = template["bbox"]
    frame_left = int(frame_box[1])
    search = (grid != bg) & (~frame_mask)
    best: dict[str, object] | None = None
    for coords in _components(search, connectivity=4):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if c1 >= frame_left:
            continue
        if c0 != c1:
            continue
        if len(coords) != r1 - r0 + 1:
            continue
        rows = sorted(coords)
        colors = [int(grid[r, c]) for r, c in rows]
        if len(colors) < 2 or len(set(colors)) != len(colors):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        cand = {"colors": colors, "mask": mask, "bbox": box}
        if best is None or len(colors) > len(best["colors"]):
            best = cand
    assert best is not None, "legend not found"
    return best


def _parse_flower(subgrid: np.ndarray, bg: int) -> dict[str, object] | None:
    if np.any(subgrid == bg):
        return None

    if subgrid.shape == (3, 3):
        heart = int(subgrid[1, 1])
        petals = [int(v) for v in np.delete(subgrid.reshape(-1), 4)]
        if len(set(petals)) == 1 and petals[0] != heart:
            return {"layers": 1, "heart": heart, "pattern": subgrid.copy()}
        return None

    if subgrid.shape == (5, 5):
        heart = int(subgrid[2, 2])
        inner = []
        outer = []
        for r in range(5):
            for c in range(5):
                dist = max(abs(r - 2), abs(c - 2))
                value = int(subgrid[r, c])
                if dist == 0:
                    continue
                if dist == 1:
                    inner.append(value)
                else:
                    outer.append(value)
        if len(set(inner)) == 1 and len(set(outer)) == 1:
            inner_color = inner[0]
            outer_color = outer[0]
            if len({heart, inner_color, outer_color}) == 3:
                return {"layers": 2, "heart": heart, "pattern": subgrid.copy()}
    return None


def _find_flowers(
    grid: np.ndarray,
    bg: int,
    template: dict[str, object],
    legend: dict[str, object],
) -> list[dict[str, object]]:
    blocked = np.asarray(template["mask"], dtype=bool) | np.asarray(legend["mask"], dtype=bool)
    flowers: list[dict[str, object]] = []
    for coords in _components((grid != bg) & (~blocked), connectivity=8):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if len(coords) != sub.size:
            continue
        parsed = _parse_flower(sub, bg)
        if parsed is None:
            continue
        parsed["coords"] = coords
        parsed["bbox"] = box
        flowers.append(parsed)
    assert flowers, "no flowers found"
    return flowers


def _slot_origin(template: dict[str, object], row_idx: int, col_idx: int) -> tuple[int, int]:
    r0, c0, _r1, _c1 = template["bbox"]
    return r0 + 1 + row_idx * (SLOT_SIZE + 1), c0 + 1 + col_idx * (SLOT_SIZE + 1)


def _paste_flower(canvas: np.ndarray, flower: np.ndarray, top: int, left: int) -> None:
    fh, fw = flower.shape
    offset_r = (SLOT_SIZE - fh) // 2
    offset_c = (SLOT_SIZE - fw) // 2
    canvas[top + offset_r : top + offset_r + fh, left + offset_c : left + offset_c + fw] = flower


def _arrange_flowers(
    base: np.ndarray,
    bg: int,
    template: dict[str, object],
    flowers: list[dict[str, object]],
    order: dict[int, int],
) -> np.ndarray:
    out = np.array(base, copy=True)
    slot_count = int(template["slots"])

    for flower in flowers:
        for r, c in flower["coords"]:
            out[r, c] = bg

    for row_idx in range(2):
        for col_idx in range(slot_count):
            top, left = _slot_origin(template, row_idx, col_idx)
            out[top : top + SLOT_SIZE, left : left + SLOT_SIZE] = bg

    one_layer = sorted((f for f in flowers if int(f["layers"]) == 1), key=lambda f: order[int(f["heart"])])
    two_layer = sorted((f for f in flowers if int(f["layers"]) == 2), key=lambda f: order[int(f["heart"])])
    assert len(one_layer) <= slot_count and len(two_layer) <= slot_count

    for col_idx, flower in enumerate(one_layer):
        top, left = _slot_origin(template, 0, col_idx)
        _paste_flower(out, np.asarray(flower["pattern"], dtype=int), top, left)
    for col_idx, flower in enumerate(two_layer):
        top, left = _slot_origin(template, 1, col_idx)
        _paste_flower(out, np.asarray(flower["pattern"], dtype=int), top, left)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move scattered concentric "flowers" into a two-row template.

    Flowers are grouped by petal-layer count (3x3 flowers above, 5x5 flowers below),
    sorted within each group by the vertical color-wheel legend, then centered into
    the template's 5x5 slots from left to right. The original flowers are cleared to
    the background while the legend and template frame remain in place.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    template = _find_template(g, bg)
    legend = _find_legend(g, bg, template)
    order = {color: idx for idx, color in enumerate(legend["colors"])}
    flowers = _find_flowers(g, bg, template, legend)
    assert all(int(f["heart"]) in order for f in flowers), "legend must cover all flower centers"
    return _arrange_flowers(g, bg, template, flowers, order)


def _make_flower(heart: int, petal1: int, petal2: int | None = None) -> np.ndarray:
    if petal2 is None:
        out = np.full((3, 3), petal1, dtype=int)
        out[1, 1] = heart
        return out
    out = np.full((5, 5), petal2, dtype=int)
    out[1:4, 1:4] = petal1
    out[2, 2] = heart
    return out


def _draw_template(canvas: np.ndarray, top: int, left: int, slots: int, color: int) -> None:
    width = slots * (SLOT_SIZE + 1) + 1
    bottom = top + FRAME_HEIGHT - 1
    right = left + width - 1
    canvas[top, left : right + 1] = color
    canvas[top + SLOT_SIZE + 1, left : right + 1] = color
    canvas[bottom, left : right + 1] = color
    for c in range(left, right + 1, SLOT_SIZE + 1):
        canvas[top : bottom + 1, c] = color


def _place_scattered(
    canvas: np.ndarray,
    bg: int,
    flowers: list[dict[str, object]],
    max_bottom: int,
) -> list[tuple[int, int]] | None:
    height, width = canvas.shape
    occupied = np.zeros_like(canvas, dtype=bool)
    order = sorted(flowers, key=lambda f: -np.asarray(f["pattern"]).shape[0])
    placed: dict[int, tuple[int, int]] = {}

    for flower in order:
        pattern = np.asarray(flower["pattern"], dtype=int)
        fh, fw = pattern.shape
        low_r = 1
        high_r = max_bottom - fh
        low_c = 1
        high_c = width - fw - 1
        if high_r < low_r or high_c < low_c:
            return None
        for _ in range(PLACE_RETRIES):
            r = random.randint(low_r, high_r)
            c = random.randint(low_c, high_c)
            rr0 = max(0, r - 1)
            rr1 = min(height, r + fh + 1)
            cc0 = max(0, c - 1)
            cc1 = min(width, c + fw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            if np.any(canvas[r : r + fh, c : c + fw] != bg):
                continue
            canvas[r : r + fh, c : c + fw] = pattern
            occupied[r : r + fh, c : c + fw] = True
            placed[id(flower)] = (r, c)
            break
        else:
            return None

    return [placed[id(flower)] for flower in flowers]


def _build_expected(
    grid: np.ndarray,
    bg: int,
    template_top: int,
    template_left: int,
    slot_count: int,
    legend_colors: list[int],
    flowers: list[dict[str, object]],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for flower, (r, c) in zip(flowers, positions, strict=False):
        pattern = np.asarray(flower["pattern"], dtype=int)
        fh, fw = pattern.shape
        out[r : r + fh, c : c + fw] = bg

    for row_idx in range(2):
        for col_idx in range(slot_count):
            top = template_top + 1 + row_idx * (SLOT_SIZE + 1)
            left = template_left + 1 + col_idx * (SLOT_SIZE + 1)
            out[top : top + SLOT_SIZE, left : left + SLOT_SIZE] = bg

    order = {color: idx for idx, color in enumerate(legend_colors)}
    one_layer = sorted((f for f in flowers if int(f["layers"]) == 1), key=lambda f: order[int(f["heart"])])
    two_layer = sorted((f for f in flowers if int(f["layers"]) == 2), key=lambda f: order[int(f["heart"])])

    for col_idx, flower in enumerate(one_layer):
        top = template_top + 1
        left = template_left + 1 + col_idx * (SLOT_SIZE + 1)
        _paste_flower(out, np.asarray(flower["pattern"], dtype=int), top, left)
    for col_idx, flower in enumerate(two_layer):
        top = template_top + 1 + (SLOT_SIZE + 1)
        left = template_left + 1 + col_idx * (SLOT_SIZE + 1)
        _paste_flower(out, np.asarray(flower["pattern"], dtype=int), top, left)

    return out


def _gen_flower_specs(bg: int, frame_color: int) -> tuple[list[int], list[dict[str, object]], int]:
    n_small = random.randint(1, 3)
    n_large = random.randint(1, 3)
    total = n_small + n_large
    hearts = random.sample([c for c in COLOR_POOL if c not in (bg, frame_color)], total)
    legend_colors = hearts[:]
    random.shuffle(legend_colors)

    flower_info: list[dict[str, object]] = []
    for idx, heart in enumerate(hearts):
        usable = [c for c in COLOR_POOL if c not in (bg, frame_color, heart)]
        if idx < n_small:
            petal = random.choice(usable)
            pattern = _make_flower(heart, petal)
            layers = 1
        else:
            petal1, petal2 = random.sample(usable, 2)
            pattern = _make_flower(heart, petal1, petal2)
            layers = 2
        flower_info.append({"heart": heart, "layers": layers, "pattern": pattern})
    random.shuffle(flower_info)
    return legend_colors, flower_info, max(n_small, n_large)


def _background_is_unique(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    ranked = sorted(zip(counts.tolist(), values.tolist()), reverse=True)
    return bool(ranked and int(ranked[0][1]) == bg and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]))


def _gen_attempt() -> np.ndarray | None:
    bg, frame_color = random.sample(COLOR_POOL, 2)
    legend_colors, flowers, min_slots = _gen_flower_specs(bg, frame_color)
    slot_count = random.randint(min_slots, min(5, min_slots + 2))
    frame_width = slot_count * (SLOT_SIZE + 1) + 1

    extra_rows = max(0, len(flowers) - 4) * 2
    upper_height = random.randint(12 + extra_rows, 18 + extra_rows)
    bottom_margin = random.randint(1, 4)
    height = upper_height + FRAME_HEIGHT + bottom_margin

    legend_gap = random.randint(2, 4)
    width = random.randint(frame_width + legend_gap + 4, frame_width + legend_gap + 14)
    frame_left = random.randint(legend_gap + 1, width - frame_width - 2)
    frame_top = upper_height
    legend_col = frame_left - legend_gap
    legend_top = random.randint(frame_top, frame_top + FRAME_HEIGHT - len(legend_colors))

    grid = np.full((height, width), bg, dtype=int)
    _draw_template(grid, frame_top, frame_left, slot_count, frame_color)
    for idx, color in enumerate(legend_colors):
        grid[legend_top + idx, legend_col] = color

    positions = _place_scattered(grid, bg, flowers, max_bottom=frame_top - 1)
    if positions is None:
        return None

    expected = _build_expected(
        grid,
        bg,
        frame_top,
        frame_left,
        slot_count,
        legend_colors,
        flowers,
        positions,
    )
    if not _background_is_unique(grid, bg):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flower-template input")

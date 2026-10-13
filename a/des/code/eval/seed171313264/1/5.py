from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
OBJECT_COLORS = (1, 2, 3, 4, 6, 8, 9)

TEMPLATE_COUNT_OPTIONS = (3, 4)
TEMPLATE_HEIGHT_RANGE = (2, 4)
TEMPLATE_WIDTH_RANGE = (2, 4)
TEMPLATE_DENSITY_RANGE = (0.35, 0.75)
TEMPLATE_RETRY_LIMIT = 200

OBJECT_RETRY_LIMIT = 200
GENERATION_RETRY_LIMIT = 400
MAX_GRID_SIDE = 30


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = [tuple(map(int, coords[0]))]
    seen[stack[0]] = True
    count = 0

    while stack:
        r, c = stack.pop()
        count += 1
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((int(rr), int(cc)))

    return count == int(mask.sum())


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    bg_mask = ~mask

    for r0 in range(h):
        for c0 in range(w):
            if not bg_mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c, h, w):
                    if bg_mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))
            if not touches_border:
                holes += 1

    return holes


def _find_divider(g: np.ndarray) -> int:
    divider_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))]
    assert len(divider_cols) == 1, "expected exactly one full-height grey divider"
    divider = int(divider_cols[0])
    assert 0 < divider < g.shape[1] - 1, "divider must split left/right regions"
    return divider


def _extract_templates(g: np.ndarray, divider: int) -> list[np.ndarray]:
    right = g[:, divider + 1 :]
    h, _w = right.shape
    orange_rows = [r for r in range(h) if bool(np.any(right[r] == ORANGE))]
    assert orange_rows, "expected at least one template on the right side"

    runs: list[tuple[int, int]] = []
    start = prev = int(orange_rows[0])
    for r in orange_rows[1:]:
        r = int(r)
        if r != prev + 1:
            runs.append((start, prev))
            start = r
        prev = r
    runs.append((start, prev))

    templates: list[np.ndarray] = []
    for r0, r1 in runs:
        band = right[r0 : r1 + 1]
        coords = np.argwhere(band == ORANGE)
        assert coords.size, "template band must contain orange cells"
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        template = band[:, c0 : c1 + 1]
        values = set(map(int, np.unique(template)))
        assert values <= {BG, ORANGE}, "template area may only contain bg/orange"
        assert bool(np.any(template == BG)), "template must contain bg pixels"
        assert bool(np.all(np.any(template == ORANGE, axis=1))), "template rows must be non-empty"
        assert bool(np.all(np.any(template == ORANGE, axis=0))), "template cols must be non-empty"
        templates.append(template == ORANGE)

    return templates


def _left_components(g: np.ndarray, divider: int) -> list[tuple[int, np.ndarray]]:
    left = g[:, :divider]
    h, w = left.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(left[r0, c0])
            if color in (BG, GREY) or seen[r0, c0]:
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []

            while stack:
                r, c = stack.pop()
                pts.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c, h, w):
                    if not seen[rr, cc] and int(left[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))

            comps.append((color, np.asarray(pts, dtype=int)))

    return comps


def _tile_mask(template: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % int(template.shape[0])
    cc = np.arange(width)[None, :] % int(template.shape[1])
    return template[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count holes in each left-side object, index the right-side template list with that
    hole count, then tile the chosen template as a keep/erase mask over the object's bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider(g)
    templates = _extract_templates(g, divider)
    out = g.copy()

    for color, coords in _left_components(g, divider):
        r0, c0, r1, c1 = _bbox(coords)
        crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        obj_mask = crop == int(color)
        holes = _hole_count(obj_mask)
        assert holes < len(templates), "object hole count must index an existing template"

        keep = _tile_mask(templates[holes], crop.shape[0], crop.shape[1])
        crop[~keep] = BG
        out[r0 : r1 + 1, c0 : c1 + 1] = crop

    return out


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    view = dst[top : top + h, left : left + w]
    mask = src != BG
    view[mask] = src[mask]


def _sample_template(existing: set[tuple[int, int, tuple[int, ...]]]) -> np.ndarray:
    for _ in range(TEMPLATE_RETRY_LIMIT):
        h = random.randint(*TEMPLATE_HEIGHT_RANGE)
        w = random.randint(*TEMPLATE_WIDTH_RANGE)
        mask = np.zeros((h, w), dtype=bool)

        for r in range(h):
            mask[r, random.randrange(w)] = True
        for c in range(w):
            mask[random.randrange(h), c] = True

        density = random.uniform(*TEMPLATE_DENSITY_RANGE)
        for r in range(h):
            for c in range(w):
                if not mask[r, c] and random.random() < density:
                    mask[r, c] = True

        if bool(mask.all()):
            rr = random.randrange(h)
            cc = random.randrange(w)
            mask[rr, cc] = False

        if int(mask.sum()) <= 1 or int((~mask).sum()) == 0:
            continue

        key = (h, w, tuple(int(v) for v in mask.ravel()))
        if key in existing:
            continue
        existing.add(key)
        return mask

    raise AssertionError("failed to sample a valid template")


def _apply_random_notches(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape

    for _ in range(random.randint(0, 2)):
        candidate = out.copy()
        side = random.choice(("top", "bottom", "left", "right"))

        if side in ("top", "bottom") and h >= 4:
            depth = random.randint(1, min(2, h - 2))
            start = random.randint(1, w - 2)
            stop = random.randint(start + 1, w)
            if side == "top":
                candidate[:depth, start:stop] = False
            else:
                candidate[h - depth :, start:stop] = False
        elif side in ("left", "right") and w >= 4:
            depth = random.randint(1, min(2, w - 2))
            start = random.randint(1, h - 2)
            stop = random.randint(start + 1, h)
            if side == "left":
                candidate[start:stop, :depth] = False
            else:
                candidate[start:stop, w - depth :] = False
        else:
            continue

        if _is_connected(candidate):
            out = candidate

    return out


def _sample_object_mask(hole_count: int) -> np.ndarray:
    for _ in range(OBJECT_RETRY_LIMIT):
        h_min = 4 if hole_count == 0 else 5 + hole_count // 2
        w_min = 4 if hole_count == 0 else 5 + hole_count
        h = random.randint(h_min, h_min + 3)
        w = random.randint(w_min, w_min + 3)

        mask = np.ones((h, w), dtype=bool)
        hole_sizes = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 3),
            (3, 1),
        ]
        hole_sizes = [(hh, ww) for hh, ww in hole_sizes if hh <= h - 2 and ww <= w - 2]
        if not hole_sizes:
            continue

        for _idx in range(hole_count):
            hh, ww = random.choice(hole_sizes)
            top = random.randint(1, h - hh - 1)
            left = random.randint(1, w - ww - 1)
            mask[top : top + hh, left : left + ww] = False

        mask = _apply_random_notches(mask)

        if _hole_count(mask) != hole_count:
            continue
        if not _is_connected(mask):
            continue
        if int(mask.sum()) < 5:
            continue

        turns = random.randrange(4)
        mask = np.rot90(mask, turns)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask.copy()

    raise AssertionError(f"failed to sample object with {hole_count} holes")


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _build_left_panel(objects: list[np.ndarray]) -> np.ndarray | None:
    n = len(objects)
    min_rows = 1 if n <= 3 else 2
    n_rows = random.randint(min_rows, min(3, n))

    items = list(objects)
    random.shuffle(items)
    items.sort(key=lambda obj: int(obj.shape[1]), reverse=True)

    rows: list[list[np.ndarray]] = [[] for _ in range(n_rows)]
    row_loads = [0] * n_rows
    for obj in items:
        idx = min(range(n_rows), key=lambda i: row_loads[i])
        rows[idx].append(obj)
        row_loads[idx] += int(obj.shape[1]) + 2

    for row in rows:
        random.shuffle(row)

    row_heights = [max(int(obj.shape[0]) for obj in row) for row in rows]
    row_gaps = [random.randint(1, 3) for _ in range(max(0, n_rows - 1))]
    row_widths: list[int] = []
    col_gaps_per_row: list[list[int]] = []
    for row in rows:
        gaps = [random.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        width = sum(int(obj.shape[1]) for obj in row) + sum(gaps)
        row_widths.append(width)
        col_gaps_per_row.append(gaps)

    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    panel_h = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
    panel_w = left_margin + max(row_widths) + right_margin + random.randint(0, 3)
    if panel_h > MAX_GRID_SIDE or panel_w > MAX_GRID_SIDE - 5:
        return None

    panel = np.zeros((panel_h, panel_w), dtype=int)
    row_top = top_margin
    for row_idx, row in enumerate(rows):
        row_width = row_widths[row_idx]
        row_height = row_heights[row_idx]
        row_left = random.randint(left_margin, panel_w - right_margin - row_width)
        cur_left = row_left

        for obj_idx, obj in enumerate(row):
            obj_h, obj_w = map(int, obj.shape)
            spare = row_height - obj_h
            if spare <= 0:
                offset = 0
            else:
                mode = random.choice(("top", "center", "bottom"))
                if mode == "top":
                    offset = 0
                elif mode == "bottom":
                    offset = spare
                else:
                    offset = spare // 2
            _paste_nonzero(panel, obj, row_top + offset, cur_left)
            cur_left += obj_w
            if obj_idx < len(row) - 1:
                cur_left += col_gaps_per_row[row_idx][obj_idx]

        row_top += row_height
        if row_idx < len(rows) - 1:
            row_top += row_gaps[row_idx]

    return panel


def _build_candidate() -> np.ndarray | None:
    template_count = random.choice(TEMPLATE_COUNT_OPTIONS)

    seen_templates: set[tuple[int, int, tuple[int, ...]]] = set()
    templates = [_sample_template(seen_templates) for _ in range(template_count)]

    hole_counts = list(range(template_count))
    if random.random() < 0.5:
        hole_counts.append(random.randrange(template_count))
    random.shuffle(hole_counts)

    colors = random.sample(OBJECT_COLORS, len(hole_counts))
    objects = [_colorize(_sample_object_mask(hc), color) for hc, color in zip(hole_counts, colors)]
    left_panel = _build_left_panel(objects)
    if left_panel is None:
        return None

    right_core_height = sum(int(t.shape[0]) for t in templates) + (len(templates) - 1)
    right_region_width = max(int(t.shape[1]) for t in templates) + random.randint(2, 5)
    left_region_width = int(left_panel.shape[1]) + random.randint(0, 3)

    final_h = max(int(left_panel.shape[0]), right_core_height) + random.randint(0, 3)
    final_w = left_region_width + 1 + right_region_width
    if final_h > MAX_GRID_SIDE or final_w > MAX_GRID_SIDE:
        return None

    grid = np.zeros((final_h, final_w), dtype=int)
    divider = left_region_width
    grid[:, divider] = GREY

    left_top = random.randint(0, final_h - int(left_panel.shape[0]))
    left_left = random.randint(0, left_region_width - int(left_panel.shape[1]))
    _paste_nonzero(grid, left_panel, left_top, left_left)

    right_top = random.randint(0, final_h - right_core_height)
    cursor = right_top
    for idx, template in enumerate(templates):
        tile = np.where(template, ORANGE, BG).astype(int)
        slack = right_region_width - int(tile.shape[1])
        if slack <= 1:
            x_off = random.randint(0, max(0, slack))
        else:
            x_off = random.randint(1, slack - 1)
        col = divider + 1 + x_off
        grid[cursor : cursor + tile.shape[0], col : col + tile.shape[1]] = tile
        cursor += int(tile.shape[0])
        if idx < len(templates) - 1:
            cursor += 1

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if not bool(np.any(out != grid)):
            continue
        return grid

    raise AssertionError("failed to generate a valid input grid")

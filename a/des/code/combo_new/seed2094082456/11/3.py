from __future__ import annotations

import random

import numpy as np


DIVIDER_COLOR = 5
TEMPLATE_COLOR = 9
CELL_SIZE = 4
MAX_GRID_SIZE = 30
GEN_ATTEMPTS = 400

TEMPLATE_CATALOG = (
    ("##", "##"),
    ("###", ".#."),
    ("#..", "###"),
    ("##.", ".##"),
    (".#.", "###", ".#."),
    ("###", "..#"),
    ("#.#", "###"),
    (".##", "##."),
    ("####", ".#.."),
    ("##.", "###"),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_border:
                holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start_r, start_c = map(int, coords[0])
    seen = np.zeros((h, w), dtype=bool)
    seen[start_r, start_c] = True
    stack = [(start_r, start_c)]
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return count == int(mask.sum())


def _parse_template(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _extract_templates(block: np.ndarray, divider: int, template_color: int) -> list[np.ndarray]:
    divider_cols = np.all(block == int(divider), axis=0)
    segments: list[tuple[int, int]] = []
    c = 0
    while c < block.shape[1]:
        if divider_cols[c]:
            c += 1
            continue
        start = c
        while c < block.shape[1] and not divider_cols[c]:
            c += 1
        segments.append((start, c))
    if len(segments) != 4:
        raise AssertionError("expected exactly four template segments")

    templates = []
    for c0, c1 in segments:
        mask = block[:, c0:c1] == int(template_color)
        if not mask.any():
            raise AssertionError("template segment must contain template pixels")
        templates.append(_crop_mask(mask))
    return templates


def _find_template_band(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray]]:
    h, _w = grid.shape
    for band_h in range(min(4, h), 1, -1):
        strip = grid[h - band_h : h]
        non_bg_cols = np.where(np.any(strip != int(bg), axis=0))[0]
        if non_bg_cols.size == 0:
            continue
        c0 = int(non_bg_cols[0])
        c1 = int(non_bg_cols[-1] + 1)
        block = strip[:, c0:c1]
        colors = [int(color) for color in np.unique(block) if int(color) != int(bg)]
        if len(colors) != 2:
            continue
        for divider in colors:
            divider_cols = np.all(block == int(divider), axis=0)
            if int(divider_cols.sum()) < 5:
                continue
            other = next(color for color in colors if color != divider)
            try:
                templates = _extract_templates(block, divider, other)
            except AssertionError:
                continue
            return h - band_h, templates
    raise AssertionError("failed to locate bottom template band")


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == int(bg):
                continue
            color = int(grid[r0, c0])
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(grid[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            bbox = _bbox_from_coords(coords)
            components.append({"color": color, "coords": coords, "bbox": bbox})
    return components


def _component_hole_count(component: dict[str, object]) -> int:
    coords = component["coords"]
    assert isinstance(coords, list)
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return _count_holes(mask)


def _scaled_mask(template: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    th, tw = template.shape
    scale = min(target_h // th, target_w // tw)
    if scale < 1:
        scale = 1
    return np.kron(template.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    for dr, dc in zip(rows.tolist(), cols.tolist()):
        rr = int(top + dr)
        cc = int(left + dc)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each colored object by the template indexed by its hole count, centered in the same bbox.

    Input structure:
    - The background is the dominant color.
    - Above the bottom band are several 4-connected monochrome objects in distinct colors; each object may contain 0, 1, 2, or 3 enclosed background holes.
    - The bottom band is a horizontal template library: four small maroon binary patterns, separated by full-height grey columns, ordered left to right as indices 0, 1, 2, and 3.

    Rule:
    - Remove the bottom template band from the output.
    - For each object, count the number of enclosed background regions inside its bounding box.
    - Select the template with the same index, enlarge it by the largest integer scale that fits inside the object's bbox, and center the scaled template inside that bbox.
    - Erase the object's original shape and draw the scaled template in the object's original color.

    Key insight:
    - The object's topology chooses which template to use: hole count is the lookup key, while color and bbox center are the only properties preserved from the source object.
    """
    g = np.asarray(grid, dtype=int)
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()])

    band_top, templates = _find_template_band(g, bg)
    body = g[:band_top].copy()
    components = _find_components(body, bg)

    out = body.copy()
    for component in components:
        color = int(component["color"])
        coords = component["coords"]
        assert isinstance(coords, list)
        for r, c in coords:
            out[r, c] = int(bg)

        holes = _component_hole_count(component)
        assert 0 <= holes < 4
        template = templates[holes]

        r0, c0, r1, c1 = component["bbox"]
        target_h = r1 - r0 + 1
        target_w = c1 - c0 + 1
        scaled = _scaled_mask(template, target_h, target_w)
        sh, sw = scaled.shape
        top = r0 + (target_h - sh) // 2
        left = c0 + (target_w - sw) // 2
        _paint_mask(out, scaled, top, left, color)

    return out


def _candidate_dims(holes: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(4, 12):
        for w in range(4, 12):
            if holes == 0:
                dims.append((h, w))
                continue
            rows = list(range(2, h - 2, 2))
            cols = list(range(2, w - 2, 2))
            if len(rows) * len(cols) >= holes:
                dims.append((h, w))
    return dims


def _make_object_mask(holes: int) -> np.ndarray:
    dims = _candidate_dims(holes)
    for _ in range(120):
        h, w = random.choice(dims)
        mask = np.ones((h, w), dtype=bool)

        if holes:
            seeds = [(r, c) for r in range(2, h - 2, 2) for c in range(2, w - 2, 2)]
            for r, c in random.sample(seeds, holes):
                mask[r, c] = False

        notch_tries = random.randint(0, min(4, h + w - 8))
        for _ in range(notch_tries):
            trial = mask.copy()
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                if w <= 2:
                    continue
                span = random.randint(1, min(2, w - 2))
                start = random.randint(1, w - 1 - span)
                row = 0 if side == "top" else h - 1
                trial[row, start : start + span] = False
            else:
                if h <= 2:
                    continue
                span = random.randint(1, min(2, h - 2))
                start = random.randint(1, h - 1 - span)
                col = 0 if side == "left" else w - 1
                trial[start : start + span, col] = False

            if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any() or not trial[:, -1].any():
                continue
            if not _is_connected(trial):
                continue
            if _count_holes(trial) != holes:
                continue
            mask = trial

        if _is_connected(mask) and _count_holes(mask) == holes:
            return mask
    raise RuntimeError(f"failed to build object with {holes} holes")


def _render_library(templates: list[np.ndarray], bg: int) -> np.ndarray:
    band_h = max(template.shape[0] for template in templates)
    band_w = 5 + 4 * CELL_SIZE
    band = np.full((band_h, band_w), int(bg), dtype=int)
    for sep_col in range(0, band_w, CELL_SIZE + 1):
        band[:, sep_col] = DIVIDER_COLOR
    for idx, template in enumerate(templates):
        th, tw = template.shape
        cell_left = 1 + idx * (CELL_SIZE + 1)
        x_pad = random.randint(0, CELL_SIZE - tw)
        y0 = band_h - th
        patch = band[y0 : y0 + th, cell_left + x_pad : cell_left + x_pad + tw]
        patch[template] = TEMPLATE_COLOR
    return band


def _place_objects(
    body_h: int,
    body_w: int,
    masks: list[np.ndarray],
    colors: list[int],
    bg: int,
) -> np.ndarray | None:
    grid = np.full((body_h, body_w), int(bg), dtype=int)
    blocked = np.zeros((body_h, body_w), dtype=bool)
    order = sorted(range(len(masks)), key=lambda idx: (masks[idx].shape[0] * masks[idx].shape[1]), reverse=True)

    for idx in order:
        mask = masks[idx]
        color = colors[idx]
        h, w = mask.shape
        placed = False
        for _ in range(250):
            r0 = random.randint(0, body_h - h)
            c0 = random.randint(0, body_w - w)
            rr0 = max(0, r0 - 1)
            rr1 = min(body_h, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(body_w, c0 + w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            patch = grid[r0 : r0 + h, c0 : c0 + w]
            patch[mask] = int(color)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _choose_templates() -> list[np.ndarray]:
    raw = random.sample(TEMPLATE_CATALOG, 4)
    return [_parse_template(rows) for rows in raw]


def _attempt_generate() -> np.ndarray | None:
    bg_choices = [color for color in range(10) if color not in (DIVIDER_COLOR, TEMPLATE_COLOR)]
    bg = random.choice(bg_choices)
    templates = _choose_templates()
    library = _render_library(templates, bg)
    band_h, library_w = library.shape

    extra = random.randint(0, 2)
    hole_counts = [0, 1, 2, 3] + [random.randint(0, 3) for _ in range(extra)]
    random.shuffle(hole_counts)
    masks = [_make_object_mask(holes) for holes in hole_counts]

    min_body_h = max(mask.shape[0] for mask in masks) + 2
    min_body_w = max(max(mask.shape[1] for mask in masks) + 2, library_w)
    body_h = random.randint(max(12, min_body_h), MAX_GRID_SIZE - band_h)
    body_w = random.randint(max(23, min_body_w), MAX_GRID_SIZE)

    color_pool = [color for color in range(10) if color not in (bg, DIVIDER_COLOR, TEMPLATE_COLOR)]
    if len(color_pool) < len(masks):
        return None
    colors = random.sample(color_pool, len(masks))
    body = _place_objects(body_h, body_w, masks, colors, bg)
    if body is None:
        return None

    grid = np.full((body_h + band_h, body_w), int(bg), dtype=int)
    grid[:body_h] = body
    lib_left = random.randint(0, body_w - library_w)
    grid[body_h:, lib_left : lib_left + library_w] = library

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[counts.argmax()]) != int(bg):
        return None

    out = main(grid)
    if out.shape != (body_h, body_w):
        return None
    if max(grid.shape) > MAX_GRID_SIZE or max(out.shape) > MAX_GRID_SIZE:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")

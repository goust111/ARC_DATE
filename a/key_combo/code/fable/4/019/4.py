from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# Make `arc_utils` importable when this file is loaded directly via SourceFileLoader.
_THIS = Path(__file__).resolve()
for _parent in [_THIS, *_THIS.parents]:
    _candidate = _parent / "new" / "solvers"
    if (_candidate / "arc_utils.py").is_file():
        _candidate = str(_candidate)
        if _candidate not in sys.path:
            sys.path.append(_candidate)
        break

from arc_utils import Color, colors_present, find_components, get_bg, paste


def _component_mask(component: dict[str, np.ndarray | int | tuple[int, int, int, int]]) -> np.ndarray:
    """Build a local boolean mask for one connected component and its bbox."""
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in map(tuple, component["coords"]):
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _punch_template(
    out: np.ndarray,
    bg: int,
    rect: dict[str, np.ndarray | int | tuple[int, int, int, int]],
    template: np.ndarray,
) -> None:
    """Punch the template mask into `rect` of `out` using background color."""
    r0, c0, r1, c1 = map(int, rect["bbox"])
    template_h, template_w = map(int, template.shape)
    rect_h = (r1 - r0 + 1)
    rect_w = (c1 - c0 + 1)
    assert template_h <= rect_h and template_w <= rect_w

    dr = (rect_h - template_h) // 2
    dc = (rect_w - template_w) // 2
    for tr, tc in map(tuple, np.argwhere(template)):
        rr = int(r0 + dr + tr)
        cc = int(c0 + dc + tc)
        out[rr, cc] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Pair top-row template masks to rectangles by area rank and punch each rectangle with its paired template.

    Input structure:
    - A global background color dominates the grid.
    - The top area contains several non-background template masks in one color, arranged left to right and separated by background.
    - The rest of the grid contains the same number of solid monochrome rectangles in unique colors and unique areas; their positions are unrelated to template order.

    Rule:
    - Read only template components in top-band; sort them by left-to-right order.
    - Read all solid rectangular payload objects below the template band, sort them by area from small to large.
    - Pair the i-th template with the i-th area-ranked rectangle, center the template mask inside the rectangle, and set those cells to background.
    - Keep all non-template/punched regions unchanged, and clear the entire top template band back to background in the output.

    Key insight:
    - The true pairing key is area rank, not color or geometric proximity; the template acts as a stencil removing cells from a matched rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    nonbg_rows = np.where(np.any(g != bg, axis=1))[0]
    assert nonbg_rows.size > 0, "expected non-background input"

    top_r0 = int(nonbg_rows.min())
    top_r1 = top_r0
    while top_r1 + 1 < g.shape[0] and np.any(g[top_r1 + 1, :] != bg):
        top_r1 += 1

    top_band = g[top_r0 : top_r1 + 1, :]
    top_colors = colors_present(top_band, exclude={bg})
    assert top_colors, "expected at least one template color in top band"

    top_color = None
    for color in sorted(top_colors, key=lambda c: int(np.sum(top_band == c)), reverse=True):
        rows = np.flatnonzero(np.any(g == color, axis=1))
        if rows.size == 0:
            continue
        if rows.min() >= top_r0 and rows.max() <= top_r1:
            top_color = int(color)
            break
    assert top_color is not None, "expected a unique template color confined to the top band"

    template_components = [
        comp
        for comp in find_components(
            g, bg=bg, colors=[top_color], monochromatic=True, connectivity=4, return_format="dict"
        )
        if top_r0 <= int(comp["bbox"][0]) and int(comp["bbox"][2]) <= top_r1
    ]
    template_components.sort(key=lambda comp: int(comp["bbox"][1]))
    assert template_components, "expected at least one template component"

    templates = [(_component_mask(comp), comp["bbox"]) for comp in template_components]

    rectangle_colors = [int(c) for c in colors_present(g, exclude={bg, top_color})]
    rectangle_components = []
    for comp in find_components(
        g,
        bg=bg,
        colors=rectangle_colors,
        monochromatic=True,
        connectivity=4,
        return_format="dict",
    ):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if r0 <= top_r1:
            continue
        if int(comp["area"]) != h * w:
            continue
        if int(comp["area"]) < 1:
            continue
        rectangle_components.append(comp)

    rectangle_components.sort(key=lambda comp: int(comp["area"]))
    assert len(rectangles := rectangle_components) == len(templates), "paired count mismatch"
    assert len(rectangles) > 0, "expected at least one rectangle"
    assert len({int(comp["area"]) for comp in rectangles}) == len(rectangles), "rectangle areas must be unique"

    out = g.copy()
    out[top_r0 : top_r1 + 1, :] = bg

    for comp, (template, _bbox) in zip(rectangles, templates):
        _punch_template(out, bg, comp, template)

    return out


def _random_connected_mask(h: int, w: int, cells: int) -> np.ndarray:
    """Create a random connected boolean mask with exactly `cells` filled cells."""
    h = int(h)
    w = int(w)
    cells = int(cells)
    assert 1 <= cells <= h * w

    mask = np.zeros((h, w), dtype=bool)
    start_r = random.randrange(h)
    start_c = random.randrange(w)
    filled = [(start_r, start_c)]
    mask[start_r, start_c] = True

    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while len(filled) < cells:
        candidates = set()
        for r, c in filled:
            for dr, dc in neighbors:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    candidates.add((rr, cc))
        assert candidates, "failed to grow connected mask"
        r, c = random.choice(tuple(candidates))
        mask[r, c] = True
        filled.append((r, c))
    return mask


def _place_template_band(
    width: int, top_color: int, bg: int, required: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int, np.ndarray]]]:
    """Generate a top band with `required` templates and return the band plus descriptors."""
    top_h = random.choice((1, 2))
    band = np.full((top_h, width), bg, dtype=int)
    descriptors: list[tuple[int, int, int, int, np.ndarray]] = []
    margin = 1
    cursor = random.randint(0, margin)
    placed = 0

    for _ in range(required):
        tw = random.randint(2, 4)
        if cursor + tw + margin >= width:
            raise RuntimeError("template layout overflow")

        th = top_h
        target_cells = random.randint(max(1, th), th * tw)
        mask = _random_connected_mask(th, tw, target_cells)
        c0 = cursor
        paste(band, np.where(mask, top_color, bg), 0, c0)
        r0, c1 = 0, c0 + tw - 1
        r1 = top_h - 1
        descriptors.append((r0, c0, r1, c1, mask))
        placed += 1
        cursor = c1 + random.randint(1, 3)
        if placed == required:
            break
        if cursor + 2 >= width:
            break

    if len(descriptors) != required:
        raise RuntimeError("template generation failed")
    return band, descriptors


def _fit_rectangles(
    grid: np.ndarray,
    start_row: int,
    rectangles: list[tuple[int, int, int]],
    colors: list[int],
    bg: int,
) -> list[tuple[int, int, int, int, int]]:
    """Place non-overlapping solid rectangles in lower rows and return their descriptors."""
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[:start_row, :] = True
    filled: list[tuple[int, int, int, int, int]] = []

    max_row, max_col = grid.shape
    for (h, w, area), color in zip(rectangles, colors):
        del area
        found = False
        for _ in range(400):
            r0 = random.randint(start_row, max_row - h)
            c0 = random.randint(0, max_col - w)
            r1 = r0 + h
            c1 = c0 + w
            if occupied[r0:r1, c0:c1].any():
                continue
            rect = np.full((h, w), int(color), dtype=int)
            paste(grid, rect, r0, c0)
            occupied[r0:r1, c0:c1] = True
            filled.append((r0, c0, r1 - 1, c1 - 1, color))
            found = True
            break
        if not found:
            raise RuntimeError("failed to place rectangles")
    return filled


def _add_distractor_shape(grid: np.ndarray, bg: int, used_colors: set[int]) -> None:
    """Paint a small non-rectangular distractor in the lower area."""
    palette = [c for c in Color.ALL_COLORS if c not in used_colors and c != bg and c != Color.BACKGROUND]
    if not palette:
        return
    color = random.choice(palette)
    h, w = grid.shape

    shape_options = [
        np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    ]
    shape = random.choice(shape_options)
    sh, sw = shape.shape

    anchors = [(r, c) for r in range(2, h - sh - 1) for c in range(1, w - sw - 1)]
    random.shuffle(anchors)
    for r0, c0 in anchors:
        region = (grid[r0 : r0 + sh, c0 : c0 + sw] != bg)
        if np.any(region):
            continue
        for rr, cc in np.argwhere(shape):
            grid[r0 + rr, c0 + cc] = color
        return


def generate_input() -> np.ndarray:
    max_attempts = 300

    for _ in range(max_attempts):
        H = random.randint(16, 28)
        W = random.randint(16, 28)
        bg = random.choice([c for c in Color.ALL_COLORS if c != Color.BACKGROUND])

        pair_count = random.randint(2, 5)

        template_color = random.choice([c for c in Color.ALL_COLORS if c != bg])

        try:
            top_band, template_data = _place_template_band(W, template_color, bg, pair_count)
        except RuntimeError:
            continue

        top_h = top_band.shape[0]
        start_row = top_h + 1
        if H <= start_row + 3:
            continue

        out = np.full((H, W), bg, dtype=int)
        out[:top_h, :] = top_band

        templates = template_data
        max_template_h = max(r1 - r0 + 1 for r0, c0, r1, c1, _ in templates)
        max_template_w = max(c1 - c0 + 1 for r0, c0, r1, c1, _ in templates)
        avail_h = H - start_row
        if avail_h < 2 * max_template_h:
            continue

        rect_colors = random.sample(
            [c for c in Color.ALL_COLORS if c not in {bg, template_color}],
            pair_count,
        )

        rect_sizes: list[tuple[int, int, int]] = []
        used_areas = set()
        for _ in range(pair_count * 2):
            h = random.randint(max(2, max_template_h), 10)
            w = random.randint(max(2, max_template_w), 10)
            area = h * w
            if area in used_areas:
                continue
            if h > avail_h or h < max_template_h or w < max_template_w or w >= W:
                continue
            rect_sizes.append((h, w, area))
            used_areas.add(area)
            if len(rect_sizes) >= pair_count:
                break

        if len(rect_sizes) < pair_count:
            continue

        rect_sizes = rect_sizes[:pair_count]
        rect_sizes.sort(key=lambda t: t[2])

        try:
            rect_fill = _fit_rectangles(out, start_row, rect_sizes, rect_colors, bg)
        except RuntimeError:
            continue

        if random.random() < 0.45:
            _add_distractor_shape(
                out, bg, set(rect_colors) | {template_color, bg}
            )
        if random.random() < 0.45:
            _add_distractor_shape(
                out, bg, set(rect_colors) | {template_color, bg}
            )

        try:
            solved = main(out)
            if not isinstance(solved, np.ndarray):
                continue
            if solved.shape != out.shape:
                continue
            if not np.all(solved[:start_row, :] == bg):
                continue
            return out
        except Exception:
            continue

    raise RuntimeError("failed to generate valid 4.py input")

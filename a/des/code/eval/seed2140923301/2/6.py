import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300
TEMPLATE_RETRIES = 200
PLACEMENT_RETRIES = 300


def _rotate_cw(arr: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(arr, -steps)


def _separator_col(grid: np.ndarray) -> int:
    cols = np.flatnonzero(np.all(grid == SEP, axis=0))
    assert cols.size >= 1, "expected a full-height grey separator column"
    return int(cols[0])


def _panel_spans(left: np.ndarray) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    sep_rows = set(int(r) for r in np.flatnonzero(np.all(left == SEP, axis=1)))
    start = 0
    for r in range(left.shape[0] + 1):
        if r == left.shape[0] or r in sep_rows:
            if start < r:
                spans.append((start, r))
            start = r + 1
    return spans


def _extract_templates(grid: np.ndarray) -> Dict[int, np.ndarray]:
    sep_col = _separator_col(grid)
    left = grid[:, :sep_col]
    templates: Dict[int, np.ndarray] = {}
    for r0, r1 in _panel_spans(left):
        panel = left[r0:r1, :]
        mask = (panel != BG) & (panel != SEP)
        assert mask.any(), "each left panel must contain a template"
        colors = np.unique(panel[mask])
        assert colors.size == 1, "each template must be monochrome"
        color = int(colors[0])
        coords = np.argwhere(mask)
        rr0, cc0 = coords.min(axis=0)
        rr1, cc1 = coords.max(axis=0) + 1
        template = np.where(mask[rr0:rr1, cc0:cc1], color, BG).astype(int)
        assert color not in templates, "template colors must be unique"
        templates[color] = template
    assert templates, "expected at least one template"
    return templates


def _quadrant_rotation(local_r: int, local_c: int, canvas_h: int, canvas_w: int) -> int:
    top = local_r < int(np.ceil(canvas_h / 2.0))
    left = local_c < int(np.ceil(canvas_w / 2.0))
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and not left:
        return 2
    return 3


def _stamp_cells(template: np.ndarray, anchor_r: int, anchor_c: int) -> List[Tuple[int, int, int]]:
    coords = np.argwhere(template != BG)
    center_r = template.shape[0] // 2
    center_c = template.shape[1] // 2
    cells: List[Tuple[int, int, int]] = []
    for tr, tc in coords:
        rr = anchor_r + int(tr) - center_r
        cc = anchor_c + int(tc) - center_c
        cells.append((rr, cc, int(template[tr, tc])))
    return cells


def _paint_stamp(out: np.ndarray, template: np.ndarray, anchor_r: int, anchor_c: int, canvas_c0: int) -> None:
    for rr, cc, color in _stamp_cells(template, anchor_r, anchor_c):
        if 0 <= rr < out.shape[0] and canvas_c0 <= cc < out.shape[1]:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each canvas marker to the same-colored stamp template on the left and rotate by quadrant.

    Input structure:
    - A full-height grey column splits the grid into a left template strip and a right black canvas.
    - The left strip is further partitioned by grey rows into several panels, one monochrome stamp per panel.
    - The right canvas contains only singleton colored markers whose colors match the left templates.

    Rule:
    - A marker selects the template with the same color.
    - Marker quadrant on the canvas determines rotation:
      top-left -> 0, top-right -> 90 clockwise, bottom-right -> 180, bottom-left -> 270 clockwise.
    - Stamp the rotated template centered on the marker.
    - The left template strip stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    sep_col = _separator_col(g)
    canvas_c0 = sep_col + 1
    templates = _extract_templates(g)

    out = g.copy()
    out[:, canvas_c0:] = BG
    canvas = g[:, canvas_c0:]
    canvas_h, canvas_w = canvas.shape

    for local_r, local_c in np.argwhere(canvas != BG):
        color = int(canvas[local_r, local_c])
        assert color in templates, "every marker color must have a matching template"
        steps = _quadrant_rotation(int(local_r), int(local_c), canvas_h, canvas_w)
        rotated = _rotate_cw(templates[color], steps)
        _paint_stamp(out, rotated, int(local_r), canvas_c0 + int(local_c), canvas_c0)

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _mask_has_distinct_rotations(mask: np.ndarray) -> bool:
    rots = [_rotate_cw(mask.astype(int), k).astype(bool) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rots[i], rots[j]):
                return False
    return True


def _random_template_mask(rng: random.Random) -> np.ndarray:
    for _ in range(TEMPLATE_RETRIES):
        h = rng.choice((3, 5))
        w = rng.choice((3, 5))
        center = (h // 2, w // 2)
        cells = {center}

        border_targets = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        rng.shuffle(border_targets)
        for tr, tc in border_targets:
            r, c = center
            while (r, c) != (tr, tc):
                options = []
                if r < tr:
                    options.append((r + 1, c))
                elif r > tr:
                    options.append((r - 1, c))
                if c < tc:
                    options.append((r, c + 1))
                elif c > tc:
                    options.append((r, c - 1))
                r, c = rng.choice(options)
                cells.add((r, c))

        frontier = {
            nbr
            for r, c in list(cells)
            for nbr in _neighbors4(r, c, h, w)
            if nbr not in cells
        }
        if len(cells) >= h * w:
            continue
        max_area = min(h * w - 1, len(cells) + rng.randint(0, max(2, (h * w) // 3)))
        if max_area < len(cells):
            continue
        target_area = rng.randint(len(cells), max_area)
        while len(cells) < target_area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors4(cell[0], cell[1], h, w):
                if nbr not in cells:
                    frontier.add(nbr)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if mask.sum() < 4:
            continue
        if np.all(mask.sum(axis=0) <= 1) or np.all(mask.sum(axis=1) <= 1):
            continue
        if not _mask_has_distinct_rotations(mask):
            continue
        return mask

    raise RuntimeError("failed to build a non-degenerate template")


def _marker_specs(rng: random.Random, colors: Sequence[int]) -> List[Tuple[int, int]]:
    specs = [(quadrant, rng.choice(colors)) for quadrant in range(4)]
    used = {color for _, color in specs}
    for color in colors:
        if color not in used:
            specs.append((rng.randrange(4), color))

    repeat_color = rng.choice(colors)
    repeat_quadrants = {quadrant for quadrant, color in specs if color == repeat_color}
    choices = [q for q in range(4) if q not in repeat_quadrants]
    specs.append((rng.choice(choices) if choices else rng.randrange(4), repeat_color))

    for _ in range(rng.randint(0, 2)):
        specs.append((rng.randrange(4), rng.choice(colors)))
    return specs


def _quadrant_bounds(quadrant: int, h: int, w: int) -> Tuple[int, int, int, int]:
    split_r = int(np.ceil(h / 2.0))
    split_c = int(np.ceil(w / 2.0))
    if quadrant == 0:
        return 0, split_r - 1, 0, split_c - 1
    if quadrant == 1:
        return 0, split_r - 1, split_c, w - 1
    if quadrant == 2:
        return split_r, h - 1, split_c, w - 1
    return split_r, h - 1, 0, split_c - 1


def _place_markers(
    rng: random.Random,
    grid_h: int,
    canvas_w: int,
    templates: Dict[int, np.ndarray],
    specs: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    blocked = np.zeros((grid_h, canvas_w), dtype=bool)
    placements: List[Tuple[int, int, int]] = []

    ordered_specs = sorted(
        specs,
        key=lambda item: int(np.count_nonzero(templates[item[1]])),
        reverse=True,
    )

    for quadrant, color in ordered_specs:
        rotated = _rotate_cw(templates[color], quadrant)
        center_r = rotated.shape[0] // 2
        center_c = rotated.shape[1] // 2
        top = center_r
        bottom = rotated.shape[0] - 1 - center_r
        left = center_c
        right = rotated.shape[1] - 1 - center_c
        qr0, qr1, qc0, qc1 = _quadrant_bounds(quadrant, grid_h, canvas_w)

        row_lo = max(qr0, top)
        row_hi = min(qr1, grid_h - 1 - bottom)
        col_lo = max(qc0, left)
        col_hi = min(qc1, canvas_w - 1 - right)
        if row_lo > row_hi or col_lo > col_hi:
            raise RuntimeError("quadrant too small for template placement")

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            anchor_r = rng.randint(row_lo, row_hi)
            anchor_c = rng.randint(col_lo, col_hi)
            cells = _stamp_cells(rotated, anchor_r, anchor_c)
            if any(blocked[rr, cc] for rr, cc, _ in cells):
                continue

            placements.append((anchor_r, anchor_c, color))
            for rr, cc, _ in cells:
                for pr in range(max(0, rr - 1), min(grid_h, rr + 2)):
                    for pc in range(max(0, cc - 1), min(canvas_w, cc + 2)):
                        blocked[pr, pc] = True
            placed = True
            break

        if not placed:
            raise RuntimeError("failed to place all markers without overlap")

    return placements


def _build_input(rng: random.Random) -> np.ndarray:
    n_templates = rng.randint(4, 5)
    colors = list(rng.sample(PALETTE, n_templates))

    templates: Dict[int, np.ndarray] = {}
    for color in colors:
        templates[color] = np.where(_random_template_mask(rng), color, BG).astype(int)

    max_w = max(template.shape[1] for template in templates.values())
    left_w = max_w + rng.randint(3, 6)
    panel_heights = [template.shape[0] + rng.randint(2, 5) for template in templates.values()]
    grid_h = sum(panel_heights) + (n_templates - 1)
    canvas_w = rng.choice(tuple(range(max(14, max_w * 2 + 4), 23, 2)))
    sep_col = left_w
    grid = np.full((grid_h, left_w + 1 + canvas_w), BG, dtype=int)
    grid[:, sep_col] = SEP

    row0 = 0
    for index, color in enumerate(colors):
        template = templates[color]
        panel_h = panel_heights[index]
        top_offset = rng.randint(1, panel_h - template.shape[0] - 1)
        left_offset = rng.randint(1, left_w - template.shape[1] - 1)
        panel_r0 = row0
        grid[
            panel_r0 + top_offset : panel_r0 + top_offset + template.shape[0],
            left_offset : left_offset + template.shape[1],
        ] = np.where(template != BG, color, grid[
            panel_r0 + top_offset : panel_r0 + top_offset + template.shape[0],
            left_offset : left_offset + template.shape[1],
        ])
        row0 += panel_h
        if index != n_templates - 1:
            grid[row0, :sep_col] = SEP
            row0 += 1

    specs = _marker_specs(rng, colors)
    placements = _place_markers(rng, grid_h, canvas_w, templates, specs)
    canvas_c0 = sep_col + 1
    for r, c, color in placements:
        grid[r, canvas_c0 + c] = color

    expected = main(grid)
    canvas = expected[:, canvas_c0:]
    if np.count_nonzero(canvas) <= len(placements):
        raise RuntimeError("stamped output is too trivial")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            return _build_input(rng)
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid stamp-rotation puzzle")

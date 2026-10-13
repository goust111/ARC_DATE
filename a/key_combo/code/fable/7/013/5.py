from __future__ import annotations

from collections import deque
import random

import numpy as np


# This file is deliberately self-contained so SourceFileLoader can load it from
# any working directory.  Helpers follow arc_utils' (row, column) and inclusive
# bounding-box conventions.


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if not coords.size:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    unseen = mask.astype(bool).copy()
    result: list[np.ndarray] = []
    while unseen.any():
        start = tuple(map(int, np.argwhere(unseen)[0]))
        unseen[start] = False
        stack = [start]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _normalize(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "shape must be nonempty"
    return _crop(mask.astype(bool), box)


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    """All distinct rotations and reflections, normalized to their bboxes."""
    base = _normalize(mask)
    variants: list[np.ndarray] = []
    for reflected in (base, np.fliplr(base)):
        for k in range(4):
            candidate = _normalize(np.rot90(reflected, k))
            if not any(candidate.shape == old.shape and np.array_equal(candidate, old) for old in variants):
                variants.append(candidate)
    return variants


def _congruent(a: np.ndarray, b: np.ndarray) -> bool:
    aa = _normalize(a)
    return any(aa.shape == variant.shape and np.array_equal(aa, variant) for variant in _transforms(b))


def _enclosed_cavity(component_crop: np.ndarray) -> np.ndarray:
    """Return background cells 4-enclosed by one outline in its own bbox."""
    wall = component_crop.astype(bool)
    h, w = wall.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    boundary = ([(0, c) for c in range(w)] + [(h - 1, c) for c in range(w)]
                + [(r, 0) for r in range(h)] + [(r, w - 1) for r in range(h)])
    for cell in boundary:
        if not wall[cell] and not outside[cell]:
            outside[cell] = True
            queue.append(cell)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                queue.append((rr, cc))
    cavity = ~wall & ~outside
    assert cavity.any(), "every candidate outline must enclose a nonempty cavity"
    return _normalize(cavity)


def _parse_layout(g: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Find separator, aligned top panels, and lower candidate region."""
    h, w = g.shape
    horizontal: list[tuple[int, int]] = []
    for r in range(h):
        if np.all(g[r] == g[r, 0]):
            horizontal.append((r, int(g[r, 0])))
    assert horizontal, "expected a full-width horizontal separator"

    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(horizontal):
        start, color = horizontal[i]
        end = start
        i += 1
        while i < len(horizontal) and horizontal[i] == (end + 1, color):
            end = horizontal[i][0]
            i += 1
        if start > 0 and end < h - 1:
            runs.append((start, end, color))
    structural_runs: list[tuple[int, int, int, list[int]]] = []
    for start, end, color in runs:
        candidate_top = g[:start]
        cols = [c for c in range(w) if np.all(candidate_top[:, c] == color)]
        if (cols and cols == list(range(min(cols), max(cols) + 1))
                and min(cols) == w - 1 - max(cols)):
            structural_runs.append((start, end, color, cols))
    assert len(structural_runs) == 1, "expected one T-shaped divider producing equal top panels"
    sep_r0, sep_r1, sep, sep_cols = structural_runs[0]

    top = g[:sep_r0]
    assert sep_cols, "expected a vertical separator between the top panels"
    assert sep_cols == list(range(min(sep_cols), max(sep_cols) + 1)), "separator columns must be contiguous"
    c0, c1 = min(sep_cols), max(sep_cols)
    left = top[:, :c0]
    right = top[:, c1 + 1 :]
    assert left.shape == right.shape and left.size, "top panels must be equal and nonempty"
    assert not np.any(left == sep) and not np.any(right == sep), "separator color is reserved for dividers"
    lower = g[sep_r1 + 1 :]
    assert lower.size, "expected candidates below the separator"
    return sep, sep_r1, left, right, lower


def _candidate_records(lower: np.ndarray, bg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    records: list[tuple[np.ndarray, np.ndarray]] = []
    for color in map(int, np.unique(lower)):
        if color == bg:
            continue
        for coords in _components(lower == color, connectivity=8):
            box = (int(coords[:, 0].min()), int(coords[:, 1].min()),
                   int(coords[:, 0].max()), int(coords[:, 1].max()))
            crop = _crop(lower, box)
            entity = crop == color
            assert int(entity.sum()) == len(coords), "candidate bbox may contain no other foreground"
            assert np.all((crop == bg) | (crop == color)), "candidate colors must remain separate"
            records.append((crop, _enclosed_cavity(entity)))
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the outline sprite whose enclosed cavity matches the aligned panels' shared background shape.

    Input structure:
    - A divider splits two equal top mask panels from a lower row of separated,
      differently colored hollow outline sprites on the common background.
    - The two panels use the same cell coordinate system, and every lower sprite
      encloses a nonempty background cavity inside its own bounding box.

    Rule:
    - Align the top panels cell for cell and intersect their background masks.
    - Compare that normalized hidden shape, up to rotation or reflection, with
      each background cavity 4-enclosed by a lower outline.
    - Return the bounding-box crop of the unique matching outline, preserving color.

    Key insight:
    - Neither matching object is directly painted: the source is the panels'
      background intersection and the destination descriptor is an outline's cavity,
      not the outline's colored entity shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    sep, _sep_end, left, right, lower = _parse_layout(g)

    values, counts = np.unique(lower, return_counts=True)
    bg_choices = [(int(n), int(v)) for v, n in zip(values, counts) if int(v) != sep]
    bg = max(bg_choices)[1]
    assert bg != sep

    hidden = (left == bg) & (right == bg)
    assert hidden.any(), "the aligned background intersection must be nonempty"
    hidden = _normalize(hidden)

    records = _candidate_records(lower, bg)
    assert len(records) >= 2, "expected multiple candidate outlines"
    matches = [crop for crop, cavity in records if _congruent(cavity, hidden)]
    assert len(matches) == 1, "expected exactly one congruent enclosed cavity"
    return matches[0].copy()


# ------------------------------ generator ------------------------------


def _random_polyomino(min_area: int = 3, max_area: int = 8) -> np.ndarray:
    area = random.randint(min_area, max_area)
    cells = {(0, 0)}
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    cells = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask


def _outline(cavity: np.ndarray) -> np.ndarray:
    """Make a one-cell 8-neighbor ring that 4-encloses the cavity."""
    h, w = cavity.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1 : h + 1, 1 : w + 1] = cavity
    dilated = padded.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            shifted = np.zeros_like(padded)
            rs = slice(max(0, dr), min(h + 2, h + 2 + dr))
            cs = slice(max(0, dc), min(w + 2, w + 2 + dc))
            src_rs = slice(max(0, -dr), min(h + 2, h + 2 - dr))
            src_cs = slice(max(0, -dc), min(w + 2, w + 2 - dc))
            shifted[rs, cs] = padded[src_rs, src_cs]
            dilated |= shifted
    return dilated & ~padded


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_transforms(mask)).copy()


def _shape_set(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    for _ in range(300):
        candidate = _random_polyomino()
        # Avoid rectangular cavities and enforce genuinely different congruence classes.
        if candidate.all() or any(_congruent(candidate, old) for old in shapes):
            continue
        shapes.append(candidate)
        if len(shapes) == count:
            return shapes
    return None


def _place_exclusive_cells(base_forbidden: np.ndarray, number: int) -> np.ndarray | None:
    choices = [tuple(map(int, p)) for p in np.argwhere(~base_forbidden)]
    if len(choices) < number:
        return None
    result = np.zeros_like(base_forbidden, dtype=bool)
    for cell in random.sample(choices, number):
        result[cell] = True
    return result


def _gen_attempt() -> np.ndarray | None:
    n_candidates = random.choice((3, 4))
    canonical = _shape_set(n_candidates)
    if canonical is None:
        return None
    target_index = random.randrange(n_candidates)

    displayed_cavities = [_random_variant(shape) for shape in canonical]
    outlines = [_outline(cavity) for cavity in displayed_cavities]
    total_width = sum(mask.shape[1] for mask in outlines) + 2 * (n_candidates - 1) + 2
    if total_width > 29:
        return None

    sep_width = random.choice((1, 2))
    panel_w = max(9, (total_width - sep_width + 1) // 2)
    grid_w = 2 * panel_w + sep_width
    if grid_w < total_width or grid_w > 30:
        return None
    panel_h = random.randint(max(7, displayed_cavities[target_index].shape[0] + 2), 10)
    sep_height = random.choice((1, 2))
    lower_h = max(mask.shape[0] for mask in outlines) + 2
    grid_h = panel_h + sep_height + lower_h
    if grid_h > 30:
        return None

    palette = random.sample(range(1, 10), 3 + n_candidates)
    sep_color, left_color, right_color = palette[:3]
    sprite_colors = palette[3:]
    bg = 0

    hidden_shape = _random_variant(canonical[target_index])
    hh, hw = hidden_shape.shape
    if hh + 2 > panel_h or hw + 2 > panel_w:
        return None
    hr = random.randint(1, panel_h - hh - 1)
    hc = random.randint(1, panel_w - hw - 1)
    hidden = np.zeros((panel_h, panel_w), dtype=bool)
    hidden[hr : hr + hh, hc : hc + hw] = hidden_shape

    forbidden = hidden.copy()
    left_extra = _place_exclusive_cells(forbidden, random.randint(2, 5))
    if left_extra is None:
        return None
    forbidden |= left_extra
    right_extra = _place_exclusive_cells(forbidden, random.randint(2, 5))
    if right_extra is None:
        return None
    left_bg = hidden | left_extra
    right_bg = hidden | right_extra

    # Each aligned panel must add indispensable information: neither alone may
    # already be congruent with any candidate cavity.
    for one_panel in (left_bg, right_bg):
        if any(_congruent(one_panel, cavity) for cavity in displayed_cavities):
            return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    left_panel = np.full((panel_h, panel_w), left_color, dtype=int)
    right_panel = np.full((panel_h, panel_w), right_color, dtype=int)
    left_panel[left_bg] = bg
    right_panel[right_bg] = bg
    grid[:panel_h, :panel_w] = left_panel
    grid[:panel_h, panel_w : panel_w + sep_width] = sep_color
    grid[:panel_h, panel_w + sep_width :] = right_panel
    grid[panel_h : panel_h + sep_height, :] = sep_color

    order = list(range(n_candidates))
    random.shuffle(order)
    c = random.randint(1, grid_w - total_width + 1)
    expected = None
    for idx in order:
        ring = outlines[idx]
        oh, ow = ring.shape
        r = panel_h + sep_height + random.randint(1, lower_h - oh - 1)
        crop = np.full((oh, ow), bg, dtype=int)
        crop[ring] = sprite_colors[idx]
        grid[r : r + oh, c : c + ow] = crop
        if idx == target_index:
            expected = crop.copy()
        c += ow + 2
    assert expected is not None

    # Semantic checks beyond solver closure.
    if not np.array_equal(left_bg & right_bg, hidden):
        return None
    matching = [i for i, cavity in enumerate(displayed_cavities) if _congruent(cavity, hidden)]
    if matching != [target_index]:
        return None
    if any(_congruent(outline, hidden) for outline in outlines):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


_SEEN_GENERATED_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is None:
            continue
        output = main(result)
        signature = (tuple(map(int, output.shape)), output.tobytes())
        if signature in _SEEN_GENERATED_OUTPUTS:
            continue
        _SEEN_GENERATED_OUTPUTS.add(signature)
        return result
    raise RuntimeError("failed to generate a valid double-background cavity puzzle")

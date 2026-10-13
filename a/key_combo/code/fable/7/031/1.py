"""ARC task: reconstruct two complementary fragment templates and swap them."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
NEIGHBORS_8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in NEIGHBORS_8:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and mask[nr, nc]
                    and not seen[nr, nc]
                ):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox in the common ARC (r0, c0, r1, c1) convention."""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _copies_by_color(
    grid: np.ndarray, colors: list[int]
) -> dict[int, list[tuple[tuple[int, int], np.ndarray]]]:
    copies: dict[int, list[tuple[tuple[int, int], np.ndarray]]] = {}
    for color in colors:
        copies[color] = []
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            assert int(local.sum()) == len(coords), "a fragment bbox may not enclose another same-color fragment"
            copies[color].append(((r0, c0), local))
        assert len(copies[color]) >= 2, "each color needs several complementary fragments"
    return copies


def _recover_templates(
    copies: dict[int, list[tuple[tuple[int, int], np.ndarray]]], colors: list[int]
) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for color in colors:
        masks = [mask for _anchor, mask in copies[color]]
        height = max(mask.shape[0] for mask in masks)
        width = max(mask.shape[1] for mask in masks)
        template = np.zeros((height, width), dtype=bool)
        for mask in masks:
            h, w = mask.shape
            template[:h, :w] |= mask
        assert all(not np.array_equal(mask, template) for mask in masks), "no fragment may already be complete"
        templates[color] = template
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct two complete templates from complementary fragments, then swap the templates at every anchor.

    Input structure:
    - A black-background grid contains exactly two foreground colors.
    - Each color has several separated 8-connected fragments of its own hidden template.
    - Missing pixels occur to the right or below the reliable top-left anchor, and both hidden templates have the same bbox size.

    Rule:
    - For each color, crop every fragment to its bbox and align the crops at their top-left corners; their union is that color's complete template.
    - Clear all fragments, then simultaneously draw the other color's complete template, in that other color, at every original fragment anchor.

    Key insight:
    - Neither complete shape is visible in one place: both must be recovered across complementary copies before their shapes and colors can be exchanged at the original top-left anchors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = sorted(int(c) for c in np.unique(g) if int(c) != BACKGROUND)
    assert len(colors) == 2, "expected exactly two foreground colors on black"

    copies = _copies_by_color(g, colors)
    templates = _recover_templates(copies, colors)
    color_a, color_b = colors
    assert templates[color_a].shape == templates[color_b].shape, "the two templates must have equal bbox size"

    other = {color_a: color_b, color_b: color_a}
    out = g.copy()
    out[out != BACKGROUND] = BACKGROUND
    height, width = out.shape
    for source_color in colors:
        target_color = other[source_color]
        target = templates[target_color]
        th, tw = target.shape
        for (r0, c0), _fragment in copies[source_color]:
            assert r0 + th <= height and c0 + tw <= width, "replacement template must fit at its anchor"
            region = out[r0 : r0 + th, c0 : c0 + tw]
            assert np.all(region[target] == BACKGROUND), "replacement templates may not overlap"
            region[target] = target_color
    return out


# ------------------------------ generator ------------------------------

GRID_SIDE_RANGE = (18, 28)
TEMPLATE_SIDE_RANGE = (4, 6)
COPY_COUNT_RANGE = (2, 3)
GENERATION_RETRIES = 400


def _connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and len(comps[0]) == int(mask.sum())


def _make_template(height: int, width: int) -> np.ndarray:
    """Make a connected full-bbox mask with a visible top/left anchor spine."""
    template = np.zeros((height, width), dtype=bool)
    template[0, :] = True
    template[:, 0] = True
    interior = [(r, c) for r in range(1, height) for c in range(1, width)]
    target = random.randint(max(4, len(interior) // 3), max(4, (3 * len(interior)) // 4))
    for r, c in random.sample(interior, min(target, len(interior))):
        template[r, c] = True
    return template


def _damaged_copies(template: np.ndarray, count: int) -> list[np.ndarray] | None:
    """Make incomplete masks whose union needs every copy."""
    removable = [
        (int(r), int(c))
        for r, c in np.argwhere(template)
        if int(r) > 0 and int(c) > 0
    ]
    if len(removable) < count + 1:
        return None
    witnesses = random.sample(removable, count)
    masks = [template.copy() for _ in range(count)]

    # Witness i occurs only in copy i: every copy is necessary to the union.
    for i, witness in enumerate(witnesses):
        for j in range(count):
            if i != j:
                masks[j][witness] = False

    for cell in removable:
        if cell in witnesses or random.random() >= 0.45:
            continue
        missing_from = random.sample(range(count), random.randint(1, count - 1))
        for idx in missing_from:
            masks[idx][cell] = False

    if any(np.array_equal(mask, template) or not _connected(mask) for mask in masks):
        return None
    if not np.array_equal(np.logical_or.reduce(masks), template):
        return None
    for i, witness in enumerate(witnesses):
        if not masks[i][witness] or any(masks[j][witness] for j in range(count) if j != i):
            return None
    return masks


def _templates_are_contrasting(a: np.ndarray, b: np.ndarray) -> bool:
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(b & ~a))
    return only_a >= 2 and only_b >= 2


def _place_windows(
    grid_shape: tuple[int, int], box_shape: tuple[int, int], total: int
) -> list[tuple[int, int]] | None:
    height, width = grid_shape
    bh, bw = box_shape
    candidates = [
        (r, c)
        for r in range(1, height - bh)
        for c in range(1, width - bw)
    ]
    def separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ar, ac = a
        br, bc = b
        return ar + bh + 1 <= br or br + bh + 1 <= ar or ac + bw + 1 <= bc or bc + bw + 1 <= ac

    # Bounded randomized packing avoids an exponential search on small canvases.
    for _ in range(80):
        placed: list[tuple[int, int]] = []
        trial = candidates[:]
        random.shuffle(trial)
        while len(placed) < total:
            valid = [pos for pos in trial if all(separated(pos, old) for old in placed)]
            if not valid:
                break
            pos = random.choice(valid)
            placed.append(pos)
            trial.remove(pos)
        if len(placed) == total:
            return placed
    return None


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    colors: tuple[int, int],
    templates: tuple[np.ndarray, np.ndarray],
    fragments: tuple[list[np.ndarray], list[np.ndarray]],
    anchors: tuple[list[tuple[int, int]], list[tuple[int, int]]],
) -> bool:
    """Check the described semantics independently of reconstruction by main."""
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if set(map(int, np.unique(inp))) != {BACKGROUND, *colors}:
        return False
    if not _templates_are_contrasting(*templates):
        return False
    for color_index, color in enumerate(colors):
        masks = fragments[color_index]
        template = templates[color_index]
        if len(_components(inp == color)) != len(masks):
            return False
        if any(np.array_equal(mask, template) or not _connected(mask) for mask in masks):
            return False
        if not np.array_equal(np.logical_or.reduce(masks), template):
            return False
        # Removing any one copy must lose information from the recovered union.
        for skip in range(len(masks)):
            reduced = np.logical_or.reduce([m for i, m in enumerate(masks) if i != skip])
            if np.array_equal(reduced, template):
                return False
        other = 1 - color_index
        target = templates[other]
        target_color = colors[other]
        for r0, c0 in anchors[color_index]:
            window = expected[r0 : r0 + target.shape[0], c0 : c0 + target.shape[1]]
            if not np.array_equal(window == target_color, target):
                return False
    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(solved, expected) and not np.array_equal(inp, expected)


def _generation_attempt() -> np.ndarray | None:
    grid_shape = (random.randint(*GRID_SIDE_RANGE), random.randint(*GRID_SIDE_RANGE))
    box_shape = (random.randint(*TEMPLATE_SIDE_RANGE), random.randint(*TEMPLATE_SIDE_RANGE))
    counts = (random.randint(*COPY_COUNT_RANGE), random.randint(*COPY_COUNT_RANGE))
    colors = tuple(random.sample(range(1, 10), 2))

    template_a = _make_template(*box_shape)
    template_b = _make_template(*box_shape)
    if not _templates_are_contrasting(template_a, template_b):
        return None
    fragments_a = _damaged_copies(template_a, counts[0])
    fragments_b = _damaged_copies(template_b, counts[1])
    if fragments_a is None or fragments_b is None:
        return None

    positions = _place_windows(grid_shape, box_shape, sum(counts))
    if positions is None:
        return None
    random.shuffle(positions)
    anchors = (positions[: counts[0]], positions[counts[0] :])

    inp = np.zeros(grid_shape, dtype=int)
    expected = np.zeros(grid_shape, dtype=int)
    templates = (template_a, template_b)
    fragments = (fragments_a, fragments_b)
    for group in range(2):
        for (r0, c0), fragment in zip(anchors[group], fragments[group]):
            inp[r0 : r0 + box_shape[0], c0 : c0 + box_shape[1]][fragment] = colors[group]
            target_group = 1 - group
            target = templates[target_group]
            expected[r0 : r0 + box_shape[0], c0 : c0 + box_shape[1]][target] = colors[target_group]

    if _semantic_check(inp, expected, colors, templates, fragments, anchors):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate valid puzzle input."""
    for _ in range(GENERATION_RETRIES):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-template swap input")

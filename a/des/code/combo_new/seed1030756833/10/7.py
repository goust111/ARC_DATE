from __future__ import annotations

from collections import deque
import random

import numpy as np


ORANGE = 7
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return bool(np.any(mask)) and len(_components(mask)) == 1


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored damaged copy by using the orange sprite as a shape template.

    Input structure:
    - The dominant color is the background. Exactly one complete orange shape is
      placed in a corner area and acts as a reference template.
    - Elsewhere there are three or four differently colored copies whose full
      shape has the same bounding-box size as the orange template.
    - Each non-orange copy has some template pixels missing, shown as background,
      while enough visible pixels remain to keep the same bounding box.

    Rule:
    - Take the orange shape's bounding box as the template coordinate frame.
    - For every non-background, non-orange color, align that color's bounding box
      to the template bounding box.
    - Wherever the template contains orange and the aligned copy contains
      background, fill that cell with the copy's own color.
    - Leave the orange template and all already visible copy pixels unchanged.

    Key insight:
    - The orange object is not another copy to recolor; it supplies only the
      shape mask. Missing pixels are restored in each damaged copy's own color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    assert bg != ORANGE

    template_points = np.argwhere(g == ORANGE)
    assert len(template_points) > 0, "expected one orange template"
    tr0, tc0, tr1, tc1 = _bbox(template_points)
    template_mask = g[tr0 : tr1 + 1, tc0 : tc1 + 1] == ORANGE
    th, tw = template_mask.shape
    assert _touches_all_edges(template_mask), "template bbox must be tight"

    out = g.copy()
    for color in np.unique(g):
        color = int(color)
        if color in (bg, ORANGE):
            continue

        points = np.argwhere(g == color)
        if len(points) == 0:
            continue
        r0, c0, r1, c1 = _bbox(points)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (th, tw), "copy bbox must match template bbox"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == bg) | (patch == color)), "copy bbox must not contain other colors"
        assert np.all((patch != color) | template_mask), "copy pixels must lie on template cells"
        missing = template_mask & (patch == bg)
        out[r0 : r1 + 1, c0 : c1 + 1][missing] = color

    return out


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    if random.choice([False, True]):
        step = 1 if er >= r else -1
        for rr in range(r, er + step, step):
            mask[rr, c] = True
        step = 1 if ec >= c else -1
        for cc in range(c, ec + step, step):
            mask[er, cc] = True
    else:
        step = 1 if ec >= c else -1
        for cc in range(c, ec + step, step):
            mask[r, cc] = True
        step = 1 if er >= r else -1
        for rr in range(r, er + step, step):
            mask[rr, ec] = True


def _grow_template_mask() -> np.ndarray | None:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    mask = np.zeros((h, w), dtype=bool)
    center = (random.randint(1, h - 2), random.randint(1, w - 2))
    edge_targets = [
        (0, random.randint(0, w - 1)),
        (h - 1, random.randint(0, w - 1)),
        (random.randint(0, h - 1), 0),
        (random.randint(0, h - 1), w - 1),
    ]
    random.shuffle(edge_targets)
    for target in edge_targets:
        _draw_path(mask, center, target)

    max_area = max(int(h * w * 0.68), int(mask.sum()))
    max_area = min(max_area, h * w - 1)
    min_area = max(15, int(mask.sum()))
    if min_area > max_area:
        min_area = int(mask.sum())
    target_area = random.randint(min_area, max_area)

    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        mask[random.choice(frontier)] = True

    if random.choice([False, True]):
        mask = np.fliplr(mask)
    if random.choice([False, True]):
        mask = np.flipud(mask)
    k = random.randint(0, 3)
    if k:
        mask = np.rot90(mask, k)

    if int(mask.sum()) < 15 or not _touches_all_edges(mask) or not _is_connected(mask):
        return None
    return mask.copy()


def _partial_mask(template: np.ndarray, missing_count: int) -> np.ndarray | None:
    coords = [tuple(map(int, p)) for p in np.argwhere(template)]
    total = len(coords)
    for _ in range(300):
        removed = set(random.sample(coords, missing_count))
        keep = template.copy()
        for r, c in removed:
            keep[r, c] = False
        if int(keep.sum()) * 2 <= total:
            continue
        if not _touches_all_edges(keep):
            continue
        if not _is_connected(keep):
            continue
        return keep
    return None


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[mask] = int(color)


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rows, cols = occupied.shape
    occupied[max(0, r0 - 1) : min(rows, r0 + h + 1), max(0, c0 - 1) : min(cols, c0 + w + 1)] = True


def _available_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(rows - h + 1):
        for c0 in range(cols - w + 1):
            if not occupied[max(0, r0 - 1) : min(rows, r0 + h + 1), max(0, c0 - 1) : min(cols, c0 + w + 1)].any():
                positions.append((r0, c0))
    return positions


def _corner_position(rows: int, cols: int, h: int, w: int) -> tuple[int, int]:
    margin = random.randint(0, 2)
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        return margin, margin
    if corner == "tr":
        return margin, cols - w - margin
    if corner == "bl":
        return rows - h - margin, margin
    return rows - h - margin, cols - w - margin


def _gen_attempt() -> np.ndarray | None:
    template = _grow_template_mask()
    if template is None:
        return None
    th, tw = template.shape
    template_area = int(template.sum())

    copy_count = random.randint(3, 4)
    max_missing = template_area // 2 - 1
    if max_missing < copy_count:
        return None
    missing_counts = random.sample(range(1, max_missing + 1), copy_count)

    min_rows = max(16, 2 * th + 6)
    min_cols = max(16, 2 * tw + 6)
    if min_rows > 30 or min_cols > 30:
        return None
    rows = random.randint(min_rows, 30)
    cols = random.randint(min_cols, 30)

    all_colors = list(range(10))
    bg_choices = [c for c in all_colors if c != ORANGE]
    bg = random.choice(bg_choices)
    copy_colors = random.sample([c for c in all_colors if c not in (bg, ORANGE)], copy_count)

    inp = np.full((rows, cols), bg, dtype=int)
    expected = np.full((rows, cols), bg, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    tr0, tc0 = _corner_position(rows, cols, th, tw)
    _paint(inp, template, tr0, tc0, ORANGE)
    _paint(expected, template, tr0, tc0, ORANGE)
    _mark_occupied(occupied, tr0, tc0, th, tw)

    for color, missing_count in zip(copy_colors, missing_counts):
        partial = _partial_mask(template, missing_count)
        if partial is None:
            return None
        positions = _available_positions(occupied, th, tw)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paint(inp, partial, r0, c0, color)
        _paint(expected, template, r0, c0, color)
        _mark_occupied(occupied, r0, c0, th, tw)

    if _background_color(inp) != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")

from __future__ import annotations

from collections import deque
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNER_CELLS = {
    "tl": lambda h, w: (0, 0),
    "tr": lambda h, w: (0, w - 1),
    "bl": lambda h, w: (h - 1, 0),
    "br": lambda h, w: (h - 1, w - 1),
}


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_corner_patch(patch: np.ndarray, corner: str) -> np.ndarray:
    out = patch.copy()
    if corner in ("bl", "br"):
        out = np.flipud(out)
    if corner in ("tr", "br"):
        out = np.fliplr(out)
    return out


def _parse_fragment_patch(patch: np.ndarray, bg: int) -> tuple[list[int], int]:
    h, w = patch.shape
    assert h == w and h >= 4, "corner fragment must be a square at least 4 cells wide"
    assert not np.any(patch == int(bg)), "fragment bbox must be fully occupied"

    diagonal = [int(patch[i, i]) for i in range(h)]
    runs: list[tuple[int, int]] = []
    current = diagonal[0]
    length = 1
    for color in diagonal[1:]:
        if color == current:
            length += 1
        else:
            runs.append((current, length))
            current = color
            length = 1
    runs.append((current, length))

    colors = [color for color, _length in runs]
    widths = [width for _color, width in runs]
    assert len(colors) >= 2, "expected at least two visible onion layers"
    assert len(set(colors)) == len(colors), "layer colors must be unique"
    assert len(set(widths)) == 1 and widths[0] >= 2, "fragment layer thickness must be uniform and greater than one"

    fragment_thickness = int(widths[0])
    assert h == fragment_thickness * len(colors), "fragment side must match layer count times thickness"

    palette = np.array(colors, dtype=int)
    rr, cc = np.indices((h, w))
    expected = palette[np.minimum(rr, cc) // fragment_thickness]
    assert np.array_equal(patch, expected), "fragment must be nested L-corners in its own corner frame"
    return colors, fragment_thickness


def _component_corner(
    coords: np.ndarray, bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> str | None:
    h, w = shape
    r0, c0, r1, c1 = bbox
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 4:
        return None

    cells = {tuple(map(int, p)) for p in coords.tolist()}
    candidates: list[str] = []
    if r0 == 0 and c0 == 0 and (0, 0) in cells:
        candidates.append("tl")
    if r0 == 0 and c1 == w - 1 and (0, w - 1) in cells:
        candidates.append("tr")
    if r1 == h - 1 and c0 == 0 and (h - 1, 0) in cells:
        candidates.append("bl")
    if r1 == h - 1 and c1 == w - 1 and (h - 1, w - 1) in cells:
        candidates.append("br")
    assert len(candidates) <= 1, "fragment component cannot occupy multiple corners"
    return candidates[0] if candidates else None


def _locate_fragment(
    grid: np.ndarray, bg: int, comps: list[np.ndarray]
) -> tuple[int, str, tuple[int, int, int, int], list[int], int]:
    h, w = grid.shape
    candidates: list[tuple[int, str, tuple[int, int, int, int], list[int], int]] = []
    for index, coords in enumerate(comps):
        if len(coords) < 4:
            continue
        box = _bbox(coords)
        corner = _component_corner(coords, box, (h, w))
        if corner is None:
            continue
        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        try:
            colors, fragment_thickness = _parse_fragment_patch(
                _normalize_corner_patch(patch, corner), bg
            )
        except AssertionError:
            continue
        candidates.append((index, corner, box, colors, fragment_thickness))

    assert len(candidates) == 1, "expected exactly one valid corner fragment"
    return candidates[0]


def _read_input(
    grid: np.ndarray, *, details: bool = False
) -> tuple[list[int], list[int]] | tuple[list[int], list[int], dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC input dimensions must not exceed 30"

    bg = _background(g)
    comps = _components(g != bg)
    assert comps, "expected a corner fragment and dot clues"

    fragment_index, corner, box, colors, fragment_thickness = _locate_fragment(g, bg, comps)
    counts_by_color = {color: 0 for color in colors}
    distractor_count = 0
    point_colors: list[int] = []

    for index, coords in enumerate(comps):
        if index == fragment_index:
            continue
        assert len(coords) == 1, "all non-fragment foreground components must be single-pixel dots"
        r, c = map(int, coords[0])
        color = int(g[r, c])
        point_colors.append(color)
        if color in counts_by_color:
            counts_by_color[color] += 1
        else:
            distractor_count += 1

    counts = [counts_by_color[color] for color in colors]
    assert all(count > 0 for count in counts), "each fragment color needs at least one counted dot"
    assert 2 * sum(counts) - 1 <= 30, "output onion would exceed ARC dimensions"

    if details:
        return colors, counts, {
            "background": bg,
            "fragment_corner": corner,
            "fragment_bbox": box,
            "fragment_thickness": fragment_thickness,
            "distractor_count": distractor_count,
            "point_colors": point_colors,
        }
    return colors, counts


def _draw_onion(colors: list[int], counts: list[int]) -> np.ndarray:
    total = int(sum(counts))
    side = 2 * total - 1
    assert 1 <= side <= 30
    out = np.empty((side, side), dtype=int)
    offset = 0
    for color, thickness in zip(colors, counts):
        assert int(thickness) > 0
        out[offset : side - offset, offset : side - offset] = int(color)
        offset += int(thickness)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a complete concentric square onion from a corner fragment order and same-color dot counts.

    Input structure:
    - A dominant background grid contains one multicolor corner fragment of a nested square-ring onion.
    - The fragment is attached to exactly one grid corner and shows equal-thickness L-corners for colors c1..ck from outside to inside.
    - Away from the fragment, all foreground components are isolated single-pixel dots; some match c1..ck and others are distractors.

    Rule:
    - Read the layer color order c1..ck from the corner fragment in the fragment's own corner coordinate frame.
    - Ignore the fragment's visible thickness, and count only the isolated dots whose colors are in c1..ck.
    - Return a square of side 2*(t1+...+tk)-1, filling nested concentric square layers in order with thicknesses t1..tk.

    Key insight:
    - The corner fragment supplies only the ordered participating colors; the complete onion's layer sizes come from scattered dot counts, not from the fragment geometry or distractor dots.
    """
    colors, counts = _read_input(grid)
    return _draw_onion(colors, counts)


# --------------- generator ---------------

ARC_COLORS = tuple(range(10))
LAYER_COUNT_OPTIONS = (2, 3, 4)
FRAGMENT_THICKNESS_OPTIONS = (2, 3, 4)
COUNT_MIN, COUNT_MAX = 1, 5
MAX_OUTPUT_RADIUS = 15
GRID_MIN_SIDE, GRID_MAX_SIDE = 12, 30
DISTRACTOR_COUNT_RANGE = (2, 7)
GENERATION_ATTEMPTS = 500


def _fragment_patch(colors: list[int], thickness: int) -> np.ndarray:
    side = int(thickness) * len(colors)
    rr, cc = np.indices((side, side))
    return np.array(colors, dtype=int)[np.minimum(rr, cc) // int(thickness)]


def _orient_patch_for_corner(patch: np.ndarray, corner: str) -> np.ndarray:
    out = patch.copy()
    if corner in ("bl", "br"):
        out = np.flipud(out)
    if corner in ("tr", "br"):
        out = np.fliplr(out)
    return out


def _paste_fragment(grid: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    side = patch.shape[0]
    r0 = 0 if corner in ("tl", "tr") else h - side
    c0 = 0 if corner in ("tl", "bl") else w - side
    oriented = _orient_patch_for_corner(patch, corner)
    grid[r0 : r0 + side, c0 : c0 + side] = oriented
    return r0, c0, r0 + side - 1, c0 + side - 1


def _block_rect(blocked: np.ndarray, box: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    blocked[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _block_point(blocked: np.ndarray, r: int, c: int, pad: int) -> None:
    h, w = blocked.shape
    blocked[max(0, r - pad) : min(h, r + pad + 1), max(0, c - pad) : min(w, c + pad + 1)] = True


def _sample_counts(k: int, fragment_thickness: int) -> list[int] | None:
    for _ in range(200):
        counts = [random.randint(COUNT_MIN, COUNT_MAX) for _ in range(k)]
        if sum(counts) > MAX_OUTPUT_RADIUS:
            continue
        if len(set(counts)) < 2:
            continue
        if all(count == int(fragment_thickness) for count in counts):
            continue
        return counts
    return None


def _place_points(grid: np.ndarray, colors: list[int], bg: int, fragment_box: tuple[int, int, int, int]) -> bool:
    blocked = np.zeros(grid.shape, dtype=bool)
    _block_rect(blocked, fragment_box, 1)
    h, w = grid.shape
    for corner_fn in CORNER_CELLS.values():
        r, c = corner_fn(h, w)
        _block_point(blocked, r, c, 1)

    for color in colors:
        candidates = np.argwhere((grid == int(bg)) & (~blocked))
        if len(candidates) == 0:
            return False
        r, c = map(int, candidates[random.randrange(len(candidates))])
        grid[r, c] = int(color)
        _block_point(blocked, r, c, 1)
    return True


def _validate_generated(
    grid: np.ndarray,
    *,
    bg: int,
    colors: list[int],
    counts: list[int],
    fragment_thickness: int,
    distractor_count: int,
) -> bool:
    try:
        parsed_colors, parsed_counts, facts = _read_input(grid, details=True)
        expected = _draw_onion(colors, counts)
        solved = main(grid)
    except AssertionError:
        return False

    if int(facts["background"]) != int(bg):
        return False
    if parsed_colors != colors or parsed_counts != counts:
        return False
    if int(facts["fragment_thickness"]) != int(fragment_thickness):
        return False
    if int(facts["distractor_count"]) != int(distractor_count):
        return False
    if not np.array_equal(solved, expected):
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not any(count != int(fragment_thickness) for count in counts):
        return False
    if len(set(counts)) < 2:
        return False
    if distractor_count < 1:
        return False
    layer_point_total = sum(1 for color in facts["point_colors"] if color in set(colors))
    if layer_point_total != sum(counts):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(ARC_COLORS)
    available = [color for color in ARC_COLORS if color != bg]
    random.shuffle(available)

    k = random.choice(LAYER_COUNT_OPTIONS)
    if len(available) <= k:
        return None
    colors = [int(color) for color in available[:k]]
    distractor_palette = [int(color) for color in available[k:]]
    if not distractor_palette:
        return None

    fragment_thickness = random.choice(FRAGMENT_THICKNESS_OPTIONS)
    counts = _sample_counts(k, fragment_thickness)
    if counts is None:
        return None

    fragment = _fragment_patch(colors, fragment_thickness)
    fragment_side = fragment.shape[0]
    min_h = max(GRID_MIN_SIDE, fragment_side + 5)
    min_w = max(GRID_MIN_SIDE, fragment_side + 5)
    if min_h > GRID_MAX_SIDE or min_w > GRID_MAX_SIDE:
        return None
    h = random.randint(min_h, GRID_MAX_SIDE)
    w = random.randint(min_w, GRID_MAX_SIDE)
    grid = np.full((h, w), int(bg), dtype=int)

    corner = random.choice(tuple(CORNER_CELLS.keys()))
    fragment_box = _paste_fragment(grid, fragment, corner)

    dot_colors: list[int] = []
    for color, count in zip(colors, counts):
        dot_colors.extend([int(color)] * int(count))
    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    dot_colors.extend(random.choice(distractor_palette) for _ in range(distractor_count))
    random.shuffle(dot_colors)

    if not _place_points(grid, dot_colors, int(bg), fragment_box):
        return None

    if not _validate_generated(
        grid,
        bg=int(bg),
        colors=colors,
        counts=counts,
        fragment_thickness=int(fragment_thickness),
        distractor_count=int(distractor_count),
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/4/002/5 input")

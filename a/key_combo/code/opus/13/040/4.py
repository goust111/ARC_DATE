from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5

INPUT_HEIGHT = 5
BIG_PANEL_SIDE = 5
SMALL_SWATCH_SIDES = (2, 3)
FG_COLOR_POOL = (1, 2, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400

ORTHOGONAL_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "cannot crop an empty mask"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def _split_regions(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == BLACK)]
    assert len(sep_cols) == 3, "expected exactly three separator columns"
    bounds = [-1, *sep_cols, g.shape[1]]
    spans = [(bounds[i] + 1, bounds[i + 1]) for i in range(4)]
    assert all(c0 < c1 for c0, c1 in spans), "every panel must be non-empty"
    return [g[:, c0:c1] for c0, c1 in spans]


def _extract_filtered_mask(panel: np.ndarray) -> tuple[np.ndarray, int]:
    colors = [int(c) for c in np.unique(panel) if int(c) != BLACK]
    assert len(colors) == 2 and GREY in colors, "mask panels must use grey plus one foreground color"
    fg = next(color for color in colors if color != GREY)
    comps = _components(panel == fg)
    assert len(comps) == 2, "foreground must split into exactly two 4-connected components"
    comps.sort(key=lambda comp: int(comp.shape[0]), reverse=True)
    assert int(comps[0].shape[0]) > int(comps[1].shape[0]), "largest component must be unique"
    assert int(comps[0].shape[0]) - int(comps[1].shape[0]) >= 2, "area gap must be visually clear"
    keep = _mask_from_coords(comps[0], panel.shape)
    return _crop_mask(keep), fg


def _extract_all_foreground_mask(panel: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(panel) if int(c) not in (BLACK, GREY)]
    assert len(colors) == 1, "panel should contain exactly one non-grey foreground color"
    return _crop_mask(panel == colors[0])


def _extract_swatch_color(panel: np.ndarray) -> int:
    coords = np.argwhere(panel != BLACK)
    assert coords.size > 0, "palette panel must contain a solid square"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = panel[r0 : r1 + 1, c0 : c1 + 1]
    colors = np.unique(patch)
    assert len(colors) == 1, "palette patch must be solid"
    assert patch.shape[0] == patch.shape[1], "palette patch must be square"
    assert patch.shape[0] in SMALL_SWATCH_SIDES, "palette patch side must be 2 or 3"
    return int(colors[0])


def _kron_paint(inner: np.ndarray, outer: np.ndarray, fg: int, bg: int) -> np.ndarray:
    painted = np.kron(outer.astype(np.uint8), inner.astype(np.uint8)).astype(bool)
    out = np.full((outer.shape[0] * inner.shape[0], outer.shape[1] * inner.shape[1]), int(bg), dtype=int)
    out[painted] = int(fg)
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _c in cells) + 1
    w = max(c for _r, c in cells) + 1
    out = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(
        (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
        for variant in _dihedral_variants(mask)
    )


def _is_asymmetric(mask: np.ndarray) -> bool:
    return len(_dihedral_variants(mask)) == 8


def _random_polyomino(area: int) -> np.ndarray:
    cells: set[tuple[int, int]] = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))
        cells = set(_normalize_cells(cells))
    return _cells_to_mask(tuple(sorted(cells)))


def _sample_large_mask() -> np.ndarray:
    for _ in range(200):
        mask = _random_polyomino(random.randint(4, 6))
        mask = random.choice(_dihedral_variants(mask))
        if min(mask.shape) < 2 or max(mask.shape) > 4:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to sample a suitable asymmetric mask")


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _distractor_candidates(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    blocked = _dilate4(occupied)
    valid = ~blocked
    h, w = occupied.shape
    inside: list[list[tuple[int, int]]] = []
    expand: list[list[tuple[int, int]]] = []
    r0, c0, r1, c1 = bbox

    def add_candidate(cells: list[tuple[int, int]]) -> None:
        target = inside if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells) else expand
        target.append(cells)

    for r, c in np.argwhere(valid):
        add_candidate([(int(r), int(c))])

    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and valid[nr, nc]:
                    add_candidate([(r, c), (nr, nc)])

    return inside, expand


def _build_mask_panel(base_mask: np.ndarray, fg_color: int, prefer_inside: bool) -> np.ndarray | None:
    panel = np.full((INPUT_HEIGHT, BIG_PANEL_SIDE), GREY, dtype=int)
    h, w = base_mask.shape
    top = random.randint(0, INPUT_HEIGHT - h)
    left = random.randint(0, BIG_PANEL_SIDE - w)

    occupied = np.zeros(panel.shape, dtype=bool)
    occupied[top : top + h, left : left + w] = base_mask
    bbox = (top, left, top + h - 1, left + w - 1)
    inside, expand = _distractor_candidates(occupied, bbox)

    preferred = inside if prefer_inside else expand
    fallback = expand if prefer_inside else inside
    choices = preferred or fallback
    if not choices:
        return None

    distractor = random.choice(choices)
    panel[occupied] = int(fg_color)
    for r, c in distractor:
        panel[int(r), int(c)] = int(fg_color)
    return panel


def _make_swatch_panel(side: int, color: int) -> np.ndarray:
    panel = np.zeros((INPUT_HEIGHT, side), dtype=int)
    top = random.randint(0, INPUT_HEIGHT - side)
    panel[top : top + side, :] = int(color)
    return panel


def _validate_generated(grid: np.ndarray) -> None:
    regions = _split_regions(grid)
    a, _ = _extract_filtered_mask(regions[0])
    b, _ = _extract_filtered_mask(regions[1])
    all_a = _extract_all_foreground_mask(regions[0])
    all_b = _extract_all_foreground_mask(regions[1])
    green = _extract_swatch_color(regions[2])
    yellow = _extract_swatch_color(regions[3])

    assert _is_asymmetric(a) and _is_asymmetric(b), "kept masks should be asymmetric"
    assert _canonical_key(a) != _canonical_key(b), "the two kept masks must be different up to dihedral symmetry"

    true_out = _kron_paint(a, b, green, yellow)
    wrong_a = _kron_paint(all_a, b, green, yellow)
    wrong_b = _kron_paint(a, all_b, green, yellow)
    swapped = _kron_paint(b, a, green, yellow)

    assert not np.array_equal(true_out, wrong_a), "panel 1 filtering must matter"
    assert not np.array_equal(true_out, wrong_b), "panel 2 filtering must matter"
    assert true_out.shape != swapped.shape or not np.array_equal(true_out, swapped), "outer/inner order must matter"
    assert set(map(int, np.unique(true_out))) == {green, yellow}, "output should visibly use both palette colors"
    assert max(grid.shape) <= 30 and max(true_out.shape) <= 30, "ARC size bound violated"
    assert np.array_equal(main(grid), true_out), "solver and semantic validator disagree"


def main(grid: np.ndarray) -> np.ndarray:
    """Filter each noisy mask panel down to its largest component, then use those filtered masks in a palette-colored Kronecker product.

    Input structure:
    - A black background grid is split by three full-height black separator columns into four horizontal regions.
    - The first two regions are grey panels containing one foreground color, and that foreground is split into exactly two 4-connected components: one larger target component and one smaller distractor.
    - The last two regions each contain a solid colored square swatch on black, providing the foreground and background palette colors for the output.

    Rule:
    - In each of the first two panels, detect the foreground color, find its two 4-connected components, discard the smaller distractor, and crop the larger component to its tight bounding box to obtain binary masks A and B.
    - Use the third panel's swatch color as the output foreground color and the fourth panel's swatch color as the output background color.
    - Compute a Kronecker product with B as the outer mask and A as the inner mask: every foreground cell of B becomes an A-shaped tile colored with the palette, while every background cell of B becomes a solid background-colored tile.

    Key insight:
    - The two Kronecker operands are not given directly: they must first be recovered by filtering away the smaller connected distractor in each noisy mask panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == INPUT_HEIGHT, "expected a 5-row ARC grid"

    panels = _split_regions(g)
    assert len(panels) == 4, "expected four separator-delimited panels"
    assert panels[0].shape[1] == panels[1].shape[1], "the two noisy mask panels must have the same width"

    a, _ = _extract_filtered_mask(panels[0])
    b, _ = _extract_filtered_mask(panels[1])
    fg = _extract_swatch_color(panels[2])
    bg = _extract_swatch_color(panels[3])
    assert fg != bg, "palette colors must differ"

    return _kron_paint(a, b, fg, bg)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inner = _sample_large_mask()
        outer = _sample_large_mask()
        if _canonical_key(inner) == _canonical_key(outer):
            continue

        fg1, fg2 = random.sample(FG_COLOR_POOL, 2)
        side = random.choice(SMALL_SWATCH_SIDES)

        panel1 = _build_mask_panel(inner, fg1, prefer_inside=random.random() < 0.5)
        panel2 = _build_mask_panel(outer, fg2, prefer_inside=random.random() < 0.5)
        if panel1 is None or panel2 is None:
            continue

        swatch_green = _make_swatch_panel(side, GREEN)
        swatch_yellow = _make_swatch_panel(side, YELLOW)
        sep = np.zeros((INPUT_HEIGHT, 1), dtype=int)
        grid = np.concatenate([panel1, sep, panel2, sep, swatch_green, sep, swatch_yellow], axis=1)

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError(f"failed to generate a valid example after {GENERATION_ATTEMPTS} attempts")

from __future__ import annotations

import random

import numpy as np


BG = 0
OUTER_THICKNESSES = (1, 2, 3)
GEN_ATTEMPTS = 500


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _filled_rectangle_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _prefix_sums(mask: np.ndarray) -> np.ndarray:
    sums = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return np.pad(sums, ((1, 0), (1, 0)))


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _square_candidates(grid: np.ndarray, visible_mask: np.ndarray, occ: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere(visible_mask)
    if coords.size == 0:
        return []

    H, W = grid.shape
    vr0, vc0, vr1, vc1 = _bbox_coords(coords)
    min_side = max(vr1 - vr0 + 1, vc1 - vc0 + 1)
    non_bg_prefix = _prefix_sums(grid != BG)
    occ_prefix = _prefix_sums(grid == occ)

    candidates: list[tuple[int, int, int]] = []
    for side in range(min_side, min(H, W) + 1):
        r0_lo = max(0, vr1 - side + 1)
        r0_hi = min(vr0, H - side)
        c0_lo = max(0, vc1 - side + 1)
        c0_hi = min(vc0, W - side)
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        for r0 in range(r0_lo, r0_hi + 1):
            r1 = r0 + side - 1
            for c0 in range(c0_lo, c0_hi + 1):
                c1 = c0 + side - 1
                if _rect_sum(non_bg_prefix, r0, c0, r1, c1) != side * side:
                    continue
                if _rect_sum(occ_prefix, r0, c0, r1, c1) == 0:
                    continue
                candidates.append((r0, c0, side))
    return candidates


def _infer_layers(square: np.ndarray, occ: int) -> list[tuple[int, int]] | None:
    side = int(square.shape[0])
    visible_colors = _colors_present(square, exclude={occ})
    if not visible_colors:
        return None
    if len(visible_colors) == 1:
        return [(visible_colors[0], side)]

    rows, cols = np.indices(square.shape)
    for thickness in OUTER_THICKNESSES:
        inner_side = side - 2 * thickness
        if inner_side < 1:
            continue

        band = (
            (rows < thickness)
            | (rows >= side - thickness)
            | (cols < thickness)
            | (cols >= side - thickness)
        )
        band_colors = _colors_present(square[band], exclude={occ})
        if len(band_colors) != 1:
            continue
        color = band_colors[0]

        if np.any((square[band] != occ) & (square[band] != color)):
            continue
        if np.any((square[~band] != occ) & (square[~band] == color)):
            continue

        inner = _infer_layers(square[thickness:-thickness, thickness:-thickness], occ)
        if inner is None:
            continue
        if any(layer_color == color for layer_color, _ in inner):
            continue
        return inner + [(color, thickness)]

    return None


def _encode_layers(layers: list[tuple[int, int]]) -> np.ndarray:
    total = sum(int(height) for _, height in layers)
    out = np.zeros((total, 1), dtype=int)
    cursor = total
    for color, height in layers:
        cursor -= int(height)
        out[cursor : cursor + height, 0] = int(color)
    return out


def _solve(grid: np.ndarray) -> list[tuple[int, int]]:
    colors = _colors_present(grid, exclude={BG})
    occ_candidates = [color for color in colors if _filled_rectangle_mask(grid == color)]
    if not occ_candidates:
        occ_candidates = colors[:]

    solutions: list[tuple[tuple[int, int], ...]] = []
    for occ in occ_candidates:
        visible_mask = (grid != BG) & (grid != occ)
        for r0, c0, side in _square_candidates(grid, visible_mask, occ):
            square = grid[r0 : r0 + side, c0 : c0 + side]
            layers = _infer_layers(square, occ)
            if layers is None:
                continue
            if set(color for color, _ in layers) != set(_colors_present(square, exclude={occ})):
                continue
            solutions.append(tuple(layers))

    unique = []
    seen = set()
    for layers in solutions:
        if layers not in seen:
            seen.add(layers)
            unique.append(layers)

    assert len(unique) == 1, "expected a unique concentric-layer decomposition"
    return list(unique[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the occluded concentric square layers and serialize them into a one-column stripe.

    Input structure:
    - The background is black.
    - One filled square region contains a solid center square wrapped by several
      concentric square frames of distinct colors.
    - A monochrome rectangular occluder in a different color covers part of the
      square, but every layer still has at least one visible cell.

    Rule:
    - Identify the occluder and the unique non-background square that contains
      the full concentric structure.
    - Peel the square from the outside inward: each outer layer is a 1-3 cell
      thick frame whose visible cells all share one color, and the final
      remaining square is the solid center block.
    - Output a width-1 vertical stripe built from bottom to top: center color
      repeated by its side length, then each outer frame color repeated by that
      frame's thickness.

    Key insight:
    - Because the hidden object is a fully filled concentric square, the true
      structure is the unique all-nonzero square whose visible border bands can
      be recursively stripped even when a rectangle masks part of each layer.
    """
    g = np.asarray(grid, dtype=int)
    layers = _solve(g)
    return _encode_layers(layers)


def _build_structure(layers: list[tuple[int, int]]) -> np.ndarray:
    color0, center_side = layers[0]
    square = np.full((center_side, center_side), color0, dtype=int)
    for color, thickness in layers[1:]:
        nxt_side = square.shape[0] + 2 * thickness
        nxt = np.full((nxt_side, nxt_side), color, dtype=int)
        nxt[thickness:-thickness, thickness:-thickness] = square
        square = nxt
    return square


def _layer_masks(structure: np.ndarray, layers: list[tuple[int, int]]) -> list[np.ndarray]:
    return [structure == color for color, _ in layers]


def _sample_layers() -> list[tuple[int, int]] | None:
    outer_count = random.randint(1, 4)
    center_side = random.randint(1, 8)
    thicknesses = [random.choice(OUTER_THICKNESSES) for _ in range(outer_count)]
    total_side = center_side + 2 * sum(thicknesses)
    if total_side > 24:
        return None

    colors = random.sample(range(1, 10), outer_count + 2)
    structure_colors = colors[: outer_count + 1]
    layers = [(structure_colors[0], center_side)]
    for idx, thickness in enumerate(thicknesses, start=1):
        layers.append((structure_colors[idx], thickness))
    return layers


def _place_structure(structure: np.ndarray) -> tuple[np.ndarray, int, int]:
    side = int(structure.shape[0])
    H = random.randint(side + 2, min(30, side + 9))
    W = random.randint(side + 2, min(30, side + 9))
    r0 = random.randint(1, H - side - 1)
    c0 = random.randint(1, W - side - 1)
    grid = np.zeros((H, W), dtype=int)
    grid[r0 : r0 + side, c0 : c0 + side] = structure
    return grid, r0, c0


def _sample_occluder(structure: np.ndarray, layers: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    side = int(structure.shape[0])
    masks = _layer_masks(structure, layers)
    for _ in range(200):
        height = random.randint(1, side)
        width = random.randint(1, side)
        r0 = random.randint(0, side - height)
        c0 = random.randint(0, side - width)
        rect = np.zeros_like(structure, dtype=bool)
        rect[r0 : r0 + height, c0 : c0 + width] = True

        visible_ok = all(np.any(mask & ~rect) for mask in masks)
        if not visible_ok:
            continue

        occluded_layers = sum(int(np.any(mask & rect)) for mask in masks)
        if occluded_layers < 2:
            continue
        if int(rect.sum()) > int(0.6 * side * side):
            continue
        return r0, c0, height, width
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layers = _sample_layers()
        if layers is None:
            continue

        structure = _build_structure(layers)
        grid, sr0, sc0 = _place_structure(structure)
        occ_rect = _sample_occluder(structure, layers)
        if occ_rect is None:
            continue

        occ_color = random.choice([c for c in range(1, 10) if c not in {color for color, _ in layers}])
        rr0, cc0, height, width = occ_rect
        inp = grid.copy()
        inp[sr0 + rr0 : sr0 + rr0 + height, sc0 + cc0 : sc0 + cc0 + width] = occ_color

        expected = _encode_layers(layers)
        try:
            produced = main(inp)
        except AssertionError:
            continue
        if np.array_equal(produced, expected):
            return inp

    raise RuntimeError("failed to generate a valid concentric-serialization input")

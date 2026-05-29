# Combo variant of b0039139 (index 28)
# combo_spec: D3/2+D10/0
# Name: 像素总数决定重复配正方形输出

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, crop_bbox, random_sprite

# Design analysis for the fused variant
# a) Concrete behavior:
#    - Original: section 2 contributes one repeat per 4-connected green component, and the output is a
#      one-dimensional strip along the layout axis.
#    - Fused: section 2 contributes one repeat per green pixel, so a 2-cell blob and a 5-cell blob do not
#      count the same anymore. The repeated template copies are then packed into the smallest square grid
#      of square slots rather than into one strip.
#    - Example: if section 2 has component sizes 2, 1, 4 then the original task would output 3 copies,
#      while this combo must output 7 copies. Those 7 copies are placed into a 3x3 square slot board, with
#      two empty slots left as background.
# b) Complexity calibration:
#    - The stencil generator uses multiple components with non-uniform sizes, so "count components" and
#      "count pixels" visibly diverge.
#    - The output uses square slots with visible padding when the cropped template is rectangular, making
#      the D10 change easy to notice instead of looking like a strip with extra blank rows/cols.
# c) Implicit-assumption audit against the base solver:
#    - Base assumption "non-black components encode repeats" is invalid; the new solver must inspect only
#      green pixels and sum them directly.
#    - Base assumption "orientation decides a single repeat axis" is invalid; orientation now decides slot
#      fill order inside a square board: horizontal layouts fill rows first, vertical layouts fill columns
#      first.
#    - Base assumption "output cross-axis equals cropped template cross-axis" is invalid; each copy now
#      sits inside a square slot of side max(tile_h, tile_w).
# d) Placement legality:
#    - Template still has a black border and is cropped by bbox.
#    - Stencil components are generated as connected shapes, laid out with at least one blank row/column
#      between their bounding boxes, so they never overlap or merge.
#    - Output copies are stamped into non-overlapping square slots on a square canvas.
# e) Visual observability:
#    - The green-pixel count is human-visible because section 2 contains only a few separated components
#      with varied areas.
#    - The square packing is human-visible because the final board is square and usually contains one
#      partially filled last row or last column.
# f) Visual cleanliness:
#    - Input colors keep their original roles: template uses 4, stencil uses 3, separators use 1, and
#      output swatches avoid 1/3/4 to reduce ambiguity.
#    - Empty square slots remain pure background instead of introducing extra markers.
# g) Generator guarantees for solver assumptions:
#    - The total green-pixel count is chosen from non-square values whose leftover slot pattern makes the
#      row-major vs column-major fill rule visible.
#    - The cropped template is forced to be rectangular, so the square-slot padding remains observable.


TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
COLOR_POOL = [2, 5, 6, 7, 8, 9]

CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
SWATCH_DIM_RANGE = (3, 8)
TOTAL_PIXEL_CHOICES = [5, 6, 7, 10, 11, 12]
COMPONENT_COUNT_RANGE = (2, 4)
COMPONENT_BUILD_TRIES = 80
GEN_RETRIES = 400


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = SEPARATOR_COLOR

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        parts = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", parts

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    parts = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", parts


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _square_slot_tile(tile: np.ndarray, bg_color: int) -> np.ndarray:
    h, w = tile.shape
    side = max(h, w)
    out = np.full((side, side), bg_color, dtype=int)
    r0 = (side - h) // 2
    c0 = (side - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = tile
    return out


def _pack_square(tile: np.ndarray, n_repeats: int, bg_color: int, row_major: bool) -> np.ndarray:
    if n_repeats <= 0:
        return np.empty((0, 0), dtype=int)

    slot = _square_slot_tile(tile, bg_color)
    slot_side = slot.shape[0]
    slots_per_side = int(math.ceil(math.sqrt(n_repeats)))
    canvas_side = slots_per_side * slot_side + (slots_per_side - 1)
    out = np.full((canvas_side, canvas_side), bg_color, dtype=int)

    for idx in range(min(n_repeats, slots_per_side * slots_per_side)):
        if row_major:
            sr = idx // slots_per_side
            sc = idx % slots_per_side
        else:
            sc = idx // slots_per_side
            sr = idx % slots_per_side
        r0 = sr * (slot_side + 1)
        c0 = sc * (slot_side + 1)
        out[r0 : r0 + slot_side, c0 : c0 + slot_side] = slot

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template, count green stencil pixels, and pack the copies into a square board.

    Input structure:
    - Blue full-row or full-column separators split the input into four ordered sections.
    - Section 1 is the black-background template, section 2 is the green counting board,
      section 3 is the foreground color swatch, and section 4 is the background color swatch.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels and recolor the crop with the two swatches.
    - Let the repeat count equal the total number of color-3 pixels in section 2.
    - Put each recolored copy inside a square slot whose side is the larger cropped template dimension.
    - Arrange those slots on the smallest square grid that can hold all copies; unused slots stay background.

    Key insight:
    - The second section now encodes area, not component count, and the layout is no longer a strip:
      the separator orientation only decides whether the square board is filled row-first or column-first.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = int(np.count_nonzero(repetition_pattern == STENCIL_COLOR))
    return _pack_square(tile, n_repeats, background, row_major=(orientation == "horizontal"))


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)

    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
    return np.vstack(parts)


def _make_exact_component(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    if area <= 0 or max_h <= 0 or max_w <= 0 or area > max_h * max_w:
        return None

    candidates: list[tuple[int, int]] = []
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            if h * w >= area:
                candidates.append((h, w))
    random.shuffle(candidates)

    for sh, sw in candidates[: min(len(candidates), COMPONENT_BUILD_TRIES)]:
        mask = np.zeros((sh, sw), dtype=bool)
        start = (random.randrange(sh), random.randrange(sw))
        cells = {start}
        mask[start] = True
        frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < sh and 0 <= nc < sw and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < area and frontier:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            cells.add((nr, nc))
            mask[nr, nc] = True
            add_frontier(nr, nc)

        if len(cells) != area:
            continue

        box = bbox(mask)
        assert box is not None
        cropped = crop_bbox(mask, box)
        if int(np.count_nonzero(cropped)) == area:
            return cropped

    return None


def _random_partition(total: int, n_parts: int) -> list[int] | None:
    if n_parts <= 0 or total < n_parts:
        return None

    for _ in range(COMPONENT_BUILD_TRIES):
        cuts = sorted(random.sample(range(1, total), n_parts - 1))
        parts = []
        prev = 0
        for cut in cuts + [total]:
            parts.append(cut - prev)
            prev = cut
        if max(parts) >= 3 and len(set(parts)) >= 2 and sum(parts) == total:
            random.shuffle(parts)
            return parts
    return None


def _build_stencil(total_pixels: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    n_components = random.randint(COMPONENT_COUNT_RANGE[0], min(COMPONENT_COUNT_RANGE[1], total_pixels))
    parts = _random_partition(total_pixels, n_components)
    if parts is None:
        return None

    pieces: list[np.ndarray] = []
    cross_cap = shared_dim - 2
    if cross_cap <= 0:
        return None

    for area in parts:
        if horizontal:
            piece = _make_exact_component(area, cross_cap, max(area, 2))
        else:
            piece = _make_exact_component(area, max(area, 2), cross_cap)
        if piece is None:
            return None
        pieces.append(piece.astype(int) * STENCIL_COLOR)

    if horizontal:
        width = 2 + sum(piece.shape[1] for piece in pieces) + (len(pieces) - 1)
        grid = np.zeros((shared_dim, width), dtype=int)
        cursor = 1
        for piece in pieces:
            ph, pw = piece.shape
            r0 = (shared_dim - ph) // 2
            grid[r0 : r0 + ph, cursor : cursor + pw] = piece
            cursor += pw + 1
        return grid

    height = 2 + sum(piece.shape[0] for piece in pieces) + (len(pieces) - 1)
    grid = np.zeros((height, shared_dim), dtype=int)
    cursor = 1
    for piece in pieces:
        ph, pw = piece.shape
        c0 = (shared_dim - pw) // 2
        grid[cursor : cursor + ph, c0 : c0 + pw] = piece
        cursor += ph + 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    total_pixels = random.choice(TOTAL_PIXEL_CHOICES)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    if crop_h == crop_w:
        crop_w = crop_w - 1 if crop_w > CROP_SIZE_RANGE[0] else crop_w + 1

    density = random.uniform(*DENSITY_RANGE)
    shape = random_sprite(
        crop_h,
        crop_w,
        density=density,
        color_palette=[TEMPLATE_COLOR],
        connectivity=4,
        symmetry="not_symmetric",
    )
    if shape.shape[0] == shape.shape[1]:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _pack_square(tile, total_pixels, bg_color, row_major=horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = template.shape[0] if horizontal else template.shape[1]
    stencil = _build_stencil(total_pixels, shared_dim, horizontal)
    if stencil is None:
        return None

    if int(np.count_nonzero(stencil == STENCIL_COLOR)) != total_pixels:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

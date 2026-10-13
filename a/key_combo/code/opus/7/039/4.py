from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

TEMPLATE_COLORS = (BLUE, RED)
TEMPLATE_SIZE_OPTIONS = (3, 5)
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 24
COMPLETE_COUNT_OPTIONS = (2, 3)
PARTIAL_COUNT_OPTIONS = (3, 4)
PLACEMENT_GAP = 2
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 800

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: Iterable[Coord]) -> BBox:
    rows, cols = zip(*coords)
    return min(rows), min(cols), max(rows), max(cols)


def _extract_components(mask: np.ndarray) -> list[list[Coord]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _objects(grid: np.ndarray) -> list[tuple[BBox, np.ndarray]]:
    found: list[tuple[BBox, np.ndarray]] = []
    for coords in _extract_components(grid != BLACK):
        box = _bbox_from_coords(coords)
        r0, c0, r1, c1 = box
        found.append((box, grid[r0 : r1 + 1, c0 : c1 + 1].copy()))
    return found


def _horizontal_reflection(size: int, cell: Coord) -> Coord:
    r, c = cell
    return size - 1 - r, c


def _vertical_reflection(size: int, cell: Coord) -> Coord:
    r, c = cell
    return r, size - 1 - c


def _recolor_filled(base_color: int, parity: int) -> int:
    if parity == 0:
        return int(base_color)
    if base_color == RED:
        return ORANGE
    if base_color == BLUE:
        return TEAL
    raise AssertionError(f"unexpected fill color {base_color}")


def _symmetry_orbits(size: int) -> list[tuple[Coord, ...]]:
    seen: set[tuple[Coord, ...]] = set()
    orbits: list[tuple[Coord, ...]] = []
    for r in range(size):
        for c in range(size):
            orbit = tuple(
                sorted(
                    {
                        (r, c),
                        (size - 1 - r, c),
                        (r, size - 1 - c),
                        (size - 1 - r, size - 1 - c),
                    }
                )
            )
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _make_template(size: int, center_color: int) -> np.ndarray:
    other_color = RED if center_color == BLUE else BLUE
    template = np.zeros((size, size), dtype=int)
    center = size // 2

    non_center_orbits: list[tuple[Coord, ...]] = []
    for orbit in _symmetry_orbits(size):
        if orbit == ((center, center),):
            template[center, center] = center_color
            continue
        non_center_orbits.append(orbit)
        color = center_color if random.random() < 0.5 else other_color
        for r, c in orbit:
            template[r, c] = color

    if all(template[r, c] == center_color for orbit in non_center_orbits for r, c in orbit):
        orbit = random.choice(non_center_orbits)
        for r, c in orbit:
            template[r, c] = other_color

    return template


def _bboxes_separated(a: BBox, b: BBox, gap: int = PLACEMENT_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_boxes(sizes: list[int], height: int, width: int) -> list[Coord] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx], reverse=True)
    placements: list[Coord | None] = [None] * len(sizes)
    placed_boxes: list[BBox] = []

    for idx in order:
        size = sizes[idx]
        ok = False
        for _ in range(400):
            r0 = random.randint(0, height - size)
            c0 = random.randint(0, width - size)
            candidate = (r0, c0, r0 + size - 1, c0 + size - 1)
            if all(_bboxes_separated(candidate, existing) for existing in placed_boxes):
                placements[idx] = (r0, c0)
                placed_boxes.append(candidate)
                ok = True
                break
        if not ok:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _mask_conditions(masked: set[Coord], origin: Coord, size: int, require_reference: bool) -> bool:
    center = (size // 2, size // 2)
    if center in masked:
        return False

    even_branch = False
    odd_branch = False
    reference_branch = not require_reference

    for cell in masked:
        global_parity = (origin[0] + cell[0] + origin[1] + cell[1]) % 2
        h_src = _horizontal_reflection(size, cell)
        v_src = _vertical_reflection(size, cell)

        if global_parity == 0 and v_src != cell and h_src in masked and v_src not in masked:
            even_branch = True
        if global_parity == 1 and h_src != cell and v_src in masked and h_src not in masked:
            odd_branch = True
        if h_src in masked and v_src in masked:
            reference_branch = True

    return even_branch and odd_branch and reference_branch


def _choose_masked_cells(size: int, origin: Coord, require_reference: bool) -> set[Coord] | None:
    center = (size // 2, size // 2)
    candidates = [(r, c) for r in range(size) for c in range(size) if (r, c) != center]
    max_mask = min(len(candidates) - 1, 6)

    for _ in range(MASK_ATTEMPTS):
        count = random.randint(3, max_mask)
        masked = set(random.sample(candidates, count))
        if _mask_conditions(masked, origin, size, require_reference):
            return masked
    return None


def _apply_masks(template: np.ndarray, masked: set[Coord]) -> np.ndarray:
    out = template.copy()
    for r, c in masked:
        out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete masked symmetric motif copies, then recolor the restored cells by checkerboard parity.

    Input structure:
    - A black background contains several small odd-sized square motifs, each separated from the others by clear gaps.
    - Complete motifs contain only blue and red pixels and are symmetric across both the horizontal and vertical axes.
    - Partial copies of those motifs have some cells replaced by grey masks, while each copy's center pixel stays visible.

    Rule:
    - Treat every non-black connected component as one motif and use its center pixel color to match partial copies with a complete reference motif of the same shape.
    - For each grey cell, inspect its global checkerboard parity `(r + c) mod 2`: even cells first mirror across the motif's horizontal axis, odd cells first mirror across the vertical axis.
    - If the preferred mirrored source cell is also grey, try the other axis; if both mirrored source cells are grey, use the color at that position in the complete reference motif.
    - Preserve every original non-grey pixel, and recolor only restored odd-parity cells from blue to teal and from red to orange.

    Key insight:
    - The same checkerboard parity controls both which symmetry lookup is attempted first and whether a newly restored pixel keeps its original color or is shifted to a parity-marked recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    templates: dict[int, np.ndarray] = {}
    partials: list[tuple[BBox, np.ndarray, int]] = []

    for box, crop in _objects(g):
        height, width = crop.shape
        assert height == width and height % 2 == 1, "each motif must be an odd square"
        center = height // 2
        center_color = int(crop[center, center])
        assert center_color in TEMPLATE_COLORS, "center must identify a reference template"

        if np.any(crop == GREY):
            partials.append((box, crop, center_color))
            continue

        previous = templates.get(center_color)
        if previous is None:
            templates[center_color] = crop
        else:
            assert previous.shape == crop.shape and np.array_equal(previous, crop), (
                "all complete motifs with the same center color must match"
            )

    assert templates and partials, "expected complete motifs and masked copies"

    for box, crop, center_color in partials:
        reference = templates[center_color]
        assert reference.shape == crop.shape, "partial copy must match its reference size"
        visible = crop != GREY
        assert np.array_equal(crop[visible], reference[visible]), "visible cells must match the reference"

        r0, c0, _r1, _c1 = box
        size = crop.shape[0]
        for local_r, local_c in np.argwhere(crop == GREY):
            local = (int(local_r), int(local_c))
            global_r = r0 + local[0]
            global_c = c0 + local[1]
            parity = (global_r + global_c) % 2

            horizontal = _horizontal_reflection(size, local)
            vertical = _vertical_reflection(size, local)
            primary, secondary = (horizontal, vertical) if parity == 0 else (vertical, horizontal)

            if crop[primary] != GREY:
                base_color = int(crop[primary])
            elif crop[secondary] != GREY:
                base_color = int(crop[secondary])
            else:
                base_color = int(reference[local])

            out[global_r, global_c] = _recolor_filled(base_color, parity)

    return out


def _build_objects() -> tuple[list[dict[str, object]], dict[int, np.ndarray]]:
    sizes = [5, random.choice(TEMPLATE_SIZE_OPTIONS)]
    random.shuffle(sizes)

    templates = {
        color: _make_template(size, color) for color, size in zip(TEMPLATE_COLORS, sizes)
    }

    complete_specs: list[dict[str, object]] = [{"kind": "complete", "color": color} for color in TEMPLATE_COLORS]
    if random.choice(COMPLETE_COUNT_OPTIONS) == 3:
        complete_specs.append({"kind": "complete", "color": random.choice(TEMPLATE_COLORS)})

    partial_total = random.choice(PARTIAL_COUNT_OPTIONS)
    partial_counts = {color: 1 for color in TEMPLATE_COLORS}
    for _ in range(partial_total - len(TEMPLATE_COLORS)):
        partial_counts[random.choice(TEMPLATE_COLORS)] += 1

    partial_specs: list[dict[str, object]] = []
    size5_color = next(color for color, template in templates.items() if template.shape[0] == 5)
    for color in TEMPLATE_COLORS:
        for index in range(partial_counts[color]):
            partial_specs.append(
                {
                    "kind": "partial",
                    "color": color,
                    "require_reference": color == size5_color and index == 0,
                }
            )

    specs = complete_specs + partial_specs
    random.shuffle(specs)
    return specs, templates


def _gen_attempt() -> np.ndarray | None:
    specs, templates = _build_objects()
    sizes = [int(templates[int(spec["color"])].shape[0]) for spec in specs]
    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    placements = _place_boxes(sizes, height, width)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)

    for spec, (r0, c0) in zip(specs, placements):
        color = int(spec["color"])
        template = templates[color]
        sprite = template
        if spec["kind"] == "partial":
            masked = _choose_masked_cells(
                template.shape[0],
                (r0, c0),
                bool(spec["require_reference"]),
            )
            if masked is None:
                return None
            sprite = _apply_masks(template, masked)
        h, w = sprite.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sprite

    if len(_objects(grid)) != len(specs):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

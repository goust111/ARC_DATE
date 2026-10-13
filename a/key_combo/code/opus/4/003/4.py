from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

TEMPLATE_COLOR = TEAL
SEPARATOR_COLOR = GREY
RIGHT_COLORS = (BLUE, RED, GREEN)
RECOLOR_BY_HOLES = {0: ORANGE, 1: YELLOW, 2: MAROON}

SHAPE_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "S0": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "S1": ((0, 0), (0, 1), (0, 2)),
    "S2": ((0, 0), (1, 0), (2, 0)),
}

ALLOWED_TRANSFORMS: dict[str, tuple[tuple[int, bool], ...]] = {
    "S0": tuple((k, flip) for k in range(4) for flip in (False, True)),
    "S1": ((0, False), (0, True), (2, False), (2, True)),
    "S2": ((0, False), (0, True), (2, False), (2, True)),
}

# Each 5x5 mask is an input template body before the missing piece is filled.
# The library covers every pair (final hole count, missing-piece shape kind).
TEMPLATE_LIBRARY: dict[tuple[int, str], np.ndarray] = {
    (0, "S0"): np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 0, 1],
        ],
        dtype=bool,
    ),
    (0, "S1"): np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
        ],
        dtype=bool,
    ),
    (0, "S2"): np.array(
        [
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1],
        ],
        dtype=bool,
    ),
    (1, "S0"): np.array(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    (1, "S1"): np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    (1, "S2"): np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
    (2, "S0"): np.array(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    (2, "S1"): np.array(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=bool,
    ),
    (2, "S2"): np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox_from_coords(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _count_holes(occupied: np.ndarray) -> int:
    holes = 0
    h, w = map(int, occupied.shape)
    for comp in _components(~occupied):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            holes += 1
    return holes


def _transform_mask(mask: np.ndarray, rotation: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, rotation)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _shape_dims(shape_offsets: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape_offsets) + 1, max(c for _, c in shape_offsets) + 1


def _enumerate_fill_candidates(
    template_mask: np.ndarray, shape_offsets: tuple[tuple[int, int], ...]
) -> list[dict]:
    h, w = map(int, template_mask.shape)
    shape_h, shape_w = _shape_dims(shape_offsets)
    options: list[dict] = []
    for r0 in range(h - shape_h + 1):
        for c0 in range(w - shape_w + 1):
            cells = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape_offsets))
            if not all(not template_mask[r, c] for r, c in cells):
                continue
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
                continue

            filled = template_mask.copy()
            for r, c in cells:
                filled[r, c] = True
            if len(_components(filled)) != 1:
                continue

            options.append({"cells_local": cells, "holes": _count_holes(filled)})
    return options


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    parts = _components(grid == int(color))
    out = []
    for comp in parts:
        out.append(
            {
                "color": int(color),
                "coords": comp,
                "bbox": _bbox_from_coords(comp),
            }
        )
    return out


def _right_shapes(grid: np.ndarray, sep_col: int) -> list[dict]:
    shapes: list[dict] = []
    for color in RIGHT_COLORS:
        comps = []
        for comp in _find_color_components(grid[:, sep_col + 1 :], color):
            shifted = [(r, c + sep_col + 1) for r, c in comp["coords"]]
            comps.append(
                {
                    "color": color,
                    "coords": shifted,
                    "bbox": _bbox_from_coords(shifted),
                    "shape": _normalize(shifted),
                }
            )
        shapes.extend(comps)
    assert len(shapes) == 3, "expected three right-side colored shapes"
    assert len({item["shape"] for item in shapes}) == 3, "gap shapes must be unique"
    return shapes


def _teal_templates(grid: np.ndarray, sep_col: int) -> list[dict]:
    templates = [obj for obj in _find_color_components(grid[:, :sep_col], TEMPLATE_COLOR)]
    assert len(templates) == 3, "expected three left-side templates"
    templates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return templates


def _solve(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sep_cols = np.flatnonzero(np.all(g == SEPARATOR_COLOR, axis=0))
    assert len(sep_cols) == 1, "expected one full-height grey separator"
    sep_col = int(sep_cols[0])

    templates = _teal_templates(g, sep_col)
    shapes = _right_shapes(g, sep_col)

    pair_options: dict[tuple[int, int], list[dict]] = {}
    for ti, template in enumerate(templates):
        r0, c0, r1, c1 = template["bbox"]
        local_mask = g[r0 : r1 + 1, c0 : c1 + 1] == TEMPLATE_COLOR
        for si, shape in enumerate(shapes):
            pair_options[(ti, si)] = _enumerate_fill_candidates(local_mask, shape["shape"])

    solutions: list[dict[int, dict]] = []
    for perm in itertools.permutations(range(len(shapes))):
        choice_lists = [pair_options[(ti, si)] for ti, si in enumerate(perm)]
        if any(not choices for choices in choice_lists):
            continue
        for chosen in itertools.product(*choice_lists):
            hole_counts = [entry["holes"] for entry in chosen]
            if sorted(hole_counts) != [0, 1, 2]:
                continue
            assignment = {
                ti: {
                    "shape_idx": perm[ti],
                    "cells_local": chosen[ti]["cells_local"],
                    "holes": chosen[ti]["holes"],
                }
                for ti in range(len(templates))
            }
            solutions.append(assignment)

    assert len(solutions) == 1, "expected one valid shape-to-gap assignment"
    return {
        "sep_col": sep_col,
        "templates": templates,
        "shapes": shapes,
        "pair_options": pair_options,
        "assignment": solutions[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the right-side missing pieces into the left templates, then recolor each template by its post-fill hole count.

    Input structure:
    - Black background dominates the canvas, and one full-height grey column splits the grid into left and right halves.
    - The left half contains three separated teal 4-connected template bodies; each template has one boundary-adjacent missing piece shape and may also contain enclosed background holes.
    - The right half contains three small monochrome shapes in blue, red, and green, and each normalized shape matches exactly one left-side missing piece.

    Rule:
    - Normalize each right-side shape and each possible boundary missing-piece placement within every template bbox by translating them to their top-left corner.
    - Find the unique one-to-one assignment that fills one missing piece per template and yields three distinct post-fill hole counts 0, 1, and 2.
    - Draw each matched right-side shape into its left-side missing piece using the shape's own color, erase the original right-side shapes and the grey separator, then recolor only the original teal template pixels by hole count: 0->orange, 1->yellow, 2->maroon.

    Key insight:
    - The right-side shapes identify which boundary gaps must be filled, and only after that fill can the template's enclosed-background topology be counted correctly.
    """
    g = np.asarray(grid, dtype=int)
    solved = _solve(g)

    out = g.copy()
    out[:, solved["sep_col"]] = BG
    for shape in solved["shapes"]:
        for r, c in shape["coords"]:
            out[r, c] = BG

    for ti, template in enumerate(solved["templates"]):
        fill = solved["assignment"][ti]
        shape = solved["shapes"][fill["shape_idx"]]
        r0, c0, _r1, _c1 = template["bbox"]
        for dr, dc in fill["cells_local"]:
            out[r0 + dr, c0 + dc] = shape["color"]
        for r, c in template["coords"]:
            out[r, c] = RECOLOR_BY_HOLES[fill["holes"]]

    return out


def _shape_patch(shape_kind: str, color: int) -> np.ndarray:
    offsets = SHAPE_OFFSETS[shape_kind]
    h, w = _shape_dims(offsets)
    patch = np.zeros((h, w), dtype=int)
    for r, c in offsets:
        patch[r, c] = int(color)
    return patch


def _boxes_touch_or_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _sample_right_boxes(
    rng: random.Random, height: int, sep_col: int, width: int, kinds: list[str]
) -> dict[str, tuple[int, int]]:
    placements: dict[str, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []
    order = kinds[:]
    rng.shuffle(order)
    for kind in order:
        patch_h, patch_w = _shape_dims(SHAPE_OFFSETS[kind])
        ok = False
        for _ in range(400):
            r0 = rng.randint(0, height - patch_h)
            c0 = rng.randint(sep_col + 1, width - patch_w)
            box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if any(_boxes_touch_or_overlap(box, other, pad=1) for other in boxes):
                continue
            placements[kind] = (r0, c0)
            boxes.append(box)
            ok = True
            break
        if not ok:
            raise RuntimeError("failed to place right-side shapes")
    return placements


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    hole_counts = [0, 1, 2]
    shape_kinds = list(SHAPE_OFFSETS)
    rng.shuffle(shape_kinds)
    kind_by_holes = {holes: kind for holes, kind in zip(hole_counts, shape_kinds)}

    band_holes = hole_counts[:]
    rng.shuffle(band_holes)

    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    gap1 = rng.randint(2, 4)
    gap2 = rng.randint(2, 4)
    left_width = rng.randint(7, 9)
    right_width = rng.randint(5, 7)

    row_starts = [
        top_margin,
        top_margin + 5 + gap1,
        top_margin + 10 + gap1 + gap2,
    ]
    height = row_starts[-1] + 5 + bottom_margin
    width = left_width + 1 + right_width
    sep_col = left_width

    grid = np.zeros((height, width), dtype=int)
    grid[:, sep_col] = SEPARATOR_COLOR

    template_specs: list[dict] = []
    for row0, holes in zip(row_starts, band_holes):
        kind = kind_by_holes[holes]
        base_mask = TEMPLATE_LIBRARY[(holes, kind)]
        rotation, flip_lr = rng.choice(ALLOWED_TRANSFORMS[kind])
        mask = _transform_mask(base_mask, rotation, flip_lr)
        col0 = rng.randint(0, left_width - mask.shape[1])
        local_candidates = _enumerate_fill_candidates(mask, SHAPE_OFFSETS[kind])
        assert len(local_candidates) == 1
        assert local_candidates[0]["holes"] == holes
        for rr, cc in zip(*np.where(mask)):
            grid[row0 + int(rr), col0 + int(cc)] = TEMPLATE_COLOR
        template_specs.append(
            {
                "holes": holes,
                "shape_kind": kind,
                "row0": row0,
                "col0": col0,
                "mask": mask,
                "gap_cells": local_candidates[0]["cells_local"],
            }
        )

    colors = list(RIGHT_COLORS)
    rng.shuffle(colors)
    color_by_kind = {kind: color for kind, color in zip(shape_kinds, colors)}
    right_positions = _sample_right_boxes(rng, height, sep_col, width, shape_kinds)
    for kind in shape_kinds:
        patch = _shape_patch(kind, color_by_kind[kind])
        r0, c0 = right_positions[kind]
        for rr, cc in zip(*np.where(patch)):
            grid[r0 + int(rr), c0 + int(cc)] = int(patch[rr, cc])

    expected = grid.copy()
    expected[:, sep_col] = BG
    for kind in shape_kinds:
        patch = _shape_patch(kind, color_by_kind[kind])
        r0, c0 = right_positions[kind]
        for rr, cc in zip(*np.where(patch)):
            expected[r0 + int(rr), c0 + int(cc)] = BG
    for spec in template_specs:
        for dr, dc in spec["gap_cells"]:
            expected[spec["row0"] + dr, spec["col0"] + dc] = color_by_kind[spec["shape_kind"]]
        for rr, cc in zip(*np.where(spec["mask"])):
            expected[spec["row0"] + int(rr), spec["col0"] + int(cc)] = RECOLOR_BY_HOLES[
                spec["holes"]
            ]

    return grid, expected


def _semantic_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        solved = _solve(inp)
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if sorted(solved["assignment"][ti]["holes"] for ti in range(3)) != [0, 1, 2]:
        return False
    for ti in range(3):
        matched = sum(1 for si in range(3) if solved["pair_options"][(ti, si)])
        if matched != 1:
            return False
    for si in range(3):
        matched = sum(1 for ti in range(3) if solved["pair_options"][(ti, si)])
        if matched != 1:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        inp, expected = _build_case(rng)
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if _semantic_ok(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")

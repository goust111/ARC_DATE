from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = Color.BLACK
TEMPLATE_COLOR = Color.GREY
GLYPH_COLOR = Color.PINK
TEMPLATE_STRIDE = 4
TEMPLATE_BBOX_SIZE = 3
GLYPH_MIN_SEP = 3
MAX_GENERATION_ATTEMPTS = 400

_TEMPLATE_MASKS = (
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
)

_CODE_TO_PATCH = {
    0: np.array([[GLYPH_COLOR, 0], [0, GLYPH_COLOR]], dtype=int),
    1: np.array([[GLYPH_COLOR], [GLYPH_COLOR]], dtype=int),
    2: np.array([[GLYPH_COLOR, GLYPH_COLOR]], dtype=int),
}
_CODE_TO_COLOR = {
    0: Color.RED,
    1: Color.GREEN,
    2: Color.BLUE,
}
_PATCH_SIGNATURE_TO_COLOR = {
    ((1, 0), (0, 1)): Color.RED,
    ((1,), (1,)): Color.GREEN,
    ((1, 1),): Color.BLUE,
}


def _center(obj: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _component_signature(grid: np.ndarray, obj: dict, color: int) -> tuple[tuple[int, ...], ...]:
    patch = crop_bbox(grid, tuple(map(int, obj["bbox"])))
    mask = (patch == int(color)).astype(int)
    return tuple(tuple(int(v) for v in row) for row in mask)


def _row_major_order(objects: list[dict]) -> list[dict]:
    assert len(objects) == 9, "expected 9 objects"
    by_y = sorted(objects, key=lambda obj: (_center(obj)[0], _center(obj)[1]))
    rows = [by_y[0:3], by_y[3:6], by_y[6:9]]
    for upper, lower in zip(rows, rows[1:]):
        assert max(_center(obj)[0] for obj in upper) < min(_center(obj)[0] for obj in lower)

    ordered: list[dict] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda obj: (_center(obj)[1], _center(obj)[0])))
    return ordered


def _decode_glyph(grid: np.ndarray, obj: dict) -> int:
    signature = _component_signature(grid, obj, GLYPH_COLOR)
    assert signature in _PATCH_SIGNATURE_TO_COLOR, "unknown glyph shape"
    return int(_PATCH_SIGNATURE_TO_COLOR[signature])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the scattered magenta glyphs into colors and use their row-major positions to recolor the matching template masks.

    Input structure:
    - The background is black.
    - Nine gray 4-connected template objects occupy a 3x3 grid near the upper-left; each template fits in a 3x3 bbox and all nine share the same asymmetric mask.
    - Nine magenta glyphs lie in blank space to the right and/or below the templates; each glyph is a two-pixel pattern that is diagonal, vertical, or horizontal.

    Rule:
    - Decode each magenta glyph by shape: diagonal -> red, vertical -> green, horizontal -> blue.
    - Sort the nine glyphs into row-major order by their bbox centers: first split the y-sorted glyph centers into three groups of three, then sort each group by x.
    - Apply that same row-major indexing to the nine gray templates and recolor each template's gray pixels with the decoded color from the glyph at the same index.
    - Erase all magenta glyph pixels to black and leave template holes/background unchanged.

    Key insight:
    - The glyphs do not get copied directly; their shapes encode colors, and their spatial arrangement supplies the indices telling which repeated template instance receives each decoded color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    templates = find_components(
        g, bg=BG, connectivity=4, monochromatic=True, colors=[TEMPLATE_COLOR]
    )
    glyphs = find_components(
        g, bg=BG, connectivity=8, monochromatic=True, colors=[GLYPH_COLOR]
    )
    assert len(templates) == 9, "expected 9 gray templates"
    assert len(glyphs) == 9, "expected 9 magenta glyphs"

    template_signatures = {_component_signature(g, obj, TEMPLATE_COLOR) for obj in templates}
    assert len(template_signatures) == 1, "all templates must share one mask"
    template_signature = next(iter(template_signatures))
    assert len(template_signature) == TEMPLATE_BBOX_SIZE
    assert len(template_signature[0]) == TEMPLATE_BBOX_SIZE

    ordered_templates = _row_major_order(templates)
    ordered_glyphs = _row_major_order(glyphs)

    out = g.copy()
    out[out == GLYPH_COLOR] = BG
    for template_obj, glyph_obj in zip(ordered_templates, ordered_glyphs):
        color = _decode_glyph(g, glyph_obj)
        coords = template_obj["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _sample_control_codes() -> np.ndarray:
    while True:
        codes = np.random.randint(0, 3, size=(3, 3))
        if set(map(int, codes.ravel())) != {0, 1, 2}:
            continue
        row_tuples = {tuple(map(int, row)) for row in codes}
        col_tuples = {tuple(map(int, col)) for col in codes.T}
        if len(row_tuples) < 2 or len(col_tuples) < 2:
            continue
        if not any(len(set(map(int, row))) == 3 for row in codes):
            continue
        if not any(len(set(map(int, col))) == 3 for col in codes.T):
            continue
        row_major = tuple(map(int, codes.ravel()))
        col_major = tuple(int(codes[r, c]) for c in range(3) for r in range(3))
        if row_major == col_major:
            continue
        return codes


def _triplets(lo: int, hi: int, min_sep: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for a in range(lo, hi + 1):
        for b in range(a + min_sep, hi + 1):
            for c in range(b + min_sep, hi + 1):
                out.append((a, b, c))
    return out


def _build_expected(
    grid: np.ndarray,
    template_mask: np.ndarray,
    template_r0: int,
    template_c0: int,
    codes: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    out[out == GLYPH_COLOR] = BG
    mask_coords = np.argwhere(template_mask)
    for rr in range(3):
        for cc in range(3):
            color = int(_CODE_TO_COLOR[int(codes[rr, cc])])
            top = template_r0 + rr * TEMPLATE_STRIDE
            left = template_c0 + cc * TEMPLATE_STRIDE
            coords = mask_coords + np.array([top, left])
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _gaps_are_visual(ordered: list[dict], axis: int) -> bool:
    coords = [float(_center(obj)[axis]) for obj in ordered]
    gaps = [coords[i + 1] - coords[i] for i in range(8)]
    inner = max(gaps[0:2] + gaps[3:5] + gaps[6:8], default=0.0)
    return gaps[2] > inner and gaps[5] > inner


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, codes: np.ndarray) -> bool:
    templates = find_components(
        grid, bg=BG, connectivity=4, monochromatic=True, colors=[TEMPLATE_COLOR]
    )
    glyphs = find_components(
        grid, bg=BG, connectivity=8, monochromatic=True, colors=[GLYPH_COLOR]
    )
    if len(templates) != 9 or len(glyphs) != 9:
        return False

    template_signatures = {_component_signature(grid, obj, TEMPLATE_COLOR) for obj in templates}
    if len(template_signatures) != 1:
        return False

    glyph_by_y = sorted(glyphs, key=lambda obj: (_center(obj)[0], _center(obj)[1]))
    if not _gaps_are_visual(glyph_by_y, axis=0):
        return False

    ordered_glyphs = _row_major_order(glyphs)
    ordered_templates = _row_major_order(templates)
    correct_colors = tuple(_decode_glyph(grid, obj) for obj in ordered_glyphs)
    expected_colors = tuple(int(_CODE_TO_COLOR[int(code)]) for code in codes.ravel())
    if correct_colors != expected_colors:
        return False

    x_major_colors = tuple(
        _decode_glyph(grid, obj) for obj in sorted(glyphs, key=lambda obj: (_center(obj)[1], _center(obj)[0]))
    )
    if x_major_colors == correct_colors:
        return False

    wrong = grid.copy()
    wrong[wrong == GLYPH_COLOR] = BG
    for template_obj, color in zip(ordered_templates, x_major_colors):
        coords = template_obj["coords"]
        wrong[coords[:, 0], coords[:, 1]] = int(color)
    if np.array_equal(wrong, expected):
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _sample_geometry(layout: str) -> tuple[int, int, int, int, tuple[int, int, int], tuple[int, int, int]] | None:
    if layout == "right":
        height = int(np.random.randint(15, 29))
        width = int(np.random.randint(23, 31))
    elif layout == "bottom":
        height = int(np.random.randint(23, 31))
        width = int(np.random.randint(15, 29))
    else:
        height = int(np.random.randint(23, 31))
        width = int(np.random.randint(23, 31))

    template_r0 = int(np.random.randint(0, 3))
    template_c0 = int(np.random.randint(0, 3))
    template_bottom = template_r0 + 3 * TEMPLATE_BBOX_SIZE + 2
    template_right = template_c0 + 3 * TEMPLATE_BBOX_SIZE + 2

    row_lo = 0 if layout == "right" else template_bottom + 2
    col_lo = 0 if layout == "bottom" else template_right + 2
    row_hi = height - 2
    col_hi = width - 2

    row_triplets = _triplets(row_lo, row_hi, GLYPH_MIN_SEP)
    col_triplets = _triplets(col_lo, col_hi, GLYPH_MIN_SEP)
    if not row_triplets or not col_triplets:
        return None

    return (
        height,
        width,
        template_r0,
        template_c0,
        row_triplets[int(np.random.randint(len(row_triplets)))],
        col_triplets[int(np.random.randint(len(col_triplets)))],
    )


def _gen_attempt() -> np.ndarray | None:
    layout = str(np.random.choice(["right", "bottom", "corner"]))
    geometry = _sample_geometry(layout)
    if geometry is None:
        return None

    height, width, template_r0, template_c0, row_tops, col_lefts = geometry
    template_mask = _TEMPLATE_MASKS[int(np.random.randint(len(_TEMPLATE_MASKS)))]
    codes = _sample_control_codes()

    grid = np.full((height, width), BG, dtype=int)
    template_patch = template_mask * TEMPLATE_COLOR
    for rr in range(3):
        for cc in range(3):
            paste(
                grid,
                template_patch,
                template_r0 + rr * TEMPLATE_STRIDE,
                template_c0 + cc * TEMPLATE_STRIDE,
                transparent=BG,
            )

    for rr, row_top in enumerate(row_tops):
        for cc, col_left in enumerate(col_lefts):
            patch = _CODE_TO_PATCH[int(codes[rr, cc])]
            paste(grid, patch, int(row_top), int(col_left), transparent=BG)

    expected = _build_expected(grid, template_mask, template_r0, template_c0, codes)
    if not _semantic_ok(grid, expected, codes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

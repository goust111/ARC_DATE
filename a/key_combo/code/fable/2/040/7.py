from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components


BLACK = Color.BLACK
RED = Color.RED
GREY = Color.GREY

OP_CCW = "rotate_ccw"
OP_CW = "rotate_cw"
OP_MIRROR_LR = "mirror_lr"
OP_FLIP_UD = "flip_ud"
OPS = (OP_CCW, OP_CW, OP_MIRROR_LR, OP_FLIP_UD)

MISSING_TO_OP = {
    (0, 0): OP_CCW,
    (0, 1): OP_CW,
    (1, 0): OP_MIRROR_LR,
    (1, 1): OP_FLIP_UD,
}
OP_TO_MISSING = {op: corner for corner, op in MISSING_TO_OP.items()}


def _same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _decode_glyphs(g: np.ndarray) -> list[str]:
    glyph_zone = g[:2, :]
    assert set(map(int, np.unique(glyph_zone))) <= {BLACK, RED}, "glyph rows must contain only black and red"

    components = find_components(
        glyph_zone,
        bg=BLACK,
        colors=[RED],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected at least one red glyph"
    components = sorted(components, key=lambda comp: int(comp["bbox"][1]))

    ops: list[str] = []
    previous_c1 = -2
    for comp in components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert (r0, r1) == (0, 1), "each glyph must occupy both glyph rows"
        assert c1 - c0 + 1 == 2, "each glyph must occupy a 2x2 frame"
        assert int(comp["area"]) == 3, "each glyph must be a 2x2 square missing one corner"
        if previous_c1 >= 0:
            assert c0 >= previous_c1 + 2, "glyphs must be separated by at least one black column"
            assert np.all(glyph_zone[:, previous_c1 + 1 : c0] == BLACK), "separator columns must be black"

        patch = crop_bbox(glyph_zone, comp["bbox"])
        missing = np.argwhere(patch == BLACK)
        assert missing.shape[0] == 1, "each glyph must have exactly one missing corner"
        corner = tuple(map(int, missing[0]))
        assert corner in MISSING_TO_OP, "missing cell must be one of the four corners"
        ops.append(MISSING_TO_OP[corner])
        previous_c1 = c1

    assert np.all(glyph_zone[:, : components[0]["bbox"][1]] == BLACK), "left glyph margin must be black"
    assert np.all(glyph_zone[:, previous_c1 + 1 :] == BLACK), "right glyph margin must be black"
    return ops


def _sprite_patch(g: np.ndarray) -> np.ndarray:
    body = g[3:, :]
    assert body.size > 0, "expected a sprite area below the separator row"
    assert set(map(int, np.unique(body))) != {BLACK}, "expected a non-empty sprite"

    components = find_components(
        body,
        bg=BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 1, "expected exactly one 4-connected multicolor sprite"
    return crop_bbox(body, components[0]["bbox"])


def _apply_op(sprite: np.ndarray, op: str) -> np.ndarray:
    if op == OP_CCW:
        return np.rot90(sprite, 1).copy()
    if op == OP_CW:
        return np.rot90(sprite, -1).copy()
    if op == OP_MIRROR_LR:
        return np.fliplr(sprite).copy()
    if op == OP_FLIP_UD:
        return np.flipud(sprite).copy()
    raise AssertionError(f"unknown operation {op}")


def _apply_program(sprite: np.ndarray, ops: list[str] | tuple[str, ...]) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    for op in ops:
        out = _apply_op(out, op)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the top red missing-corner glyph program and apply it to the lower sprite.

    Input structure:
    - The top two rows contain red 2x2 glyphs on black; each glyph is missing exactly one corner and glyphs are separated by black columns.
    - Row 3 is a full-width grey separator row.
    - Below the separator is a black field containing exactly one 4-connected multicolor sprite.

    Rule:
    - Read glyphs left to right: missing upper-left means rotate the sprite 90 degrees counterclockwise, missing upper-right means rotate 90 degrees clockwise, missing lower-left means mirror left-right, and missing lower-right means flip top-bottom.
    - Apply the decoded operations in that order to the sprite cropped from its minimal bounding box.
    - Return only the final sprite patch, without the glyph rows, separator row, or surrounding black background.

    Key insight:
    - The red shapes are operation codes, not output material; the missing corner of each glyph encodes one geometric transform in a sequential program.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 4 and g.shape[1] >= 2, "expected a 2D ARC grid with a sprite area"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"
    assert np.all(g[2, :] == GREY), "the third row must be a full-width grey separator"

    ops = _decode_glyphs(g)
    sprite = _sprite_patch(g)
    out = _apply_program(sprite, ops)
    assert max(out.shape) <= 30, "output must fit ARC dimensions"
    return out


# --------------- generator ---------------

SPRITE_PALETTE = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
SPRITE_SIZE_RANGE = (4, 8)
SPRITE_DENSITY_RANGE = (0.35, 0.72)
GLYPH_COUNT_RANGE = (4, 8)
BODY_MARGIN_RANGE = (0, 4)
GENERATION_ATTEMPTS = 800


def _carve_path(mask: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    r, c = start
    tr, tc = target
    mask[r, c] = True
    while (r, c) != (tr, tc):
        choices: list[tuple[int, int]] = []
        if r < tr:
            choices.append((1, 0))
        elif r > tr:
            choices.append((-1, 0))
        if c < tc:
            choices.append((0, 1))
        elif c > tc:
            choices.append((0, -1))
        dr, dc = random.choice(choices)
        r += dr
        c += dc
        mask[r, c] = True
    return int(r), int(c)


def _grow_connected_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    cur = (random.randrange(h), random.randrange(w))
    targets = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    random.shuffle(targets)
    for target in targets:
        cur = _carve_path(mask, cur, target)

    area_now = int(mask.sum())
    max_area = h * w
    if area_now >= max_area - 1:
        return mask

    low = max(area_now + 1, int(round(SPRITE_DENSITY_RANGE[0] * max_area)))
    high = max(low, int(round(SPRITE_DENSITY_RANGE[1] * max_area)))
    high = min(high, max_area)
    target_area = random.randint(low, high)

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(rr) + dr
                nc = int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _dihedral_forms(sprite: np.ndarray) -> list[np.ndarray]:
    forms: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(sprite, k)
        forms.append(rotated.copy())
        forms.append(np.fliplr(rotated).copy())
    return forms


def _unique_dihedral(sprite: np.ndarray) -> bool:
    signatures = {(form.shape, form.tobytes()) for form in _dihedral_forms(sprite)}
    return len(signatures) == 8


def _paint_sprite(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h, w), dtype=int)
    color_count = random.randint(2, min(4, len(SPRITE_PALETTE)))
    colors = random.sample(list(SPRITE_PALETTE), color_count)
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    random.shuffle(coords)

    for i, (r, c) in enumerate(coords):
        if i < color_count:
            patch[r, c] = colors[i]
        else:
            patch[r, c] = random.choice(colors)
    return patch


def _random_sprite() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(*SPRITE_SIZE_RANGE)
        w = random.randint(*SPRITE_SIZE_RANGE)
        mask = _grow_connected_mask(h, w)
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        patch = _paint_sprite(mask)
        components = find_components(
            patch,
            bg=BLACK,
            connectivity=4,
            monochromatic=False,
            return_format="dict",
        )
        if len(components) != 1:
            continue
        if not _unique_dihedral(patch):
            continue
        return patch
    return None


def _program_is_non_degenerate(sprite: np.ndarray, ops: list[str]) -> bool:
    expected = _apply_program(sprite, ops)
    if _same_grid(expected, sprite):
        return False
    if any(_same_grid(expected, _apply_program(sprite, [op])) for op in OPS):
        return False
    if _same_grid(expected, _apply_program(sprite, list(reversed(ops)))):
        return False
    for i in range(len(ops)):
        shortened = ops[:i] + ops[i + 1 :]
        if _same_grid(expected, _apply_program(sprite, shortened)):
            return False
    return True


def _random_program(sprite: np.ndarray) -> list[str] | None:
    for _ in range(100):
        ops = list(OPS)
        extras = [random.choice(OPS) for _ in range(random.randint(0, GLYPH_COUNT_RANGE[1] - len(OPS)))]
        ops.extend(extras)
        random.shuffle(ops)
        if not (GLYPH_COUNT_RANGE[0] <= len(ops) <= GLYPH_COUNT_RANGE[1]):
            continue
        if _program_is_non_degenerate(sprite, ops):
            return ops
    return None


def _render_glyph_rows(ops: list[str], width: int, start_c: int) -> np.ndarray:
    rows = np.zeros((2, width), dtype=int)
    for i, op in enumerate(ops):
        c = start_c + 3 * i
        glyph = np.full((2, 2), RED, dtype=int)
        miss_r, miss_c = OP_TO_MISSING[op]
        glyph[miss_r, miss_c] = BLACK
        rows[:, c : c + 2] = glyph
    return rows


def _render_candidate(sprite: np.ndarray, ops: list[str]) -> np.ndarray | None:
    glyph_w = 3 * len(ops) - 1
    top_margin = random.randint(*BODY_MARGIN_RANGE)
    bottom_margin = random.randint(*BODY_MARGIN_RANGE)
    left_margin = random.randint(*BODY_MARGIN_RANGE)
    right_margin = random.randint(*BODY_MARGIN_RANGE)

    body_h = sprite.shape[0] + top_margin + bottom_margin
    min_w = max(glyph_w, sprite.shape[1] + left_margin + right_margin)
    if body_h + 3 > 30 or min_w > 30:
        return None

    width = random.randint(min_w, min(30, min_w + 5))
    glyph_start = random.randint(0, width - glyph_w)
    sprite_r = top_margin
    sprite_c = random.randint(left_margin, width - sprite.shape[1] - right_margin)

    grid = np.zeros((body_h + 3, width), dtype=int)
    grid[:2, :] = _render_glyph_rows(ops, width, glyph_start)
    grid[2, :] = GREY
    grid[3 + sprite_r : 3 + sprite_r + sprite.shape[0], sprite_c : sprite_c + sprite.shape[1]] = sprite
    return grid


def _gen_attempt() -> np.ndarray | None:
    sprite = _random_sprite()
    if sprite is None:
        return None
    ops = _random_program(sprite)
    if ops is None:
        return None

    grid = _render_candidate(sprite, ops)
    if grid is None:
        return None

    expected = _apply_program(sprite, ops)
    try:
        if _decode_glyphs(grid) != ops:
            return None
        if not _same_grid(_sprite_patch(grid), sprite):
            return None
        actual = main(grid)
    except AssertionError:
        return None
    if not _same_grid(actual, expected):
        return None
    if not _program_is_non_degenerate(sprite, ops):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid glyph-program input")

import itertools
import random

import numpy as np


BG = 0
GLYPH = 5
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 8):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _flip_lr(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(arr)


def _flip_ud(arr: np.ndarray) -> np.ndarray:
    return np.flipud(arr)


def _transpose_main(arr: np.ndarray) -> np.ndarray:
    return arr.T.copy()


def _transpose_anti(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(arr)).T.copy()


def _rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2).copy()


OPS = {
    "lr": _flip_lr,
    "ud": _flip_ud,
    "main": _transpose_main,
    "anti": _transpose_anti,
    "rot180": _rot180,
}

CANONICAL = {
    "id": lambda a: a.copy(),
    "lr": _flip_lr,
    "ud": _flip_ud,
    "main": _transpose_main,
    "anti": _transpose_anti,
    "rot180": _rot180,
    "rot90": lambda a: np.rot90(a, -1).copy(),
    "rot270": lambda a: np.rot90(a, 1).copy(),
}


def _apply_sequence(arr: np.ndarray, seq) -> np.ndarray:
    out = np.array(arr, dtype=int, copy=True)
    for name in seq:
        out = OPS[name](out)
    return out


_REF_FORM = np.arange(1, 7, dtype=int).reshape(2, 3)
_CANONICAL_FORMS = {name: fn(_REF_FORM) for name, fn in CANONICAL.items()}


def _canonical_name(seq) -> str:
    out = _apply_sequence(_REF_FORM, seq)
    for name, target in _CANONICAL_FORMS.items():
        if np.array_equal(out, target):
            return name
    raise AssertionError("unexpected transform composition")


def _build_sequence_pools():
    pools = {1: [], 2: [], 3: []}
    for name in OPS:
        pools[1].append((name,))

    for length in (2, 3):
        for seq in itertools.permutations(tuple(OPS), length):
            if _canonical_name(seq) not in {"rot90", "rot270"}:
                continue
            target = _apply_sequence(_REF_FORM, seq)
            if any(np.array_equal(_apply_sequence(_REF_FORM, seq[:i] + seq[i + 1 :]), target) for i in range(length)):
                continue
            if np.array_equal(_apply_sequence(_REF_FORM, seq[::-1]), target):
                continue
            pools[length].append(seq)
    return pools


SEQUENCE_POOLS = _build_sequence_pools()


def _build_glyph_bank():
    bank = {name: [] for name in OPS}
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
        if any(mask[r].sum() == 0 for r in range(3)):
            continue
        if any(mask[:, c].sum() == 0 for c in range(3)):
            continue
        if not (3 <= int(mask.sum()) <= 7):
            continue
        matches = [name for name, fn in OPS.items() if np.array_equal(mask, fn(mask))]
        if len(matches) != 1:
            continue
        bank[matches[0]].append(np.where(mask == 1, GLYPH, BG))
    for name in bank:
        bank[name].sort(key=lambda arr: tuple(int(x) for x in arr.flatten()))
    return bank


GLYPH_BANK = _build_glyph_bank()


def _split_intervals(indices):
    if len(indices) == 0:
        return []
    groups = [[int(indices[0])]]
    for idx in indices[1:]:
        idx = int(idx)
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return [(group[0], group[-1]) for group in groups]


def _decode_glyph(glyph: np.ndarray) -> str:
    mask = (glyph == GLYPH).astype(int)
    matches = [name for name, fn in OPS.items() if np.array_equal(mask, fn(mask))]
    assert len(matches) == 1, "each glyph must encode exactly one symmetry"
    return matches[0]


def _parse_input(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    glyph_rows = np.where((g == GLYPH).any(axis=1))[0]
    assert len(glyph_rows) == 3 and int(glyph_rows[-1] - glyph_rows[0]) == 2, "glyph band must be 3 rows tall"
    r0 = int(glyph_rows[0])
    glyph_band = g[r0 : r0 + 3, :]

    glyph_cols = np.where((glyph_band == GLYPH).any(axis=0))[0]
    spans = _split_intervals(glyph_cols)
    assert 1 <= len(spans) <= 3, "expected one to three glyphs"
    assert all(lo >= 0 and hi - lo + 1 == 3 for lo, hi in spans), "each glyph must occupy a 3x3 window"

    glyphs = []
    for lo, hi in spans:
        patch = glyph_band[:, lo : hi + 1].copy()
        assert patch.shape == (3, 3)
        assert set(np.unique(patch)).issubset({BG, GLYPH})
        glyphs.append(patch)

    sprite_mask = (g != BG) & (g != GLYPH)
    assert sprite_mask.any(), "expected one multicolor sprite"
    comps = _components(sprite_mask, connectivity=8)
    assert len(comps) == 1, "expected a single sprite component"
    sprite_box = bbox(sprite_mask)
    assert sprite_box is not None
    sprite = crop_bbox(g, sprite_box)
    assert sprite_box[0] > r0 + 2, "sprite must lie below the glyph band"
    return glyphs, sprite


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode each gray glyph's unique self-symmetry into a transform and apply them left-to-right to the sprite.

    Input structure:
    - Black background.
    - The upper area contains one to three 3x3 gray/black glyph windows separated by black columns.
    - The lower area contains one multicolor sprite with non-gray colors.

    Rule:
    - Read the glyph windows from left to right.
    - For each glyph, detect which single non-trivial symmetry leaves that glyph unchanged:
      left-right mirror, up-down mirror, main-diagonal transpose, anti-diagonal transpose, or 180-degree rotation.
    - Apply those decoded transforms to the sprite in the same left-to-right order.
    - Return the transformed sprite cropped to its foreground bounding box.

    Key insight:
    - The glyphs are self-describing codewords: the symmetry they themselves preserve is exactly the operation to perform.
    """
    glyphs, sprite = _parse_input(np.asarray(input_grid, dtype=int))
    out = np.array(sprite, dtype=int, copy=True)
    for glyph in glyphs:
        out = OPS[_decode_glyph(glyph)](out)
    box = bbox(out != BG)
    assert box is not None
    return crop_bbox(out, box)


def _random_connected_mask(h: int, w: int):
    target = random.randint(max(5, (h * w + 1) // 2), h * w)
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if any(mask[r].sum() == 0 for r in range(h)):
        return None
    if any(mask[:, c].sum() == 0 for c in range(w)):
        return None
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    coords = [tuple(rc) for rc in np.argwhere(mask)]
    n_colors = random.randint(2, min(5, len(coords)))
    palette = random.sample(SPRITE_COLORS, n_colors)
    random.shuffle(coords)
    for color, (r, c) in zip(palette, coords):
        sprite[r, c] = color
    for r, c in coords[n_colors:]:
        sprite[r, c] = random.choice(palette)
    return sprite


def _stepwise_changes(sprite: np.ndarray, seq) -> bool:
    cur = np.array(sprite, dtype=int, copy=True)
    for name in seq:
        nxt = OPS[name](cur)
        if np.array_equal(nxt, cur):
            return False
        cur = nxt
    return True


def _sequence_is_informative(sprite: np.ndarray, seq) -> bool:
    final = _apply_sequence(sprite, seq)
    if np.array_equal(final, sprite):
        return False
    if not _stepwise_changes(sprite, seq):
        return False
    for i in range(len(seq)):
        if np.array_equal(final, _apply_sequence(sprite, seq[:i] + seq[i + 1 :])):
            return False
    for i, name in enumerate(seq):
        for alt in OPS:
            if alt == name:
                continue
            alt_seq = list(seq)
            alt_seq[i] = alt
            if np.array_equal(final, _apply_sequence(sprite, tuple(alt_seq))):
                return False
    if len(seq) >= 2 and np.array_equal(final, _apply_sequence(sprite, seq[::-1])):
        return False
    return True


def _sample_sprite(seq):
    for _ in range(400):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        if len(seq) >= 2 and random.random() < 0.7 and h == w:
            w = 3 + ((w + random.randint(1, 2)) % 4)
        mask = _random_connected_mask(h, w)
        if mask is None:
            continue
        sprite = _colorize_mask(mask)
        if _sequence_is_informative(sprite, seq):
            return sprite
    raise RuntimeError("failed to sample a non-degenerate sprite")


def _assemble_input(glyphs, sprite):
    count = len(glyphs)
    top_pad = random.randint(0, 2)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    bottom_pad = random.randint(0, 2)
    gap = random.randint(2, 5)
    seps = [random.randint(1, 2) for _ in range(max(0, count - 1))]
    glyph_width = 3 * count + sum(seps)

    sprite_left = random.randint(1, 4)
    sprite_right = random.randint(1, 4)
    width = max(left_pad + glyph_width + right_pad, sprite_left + sprite.shape[1] + sprite_right)
    height = top_pad + 3 + gap + sprite.shape[0] + bottom_pad
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)

    c = left_pad
    for i, glyph in enumerate(glyphs):
        grid[top_pad : top_pad + 3, c : c + 3] = glyph
        if i < count - 1:
            c += 3 + seps[i]

    sprite_top = top_pad + 3 + gap
    sprite_col = random.randint(1, width - sprite.shape[1] - 1)
    sprite_mask = sprite != BG
    block = grid[sprite_top : sprite_top + sprite.shape[0], sprite_col : sprite_col + sprite.shape[1]]
    block[sprite_mask] = sprite[sprite_mask]
    return grid


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    rows = np.where((g == GLYPH).any(axis=1))[0]
    band = g[int(rows[0]) : int(rows[0]) + 3]
    cols = np.where((band == GLYPH).any(axis=0))[0]
    spans = _split_intervals(cols)
    seq = []
    for lo, hi in spans:
        glyph = band[:, lo : hi + 1]
        mask = (glyph == GLYPH).astype(int)
        matches = [name for name, fn in OPS.items() if np.array_equal(mask, fn(mask))]
        if len(matches) != 1:
            raise AssertionError("glyph lost unique symmetry")
        seq.append(matches[0])

    sprite_mask = (g != BG) & (g != GLYPH)
    sprite = crop_bbox(g, bbox(sprite_mask))
    out = np.array(sprite, dtype=int, copy=True)
    for name in seq:
        out = OPS[name](out)
    return crop_bbox(out, bbox(out != BG))


def _validate_instance(grid: np.ndarray) -> bool:
    glyphs, sprite = _parse_input(grid)
    seq = tuple(_decode_glyph(glyph) for glyph in glyphs)
    if len(seq) not in (1, 2, 3):
        return False
    if len(seq) >= 2 and _canonical_name(seq) not in {"rot90", "rot270"}:
        return False
    if not _sequence_is_informative(sprite, seq):
        return False
    expected = _independent_expected(grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False
    return actual.ndim == 2 and max(grid.shape) <= 30 and max(actual.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(600):
        count = random.choices([1, 2, 3], weights=[3, 4, 3])[0]
        seq = random.choice(SEQUENCE_POOLS[count])
        glyphs = [random.choice(GLYPH_BANK[name]).copy() for name in seq]
        sprite = _sample_sprite(seq)
        grid = _assemble_input(glyphs, sprite)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid self-decoding glyph puzzle")

from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 2
SEPARATOR = 5
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
PANEL_SIDE_CHOICES = (7, 9, 11, 13)
PATTERN_SIDE_CHOICES = (3, 5, 7, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400


def _find_cross_lines(g: np.ndarray) -> tuple[int, int]:
    sep_rows = np.flatnonzero(np.all(g == SEPARATOR, axis=1))
    sep_cols = np.flatnonzero(np.all(g == SEPARATOR, axis=0))
    assert len(sep_rows) == 1 and len(sep_cols) == 1
    sep_r = int(sep_rows[0])
    sep_c = int(sep_cols[0])
    assert 0 < sep_r < g.shape[0] - 1 and 0 < sep_c < g.shape[1] - 1
    assert sep_r == g.shape[0] - sep_r - 1
    assert sep_c == g.shape[1] - sep_c - 1
    return sep_r, sep_c


def _quadrant_slices(g: np.ndarray) -> list[tuple[slice, slice]]:
    sep_r, sep_c = _find_cross_lines(g)
    return [
        (slice(0, sep_r), slice(0, sep_c)),
        (slice(0, sep_r), slice(sep_c + 1, g.shape[1])),
        (slice(sep_r + 1, g.shape[0]), slice(0, sep_c)),
        (slice(sep_r + 1, g.shape[0]), slice(sep_c + 1, g.shape[1])),
    ]


def _crop_foreground(panel: np.ndarray) -> np.ndarray:
    coords = np.argwhere(panel != BG)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return panel[r0 : r1 + 1, c0 : c1 + 1].copy()


def _marker_panel(panel: np.ndarray) -> bool:
    h, w = panel.shape
    center = (h // 2, w // 2)
    coords = np.argwhere(panel != BG)
    return len(coords) == 1 and tuple(map(int, coords[0])) == center and int(panel[center]) == MARKER


def _rotation_offset(reference: np.ndarray, target: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(reference, k), target)]
    assert len(matches) == 1
    return matches[0]


def _paste_centered(panel: np.ndarray, sprite: np.ndarray) -> None:
    ph, pw = panel.shape
    sh, sw = sprite.shape
    assert sh <= ph and sw <= pw
    r0 = (ph - sh) // 2
    c0 = (pw - sw) // 2
    panel[r0 : r0 + sh, c0 : c0 + sw] = sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the marked quadrant with the missing rotation of the shared asymmetric motif.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column form a 2x2 partition of equal rectangular quadrants.
    - Three quadrants each contain the same non-symmetric colored motif under three different 90-degree rotations, centered within their quadrant.
    - The remaining quadrant contains only a single red marker at its center.

    Rule:
    - Detect the grey cross and split the grid into its four quadrants.
    - Crop the three existing motifs to their non-black bounding boxes and compare them under 90-degree rotations.
    - Infer which of the four rotations is absent, rotate the shared motif to that missing orientation, and place it centered in the marked quadrant.
    - Remove the red marker while preserving the grey separator lines and all existing motifs.

    Key insight:
    - The absolute orientation does not matter: once one observed motif is treated as a reference, the other two reveal which relative quarter-turn is still missing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    quadrants = _quadrant_slices(g)
    marker_index = None
    observed: list[np.ndarray] = []
    for idx, (rs, cs) in enumerate(quadrants):
        panel = g[rs, cs]
        if _marker_panel(panel):
            assert marker_index is None
            marker_index = idx
        else:
            observed.append(_crop_foreground(panel))

    assert marker_index is not None
    assert len(observed) == 3

    reference = observed[0]
    seen_offsets = {0}
    for sprite in observed[1:]:
        seen_offsets.add(_rotation_offset(reference, sprite))
    assert len(seen_offsets) == 3

    missing_offset = ({0, 1, 2, 3} - seen_offsets).pop()
    missing_sprite = np.rot90(reference, missing_offset)

    out = g.copy()
    rs, cs = quadrants[marker_index]
    out[rs, cs] = BG
    _paste_centered(out[rs, cs], missing_sprite)
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _connected_mask(h: int, w: int, target: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = []
        seen = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c, h, w):
                if (rr, cc) not in cells and (rr, cc) not in seen:
                    seen.add((rr, cc))
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _color_sprite(mask: np.ndarray) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(1, min(4, len(coords)))
    palette = random.sample(PATTERN_COLORS, color_count)
    key = random.choice(
        (
            lambda rc: (rc[0], rc[1]),
            lambda rc: (rc[1], rc[0]),
            lambda rc: (rc[0] + rc[1], rc[0], rc[1]),
            lambda rc: (rc[0] - rc[1], rc[0], rc[1]),
        )
    )
    coords.sort(key=key)
    for color, group in zip(palette, np.array_split(np.array(coords, dtype=int), color_count)):
        for r, c in group:
            sprite[int(r), int(c)] = color
    return _crop_foreground(sprite)


def _distinct_rotations(sprite: np.ndarray) -> bool:
    rotations = [np.rot90(sprite, k) for k in range(4)]
    return len({rot.tobytes() + str(rot.shape).encode() for rot in rotations}) == 4


def _make_base_sprite(panel_h: int, panel_w: int) -> np.ndarray | None:
    max_side = min(panel_h, panel_w) - 2
    size_options = [side for side in PATTERN_SIDE_CHOICES if side <= max_side]
    if not size_options:
        return None

    h = random.choice(size_options)
    w = random.choice(size_options)
    area = h * w
    low = max(5, area // 4)
    high = min(area - 1, max(low, (area * 3) // 5))
    if low > high:
        return None

    for _ in range(80):
        mask = _connected_mask(h, w, random.randint(low, high))
        cropped = _crop_foreground(mask.astype(int)) > 0
        ch, cw = cropped.shape
        if min(ch, cw) < 2:
            continue
        sprite = _color_sprite(cropped)
        if _distinct_rotations(sprite):
            return sprite
    return None


def _render_pair(panel_h: int, panel_w: int, sprite: np.ndarray, missing_rotation: int, marker_idx: int):
    grid_h = panel_h * 2 + 1
    grid_w = panel_w * 2 + 1
    out = np.full((grid_h, grid_w), BG, dtype=int)
    out[panel_h, :] = SEPARATOR
    out[:, panel_w] = SEPARATOR
    inp = out.copy()

    quadrants = _quadrant_slices(out)
    shown_rotations = [rot for rot in range(4) if rot != missing_rotation]
    random.shuffle(shown_rotations)
    shown_iter = iter(shown_rotations)

    for idx, (rs, cs) in enumerate(quadrants):
        panel_out = out[rs, cs]
        panel_in = inp[rs, cs]
        if idx == marker_idx:
            panel_in[panel_h // 2, panel_w // 2] = MARKER
            _paste_centered(panel_out, np.rot90(sprite, missing_rotation))
            continue

        rot = next(shown_iter)
        rotated = np.rot90(sprite, rot)
        _paste_centered(panel_out, rotated)
        _paste_centered(panel_in, rotated)

    return inp, out


def _candidate() -> np.ndarray | None:
    panel_h = random.choice(PANEL_SIDE_CHOICES)
    panel_w = random.choice(PANEL_SIDE_CHOICES)
    sprite = _make_base_sprite(panel_h, panel_w)
    if sprite is None:
        return None

    missing_rotation = random.randrange(4)
    marker_idx = random.randrange(4)
    inp, expected = _render_pair(panel_h, panel_w, sprite, missing_rotation, marker_idx)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-completion puzzle")

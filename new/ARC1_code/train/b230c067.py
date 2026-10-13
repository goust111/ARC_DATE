from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste, random_free_location
import random


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each 8-connected foreground object by whether its normalized shape is repeated or unique.

    Input structure
    - The dominant background contains several same-color foreground objects.
    - Objects are 8-connected; diagonal contact joins pixels into the same object.
    - Up to translation within each bbox, exactly one object shape appears once
      and exactly one other object shape appears multiple times.

    Rule
    - Group foreground objects by their normalized bbox mask.
    - Recolor the singleton-shape object red.
    - Recolor every repeated-shape object blue.
    - Keep the background unchanged.

    Key insight
    - The recolor is determined by object-shape frequency, not by position or size alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected one foreground color"

    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=fg_colors,
        return_format="dict",
    )
    assert len(components) >= 2, "expected multiple foreground objects"

    shape_keys = [_shape_key(component["coords"]) for component in components]
    counts = Counter(shape_keys)
    singleton_shapes = {key for key, count in counts.items() if count == 1}
    repeated_shapes = {key for key, count in counts.items() if count > 1}
    assert len(singleton_shapes) == 1, "expected exactly one singleton object shape"
    assert len(repeated_shapes) == 1, "expected exactly one repeated object shape"

    out = g.copy()
    for component, shape_key in zip(components, shape_keys):
        coords = component["coords"]
        color = 2 if shape_key in singleton_shapes else 1
        out[coords[:, 0], coords[:, 1]] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas/colors.
GEN_GRID_SHAPE = (10, 10)
GEN_BG, GEN_FG = 0, 8
GEN_OUT_REPEATED, GEN_OUT_SINGLETON = 1, 2
# Shape sampling.
SHAPE_HEIGHT_RANGE = (2, 4)
SHAPE_WIDTH_RANGE = (2, 5)
SHAPE_AREA_RANGE = (4, 10)
SHAPE_FLIP_PROB = 0.5
SHAPE_PAIR_RETRIES = 60
NEIGHBOR_DELTAS_8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
# Placement/retries.
PLACEMENT_PADDING, PLACEMENT_CONNECTIVITY = 1, 8
GENERATION_RETRIES = 300


def _gen_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _gen_dihedral_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    forms = []
    base = mask.astype(bool)
    for k in range(4):
        rot = np.rot90(base, k)
        forms.append(_gen_key(rot))
        forms.append(_gen_key(np.fliplr(rot)))
    return min(forms)


def _gen_random_shape() -> np.ndarray:
    h = random.randint(*SHAPE_HEIGHT_RANGE)
    w = random.randint(*SHAPE_WIDTH_RANGE)
    area = random.randint(max(SHAPE_AREA_RANGE[0], h, w), min(SHAPE_AREA_RANGE[1], h * w))

    path_len = random.randint(max(h, w), min(h + w - 1, area))
    diag_count = h + w - 1 - path_len
    steps = (
        [(1, 1)] * diag_count
        + [(1, 0)] * (h - 1 - diag_count)
        + [(0, 1)] * (w - 1 - diag_count)
    )
    random.shuffle(steps)

    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    for dr, dc in steps:
        r += dr
        c += dc
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in NEIGHBOR_DELTAS_8:
                nr = int(rr) + dr
                nc = int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    if random.random() < SHAPE_FLIP_PROB:
        mask = np.fliplr(mask)
    if random.random() < SHAPE_FLIP_PROB:
        mask = np.flipud(mask)
    return mask


def _gen_pair_shapes() -> tuple[np.ndarray, np.ndarray] | None:
    shape_a = _gen_random_shape()
    key_a = _gen_dihedral_key(shape_a)
    for _ in range(SHAPE_PAIR_RETRIES):
        shape_b = _gen_random_shape()
        if _gen_dihedral_key(shape_b) != key_a:
            return shape_a, shape_b
    return None


def _render_candidate(shape_a: np.ndarray, shape_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(GEN_GRID_SHAPE, dtype=int)
    expected = np.zeros(GEN_GRID_SHAPE, dtype=int)
    pieces = [
        (shape_a, GEN_OUT_REPEATED),
        (shape_a, GEN_OUT_REPEATED),
        (shape_b, GEN_OUT_SINGLETON),
    ]
    random.shuffle(pieces)

    for mask, out_color in pieces:
        in_patch = np.where(mask, GEN_FG, GEN_BG)
        out_patch = np.where(mask, out_color, GEN_BG)
        r, c = random_free_location(
            inp,
            in_patch,
            transparent=GEN_BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_CONNECTIVITY,
        )
        paste(inp, in_patch, r, c, transparent=GEN_BG)
        paste(expected, out_patch, r, c, transparent=GEN_BG)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    pair = _gen_pair_shapes()
    if pair is None:
        return None
    try:
        inp, expected = _render_candidate(*pair)
        actual = main(inp)
    except Exception:
        return None

    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid b230c067 input")

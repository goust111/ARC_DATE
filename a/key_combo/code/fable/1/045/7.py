from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


X_POSITIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (0, 4),
    (3, 1),
    (2, 2),
    (1, 3),
    (4, 0),
    (3, 3),
    (4, 4),
)
KEY_POSITIONS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
X_MASK = np.eye(5, dtype=bool) | np.fliplr(np.eye(5, dtype=bool))
OUTER_RING_MASK = np.ones((5, 5), dtype=bool)
OUTER_RING_MASK[1:4, 1:4] = False
PALETTE = tuple(int(c) for c in Color.NOT_BLACK)
GENERATION_RETRIES = 300


def _fold_x(candidate: np.ndarray, bg: int = Color.BLACK) -> np.ndarray:
    out = np.full((3, 3), int(bg), dtype=int)
    for (sr, sc), (tr, tc) in zip(X_POSITIONS, KEY_POSITIONS):
        out[tr, tc] = int(candidate[sr, sc])
    return out


def _render_x(colors: tuple[int, ...], bg: int = Color.BLACK) -> np.ndarray:
    patch = np.full((5, 5), int(bg), dtype=int)
    for color, (r, c) in zip(colors, X_POSITIONS):
        patch[r, c] = int(color)
    return patch


def _render_key_frame(key: np.ndarray, border_color: int) -> np.ndarray:
    patch = np.full((5, 5), int(border_color), dtype=int)
    patch[1:4, 1:4] = key
    return patch


def _is_x_crop(crop: np.ndarray, bg: int) -> bool:
    return crop.shape == (5, 5) and np.array_equal(crop != int(bg), X_MASK)


def _is_key_crop(crop: np.ndarray, bg: int) -> bool:
    if crop.shape != (5, 5):
        return False
    ring = crop[OUTER_RING_MASK]
    if np.any(ring == int(bg)) or len(set(map(int, ring))) != 1:
        return False
    return bool(np.all(crop[1:4, 1:4] != int(bg)))


def _parse_grid(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )

    candidates: list[np.ndarray] = []
    key: np.ndarray | None = None
    for comp in components:
        patch = crop_bbox(g, comp["bbox"])
        if _is_x_crop(patch, bg):
            candidates.append(patch)
        elif _is_key_crop(patch, bg):
            if key is not None:
                raise AssertionError("expected exactly one framed 3x3 key")
            key = patch[1:4, 1:4].copy()

    if key is None:
        raise AssertionError("missing framed key")
    if not candidates:
        raise AssertionError("missing 5x5 X candidates")
    return candidates, key, bg


def _x_color_multiset(candidate: np.ndarray, bg: int) -> tuple[int, ...]:
    colors = candidate[candidate != int(bg)]
    return tuple(sorted(map(int, colors)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the 5x5 X whose inward-folded 3x3 matches the framed key, and output that original X.

    Input structure:
    - The background is black.
    - Several 5x5 multicolor candidate objects have foreground only on the two diagonals, making X shapes.
    - Elsewhere, one 3x3 solid key is enclosed by a one-cell monochrome frame.

    Rule:
    - For each candidate X, fold the northwest arm onto the top edge, the northeast arm onto the right edge,
      the southwest arm onto the left edge, the southeast arm onto the bottom edge, and keep the center in place.
    - This turns the 5x5 X into a 3x3 grid whose nine cells come from the nine diagonal pixels.
    - Compare every folded candidate against the framed 3x3 key and select the unique exact match.

    Key insight:
    - The key is not a crop or rotation of any candidate; it is a one-to-one remapping of the X's diagonal pixels
      into the perimeter and center of a 3x3 square.
    """
    candidates, key, bg = _parse_grid(grid)
    matches = [candidate for candidate in candidates if np.array_equal(_fold_x(candidate, bg), key)]
    if len(matches) != 1:
        raise AssertionError("expected exactly one candidate to match the folded key")
    return matches[0].copy()


def _place_objects(objects: list[np.ndarray], shape: tuple[int, int], *, border: int, padding: int) -> np.ndarray | None:
    canvas = np.full(shape, Color.BLACK, dtype=int)
    order = list(range(len(objects)))
    random.shuffle(order)
    for idx in order:
        patch = objects[idx]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=Color.BLACK,
                border=border,
                padding=padding,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=Color.BLACK)
    return canvas


def _semantic_checks(inp: np.ndarray, expected_output: np.ndarray, expected_candidates: int) -> bool:
    try:
        candidates, key, bg = _parse_grid(inp)
    except AssertionError:
        return False

    if len(candidates) != expected_candidates:
        return False

    multiset_counts: dict[tuple[int, ...], int] = {}
    for candidate in candidates:
        signature = _x_color_multiset(candidate, bg)
        multiset_counts[signature] = multiset_counts.get(signature, 0) + 1
    if max(multiset_counts.values()) < 2:
        return False

    matches = [candidate for candidate in candidates if np.array_equal(_fold_x(candidate, bg), key)]
    if len(matches) != 1:
        return False

    if not np.array_equal(matches[0], expected_output):
        return False
    if not np.array_equal(main(inp), expected_output):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_candidates = random.randint(2, 5)
    candidate_perms: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(candidate_perms) < n_candidates:
        perm = tuple(random.sample(PALETTE, len(PALETTE)))
        if perm in seen:
            continue
        seen.add(perm)
        candidate_perms.append(perm)

    candidates = [_render_x(perm) for perm in candidate_perms]
    winner_idx = random.randrange(n_candidates)
    winner = candidates[winner_idx]
    key = _fold_x(winner)
    border_color = random.choice(PALETTE)
    key_frame = _render_key_frame(key, border_color)

    n_objects = n_candidates + 1
    min_side = max(16, 6 + 6 * ((n_objects + 1) // 3))
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    border = random.randint(0, 1)
    padding = random.randint(1, 2)

    objects = candidates + [key_frame]
    inp = _place_objects(objects, (height, width), border=border, padding=padding)
    if inp is None:
        return None
    if not _semantic_checks(inp, winner, n_candidates):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-key X puzzle")

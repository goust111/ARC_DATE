from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, paste


BG = Color.BLACK
SPRITE = Color.RED
SEPARATOR = Color.ORANGE
OBSERVED_FRAMES = 4
TOTAL_FRAMES = 5
MAX_GENERATION_ATTEMPTS = 400
PANEL_HEIGHTS = (4, 5, 6)
PANEL_WIDTHS = (5, 6, 7, 8, 9)
MOTION_MODES = ("horizontal", "vertical", "diagonal")


def _rotate_cw(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, -1)


def _rotate_cw_n(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -int(turns) % 4)


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    assert grid.ndim == 2, "expected a 2D grid"
    separator_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    assert len(separator_rows) == OBSERVED_FRAMES - 1, "expected exactly three separator rows"

    panel_h = separator_rows[0]
    expected = [panel_h + i * (panel_h + 1) for i in range(OBSERVED_FRAMES - 1)]
    assert separator_rows == expected, "separator rows must be evenly spaced"
    assert grid.shape[0] == OBSERVED_FRAMES * panel_h + (OBSERVED_FRAMES - 1)

    panels = []
    start = 0
    for sep in separator_rows:
        panels.append(grid[start:sep, :])
        start = sep + 1
    panels.append(grid[start:, :])
    assert len({panel.shape for panel in panels}) == 1, "all panels must match in size"
    return panels


def _extract_sprite(panel: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    assert set(map(int, np.unique(panel))).issubset({BG, SPRITE}), "unexpected panel colors"
    objects = find_components(
        panel,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[SPRITE],
    )
    assert len(objects) == 1, "expected exactly one red sprite per panel"
    obj = objects[0]
    assert 3 <= int(obj["area"]) <= 5, "sprite area must stay within the intended range"
    return tuple(map(int, obj["bbox"])), crop_bbox(panel, obj["bbox"])


def _frame_out(panel_shape: tuple[int, int], patch: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.full(panel_shape, BG, dtype=int)
    paste(out, patch, top, left, transparent=BG)
    return out


def _safe_frame_out(
    panel_shape: tuple[int, int],
    patch: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray | None:
    h, w = map(int, patch.shape)
    panel_h, panel_w = map(int, panel_shape)
    if top < 0 or left < 0 or top + h > panel_h or left + w > panel_w:
        return None
    return _frame_out(panel_shape, patch, top, left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the fifth panel of a four-panel strip where one sprite both translates and rotates 90 degrees clockwise each frame.

    Input structure:
    - The grid is a vertical stack of four equal black panels separated by solid full-width orange rows.
    - Each panel contains exactly one red 4-connected sprite.
    - From top to bottom, the sprite's bbox upper-left corner shifts by a constant (dr, dc), and the cropped sprite turns 90 degrees clockwise each frame.

    Rule:
    - Split the strip into its four panels using the orange separator rows.
    - In each panel, extract the red sprite's bbox position and the cropped sprite patch.
    - Infer the constant translation from consecutive bbox upper-left differences, and verify each next patch is the previous patch rotated 90 degrees clockwise.
    - Output one panel of the same size containing the next sprite: rotate the fourth patch another 90 degrees clockwise and place it at the translated bbox position.

    Key insight:
    - The sequence hides two independent parameters at once: position and orientation. Extrapolating only the translation gives the wrong shape, and extrapolating only the rotation gives the wrong location.
    """
    g = np.asarray(grid, dtype=int)
    panels = _split_panels(g)

    frames = [_extract_sprite(panel) for panel in panels]
    positions = [(bbox[0], bbox[1]) for bbox, _patch in frames]
    patches = [patch for _bbox, patch in frames]

    deltas = [
        (positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
        for i in range(len(positions) - 1)
    ]
    assert len(set(deltas)) == 1, "sprite translation must be constant"
    dr, dc = deltas[0]
    assert dr != 0 or dc != 0, "translation must be visible"

    for i in range(len(patches) - 1):
        assert np.array_equal(_rotate_cw(patches[i]), patches[i + 1]), "sprite rotation must be 90° clockwise"

    next_patch = _rotate_cw(patches[-1])
    next_top = positions[-1][0] + dr
    next_left = positions[-1][1] + dc
    return _frame_out(panels[0].shape, next_patch, next_top, next_left)


# --------------- generator (o5x) ---------------


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(int)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.tolist())


def _distinct_rotations(mask: np.ndarray) -> bool:
    rotations = {_mask_key(_rotate_cw_n(mask, k)) for k in range(4)}
    return len(rotations) == 4


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 5)
        mask = np.zeros((3, 3), dtype=bool)
        r = random.randrange(3)
        c = random.randrange(3)
        mask[r, c] = True

        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr) + dr
                    nc = int(cc) + dc
                    if 0 <= nr < 3 and 0 <= nc < 3 and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        if int(mask.sum()) != area:
            continue
        cropped = _normalize_mask(mask)
        if _distinct_rotations(cropped):
            return cropped

    # Deterministic fallback: an L triomino has four distinct 90-degree rotations.
    return np.array([[1, 0], [1, 1]], dtype=int)


def _candidate_vectors(mode: str) -> list[tuple[int, int]]:
    if mode == "horizontal":
        return [(0, -2), (0, -1), (0, 1), (0, 2)]
    if mode == "vertical":
        return [(-1, 0), (1, 0)]
    return [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _all_vectors() -> list[tuple[int, int]]:
    vectors: list[tuple[int, int]] = []
    for mode in MOTION_MODES:
        vectors.extend(_candidate_vectors(mode))
    return vectors


def _feasible_starts(
    panel_h: int,
    panel_w: int,
    patches: list[np.ndarray],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    starts = []
    for top in range(panel_h):
        for left in range(panel_w):
            ok = True
            for t, patch in enumerate(patches):
                h, w = map(int, patch.shape)
                rr = top + t * dr
                cc = left + t * dc
                if rr < 0 or cc < 0 or rr + h > panel_h or cc + w > panel_w:
                    ok = False
                    break
            if ok:
                starts.append((top, left))
    return starts


def _choose_layout(patches: list[np.ndarray]) -> tuple[int, int, int, int, int, int] | None:
    mode_order = list(MOTION_MODES)
    random.shuffle(mode_order)
    width_order = list(PANEL_WIDTHS)
    random.shuffle(width_order)
    height_order = list(PANEL_HEIGHTS)
    random.shuffle(height_order)

    for mode in mode_order + ["fallback"]:
        vectors = _candidate_vectors(mode) if mode != "fallback" else _all_vectors()
        random.shuffle(vectors)
        options: list[tuple[int, int, int, int, int, int]] = []
        for panel_h in height_order:
            for panel_w in width_order:
                for dr, dc in vectors:
                    starts = _feasible_starts(panel_h, panel_w, patches, dr, dc)
                    if not starts:
                        continue
                    top, left = random.choice(starts)
                    options.append((panel_h, panel_w, dr, dc, top, left))
        if options:
            return random.choice(options)
    return None


def _build_strip(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = map(int, panels[0].shape)
    out = np.full((OBSERVED_FRAMES * panel_h + (OBSERVED_FRAMES - 1), panel_w), BG, dtype=int)
    for i, panel in enumerate(panels):
        row0 = i * (panel_h + 1)
        out[row0 : row0 + panel_h, :] = panel
        if i < OBSERVED_FRAMES - 1:
            out[row0 + panel_h, :] = SEPARATOR
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    observed_panels: list[np.ndarray],
    patches: list[np.ndarray],
    positions: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    panel_shape = observed_panels[0].shape
    last_top, last_left = positions[-1]
    translation_only = _safe_frame_out(panel_shape, patches[-1], last_top + dr, last_left + dc)
    rotation_only = _safe_frame_out(panel_shape, _rotate_cw(patches[-1]), last_top, last_left)

    if translation_only is not None and np.array_equal(translation_only, expected):
        return False
    if rotation_only is not None and np.array_equal(rotation_only, expected):
        return False
    if any(np.array_equal(panel, expected) for panel in observed_panels):
        return False

    # The observed history should show both parameters changing on-screen.
    if len({_mask_key(patch) for patch in patches[:OBSERVED_FRAMES]}) != OBSERVED_FRAMES:
        return False
    if len(set(positions[:OBSERVED_FRAMES])) != OBSERVED_FRAMES:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    base = _random_connected_mask()
    start_turn = random.randrange(4)
    patches = [_rotate_cw_n(base, start_turn + i) * SPRITE for i in range(TOTAL_FRAMES)]

    layout = _choose_layout(patches)
    if layout is None:
        return None
    panel_h, panel_w, dr, dc, top0, left0 = layout

    panels = []
    positions = []
    for t in range(TOTAL_FRAMES):
        top = top0 + t * dr
        left = left0 + t * dc
        positions.append((top, left))
        panels.append(_frame_out((panel_h, panel_w), patches[t], top, left))

    inp = _build_strip(panels[:OBSERVED_FRAMES])
    expected = panels[-1]
    if not _semantic_checks(inp, expected, panels[:OBSERVED_FRAMES], patches, positions, dr, dc):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")

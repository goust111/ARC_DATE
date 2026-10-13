import numpy as np

from arc_utils import Color, colors_present, get_bg, paste
import random


_RING_MASK = np.ones((3, 3), dtype=bool)
_RING_MASK[1, 1] = False
_CORNER_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_PLUS_OFFSETS = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))


def _target_centers(g: np.ndarray, bg: int, fg: int) -> list[tuple[int, int]]:
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3

    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != bg:
                continue
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            if bool(np.all(block[_RING_MASK] == fg)):
                centers.append((r, c))

    assert len(centers) == 2
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each hollow 3x3 foreground square with a red plus sign in the same window.

    Input structure:
    - A background grid containing one foreground color.
    - Exactly two target motifs are 3x3 hollow squares: a background center surrounded by eight foreground pixels.

    Rule:
    - For each target motif, erase the four corner pixels to background.
    - Recolor the four edge-neighbor pixels and fill the center pixel red.
    - Leave all other pixels unchanged.

    Key insight:
    The changing regions are local masks: each square outline is rewritten in place as a plus mask.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = fg_colors[0]
    assert fg != Color.RED

    out = g.copy()
    written: set[tuple[int, int]] = set()
    for r, c in _target_centers(g, bg, fg):
        cells = {(r + dr, c + dc) for dr, dc in (*_CORNER_OFFSETS, *_PLUS_OFFSETS)}
        assert written.isdisjoint(cells)
        written.update(cells)

        for dr, dc in _CORNER_OFFSETS:
            out[r + dr, c + dc] = bg
        for dr, dc in _PLUS_OFFSETS:
            out[r + dr, c + dc] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and color choices.
_GEN_GRID_SIZE = 10
_GEN_BG = 0
_GEN_FG_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]

# Target geometry and rewrite masks.
_GEN_TARGET_COUNT = 2
_GEN_TARGET_RADIUS = 1
_GEN_TARGET_CENTER_RANGE = range(_GEN_TARGET_RADIUS, _GEN_GRID_SIZE - _GEN_TARGET_RADIUS)
_GEN_TARGET_SEPARATION = 1

_GEN_RING = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
_GEN_RING_MASK = _GEN_RING.astype(bool)

_GEN_PLUS = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]], dtype=int)
_GEN_FULL_TARGET_MASK = np.ones_like(_GEN_RING, dtype=bool)
_GEN_RED_PLUS_PATCH = np.where(_GEN_PLUS == Color.RED, Color.RED, _GEN_BG)

# Distractor library and placement rules.
_GEN_DISTRACTOR_COUNT = 3
_GEN_DISTRACTOR_MARGIN_CONNECTIVITY = 4
_GEN_DISTRACTOR_BASES = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
]

# Retry budget for rejection sampling.
_GEN_MAX_ATTEMPTS = 300


def _gen_patch_orientations(patch: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out = []
    for k in range(4):
        rot = np.rot90(patch, k)
        for flipped in (rot, np.fliplr(rot)):
            rows = np.where(flipped.any(axis=1))[0]
            cols = np.where(flipped.any(axis=0))[0]
            trimmed = flipped[
                rows.min() : rows.max() + 1, cols.min() : cols.max() + 1
            ].copy()
            key = tuple(map(tuple, trimmed.tolist()))
            if key not in seen:
                seen.add(key)
                out.append(trimmed)
    return out


_GEN_DISTRACTOR_PATCHES = [
    patch for base in _GEN_DISTRACTOR_BASES for patch in _gen_patch_orientations(base)
]


def _gen_blocked_with_margin(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Return occupied cells plus their 4- or 8-neighbor margin."""
    blocked = mask.copy()
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    if connectivity == 8:
        blocked[1:, 1:] |= mask[:-1, :-1]
        blocked[1:, :-1] |= mask[:-1, 1:]
        blocked[:-1, 1:] |= mask[1:, :-1]
        blocked[:-1, :-1] |= mask[1:, 1:]
    return blocked


def _gen_would_make_extra_ring(g: np.ndarray, fg: int) -> bool:
    count = 0
    for r in _GEN_TARGET_CENTER_RANGE:
        for c in _GEN_TARGET_CENTER_RANGE:
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            if int(g[r, c]) == _GEN_BG and bool(np.all(block[_GEN_RING_MASK] == fg)):
                count += 1
    return count != _GEN_TARGET_COUNT


def _gen_target_window_is_separate(
    existing_centers: list[tuple[int, int]], r: int, c: int
) -> bool:
    new_cells = {
        (rr, cc)
        for rr in range(r - _GEN_TARGET_RADIUS, r + _GEN_TARGET_RADIUS + 1)
        for cc in range(c - _GEN_TARGET_RADIUS, c + _GEN_TARGET_RADIUS + 1)
    }
    for er, ec in existing_centers:
        for rr in range(er - _GEN_TARGET_RADIUS, er + _GEN_TARGET_RADIUS + 1):
            for cc in range(ec - _GEN_TARGET_RADIUS, ec + _GEN_TARGET_RADIUS + 1):
                if any(
                    abs(rr - nr) + abs(cc - nc) <= _GEN_TARGET_SEPARATION
                    for nr, nc in new_cells
                ):
                    return False
    return True


def _gen_target_center_pair() -> list[tuple[int, int]]:
    centers = [
        (r, c) for r in _GEN_TARGET_CENTER_RANGE for c in _GEN_TARGET_CENTER_RANGE
    ]
    pairs = [
        [a, b]
        for i, a in enumerate(centers)
        for b in centers[i + 1 :]
        if _gen_target_window_is_separate([a], b[0], b[1])
    ]
    return random.choice(pairs)


def _gen_expected_output(inp: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for r, c in centers:
        paste(out, _GEN_RED_PLUS_PATCH, r - 1, c - 1, mask=_GEN_FULL_TARGET_MASK)
    return out


def _gen_place_distractor(g: np.ndarray, occupied: np.ndarray, fg: int) -> bool:
    blocked = _gen_blocked_with_margin(
        occupied, connectivity=_GEN_DISTRACTOR_MARGIN_CONNECTIVITY
    )
    for patch in random.sample(_GEN_DISTRACTOR_PATCHES, len(_GEN_DISTRACTOR_PATCHES)):
        colored_patch = np.where(patch != 0, fg, _GEN_BG)
        h, w = patch.shape
        local_cells = np.argwhere(patch != 0)
        positions = []
        for r in range(_GEN_GRID_SIZE - h + 1):
            for c in range(_GEN_GRID_SIZE - w + 1):
                cells = local_cells + np.array([r, c])
                if np.any(blocked[cells[:, 0], cells[:, 1]]):
                    continue
                trial = g.copy()
                paste(trial, colored_patch, r, c, transparent=_GEN_BG)
                if not _gen_would_make_extra_ring(trial, fg):
                    positions.append((r, c, cells))
        if positions:
            r, c, cells = random.choice(positions)
            paste(g, colored_patch, r, c, transparent=_GEN_BG)
            occupied[cells[:, 0], cells[:, 1]] = True
            return True
    return False


def _gen_place_targets(
    g: np.ndarray, occupied: np.ndarray, fg: int, centers: list[tuple[int, int]]
) -> None:
    ring_patch = np.where(_GEN_RING != 0, fg, _GEN_BG)
    for r, c in centers:
        paste(g, ring_patch, r - 1, c - 1, transparent=_GEN_BG)
        occupied[
            r - _GEN_TARGET_RADIUS : r + _GEN_TARGET_RADIUS + 1,
            c - _GEN_TARGET_RADIUS : c + _GEN_TARGET_RADIUS + 1,
        ] = True


def _gen_matches_solver(g: np.ndarray, centers: list[tuple[int, int]]) -> bool:
    expected = _gen_expected_output(g, centers)
    try:
        actual = main(g)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    fg = random.choice(_GEN_FG_COLORS)
    g = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    occupied = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=bool)

    target_centers = _gen_target_center_pair()
    _gen_place_targets(g, occupied, fg, target_centers)
    for _ in range(_GEN_DISTRACTOR_COUNT):
        if not _gen_place_distractor(g, occupied, fg):
            return None
    if not _gen_matches_solver(g, target_centers):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 6c434453 input")

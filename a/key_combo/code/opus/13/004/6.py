import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

RING_OFFSETS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
CORNER_OFFSETS = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDE_OFFSETS = ((0, 1), (1, 2), (2, 1), (1, 0))

PANEL_SIDE_MIN = 9
PANEL_SIDE_MAX = 13
RING_COUNT_CHOICES = (2, 3)
GEN_ATTEMPTS = 400
INNER_PLACE_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _separator_col(grid: np.ndarray, bg: int) -> int:
    matches = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            matches.append(c)
    assert len(matches) == 1, "expected exactly one solid non-background separator"
    return int(matches[0])


def _component_coords(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    stack = [start]
    seen[start] = True
    coords = []
    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return coords


def _find_rings(panel: np.ndarray, bg: int) -> dict[tuple[int, int], np.ndarray]:
    mask = panel != bg
    seen = np.zeros_like(mask, dtype=bool)
    rings: dict[tuple[int, int], np.ndarray] = {}
    for r in range(panel.shape[0]):
        for c in range(panel.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            coords = _component_coords(mask, (r, c), seen)
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(coords) == 8, "foreground objects must be 3x3 rings"
            assert r1 == r0 + 2 and c1 == c0 + 2, "ring bbox must be 3x3"
            assert int(panel[r0 + 1, c0 + 1]) == bg, "ring center must be background"
            expected = {(r0 + dr, c0 + dc) for dr, dc in RING_OFFSETS}
            assert set(coords) == expected, "component must occupy the ring perimeter"
            rings[(r0, c0)] = panel[r0 : r0 + 3, c0 : c0 + 3].copy()
    assert rings, "expected at least one ring"
    return rings


def _cycled(values: list[int], *, clockwise: bool) -> list[int]:
    if clockwise:
        return [values[-1], *values[:-1]]
    return [*values[1:], values[0]]


def _cycle_composite_ring(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    corner_values = [int(patch[r, c]) for r, c in CORNER_OFFSETS]
    side_values = [int(patch[r, c]) for r, c in SIDE_OFFSETS]
    for (r, c), value in zip(CORNER_OFFSETS, _cycled(corner_values, clockwise=True)):
        out[r, c] = value
    for (r, c), value in zip(SIDE_OFFSETS, _cycled(side_values, clockwise=False)):
        out[r, c] = value
    return out


def _compose_pair(left_patch: np.ndarray, right_patch: np.ndarray, bg: int) -> np.ndarray:
    out = np.full((3, 3), bg, dtype=int)
    for r, c in CORNER_OFFSETS:
        out[r, c] = int(left_patch[r, c])
    for r, c in SIDE_OFFSETS:
        out[r, c] = int(right_patch[r, c])
    return _cycle_composite_ring(out)


def _paste_ring(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int], bg: int) -> None:
    r0, c0 = top_left
    for dr, dc in RING_OFFSETS:
        rr, cc = r0 + dr, c0 + dc
        assert grid[rr, cc] == bg
        grid[rr, cc] = int(patch[dr, dc])


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the two ring panels across the separator, then recolor only the merged rings by opposite corner/side cycles.

    Input structure:
    - A black-background grid is split by one full-height grey separator column into equal-width left and right panels.
    - Each side panel contains isolated multicolor 3x3 ring objects: eight foreground cells around a black center.
    - Some rings occupy mirror-symmetric positions across the separator and some appear on only one side.

    Rule:
    - Remove the separator and mirror the right panel horizontally into the left panel's coordinate system.
    - At every position containing rings on both sides, build a composite ring whose corner cells come from the left ring and whose side-center cells come from the mirrored right ring.
    - Cycle each composite ring by one step: corner colors move clockwise, side-center colors move counterclockwise.
    - Rings that appear on only one side are copied unchanged, and all background cells stay black.

    Key insight:
    - The fold does not simply overlay pixels: overlap first creates a new ring by role-based sampling (corners from left, sides from right), and only those newly composite rings are cycled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    sep = _separator_col(g, bg)
    h, w = g.shape
    assert 2 * sep + 1 == w, "separator must split equal-width panels"

    left = g[:, :sep]
    right = np.fliplr(g[:, sep + 1 :])
    assert left.shape == right.shape

    left_rings = _find_rings(left, bg)
    right_rings = _find_rings(right, bg)

    out = np.full_like(left, bg)
    for pos in sorted(set(left_rings) | set(right_rings)):
        left_patch = left_rings.get(pos)
        right_patch = right_rings.get(pos)
        if left_patch is not None and right_patch is not None:
            patch = _compose_pair(left_patch, right_patch, bg)
        else:
            patch = left_patch if left_patch is not None else right_patch
        assert patch is not None
        _paste_ring(out, patch, pos, bg)
    return out


def _bbox_separated(a: tuple[int, int], b: tuple[int, int], pad: int = 1) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 2, ac0 + 2
    br1, bc1 = br0 + 2, bc0 + 2
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _choose_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for pos in candidates:
        if all(_bbox_separated(pos, other, pad=1) for other in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return chosen
    return None


def _sample_role_values(require_variation: bool) -> list[int]:
    while True:
        pool_size = random.randint(2, min(5, len(PALETTE)))
        pool = random.sample(PALETTE, pool_size)
        values = [random.choice(pool) for _ in range(4)]
        if not require_variation or len(set(values)) >= 2:
            return values


def _make_ring(require_corner_variation: bool, require_side_variation: bool) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    corner_values = _sample_role_values(require_corner_variation)
    side_values = _sample_role_values(require_side_variation)
    for (r, c), value in zip(CORNER_OFFSETS, corner_values):
        patch[r, c] = value
    for (r, c), value in zip(SIDE_OFFSETS, side_values):
        patch[r, c] = value
    return patch


def _pair_layout() -> tuple[int, int, int]:
    options = []
    for left_count in RING_COUNT_CHOICES:
        for right_count in RING_COUNT_CHOICES:
            for pair_count in range(1, min(left_count, right_count) + 1):
                single_total = left_count + right_count - 2 * pair_count
                if single_total >= 1:
                    options.append((left_count, right_count, pair_count))
    return random.choice(options)


def _assemble_input(
    h: int,
    w: int,
    left_rings: dict[tuple[int, int], np.ndarray],
    right_rings: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    left_panel = np.full((h, w), BG, dtype=int)
    right_panel = np.full((h, w), BG, dtype=int)
    for pos, patch in left_rings.items():
        _paste_ring(left_panel, patch, pos, BG)
    for (r0, c0), patch in right_rings.items():
        mirrored_input_c = w - 3 - c0
        _paste_ring(right_panel, np.fliplr(patch), (r0, mirrored_input_c), BG)
    return np.concatenate([left_panel, np.full((h, 1), SEP, dtype=int), right_panel], axis=1)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX)
    w = random.randint(PANEL_SIDE_MIN, PANEL_SIDE_MAX)
    left_count, right_count, pair_count = _pair_layout()
    union_count = left_count + right_count - pair_count

    positions = _choose_positions(h, w, union_count)
    if positions is None:
        return None
    random.shuffle(positions)

    pair_positions = positions[:pair_count]
    remaining = positions[pair_count:]
    left_only_count = left_count - pair_count
    right_only_count = right_count - pair_count
    left_only_positions = remaining[:left_only_count]
    right_only_positions = remaining[left_only_count : left_only_count + right_only_count]

    left_rings: dict[tuple[int, int], np.ndarray] = {}
    right_rings: dict[tuple[int, int], np.ndarray] = {}

    for pos in pair_positions:
        left_rings[pos] = _make_ring(require_corner_variation=True, require_side_variation=False)
        right_rings[pos] = _make_ring(require_corner_variation=False, require_side_variation=True)
        composite = _compose_pair(left_rings[pos], right_rings[pos], BG)
        raw = np.full((3, 3), BG, dtype=int)
        for r, c in CORNER_OFFSETS:
            raw[r, c] = left_rings[pos][r, c]
        for r, c in SIDE_OFFSETS:
            raw[r, c] = right_rings[pos][r, c]
        if np.array_equal(composite, raw):
            return None

    for pos in left_only_positions:
        left_rings[pos] = _make_ring(require_corner_variation=True, require_side_variation=True)
    for pos in right_only_positions:
        right_rings[pos] = _make_ring(require_corner_variation=True, require_side_variation=True)

    grid = _assemble_input(h, w, left_rings, right_rings)
    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (h, w):
        return None
    if len(left_rings) not in RING_COUNT_CHOICES or len(right_rings) not in RING_COUNT_CHOICES:
        return None
    if not pair_positions or not (left_only_positions or right_only_positions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-ring task instance")

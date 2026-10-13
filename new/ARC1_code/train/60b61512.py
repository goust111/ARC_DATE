import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, is_contiguous, paste
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete 3x3 yellow object by coloring the missing cells in its bounding box orange.

    Input structure
    - The grid has a black background and one or more 8-connected yellow objects.
    - Each yellow object fits in a 3x3 bounding box and has black gaps inside that box.

    Rule
    - Find each 8-connected yellow object.
    - For the object's 3x3 bounding box, recolor every black cell in that box to orange.
    - Keep the existing yellow pixels and all background outside those boxes unchanged.

    Key insight
    - The yellow pixels mark partial 3x3 blocks; orange fills the missing pixels needed to make each block solid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.YELLOW}, "expected only black and yellow in input"

    objects = find_components(g, bg=bg, connectivity=8, monochromatic=True, colors=[Color.YELLOW])
    assert objects, "expected at least one yellow object"

    out = g.copy()
    for obj in objects:
        assert obj["color"] == Color.YELLOW, "expected yellow objects"
        box = obj["bbox"]
        assert bbox_h(box) == 3 and bbox_w(box) == 3, "expected each object bbox to be 3x3"
        r0, c0, r1, c1 = box

        src_region = g[r0 : r1 + 1, c0 : c1 + 1]
        missing = src_region == bg
        assert np.any(missing), "expected at least one missing cell inside each bbox"

        dst_region = out[r0 : r1 + 1, c0 : c1 + 1]
        dst_region[missing] = Color.ORANGE

    return out

# --------------- generator (o5x) ---------------
# Palette used by this task.
_GEN_BLACK = Color.BLACK
_GEN_YELLOW = Color.YELLOW
_GEN_ORANGE = Color.ORANGE

# Canvas and object geometry.
_GEN_GRID_SIZE = 9
_GEN_BOX_SIZE = 3
_GEN_OBJECT_COUNT = 2
_GEN_MIN_BOX_GAP = 1
_GEN_TOP_LEFT_MAX = _GEN_GRID_SIZE - _GEN_BOX_SIZE

# 3x3 object pattern density.
_GEN_MIN_YELLOW_CELLS = 4
_GEN_MAX_YELLOW_CELLS = 8

# Rejection sampling budget.
_GEN_MAX_ATTEMPTS = 300

def _gen_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(np.all(np.any(mask, axis=0)) and np.all(np.any(mask, axis=1)))


def _gen_patterns_by_count() -> dict[int, list[np.ndarray]]:
    patterns: dict[int, list[np.ndarray]] = {k: [] for k in range(_GEN_MIN_YELLOW_CELLS, _GEN_MAX_YELLOW_CELLS + 1)}
    for k in patterns:
        for cells in itertools.combinations(range(_GEN_BOX_SIZE * _GEN_BOX_SIZE), k):
            mask = np.zeros((_GEN_BOX_SIZE, _GEN_BOX_SIZE), dtype=int)
            mask.flat[list(cells)] = 1
            if _gen_has_full_bbox(mask) and is_contiguous(mask, background=_GEN_BLACK, connectivity=8):
                patterns[k].append(np.where(mask, _GEN_YELLOW, _GEN_BLACK).astype(int))
    return patterns


def _gen_position_pairs() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    positions = [(r, c) for r in range(_GEN_TOP_LEFT_MAX + 1) for c in range(_GEN_TOP_LEFT_MAX + 1)]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, (r0, c0) in enumerate(positions):
        for r1, c1 in positions[i + 1 :]:
            separated = (
                r0 + _GEN_BOX_SIZE + _GEN_MIN_BOX_GAP <= r1
                or r1 + _GEN_BOX_SIZE + _GEN_MIN_BOX_GAP <= r0
                or c0 + _GEN_BOX_SIZE + _GEN_MIN_BOX_GAP <= c1
                or c1 + _GEN_BOX_SIZE + _GEN_MIN_BOX_GAP <= c0
            )
            leaves_top_or_bottom_black = not (min(r0, r1) == 0 and max(r0, r1) == _GEN_TOP_LEFT_MAX)
            if separated and leaves_top_or_bottom_black:
                pairs.append(((r0, c0), (r1, c1)))
    return pairs


_GEN_PATTERNS_BY_COUNT = _gen_patterns_by_count()
_GEN_POSITION_PAIRS = _gen_position_pairs()


def _gen_sample_patches() -> list[np.ndarray]:
    counts = [random.randint(_GEN_MIN_YELLOW_CELLS, _GEN_MAX_YELLOW_CELLS) for _ in range(_GEN_OBJECT_COUNT)]
    return [random.choice(_GEN_PATTERNS_BY_COUNT[k]).copy() for k in counts]


def _gen_expected_output(grid: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()
    for r, c in positions:
        region = expected[r : r + _GEN_BOX_SIZE, c : c + _GEN_BOX_SIZE]
        region[region == _GEN_BLACK] = _GEN_ORANGE
    return expected


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((_GEN_GRID_SIZE, _GEN_GRID_SIZE), _GEN_BLACK, dtype=int)
    patches = _gen_sample_patches()
    positions = list(random.choice(_GEN_POSITION_PAIRS))
    random.shuffle(positions)

    for patch, (r, c) in zip(patches, positions):
        paste(grid, patch, r, c, transparent=_GEN_BLACK)

    expected = _gen_expected_output(grid, positions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 60b61512 input")

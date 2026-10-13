from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

_RNG = np.random.default_rng()

_DIM_CHOICES = (12, 14, 16, 18, 20, 22, 24, 26)
_MAX_GENERATION_ATTEMPTS = 500
_DIRECTIONS = ("up", "down", "left", "right")
_COUNT_PATTERNS = (
    (1, 1, 2, 3),
    (1, 2, 2, 3),
    (1, 2, 3, 3),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_block(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == GREEN
    box = _bbox(mask)
    assert box is not None, "expected one green 2x2 block"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "green anchor must be 2x2"
    assert int(mask.sum()) == 4, "green anchor must be a filled 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN), "green anchor must fill its bbox"
    return r0, c0, r1, c1


def _classify_rays(grid: np.ndarray, block_bbox: tuple[int, int, int, int]) -> dict[str, dict[str, object]]:
    r0, c0, r1, c1 = block_bbox
    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert red_coords, "expected red markers on the four rays"

    groups = {
        "up": {c0: [], c1: []},
        "down": {c0: [], c1: []},
        "left": {r0: [], r1: []},
        "right": {r0: [], r1: []},
    }

    for r, c in red_coords:
        if r <= r0 - 2 and c in (c0, c1):
            groups["up"][c].append(r0 - r)
        elif r >= r1 + 2 and c in (c0, c1):
            groups["down"][c].append(r - r1)
        elif c <= c0 - 2 and r in (r0, r1):
            groups["left"][r].append(c0 - c)
        elif c >= c1 + 2 and r in (r0, r1):
            groups["right"][r].append(c - c1)
        else:
            raise AssertionError("every red marker must lie on one of the four cardinal rays")

    ray_info: dict[str, dict[str, object]] = {}
    for direction, candidates in groups.items():
        nonempty = [(axis, sorted(map(int, dists))) for axis, dists in candidates.items() if dists]
        assert len(nonempty) == 1, f"{direction} must use exactly one candidate row/column"
        axis_value, dists = nonempty[0]
        count = len(dists)
        assert 1 <= count <= 3, f"{direction} must contain 1..3 red markers"
        ray_info[direction] = {"axis": int(axis_value), "dists": tuple(dists), "count": int(count)}

    total = sum(int(info["count"]) for info in ray_info.values())
    assert total == len(red_coords), "all red markers must belong to exactly one ray"
    counts = [int(ray_info[direction]["count"]) for direction in _DIRECTIONS]
    assert len(set(counts)) >= 2, "the four ray counts must not all be equal"
    return ray_info


def _frame_mask(
    shape: tuple[int, int],
    block_bbox: tuple[int, int, int, int],
    counts: dict[str, int],
) -> np.ndarray:
    H, W = map(int, shape)
    r0, c0, r1, c1 = block_bbox
    n_up = int(counts["up"])
    n_down = int(counts["down"])
    n_left = int(counts["left"])
    n_right = int(counts["right"])

    tl = max(n_up, n_left)
    tr = max(n_up, n_right)
    bl = max(n_down, n_left)
    br = max(n_down, n_right)

    assert r0 - tl >= 0 and r0 - tr >= 0
    assert c0 - tl >= 0 and c0 - bl >= 0
    assert r1 + bl < H and r1 + br < H
    assert c1 + tr < W and c1 + br < W

    mask = np.zeros((H, W), dtype=bool)
    mask[r0 - n_up : r0, c0 - tl : c1 + tr + 1] = True
    mask[r1 + 1 : r1 + n_down + 1, c0 - bl : c1 + br + 1] = True
    mask[r0 - tl : r1 + bl + 1, c0 - n_left : c0] = True
    mask[r0 - tr : r1 + br + 1, c1 + 1 : c1 + n_right + 1] = True
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _counts_from_ray_info(ray_info: dict[str, dict[str, object]]) -> dict[str, int]:
    return {direction: int(ray_info[direction]["count"]) for direction in _DIRECTIONS}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red markers on the four rays around the green 2x2 anchor and convert those counts into an asymmetric yellow concentric frame.

    Input structure:
    - The grid is mostly black and contains one filled green 2x2 block acting as the anchor.
    - Red marker pixels lie on four cardinal rays extending away from the block: one upward, one downward, one leftward, and one rightward.
    - For each direction, markers occupy exactly one candidate row or column aligned with the corresponding side of the 2x2 block, and each direction has 1 to 3 markers.

    Rule:
    - Locate the green 2x2 block.
    - Count red markers on the upward, downward, leftward, and rightward rays to obtain `n_up`, `n_down`, `n_left`, and `n_right`.
    - Erase all red markers.
    - Draw a yellow frame tightly around the block: the top, bottom, left, and right side thicknesses are `n_up`, `n_down`, `n_left`, and `n_right` respectively, while each corner uses the larger thickness of its two adjacent sides.
    - Keep the green 2x2 block unchanged.

    Key insight:
    - The red markers are not copied or moved; their per-direction counts independently encode the geometric thickness of the four sides of the output frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREEN}, "unexpected colors present"

    block_bbox = _find_block(g)
    ray_info = _classify_rays(g, block_bbox)
    counts = _counts_from_ray_info(ray_info)

    out = g.copy()
    out[out == RED] = BLACK
    out[_frame_mask(g.shape, block_bbox, counts)] = YELLOW
    return out


def _sample_counts() -> dict[str, int]:
    while True:
        pattern = _COUNT_PATTERNS[int(_RNG.integers(0, len(_COUNT_PATTERNS)))]
        permuted = list(pattern)
        _RNG.shuffle(permuted)
        counts = {direction: int(permuted[i]) for i, direction in enumerate(_DIRECTIONS)}
        adjacent = [
            counts["up"] != counts["left"],
            counts["up"] != counts["right"],
            counts["down"] != counts["left"],
            counts["down"] != counts["right"],
        ]
        if sum(adjacent) >= 3:
            return counts


def _choose_positions(available: Iterable[int], count: int) -> tuple[int, ...]:
    choices = np.array(sorted(set(int(v) for v in available)), dtype=int)
    picked = _RNG.choice(choices, size=int(count), replace=False)
    return tuple(sorted(map(int, picked.tolist())))


def _sample_distances(
    H: int,
    W: int,
    block_bbox: tuple[int, int, int, int],
    counts: dict[str, int],
) -> dict[str, tuple[int, ...]] | None:
    r0, c0, r1, c1 = block_bbox
    pools = {
        "up": range(2, r0 + 1),
        "down": range(2, H - r1),
        "left": range(2, c0 + 1),
        "right": range(2, W - c1),
    }

    if any(len(list(pool)) < 3 for pool in pools.values()):
        return None

    for _ in range(120):
        dists = {
            direction: _choose_positions(pools[direction], counts[direction])
            for direction in _DIRECTIONS
        }
        nearest = {direction: min(vals) for direction, vals in dists.items()}
        farthest = {direction: max(vals) for direction, vals in dists.items()}
        if not any(
            counts[a] != counts[b] and nearest[a] == nearest[b]
            for a, b in itertools.combinations(_DIRECTIONS, 2)
        ):
            continue
        if not any(
            counts[a] == counts[b] and dists[a] != dists[b]
            for a, b in itertools.combinations(_DIRECTIONS, 2)
        ):
            continue
        if not any(
            len(vals) >= 2 and any(b - a > 1 for a, b in zip(vals, vals[1:]))
            for vals in dists.values()
        ):
            continue
        if not any(
            counts[a] != counts[b] and farthest[a] == farthest[b]
            for a, b in itertools.combinations(_DIRECTIONS, 2)
        ):
            continue
        return dists
    return None


def _render_input(
    H: int,
    W: int,
    block_bbox: tuple[int, int, int, int],
    dists: dict[str, tuple[int, ...]],
) -> np.ndarray:
    r0, c0, r1, c1 = block_bbox
    grid = np.zeros((H, W), dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREEN

    up_col = c0
    down_col = c1
    left_row = r0
    right_row = r1

    for dist in dists["up"]:
        grid[r0 - dist, up_col] = RED
    for dist in dists["down"]:
        grid[r1 + dist, down_col] = RED
    for dist in dists["left"]:
        grid[left_row, c0 - dist] = RED
    for dist in dists["right"]:
        grid[right_row, c1 + dist] = RED
    return grid


def _corner_witnesses(
    block_bbox: tuple[int, int, int, int],
    counts: dict[str, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = block_bbox
    witnesses: list[tuple[int, int]] = []
    tl = max(counts["up"], counts["left"])
    tr = max(counts["up"], counts["right"])
    bl = max(counts["down"], counts["left"])
    br = max(counts["down"], counts["right"])
    if tl > min(counts["up"], counts["left"]):
        witnesses.append((r0 - tl, c0 - tl))
    if tr > min(counts["up"], counts["right"]):
        witnesses.append((r0 - tr, c1 + tr))
    if bl > min(counts["down"], counts["left"]):
        witnesses.append((r1 + bl, c0 - bl))
    if br > min(counts["down"], counts["right"]):
        witnesses.append((r1 + br, c1 + br))
    return witnesses


def _validate_sample(inp: np.ndarray) -> bool:
    try:
        block_bbox = _find_block(inp)
        ray_info = _classify_rays(inp, block_bbox)
        counts = _counts_from_ray_info(ray_info)
        out = main(inp)
    except AssertionError:
        return False

    H, W = map(int, inp.shape)
    r0, c0, r1, c1 = block_bbox
    if (r0, c0) != (H // 2 - 1, W // 2 - 1):
        return False
    if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], np.full((2, 2), GREEN, dtype=int)):
        return False
    if int((out == RED).sum()) != 0:
        return False
    if set(map(int, np.unique(out))) - {BLACK, GREEN, YELLOW}:
        return False

    expected_mask = _frame_mask(inp.shape, block_bbox, counts)
    if not np.array_equal(out == YELLOW, expected_mask):
        return False
    if int((out == YELLOW).sum()) == 0:
        return False

    witnesses = _corner_witnesses(block_bbox, counts)
    if not witnesses:
        return False
    for r, c in witnesses:
        if int(out[r, c]) != YELLOW:
            return False

    dists = {direction: tuple(ray_info[direction]["dists"]) for direction in _DIRECTIONS}
    nearest = {direction: min(vals) for direction, vals in dists.items()}
    if not any(
        counts[a] != counts[b] and nearest[a] == nearest[b]
        for a, b in itertools.combinations(_DIRECTIONS, 2)
    ):
        return False
    if not any(
        counts[a] == counts[b] and dists[a] != dists[b]
        for a, b in itertools.combinations(_DIRECTIONS, 2)
    ):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = int(_RNG.choice(np.array(_DIM_CHOICES, dtype=int)))
    W = int(_RNG.choice(np.array(_DIM_CHOICES, dtype=int)))
    block_bbox = (H // 2 - 1, W // 2 - 1, H // 2, W // 2)
    counts = _sample_counts()
    dists = _sample_distances(H, W, block_bbox, counts)
    if dists is None:
        return None
    inp = _render_input(H, W, block_bbox, dists)
    return inp if _validate_sample(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample for puzzle 6")

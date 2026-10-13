from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

try:
    from arc_utils import paste
except Exception:
    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
    ) -> np.ndarray:
        h, w = patch.shape
        view = dst[r0 : r0 + h, c0 : c0 + w]
        if transparent is None:
            view[:] = patch
        else:
            mask = patch != int(transparent)
            view[mask] = patch[mask]
        return dst


BG = 0
GREY = 5
FRAME_BY_ROTATION = {0: 5, 1: 1, 2: 2, 3: 3}
SAMPLE_COLORS = (4, 6, 7, 8, 9)
GRID_MIN = 21
GRID_MAX = 28
MAX_GENERATION_ATTEMPTS = 400


def _rotate_cw(pattern: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(pattern, -int(quarter_turns))


def _rotation_lookup(pattern: np.ndarray) -> dict[bytes, int]:
    lookup = {_rotate_cw(pattern, k).tobytes(): k for k in range(4)}
    if len(lookup) != 4:
        raise ValueError("sample must have four distinct rotations")
    return lookup


def _draw_outer_frame(grid: np.ndarray, r: int, c: int, color: int) -> None:
    grid[r - 1, c - 1 : c + 4] = color
    grid[r + 3, c - 1 : c + 4] = color
    grid[r - 1 : r + 4, c - 1] = color
    grid[r - 1 : r + 4, c + 3] = color


def _sample_patch(sample: np.ndarray) -> np.ndarray:
    patch = np.zeros((4, 4), dtype=int)
    patch[:3, :3] = sample
    patch[:3, 3] = GREY
    patch[3, :3] = GREY
    return patch


def _locate_sample(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    grey_coords = np.argwhere(grid == GREY)
    if grey_coords.shape[0] != 6:
        raise ValueError("expected exactly six grey L-marker cells")

    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (4, 4):
        raise ValueError("grey marker must have a 4x4 bbox")

    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected_mask = np.zeros((4, 4), dtype=bool)
    expected_mask[:3, 3] = True
    expected_mask[3, :3] = True
    if not np.array_equal(local == GREY, expected_mask):
        raise ValueError("grey cells do not form the required L marker")

    sample = local[:3, :3].copy()
    if np.any(sample == BG) or np.any(sample == GREY):
        raise ValueError("sample must be a full 3x3 pattern with non-black non-grey colors")
    return (int(r0), int(c0)), sample


def _find_rotated_matches(
    grid: np.ndarray,
    sample_pos: tuple[int, int],
    sample: np.ndarray,
) -> dict[tuple[int, int], int]:
    lookup = _rotation_lookup(sample)
    matches: dict[tuple[int, int], int] = {}
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if (r, c) == sample_pos:
                continue
            rot = lookup.get(grid[r : r + 3, c : c + 3].tobytes())
            if rot is not None:
                matches[(r, c)] = int(rot)
    return matches


def _expected_output(grid: np.ndarray) -> np.ndarray:
    sample_pos, sample = _locate_sample(grid)
    matches = _find_rotated_matches(grid, sample_pos, sample)

    out = np.asarray(grid, dtype=int).copy()
    sr, sc = sample_pos
    out[sr : sr + 3, sc + 3] = BG
    out[sr + 3, sc : sc + 3] = BG

    for (r, c), rot in matches.items():
        if not (1 <= r <= grid.shape[0] - 4 and 1 <= c <= grid.shape[1] - 4):
            raise ValueError("matched copy must have a 1-cell border for its frame")
        _draw_outer_frame(out, r, c, FRAME_BY_ROTATION[rot])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find every rotated copy of the 3x3 sample and frame it with a color that encodes the rotation.

    Input structure:
    - Black is background.
    - A filled 3x3 sample sits near the upper-left beside a grey L marker: a 3-cell grey column on the sample's right and a 3-cell grey row below it.
    - Elsewhere, separated 3x3 copies of the sample appear, each equal to one of the sample's four rotations.

    Rule:
    - Read the sample from the inside of the grey L marker.
    - Search the whole grid, excluding the sample itself, for exact 3x3 matches to the sample rotated by 0, 90, 180, or 270 degrees clockwise.
    - Draw a 1-pixel-thick 5x5 frame around each matched copy using grey for 0 degrees, blue for 90 degrees, red for 180 degrees, and green for 270 degrees.
    - Remove the original grey L marker and keep the sample unchanged.

    Key insight:
    - The search is rotation-invariant, and the matched rotation is not used to transform the copy itself but to encode information through the frame color.
    """
    return _expected_output(np.asarray(grid, dtype=int))


def _make_sample(rng: random.Random) -> np.ndarray:
    colors = rng.sample(SAMPLE_COLORS, 2)
    while True:
        flat = [colors[0]] * rng.randint(3, 6) + [colors[1]] * 0
        flat += [colors[1]] * (9 - len(flat))
        rng.shuffle(flat)
        sample = np.array(flat, dtype=int).reshape(3, 3)
        if len(set(sample.reshape(-1))) != 2:
            continue
        try:
            _rotation_lookup(sample)
            return sample
        except ValueError:
            continue


def _make_histogram_distractor(sample: np.ndarray, rng: random.Random) -> np.ndarray:
    rotations = {rot.tobytes() for rot in (_rotate_cw(sample, k) for k in range(4))}
    flat = list(map(int, sample.reshape(-1)))
    while True:
        rng.shuffle(flat)
        distractor = np.array(flat, dtype=int).reshape(3, 3)
        if distractor.tobytes() not in rotations:
            return distractor


def _reserve_region(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(mask.shape[0], int(r1))
    cc1 = min(mask.shape[1], int(c1))
    mask[rr0:rr1, cc0:cc1] = True


def _try_place_3x3(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(1, grid.shape[0] - 3) for c in range(1, grid.shape[1] - 3)]
    rng.shuffle(candidates)
    for r, c in candidates:
        if reserved[max(0, r - 2) : min(grid.shape[0], r + 5), max(0, c - 2) : min(grid.shape[1], c + 5)].any():
            continue
        paste(grid, patch, r, c)
        _reserve_region(reserved, r - 2, c - 2, r + 5, c + 5)
        return (r, c)
    return None


def _build_attempt(rng: random.Random) -> tuple[np.ndarray, dict[tuple[int, int], int], list[tuple[int, int]]] | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    sample = _make_sample(rng)
    sample_r = rng.randint(0, 2)
    sample_c = rng.randint(0, 2)
    paste(grid, _sample_patch(sample), sample_r, sample_c, transparent=BG)
    _reserve_region(reserved, sample_r - 1, sample_c - 1, sample_r + 5, sample_c + 5)

    if rng.random() < 0.7:
        rotations = [0, 1, 2, 3]
    else:
        omitted = rng.choice([1, 2, 3])
        rotations = [rot for rot in [0, 1, 2, 3] if rot != omitted]
    rng.shuffle(rotations)

    placements: dict[tuple[int, int], int] = {}
    for rot in rotations:
        pos = _try_place_3x3(grid, reserved, _rotate_cw(sample, rot), rng)
        if pos is None:
            return None
        placements[pos] = rot

    distractor_count = rng.randint(1, 2)
    distractor_positions: list[tuple[int, int]] = []
    seen_distractors: set[bytes] = set()
    while len(distractor_positions) < distractor_count:
        distractor = _make_histogram_distractor(sample, rng)
        if distractor.tobytes() in seen_distractors:
            continue
        pos = _try_place_3x3(grid, reserved, distractor, rng)
        if pos is None:
            return None
        distractor_positions.append(pos)
        seen_distractors.add(distractor.tobytes())

    return grid, placements, distractor_positions


def _validate_attempt(
    grid: np.ndarray,
    placements: dict[tuple[int, int], int],
    distractor_positions: list[tuple[int, int]],
) -> bool:
    try:
        sample_pos, sample = _locate_sample(grid)
    except Exception:
        return False

    found = _find_rotated_matches(grid, sample_pos, sample)
    if found != placements:
        return False

    if 0 not in found.values():
        return False
    if not any(rot != 0 for rot in found.values()):
        return False
    if len(set(found.values())) < 3:
        return False

    unrotated_only = {
        (r, c)
        for (r, c), rot in found.items()
        if rot == 0
    }
    if len(unrotated_only) == len(found):
        return False

    target_hist = np.bincount(sample.reshape(-1), minlength=10)
    for r, c in distractor_positions:
        window = grid[r : r + 3, c : c + 3]
        if np.array_equal(window, sample):
            return False
        if window.tobytes() in _rotation_lookup(sample):
            return False
        if not np.array_equal(np.bincount(window.reshape(-1), minlength=10), target_hist):
            return False

    expected = grid.copy()
    sample_r, sample_c = sample_pos
    expected[sample_r : sample_r + 3, sample_c + 3] = BG
    expected[sample_r + 3, sample_c : sample_c + 3] = BG
    for (r, c), rot in placements.items():
        _draw_outer_frame(expected, r, c, FRAME_BY_ROTATION[rot])

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        built = _build_attempt(rng)
        if built is None:
            continue
        grid, placements, distractor_positions = built
        if _validate_attempt(grid, placements, distractor_positions):
            return grid
    raise RuntimeError("failed to generate a valid input")

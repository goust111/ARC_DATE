from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import Color, bbox, crop_bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BG = Color.BLACK
MARK = Color.GREEN
INPUT_COLORS = (Color.BLUE, Color.RED, Color.YELLOW, Color.ORANGE)
GRID_SIDES = (12, 14, 16)
MIN_TILE_SIDE = 4
MAX_GENERATION_ATTEMPTS = 500
QUADRANTS = ("ur", "ll", "lr")


def _expected_quadrants(tile: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ul": tile.copy(),
        "ur": np.rot90(tile, -1),
        "ll": np.rot90(tile, 1),
        "lr": np.rot90(tile, 2),
    }


def _source_to_quadrant(src: tuple[int, int], quadrant: str, n: int) -> tuple[int, int]:
    r, c = src
    if quadrant == "ur":
        return int(c), int(n - 1 - r)
    if quadrant == "ll":
        return int(n - 1 - c), int(r)
    if quadrant == "lr":
        return int(n - 1 - r), int(n - 1 - c)
    raise ValueError(f"unknown quadrant: {quadrant}")


def _different_value(expected: int) -> int:
    if expected == BG:
        return random.choice(INPUT_COLORS)

    if random.random() < 0.45:
        return BG
    choices = [color for color in INPUT_COLORS if color != expected]
    return random.choice(choices)


def _sample_reference_tile(n: int) -> np.ndarray | None:
    for _ in range(200):
        palette = random.sample(INPUT_COLORS, random.randint(2, len(INPUT_COLORS)))
        tile = np.zeros((n, n), dtype=int)

        top_count = random.randint(4, n)
        for c in random.sample(range(n), top_count):
            tile[0, c] = random.choice(palette)

        fill_prob = random.uniform(0.35, 0.62)
        for r in range(1, n):
            for c in range(n):
                if random.random() < fill_prob:
                    tile[r, c] = random.choice(palette)

        nonzero = int(np.count_nonzero(tile))
        if nonzero < 8:
            continue
        if int(np.count_nonzero(tile == BG)) < 3:
            continue
        if int(np.count_nonzero(tile[1:, :] != BG)) < 4:
            continue
        if len(set(map(int, tile[tile != BG].tolist()))) < 2:
            continue
        if any(np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3)):
            continue
        return tile
    return None


def _build_input_from_tile(tile: np.ndarray, height: int, width: int) -> tuple[np.ndarray, dict[str, object]] | None:
    n = int(tile.shape[0])
    expected = _expected_quadrants(tile)
    actual = {name: quad.copy() for name, quad in expected.items()}

    all_src = [(r, c) for r in range(n) for c in range(n)]
    shared_src = random.choice(all_src)
    shared_wrong = _different_value(int(tile[shared_src]))

    defect_srcs: dict[str, list[tuple[int, int]]] = {}
    counts = {name: random.randint(1, 3) for name in QUADRANTS}
    for name in QUADRANTS:
        extras = random.sample([cell for cell in all_src if cell != shared_src], counts[name] - 1)
        defect_srcs[name] = [shared_src, *extras]

    for name in QUADRANTS:
        quad = actual[name]
        for idx, src in enumerate(defect_srcs[name]):
            rr, cc = _source_to_quadrant(src, name, n)
            if idx == 0:
                quad[rr, cc] = shared_wrong
            else:
                quad[rr, cc] = _different_value(int(quad[rr, cc]))

    motif = np.zeros((2 * n, 2 * n), dtype=int)
    motif[:n, :n] = actual["ul"]
    motif[:n, n:] = actual["ur"]
    motif[n:, :n] = actual["ll"]
    motif[n:, n:] = actual["lr"]

    side = 2 * n
    if side >= min(height, width):
        return None
    r0 = (height - side) // 2
    c0 = (width - side) // 2

    grid = np.zeros((height, width), dtype=int)
    grid[r0 : r0 + side, c0 : c0 + side] = motif
    meta = {
        "tile": tile,
        "box": (r0, c0, r0 + side - 1, c0 + side - 1),
        "counts": counts,
    }
    return grid, meta


def _manual_output(inp: np.ndarray) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    box = bbox(inp != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    motif = crop_bbox(inp, box)
    side = int(motif.shape[0])
    assert side == motif.shape[1] and side % 2 == 0
    half = side // 2
    ul = motif[:half, :half]
    expectations = {
        "ur": np.rot90(ul, -1),
        "ll": np.rot90(ul, 1),
        "lr": np.rot90(ul, 2),
    }
    slices = {
        "ur": (slice(0, half), slice(half, side)),
        "ll": (slice(half, side), slice(0, half)),
        "lr": (slice(half, side), slice(half, side)),
    }
    out_motif = out[r0 : r1 + 1, c0 : c1 + 1]
    for name, (rs, cs) in slices.items():
        block = out_motif[rs, cs]
        block[block != expectations[name]] = MARK
    return out


def _aligned_majority_trap(motif: np.ndarray) -> bool:
    side = int(motif.shape[0])
    half = side // 2
    ul = motif[:half, :half]
    aligned = np.stack(
        [
            np.rot90(motif[:half, half:], 1),
            np.rot90(motif[half:, :half], -1),
            np.rot90(motif[half:, half:], 2),
        ],
        axis=0,
    )
    wrong_majority = (aligned[0] == aligned[1]) & (aligned[1] == aligned[2]) & (aligned[0] != ul)
    return bool(wrong_majority.any())


def _validate_generated(inp: np.ndarray, meta: dict[str, object]) -> bool:
    tile = np.asarray(meta["tile"], dtype=int)
    box = bbox(inp != BG)
    if box != meta["box"]:
        return False
    if np.any(inp == MARK):
        return False

    motif = crop_bbox(inp, box)
    side = int(motif.shape[0])
    if side != motif.shape[1] or side % 2 != 0:
        return False
    half = side // 2
    if half != tile.shape[0]:
        return False

    if not motif[0, :].any() or not motif[-1, :].any() or not motif[:, 0].any() or not motif[:, -1].any():
        return False

    ul = motif[:half, :half]
    ur = motif[:half, half:]
    ll = motif[half:, :half]
    lr = motif[half:, half:]
    if not np.array_equal(ul, tile):
        return False
    if any(np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3)):
        return False

    expected = {
        "ur": np.rot90(tile, -1),
        "ll": np.rot90(tile, 1),
        "lr": np.rot90(tile, 2),
    }
    actual = {"ur": ur, "ll": ll, "lr": lr}
    counts = meta["counts"]
    for name in QUADRANTS:
        mismatches = int(np.count_nonzero(actual[name] != expected[name]))
        if mismatches != counts[name] or not (1 <= mismatches <= 3):
            return False

    if not _aligned_majority_trap(motif):
        return False

    expected_out = _manual_output(inp)
    actual_out = main(inp)
    if not np.array_equal(actual_out, expected_out):
        return False

    for name, quad in actual.items():
        mismatch = quad != expected[name]
        out_quad = crop_bbox(actual_out, box)
        if name == "ur":
            out_block = out_quad[:half, half:]
        elif name == "ll":
            out_block = out_quad[half:, :half]
        else:
            out_block = out_quad[half:, half:]
        if np.any(out_block[mismatch] != MARK):
            return False
        if np.any(out_block[~mismatch] != quad[~mismatch]):
            return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the defect pixels in a near-rotationally-symmetric four-quadrant motif.

    Input structure:
    - The background is black, and one even-sided square motif sits inside it.
    - The motif is split into four equal square quadrants.
    - The upper-left quadrant is the clean reference tile; the other three are rotated copies with a few defects.

    Rule:
    - Extract the upper-left quadrant as the reference tile.
    - Compute the expected upper-right, lower-left, and lower-right quadrants by rotating that tile clockwise 90 degrees, counterclockwise 90 degrees, and 180 degrees.
    - Compare each non-reference quadrant cell-by-cell against its expected rotation.
    - Keep matching cells unchanged and recolor every mismatching cell green, while leaving the reference quadrant and the outside background unchanged.

    Key insight:
    - The missing information is not stored directly in the defective quadrants; it must be regenerated by rotating the clean upper-left tile and then used as the comparison baseline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    box = bbox(g != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    motif = crop_bbox(g, box)
    side_h, side_w = motif.shape
    assert side_h == side_w and side_h % 2 == 0
    half = side_h // 2

    ul = motif[:half, :half]
    expectations = {
        "ur": np.rot90(ul, -1),
        "ll": np.rot90(ul, 1),
        "lr": np.rot90(ul, 2),
    }
    slices = {
        "ur": (slice(0, half), slice(half, side_h)),
        "ll": (slice(half, side_h), slice(0, half)),
        "lr": (slice(half, side_h), slice(half, side_h)),
    }

    out = g.copy()
    out_motif = out[r0 : r1 + 1, c0 : c1 + 1]
    for name, (rs, cs) in slices.items():
        block = out_motif[rs, cs]
        block[block != expectations[name]] = MARK

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.choice(GRID_SIDES)
        width = random.choice(GRID_SIDES)
        max_tile = min(height, width) // 2 - 1
        if max_tile < MIN_TILE_SIDE:
            continue

        n = random.randint(MIN_TILE_SIDE, max_tile)
        tile = _sample_reference_tile(n)
        if tile is None:
            continue

        built = _build_input_from_tile(tile, height, width)
        if built is None:
            continue
        inp, meta = built
        if _validate_generated(inp, meta):
            return inp

    raise RuntimeError("failed to generate a valid rotational-defect input")

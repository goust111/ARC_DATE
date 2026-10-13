from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg  # noqa: E402


BLUE = int(Color.BLUE)
BLACK = int(Color.BLACK)
SEED_COLORS = np.array([int(Color.RED), int(Color.GREEN), int(Color.YELLOW)], dtype=int)
GEN_RETRY_LIMIT = 400
GEN_SIDES = ("top", "bottom", "left", "right")


def _gap_side_and_direction(gap: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[str, tuple[int, int]]:
    r, c = map(int, gap)
    r0, c0, r1, c1 = map(int, box)
    if r == r0:
        return "top", (-1, 0)
    if r == r1:
        return "bottom", (1, 0)
    if c == c0:
        return "left", (0, -1)
    assert c == c1
    return "right", (0, 1)


def _frame_gap(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    missing: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r not in (r0, r1) and c not in (c0, c1):
                continue
            if grid[r, c] != BLUE:
                assert (r in (r0, r1)) ^ (c in (c0, c1)), "gap must be non-corner"
                missing.append((r, c))
    assert len(missing) == 1, f"expected one border gap, got {missing}"
    return missing[0]


def _tile_horizontal(seed: np.ndarray, width: int) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    cols = np.arange(int(width)) % seed.shape[1]
    return seed[:, cols]


def _mirror_vertical(seed: np.ndarray, height: int) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    unit = np.vstack([seed, seed[-2::-1]]) if seed.shape[0] > 1 else seed.copy()
    rows = np.arange(int(height)) % unit.shape[0]
    return unit[rows, :]


def _fill_interior(interior: np.ndarray, side: str) -> np.ndarray:
    interior = np.asarray(interior, dtype=int)
    rows, cols = interior.shape
    nz = np.argwhere(interior != BLACK)
    assert nz.size > 0, "seed must be present"
    if side in ("left", "right"):
        seed_w = int(nz[:, 1].max()) + 1
        assert seed_w < cols, "periodic branch needs black space to the right"
        assert np.all(interior[:, seed_w:] == BLACK)
        seed = interior[:, :seed_w]
        return _tile_horizontal(seed, cols)

    seed_h = int(nz[:, 0].max()) + 1
    assert seed_h < rows, "mirror branch needs black space below"
    assert np.all(interior[seed_h:, :] == BLACK)
    seed = interior[:seed_h, :]
    return _mirror_vertical(seed, rows)


def _fill_interior_relaxed(interior: np.ndarray, side: str) -> np.ndarray:
    interior = np.asarray(interior, dtype=int)
    rows, cols = interior.shape
    nz = np.argwhere(interior != BLACK)
    assert nz.size > 0
    if side in ("left", "right"):
        seed_w = int(nz[:, 1].max()) + 1
        seed = interior[:, :seed_w]
        return seed.copy() if seed_w == cols else _tile_horizontal(seed, cols)
    seed_h = int(nz[:, 0].max()) + 1
    seed = interior[:seed_h, :]
    return seed.copy() if seed_h == rows else _mirror_vertical(seed, rows)


def _dominant_nonblack(grid: np.ndarray) -> int:
    counts: Counter[int] = Counter()
    first_pos: dict[int, int] = {}
    idx = 0
    for value in np.asarray(grid, dtype=int).flat:
        value = int(value)
        if value == BLACK:
            idx += 1
            continue
        counts[value] += 1
        if value not in first_pos:
            first_pos[value] = idx
        idx += 1
    assert counts, "expected non-black transformed interior"
    return min(counts, key=lambda color: (-counts[color], first_pos[color]))


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames = find_components(
        np.asarray(grid, dtype=int),
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    parsed: list[dict] = []
    for frame in sorted(frames, key=lambda item: item["bbox"]):
        box = tuple(map(int, frame["bbox"]))
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 4 and c1 - c0 >= 4
        coords = {tuple(map(int, rc)) for rc in frame["coords"]}
        for r, c in coords:
            assert r in (r0, r1) or c in (c0, c1), "blue pixels must stay on the border"
        gap = _frame_gap(grid, box)
        side, direction = _gap_side_and_direction(gap, box)
        interior = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
        parsed.append(
            {
                "bbox": box,
                "gap": gap,
                "side": side,
                "direction": direction,
                "interior": interior,
            }
        )
    assert parsed
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each blue frame by a gap-oriented seed transform, then shoot an outward ray in the transformed majority color.

    Input structure:
    - A black background contains 2 to 4 blue one-cell-thick hollow rectangular frames.
    - Each frame has exactly one missing non-corner border cell, and a small red/green/yellow seed occupies the interior top-left corner region while the rest of the interior is black.

    Rule:
    - Detect each frame's unique gap and classify whether it lies on a horizontal side or a vertical side.
    - If the gap is on the left or right side, treat the non-black interior prefix columns as a seed and repeat them horizontally across the whole interior.
    - If the gap is on the top or bottom side, treat the non-black interior prefix rows as a seed, append its vertical mirror across the seed's bottom row, and repeat that taller unit downward across the whole interior.
    - For each transformed interior, choose the most frequent non-black color, breaking ties by earliest row-major appearance inside that interior, and paint the gap cell plus the outward ray from that gap with the chosen color until the first non-black obstacle or the grid edge.
    - Keep every original blue border pixel unchanged.

    Key insight:
    - The missing-border side controls two linked consequences at once: which geometric completion rule fills the frame interior, and which direction the resulting majority-color ray exits the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK

    out = g.copy()
    rays: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for frame in _parse_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        filled = _fill_interior(frame["interior"], frame["side"])
        out[r0 + 1 : r1, c0 + 1 : c1] = filled
        rays.append((frame["gap"], frame["direction"], _dominant_nonblack(filled)))

    blocked = out != BLACK
    h, w = out.shape
    for (r, c), (dr, dc), color in rays:
        rr, cc = int(r), int(c)
        while 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc]:
            out[rr, cc] = int(color)
            blocked[rr, cc] = True
            rr += int(dr)
            cc += int(dc)
    return out


def _seed_main_color(seed: np.ndarray) -> int:
    return _dominant_nonblack(seed)


def _choose_side_set(rng: np.random.Generator) -> list[str]:
    count = int(rng.integers(2, 5))
    for _ in range(100):
        sides = list(rng.choice(GEN_SIDES, size=count, replace=False))
        if any(side in ("top", "bottom") for side in sides) and any(side in ("left", "right") for side in sides):
            return sides
    raise RuntimeError("failed to sample sides")


def _frame_shape_for_side(rng: np.random.Generator, side: str) -> tuple[int, int]:
    if side in ("left", "right"):
        outer_h = int(rng.integers(5, 8))
        outer_w = int(rng.integers(6, 8))
    else:
        outer_h = int(rng.integers(6, 8))
        outer_w = int(rng.integers(5, 8))
    return outer_h, outer_w


def _slot_bbox(side: str, shape: tuple[int, int], canvas_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    rows, cols = map(int, canvas_shape)
    if side == "top":
        r0 = 2
        c0 = cols // 2 - w // 2
    elif side == "bottom":
        r0 = rows - h - 2
        c0 = cols // 2 - w // 2
    elif side == "left":
        r0 = rows // 2 - h // 2
        c0 = 1
    else:
        r0 = rows // 2 - h // 2
        c0 = cols - w - 1
    return r0, c0, r0 + h - 1, c0 + w - 1


def _blank_gap_between(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _boxes_separated(boxes: list[tuple[int, int, int, int]], min_gap: int = 2) -> bool:
    for i, a in enumerate(boxes):
        ar0, ac0, ar1, ac1 = a
        for b in boxes[i + 1 :]:
            br0, bc0, br1, bc1 = b
            row_gap = _blank_gap_between(ar0, ar1, br0, br1)
            col_gap = _blank_gap_between(ac0, ac1, bc0, bc1)
            if row_gap == 0 and col_gap < min_gap:
                return False
            if col_gap == 0 and row_gap < min_gap:
                return False
            if row_gap > 0 and col_gap > 0 and (row_gap < min_gap or col_gap < min_gap):
                return False
    return True


def _vertical_fill_from_seed(seed: np.ndarray, inner_width: int) -> np.ndarray:
    return _tile_horizontal(seed, inner_width)


def _horizontal_fill_from_seed(seed: np.ndarray, inner_height: int) -> np.ndarray:
    return _mirror_vertical(seed, inner_height)


def _sample_vertical_seed(rng: np.random.Generator, inner_h: int, inner_w: int) -> np.ndarray:
    seed_w = int(rng.integers(2, min(3, inner_w - 1) + 1))
    color_count = int(rng.integers(1, 3))
    palette = rng.choice(SEED_COLORS, size=color_count, replace=False)
    values = np.array([BLACK, *map(int, palette)], dtype=int)
    probs = np.array([0.35, *([0.65 / color_count] * color_count)], dtype=float)
    for _ in range(200):
        seed = rng.choice(values, size=(inner_h, seed_w), p=probs)
        if not np.any(seed):
            continue
        if np.all(seed[:, -1] == BLACK):
            continue
        if len(np.unique(seed[seed != BLACK])) == 0:
            continue
        full = _vertical_fill_from_seed(seed, inner_w)
        shorter = _vertical_fill_from_seed(seed[:, :-1], inner_w) if seed_w > 2 else None
        if shorter is not None and np.array_equal(full, shorter):
            continue
        return seed
    raise RuntimeError("failed to sample vertical seed")


def _sample_horizontal_seed(
    rng: np.random.Generator,
    inner_h: int,
    inner_w: int,
    *,
    require_main_shift: bool,
) -> np.ndarray:
    seed_h = int(rng.integers(2, min(3, inner_h - 1) + 1))
    color_count = 2 if require_main_shift else int(rng.integers(1, 3))
    palette = rng.choice(SEED_COLORS, size=color_count, replace=False)
    values = np.array([BLACK, *map(int, palette)], dtype=int)
    probs = np.array([0.35, *([0.65 / color_count] * color_count)], dtype=float)
    for _ in range(400):
        seed = rng.choice(values, size=(seed_h, inner_w), p=probs)
        if not np.any(seed):
            continue
        if np.all(seed[-1, :] == BLACK):
            continue
        full = _horizontal_fill_from_seed(seed, inner_h)
        shorter = _horizontal_fill_from_seed(seed[:-1, :], inner_h) if seed_h > 2 else None
        if shorter is not None and np.array_equal(full, shorter):
            continue
        if require_main_shift and _seed_main_color(seed) == _dominant_nonblack(full):
            continue
        return seed
    raise RuntimeError("failed to sample horizontal seed")


def _render_frame(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    side: str,
    seed: np.ndarray,
    gap_index: int,
) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    if side == "top":
        grid[r0, c0 + 1 + gap_index] = BLACK
    elif side == "bottom":
        grid[r1, c0 + 1 + gap_index] = BLACK
    elif side == "left":
        grid[r0 + 1 + gap_index, c0] = BLACK
    else:
        grid[r0 + 1 + gap_index, c1] = BLACK

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner[:] = BLACK
    seed_h, seed_w = seed.shape
    inner[:seed_h, :seed_w] = seed


def _ray_cells(box: tuple[int, int, int, int], side: str, gap_index: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    rows, cols = map(int, shape)
    if side == "top":
        r, c = r0, c0 + 1 + gap_index
        dr, dc = -1, 0
    elif side == "bottom":
        r, c = r1, c0 + 1 + gap_index
        dr, dc = 1, 0
    elif side == "left":
        r, c = r0 + 1 + gap_index, c0
        dr, dc = 0, -1
    else:
        r, c = r0 + 1 + gap_index, c1
        dr, dc = 0, 1

    cells: list[tuple[int, int]] = []
    while 0 <= r < rows and 0 <= c < cols:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _frame_spec(
    rng: np.random.Generator,
    side: str,
    box: tuple[int, int, int, int],
    *,
    require_main_shift: bool,
) -> dict:
    r0, c0, r1, c1 = map(int, box)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if side in ("left", "right"):
        seed = _sample_vertical_seed(rng, inner_h, inner_w)
        gap_index = int(rng.integers(0, inner_h))
    else:
        seed = _sample_horizontal_seed(rng, inner_h, inner_w, require_main_shift=require_main_shift)
        gap_index = int(rng.integers(0, inner_w))
    return {"side": side, "bbox": box, "seed": seed, "gap_index": gap_index}


def _wrong_branch_fill(interior: np.ndarray, side: str) -> np.ndarray:
    return _fill_interior_relaxed(interior, "top" if side in ("left", "right") else "left")


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _parse_frames(grid)
    except Exception:
        return False
    if not (2 <= len(frames) <= 4):
        return False
    if not any(frame["side"] in ("top", "bottom") for frame in frames):
        return False
    if not any(frame["side"] in ("left", "right") for frame in frames):
        return False

    boxes = [frame["bbox"] for frame in frames]
    if not _boxes_separated(boxes, min_gap=2):
        return False

    ray_sets: list[set[tuple[int, int]]] = []
    saw_main_shift = False
    for frame in frames:
        interior = frame["interior"]
        side = frame["side"]
        if side in ("left", "right"):
            nz = np.argwhere(interior != BLACK)
            seed_w = int(nz[:, 1].max()) + 1
            if seed_w >= interior.shape[1]:
                return False
            if not np.all(interior[:, seed_w:] == BLACK):
                return False
            seed = interior[:, :seed_w]
            if seed_w > 2 and np.array_equal(_vertical_fill_from_seed(seed, interior.shape[1]), _vertical_fill_from_seed(seed[:, :-1], interior.shape[1])):
                return False
        else:
            nz = np.argwhere(interior != BLACK)
            seed_h = int(nz[:, 0].max()) + 1
            if seed_h >= interior.shape[0]:
                return False
            if not np.all(interior[seed_h:, :] == BLACK):
                return False
            seed = interior[:seed_h, :]
            if seed_h > 2 and np.array_equal(_horizontal_fill_from_seed(seed, interior.shape[0]), _horizontal_fill_from_seed(seed[:-1, :], interior.shape[0])):
                return False
            if _seed_main_color(seed) != _dominant_nonblack(_horizontal_fill_from_seed(seed, interior.shape[0])):
                saw_main_shift = True

        correct = _fill_interior(interior, side)
        wrong = _wrong_branch_fill(interior, side)
        if np.array_equal(correct, wrong):
            return False

        ray = set(_ray_cells(frame["bbox"], side, (frame["gap"][1] - frame["bbox"][1] - 1) if side in ("top", "bottom") else (frame["gap"][0] - frame["bbox"][0] - 1), grid.shape))
        for r, c in ray:
            if (r, c) == frame["gap"]:
                continue
            if grid[r, c] != BLACK:
                return False
        if any(ray & other for other in ray_sets):
            return False
        ray_sets.append(ray)

    if not saw_main_shift:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape


def _gap_index_from_side(box: tuple[int, int, int, int], side: str, gap: tuple[int, int]) -> int:
    r0, c0, _, _ = map(int, box)
    r, c = map(int, gap)
    return int(c - c0 - 1) if side in ("top", "bottom") else int(r - r0 - 1)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    rows = int(rng.integers(28, 30))
    cols = int(rng.integers(28, 30))
    sides = _choose_side_set(rng)

    boxes = [_slot_bbox(side, _frame_shape_for_side(rng, side), (rows, cols)) for side in sides]
    if not _boxes_separated(boxes, min_gap=2):
        return None

    horizontal_sides = [side for side in sides if side in ("top", "bottom")]
    require_shift_side = str(rng.choice(horizontal_sides))

    specs: list[dict] = []
    try:
        for side, box in zip(sides, boxes):
            specs.append(_frame_spec(rng, side, box, require_main_shift=(side == require_shift_side)))
    except RuntimeError:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    for spec in specs:
        _render_frame(grid, spec["bbox"], spec["side"], spec["seed"], spec["gap_index"])

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-directed frame puzzle")

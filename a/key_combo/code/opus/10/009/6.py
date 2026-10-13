from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox, crop_bbox, paste
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        h, w = patch.shape
        rr0 = int(r0)
        cc0 = int(c0)
        rr1 = rr0 + h
        cc1 = cc0 + w
        if not clip and (rr0 < 0 or cc0 < 0 or rr1 > dst.shape[0] or cc1 > dst.shape[1]):
            raise ValueError("paste out of bounds")
        sr0 = max(0, -rr0)
        sc0 = max(0, -cc0)
        dr0 = max(0, rr0)
        dc0 = max(0, cc0)
        dr1 = min(dst.shape[0], rr1)
        dc1 = min(dst.shape[1], cc1)
        if dr0 >= dr1 or dc0 >= dc1:
            return dst
        view = patch[sr0 : sr0 + (dr1 - dr0), sc0 : sc0 + (dc1 - dc0)]
        if transparent is None:
            dst[dr0:dr1, dc0:dc1] = view
        else:
            mask = view != int(transparent)
            dst_view = dst[dr0:dr1, dc0:dc1]
            dst_view[mask] = view[mask]
        return dst


BG = Color.BLACK
CONTROL_COLORS = [
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
SEED_COLORS = [Color.BLUE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400


def _extract_seed_and_controls(grid: np.ndarray) -> tuple[np.ndarray, list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    active_rows = np.where(np.any(g != BG, axis=1))[0]
    assert active_rows.size >= 2, "expected a seed area and a control row"
    control_row = int(active_rows[-1])
    controls = [int(v) for v in g[control_row] if int(v) != BG]
    assert 3 <= len(controls) <= 7, "control length must be 3..7"
    assert len(set(controls)) == len(controls), "control colors must be distinct"

    upper_mask = np.zeros_like(g, dtype=bool)
    if control_row > 0:
        upper_mask[:control_row, :] = g[:control_row, :] != BG
    box = bbox(upper_mask)
    assert box is not None, "expected a seed sprite above the control row"
    r0, c0, r1, c1 = box
    seed = crop_bbox(g, box)
    assert seed.shape[0] == seed.shape[1], "seed bbox must be square"
    n = int(seed.shape[0])
    assert 3 <= n <= 5, "seed size must be 3..5"
    assert not np.any(upper_mask[:r0, :]), "unexpected pixels above the seed bbox"
    assert not np.any(upper_mask[r1 + 1 :, :]), "unexpected pixels below the seed bbox"
    assert not np.any(upper_mask[r0 : r1 + 1, :c0]), "unexpected pixels left of the seed bbox"
    assert not np.any(upper_mask[r0 : r1 + 1, c1 + 1 :]), "unexpected pixels right of the seed bbox"

    seed_colors = sorted({int(v) for v in seed.flat if int(v) != BG})
    assert len(seed_colors) == 1, "seed must be monochrome"
    seed_count = int(np.count_nonzero(seed))
    assert seed_count == len(controls), "seed foreground count must equal control length"
    return seed, controls, seed_colors[0]


def _build_output(seed: np.ndarray, controls: list[int], seed_color: int) -> np.ndarray:
    mask = seed != BG
    n = int(seed.shape[0])
    coords = [(r, c) for r in range(n) for c in range(n) if bool(mask[r, c])]
    assert len(coords) == len(controls)

    out = np.zeros((n * n, n * n), dtype=int)
    stamp = np.where(mask, seed_color, BG)
    for r, c in coords:
        paste(out, stamp, r * n, c * n, transparent=BG)
    for idx, (r, c) in enumerate(coords):
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[mask] = int(controls[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome seed into self-similar copies and recolor each copy by a bottom control sequence.

    Input structure:
    - The background is black.
    - Above the bottom row there is one compact square seed sprite of size N x N, with N in 3..5, using exactly one non-black color.
    - The bottom-most non-black row is a control sequence: its non-black pixels read left-to-right give M distinct colors, where M equals the seed's foreground-pixel count.

    Rule:
    - Crop the seed by its tight square bounding box and record the positions of its foreground pixels.
    - Create an (N*N) x (N*N) black output partitioned into N x N blocks of size N x N.
    - For every foreground seed pixel, stamp the full seed pattern into the corresponding output block.
    - Visit the occupied blocks in reading order (top-to-bottom, then left-to-right) and recolor the foreground cells of the i-th stamped copy with the i-th control color.

    Key insight:
    - The same seed supplies both the placement mask for where copies appear and the internal shape of every copy, while the control row length is implicitly tied to the seed's foreground count.
    """
    seed, controls, seed_color = _extract_seed_and_controls(np.asarray(grid, dtype=int))
    return _build_output(seed, controls, seed_color)


def _boundary_contacts(mask: np.ndarray) -> int:
    n = int(mask.shape[0])
    coords = {(r, c) for r in range(n) for c in range(n) if bool(mask[r, c])}
    horiz_pairs = sum((r, c + 1) in coords for r, c in coords)
    vert_pairs = sum((r + 1, c) in coords for r, c in coords)
    horiz_touch = int(np.count_nonzero(mask[:, 0] & mask[:, -1]))
    vert_touch = int(np.count_nonzero(mask[0, :] & mask[-1, :]))
    return int(horiz_pairs) * horiz_touch + int(vert_pairs) * vert_touch


def _has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_connected(cells: set[tuple[int, int]], n: int, target: int) -> None:
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return
        cells.add(random.choice(frontier))


def _make_seed_mask(n: int, fg_count: int) -> np.ndarray | None:
    mode = random.randrange(3)
    for _ in range(120):
        cells = {
            (0, random.randrange(n)),
            (n - 1, random.randrange(n)),
            (random.randrange(n), 0),
            (random.randrange(n), n - 1),
        }
        if len(cells) > fg_count:
            continue

        if mode == 0:
            _grow_connected(cells, n, fg_count)
        elif mode == 1:
            anchors = list(cells)
            random.shuffle(anchors)
            split = max(1, fg_count // 2)
            group_a = {anchors[0]}
            group_b = {anchors[-1]}
            while len(group_a) < min(split, fg_count):
                _grow_connected(group_a, n, len(group_a) + 1)
                if len(group_a) + len(group_b) > fg_count:
                    break
            while len(group_a) + len(group_b) < fg_count:
                _grow_connected(group_b, n, len(group_b) + 1)
            cells |= group_a | group_b
        else:
            bag = [(r, c) for r in range(n) for c in range(n) if (r, c) not in cells]
            random.shuffle(bag)
            border_first = sorted(
                bag,
                key=lambda p: (
                    p[0] not in (0, n - 1) and p[1] not in (0, n - 1),
                    random.random(),
                ),
            )
            for pt in border_first[: fg_count - len(cells)]:
                cells.add(pt)

        if len(cells) != fg_count:
            continue

        mask = np.zeros((n, n), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not _has_full_bbox(mask):
            continue
        if np.unique(np.argwhere(mask)[:, 0]).size < 2 or np.unique(np.argwhere(mask)[:, 1]).size < 2:
            continue
        if int(mask.sum()) * int(mask.sum()) >= int(0.47 * (n ** 4)):
            continue
        if _boundary_contacts(mask) > max(1, fg_count // 2):
            continue
        if np.array_equal(mask, mask.astype(int).all(axis=0, keepdims=True).repeat(n, axis=0)):
            continue
        return mask
    return None


def _control_run(width: int, length: int) -> tuple[int, int]:
    start = random.randint(0, width - length)
    return start, start + length


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        seed, controls, _seed_color = _extract_seed_and_controls(inp)
    except Exception:
        return False

    n = int(seed.shape[0])
    mask = seed != BG
    coords = [(r, c) for r in range(n) for c in range(n) if bool(mask[r, c])]
    if out.shape != (n * n, n * n):
        return False
    if len(coords) != len(controls):
        return False
    if set(np.unique(out)) - {BG} != set(controls):
        return False
    if int(np.count_nonzero(out)) != len(coords) * len(coords):
        return False

    occupied_blocks = 0
    for r in range(n):
        for c in range(n):
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            if (r, c) in coords:
                idx = coords.index((r, c))
                expected = np.where(mask, controls[idx], BG)
                if not np.array_equal(block, expected):
                    return False
                occupied_blocks += 1
            elif np.any(block != BG):
                return False
    return occupied_blocks == len(coords)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    max_fg = 6 if n == 3 else 7
    fg_count = random.randint(3, min(max_fg, n * n - 1))
    seed_mask = _make_seed_mask(n, fg_count)
    if seed_mask is None:
        return None

    height = random.randint(max(n + 6, 12), 22)
    width = random.randint(max(n + 4, fg_count + 4, 12), 24)
    max_top = height - n - 4
    if max_top < 0:
        return None
    seed_top = random.randint(0, max_top)
    seed_left = random.randint(0, width - n)
    control_left, control_right = _control_run(width, fg_count)
    seed_color = random.choice(SEED_COLORS)
    control_palette = [c for c in CONTROL_COLORS if c != seed_color]
    controls = random.sample(control_palette, fg_count)

    grid = np.zeros((height, width), dtype=int)
    grid[seed_top : seed_top + n, seed_left : seed_left + n][seed_mask] = seed_color
    grid[-1, control_left:control_right] = controls

    try:
        out = main(grid)
    except Exception:
        return None

    if max(out.shape) > 30:
        return None
    if not _validate_semantics(grid, out):
        return None

    active_rows = np.where(np.any(grid != BG, axis=1))[0]
    if active_rows.size != n + 1:
        return None
    if int(np.count_nonzero(grid[-1])) != fg_count:
        return None
    if np.count_nonzero(grid[:-1]) != fg_count:
        return None
    if int(np.count_nonzero(seed_mask[1:-1, 1:-1])) == 0 and fg_count <= 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

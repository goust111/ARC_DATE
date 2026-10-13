import random
from collections import Counter

import numpy as np


BG = 0
MIRROR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (15, 19)
GRID_W_RANGE = (15, 19)
STRIPE_COUNT_RANGE = (2, 4)
STRIPE_LEN_RANGE = (3, 6)
GAP_RANGE = (1, 4)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _reflect_dir(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    dr, dc = map(int, direction)
    if mirror_kind == "\\":
        return (dc, dr)
    return (-dc, -dr)


def _mirror_segment(start: tuple[int, int], length: int, mirror_kind: str) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    if mirror_kind == "\\":
        return [(r0 + i, c0 + i) for i in range(length)]
    return [(r0 + i, c0 - i) for i in range(length)]


def _find_mirror(grid: np.ndarray, bg: int) -> tuple[int, str, list[tuple[int, int]]]:
    candidates: list[tuple[int, str, list[tuple[int, int]]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
        if len(pts) < 3:
            continue

        if len({r - c for r, c in pts}) == 1:
            ordered = sorted(pts)
            if all(
                ordered[i + 1] == (ordered[i][0] + 1, ordered[i][1] + 1)
                for i in range(len(ordered) - 1)
            ):
                candidates.append((int(color), "\\", ordered))

        if len({r + c for r, c in pts}) == 1:
            ordered = sorted(pts)
            if all(
                ordered[i + 1] == (ordered[i][0] + 1, ordered[i][1] - 1)
                for i in range(len(ordered) - 1)
            ):
                candidates.append((int(color), "/", ordered))

    assert len(candidates) == 1, "expected exactly one diagonal mirror segment"
    return candidates[0]


def _parse_stripe_object(coords: list[tuple[int, int]]) -> dict:
    pts = [(int(r), int(c)) for r, c in coords]
    candidates: list[dict] = []

    rows = sorted({r for r, _c in pts})
    for row in rows:
        cols = sorted(c for r, c in pts if r == row)
        if len(cols) < 2:
            continue
        if cols != list(range(cols[0], cols[-1] + 1)):
            continue
        extras = [(r, c) for r, c in pts if r != row]
        if len(extras) != 1:
            continue
        er, ec = extras[0]
        if abs(er - row) != 1:
            continue
        if ec == cols[0]:
            candidates.append(
                {
                    "axis": "h",
                    "far": (int(row), int(cols[0])),
                    "near": (int(row), int(cols[-1])),
                    "direction": (0, 1),
                    "side": (int(er - row), 0),
                }
            )
        if ec == cols[-1]:
            candidates.append(
                {
                    "axis": "h",
                    "far": (int(row), int(cols[-1])),
                    "near": (int(row), int(cols[0])),
                    "direction": (0, -1),
                    "side": (int(er - row), 0),
                }
            )

    cols_unique = sorted({c for _r, c in pts})
    for col in cols_unique:
        rows_here = sorted(r for r, c in pts if c == col)
        if len(rows_here) < 2:
            continue
        if rows_here != list(range(rows_here[0], rows_here[-1] + 1)):
            continue
        extras = [(r, c) for r, c in pts if c != col]
        if len(extras) != 1:
            continue
        er, ec = extras[0]
        if abs(ec - col) != 1:
            continue
        if er == rows_here[0]:
            candidates.append(
                {
                    "axis": "v",
                    "far": (int(rows_here[0]), int(col)),
                    "near": (int(rows_here[-1]), int(col)),
                    "direction": (1, 0),
                    "side": (0, int(ec - col)),
                }
            )
        if er == rows_here[-1]:
            candidates.append(
                {
                    "axis": "v",
                    "far": (int(rows_here[-1]), int(col)),
                    "near": (int(rows_here[0]), int(col)),
                    "direction": (-1, 0),
                    "side": (0, int(ec - col)),
                }
            )

    assert len(candidates) == 1, "each color must encode exactly one stripe-plus-seed object"
    return candidates[0]


def _trace_reflected_cells(
    hit: tuple[int, int],
    incoming: tuple[int, int],
    side: tuple[int, int],
    mirror_kind: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    out_dir = _reflect_dir(incoming, mirror_kind)
    r = int(hit[0] + side[0])
    c = int(hit[1] + side[1])
    cells: list[tuple[int, int]] = []

    while _in_bounds(shape, r, c):
        if (r, c) != hit:
            cells.append((int(r), int(c)))
        r += out_dir[0]
        c += out_dir[1]
        if (r, c) == hit:
            r += out_dir[0]
            c += out_dir[1]

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marked stripe to a 45-degree mirror and continue it after reflection.

    The input contains one diagonal grey mirror and several same-color stripe objects.
    Each stripe object is a straight 1-pixel bar plus one same-color seed pixel placed
    next to the endpoint farthest from the mirror. The output keeps the input and adds:
    1. the missing straight run from the stripe's near endpoint up to the mirror, and
    2. the reflected orthogonal run, shifted by the seed's side offset.
    """
    g = np.asarray(grid, dtype=int)
    shape = g.shape
    bg = _most_common_color(g)
    mirror_color, mirror_kind, mirror_cells = _find_mirror(g, bg)
    mirror_set = set(mirror_cells)

    out = g.copy()
    for color in sorted(set(map(int, np.unique(g))) - {int(bg), int(mirror_color)}):
        obj = _parse_stripe_object([(int(r), int(c)) for r, c in np.argwhere(g == int(color))])
        near = obj["near"]
        incoming = obj["direction"]
        side = obj["side"]

        r, c = near
        hit = None
        while True:
            r += incoming[0]
            c += incoming[1]
            if not _in_bounds(shape, r, c):
                break
            if (r, c) in mirror_set:
                hit = (int(r), int(c))
                break
            if int(g[r, c]) in (int(bg), int(color)):
                out[r, c] = int(color)

        assert hit is not None, "stripe extension must meet the mirror"

        rr = int(hit[0] + side[0])
        cc = int(hit[1] + side[1])
        out_dir = _reflect_dir(incoming, mirror_kind)
        while _in_bounds(shape, rr, cc):
            if (rr, cc) not in mirror_set and int(g[rr, cc]) in (int(bg), int(color)):
                out[rr, cc] = int(color)
            rr += out_dir[0]
            cc += out_dir[1]
            if (rr, cc) in mirror_set:
                rr += out_dir[0]
                cc += out_dir[1]

    return out


def _cheby_halo(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(shape[1], c + 2)):
                halo.add((int(rr), int(cc)))
    return halo


def _axis_name(direction: tuple[int, int]) -> str:
    return "h" if int(direction[0]) == 0 else "v"


def _enumerate_candidates(
    shape: tuple[int, int],
    mirror_cells: list[tuple[int, int]],
    mirror_kind: str,
    occupied_input: set[tuple[int, int]],
    object_halo: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
    used_hit_indices: set[int],
    axis_target: str | None,
    align_target: bool | None,
) -> list[dict]:
    candidates: list[dict] = []
    all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    gaps = range(GAP_RANGE[0], GAP_RANGE[1] + 1)
    lengths = range(STRIPE_LEN_RANGE[0], STRIPE_LEN_RANGE[1] + 1)

    for hit_idx, hit in enumerate(mirror_cells):
        if any(abs(hit_idx - other) <= 1 for other in used_hit_indices):
            continue

        for direction in all_dirs:
            if axis_target is not None and _axis_name(direction) != axis_target:
                continue

            if direction[0] == 0:
                side_options = [(-1, 0), (1, 0)]
            else:
                side_options = [(0, -1), (0, 1)]

            for side in side_options:
                aligned = side == _reflect_dir(direction, mirror_kind)
                if align_target is not None and aligned != bool(align_target):
                    continue

                for gap in gaps:
                    for length in lengths:
                        near = (
                            int(hit[0] - direction[0] * (gap + 1)),
                            int(hit[1] - direction[1] * (gap + 1)),
                        )
                        far = (
                            int(near[0] - direction[0] * (length - 1)),
                            int(near[1] - direction[1] * (length - 1)),
                        )
                        seed = (int(far[0] + side[0]), int(far[1] + side[1]))

                        stripe_cells = {
                            (int(far[0] + direction[0] * t), int(far[1] + direction[1] * t))
                            for t in range(length)
                        }
                        input_cells = set(stripe_cells)
                        input_cells.add(seed)
                        if len(input_cells) != length + 1:
                            continue
                        if not all(_in_bounds(shape, r, c) for r, c in input_cells):
                            continue
                        if input_cells & occupied_input:
                            continue
                        if input_cells & object_halo:
                            continue
                        if input_cells & reserved_output:
                            continue

                        extension = {
                            (int(near[0] + direction[0] * step), int(near[1] + direction[1] * step))
                            for step in range(1, gap + 1)
                        }
                        reflected = set(_trace_reflected_cells(hit, direction, side, mirror_kind, shape))
                        if len(reflected) < 2:
                            continue

                        output_cells = extension | reflected
                        if not output_cells:
                            continue
                        if output_cells & occupied_input:
                            continue
                        if output_cells & input_cells:
                            continue
                        if output_cells & reserved_output:
                            continue

                        candidates.append(
                            {
                                "hit_idx": int(hit_idx),
                                "hit": hit,
                                "direction": direction,
                                "side": side,
                                "stripe_cells": input_cells - {seed},
                                "seed": seed,
                                "output_cells": output_cells,
                            }
                        )

    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    stripe_count = random.randint(*STRIPE_COUNT_RANGE)
    mirror_kind = random.choice(["\\", "/"])
    max_len = min(h - 2, w - 2, 11)
    min_len = max(7, stripe_count * 2 + 1)
    if min_len > max_len:
        return None
    mirror_len = random.randint(min_len, max_len)

    if mirror_kind == "\\":
        start = (
            random.randint(1, h - mirror_len - 1),
            random.randint(1, w - mirror_len - 1),
        )
    else:
        start = (
            random.randint(1, h - mirror_len - 1),
            random.randint(mirror_len, w - 2),
        )

    mirror_cells = _mirror_segment(start, mirror_len, mirror_kind)
    grid = np.full(shape, BG, dtype=int)
    for r, c in mirror_cells:
        grid[r, c] = MIRROR

    occupied_input = set(mirror_cells)
    object_halo: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()
    used_hit_indices: set[int] = set()
    plans: list[dict] = []

    axis_targets = ["h", "v"] + [random.choice(["h", "v"]) for _ in range(stripe_count - 2)]
    align_targets: list[bool | None] = [None] * stripe_count
    if stripe_count >= 3:
        align_targets[0] = True
        align_targets[1] = False
    if stripe_count == 4:
        align_targets[2] = True
        align_targets[3] = False

    requirements = list(zip(axis_targets, align_targets))
    random.shuffle(requirements)

    for axis_target, align_target in requirements:
        candidates = _enumerate_candidates(
            shape,
            mirror_cells,
            mirror_kind,
            occupied_input,
            object_halo,
            reserved_output,
            used_hit_indices,
            axis_target,
            align_target,
        )
        if not candidates:
            return None
        plan = random.choice(candidates)
        plans.append(plan)
        all_input_cells = set(plan["stripe_cells"])
        all_input_cells.add(plan["seed"])
        occupied_input |= all_input_cells
        object_halo |= _cheby_halo(all_input_cells, shape)
        reserved_output |= set(plan["output_cells"])
        used_hit_indices.add(int(plan["hit_idx"]))

    colors = random.sample(COLOR_POOL, len(plans))
    for color, plan in zip(colors, plans):
        for r, c in plan["stripe_cells"]:
            grid[r, c] = int(color)
        sr, sc = plan["seed"]
        grid[sr, sc] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    changed = int(np.count_nonzero(out != grid))
    if changed < 3 * len(plans):
        return None

    for color in colors:
        if int(np.count_nonzero(out == int(color))) <= int(np.count_nonzero(grid == int(color))):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-stripe instance")

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 500
PLACE_RETRIES = 250


def _is_isolated_checker(grid: np.ndarray, r: int, c: int) -> bool:
    patch = grid[r : r + 2, c : c + 2]
    a = int(patch[0, 0])
    b = int(patch[0, 1])
    if a == BG or b == BG or a == b:
        return False
    if int(patch[1, 0]) != b or int(patch[1, 1]) != a:
        return False

    inside = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
    h, w = grid.shape
    for rr, cc in inside:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in inside:
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _find_junctions(grid: np.ndarray) -> list[dict[str, int]]:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    junctions: list[dict[str, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if used[r : r + 2, c : c + 2].any():
                continue
            if not _is_isolated_checker(grid, r, c):
                continue
            patch = grid[r : r + 2, c : c + 2]
            used[r : r + 2, c : c + 2] = True
            junctions.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "in_color": int(patch[0, 0]),
                    "out_color": int(patch[0, 1]),
                }
            )
    return junctions


def _extract_segments(grid: np.ndarray, junctions: list[dict[str, int]]) -> dict[int, dict[str, int | str]]:
    work = np.array(grid, copy=True)
    for junction in junctions:
        r = int(junction["r"])
        c = int(junction["c"])
        work[r : r + 2, c : c + 2] = BG

    segments: dict[int, dict[str, int | str]] = {}
    for color in sorted(int(v) for v in np.unique(work) if int(v) != BG):
        coords = np.argwhere(work == color)
        if coords.size == 0:
            continue
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            c0 = int(cols.min())
            c1 = int(cols.max())
            if int(c1 - c0 + 1) != int(len(cols)):
                raise AssertionError(f"horizontal segment {color} is not contiguous")
            segments[color] = {"orientation": "H", "length": int(len(coords))}
        elif len(cols) == 1:
            r0 = int(rows.min())
            r1 = int(rows.max())
            if int(r1 - r0 + 1) != int(len(rows)):
                raise AssertionError(f"vertical segment {color} is not contiguous")
            segments[color] = {"orientation": "V", "length": int(len(coords))}
        else:
            raise AssertionError(f"segment color {color} is not straight")
    return segments


def _recover_order(segments: dict[int, dict[str, int | str]], junctions: list[dict[str, int]]) -> list[int]:
    colors = list(segments.keys())
    if not colors:
        raise AssertionError("no path segments found")
    if len(colors) == 1:
        if junctions:
            raise AssertionError("single segment cannot have junctions")
        return colors

    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for junction in junctions:
        a = int(junction["in_color"])
        b = int(junction["out_color"])
        if a not in segments or b not in segments:
            raise AssertionError("junction references unknown segment color")
        if a in succ and int(succ[a]) != b:
            raise AssertionError("incoming segment branches to multiple outputs")
        if b in pred and int(pred[b]) != a:
            raise AssertionError("outgoing segment has multiple predecessors")
        succ[a] = b
        pred[b] = a

    starts = [color for color in colors if color not in pred]
    ends = [color for color in colors if color not in succ]
    if len(starts) != 1 or len(ends) != 1:
        raise AssertionError("junctions do not form a unique chain")

    order: list[int] = []
    seen: set[int] = set()
    cur = int(starts[0])
    while True:
        if cur in seen:
            raise AssertionError("junction chain contains a cycle")
        seen.add(cur)
        order.append(cur)
        if cur not in succ:
            break
        cur = int(succ[cur])

    if len(order) != len(colors):
        raise AssertionError("junction chain does not cover every segment")
    if len(junctions) != len(colors) - 1:
        raise AssertionError("junction count does not match chain length")
    return order


def _render_chain(order: list[int], segments: dict[int, dict[str, int | str]]) -> np.ndarray:
    painted: dict[tuple[int, int], int] = {}
    cur_r = 0
    cur_c = 0
    max_r = 0
    max_c = 0

    for i, color in enumerate(order):
        seg = segments[color]
        orientation = str(seg["orientation"])
        length = int(seg["length"])
        if length < 2:
            raise AssertionError("segments must have length at least 2")
        if i > 0:
            prev_orientation = str(segments[order[i - 1]]["orientation"])
            if orientation == prev_orientation:
                raise AssertionError("adjacent segments must turn at right angles")

        if orientation == "H":
            for offset in range(length):
                painted[(cur_r, cur_c + offset)] = int(color)
            cur_c += length - 1
        else:
            for offset in range(length):
                painted[(cur_r + offset, cur_c)] = int(color)
            cur_r += length - 1

        max_r = max(max_r, cur_r)
        max_c = max(max_c, cur_c)

    out = np.full((max_r + 1, max_c + 1), BG, dtype=int)
    for (r, c), color in painted.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered straight path segments into a single continuous path.

    The 2x2 checker junctions encode directed links A -> B, where the TL/BR
    diagonal is the incoming segment color and the TR/BL diagonal is the
    outgoing segment color. In the reconstructed path, incoming segments reach
    the corner from the left or from above, and outgoing segments leave the
    corner to the right or downward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    junctions = _find_junctions(g)
    segments = _extract_segments(g, junctions)
    order = _recover_order(segments, junctions)
    return _render_chain(order, segments)


def _build_expected(colors: list[int], orientations: list[str], lengths: list[int]) -> np.ndarray:
    segments = {
        int(color): {"orientation": str(orientation), "length": int(length)}
        for color, orientation, length in zip(colors, orientations, lengths)
    }
    return _render_chain([int(c) for c in colors], segments)


def _make_segment_object(color: int, orientation: str, length: int) -> dict[str, object]:
    if orientation == "H":
        return {
            "kind": "segment",
            "color": int(color),
            "h": 1,
            "w": int(length),
            "paint": [(0, c, int(color)) for c in range(length)],
        }
    return {
        "kind": "segment",
        "color": int(color),
        "h": int(length),
        "w": 1,
        "paint": [(r, 0, int(color)) for r in range(length)],
    }


def _make_junction_object(in_color: int, out_color: int) -> dict[str, object]:
    return {
        "kind": "junction",
        "h": 2,
        "w": 2,
        "paint": [
            (0, 0, int(in_color)),
            (0, 1, int(out_color)),
            (1, 0, int(out_color)),
            (1, 1, int(in_color)),
        ],
    }


def _sample_scene_size(objects: list[dict[str, object]]) -> tuple[int, int]:
    total_area = sum(int(obj["h"]) * int(obj["w"]) for obj in objects)
    longest_h = max(int(obj["h"]) for obj in objects)
    longest_w = max(int(obj["w"]) for obj in objects)
    base = max(14, int(np.ceil(np.sqrt(total_area * 6))))
    h = random.randint(max(base, longest_h + 4), max(base, longest_h + 4) + 8)
    w = random.randint(max(base, longest_w + 4), max(base, longest_w + 4) + 8)
    return h, w


def _place_objects(objects: list[dict[str, object]]) -> np.ndarray | None:
    h, w = _sample_scene_size(objects)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(
        objects,
        key=lambda obj: (int(obj["h"]) * int(obj["w"]), int(obj["h"]) + int(obj["w"])),
        reverse=True,
    )

    for obj in order:
        oh = int(obj["h"])
        ow = int(obj["w"])
        placed = False
        for _ in range(PLACE_RETRIES):
            r = random.randint(0, h - oh)
            c = random.randint(0, w - ow)
            if blocked[r : r + oh, c : c + ow].any():
                continue

            for dr, dc, color in obj["paint"]:
                grid[r + int(dr), c + int(dc)] = int(color)

            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(h, r + oh + 1)
            c1 = min(w, c + ow + 1)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(2, 6)
    colors = random.sample(PALETTE, n_segments)
    first_orientation = random.choice(("H", "V"))
    orientations = [
        first_orientation if i % 2 == 0 else ("V" if first_orientation == "H" else "H")
        for i in range(n_segments)
    ]
    lengths = [random.randint(3, 8) for _ in range(n_segments)]

    objects: list[dict[str, object]] = []
    for color, orientation, length in zip(colors, orientations, lengths):
        objects.append(_make_segment_object(int(color), str(orientation), int(length)))
    for i in range(n_segments - 1):
        objects.append(_make_junction_object(int(colors[i]), int(colors[i + 1])))

    random.shuffle(objects)
    grid = _place_objects(objects)
    if grid is None:
        return None

    expected = _build_expected(colors, orientations, lengths)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400
MIN_OBJECTS = 2
MAX_OBJECTS = 5
PADDING = 1


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))}
    comps: list[list[tuple[int, int]]] = []
    while cells:
        start = cells.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (int(r + dr), int(c + dc))
                if nxt in cells:
                    cells.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        comps.append(comp)
    return comps


def _square_info(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    side = int(r1 - r0 + 1)
    if side < 2 or side > 4 or side != int(c1 - c0 + 1):
        return None
    if len(comp) != side * side:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all(patch == int(color))):
        return None
    return {
        "color": int(color),
        "side": int(side),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _find_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    objects: list[dict] = []

    for color in colors:
        comps = _components_for_color(g, int(color))
        squares: list[dict] = []
        signals: list[tuple[int, int]] = []
        for comp in comps:
            if len(comp) == 1:
                signals.append(comp[0])
                continue
            info = _square_info(g, int(color), comp)
            if info is not None:
                squares.append(info)

        assert len(squares) == 1, "each foreground color must contain one solid square"
        assert len(signals) == 1, "each foreground color must contain one signal pixel"

        info = squares[0]
        signal_r, signal_c = signals[0]
        r0, c0, r1, c1 = info["bbox"]
        side = int(info["side"])

        if r0 <= signal_r <= r1 and signal_c < c0:
            dr, dc = DIRS["left"]
        elif r0 <= signal_r <= r1 and signal_c > c1:
            dr, dc = DIRS["right"]
        elif c0 <= signal_c <= c1 and signal_r < r0:
            dr, dc = DIRS["up"]
        elif c0 <= signal_c <= c1 and signal_r > r1:
            dr, dc = DIRS["down"]
        else:
            raise AssertionError("signal must align horizontally or vertically with its square")

        objects.append(
            {
                "color": int(color),
                "side": int(side),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "signal": (int(signal_r), int(signal_c)),
                "delta": (int(dr * side), int(dc * side)),
            }
        )

    return int(bg), objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each solid square toward its same-colored signal by exactly its own side length.

    The signal is removed. Because the translation distance equals the side length, the old
    square footprint becomes the full grey trail and the moved square occupies a disjoint box.
    """
    g = np.asarray(grid, dtype=int)
    bg, objects = _find_objects(g)
    out = g.copy()

    for obj in objects:
        sr, sc = obj["signal"]
        out[int(sr), int(sc)] = int(bg)

    for obj in objects:
        color = int(obj["color"])
        side = int(obj["side"])
        r0, c0, _r1, _c1 = obj["bbox"]
        dr, dc = obj["delta"]

        out[int(r0) : int(r0 + side), int(c0) : int(c0 + side)] = int(GREY)

        nr0 = int(r0 + dr)
        nc0 = int(c0 + dc)
        out[int(nr0) : int(nr0 + side), int(nc0) : int(nc0 + side)] = int(color)

    return out


def _expanded(cells: set[tuple[int, int]], h: int, w: int, pad: int = PADDING) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _square_cells(r0: int, c0: int, side: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(int(r0), int(r0 + side))
        for c in range(int(c0), int(c0 + side))
    }


def _center_band_positions(start: int, side: int) -> list[int]:
    if side % 2 == 1:
        return [int(start + side // 2)]
    return [int(start + side // 2 - 1), int(start + side // 2)]


def _sample_sizes(rng: random.Random, count: int) -> list[int]:
    base = [2, 3, 4]
    if count <= 3:
        return rng.sample(base, k=count)
    sizes = base + [rng.choice(base) for _ in range(count - 3)]
    rng.shuffle(sizes)
    return sizes


def _sample_directions(rng: random.Random, count: int) -> list[str]:
    if count == 2:
        dirs = [rng.choice(["left", "right"]), rng.choice(["up", "down"])]
    elif count == 3:
        dirs = [
            rng.choice(["left", "right"]),
            rng.choice(["up", "down"]),
            rng.choice(list(DIRS)),
        ]
    else:
        dirs = ["left", "right", "up", "down"] + [rng.choice(list(DIRS)) for _ in range(count - 4)]
    rng.shuffle(dirs)
    return dirs


def _sample_cover_flags(rng: random.Random, count: int) -> list[bool]:
    flags = [True, False]
    while len(flags) < count:
        flags.append(bool(rng.getrandbits(1)))
    rng.shuffle(flags)
    return flags


def _signal_choices(
    h: int,
    w: int,
    r0: int,
    c0: int,
    side: int,
    direction: str,
    cover_signal: bool,
) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    center_rows = _center_band_positions(int(r0), int(side))
    center_cols = _center_band_positions(int(c0), int(side))

    if direction == "right":
        max_gap = int(w - 1 - (c0 + side))
        gaps = range(1, min(max_gap, side - 1) + 1) if cover_signal else range(side, min(max_gap, side + 4) + 1)
        for gap in gaps:
            sig_c = int(c0 + side + gap)
            for sig_r in center_rows:
                choices.append((int(sig_r), int(sig_c)))
    elif direction == "left":
        max_gap = int(c0 - 1)
        gaps = range(1, min(max_gap, side - 1) + 1) if cover_signal else range(side, min(max_gap, side + 4) + 1)
        for gap in gaps:
            sig_c = int(c0 - 1 - gap)
            for sig_r in center_rows:
                choices.append((int(sig_r), int(sig_c)))
    elif direction == "down":
        max_gap = int(h - 1 - (r0 + side))
        gaps = range(1, min(max_gap, side - 1) + 1) if cover_signal else range(side, min(max_gap, side + 4) + 1)
        for gap in gaps:
            sig_r = int(r0 + side + gap)
            for sig_c in center_cols:
                choices.append((int(sig_r), int(sig_c)))
    else:
        max_gap = int(r0 - 1)
        gaps = range(1, min(max_gap, side - 1) + 1) if cover_signal else range(side, min(max_gap, side + 4) + 1)
        for gap in gaps:
            sig_r = int(r0 - 1 - gap)
            for sig_c in center_cols:
                choices.append((int(sig_r), int(sig_c)))
    return choices


def _place_object(
    rng: random.Random,
    h: int,
    w: int,
    side: int,
    direction: str,
    color: int,
    cover_signal: bool,
    forbidden_input: set[tuple[int, int]],
    forbidden_output: set[tuple[int, int]],
) -> dict | None:
    dr, dc = DIRS[direction]
    top_rows = list(range(0, h - side + 1))
    left_cols = list(range(0, w - side + 1))
    rng.shuffle(top_rows)
    rng.shuffle(left_cols)

    for r0 in top_rows:
        for c0 in left_cols:
            nr0 = int(r0 + dr * side)
            nc0 = int(c0 + dc * side)
            if nr0 < 0 or nc0 < 0 or nr0 + side > h or nc0 + side > w:
                continue

            old_cells = _square_cells(int(r0), int(c0), int(side))
            new_cells = _square_cells(int(nr0), int(nc0), int(side))
            sig_choices = _signal_choices(int(h), int(w), int(r0), int(c0), int(side), direction, cover_signal)
            rng.shuffle(sig_choices)

            for sig_r, sig_c in sig_choices:
                input_cells = set(old_cells)
                input_cells.add((int(sig_r), int(sig_c)))
                output_cells = set(old_cells) | set(new_cells)

                if input_cells & forbidden_input:
                    continue
                if input_cells & forbidden_output:
                    continue
                if output_cells & forbidden_input:
                    continue
                if output_cells & forbidden_output:
                    continue

                return {
                    "color": int(color),
                    "side": int(side),
                    "direction": direction,
                    "cover_signal": bool(cover_signal),
                    "old_top_left": (int(r0), int(c0)),
                    "new_top_left": (int(nr0), int(nc0)),
                    "signal": (int(sig_r), int(sig_c)),
                    "input_cells": input_cells,
                    "output_cells": output_cells,
                }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = int(rng.randint(MIN_OBJECTS, MAX_OBJECTS))
    sizes = _sample_sizes(rng, obj_count)
    lower = max(16, 6 + 3 * obj_count + max(sizes))
    h = int(rng.randint(lower, min(30, lower + 5)))
    w = int(rng.randint(lower, min(30, lower + 5)))

    bg = int(rng.choice(PALETTE))
    colors = rng.sample([c for c in PALETTE if c != bg], k=obj_count)
    directions = _sample_directions(rng, obj_count)
    cover_flags = _sample_cover_flags(rng, obj_count)

    specs = [
        {
            "color": int(colors[i]),
            "side": int(sizes[i]),
            "direction": directions[i],
            "cover_signal": bool(cover_flags[i]),
        }
        for i in range(obj_count)
    ]
    order = sorted(range(obj_count), key=lambda i: (-specs[i]["side"], specs[i]["direction"]))

    forbidden_input: set[tuple[int, int]] = set()
    forbidden_output: set[tuple[int, int]] = set()
    placed: list[dict] = [None] * obj_count  # type: ignore[list-item]

    for idx in order:
        spec = specs[idx]
        obj = _place_object(
            rng,
            int(h),
            int(w),
            int(spec["side"]),
            str(spec["direction"]),
            int(spec["color"]),
            bool(spec["cover_signal"]),
            forbidden_input,
            forbidden_output,
        )
        if obj is None:
            return None
        placed[idx] = obj
        forbidden_input |= _expanded(obj["input_cells"], int(h), int(w))
        forbidden_output |= _expanded(obj["output_cells"], int(h), int(w))

    grid = np.full((int(h), int(w)), int(bg), dtype=int)
    for obj in placed:
        r0, c0 = obj["old_top_left"]
        side = int(obj["side"])
        grid[int(r0) : int(r0 + side), int(c0) : int(c0 + side)] = int(obj["color"])
        sr, sc = obj["signal"]
        grid[int(sr), int(sc)] = int(obj["color"])

    parsed_bg, parsed_objects = _find_objects(grid)
    if int(parsed_bg) != int(bg) or len(parsed_objects) != obj_count:
        return None

    direction_counts = Counter("v" if obj["direction"] in {"up", "down"} else "h" for obj in placed)
    if not direction_counts.get("h") or not direction_counts.get("v"):
        return None
    if not any(bool(obj["cover_signal"]) for obj in placed):
        return None
    if not any(not bool(obj["cover_signal"]) for obj in placed):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if not np.any(out == GREY):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

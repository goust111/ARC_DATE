from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 3
SEED_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
SIDE_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _center_offsets(length: int) -> tuple[int, ...]:
    if length % 2 == 1:
        return (length // 2,)
    return (length // 2 - 1, length // 2)


def _gap_coord(r0: int, c0: int, h: int, w: int, side: str, offset: int | None = None) -> tuple[int, int]:
    if side in {"up", "down"}:
        assert offset is not None
        rr = r0 if side == "up" else r0 + h - 1
        return rr, c0 + offset
    assert offset is not None
    cc = c0 if side == "left" else c0 + w - 1
    return r0 + offset, cc


def _spike_cells(gap: tuple[int, int], side: str, length: int) -> list[tuple[int, int]]:
    dr, dc = SIDE_TO_VEC[side]
    gr, gc = gap
    return [(int(gr + dr * k), int(gc + dc * k)) for k in range(length)]


def _halo(cells: Iterable[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for nc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                out.add((int(nr), int(nc)))
    return out


def _parse_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    pts = {(int(r), int(c)) for r, c in comp}
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert 4 <= h <= 8 and 4 <= w <= 8, "frame size must be within the described range"

    border = _border_cells(r0, c0, h, w)
    assert pts <= border, "green component must lie on a rectangular border"
    missing = border - pts
    assert len(missing) == 1, "frame border must have exactly one gap"
    gap = next(iter(missing))
    gr, gc = gap
    assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gap cannot be at a corner"

    if gr == r0:
        side = "up"
        assert gc - c0 in _center_offsets(w), "top gap must be centered"
    elif gr == r1:
        side = "down"
        assert gc - c0 in _center_offsets(w), "bottom gap must be centered"
    elif gc == c0:
        side = "left"
        assert gr - r0 in _center_offsets(h), "left gap must be centered"
    else:
        assert gc == c1, "gap must lie on the outer border"
        side = "right"
        assert gr - r0 in _center_offsets(h), "right gap must be centered"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (inner != BG_COLOR) & (inner != FRAME_COLOR)
    assert seed_mask.any(), "each frame needs at least one seed"
    seed_colors = set(map(int, np.unique(inner[seed_mask])))
    assert len(seed_colors) == 1, "all seeds inside one frame must share a color"
    seed_color = seed_colors.pop()

    seed_count = int(seed_mask.sum())
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "side": side,
        "seed_color": seed_color,
        "seed_count": seed_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each rectangle's internal seed count as spike length and its border gap as spike direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = [_parse_frame(g, comp) for comp in _components_of_color(g, FRAME_COLOR)]
    assert frames, "expected at least one green frame"

    h, w = g.shape
    for frame in frames:
        for r, c in _spike_cells(frame["gap"], frame["side"], frame["seed_count"]):
            assert 0 <= r < h and 0 <= c < w, "spike must stay inside the grid"
            out[r, c] = int(frame["seed_color"])
    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 31)
GRID_W_RANGE = (14, 31)
RECT_SIZE_RANGE = (4, 9)
SEED_COUNT_CAP = 6
SCENE_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _render_input(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for obj in objects:
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        h = int(obj["h"])
        w = int(obj["w"])
        gap = tuple(obj["gap"])
        border = _border_cells(r0, c0, h, w)
        for r, c in border:
            if (r, c) != gap:
                grid[r, c] = FRAME_COLOR
        for r, c in obj["seeds"]:
            grid[int(r), int(c)] = int(obj["color"])
    return grid


def _render_output(inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        for r, c in _spike_cells(tuple(obj["gap"]), obj["side"], int(obj["length"])):
            out[int(r), int(c)] = int(obj["color"])
    return out


def _object_footprint(obj: dict) -> set[tuple[int, int]]:
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    h = int(obj["h"])
    w = int(obj["w"])
    box = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    box.update(_spike_cells(tuple(obj["gap"]), obj["side"], int(obj["length"])))
    return box


def _sample_count(height: int, width: int, rng: random.Random) -> int:
    area = height * width
    max_objects = min(4, max(1, area // 120 + 1))
    return rng.randint(1, max_objects)


def _sample_object(
    height: int,
    width: int,
    color: int,
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        rect_h = rng.randrange(*RECT_SIZE_RANGE)
        rect_w = rng.randrange(*RECT_SIZE_RANGE)
        side = rng.choice(tuple(SIDE_TO_VEC))
        r0 = rng.randrange(0, height - rect_h + 1)
        c0 = rng.randrange(0, width - rect_w + 1)

        if side in {"up", "down"}:
            offset = rng.choice(_center_offsets(rect_w))
        else:
            offset = rng.choice(_center_offsets(rect_h))
        gap = _gap_coord(r0, c0, rect_h, rect_w, side, offset)

        gr, gc = gap
        if side == "up":
            free_to_edge = gr + 1
        elif side == "down":
            free_to_edge = height - gr
        elif side == "left":
            free_to_edge = gc + 1
        else:
            free_to_edge = width - gc

        interior_area = (rect_h - 2) * (rect_w - 2)
        max_length = min(SEED_COUNT_CAP, free_to_edge, interior_area)
        if max_length < 1:
            continue
        length = rng.randint(1, max_length)

        spike = _spike_cells(gap, side, length)
        footprint = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
        footprint.update(spike)
        halo = _halo(footprint, height, width, pad=1)
        if footprint & blocked or halo & blocked:
            continue

        seeds = rng.sample(
            [(r, c) for r in range(r0 + 1, r0 + rect_h - 1) for c in range(c0 + 1, c0 + rect_w - 1)],
            length,
        )
        return {
            "r0": r0,
            "c0": c0,
            "h": rect_h,
            "w": rect_w,
            "side": side,
            "gap": gap,
            "length": length,
            "color": color,
            "seeds": seeds,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randrange(*GRID_H_RANGE)
    width = rng.randrange(*GRID_W_RANGE)
    count = _sample_count(height, width, rng)
    colors = list(SEED_COLORS)
    rng.shuffle(colors)

    objects: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    for idx in range(count):
        obj = _sample_object(height, width, colors[idx], blocked, rng)
        if obj is None:
            return None
        objects.append(obj)
        blocked |= _halo(_object_footprint(obj), height, width, pad=1)

    inp = _render_input((height, width), objects)
    expected = _render_output(inp, objects)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spike-frame scene")

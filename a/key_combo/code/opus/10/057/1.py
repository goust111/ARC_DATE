import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 1
RED = 2
GREEN = 3
PALETTE = (RED, GREEN)

GEN_ATTEMPTS = 500
GRID_ROWS = (13, 19)
GRID_COLS = (13, 19)
FRAME_COUNT = (3, 5)
SMALL_SIZES = ((1, 2), (1, 3), (2, 2))
LARGE_SIZE = (3, 3)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components_of_color(grid, FRAME):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "every blue component must be a rectangular frame"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(coords.shape[0]) == int(border.sum()), "blue component must be exactly the frame border"
        assert np.all(patch[border] == FRAME), "frame border must be solid blue"
        assert not np.any(patch[~border] == FRAME), "frame interior must not contain blue"
        frames.append((r0, c0, r1, c1))

    assert frames, "expected at least one blue frame"
    frames.sort()
    return frames


def _mirror_complete(patch: np.ndarray) -> np.ndarray:
    out = np.array(patch, copy=True)
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            a = int(patch[r, c])
            b = int(patch[r, mc])
            assert a in (BG, RED, GREEN) and b in (BG, RED, GREEN)
            assert not (a != BG and b != BG and a != b), "mirror mates must agree when both visible"
            if a == BG and b != BG:
                out[r, c] = b
    return out


def _rotation_orbits(n: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[list[tuple[int, int]]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit: list[tuple[int, int]] = []
            cur = (r, c)
            while cur not in orbit:
                orbit.append(cur)
                cur = (cur[1], n - 1 - cur[0])
            for cell in orbit:
                seen.add(cell)
            orbits.append(orbit)
    return orbits


def _rotation_complete(patch: np.ndarray) -> np.ndarray:
    out = np.array(patch, copy=True)
    h, w = out.shape
    assert h == w, "rotation-completed interiors must be square"
    for orbit in _rotation_orbits(h):
        colors = {int(patch[r, c]) for r, c in orbit if int(patch[r, c]) != BG}
        assert len(colors) <= 1, "rotation orbit has conflicting visible colors"
        if colors:
            color = colors.pop()
            for r, c in orbit:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each blue rectangular frame interior using mirror symmetry for small interiors and rotational symmetry for large interiors.

    Input structure:
    - The background is black.
    - Several blue one-pixel-thick rectangular frames are scattered without overlap.
    - Inside each frame, red and green cells form a partial pattern while black cells mark missing interior pixels.

    Rule:
    - Keep every blue frame border unchanged.
    - Compute each frame's enclosed interior area.
    - If the interior area is at most 4, complete missing cells by left-right mirroring within that frame.
    - If the interior area is greater than 4, complete missing cells by propagating visible colors across each 90-degree rotation orbit within that square interior.

    Key insight:
    - The enclosed area is not the final answer itself; it gates which symmetry constraint must be used to reconstruct the interior pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for r0, c0, r1, c1 in _frame_bboxes(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        area = int(interior.size)
        if area <= 4:
            solved = _mirror_complete(interior)
        else:
            assert interior.shape[0] == interior.shape[1], "large interiors must be square"
            solved = _rotation_complete(interior)
        out[r0 + 1 : r1, c0 + 1 : c1] = solved
    return out


def _small_full_interior(size: tuple[int, int], counterexample: bool) -> np.ndarray:
    h, w = map(int, size)
    full = np.zeros((h, w), dtype=int)
    if counterexample:
        assert (h, w) == (2, 2)
        top, bottom = random.sample(PALETTE, 2)
        full[0, :] = top
        full[1, :] = bottom
        return full

    for r in range(h):
        for c in range((w + 1) // 2):
            color = random.choice(PALETTE)
            full[r, c] = color
            full[r, w - 1 - c] = color

    if np.unique(full).size == 1 and random.random() < 0.7:
        rr = random.randrange(h)
        cc = random.randrange((w + 1) // 2)
        full[rr, cc] = RED if int(full[rr, cc]) == GREEN else GREEN
        full[rr, w - 1 - cc] = full[rr, cc]
    return full


def _small_partial_interior(full: np.ndarray, counterexample: bool) -> np.ndarray:
    part = np.array(full, copy=True)
    h, w = part.shape
    if counterexample:
        part[:, 1] = BG
        return part

    hidden = 0
    pair_cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w // 2):
            pair_cells.append((r, c))
            mc = w - 1 - c
            mode = random.choices(("left", "right", "both"), weights=(0.4, 0.4, 0.2), k=1)[0]
            if mode == "left":
                part[r, mc] = BG
                hidden += 1
            elif mode == "right":
                part[r, c] = BG
                hidden += 1

    if hidden == 0 and pair_cells:
        r, c = random.choice(pair_cells)
        part[r, w - 1 - c] = BG
    return part


def _large_full_interior(counterexample: bool) -> np.ndarray:
    full = np.zeros((3, 3), dtype=int)
    corner = random.choice(PALETTE)
    edge = RED if corner == GREEN else GREEN
    center = random.choice(PALETTE)

    full[1, 1] = center
    for r, c in ((0, 0), (0, 2), (2, 2), (2, 0)):
        full[r, c] = corner
    for r, c in ((0, 1), (1, 2), (2, 1), (1, 0)):
        full[r, c] = edge

    if not counterexample and random.random() < 0.35:
        full[1, 1] = edge
    return full


def _large_partial_interior(full: np.ndarray, counterexample: bool) -> np.ndarray:
    part = np.array(full, copy=True)
    if counterexample:
        part[0, 0] = BG
        part[0, 1] = BG
        part[0, 2] = BG
        if random.random() < 0.5:
            part[1, 0] = BG
        return part

    corner_orbit = [(0, 0), (0, 2), (2, 2), (2, 0)]
    edge_orbit = [(0, 1), (1, 2), (2, 1), (1, 0)]
    for orbit in (corner_orbit, edge_orbit):
        keep = random.choice(orbit)
        for cell in orbit:
            if cell != keep and random.random() < 0.6:
                part[cell] = BG
        if np.all([part[cell] == BG for cell in orbit]):
            part[keep] = full[keep]
    return part


def _build_frame(kind: str, counterexample: bool) -> dict:
    if kind == "large":
        interior_full = _large_full_interior(counterexample)
        interior_part = _large_partial_interior(interior_full, counterexample)
    else:
        if counterexample:
            size = (2, 2)
        elif kind == "small_odd":
            size = (1, 3)
        else:
            size = random.choice(SMALL_SIZES)
        interior_full = _small_full_interior(size, counterexample)
        interior_part = _small_partial_interior(interior_full, counterexample)

    ih, iw = interior_full.shape
    oh, ow = ih + 2, iw + 2
    frame_in = np.full((oh, ow), FRAME, dtype=int)
    frame_out = np.full((oh, ow), FRAME, dtype=int)
    frame_in[1:-1, 1:-1] = interior_part
    frame_out[1:-1, 1:-1] = interior_full
    return {
        "kind": "large" if interior_full.size > 4 else "small",
        "input_patch": frame_in,
        "output_patch": frame_out,
        "interior_input": interior_part,
        "interior_output": interior_full,
        "counterexample": counterexample,
        "shape": (oh, ow),
    }


def _rectangles_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1)


def _place_frames(frames: list[dict], rows: int, cols: int) -> list[tuple[int, int, int, int]] | None:
    placements: list[tuple[int, int, int, int]] = []
    ordered = sorted(enumerate(frames), key=lambda item: int(np.prod(item[1]["shape"])), reverse=True)
    by_index: list[tuple[int, int, int, int] | None] = [None] * len(frames)
    for idx, frame in ordered:
        h, w = frame["shape"]
        options: list[tuple[int, int, int, int]] = []
        for r0 in range(rows - h + 1):
            for c0 in range(cols - w + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(not _rectangles_too_close(rect, other) for other in placements):
                    options.append(rect)
        if not options:
            return None
        rect = random.choice(options)
        placements.append(rect)
        by_index[idx] = rect
    return [rect for rect in by_index if rect is not None]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _build_attempt() -> tuple[np.ndarray, np.ndarray, list[dict], list[tuple[int, int, int, int]]] | None:
    count = random.randint(*FRAME_COUNT)
    frames = [
        _build_frame("large", counterexample=True),
        _build_frame("small_counter", counterexample=True),
        _build_frame("small_odd", counterexample=False),
    ]
    while len(frames) < count:
        kind = random.choice(("large", "small_generic", "small_odd"))
        frames.append(_build_frame(kind, counterexample=False))

    random.shuffle(frames)
    rows = random.randint(*GRID_ROWS)
    cols = random.randint(*GRID_COLS)
    placements = _place_frames(frames, rows, cols)
    if placements is None:
        return None

    inp = np.zeros((rows, cols), dtype=int)
    out = np.zeros((rows, cols), dtype=int)
    for frame, (r0, c0, _, _) in zip(frames, placements):
        _paste(inp, frame["input_patch"], r0, c0)
        _paste(out, frame["output_patch"], r0, c0)
    return inp, out, frames, placements


def _is_rotation_counterexample(interior_in: np.ndarray, interior_out: np.ndarray) -> bool:
    try:
        rotated = _rotation_complete(interior_in)
    except AssertionError:
        return True
    return not np.array_equal(rotated, interior_out)


def _is_mirror_counterexample(interior_in: np.ndarray, interior_out: np.ndarray) -> bool:
    mirrored = _mirror_complete(interior_in)
    return not np.array_equal(mirrored, interior_out)


def _validate_generated(inp: np.ndarray, out: np.ndarray, frames: Iterable[dict]) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or inp.shape != out.shape:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    parsed = _frame_bboxes(inp)
    if len(parsed) != len(list(frames)):
        return False
    if np.any((inp != BG) & (inp != FRAME) & (inp != RED) & (inp != GREEN)):
        return False
    if np.any((out != BG) & (out != FRAME) & (out != RED) & (out != GREEN)):
        return False

    small_count = 0
    large_count = 0
    saw_small_counterexample = False
    saw_large_counterexample = False
    for frame in frames:
        interior_in = frame["interior_input"]
        interior_out = frame["interior_output"]
        if np.any(interior_out == BG):
            return False
        if not np.any(interior_in == BG):
            return False
        if frame["kind"] == "small":
            small_count += 1
            if not np.array_equal(_mirror_complete(interior_in), interior_out):
                return False
            if frame["counterexample"] and _is_rotation_counterexample(interior_in, interior_out):
                saw_small_counterexample = True
        else:
            large_count += 1
            if not np.array_equal(_rotation_complete(interior_in), interior_out):
                return False
            if frame["counterexample"] and _is_mirror_counterexample(interior_in, interior_out):
                saw_large_counterexample = True

    if small_count == 0 or large_count == 0:
        return False
    if not saw_small_counterexample or not saw_large_counterexample:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, out, frames, _placements = built
        if _validate_generated(inp, out, frames):
            return inp
    raise RuntimeError("generate_input failed to build a valid area-gated symmetry puzzle")

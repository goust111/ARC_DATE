from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RING_SIZE = 5
SIDES = ("top", "bottom", "left", "right")
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
LOCK_COLOR = 5
GEN_ATTEMPTS = 400
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _border_mask() -> np.ndarray:
    mask = np.zeros((RING_SIZE, RING_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


BORDER_MASK = _border_mask()


def _scan_runs(line: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(map(int, line)):
        if value == int(color):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(line) - 1)))
    return runs


def _boundary_colors(grid: np.ndarray, bg: int) -> set[int]:
    boundary = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    return set(map(int, boundary)) - {int(bg)}


def _lockholes(grid: np.ndarray, lock_color: int) -> list[dict]:
    H, W = grid.shape
    holes: list[dict] = []

    for start, end in _scan_runs(grid[0, :], lock_color):
        holes.append({"side": "top", "start": int(start), "end": int(end), "length": int(end - start + 1)})
    for start, end in _scan_runs(grid[H - 1, :], lock_color):
        holes.append({"side": "bottom", "start": int(start), "end": int(end), "length": int(end - start + 1)})
    for start, end in _scan_runs(grid[:, 0], lock_color):
        holes.append({"side": "left", "start": int(start), "end": int(end), "length": int(end - start + 1)})
    for start, end in _scan_runs(grid[:, W - 1], lock_color):
        holes.append({"side": "right", "start": int(start), "end": int(end), "length": int(end - start + 1)})

    return holes


def _monochrome_components(grid: np.ndarray, excluded: set[int]) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color in excluded or seen[r, c]:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _contiguous(values: Iterable[int]) -> bool:
    ordered = sorted(map(int, values))
    return all(b == a + 1 for a, b in zip(ordered, ordered[1:]))


def _analyze_ring(grid: np.ndarray, comp: dict) -> dict:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    assert (r1 - r0 + 1, c1 - c0 + 1) == (RING_SIZE, RING_SIZE), "ring bbox must be 5x5"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == color
    assert not bool((mask & ~BORDER_MASK).any()), "ring pixels must lie on the 5x5 border"

    missing = BORDER_MASK & ~mask
    miss_pts = np.argwhere(missing)
    assert 1 <= int(len(miss_pts)) <= 3, "ring must have a 1-3 pixel gap"

    side: str
    gap_start: int
    if np.all(miss_pts[:, 0] == 0):
        side = "top"
        values = miss_pts[:, 1]
        gap_start = int(values.min())
    elif np.all(miss_pts[:, 0] == RING_SIZE - 1):
        side = "bottom"
        values = miss_pts[:, 1]
        gap_start = int(values.min())
    elif np.all(miss_pts[:, 1] == 0):
        side = "left"
        values = miss_pts[:, 0]
        gap_start = int(values.min())
    elif np.all(miss_pts[:, 1] == RING_SIZE - 1):
        side = "right"
        values = miss_pts[:, 0]
        gap_start = int(values.min())
    else:
        raise AssertionError("gap must lie on exactly one side")

    assert _contiguous(values), "gap pixels must form a contiguous run"

    return {
        "color": color,
        "side": side,
        "gap_start": int(gap_start),
        "gap_len": int(len(miss_pts)),
        "source_bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _gap_cells(side: str, gap_start: int, gap_len: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, int(c)) for c in range(gap_start, gap_start + gap_len)]
    if side == "bottom":
        return [(RING_SIZE - 1, int(c)) for c in range(gap_start, gap_start + gap_len)]
    if side == "left":
        return [(int(r), 0) for r in range(gap_start, gap_start + gap_len)]
    return [(int(r), RING_SIZE - 1) for r in range(gap_start, gap_start + gap_len)]


def _placement_from_lock(ring: dict, lock: dict, H: int, W: int) -> tuple[int, int]:
    side = ring["side"]
    gap_start = int(ring["gap_start"])
    if side in ("top", "bottom"):
        c0 = int(lock["start"] - gap_start)
        r0 = 0 if side == "top" else int(H - RING_SIZE)
        assert 0 <= c0 <= W - RING_SIZE
    else:
        r0 = int(lock["start"] - gap_start)
        c0 = 0 if side == "left" else int(W - RING_SIZE)
        assert 0 <= r0 <= H - RING_SIZE
    return int(r0), int(c0)


def _paint_ring(out: np.ndarray, color: int, r0: int, c0: int, side: str, gap_start: int, gap_len: int) -> None:
    missing = set(_gap_cells(side, gap_start, gap_len))
    for rr in range(RING_SIZE):
        for cc in range(RING_SIZE):
            if not BORDER_MASK[rr, cc] or (rr, cc) in missing:
                continue
            out[int(r0 + rr), int(c0 + cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each incomplete 5x5 ring to the edge lockhole with the same side and gap length,
    then move the ring so the lockhole fills its gap.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = BG

    boundary_non_bg = _boundary_colors(g, bg)
    assert len(boundary_non_bg) == 1, "expected one boundary marker color"
    lock_color = int(next(iter(boundary_non_bg)))

    holes = _lockholes(g, lock_color)
    hole_map: dict[tuple[str, int], dict] = {}
    for hole in holes:
        key = (str(hole["side"]), int(hole["length"]))
        assert key not in hole_map, "each side/length lockhole pair must be unique"
        hole_map[key] = hole

    rings = [_analyze_ring(g, comp) for comp in _monochrome_components(g, {bg, lock_color})]
    assert len(rings) == len(hole_map), "each ring must match exactly one lockhole"

    out = np.full((H, W), bg, dtype=int)
    out[g == lock_color] = int(lock_color)
    occupied = np.zeros((H, W), dtype=bool)

    for ring in rings:
        key = (str(ring["side"]), int(ring["gap_len"]))
        assert key in hole_map, "missing lockhole for ring"
        r0, c0 = _placement_from_lock(ring, hole_map[key], H, W)
        gap_cells = set(_gap_cells(ring["side"], int(ring["gap_start"]), int(ring["gap_len"])))
        for rr in range(RING_SIZE):
            for cc in range(RING_SIZE):
                if not BORDER_MASK[rr, cc] or (rr, cc) in gap_cells:
                    continue
                tr = int(r0 + rr)
                tc = int(c0 + cc)
                assert not occupied[tr, tc], "moved rings must not overlap"
                occupied[tr, tc] = True
        _paint_ring(out, int(ring["color"]), r0, c0, ring["side"], int(ring["gap_start"]), int(ring["gap_len"]))

    return out


def _rects_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _gap_start_choices(gap_len: int) -> list[int]:
    return list(range(1, RING_SIZE - gap_len))


def _ring_patch(color: int, side: str, gap_start: int, gap_len: int, bg: int) -> np.ndarray:
    patch = np.full((RING_SIZE, RING_SIZE), bg, dtype=int)
    mask = BORDER_MASK.copy()
    for rr, cc in _gap_cells(side, gap_start, gap_len):
        mask[rr, cc] = False
    patch[mask] = int(color)
    return patch


def _place_target_boxes(H: int, W: int, specs: list[dict]) -> bool:
    placed: list[tuple[int, int, int, int]] = []
    order = specs[:]
    random.shuffle(order)

    for spec in order:
        side = str(spec["side"])
        gap_start = int(spec["gap_start"])
        gap_len = int(spec["gap_len"])
        success = False
        for _ in range(200):
            if side == "top":
                r0 = 0
                c0 = random.randint(0, W - RING_SIZE)
                lock_start = c0 + gap_start
                lock_end = lock_start + gap_len - 1
                if lock_start == 0 or lock_end == W - 1:
                    continue
            elif side == "bottom":
                r0 = H - RING_SIZE
                c0 = random.randint(0, W - RING_SIZE)
                lock_start = c0 + gap_start
                lock_end = lock_start + gap_len - 1
                if lock_start == 0 or lock_end == W - 1:
                    continue
            elif side == "left":
                r0 = random.randint(0, H - RING_SIZE)
                c0 = 0
                lock_start = r0 + gap_start
                lock_end = lock_start + gap_len - 1
                if lock_start == 0 or lock_end == H - 1:
                    continue
            else:
                r0 = random.randint(0, H - RING_SIZE)
                c0 = W - RING_SIZE
                lock_start = r0 + gap_start
                lock_end = lock_start + gap_len - 1
                if lock_start == 0 or lock_end == H - 1:
                    continue

            bbox = (int(r0), int(c0), int(r0 + RING_SIZE - 1), int(c0 + RING_SIZE - 1))
            if any(_rects_conflict(bbox, other, pad=1) for other in placed):
                continue

            spec["target_bbox"] = bbox
            placed.append(bbox)
            success = True
            break
        if not success:
            return False
    return True


def _place_source_boxes(H: int, W: int, count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(300):
            r0 = random.randint(1, H - RING_SIZE - 1)
            c0 = random.randint(1, W - RING_SIZE - 1)
            bbox = (int(r0), int(c0), int(r0 + RING_SIZE - 1), int(c0 + RING_SIZE - 1))
            if any(_rects_conflict(bbox, other, pad=1) for other in boxes):
                continue
            boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _draw_lockhole(grid: np.ndarray, side: str, start: int, length: int, color: int) -> None:
    if side == "top":
        grid[0, start : start + length] = int(color)
    elif side == "bottom":
        grid[-1, start : start + length] = int(color)
    elif side == "left":
        grid[start : start + length, 0] = int(color)
    else:
        grid[start : start + length, -1] = int(color)


def _build_expected(H: int, W: int, specs: list[dict]) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for spec in specs:
        r0, c0, _r1, _c1 = spec["target_bbox"]
        side = str(spec["side"])
        gap_start = int(spec["gap_start"])
        gap_len = int(spec["gap_len"])
        lock_start = int(spec["lock_start"])
        _draw_lockhole(out, side, lock_start, gap_len, LOCK_COLOR)
        _paint_ring(out, int(spec["color"]), r0, c0, side, gap_start, gap_len)
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(17, 24)
    W = random.randint(17, 24)
    n_rings = random.randint(2, 5)
    if n_rings > len(PALETTE):
        return None

    pair_pool = [(side, gap_len) for side in SIDES for gap_len in (1, 2, 3)]
    chosen_pairs = random.sample(pair_pool, n_rings)
    colors = random.sample(PALETTE, n_rings)

    specs: list[dict] = []
    for color, (side, gap_len) in zip(colors, chosen_pairs):
        specs.append(
            {
                "color": int(color),
                "side": str(side),
                "gap_len": int(gap_len),
                "gap_start": random.choice(_gap_start_choices(gap_len)),
            }
        )

    if not _place_target_boxes(H, W, specs):
        return None

    source_boxes = _place_source_boxes(H, W, n_rings)
    if source_boxes is None:
        return None

    for spec, source_bbox in zip(specs, source_boxes):
        spec["source_bbox"] = source_bbox
        tr0, tc0, _tr1, _tc1 = spec["target_bbox"]
        if spec["side"] in ("top", "bottom"):
            spec["lock_start"] = int(tc0 + spec["gap_start"])
        else:
            spec["lock_start"] = int(tr0 + spec["gap_start"])

    expected = _build_expected(H, W, specs)
    inp = np.full((H, W), BG, dtype=int)

    for spec in specs:
        _draw_lockhole(inp, str(spec["side"]), int(spec["lock_start"]), int(spec["gap_len"]), LOCK_COLOR)

    for spec in specs:
        sr0, sc0, _sr1, _sc1 = spec["source_bbox"]
        patch = _ring_patch(int(spec["color"]), str(spec["side"]), int(spec["gap_start"]), int(spec["gap_len"]), BG)
        target = inp[sr0 : sr0 + RING_SIZE, sc0 : sc0 + RING_SIZE]
        if np.any((target != BG) & (patch != BG)):
            return None
        target[:] = np.where(patch != BG, patch, target)

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

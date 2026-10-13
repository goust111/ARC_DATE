import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, find_components, in_bounds


BG = Color.BLACK
ANCHOR = Color.GREY
PALETTE_POOL = [Color.YELLOW, Color.ORANGE, Color.PINK, Color.MAROON, Color.GREEN]
GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 250
EXTRA_ATTEMPTS = 150


def _extract_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    row = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(row != BG)
    assert cols.size > 0, "expected a top-row palette strip"
    start = int(cols.min())
    end = int(cols.max())
    assert end - start + 1 == cols.size, "palette strip must be contiguous"
    palette = row[start : end + 1].astype(int).tolist()
    assert 3 <= len(palette) <= 5, "palette length must be 3..5"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert ANCHOR not in palette, "palette must not use grey"
    return start, palette


def _extract_anchors(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[ANCHOR],
        return_format="dict",
    )
    anchors: list[tuple[int, int]] = []
    for comp in comps:
        assert int(comp["area"]) == 1, "anchors must be single grey pixels"
        r, c = map(int, comp["coords"][0])
        assert r > 0, "anchors must be below the palette row"
        anchors.append((r, c))
    assert 2 <= len(anchors) <= 4, "expected 2..4 anchors"
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            assert dist >= 6, "anchors must be well separated"
    anchors.sort(key=lambda rc: (rc[1], rc[0]))
    return anchors


def _validate_input_structure(grid: np.ndarray, palette_start: int, palette: list[int], anchors: list[tuple[int, int]]) -> None:
    allowed = {(0, c) for c in range(palette_start, palette_start + len(palette))}
    allowed |= set(anchors)
    nz = np.argwhere(np.asarray(grid, dtype=int) != BG)
    for r, c in map(tuple, nz.tolist()):
        assert (int(r), int(c)) in allowed, "unexpected non-background pixel in input"


def _arm_dirs(name: str) -> list[tuple[int, int]]:
    if name == "up_right":
        return [(-1, 0), (0, 1)]
    if name == "down_left":
        return [(1, 0), (0, -1)]
    raise ValueError(name)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = map(int, g.shape)
    palette_start, palette = _extract_palette(g)
    anchors = _extract_anchors(g)
    _validate_input_structure(g, palette_start, palette, anchors)
    step_len = len(palette)

    occupied = g.copy()
    owner = np.zeros((H, W), dtype=int)
    owner[0, palette_start : palette_start + step_len] = -1
    for idx, (r, c) in enumerate(anchors, start=1):
        owner[r, c] = -idx

    records = []
    events = []
    for anchor_idx, anchor in enumerate(anchors, start=1):
        anchor_rec = {"anchor": anchor, "arms": {}}
        for arm_name in ("up_right", "down_left"):
            dirs = _arm_dirs(arm_name)
            r, c = anchor
            steps = []
            stop_reason = None
            stop_step = None
            blockers = []
            for step_idx in range(1, max(H, W) + 1):
                dr, dc = dirs[(step_idx - 1) % 2]
                cells = [(r + dr * k, c + dc * k) for k in range(1, step_len + 1)]
                if not all(in_bounds(g.shape, rr, cc) for rr, cc in cells):
                    stop_reason = "bounds"
                    stop_step = step_idx
                    break
                step_blockers = sorted({int(owner[rr, cc]) for rr, cc in cells if int(occupied[rr, cc]) != BG})
                if step_blockers:
                    stop_reason = "collision"
                    stop_step = step_idx
                    blockers = step_blockers
                    events.append(
                        {
                            "anchor_idx": anchor_idx,
                            "arm": arm_name,
                            "step": step_idx,
                            "blockers": step_blockers,
                        }
                    )
                    break
                steps.append(cells)
                for rr, cc in cells:
                    occupied[rr, cc] = 1
                    owner[rr, cc] = anchor_idx
                r += dr * step_len
                c += dc * step_len
            anchor_rec["arms"][arm_name] = {
                "steps": steps,
                "stop_reason": stop_reason,
                "stop_step": stop_step,
                "blockers": blockers,
            }
        records.append(anchor_rec)

    out = g.copy()
    for rec in records:
        for arm_name in ("up_right", "down_left"):
            for depth, cells in enumerate(rec["arms"][arm_name]["steps"], start=1):
                color = int(palette[(depth - 1) % step_len])
                for rr, cc in cells:
                    out[rr, cc] = color
    for r, c in anchors:
        out[r, c] = ANCHOR

    meta = {
        "palette_start": palette_start,
        "palette": palette,
        "anchors": anchors,
        "records": records,
        "events": events,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow paired staircases from all grey anchors using the top-row palette as both step length and cyclic colors.

    Input structure:
    - The background is black.
    - The top row contains one contiguous palette strip of 3-5 distinct non-grey colors.
    - Below it are 2-4 isolated grey anchor pixels; no other foreground pixels are present.

    Rule:
    - Let P be the palette length.
    - Process anchors from left to right, breaking ties by row.
    - For each anchor, grow one arm in repeated steps of up P then right P, and another arm in repeated steps of down P then left P.
    - Before drawing any P-pixel step, stop that arm if any pixel of the whole step would leave the grid or hit an already non-black pixel.
    - After all growth, recolor each staircase step by depth from its anchor: step d uses palette[(d-1) mod P], while anchors stay grey.

    Key insight:
    - The palette strip is doubly encoded: its length determines the staircase geometry, and the same ordered colors determine the depth-based recoloring.
    """
    out, _meta = _simulate(np.asarray(grid, dtype=int))
    return out


def _render_input(shape: tuple[int, int], palette_start: int, palette: list[int], anchors: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[0, palette_start : palette_start + len(palette)] = np.array(palette, dtype=int)
    for r, c in anchors:
        grid[r, c] = ANCHOR
    return grid


def _all_arms_have_two_steps(meta: dict) -> bool:
    for rec in meta["records"]:
        for arm in rec["arms"].values():
            if len(arm["steps"]) < 2:
                return False
    return True


def _anchors_show_stair_pattern(meta: dict) -> bool:
    for rec in meta["records"]:
        lengths = [len(rec["arms"][name]["steps"]) for name in ("up_right", "down_left")]
        if min(lengths) < 1:
            return False
        if max(lengths) < 2:
            return False
    return True


def _has_cross_anchor_collision(meta: dict) -> bool:
    for event in meta["events"]:
        for blocker in event["blockers"]:
            if blocker > 0 and blocker != int(event["anchor_idx"]):
                return True
    return False


def _total_stair_pixels(meta: dict) -> int:
    total = 0
    for rec in meta["records"]:
        for arm in rec["arms"].values():
            total += len(arm["steps"]) * len(meta["palette"])
    return total


def _stop_modes(meta: dict) -> set[str]:
    modes = set()
    for rec in meta["records"]:
        for arm in rec["arms"].values():
            if arm["stop_reason"] is not None:
                modes.add(str(arm["stop_reason"]))
    return modes


def _candidate_positions(H: int, W: int, P: int) -> list[tuple[int, int]]:
    rows = range(P + 1, H - P)
    cols = range(P, W - P)
    return [(r, c) for r in rows for c in cols]


def _build_valid_sample(shape: tuple[int, int], palette_start: int, palette: list[int], n_anchors: int, require_both_stop_modes: bool) -> np.ndarray | None:
    H, W = shape
    P = len(palette)
    candidates = _candidate_positions(H, W, P)
    if len(candidates) < n_anchors:
        return None

    pair_found = None
    for _ in range(PAIR_ATTEMPTS):
        a, b = random.sample(candidates, 2)
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 6:
            continue
        test_anchors = [a, b]
        grid = _render_input(shape, palette_start, palette, test_anchors)
        try:
            _out, meta = _simulate(grid)
        except AssertionError:
            continue
        if not _anchors_show_stair_pattern(meta):
            continue
        if not _has_cross_anchor_collision(meta):
            continue
        pair_found = test_anchors
        break
    if pair_found is None:
        return None

    anchors = pair_found[:]
    remaining = candidates[:]
    random.shuffle(remaining)
    for cand in remaining:
        if len(anchors) >= n_anchors:
            break
        if any(abs(cand[0] - r) + abs(cand[1] - c) < 6 for r, c in anchors):
            continue
        trial = anchors + [cand]
        grid = _render_input(shape, palette_start, palette, trial)
        try:
            _out, meta = _simulate(grid)
        except AssertionError:
            continue
        if _anchors_show_stair_pattern(meta) and _has_cross_anchor_collision(meta):
            anchors = trial

    if len(anchors) != n_anchors:
        return None

    for _ in range(EXTRA_ATTEMPTS):
        random.shuffle(anchors)
        grid = _render_input(shape, palette_start, palette, anchors)
        try:
            out, meta = _simulate(grid)
        except AssertionError:
            continue
        if not _anchors_show_stair_pattern(meta):
            continue
        if not _has_cross_anchor_collision(meta):
            continue
        if require_both_stop_modes and not {"bounds", "collision"}.issubset(_stop_modes(meta)):
            continue
        density = _total_stair_pixels(meta) / float(H * W)
        if not (0.08 <= density <= 0.45):
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    P = random.randint(3, 5)
    n_anchors = random.randint(2, 4)
    min_h = max(15, 2 * P + 6)
    min_w = max(18, 3 * P + 6)
    H = random.randint(min_h, min(30, min_h + 10))
    W = random.randint(min_w, min(30, min_w + 10))
    palette_start = random.randint(0, W - P)
    palette = random.sample(PALETTE_POOL, P)
    require_both_stop_modes = random.random() < 0.55
    grid = _build_valid_sample((H, W), palette_start, palette, n_anchors, require_both_stop_modes)
    if grid is None:
        return None

    out, meta = _simulate(grid)
    assert _anchors_show_stair_pattern(meta)
    assert _has_cross_anchor_collision(meta)
    if require_both_stop_modes:
        assert {"bounds", "collision"}.issubset(_stop_modes(meta))
    assert not np.array_equal(grid, out)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated semantic checks")

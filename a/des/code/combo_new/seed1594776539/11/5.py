import random
from collections import Counter, deque

import numpy as np


BG = 0
CORE_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
GEN_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_filled_rectangle(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _extract_structure(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _components(g, bg)
    assert comps, "expected at least one non-background component"

    core_candidates = [comp for comp in comps if _is_filled_rectangle(comp)]
    assert len(core_candidates) == 1, "expected exactly one solid core rectangle"
    core = core_candidates[0]

    frames = [comp for comp in comps if comp is not core]
    assert frames, "expected at least one frame"

    cr0, cc0, cr1, cc1 = core["bbox"]
    center_r_sum = int(cr0 + cr1)
    center_c_sum = int(cc0 + cc1)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        assert r0 < cr0 and c0 < cc0 and r1 > cr1 and c1 > cc1, "frames must surround the core"
        assert int(r0 + r1) == center_r_sum, "frames must be vertically concentric"
        assert int(c0 + c1) == center_c_sum, "frames must be horizontally concentric"

    frames.sort(key=lambda comp: (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1), reverse=True)
    return bg, core, frames


def _gap_side(grid: np.ndarray, frame: dict) -> str:
    g = np.asarray(grid, dtype=int)
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]

    candidates: list[str] = []
    side_slices = {
        "top": np.asarray(g[r0, c0 : c1 + 1], dtype=int),
        "bottom": np.asarray(g[r1, c0 : c1 + 1], dtype=int),
        "left": np.asarray(g[r0 : r1 + 1, c0], dtype=int),
        "right": np.asarray(g[r0 : r1 + 1, c1], dtype=int),
    }
    for side, vals in side_slices.items():
        missing = np.flatnonzero(vals != color)
        if missing.size == 0:
            continue
        assert missing.size >= 2, "gap must be at least two pixels long"
        assert int(missing[-1] - missing[0] + 1) == int(missing.size), "gap must be contiguous"
        assert int(missing[0]) > 0 and int(missing[-1]) < int(vals.size - 1), "gap must stay away from corners"
        candidates.append(side)

    assert len(candidates) == 1, "each frame must have exactly one gapped side"
    return candidates[0]


def _centered_span(center_sum: int, length: int) -> tuple[int, int]:
    start = (int(center_sum) - int(length) + 1) // 2
    end = start + int(length) - 1
    return int(start), int(end)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each gapped concentric frame into a compact stripe around the solid core.

    Input structure:
    - Background is Black.
    - The main object is a stack of concentric one-pixel-thick rectangular frames,
      separated by Black gaps, with one distinct color per frame.
    - Each frame has exactly one continuous mid-edge gap on one of its four sides.
    - Inside the innermost frame sits a small solid Grey rectangle acting as the core.

    Rule:
    - Identify the solid core and sort the surrounding frames from outermost to
      innermost by their bounding boxes.
    - For each frame, detect which side contains the gap; the frame collapses
      toward the opposite side of the compact output structure.
    - Replace that frame with a one-pixel stripe of the same color, adjacent to
      the core or previously collapsed stripes on that target side, with stripe
      length equal to the frame side opposite the gap.
    - Keep only the core and all collapsed stripes; every other cell becomes Black.

    Key insight:
    - The gap is a directional cue: a gap on one side means the frame is
      projected inward toward the opposite side, turning nested hollow rectangles
      into a tightly packed multi-color stripe arrangement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, core, frames = _extract_structure(g)

    out = np.full_like(g, bg)
    core_coords = core["coords"]
    out[core_coords[:, 0], core_coords[:, 1]] = int(core["color"])

    cr0, cc0, cr1, cc1 = core["bbox"]
    center_r_sum = int(cr0 + cr1)
    center_c_sum = int(cc0 + cc1)
    placed = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    for frame in frames:
        gap = _gap_side(g, frame)
        target = OPPOSITE_SIDE[gap]
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)

        if target == "top":
            row = int(cr0 - placed["top"] - 1)
            start, end = _centered_span(center_c_sum, width)
            assert 0 <= row < g.shape[0] and 0 <= start <= end < g.shape[1]
            out[row, start : end + 1] = color
            placed["top"] += 1
        elif target == "bottom":
            row = int(cr1 + placed["bottom"] + 1)
            start, end = _centered_span(center_c_sum, width)
            assert 0 <= row < g.shape[0] and 0 <= start <= end < g.shape[1]
            out[row, start : end + 1] = color
            placed["bottom"] += 1
        elif target == "left":
            col = int(cc0 - placed["left"] - 1)
            start, end = _centered_span(center_r_sum, height)
            assert 0 <= col < g.shape[1] and 0 <= start <= end < g.shape[0]
            out[start : end + 1, col] = color
            placed["left"] += 1
        else:
            col = int(cc1 + placed["right"] + 1)
            start, end = _centered_span(center_r_sum, height)
            assert 0 <= col < g.shape[1] and 0 <= start <= end < g.shape[0]
            out[start : end + 1, col] = color
            placed["right"] += 1

    return out


def _balanced_sides(rng: random.Random, count: int) -> list[str]:
    sides: list[str] = []
    while len(sides) < count:
        bag = list(SIDES)
        rng.shuffle(bag)
        take = min(len(bag), count - len(sides))
        sides.extend(bag[:take])
    return sides


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap_side: str, rng: random.Random) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if gap_side in ("top", "bottom"):
        side_len = int(c1 - c0 + 1)
        max_gap = max(2, side_len - 4)
        gap_len = rng.randint(2, max_gap)
        spare = side_len - gap_len
        lead = spare // 2
        if spare % 2 == 1 and rng.random() < 0.5:
            lead += 1
        gs = int(c0 + lead)
        ge = int(gs + gap_len - 1)
        row = r0 if gap_side == "top" else r1
        grid[row, gs : ge + 1] = BG
    else:
        side_len = int(r1 - r0 + 1)
        max_gap = max(2, side_len - 4)
        gap_len = rng.randint(2, max_gap)
        spare = side_len - gap_len
        lead = spare // 2
        if spare % 2 == 1 and rng.random() < 0.5:
            lead += 1
        gs = int(r0 + lead)
        ge = int(gs + gap_len - 1)
        col = c0 if gap_side == "left" else c1
        grid[gs : ge + 1, col] = BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    core_h = rng.randint(2, 5)
    core_w = rng.randint(2, 6)

    possible_counts = [
        count
        for count in range(2, 6)
        if core_h + 4 * count <= 26 and core_w + 4 * count <= 26
    ]
    if not possible_counts:
        return None
    n_frames = rng.choice(possible_counts)

    step_h = [rng.randint(2, 3) for _ in range(n_frames)]
    step_w = [rng.randint(2, 3) for _ in range(n_frames)]
    total_h = core_h + 2 * sum(step_h)
    total_w = core_w + 2 * sum(step_w)
    if total_h > 30 or total_w > 30:
        return None

    grid_h = rng.randint(total_h, min(30, total_h + 6))
    grid_w = rng.randint(total_w, min(30, total_w + 6))
    top = rng.randint(0, grid_h - total_h)
    left = rng.randint(0, grid_w - total_w)

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    outer_pad_h = sum(step_h)
    outer_pad_w = sum(step_w)
    core_r0 = top + outer_pad_h
    core_c0 = left + outer_pad_w
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1
    grid[core_r0 : core_r1 + 1, core_c0 : core_c1 + 1] = CORE_COLOR

    colors = list(rng.sample(FRAME_COLORS, n_frames))
    sides = _balanced_sides(rng, n_frames)

    acc_h = 0
    acc_w = 0
    bboxes: list[tuple[int, int, int, int]] = []
    for idx in range(n_frames):
        acc_h += step_h[idx]
        acc_w += step_w[idx]
        bboxes.append((core_r0 - acc_h, core_c0 - acc_w, core_r1 + acc_h, core_c1 + acc_w))

    for bbox, color, gap_side in zip(bboxes, colors, sides):
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            return None
        _draw_frame(grid, bbox, int(color), gap_side, rng)

    try:
        out = main(grid)
    except Exception:
        return None

    assert out.shape == grid.shape
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid nested-frame puzzle")

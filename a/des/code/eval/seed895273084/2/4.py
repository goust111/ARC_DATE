from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW = 2
FRAME = 5
STRIPE_PALETTE = [1, 3, 4, 6, 7, 8, 9]
DIRECTIONS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 300

ARROW_PATTERNS = {
    "right": {
        "cells": frozenset({(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)}),
        "tip": (1, 2),
    },
    "left": {
        "cells": frozenset({(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)}),
        "tip": (1, 0),
    },
    "down": {
        "cells": frozenset({(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)}),
        "tip": (2, 1),
    },
    "up": {
        "cells": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
        "tip": (0, 1),
    },
}

ARROW_OFFSETS = {
    "right": [(0, 0), (0, -1), (0, -2), (-1, 0), (1, 0)],
    "left": [(0, 0), (0, 1), (0, 2), (-1, 0), (1, 0)],
    "down": [(0, 0), (-1, 0), (-2, 0), (0, -1), (0, 1)],
    "up": [(0, 0), (1, 0), (2, 0), (0, -1), (0, 1)],
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0 + 1, r1):
            expected.add((r, c0))
            expected.add((r, c1))
        if set(comp) != expected:
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != BG):
            continue
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_h": int(r1 - r0 - 1),
                "inner_w": int(c1 - c0 - 1),
            }
        )
    return frames


def _classify_arrow(comp: list[tuple[int, int]]) -> dict | None:
    if len(comp) != 5:
        return None
    r0, c0, r1, c1 = _bbox(comp)
    if (r1 - r0, c1 - c0) != (2, 2):
        return None
    norm = frozenset((r - r0, c - c0) for r, c in comp)
    for direction, spec in ARROW_PATTERNS.items():
        if norm != spec["cells"]:
            continue
        tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
        return {
            "direction": direction,
            "tip": (int(tip[0]), int(tip[1])),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
        }
    return None


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components(grid == ARROW):
        parsed = _classify_arrow(comp)
        if parsed is not None:
            arrows.append(parsed)
    return arrows


def _line_component_info(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    colors = {int(grid[r, c]) for r, c in comp}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    r0, c0, r1, c1 = _bbox(comp)
    cells = set(comp)
    if c0 == c1 and len(comp) == r1 - r0 + 1:
        if cells == {(r, c0) for r in range(r0, r1 + 1)}:
            return {
                "orientation": "V",
                "color": int(color),
                "index": int(c0),
                "span0": int(r0),
                "span1": int(r1),
            }
    if r0 == r1 and len(comp) == c1 - c0 + 1:
        if cells == {(r0, c) for c in range(c0, c1 + 1)}:
            return {
                "orientation": "H",
                "color": int(color),
                "index": int(r0),
                "span0": int(c0),
                "span1": int(c1),
            }
    return None


def _find_stripes(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != FRAME) & (grid != ARROW)
    stripes: list[dict] = []
    for comp in _components(mask):
        parsed = _line_component_info(grid, comp)
        if parsed is not None:
            stripes.append(parsed)
    return stripes


def _matches(frame: dict, arrow: dict) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    tip_r, tip_c = arrow["tip"]
    direction = arrow["direction"]
    if direction == "right":
        return tip_c + 1 == c0 and r0 < tip_r < r1
    if direction == "left":
        return tip_c - 1 == c1 and r0 < tip_r < r1
    if direction == "down":
        return tip_r + 1 == r0 and c0 < tip_c < c1
    return tip_r - 1 == r1 and c0 < tip_c < c1


def _collect_run(values: list[dict]) -> list[dict]:
    run: list[dict] = []
    prev = None
    for item in values:
        idx = int(item["index"])
        if prev is None:
            run.append(item)
            prev = idx
            continue
        if abs(idx - prev) != 2:
            break
        run.append(item)
        prev = idx
    return run


def _frame_stripes(frame: dict, arrow: dict, stripes: list[dict]) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    ar0, ac0, ar1, ac1 = arrow["bbox"]
    direction = arrow["direction"]
    if direction == "right":
        candidates = [
            s
            for s in stripes
            if s["orientation"] == "V"
            and s["span0"] == r0 + 1
            and s["span1"] == r1 - 1
            and s["index"] < ac0
        ]
        candidates.sort(key=lambda s: -int(s["index"]))
    elif direction == "left":
        candidates = [
            s
            for s in stripes
            if s["orientation"] == "V"
            and s["span0"] == r0 + 1
            and s["span1"] == r1 - 1
            and s["index"] > ac1
        ]
        candidates.sort(key=lambda s: int(s["index"]))
    elif direction == "down":
        candidates = [
            s
            for s in stripes
            if s["orientation"] == "H"
            and s["span0"] == c0 + 1
            and s["span1"] == c1 - 1
            and s["index"] < ar0
        ]
        candidates.sort(key=lambda s: -int(s["index"]))
    else:
        candidates = [
            s
            for s in stripes
            if s["orientation"] == "H"
            and s["span0"] == c0 + 1
            and s["span1"] == c1 - 1
            and s["index"] > ar1
        ]
        candidates.sort(key=lambda s: int(s["index"]))
    assert candidates, "each frame-arrow pair needs at least one stripe"
    return _collect_run(candidates)


def _fill_frame(out: np.ndarray, frame: dict, direction: str, colors: list[int]) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    if direction == "right":
        for idx, color in enumerate(colors[: frame["inner_w"]]):
            out[r0 + 1 : r1, c1 - 1 - idx] = int(color)
    elif direction == "left":
        for idx, color in enumerate(colors[: frame["inner_w"]]):
            out[r0 + 1 : r1, c0 + 1 + idx] = int(color)
    elif direction == "down":
        for idx, color in enumerate(colors[: frame["inner_h"]]):
            out[r1 - 1 - idx, c0 + 1 : c1] = int(color)
    else:
        for idx, color in enumerate(colors[: frame["inner_h"]]):
            out[r0 + 1 + idx, c0 + 1 : c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each red arrow, use its direction to push the colored stripes into the
    adjacent gray frame, and keep only the frames plus the stacked interior fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    frames = _find_frames(g)
    arrows = _find_arrows(g)
    stripes = _find_stripes(g)
    assert frames, "expected at least one frame"
    assert arrows, "expected at least one arrow"
    assert len(frames) == len(arrows), "each frame must have one arrow"

    out = np.full_like(g, BG)
    out[g == FRAME] = FRAME

    used_frames: set[int] = set()
    for arrow in arrows:
        matches = [idx for idx, frame in enumerate(frames) if _matches(frame, arrow)]
        assert len(matches) == 1, "each arrow must point to exactly one frame"
        frame_idx = matches[0]
        assert frame_idx not in used_frames, "frames must not share arrows"
        used_frames.add(frame_idx)

        frame = frames[frame_idx]
        ordered = _frame_stripes(frame, arrow, stripes)
        _fill_frame(out, frame, arrow["direction"], [int(s["color"]) for s in ordered])

    assert len(used_frames) == len(frames), "every frame must be paired"
    return out


def _build_object(
    direction: str,
    inner_h: int,
    inner_w: int,
    stripe_colors: list[int],
    lead_gap: int,
    tip_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(stripe_colors)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    extent = 2 * count + lead_gap + 2

    if direction in {"left", "right"}:
        patch_h = frame_h
        patch_w = frame_w + extent
        fr0, fr1 = 0, frame_h - 1
        if direction == "right":
            fc0 = extent
        else:
            fc0 = 0
        fc1 = fc0 + frame_w - 1
    else:
        patch_h = frame_h + extent
        patch_w = frame_w
        fc0, fc1 = 0, frame_w - 1
        if direction == "down":
            fr0 = extent
        else:
            fr0 = 0
        fr1 = fr0 + frame_h - 1

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = np.full_like(inp, BG)
    _paint_frame(inp, fr0, fc0, fr1, fc1)
    _paint_frame(out, fr0, fc0, fr1, fc1)

    if direction == "right":
        tip_r = fr0 + 1 + tip_offset
        tip_c = fc0 - 1
        for dr, dc in ARROW_OFFSETS[direction]:
            inp[tip_r + dr, tip_c + dc] = ARROW
        for idx, color in enumerate(stripe_colors):
            col = fc0 - 4 - lead_gap - 2 * idx
            inp[fr0 + 1 : fr1, col] = int(color)
        _fill_frame(out, {"bbox": (fr0, fc0, fr1, fc1), "inner_h": inner_h, "inner_w": inner_w}, direction, stripe_colors)
    elif direction == "left":
        tip_r = fr0 + 1 + tip_offset
        tip_c = fc1 + 1
        for dr, dc in ARROW_OFFSETS[direction]:
            inp[tip_r + dr, tip_c + dc] = ARROW
        for idx, color in enumerate(stripe_colors):
            col = fc1 + 4 + lead_gap + 2 * idx
            inp[fr0 + 1 : fr1, col] = int(color)
        _fill_frame(out, {"bbox": (fr0, fc0, fr1, fc1), "inner_h": inner_h, "inner_w": inner_w}, direction, stripe_colors)
    elif direction == "down":
        tip_r = fr0 - 1
        tip_c = fc0 + 1 + tip_offset
        for dr, dc in ARROW_OFFSETS[direction]:
            inp[tip_r + dr, tip_c + dc] = ARROW
        for idx, color in enumerate(stripe_colors):
            row = fr0 - 4 - lead_gap - 2 * idx
            inp[row, fc0 + 1 : fc1] = int(color)
        _fill_frame(out, {"bbox": (fr0, fc0, fr1, fc1), "inner_h": inner_h, "inner_w": inner_w}, direction, stripe_colors)
    else:
        tip_r = fr1 + 1
        tip_c = fc0 + 1 + tip_offset
        for dr, dc in ARROW_OFFSETS[direction]:
            inp[tip_r + dr, tip_c + dc] = ARROW
        for idx, color in enumerate(stripe_colors):
            row = fr1 + 4 + lead_gap + 2 * idx
            inp[row, fc0 + 1 : fc1] = int(color)
        _fill_frame(out, {"bbox": (fr0, fc0, fr1, fc1), "inner_h": inner_h, "inner_w": inner_w}, direction, stripe_colors)

    return inp, out


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    view = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    assert np.all(view[mask] == BG), "patches must not overlap"
    view[mask] = patch[mask]


def _compose_objects(
    input_patches: list[np.ndarray],
    output_patches: list[np.ndarray],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    patches = list(zip(input_patches, output_patches))
    rng.shuffle(patches)
    margin_top = rng.randint(1, 3)
    margin_bottom = rng.randint(1, 3)
    margin_left = rng.randint(1, 3)
    margin_right = rng.randint(1, 3)
    extra_h = rng.randint(0, 3)
    extra_w = rng.randint(0, 3)

    if len(patches) == 1:
        inp_patch, out_patch = patches[0]
        h0, w0 = inp_patch.shape
        grid_h = h0 + margin_top + margin_bottom + extra_h
        grid_w = w0 + margin_left + margin_right + extra_w
        top = margin_top + rng.randint(0, extra_h)
        left = margin_left + rng.randint(0, extra_w)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        expected = np.full_like(grid, BG)
        _paste(grid, inp_patch, top, left)
        _paste(expected, out_patch, top, left)
        return grid, expected

    (inp1, out1), (inp2, out2) = patches
    h1, w1 = inp1.shape
    h2, w2 = inp2.shape
    gap = rng.randint(4, 6)
    row_size = (max(h1, h2), w1 + gap + w2)
    col_size = (h1 + gap + h2, max(w1, w2))

    row_score = max(row_size)
    col_score = max(col_size)
    if abs(row_score - col_score) <= 3:
        layout = rng.choice(["row", "col"])
    else:
        layout = "row" if row_score < col_score else "col"

    if layout == "row":
        block_h, block_w = row_size
        grid_h = block_h + margin_top + margin_bottom + extra_h
        grid_w = block_w + margin_left + margin_right + extra_w
        top0 = margin_top + rng.randint(0, extra_h)
        left0 = margin_left + rng.randint(0, extra_w)
        p1_top = top0 + rng.randint(0, block_h - h1)
        p2_top = top0 + rng.randint(0, block_h - h2)
        p1_left = left0
        p2_left = left0 + w1 + gap
    else:
        block_h, block_w = col_size
        grid_h = block_h + margin_top + margin_bottom + extra_h
        grid_w = block_w + margin_left + margin_right + extra_w
        top0 = margin_top + rng.randint(0, extra_h)
        left0 = margin_left + rng.randint(0, extra_w)
        p1_top = top0
        p2_top = top0 + h1 + gap
        p1_left = left0 + rng.randint(0, block_w - w1)
        p2_left = left0 + rng.randint(0, block_w - w2)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full_like(grid, BG)
    _paste(grid, inp1, p1_top, p1_left)
    _paste(grid, inp2, p2_top, p2_left)
    _paste(expected, out1, p1_top, p1_left)
    _paste(expected, out2, p2_top, p2_left)
    return grid, expected


def _random_object(rng: random.Random, direction: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    direction = direction or rng.choice(DIRECTIONS)
    inner_h = rng.randint(3, 5)
    inner_w = rng.randint(3, 5)
    depth = inner_w if direction in {"left", "right"} else inner_h
    allow_max = min(len(STRIPE_PALETTE), depth + 2)
    overflow = rng.random() < 0.5
    if overflow:
        stripe_count = rng.randint(depth + 1, allow_max)
    else:
        stripe_count = rng.randint(1, depth)
    stripe_colors = rng.sample(STRIPE_PALETTE, stripe_count)
    lead_gap = rng.randint(0, 1)
    tip_offset = rng.randrange(inner_h if direction in {"left", "right"} else inner_w)
    return _build_object(direction, inner_h, inner_w, stripe_colors, lead_gap, tip_offset)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = 1 if rng.random() < 0.4 else 2
    if obj_count == 2 and rng.random() < 0.75:
        chosen_dirs = rng.sample(list(DIRECTIONS), 2)
    else:
        chosen_dirs = [rng.choice(DIRECTIONS) for _ in range(obj_count)]

    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []
    for direction in chosen_dirs:
        inp, out = _random_object(rng, direction)
        input_patches.append(inp)
        output_patches.append(out)

    grid, expected = _compose_objects(input_patches, output_patches, rng)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

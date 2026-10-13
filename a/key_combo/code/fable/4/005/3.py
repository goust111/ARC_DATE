from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, paste


_BG = Color.BLACK
_MAX_ATTEMPTS = 40


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for c in range(c0, c1 + 1):
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    return coords


def _side_counts(g: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int, int, int]:
    top = int(np.sum(g[r0, c0 : c1 + 1] == color))
    bottom = int(np.sum(g[r1, c0 : c1 + 1] == color))
    left = int(np.sum(g[r0 : r1 + 1, c0] == color))
    right = int(np.sum(g[r0 : r1 + 1, c1] == color))
    return top, bottom, left, right


def _gap_limit(perimeter: int) -> int:
    return max(4, perimeter // 5)


def _frame_candidates_for_color(g: np.ndarray, color: int) -> list[dict]:
    coords = np.argwhere(g == color)
    if coords.shape[0] < 7:
        return []

    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    total_color_pixels = int(coords.shape[0])
    candidates: list[dict] = []

    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            h = r1 - r0 + 1
            if h < 3:
                continue
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    w = c1 - c0 + 1
                    if w < 3:
                        continue

                    border = _border_coords(r0, c0, r1, c1)
                    vals = np.array([int(g[r, c]) for r, c in border], dtype=int)
                    if bool(np.any((vals != _BG) & (vals != color))):
                        continue

                    perimeter = len(border)
                    border_color_count = int(np.sum(vals == color))
                    missing = perimeter - border_color_count
                    if missing > _gap_limit(perimeter):
                        continue
                    if total_color_pixels - border_color_count > 1:
                        continue

                    top, bottom, left, right = _side_counts(g, color, r0, c0, r1, c1)
                    if top < min(2, w) or bottom < min(2, w) or left < min(2, h) or right < min(2, h):
                        continue

                    interior = g[r0 + 1 : r1, c0 + 1 : c1]
                    inner_non_bg = np.argwhere(interior != _BG)
                    if inner_non_bg.shape[0] != 1:
                        continue
                    sr, sc = map(int, inner_non_bg[0])
                    fill_color = int(interior[sr, sc])
                    if fill_color == int(color):
                        continue

                    missing_cells = [(r, c) for r, c in border if int(g[r, c]) == _BG]
                    candidates.append(
                        {
                            "color": int(color),
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "r0": int(r0),
                            "c0": int(c0),
                            "r1": int(r1),
                            "c1": int(c1),
                            "fill_color": fill_color,
                            "stray": (int(r0 + 1 + sr), int(c0 + 1 + sc)),
                            "missing": int(missing),
                            "missing_cells": missing_cells,
                            "border_count": int(border_color_count),
                            "area": int(h * w),
                        }
                    )

    return candidates


def _find_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != _BG):
        candidates = _frame_candidates_for_color(g, color)
        if not candidates:
            continue
        candidates.sort(key=lambda d: (d["border_count"], -d["missing"], d["area"]), reverse=True)
        best = candidates[0]
        frames.append(best)

    assert frames, "expected at least one rectangular frame"
    assert len({f["color"] for f in frames}) == len(frames), "frame colors must be unique"
    assert len({f["bbox"] for f in frames}) == len(frames), "frame bboxes must be unique"

    for i, a in enumerate(frames):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in frames[i + 1 :]:
            br0, bc0, br1, bc1 = b["bbox"]
            disjoint = ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0
            assert disjoint, "frames must not overlap or nest"

    frames.sort(key=lambda d: d["bbox"])
    return frames


def _complete_and_fill(out: np.ndarray, frame: dict) -> None:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    for r, c in _border_coords(r0, c0, r1, c1):
        out[r, c] = color
    out[r0 + 1 : r1, c0 + 1 : c1] = int(frame["fill_color"])


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict], set[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    frames = _find_frames(g)
    color_to_frame = {int(frame["color"]): i for i, frame in enumerate(frames)}
    broken = [i for i, frame in enumerate(frames) if int(frame["missing"]) > 0]
    assert broken, "expected at least one broken frame"

    out = g.copy()
    active: set[int] = set()
    queue = list(broken)
    while queue:
        i = int(queue.pop(0))
        if i in active:
            continue
        frame = frames[i]
        _complete_and_fill(out, frame)
        active.add(i)

        next_color = int(frame["fill_color"])
        if next_color not in color_to_frame:
            continue
        j = int(color_to_frame[next_color])
        if j in active:
            raise AssertionError("the color-naming chain must be acyclic")
        if int(frames[j]["missing"]) == 0:
            queue.append(j)
        else:
            assert j in broken, "a fill color may only ignite an originally closed frame"

    return out, frames, active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair broken rectangular fuses and flood only the frames reached by the color-naming chain.

    Input structure:
    - A black background contains separated one-pixel rectangular outlines with distinct frame colors.
    - At least one outline is missing a few black border cells; the other outlines are already closed.
    - Each frame interior contains exactly one stray pixel whose color differs from the frame color, and outside distractor pixels never use frame colors.

    Rule:
    - Complete every broken frame with its own frame color, then fill its whole interior with its stray pixel color.
    - Whenever that fill color equals the color of an originally closed frame, fill that frame's interior with its own stray color and continue the same test.
    - Closed frames not reached by this chain, and all outside distractor pixels, remain unchanged.

    Key insight:
    - Flooding is caused by the repair action and then propagated by visible color equality between a fill color and the next frame color, not by every closed frame's local contents.
    """
    out, _frames, _active = _solve(grid)
    return out


def _empty_frame_patch(h: int, w: int, color: int) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _place_frame_bboxes(H: int, W: int, count: int) -> list[dict] | None:
    # Partition the canvas into a randomly oriented set of non-overlapping slots.
    # This is constructive and bounded: unlike recursive backtracking, every chosen
    # layout is known to fit before any frame dimensions or offsets are sampled.
    gap = 2
    layouts: list[tuple[int, int]] = []
    for rows in range(1, count + 1):
        cols = (count + rows - 1) // rows
        if rows * cols - count > 1:
            continue
        if H - gap * (rows - 1) >= 4 * rows and W - gap * (cols - 1) >= 4 * cols:
            layouts.append((rows, cols))
    if not layouts:
        return None

    rows, cols = random.choice(layouts)

    def split_span(total: int, parts: int) -> list[int]:
        base, extra = divmod(total - gap * (parts - 1), parts)
        spans = [base + (i < extra) for i in range(parts)]
        random.shuffle(spans)
        return spans

    row_heights = split_span(H, rows)
    col_widths = split_span(W, cols)
    row_starts: list[int] = []
    col_starts: list[int] = []
    cursor = 0
    for span in row_heights:
        row_starts.append(cursor)
        cursor += span + gap
    cursor = 0
    for span in col_widths:
        col_starts.append(cursor)
        cursor += span + gap

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(slots)
    specs: list[dict] = []
    for slot_r, slot_c in slots[:count]:
        slot_h = row_heights[slot_r]
        slot_w = col_widths[slot_c]
        h = random.randint(4, min(7, slot_h))
        w = random.randint(4, min(8, slot_w))
        r0 = row_starts[slot_r] + random.randint(0, slot_h - h)
        c0 = col_starts[slot_c] + random.randint(0, slot_w - w)
        specs.append(
            {
                "h": h,
                "w": w,
                "r0": r0,
                "c0": c0,
                "r1": r0 + h - 1,
                "c1": c0 + w - 1,
            }
        )
    return specs


def _choose_gap_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    options = [pt for pt in _border_coords(r0, c0, r1, c1) if pt not in corners]
    random.shuffle(options)

    chosen: list[tuple[int, int]] = []
    target = random.randint(1, min(3, len(options)))
    side_taken: set[str] = set()
    for r, c in options:
        if r == r0:
            side = "top"
        elif r == r1:
            side = "bottom"
        elif c == c0:
            side = "left"
        else:
            side = "right"
        if side in side_taken and len(side_taken) < 2:
            continue
        chosen.append((r, c))
        side_taken.add(side)
        if len(chosen) == target:
            break
    return chosen or options[:1]


def _render_input(H: int, W: int, frames: list[dict], distractor_count: int) -> np.ndarray:
    grid = np.full((H, W), _BG, dtype=int)
    for frame in frames:
        patch = _empty_frame_patch(frame["h"], frame["w"], frame["color"])
        paste(grid, patch, frame["r0"], frame["c0"], transparent=_BG)

    for frame in frames:
        for r, c in frame["missing_cells"]:
            grid[r, c] = _BG
        sr, sc = frame["stray"]
        grid[sr, sc] = int(frame["fill_color"])

    frame_bbox_mask = np.zeros((H, W), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    non_frame_colors = sorted(set(Color.NOT_BLACK) - {int(frame["color"]) for frame in frames})
    candidates = [(r, c) for r in range(H) for c in range(W) if not frame_bbox_mask[r, c] and grid[r, c] == _BG]
    random.shuffle(candidates)
    for r, c in candidates[:distractor_count]:
        grid[r, c] = random.choice(non_frame_colors)
    return grid


def _generated_expected(grid: np.ndarray, frames: list[dict], head: int) -> tuple[np.ndarray, set[int]] | None:
    out = grid.copy()
    by_color = {int(frame["color"]): i for i, frame in enumerate(frames)}
    active: set[int] = set()
    cur = int(head)
    while True:
        if cur in active:
            return None
        _complete_and_fill(out, frames[cur])
        active.add(cur)
        nxt = int(frames[cur]["fill_color"])
        if nxt not in by_color:
            break
        cur = int(by_color[nxt])
    return out, active


def _semantic_check(inp: np.ndarray, expected: np.ndarray, frames: list[dict], active: set[int], head: int) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(active) < 2:
        return False
    if len(frames) - len(active) < 1:
        return False
    if not frames[head]["missing_cells"]:
        return False

    for r, c in frames[head]["missing_cells"]:
        if int(expected[r, c]) != int(frames[head]["color"]):
            return False

    untouched = False
    for i, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        if i in active:
            fill = int(frame["fill_color"])
            if not bool(np.all(expected[r0 + 1 : r1, c0 + 1 : c1] == fill)):
                return False
        else:
            if not bool(np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], expected[r0 : r1 + 1, c0 : c1 + 1])):
                return False
            if bool(np.any(inp[r0 + 1 : r1, c0 + 1 : c1] == _BG)):
                untouched = True
    if not untouched:
        return False

    try:
        solved, found_frames, found_active = _solve(inp)
    except Exception:
        return False
    if len(found_frames) != len(frames):
        return False
    if len(found_active) != len(active):
        return False
    return bool(np.array_equal(solved, expected))


def _attempt_generate() -> np.ndarray | None:
    frame_count = random.randint(3, 5)
    H = random.randint(18, 30)
    W = random.randint(18, 30)
    placed = _place_frame_bboxes(H, W, frame_count)
    if placed is None:
        return None

    frame_colors = random.sample(Color.NOT_BLACK, frame_count)
    non_frame_colors = sorted(set(Color.NOT_BLACK) - set(frame_colors))
    if not non_frame_colors:
        return None

    chain_len = random.randint(2, min(4, frame_count - 1))
    chain = random.sample(range(frame_count), chain_len)
    head = chain[0]

    fill_colors = {chain[i]: frame_colors[chain[i + 1]] for i in range(chain_len - 1)}
    fill_colors[chain[-1]] = random.choice(non_frame_colors)
    for idx in range(frame_count):
        if idx not in fill_colors:
            fill_colors[idx] = random.choice(non_frame_colors)

    frames: list[dict] = []
    for idx, spec in enumerate(placed):
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        r1, c1 = int(spec["r1"]), int(spec["c1"])
        frame = {
            "h": int(spec["h"]),
            "w": int(spec["w"]),
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "bbox": (r0, c0, r1, c1),
            "color": int(frame_colors[idx]),
            "fill_color": int(fill_colors[idx]),
            "missing_cells": [],
            "missing": 0,
        }
        frame["stray"] = (random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1))
        frames.append(frame)

    frames[head]["missing_cells"] = _choose_gap_cells(frames[head])
    frames[head]["missing"] = len(frames[head]["missing_cells"])

    distractor_count = random.randint(2, 6)
    inp = _render_input(H, W, frames, distractor_count)
    expected_info = _generated_expected(inp, frames, head)
    if expected_info is None:
        return None
    expected, active = expected_info
    if not _semantic_check(inp, expected, frames, active, head):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fuse-repair chain input")

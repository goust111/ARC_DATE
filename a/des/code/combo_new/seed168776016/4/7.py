import random
from collections import deque

import numpy as np


COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_by_color(
    grid: np.ndarray, bg: int, blocked: np.ndarray | None = None
) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    if blocked is None:
        blocked = np.zeros((h, w), dtype=bool)

    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or blocked[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or blocked[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _legend_info(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    assert h >= 1 and w >= 3, "expected room for an upper-left legend"
    legend: list[int] = []
    c = 0
    while c < w and int(grid[0, c]) != bg:
        legend.append(int(grid[0, c]))
        c += 1
    assert len(legend) >= 3, "legend must contain at least three colors"
    assert len(set(legend)) == len(legend), "legend colors must be unique"
    mask = np.zeros_like(grid, dtype=bool)
    mask[0, : len(legend)] = True
    return legend, mask


def _is_rect_frame(coords: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, None
    expected = 2 * (height + width) - 4
    if len(coords) != expected:
        return False, None
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return False, None
    return True, (r0, c0, r1, c1)


def _extract_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components_by_color(grid, bg, legend_mask):
        ok, bbox = _is_rect_frame(comp["coords"])
        if not ok:
            continue
        frames.append({"color": int(comp["color"]), "bbox": bbox})
    frames.sort(
        key=lambda item: (
            (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1),
            item["bbox"][2] - item["bbox"][0],
            item["bbox"][3] - item["bbox"][1],
        ),
        reverse=True,
    )
    assert frames, "expected at least one rectangular frame"
    return frames


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    legend, legend_mask = _legend_info(g, bg)
    next_color = {legend[i]: legend[(i + 1) % len(legend)] for i in range(len(legend))}
    frames = _extract_frames(g, bg, legend_mask)

    out = g.copy()
    out[legend_mask] = bg

    for frame in frames:
        assert int(frame["color"]) in next_color, "frame colors must come from the legend"

    for i in range(1, len(frames)):
        prev_color = int(frames[i - 1]["color"])
        cur_color = int(frames[i]["color"])
        if cur_color != next_color[prev_color]:
            r0, c0, r1, c1 = frames[i]["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = bg
            break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the upper-left color-cycle legend and erase the first inner frame whose color breaks the legend order.

    Input structure:
    - The background is the most common color.
    - The upper-left corner contains a short horizontal legend: a row of colored
      pixels defining a cyclic color order.
    - Elsewhere there is one nested stack of one-cell-thick hollow rectangular
      frames, plus a small inner pattern inside the innermost frame.

    Rule:
    - Read the upper-left legend row as a color cycle.
    - Sort the rectangular frames from outermost to innermost and compare each
      inner frame against the previous frame.
    - If an inner frame's color is the next color in the legend cycle, keep it.
      At the first mismatch, erase that frame and everything inside its bbox to
      the background color.
    - Always erase the legend itself to background.

    Key insight:
    - The top-left strip is a control legend, and the frame colors must form one
      continuous walk around that cycle; the first broken link clears the whole
      remaining inner suffix.
    """
    return _apply_rule(grid)


def _pattern_templates() -> list[np.ndarray]:
    return [
        np.array([[1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    ]


def _orient_template(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    k = rng.randrange(4)
    return np.rot90(out, k)


def _choose_pattern(inner_h: int, inner_w: int, rng: random.Random) -> np.ndarray:
    usable_h = max(1, inner_h - 2)
    usable_w = max(1, inner_w - 2)
    candidates: list[np.ndarray] = []
    for template in _pattern_templates():
        oriented = _orient_template(template, rng)
        if oriented.shape[0] <= usable_h and oriented.shape[1] <= usable_w:
            candidates.append(oriented)
    if not candidates:
        return np.array([[1]], dtype=bool)
    return rng.choice(candidates)


def _frame_sequence(
    legend: list[int], n_frames: int, rng: random.Random
) -> tuple[list[int], bool]:
    start_idx = rng.randrange(len(legend))
    break_chain = bool(rng.randrange(2))
    seq = [legend[start_idx]]
    break_at = rng.randrange(1, n_frames) if break_chain else -1
    idx = start_idx
    for i in range(1, n_frames):
        expected_idx = (idx + 1) % len(legend)
        expected_color = legend[expected_idx]
        if i == break_at:
            options = [color for color in legend if color != expected_color]
            color = rng.choice(options)
            idx = legend.index(color)
        elif break_chain and i > break_at:
            color = rng.choice(legend)
            idx = legend.index(color)
        else:
            color = expected_color
            idx = expected_idx
        seq.append(color)
    return seq, break_chain


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _nested_bboxes(
    top: int, left: int, inner_h: int, inner_w: int, n_frames: int
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    outer_h = inner_h + 2 + 4 * (n_frames - 1)
    outer_w = inner_w + 2 + 4 * (n_frames - 1)
    r0 = top
    c0 = left
    r1 = top + outer_h - 1
    c1 = left + outer_w - 1
    for _ in range(n_frames):
        boxes.append((r0, c0, r1, c1))
        r0 += 2
        c0 += 2
        r1 -= 2
        c1 -= 2
    return boxes


def _place_pattern(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    mask = _choose_pattern(inner_h, inner_w, rng)
    margin_top = 1 if inner_h - mask.shape[0] >= 2 else 0
    margin_left = 1 if inner_w - mask.shape[1] >= 2 else 0
    top_min = r0 + 1 + margin_top
    left_min = c0 + 1 + margin_left
    top_max = r1 - margin_top - mask.shape[0]
    left_max = c1 - margin_left - mask.shape[1]
    pr = rng.randint(top_min, top_max)
    pc = rng.randint(left_min, left_max)
    for rr in range(mask.shape[0]):
        for cc in range(mask.shape[1]):
            if mask[rr, cc]:
                grid[pr + rr, pc + cc] = color


def _build_input(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(COLORS)
    usable_colors = [c for c in COLORS if c != bg]
    legend_len = rng.randint(3, 6)
    legend = rng.sample(usable_colors, legend_len)
    remaining = [c for c in usable_colors if c not in legend]
    if not remaining:
        return None
    pattern_color = rng.choice(remaining)

    n_frames = rng.randint(2, 5)
    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 6)
    outer_h = inner_h + 2 + 4 * (n_frames - 1)
    outer_w = inner_w + 2 + 4 * (n_frames - 1)

    top_margin = rng.randint(3, 5)
    left_margin = rng.randint(1, 4)
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 4)

    height = top_margin + outer_h + bottom_margin
    width = max(left_margin + outer_w + right_margin, legend_len + rng.randint(1, 4))
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[0, :legend_len] = legend

    boxes = _nested_bboxes(top_margin, left_margin, inner_h, inner_w, n_frames)
    seq, _break_chain = _frame_sequence(legend, n_frames, rng)
    for bbox, color in zip(boxes, seq):
        _draw_frame(grid, bbox, color)
    _place_pattern(grid, boxes[-1], pattern_color, rng)

    # Keep the legend isolated and the structure unambiguous.
    assert np.all(grid[1 : top_margin, :legend_len] == bg), "legend must be separated vertically"

    frame_colors = set(seq)
    assert pattern_color not in frame_colors, "pattern color should not mimic a frame"

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")

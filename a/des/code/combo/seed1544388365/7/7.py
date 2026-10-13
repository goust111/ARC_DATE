from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
FRAME_COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(points) == expected


def _find_priority_sequence(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    for r in range(h):
        if int(grid[r, 0]) == bg:
            continue
        seq: list[int] = []
        c = 0
        while c < w and int(grid[r, c]) != bg:
            seq.append(int(grid[r, c]))
            c += 1
        if len(seq) >= 2 and len(set(seq)) == len(seq):
            return seq
    raise AssertionError("priority sequence not found")


def _largest_frame_for_color(grid: np.ndarray, color: int) -> dict[str, int | tuple[int, int, int, int]]:
    comps = _components(grid == int(color))
    for comp in sorted(comps, key=len, reverse=True):
        box = _bbox(comp)
        if _is_rect_frame(comp, box):
            return {"color": int(color), "bbox": box}
    raise AssertionError(f"frame for color {color} not found")


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, list[int], dict[int, int], int | None, list[dict[str, int | tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    seq = _find_priority_sequence(g, bg)
    priority = {color: idx + 1 for idx, color in enumerate(seq)}

    non_bg = {int(v) for v in np.unique(g) if int(v) != bg}
    fill_candidates = sorted(non_bg - set(seq))
    assert len(fill_candidates) <= 1, "expected at most one fill color"
    fill_color = fill_candidates[0] if fill_candidates else None

    frames = [_largest_frame_for_color(g, color) for color in seq]
    frames.sort(
        key=lambda item: -(
            (int(item["bbox"][2]) - int(item["bbox"][0]) + 1)
            * (int(item["bbox"][3]) - int(item["bbox"][1]) + 1)
        )
    )

    assert len(frames) == len(seq), "priority length must match frame count"
    for idx in range(len(frames) - 1):
        outer = frames[idx]["bbox"]
        inner = frames[idx + 1]["bbox"]
        or0, oc0, or1, oc1 = map(int, outer)
        ir0, ic0, ir1, ic1 = map(int, inner)
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must be nested"

    return bg, seq, priority, fill_color, frames


def _inner_region_mask(
    shape: tuple[int, int],
    outer_box: tuple[int, int, int, int],
    next_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = map(int, outer_box)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return mask
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    if next_box is not None:
        nr0, nc0, nr1, nc1 = map(int, next_box)
        mask[nr0 : nr1 + 1, nc0 : nc1 + 1] = False
    return mask


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _region_coords(
    outer_box: tuple[int, int, int, int],
    next_box: tuple[int, int, int, int] | None,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, outer_box)
    coords: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if next_box is not None:
                nr0, nc0, nr1, nc1 = map(int, next_box)
                if nr0 <= r <= nr1 and nc0 <= c <= nc1:
                    continue
            coords.append((r, c))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top-left priority row, match colors to nested frame layers,
    and clear the fill inside even-priority layers.
    """
    g = np.asarray(grid, dtype=int)
    bg, _seq, priority, fill_color, frames = _extract_structure(g)
    out = g.copy()

    if fill_color is None:
        return out

    for idx, frame in enumerate(frames):
        color = int(frame["color"])
        if priority[color] % 2 == 1:
            continue
        next_box = None if idx + 1 == len(frames) else frames[idx + 1]["bbox"]
        region = _inner_region_mask(g.shape, frame["bbox"], next_box)
        out[region & (out == int(fill_color))] = int(bg)

    return out


def _make_grid(rng: random.Random) -> np.ndarray:
    layers = rng.randint(2, 5)
    inner_h = rng.randint(1, 7)
    inner_w = rng.randint(1, 9)
    outer_h = inner_h + 2 + 4 * (layers - 1)
    outer_w = inner_w + 2 + 4 * (layers - 1)

    top_min, bottom_min = 2, 1
    left_min, right_min = max(layers + 1, 3), 1
    if outer_h + top_min + bottom_min > 30 or outer_w + left_min + right_min > 30:
        raise ValueError("sampled geometry does not fit canvas bounds")

    top = top_min + rng.randint(0, min(6, 30 - outer_h - top_min - bottom_min))
    bottom = bottom_min + rng.randint(0, min(6, 30 - outer_h - top - bottom_min))
    left = left_min + rng.randint(0, min(8, 30 - outer_w - left_min - right_min))
    right = right_min + rng.randint(0, min(8, 30 - outer_w - left - right_min))

    height = top + outer_h + bottom
    width = left + outer_w + right
    grid = np.zeros((height, width), dtype=int)

    frame_colors = rng.sample(FRAME_COLOR_POOL, layers)
    priority_seq = frame_colors[:]
    rng.shuffle(priority_seq)
    if priority_seq == frame_colors:
        priority_seq = priority_seq[1:] + priority_seq[:1]
    grid[0, :layers] = np.array(priority_seq, dtype=int)

    outer_box = (top, left, top + outer_h - 1, left + outer_w - 1)
    boxes: list[tuple[int, int, int, int]] = []
    for idx, color in enumerate(frame_colors):
        box = (
            outer_box[0] + 2 * idx,
            outer_box[1] + 2 * idx,
            outer_box[2] - 2 * idx,
            outer_box[3] - 2 * idx,
        )
        boxes.append(box)
        _draw_frame(grid, box, color)

    for idx, box in enumerate(boxes):
        next_box = None if idx + 1 == len(boxes) else boxes[idx + 1]
        coords = _region_coords(box, next_box)
        assert coords, "every layer must have an interior region"
        if len(coords) == 1:
            chosen = coords
        else:
            yellow_count = rng.randint(1, len(coords) - 1)
            chosen = rng.sample(coords, yellow_count)
        for r, c in chosen:
            grid[r, c] = YELLOW

    return grid


def _is_valid_instance(grid: np.ndarray) -> bool:
    try:
        bg, seq, priority, fill_color, frames = _extract_structure(grid)
    except AssertionError:
        return False

    if bg != BLACK:
        return False
    if fill_color != YELLOW:
        return False
    if len(seq) != len(frames):
        return False

    for idx, frame in enumerate(frames):
        next_box = None if idx + 1 == len(frames) else frames[idx + 1]["bbox"]
        region = _inner_region_mask(grid.shape, frame["bbox"], next_box)
        if int((grid[region] == YELLOW).sum()) == 0:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    for idx, frame in enumerate(frames):
        next_box = None if idx + 1 == len(frames) else frames[idx + 1]["bbox"]
        region = _inner_region_mask(grid.shape, frame["bbox"], next_box)
        color = int(frame["color"])
        if priority[color] % 2 == 0 and bool((out[region] == YELLOW).any()):
            return False
        if priority[color] % 2 == 1 and not np.array_equal(out[region], grid[region]):
            return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _make_grid(rng)
        except ValueError:
            continue
        if _is_valid_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")

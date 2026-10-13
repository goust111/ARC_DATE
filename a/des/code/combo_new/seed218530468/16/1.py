from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_TRIES = 200
GAP_SAMPLING_TRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int, int, list[int]]] = []
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != BG)
        if len(rows) < 2:
            continue
        r0, r1 = int(rows[0]), int(rows[-1])
        if r1 - r0 + 1 != len(rows):
            continue
        segment = [int(v) for v in grid[r0 : r1 + 1, c]]
        if len(set(segment)) < 2:
            continue
        isolated = True
        for r in range(r0, r1 + 1):
            row_non_bg = 0
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if int(grid[r, nc]) != BG:
                    row_non_bg += 1
            if row_non_bg != 1:
                isolated = False
                break
        if not isolated:
            continue
        side_dist = min(c, w - 1 - c)
        candidates.append((side_dist, -len(segment), c, r0, segment))

    assert candidates, "legend column not found"
    candidates.sort()
    _, _, col, start_row, colors = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[start_row : start_row + len(colors), col] = True
    return colors, mask


def _frame_from_bbox(
    work: np.ndarray,
    frame_mask: np.ndarray,
    frame_color: int,
    bbox: tuple[int, int, int, int],
    max_gaps: int,
) -> dict[str, int | tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter = _perimeter_coords(r0, c0, r1, c1)
    values = [int(work[r, c]) for r, c in perimeter]
    if any(v not in (BG, frame_color) for v in values):
        return None

    gap_count = sum(v == BG for v in values)
    if not (1 <= gap_count <= max_gaps):
        return None

    side_sums = [
        int((work[r0, c0 : c1 + 1] == frame_color).sum()),
        int((work[r1, c0 : c1 + 1] == frame_color).sum()),
        int((work[r0 : r1 + 1, c0] == frame_color).sum()),
        int((work[r0 : r1 + 1, c1] == frame_color).sum()),
    ]
    if any(v == 0 for v in side_sums):
        return None

    inside = work[r0 + 1 : r1, c0 + 1 : c1]
    if inside.size == 0 or not bool((inside == BG).all()):
        return None

    local_points = np.argwhere(frame_mask[r0 : r1 + 1, c0 : c1 + 1])
    for rr, cc in local_points:
        gr, gc = int(r0 + rr), int(c0 + cc)
        if gr not in (r0, r1) and gc not in (c0, c1):
            return None

    colored_perimeter = sum(v == frame_color for v in values)
    if int(local_points.shape[0]) != colored_perimeter:
        return None

    return {"bbox": bbox, "gaps": gap_count}


def _find_frames(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int]]]]:
    work = np.array(grid, dtype=int)
    work[legend_mask] = BG
    non_bg = work[work != BG]
    assert non_bg.size > 0, "frame color not found"

    colors, counts = np.unique(non_bg, return_counts=True)
    frame_color = int(colors[int(np.argmax(counts))])
    frame_mask = work == frame_color
    dilated = _dilate4(frame_mask)

    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    max_gaps = int(legend_mask.sum())
    for comp in _components(dilated):
        original = [(r, c) for r, c in comp if bool(frame_mask[r, c])]
        if not original:
            continue
        rows = [r for r, _ in original]
        cols = [c for _, c in original]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        if bbox in seen_bboxes:
            continue
        frame = _frame_from_bbox(work, frame_mask, frame_color, bbox, max_gaps)
        if frame is None:
            continue
        seen_bboxes.add(bbox)
        frames.append(frame)

    frames.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    assert frames, "no frames detected"
    return frame_color, frames


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _find_legend(g)
    _frame_color, frames = _find_frames(g, legend_mask)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        fill_color = int(legend_colors[int(frame["gaps"]) - 1])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame using a side legend indexed by the number of border gaps.

    Input structure:
    - The background is black.
    - One side of the grid contains a vertical single-pixel color legend, ordered from top to bottom.
    - Elsewhere there are several same-colored hollow rectangular frames with 1-pixel-thick borders.
    - Each frame border contains a small number of 1-pixel background gaps.

    Rule:
    - Locate the side legend and read its colors from top to bottom as indices 1, 2, 3, ...
    - For each hollow frame, count how many background gaps appear on its border.
    - Use the color at that legend index to fill every background cell strictly inside the frame.
    - Keep the frame border unchanged, and keep the border gaps black.

    Key insight: The missing border pixels are not damage; their count is a visible numeric code that selects a fill color from the ordered legend.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sample_gap_positions(height: int, width: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    candidates = _perimeter_coords(0, 0, height - 1, width - 1)
    candidates = [pos for pos in candidates if pos not in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))]
    if len(candidates) < count:
        return None

    for _ in range(GAP_SAMPLING_TRIES):
        order = candidates[:]
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for cell in order:
            if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gaps: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    for r, c in _perimeter_coords(r0, c0, r1, c1):
        grid[r, c] = color
    for gr, gc in gaps:
        grid[r0 + gr, c0 + gc] = BG


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = map(int, mask.shape)
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r1 + pad + 1)
    cc1 = min(w, c1 + pad + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)
    legend_len = rng.randint(2, 5)
    frame_count = rng.randint(1, min(4, legend_len))
    side = rng.choice(["left", "right"])

    frame_color = rng.choice(COLORS)
    legend_colors = rng.sample([c for c in COLORS if c != frame_color], legend_len)
    chosen_gap_counts = rng.sample(list(range(1, legend_len + 1)), frame_count)

    if h < legend_len + 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    legend_col = 0 if side == "left" else w - 1
    legend_row0 = rng.randint(1, h - legend_len - 1)
    for i, color in enumerate(legend_colors):
        grid[legend_row0 + i, legend_col] = color

    reserved = np.zeros((h, w), dtype=bool)
    if side == "left":
        reserved[:, :2] = True
    else:
        reserved[:, w - 2 :] = True

    placed: list[tuple[tuple[int, int, int, int], int]] = []
    for gap_count in chosen_gap_counts:
        placed_ok = False
        for _ in range(PLACEMENT_TRIES):
            fh = rng.randint(4 + int(gap_count >= 5), 9)
            fw = rng.randint(4 + int(gap_count >= 5), 9)
            gaps = _sample_gap_positions(fh, fw, gap_count, rng)
            if gaps is None:
                continue

            max_r0 = h - fh - 1
            if max_r0 < 1:
                continue
            min_c0 = 1
            max_c0 = w - fw - 1
            if side == "left":
                min_c0 = max(min_c0, 4)
            else:
                max_c0 = min(max_c0, w - fw - 5)
            if min_c0 > max_c0:
                continue

            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(min_c0, max_c0)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)

            rr0 = max(0, bbox[0] - 2)
            cc0 = max(0, bbox[1] - 2)
            rr1 = min(h, bbox[2] + 3)
            cc1 = min(w, bbox[3] + 3)
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue

            _draw_frame(grid, bbox, frame_color, gaps)
            _reserve(reserved, bbox, 2)
            placed.append((bbox, gap_count))
            placed_ok = True
            break
        if not placed_ok:
            return None

    try:
        found_legend, legend_mask = _find_legend(grid)
        _frame_color, frames = _find_frames(grid, legend_mask)
    except AssertionError:
        return None

    if found_legend != legend_colors or len(frames) != len(placed):
        return None

    expected = sorted(placed)
    detected = sorted((tuple(frame["bbox"]), int(frame["gaps"])) for frame in frames)  # type: ignore[arg-type]
    if detected != expected:
        return None

    out = main(grid)
    for bbox, gap_count in placed:
        r0, c0, r1, c1 = bbox
        fill_color = legend_colors[gap_count - 1]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not bool((interior == fill_color).all()):
            return None
        if int(out[r0:r1 + 1, c0:c1 + 1].max()) == BG:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

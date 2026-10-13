from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in np.unique(grid):
        if int(color) == BG:
            continue
        for coords in _components(grid == int(color)):
            if _is_rect_frame(coords):
                frames.append(
                    {
                        "bbox": _bbox(coords),
                        "wall_color": int(color),
                        "coords": coords,
                    }
                )
    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # type: ignore[arg-type]
    return frames


def _majority_color(grid: np.ndarray, frame: dict[str, object]) -> int:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
    wall_color = int(frame["wall_color"])
    interior = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)].ravel()
    counts = Counter(int(v) for v in interior if int(v) not in (BG, wall_color))
    if not counts:
        raise ValueError("frame interior must contain vote pixels")
    ordered = counts.most_common()
    if len(ordered) > 1 and int(ordered[0][1]) <= int(ordered[1][1]):
        raise ValueError("vote pixels must have a unique majority color")
    return int(ordered[0][0])


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _expand_bboxes(bboxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape
    n = len(bboxes)
    orig = [tuple(map(int, box)) for box in bboxes]

    tops = np.array([box[0] for box in orig], dtype=int)
    lefts = np.array([box[1] for box in orig], dtype=int)
    bottoms = np.array([box[2] for box in orig], dtype=int)
    rights = np.array([box[3] for box in orig], dtype=int)

    # Internal gaps with even width need a tie-break. Bias the spare cell toward the
    # top/left rectangle, which matches the generator and keeps a 1-cell separator.
    for _ in range(max(8, 2 * n + 2)):
        prev = (tops.copy(), lefts.copy(), bottoms.copy(), rights.copy())

        new_lefts = np.zeros_like(lefts)
        new_rights = np.zeros_like(rights)
        for i, (_r0, c0, _r1, c1) in enumerate(orig):
            left_candidates = [
                oc1
                for j, (_or0, _oc0, _or1, oc1) in enumerate(orig)
                if j != i
                and oc1 < c0
                and _interval_overlap(int(tops[i]), int(bottoms[i]), int(tops[j]), int(bottoms[j]))
            ]
            if left_candidates:
                new_lefts[i] = max(0, (max(left_candidates) + int(c0) - 1) // 2 + 2)
            else:
                new_lefts[i] = 0

            right_candidates = [
                oc0
                for j, (_or0, oc0, _or1, _oc1) in enumerate(orig)
                if j != i
                and oc0 > c1
                and _interval_overlap(int(tops[i]), int(bottoms[i]), int(tops[j]), int(bottoms[j]))
            ]
            if right_candidates:
                new_rights[i] = min(w - 1, (int(c1) + min(right_candidates) - 1) // 2)
            else:
                new_rights[i] = w - 1

        new_tops = np.zeros_like(tops)
        new_bottoms = np.zeros_like(bottoms)
        for i, (r0, _c0, r1, _c1) in enumerate(orig):
            top_candidates = [
                or1
                for j, (_or0, _oc0, or1, _oc1) in enumerate(orig)
                if j != i
                and or1 < r0
                and _interval_overlap(int(new_lefts[i]), int(new_rights[i]), int(new_lefts[j]), int(new_rights[j]))
            ]
            if top_candidates:
                new_tops[i] = max(0, (max(top_candidates) + int(r0) - 1) // 2 + 2)
            else:
                new_tops[i] = 0

            bottom_candidates = [
                or0
                for j, (or0, _oc0, _or1, _oc1) in enumerate(orig)
                if j != i
                and or0 > r1
                and _interval_overlap(int(new_lefts[i]), int(new_rights[i]), int(new_lefts[j]), int(new_rights[j]))
            ]
            if bottom_candidates:
                new_bottoms[i] = min(h - 1, (int(r1) + min(bottom_candidates) - 1) // 2)
            else:
                new_bottoms[i] = h - 1

        tops, lefts, bottoms, rights = new_tops, new_lefts, new_bottoms, new_rights
        if all(np.array_equal(a, b) for a, b in zip(prev, (tops, lefts, bottoms, rights))):
            break

    return [
        (int(tops[i]), int(lefts[i]), int(bottoms[i]), int(rights[i]))
        for i in range(n)
    ]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pick each frame's majority interior color, recolor the frame, clear the votes,
    then expand the rectangular frame outward until neighboring expansions or the
    grid boundary stop it, leaving a 1-cell background gap between neighbors.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    if not frames:
        return g.copy()

    colors = [_majority_color(g, frame) for frame in frames]
    expanded = _expand_bboxes([frame["bbox"] for frame in frames], g.shape)  # type: ignore[list-item]

    out = np.full_like(g, BG)
    for bbox, color in zip(expanded, colors):
        _draw_frame(out, bbox, int(color))
    return out


def _positive_partition(rng: np.random.Generator, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    vals: list[int] = []
    prev = 0
    for cut in cuts + [int(total)]:
        vals.append(int(cut - prev))
        prev = cut
    return vals


def _sample_cell_sizes(rng: np.random.Generator, count: int) -> list[int]:
    hi = {1: 12, 2: 11, 3: 9, 4: 7}[count]
    return [int(rng.integers(6, hi + 1)) for _ in range(count)]


def _sample_layout(rng: np.random.Generator) -> tuple[int, int]:
    mode = str(rng.choice(["row", "col", "grid"]))
    if mode == "row":
        return 1, int(rng.integers(2, 5))
    if mode == "col":
        return int(rng.integers(2, 5)), 1
    return 2, 2


def _layout_bboxes(row_heights: list[int], col_widths: list[int]) -> list[tuple[int, int, int, int, int, int]]:
    row_starts: list[int] = []
    cursor = 0
    for h in row_heights:
        row_starts.append(int(cursor))
        cursor += int(h) + 1

    col_starts: list[int] = []
    cursor = 0
    for w in col_widths:
        col_starts.append(int(cursor))
        cursor += int(w) + 1

    out: list[tuple[int, int, int, int, int, int]] = []
    for i, h in enumerate(row_heights):
        for j, w in enumerate(col_widths):
            r0 = int(row_starts[i])
            c0 = int(col_starts[j])
            out.append((r0, c0, r0 + int(h) - 1, c0 + int(w) - 1, i, j))
    return out


def _place_votes(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    area = len(cells)
    max_colors = 3 if area >= 5 else 2
    color_count = int(rng.choice([2, max_colors])) if max_colors == 3 else 2

    min_votes = max(3, 2 * color_count - 1)
    max_votes = min(area, 8)
    if max_votes < min_votes:
        color_count = 2
        min_votes = 3
        max_votes = min(area, 8)
    total_votes = int(rng.integers(min_votes, max_votes + 1))

    majority_color = int(rng.choice(COLOR_POOL))
    others = [c for c in COLOR_POOL if c != majority_color]
    other_colors = [int(c) for c in rng.choice(others, size=color_count - 1, replace=False)]

    majority_min = total_votes // 2 + 1
    majority_max = total_votes - (color_count - 1)
    majority_count = int(rng.integers(majority_min, majority_max + 1))
    other_counts = _positive_partition(rng, total_votes - majority_count, color_count - 1)

    vote_cells = [cells[idx] for idx in rng.choice(len(cells), size=total_votes, replace=False)]
    rng.shuffle(vote_cells)

    assignments = [majority_color] * majority_count
    for color, count in zip(other_colors, other_counts):
        assignments.extend([int(color)] * int(count))
    rng.shuffle(assignments)

    for (r, c), color in zip(vote_cells, assignments):
        grid[int(r), int(c)] = int(color)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    nrows, ncols = _sample_layout(rng)
    row_heights = _sample_cell_sizes(rng, nrows)
    col_widths = _sample_cell_sizes(rng, ncols)

    h = int(sum(row_heights) + (nrows - 1))
    w = int(sum(col_widths) + (ncols - 1))
    grid = np.full((h, w), BG, dtype=int)

    for r0, c0, r1, c1, row_idx, col_idx in _layout_bboxes(row_heights, col_widths):
        top_margin = int(rng.integers(0, 2)) if row_idx == 0 else 0
        left_margin = int(rng.integers(0, 2)) if col_idx == 0 else 0
        bottom_margin = 1 if row_idx < nrows - 1 else int(rng.integers(0, 2))
        right_margin = 1 if col_idx < ncols - 1 else int(rng.integers(0, 2))

        fr0 = int(r0 + top_margin)
        fc0 = int(c0 + left_margin)
        fr1 = int(r1 - bottom_margin)
        fc1 = int(c1 - right_margin)
        if fr1 - fr0 + 1 < 4 or fc1 - fc0 + 1 < 4:
            return None

        _draw_frame(grid, (fr0, fc0, fr1, fc1), GREY)
        _place_votes(grid, (fr0, fc0, fr1, fc1), rng)

    frames = _extract_frames(grid)
    if len(frames) != nrows * ncols:
        return None

    try:
        out = main(grid)
    except ValueError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

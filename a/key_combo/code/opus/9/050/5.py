from __future__ import annotations

from collections import deque
import itertools

import numpy as np


BG = 0
GUIDE = 1
FILL = 3
FRAME = 4

GEN_HEIGHT_RANGE = (17, 25)  # high exclusive
GEN_WIDTH_RANGE = (17, 25)  # high exclusive
GEN_FRAME_COUNT_CHOICES = (3, 4)
GEN_TOTAL_LINE_COUNTS = (2, 3)
GEN_FRAME_SIDE_RANGE = (4, 9)  # high exclusive
GEN_FRAME_MARGIN = 2
GEN_FRAME_GAP = 1
GEN_LINE_GAP = 3
GEN_ATTEMPTS = 500
GEN_CANDIDATE_CAP = 700


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_frames(grid: np.ndarray) -> list[dict[str, tuple[int, int, int, int]]]:
    frames: list[dict[str, tuple[int, int, int, int]]] = []
    for coords in _components(grid == FRAME):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 4 or width < 4:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((height, width), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        if np.count_nonzero(patch == FRAME) != 2 * height + 2 * width - 4:
            continue
        if not np.all(patch[border] == FRAME):
            continue
        if not np.all(patch[1:-1, 1:-1] == BG):
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "interior": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _guide_rows_cols(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    rows = [r for r in range(1, h - 1) if grid[r, 0] == GUIDE and grid[r, w - 1] == GUIDE]
    cols = [c for c in range(1, w - 1) if grid[0, c] == GUIDE and grid[h - 1, c] == GUIDE]
    return rows, cols


def _draw_guides(base: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    out = np.array(base, copy=True)
    frame_mask = out == FRAME
    for r in rows:
        visible = ~frame_mask[r, :]
        out[r, visible] = GUIDE
    for c in cols:
        visible = ~frame_mask[:, c]
        out[visible, c] = GUIDE
    return out


def _frame_penetrated(frame: dict[str, tuple[int, int, int, int]], rows: list[int], cols: list[int]) -> bool:
    r0, c0, r1, c1 = frame["interior"]
    return any(r0 <= r <= r1 for r in rows) or any(c0 <= c <= c1 for c in cols)


def _frame_graze_only(frame: dict[str, tuple[int, int, int, int]], rows: list[int], cols: list[int]) -> bool:
    if _frame_penetrated(frame, rows, cols):
        return False
    r0, c0, r1, c1 = frame["bbox"]
    return (
        any(r in {r0, r1, r0 - 1, r1 + 1} for r in rows)
        or any(c in {c0, c1, c0 - 1, c1 + 1} for c in cols)
    )


def _classify_box(box: tuple[int, int, int, int], rows: list[int], cols: list[int]) -> str:
    r0, c0, r1, c1 = box
    if any(r0 < r < r1 for r in rows) or any(c0 < c < c1 for c in cols):
        return "hit"
    if any(r in {r0, r1, r0 - 1, r1 + 1} for r in rows) or any(c in {c0, c1, c0 - 1, c1 + 1} for c in cols):
        return "graze"
    return "far"


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = GEN_FRAME_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_spaced_positions(rng: np.random.Generator, lo: int, hi: int, count: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if count == 0:
        return []
    if count == 1:
        return [int(rng.choice(values))]
    combos = [
        combo
        for combo in itertools.combinations(values, count)
        if all(combo[i + 1] - combo[i] >= GEN_LINE_GAP for i in range(len(combo) - 1))
    ]
    if not combos:
        return None
    choice = combos[int(rng.integers(len(combos)))]
    return [int(v) for v in choice]


def _sample_lines(rng: np.random.Generator, h: int, w: int) -> tuple[list[int], list[int]] | None:
    total = int(rng.choice(np.array(GEN_TOTAL_LINE_COUNTS, dtype=int)))
    splits = [(hr, total - hr) for hr in range(total + 1)]
    rng.shuffle(splits)
    for h_count, v_count in splits:
        rows = _sample_spaced_positions(rng, 1, h - 2, h_count)
        cols = _sample_spaced_positions(rng, 1, w - 2, v_count)
        if rows is not None and cols is not None:
            return rows, cols
    return None


def _candidate_boxes(h: int, w: int, rows: list[int], cols: list[int]) -> dict[str, list[tuple[int, int, int, int]]]:
    buckets: dict[str, list[tuple[int, int, int, int]]] = {"hit": [], "graze": [], "far": []}
    for rh in range(*GEN_FRAME_SIDE_RANGE):
        for cw in range(*GEN_FRAME_SIDE_RANGE):
            for r0 in range(GEN_FRAME_MARGIN, h - rh - GEN_FRAME_MARGIN + 1):
                r1 = r0 + rh - 1
                for c0 in range(GEN_FRAME_MARGIN, w - cw - GEN_FRAME_MARGIN + 1):
                    c1 = c0 + cw - 1
                    box = (int(r0), int(c0), int(r1), int(c1))
                    buckets[_classify_box(box, rows, cols)].append(box)
    return buckets


def _choose_frame_labels(rng: np.random.Generator, n_frames: int) -> list[str]:
    labels = ["hit", "graze", "far"]
    if n_frames == 4:
        labels.append(str(rng.choice(np.array(["hit", "graze", "far"], dtype=object))))
    rng.shuffle(labels)
    return labels


def _pick_boxes(
    rng: np.random.Generator,
    candidates: dict[str, list[tuple[int, int, int, int]]],
    n_frames: int,
) -> list[tuple[int, int, int, int]] | None:
    labels = _choose_frame_labels(rng, n_frames)
    pools: dict[str, list[tuple[int, int, int, int]]] = {}
    for label, boxes in candidates.items():
        if label in labels and not boxes:
            return None
        if len(boxes) > GEN_CANDIDATE_CAP:
            idx = rng.choice(len(boxes), size=GEN_CANDIDATE_CAP, replace=False)
            sampled = [boxes[int(i)] for i in idx]
        else:
            sampled = list(boxes)
        rng.shuffle(sampled)
        pools[label] = sampled

    chosen: list[tuple[int, int, int, int]] = []

    def backtrack(i: int) -> bool:
        if i == len(labels):
            return True
        for box in pools[labels[i]]:
            if any(_boxes_conflict(box, prev) for prev in chosen):
                continue
            chosen.append(box)
            if backtrack(i + 1):
                return True
            chosen.pop()
        return False

    return chosen if backtrack(0) else None


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _build_input(h: int, w: int, rows: list[int], cols: list[int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r in rows:
        grid[r, 0] = GUIDE
        grid[r, w - 1] = GUIDE
    for c in cols:
        grid[0, c] = GUIDE
        grid[h - 1, c] = GUIDE
    for box in boxes:
        _paint_frame(grid, box)
    return grid


def _expected_from_semantics(inp: np.ndarray) -> np.ndarray:
    rows, cols = _guide_rows_cols(inp)
    out = _draw_guides(inp, rows, cols)
    for frame in _find_frames(inp):
        ir0, ic0, ir1, ic1 = frame["interior"]
        if np.any(out[ir0 : ir1 + 1, ic0 : ic1 + 1] == GUIDE):
            out[ir0 : ir1 + 1, ic0 : ic1 + 1] = FILL
    return out


def _validate_input(inp: np.ndarray) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not set(map(int, np.unique(inp))).issubset({BG, GUIDE, FRAME}):
        return False

    rows, cols = _guide_rows_cols(inp)
    if len(rows) + len(cols) not in GEN_TOTAL_LINE_COUNTS:
        return False
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        return False

    h, w = inp.shape
    blue = inp == GUIDE
    expected_blue = np.zeros_like(blue, dtype=bool)
    for r in rows:
        expected_blue[r, 0] = True
        expected_blue[r, w - 1] = True
    for c in cols:
        expected_blue[0, c] = True
        expected_blue[h - 1, c] = True
    if not np.array_equal(blue, expected_blue):
        return False

    frames = _find_frames(inp)
    if len(frames) not in GEN_FRAME_COUNT_CHOICES:
        return False

    hit_flags = [_frame_penetrated(frame, rows, cols) for frame in frames]
    graze_flags = [_frame_graze_only(frame, rows, cols) for frame in frames]
    if not any(hit_flags):
        return False
    if not any(not flag for flag in hit_flags):
        return False
    if not any(graze_flags):
        return False
    return True


def _validate_output(inp: np.ndarray, out: np.ndarray) -> bool:
    if out.shape != inp.shape:
        return False

    rows, cols = _guide_rows_cols(inp)
    frames = _find_frames(inp)
    expected = _draw_guides(inp, rows, cols)

    yellow_mask = inp == FRAME
    if not np.all(out[yellow_mask] == FRAME):
        return False

    hit_count = 0
    untouched_count = 0
    for frame in frames:
        ir0, ic0, ir1, ic1 = frame["interior"]
        interior = out[ir0 : ir1 + 1, ic0 : ic1 + 1]
        if np.any(expected[ir0 : ir1 + 1, ic0 : ic1 + 1] == GUIDE):
            if not np.all(interior == FILL):
                return False
            expected[ir0 : ir1 + 1, ic0 : ic1 + 1] = FILL
            hit_count += 1
        else:
            if not np.all(interior == BG):
                return False
            untouched_count += 1

    if hit_count < 1 or untouched_count < 1:
        return False
    return np.array_equal(out, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """Draw guide lines from opposite-edge blue markers and fill only the yellow hollow frames whose interiors are pierced.

    Input structure:
    - Black is the background.
    - Blue marker pixels appear in opposite-border pairs: left-right pairs share one row, and top-bottom pairs share one column.
    - Yellow objects are single-pixel-thick hollow rectangles with black interiors, separated from one another.

    Rule:
    - For each left-right pair, draw a blue horizontal line across the full row; for each top-bottom pair, draw a blue vertical line across the full column.
    - Yellow frame-border pixels keep their yellow color, so guides are visible everywhere except on the frame outlines.
    - Inspect each frame's strict interior rectangle only; if any blue guide pixel lies inside that enclosed area, fill the whole interior green, otherwise leave it black.

    Key insight:
    - The trigger is penetration into the enclosed interior space, not merely touching or running alongside the outside of the frame.
    """
    g = np.asarray(grid, dtype=int)
    rows, cols = _guide_rows_cols(g)
    out = _draw_guides(g, rows, cols)
    for frame in _find_frames(g):
        ir0, ic0, ir1, ic1 = frame["interior"]
        if np.any(out[ir0 : ir1 + 1, ic0 : ic1 + 1] == GUIDE):
            out[ir0 : ir1 + 1, ic0 : ic1 + 1] = FILL
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_HEIGHT_RANGE))
    w = int(rng.integers(*GEN_WIDTH_RANGE))

    lines = _sample_lines(rng, h, w)
    if lines is None:
        return None
    rows, cols = lines

    candidates = _candidate_boxes(h, w, rows, cols)
    n_frames = int(rng.choice(np.array(GEN_FRAME_COUNT_CHOICES, dtype=int)))
    boxes = _pick_boxes(rng, candidates, n_frames)
    if boxes is None:
        return None

    inp = _build_input(h, w, rows, cols, boxes)
    if not _validate_input(inp):
        return None

    out = main(inp)
    if not _validate_output(inp, out):
        return None
    if not np.array_equal(out, _expected_from_semantics(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

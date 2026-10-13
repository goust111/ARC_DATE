from __future__ import annotations

import random

import numpy as np


FRAME_COLOR = 5
POINTER_COLOR = 2
GRID_MIN = 12
GRID_MAX = 28
FRAME_COUNT_MIN = 2
FRAME_COUNT_MAX = 5
INNER_SIZE_CHOICES = (3, 5, 7)
GENERATION_RETRIES = 300
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _constant_step(lines: list[int]) -> int | None:
    if len(lines) < 2:
        return None
    diffs = [int(b - a) for a, b in zip(lines, lines[1:])]
    if len(set(diffs)) != 1 or diffs[0] < 2:
        return None
    return diffs[0]


def _parse_layout(grid: np.ndarray) -> tuple[str, list[tuple[int, int, int, int]]]:
    grey_cells = np.argwhere(grid == FRAME_COLOR)
    assert grey_cells.size > 0, "expected at least one grey frame"

    r0, c0 = np.min(grey_cells, axis=0).tolist()
    r1, c1 = np.max(grey_cells, axis=0).tolist()
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    grey_mask = sub == FRAME_COLOR

    row_lines = [r for r in range(sub.shape[0]) if bool(np.all(grey_mask[r, :]))]
    col_lines = [c for c in range(sub.shape[1]) if bool(np.all(grey_mask[:, c]))]

    candidates: list[tuple[str, list[int], list[int]]] = []

    if (
        len(row_lines) == 2
        and row_lines[0] == 0
        and row_lines[1] == sub.shape[0] - 1
        and len(col_lines) >= 3
        and col_lines[0] == 0
        and col_lines[-1] == sub.shape[1] - 1
    ):
        step = _constant_step(col_lines)
        if step is not None:
            expected = np.zeros_like(grey_mask, dtype=bool)
            expected[[0, sub.shape[0] - 1], :] = True
            expected[:, col_lines] = True
            if np.array_equal(grey_mask, expected):
                candidates.append(("horizontal", row_lines, col_lines))

    if (
        len(col_lines) == 2
        and col_lines[0] == 0
        and col_lines[1] == sub.shape[1] - 1
        and len(row_lines) >= 3
        and row_lines[0] == 0
        and row_lines[-1] == sub.shape[0] - 1
    ):
        step = _constant_step(row_lines)
        if step is not None:
            expected = np.zeros_like(grey_mask, dtype=bool)
            expected[:, [0, sub.shape[1] - 1]] = True
            expected[row_lines, :] = True
            if np.array_equal(grey_mask, expected):
                candidates.append(("vertical", row_lines, col_lines))

    assert len(candidates) == 1, "expected one clean row or column of same-size shared-border frames"

    orientation, row_lines, col_lines = candidates[0]
    frames: list[tuple[int, int, int, int]] = []
    if orientation == "horizontal":
        for i in range(len(col_lines) - 1):
            frames.append((r0, c0 + col_lines[i], r1, c0 + col_lines[i + 1]))
    else:
        for i in range(len(row_lines) - 1):
            frames.append((r0 + row_lines[i], c0, r0 + row_lines[i + 1], c1))

    assert len(frames) >= FRAME_COUNT_MIN, "expected multiple frames"
    return orientation, frames


def _pointer_direction(inner: np.ndarray) -> tuple[int, int] | None:
    points = np.argwhere(inner == POINTER_COLOR)
    assert points.shape[0] <= 1, "each frame may contain at most one pointer"
    if points.shape[0] == 0:
        return None

    ih, iw = map(int, inner.shape)
    assert ih >= 3 and iw >= 3 and ih % 2 == 1 and iw % 2 == 1, "pointer midpoints must be unique"
    pr, pc = map(int, points[0])
    lookup = {
        (0, iw // 2): (-1, 0),
        (ih - 1, iw // 2): (1, 0),
        (ih // 2, 0): (0, -1),
        (ih // 2, iw - 1): (0, 1),
    }
    assert (pr, pc) in lookup, "pointer must lie at the midpoint of one interior side"
    return lookup[(pr, pc)]


def _analyze(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]], list[np.ndarray], list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    orientation, frames = _parse_layout(g)

    patterns: list[np.ndarray] = []
    outgoing: list[int | None] = [None] * len(frames)
    pointer_count = 0

    for idx, (r0, c0, r1, c1) in enumerate(frames):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern = np.full(inner.shape, bg, dtype=int)
        content_mask = (inner != bg) & (inner != FRAME_COLOR) & (inner != POINTER_COLOR)
        pattern[content_mask] = inner[content_mask]
        patterns.append(pattern)

        direction = _pointer_direction(inner)
        if direction is None:
            continue

        pointer_count += 1
        dr, dc = direction
        if orientation == "horizontal":
            assert dr == 0 and dc in (-1, 1), "horizontal frame chains must use left/right pointers"
            nxt = idx + dc
        else:
            assert dc == 0 and dr in (-1, 1), "vertical frame chains must use up/down pointers"
            nxt = idx + dr
        assert 0 <= nxt < len(frames), "pointer must target an adjacent frame"
        outgoing[idx] = nxt

    assert pointer_count == len(frames) - 1, "expected exactly one terminal frame without a pointer"

    indegree = [0] * len(frames)
    terminal_count = 0
    for nxt in outgoing:
        if nxt is None:
            terminal_count += 1
        else:
            indegree[nxt] += 1
    assert terminal_count == 1, "expected exactly one terminal frame"
    assert all(deg <= 1 for deg in indegree), "chain cannot branch"

    starts = [idx for idx, deg in enumerate(indegree) if deg == 0]
    assert len(starts) == 1, "expected exactly one start frame"

    chain: list[int] = []
    seen: set[int] = set()
    cur: int | None = starts[0]
    while cur is not None:
        assert cur not in seen, "chain must be acyclic"
        seen.add(cur)
        chain.append(cur)
        cur = outgoing[cur]
    assert len(chain) == len(frames), "chain must cover every frame"

    return bg, frames, patterns, chain


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift frame contents along the pointer-defined chain and erase the pointers.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, patterns, chain = _analyze(g)

    transferred: list[np.ndarray | None] = [None] * len(frames)
    transferred[chain[0]] = patterns[chain[0]]
    for i in range(1, len(chain)):
        transferred[chain[i]] = patterns[chain[i - 1]]

    out = g.copy()
    for idx, (r0, c0, r1, c1) in enumerate(frames):
        assert transferred[idx] is not None
        out[r0 + 1 : r1, c0 + 1 : c1] = transferred[idx]
    return out


def _pointer_position(inner_h: int, inner_w: int, direction: tuple[int, int]) -> tuple[int, int]:
    if direction == (-1, 0):
        return 0, inner_w // 2
    if direction == (1, 0):
        return inner_h - 1, inner_w // 2
    if direction == (0, -1):
        return inner_h // 2, 0
    if direction == (0, 1):
        return inner_h // 2, inner_w - 1
    raise AssertionError(f"unsupported direction: {direction}")


def _empty_patch(inner_h: int, inner_w: int, bg: int) -> np.ndarray:
    return np.full((inner_h, inner_w), bg, dtype=int)


def _grow_blob(
    inner_h: int,
    inner_w: int,
    banned: set[tuple[int, int]],
    area: int,
) -> np.ndarray | None:
    free = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) not in banned]
    if not free:
        return None

    cells = {random.choice(free)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < inner_h
                    and 0 <= nc < inner_w
                    and (nr, nc) not in banned
                    and (nr, nc) not in cells
                ):
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask if bool(mask.any()) else None


def _sample_pattern_mask(inner_h: int, inner_w: int, banned: set[tuple[int, int]]) -> np.ndarray:
    free_count = inner_h * inner_w - len(banned)
    max_area = max(1, min(free_count, (inner_h * inner_w + 1) // 2))

    for _ in range(80):
        mode = random.choice(("blob", "line", "rect", "ell"))

        if mode == "line":
            horizontal = bool(random.getrandbits(1))
            if horizontal:
                length = random.randint(1, inner_w)
                row = random.randrange(inner_h)
                col = random.randint(0, inner_w - length)
                cells = {(row, c) for c in range(col, col + length)}
            else:
                length = random.randint(1, inner_h)
                row = random.randint(0, inner_h - length)
                col = random.randrange(inner_w)
                cells = {(r, col) for r in range(row, row + length)}
            if cells.isdisjoint(banned):
                mask = np.zeros((inner_h, inner_w), dtype=bool)
                for r, c in cells:
                    mask[r, c] = True
                return mask

        if mode == "rect":
            rect_h = random.randint(1, max(1, min(inner_h, 3)))
            rect_w = random.randint(1, max(1, min(inner_w, 3)))
            row = random.randint(0, inner_h - rect_h)
            col = random.randint(0, inner_w - rect_w)
            cells = {(r, c) for r in range(row, row + rect_h) for c in range(col, col + rect_w)}
            if cells.isdisjoint(banned):
                mask = np.zeros((inner_h, inner_w), dtype=bool)
                for r, c in cells:
                    mask[r, c] = True
                return mask

        if mode == "ell":
            corner_r = random.randrange(inner_h)
            corner_c = random.randrange(inner_w)
            down = random.choice((-1, 1))
            right = random.choice((-1, 1))
            arm_h = random.randint(1, inner_h)
            arm_w = random.randint(1, inner_w)
            cells: set[tuple[int, int]] = set()
            for i in range(arm_h):
                rr = corner_r + down * i
                if not (0 <= rr < inner_h):
                    break
                cells.add((rr, corner_c))
            for i in range(arm_w):
                cc = corner_c + right * i
                if not (0 <= cc < inner_w):
                    break
                cells.add((corner_r, cc))
            if cells and cells.isdisjoint(banned):
                mask = np.zeros((inner_h, inner_w), dtype=bool)
                for r, c in cells:
                    mask[r, c] = True
                return mask

        area = random.randint(1, max_area)
        blob = _grow_blob(inner_h, inner_w, banned, area)
        if blob is not None:
            return blob

    raise RuntimeError("failed to sample a pattern mask")


def _block_size(orientation: str, frame_count: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return inner_h + 2, frame_count * (inner_w + 1) + 1
    return frame_count * (inner_h + 1) + 1, inner_w + 2


def _sample_dimensions(orientation: str, inner_h: int, inner_w: int) -> tuple[int, int, int, int, int]:
    axis_span = inner_w + 1 if orientation == "horizontal" else inner_h + 1
    max_count_fit = (GRID_MAX - 3) // axis_span
    frame_count_max = min(FRAME_COUNT_MAX, max_count_fit)
    if frame_count_max < FRAME_COUNT_MIN:
        raise RuntimeError("frame count does not fit")

    frame_count = random.randint(FRAME_COUNT_MIN, frame_count_max)
    block_h, block_w = _block_size(orientation, frame_count, inner_h, inner_w)

    extra_h_min = max(2, GRID_MIN - block_h)
    extra_w_min = max(2, GRID_MIN - block_w)
    extra_h_max = GRID_MAX - block_h
    extra_w_max = GRID_MAX - block_w
    if extra_h_min > extra_h_max or extra_w_min > extra_w_max:
        raise RuntimeError("block does not fit")

    extra_h = random.randint(extra_h_min, extra_h_max)
    extra_w = random.randint(extra_w_min, extra_w_max)
    top = random.randint(1, extra_h - 1)
    left = random.randint(1, extra_w - 1)
    return frame_count, top, extra_h - top, left, extra_w - left


def _frame_bboxes(
    orientation: str,
    frame_count: int,
    inner_h: int,
    inner_w: int,
    top: int,
    left: int,
) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    if orientation == "horizontal":
        for i in range(frame_count):
            c0 = left + i * (inner_w + 1)
            frames.append((top, c0, top + inner_h + 1, c0 + inner_w + 1))
    else:
        for i in range(frame_count):
            r0 = top + i * (inner_h + 1)
            frames.append((r0, left, r0 + inner_h + 1, left + inner_w + 1))
    return frames


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR


def _render_expected(
    grid: np.ndarray,
    bg: int,
    frames: list[tuple[int, int, int, int]],
    patterns: list[np.ndarray],
    chain: list[int],
) -> np.ndarray:
    out = grid.copy()
    transferred: list[np.ndarray | None] = [None] * len(frames)
    transferred[chain[0]] = patterns[chain[0]]
    for i in range(1, len(chain)):
        transferred[chain[i]] = patterns[chain[i - 1]]
    for idx, (r0, c0, r1, c1) in enumerate(frames):
        out[r0 + 1 : r1, c0 + 1 : c1] = transferred[idx]
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    orientation = random.choice(("horizontal", "vertical"))
    inner_h = random.choice(INNER_SIZE_CHOICES)
    inner_w = random.choice(INNER_SIZE_CHOICES)
    frame_count, top, bottom, left, right = _sample_dimensions(orientation, inner_h, inner_w)
    block_h, block_w = _block_size(orientation, frame_count, inner_h, inner_w)
    height = top + block_h + bottom
    width = left + block_w + right

    bg = random.choice([c for c in range(10) if c not in {FRAME_COLOR, POINTER_COLOR}])
    palette = [c for c in range(10) if c not in {bg, FRAME_COLOR, POINTER_COLOR}]
    pattern_colors = random.sample(palette, frame_count)

    frames = _frame_bboxes(orientation, frame_count, inner_h, inner_w, top, left)
    chain = list(range(frame_count))
    if not bool(random.getrandbits(1)):
        chain.reverse()

    grid = np.full((height, width), bg, dtype=int)
    for bbox in frames:
        _paint_frame(grid, bbox)

    patterns: list[np.ndarray] = []
    next_by_frame: dict[int, int] = {chain[i]: chain[i + 1] for i in range(frame_count - 1)}

    for idx, bbox in enumerate(frames):
        r0, c0, r1, c1 = bbox
        patch = _empty_patch(inner_h, inner_w, bg)

        banned: set[tuple[int, int]] = set()
        if idx in next_by_frame:
            nxt = next_by_frame[idx]
            direction = (
                (nxt - idx, 0) if orientation == "vertical" else (0, nxt - idx)
            )
            banned.add(_pointer_position(inner_h, inner_w, direction))

        mask = _sample_pattern_mask(inner_h, inner_w, banned)
        patch[mask] = pattern_colors[idx]
        patterns.append(patch.copy())

        grid[r0 + 1 : r1, c0 + 1 : c1] = patch
        if idx in next_by_frame:
            nxt = next_by_frame[idx]
            direction = (
                (nxt - idx, 0) if orientation == "vertical" else (0, nxt - idx)
            )
            pr, pc = _pointer_position(inner_h, inner_w, direction)
            grid[r0 + 1 + pr, c0 + 1 + pc] = POINTER_COLOR

    expected = _render_expected(grid, bg, frames, patterns, chain)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid, expected = _build_case()
        bg = _get_bg(grid)
        if bg in {FRAME_COLOR, POINTER_COLOR}:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        if int(np.count_nonzero(grid == POINTER_COLOR)) < 1:
            continue
        return grid
    raise RuntimeError("failed to generate a valid chain-of-frames input")

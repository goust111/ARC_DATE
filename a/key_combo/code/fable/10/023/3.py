import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GENERATION_RETRIES = 1000


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _neighbors8(r: int, c: int, height: int, width: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width:
                yield rr, cc


def _is_connected8(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    height, width = mask.shape
    start = tuple(map(int, points[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors8(r, c, height, width):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _nearest_unique(values: list[int], x: int) -> int:
    ranked = sorted((abs(v - x), v) for v in values)
    assert ranked, "expected a non-empty lattice axis"
    best_dist = ranked[0][0]
    tied = [v for dist, v in ranked if dist == best_dist]
    assert len(tied) == 1, "expected a unique nearest lattice point"
    return int(tied[0])


def _nearest_unique_or_none(values: list[int], x: int) -> int | None:
    ranked = sorted((abs(v - x), v) for v in values)
    if not ranked:
        return None
    best_dist = ranked[0][0]
    tied = [v for dist, v in ranked if dist == best_dist]
    if len(tied) != 1:
        return None
    return int(tied[0])


def _allowed_offsets(period: int) -> list[int]:
    return [d for d in range(-(period - 1), period) if 2 * abs(d) < period]


def _can_stamp(shape: tuple[int, int], sprite: np.ndarray, center_r: int, center_c: int) -> bool:
    height, width = shape
    half_r = sprite.shape[0] // 2
    half_c = sprite.shape[1] // 2
    r0 = center_r - half_r
    c0 = center_c - half_c
    return 0 <= r0 and 0 <= c0 and r0 + sprite.shape[0] <= height and c0 + sprite.shape[1] <= width


def _stamp(out: np.ndarray, sprite: np.ndarray, center_r: int, center_c: int) -> bool:
    if not _can_stamp(out.shape, sprite, center_r, center_c):
        return False
    half_r = sprite.shape[0] // 2
    half_c = sprite.shape[1] // 2
    r0 = center_r - half_r
    c0 = center_c - half_c
    region = out[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
    mask = sprite != BLACK
    region[mask] = sprite[mask]
    return True


def _render(
    shape: tuple[int, int],
    row_ticks: list[int],
    col_ticks: list[int],
    row_phase: int,
    col_phase: int,
    sprite: np.ndarray,
    top_color: int,
    left_color: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[0, col_ticks] = top_color
    out[row_ticks, 0] = left_color
    for r in row_ticks:
        for c in col_ticks:
            _stamp(out, sprite, r + row_phase, c + col_phase)
    return out


def _extract_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    top_cols = np.flatnonzero(g[0] != BLACK).tolist()
    left_rows = np.flatnonzero(g[:, 0] != BLACK).tolist()
    assert len(top_cols) == 2, "expected exactly two top-row ticks"
    assert len(left_rows) == 2, "expected exactly two left-column ticks"

    top_color = int(g[0, top_cols[0]])
    left_color = int(g[left_rows[0], 0])
    assert top_color != BLACK and left_color != BLACK and top_color != left_color
    assert all(int(g[0, c]) == top_color for c in top_cols), "top ticks must share one color"
    assert all(int(g[r, 0]) == left_color for r in left_rows), "left ticks must share one color"

    col_period = int(top_cols[1] - top_cols[0])
    row_period = int(left_rows[1] - left_rows[0])
    assert col_period > 0 and row_period > 0, "tick periods must be positive"

    inner_mask = g != BLACK
    inner_mask[0, :] = False
    inner_mask[:, 0] = False
    box = _bbox(inner_mask)
    assert box is not None, "expected one interior sprite"
    r0, c0, r1, c1 = box
    sprite = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    sprite_mask = sprite != BLACK
    assert _is_connected8(sprite_mask), "sprite must form one 8-connected object"
    assert sprite.shape[0] % 2 == 1 and sprite.shape[1] % 2 == 1, "sprite bbox must have odd side lengths"

    center_r = int(r0 + sprite.shape[0] // 2)
    center_c = int(c0 + sprite.shape[1] // 2)

    row_ticks = list(range(int(left_rows[0]), height, row_period))
    col_ticks = list(range(int(top_cols[0]), width, col_period))
    nearest_row = _nearest_unique(row_ticks, center_r)
    nearest_col = _nearest_unique(col_ticks, center_c)
    row_phase = int(center_r - nearest_row)
    col_phase = int(center_c - nearest_col)
    assert row_phase != 0 or col_phase != 0, "sprite center must encode a non-zero global phase"

    return {
        "grid": g,
        "sprite": sprite,
        "sprite_box": (r0, c0, r1, c1),
        "center": (center_r, center_c),
        "top_color": top_color,
        "left_color": left_color,
        "row_ticks": row_ticks,
        "col_ticks": col_ticks,
        "row_phase": row_phase,
        "col_phase": col_phase,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the two border tick sequences into a lattice and stamp the interior sprite at one shared phase.

    Input structure:
    - The grid has a black background.
    - The top row contains exactly two same-color tick pixels, and the left column
      contains exactly two tick pixels of another color; each pair gives the first
      two terms of an arithmetic border scale.
    - Away from the border there is one small multicolor sprite whose bounding-box
      center is offset from its nearest row/column tick intersection.

    Rule:
    - Use the distance between the top-row ticks as the column period and extend
      that tick color across the whole top border; use the distance between the
      left-column ticks as the row period and extend that tick color down the whole
      left border.
    - Treat every crossing of an extended tick row and tick column as a lattice
      intersection, and read the original sprite center's offset from its unique
      nearest intersection.
    - Stamp the sprite at every intersection plus that same offset, keeping black
      cells transparent, leaving out positions where the sprite would go out of
      bounds, and preserving the completed border ticks.

    Key insight:
    - The interior sample is not aligned to the lattice on purpose: its displacement
      from the nearest hidden intersection is the global phase that every copy must
      share.
    """
    info = _extract_structure(grid)
    return _render(
        info["grid"].shape,
        info["row_ticks"],
        info["col_ticks"],
        info["row_phase"],
        info["col_phase"],
        info["sprite"],
        info["top_color"],
        info["left_color"],
    )


def _grow_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    path = [(r, c)]
    while r != rr:
        r += 1 if rr > r else -1
        path.append((r, c))
    while c != cc:
        c += 1 if cc > c else -1
        path.append((r, c))
    return path


def _make_sprite(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    center = (height // 2, width // 2)
    cells = {center}
    edge_targets = [
        (0, int(np.random.randint(width))),
        (height - 1, int(np.random.randint(width))),
        (int(np.random.randint(height)), 0),
        (int(np.random.randint(height)), width - 1),
    ]
    for target in edge_targets:
        for cell in _grow_path(center, target):
            cells.add(cell)

    target_area = int(np.random.randint(max(5, height + width - 2), height * width))
    while len(cells) < target_area:
        frontier = []
        for r, c in list(cells):
            for rr, cc in _neighbors8(r, c, height, width):
                if (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(frontier[int(np.random.randint(len(frontier)))])

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if mask.all():
        removable = [(r, c) for r, c in cells if (r, c) != center]
        np.random.shuffle(removable)
        removed = False
        for r, c in removable:
            trial = mask.copy()
            trial[r, c] = False
            if _is_connected8(trial):
                mask = trial
                removed = True
                break
        if not removed:
            return None

    if not _is_connected8(mask):
        return None
    if _bbox(mask) != (0, 0, height - 1, width - 1):
        return None

    used_colors = colors[: int(np.random.randint(2, min(4, len(colors)) + 1))]
    sprite = np.zeros((height, width), dtype=int)
    positions = np.argwhere(mask)
    if len(positions) < len(used_colors):
        return None
    order = np.random.permutation(len(positions))
    assigned = np.empty(len(positions), dtype=int)
    assigned[: len(used_colors)] = used_colors
    if len(positions) > len(used_colors):
        assigned[len(used_colors) :] = np.random.choice(used_colors, size=len(positions) - len(used_colors))
    assigned = assigned[np.random.permutation(len(positions))]
    for idx, (r, c) in enumerate(positions):
        sprite[int(r), int(c)] = int(assigned[idx])
    return sprite


def _corner_phase_output(
    shape: tuple[int, int],
    row_ticks: list[int],
    col_ticks: list[int],
    sprite: np.ndarray,
    reveal_center: tuple[int, int],
    top_color: int,
    left_color: int,
    corner: str,
) -> np.ndarray | None:
    half_r = sprite.shape[0] // 2
    half_c = sprite.shape[1] // 2
    if corner == "top_left":
        anchor_r = int(reveal_center[0] - half_r)
        anchor_c = int(reveal_center[1] - half_c)
        nearest_r = _nearest_unique_or_none(row_ticks, anchor_r)
        nearest_c = _nearest_unique_or_none(col_ticks, anchor_c)
        if nearest_r is None or nearest_c is None:
            return None
        row_phase = int(anchor_r - nearest_r + half_r)
        col_phase = int(anchor_c - nearest_c + half_c)
    elif corner == "bottom_right":
        anchor_r = int(reveal_center[0] + half_r)
        anchor_c = int(reveal_center[1] + half_c)
        nearest_r = _nearest_unique_or_none(row_ticks, anchor_r)
        nearest_c = _nearest_unique_or_none(col_ticks, anchor_c)
        if nearest_r is None or nearest_c is None:
            return None
        row_phase = int(anchor_r - nearest_r - half_r)
        col_phase = int(anchor_c - nearest_c - half_c)
    else:
        raise ValueError(corner)

    return _render(shape, row_ticks, col_ticks, row_phase, col_phase, sprite, top_color, left_color)


def _safe_reveal_positions(
    shape: tuple[int, int],
    row_ticks: list[int],
    col_ticks: list[int],
    row_phase: int,
    col_phase: int,
    sprite: np.ndarray,
) -> list[tuple[int, int]]:
    half_r = sprite.shape[0] // 2
    half_c = sprite.shape[1] // 2
    height, width = shape
    safe = []
    for r in row_ticks:
        for c in col_ticks:
            center_r = int(r + row_phase)
            center_c = int(c + col_phase)
            if not _can_stamp(shape, sprite, center_r, center_c):
                continue
            if center_r - half_r <= 0 or center_c - half_c <= 0:
                continue
            if center_r + half_r >= height or center_c + half_c >= width:
                continue
            safe.append((center_r, center_c))
    return safe


def _gen_attempt() -> np.ndarray | None:
    sprite_h = int(np.random.choice([3, 5]))
    sprite_w = int(np.random.choice([3, 5]))

    row_period_min = max(sprite_h + 1, 5)
    col_period_min = max(sprite_w + 1, 5)
    if row_period_min > 8 or col_period_min > 8:
        return None

    row_period = int(np.random.randint(row_period_min, 9))
    col_period = int(np.random.randint(col_period_min, 9))

    possible_row_counts = [n for n in range(3, 6) if 4 + row_period * n <= 30]
    possible_col_counts = [n for n in range(3, 6) if 4 + col_period * n <= 30]
    if not possible_row_counts or not possible_col_counts:
        return None

    row_count = int(np.random.choice(possible_row_counts))
    col_count = int(np.random.choice(possible_col_counts))

    row_start = int(np.random.randint(1, min(4, row_period - 1) + 1))
    col_start = int(np.random.randint(1, min(4, col_period - 1) + 1))
    bottom_margin = int(np.random.randint(0, row_period))
    right_margin = int(np.random.randint(0, col_period))

    height = int(row_start + row_period * (row_count - 1) + 1 + bottom_margin)
    width = int(col_start + col_period * (col_count - 1) + 1 + right_margin)
    if height > 30 or width > 30:
        return None

    row_ticks = list(range(row_start, height, row_period))
    col_ticks = list(range(col_start, width, col_period))
    if len(row_ticks) != row_count or len(col_ticks) != col_count:
        return None

    row_phase_choices = _allowed_offsets(row_period)
    col_phase_choices = _allowed_offsets(col_period)
    if not row_phase_choices or not col_phase_choices:
        return None

    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            row_phase = 0
            nonzero_cols = [d for d in col_phase_choices if d != 0]
            if not nonzero_cols:
                return None
            col_phase = int(np.random.choice(nonzero_cols))
        else:
            col_phase = 0
            nonzero_rows = [d for d in row_phase_choices if d != 0]
            if not nonzero_rows:
                return None
            row_phase = int(np.random.choice(nonzero_rows))
    else:
        row_phase = int(np.random.choice(row_phase_choices))
        col_phase = int(np.random.choice(col_phase_choices))
        if row_phase == 0 and col_phase == 0:
            nonzero_axis = int(np.random.randint(2))
            if nonzero_axis == 0:
                row_phase = int(np.random.choice([d for d in row_phase_choices if d != 0]))
            else:
                col_phase = int(np.random.choice([d for d in col_phase_choices if d != 0]))

    colors = np.random.permutation(PALETTE).tolist()
    top_color = int(colors[0])
    left_color = int(colors[1])
    sprite_colors = [int(c) for c in colors[2:]]
    sprite = _make_sprite(sprite_h, sprite_w, sprite_colors)
    if sprite is None:
        return None

    correct = _render(
        (height, width),
        row_ticks,
        col_ticks,
        row_phase,
        col_phase,
        sprite,
        top_color,
        left_color,
    )

    fitted = [
        (r + row_phase, c + col_phase)
        for r in row_ticks
        for c in col_ticks
        if _can_stamp((height, width), sprite, r + row_phase, c + col_phase)
    ]
    if len(fitted) < 4:
        return None

    safe_reveals = _safe_reveal_positions((height, width), row_ticks, col_ticks, row_phase, col_phase, sprite)
    if not safe_reveals:
        return None

    valid_reveals = []
    zero_phase = _render((height, width), row_ticks, col_ticks, 0, 0, sprite, top_color, left_color)
    if np.array_equal(zero_phase, correct):
        return None

    for reveal_center in safe_reveals:
        alt_tl = _corner_phase_output(
            (height, width), row_ticks, col_ticks, sprite, reveal_center, top_color, left_color, "top_left"
        )
        alt_br = _corner_phase_output(
            (height, width), row_ticks, col_ticks, sprite, reveal_center, top_color, left_color, "bottom_right"
        )
        if (alt_tl is not None and not np.array_equal(alt_tl, correct)) or (
            alt_br is not None and not np.array_equal(alt_br, correct)
        ):
            valid_reveals.append(reveal_center)
    if not valid_reveals:
        return None

    reveal_center = valid_reveals[int(np.random.randint(len(valid_reveals)))]
    inp = np.zeros((height, width), dtype=int)
    inp[0, col_ticks[0]] = top_color
    inp[0, col_ticks[1]] = top_color
    inp[row_ticks[0], 0] = left_color
    inp[row_ticks[1], 0] = left_color
    if not _stamp(inp, sprite, int(reveal_center[0]), int(reveal_center[1])):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, correct):
        return None
    if np.count_nonzero(solved) <= np.count_nonzero(inp):
        return None
    if not np.array_equal(correct[0, col_ticks], np.full(len(col_ticks), top_color)):
        return None
    if not np.array_equal(correct[row_ticks, 0], np.full(len(row_ticks), left_color)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phased tick-array input")

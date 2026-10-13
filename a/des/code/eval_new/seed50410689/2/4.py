import numpy as np

BG = 0
DOT = 9
LINE_COLORS = tuple(range(1, 9))
MAX_GENERATION_RETRIES = 400
_RNG = np.random.default_rng()

CORNER_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values.astype(int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _room_bounds(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        bounds.append((start, sep - 1))
        start = sep + 1
    bounds.append((start, size - 1))
    return bounds


def _neighbor(room: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = room
    if side == "top":
        return r - 1, c
    if side == "bottom":
        return r + 1, c
    if side == "left":
        return r, c - 1
    if side == "right":
        return r, c + 1
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each room's stamp across divider-marked sides, erasing pixels that match the crossed divider color.

    Input structure:
    - The background is black.
    - One-cell-thick non-black full rows and full columns partition the grid into rectangular rooms.
    - Some rooms contain one multicolor stamp and one or two maroon single-pixel dots touching divider-adjacent room edges.

    Rule:
    - For every room that contains a stamp, inspect each maroon dot in that room.
    - The divider side touched by the dot chooses the adjacent room across that divider.
    - Copy the entire stamp without changing its orientation, placing its bounding box at the mirror-symmetric position across the chosen divider.
    - During that copy, remove every stamp pixel whose color equals the crossed divider's color; keep all other pixels.
    - Leave the original stamp, dots, and divider lines unchanged. Multiple dots create multiple filtered copies.

    Key insight:
    - The maroon dot selects the transmission direction, while the divider's own color acts as a filter that deletes matching stamp pixels during the transfer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape

    sep_rows = [int(r) for r in range(h) if bool(np.all(g[r, :] != BG))]
    sep_cols = [int(c) for c in range(w) if bool(np.all(g[:, c] != BG))]
    assert sep_rows and sep_cols, "expected both horizontal and vertical dividers"

    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    row_colors = {r: _majority_color(g[r, :]) for r in sep_rows}
    col_colors = {c: _majority_color(g[:, c]) for c in sep_cols}
    row_bounds = _room_bounds(h, sep_rows)
    col_bounds = _room_bounds(w, sep_cols)

    out = g.copy()
    for room_r0, room_r1 in row_bounds:
        for room_c0, room_c1 in col_bounds:
            room = g[room_r0 : room_r1 + 1, room_c0 : room_c1 + 1]
            dots = np.argwhere(room == DOT)
            stamp_mask = (room != BG) & (room != DOT)
            stamp_box = _bbox(stamp_mask)
            if stamp_box is None or dots.size == 0:
                continue

            sr0_l, sc0_l, sr1_l, sc1_l = stamp_box
            patch = room[sr0_l : sr1_l + 1, sc0_l : sc1_l + 1].copy()
            src_r0 = room_r0 + sr0_l
            src_c0 = room_c0 + sc0_l
            src_r1 = room_r0 + sr1_l
            src_c1 = room_c0 + sc1_l

            for dot_r_raw, dot_c_raw in dots:
                dot_r = int(dot_r_raw)
                dot_c = int(dot_c_raw)
                matches: list[tuple[str, int, int]] = []
                if dot_r == 0 and room_r0 > 0 and (room_r0 - 1) in sep_row_set:
                    line_r = room_r0 - 1
                    matches.append(("top", line_r, int(row_colors[line_r])))
                if dot_r == room.shape[0] - 1 and room_r1 < h - 1 and (room_r1 + 1) in sep_row_set:
                    line_r = room_r1 + 1
                    matches.append(("bottom", line_r, int(row_colors[line_r])))
                if dot_c == 0 and room_c0 > 0 and (room_c0 - 1) in sep_col_set:
                    line_c = room_c0 - 1
                    matches.append(("left", line_c, int(col_colors[line_c])))
                if dot_c == room.shape[1] - 1 and room_c1 < w - 1 and (room_c1 + 1) in sep_col_set:
                    line_c = room_c1 + 1
                    matches.append(("right", line_c, int(col_colors[line_c])))
                assert len(matches) == 1, "each dot must indicate exactly one divider side"

                side, line_pos, filter_color = matches[0]
                if side in ("top", "bottom"):
                    dst_r0 = 2 * line_pos - src_r1
                    dst_c0 = src_c0
                else:
                    dst_r0 = src_r0
                    dst_c0 = 2 * line_pos - src_c1

                for pr in range(patch.shape[0]):
                    for pc in range(patch.shape[1]):
                        color = int(patch[pr, pc])
                        if color in (BG, filter_color):
                            continue
                        rr = dst_r0 + pr
                        cc = dst_c0 + pc
                        assert 0 <= rr < h and 0 <= cc < w
                        assert rr not in sep_row_set and cc not in sep_col_set
                        assert int(out[rr, cc]) in (BG, color)
                        out[rr, cc] = color

    return out


def _stamp_bbox(corner: str, room_h: int, room_w: int, stamp_size: int, row_gap: int, col_gap: int) -> tuple[int, int]:
    if corner[0] == "t":
        r0 = row_gap
    else:
        r0 = room_h - row_gap - stamp_size
    if corner[1] == "l":
        c0 = col_gap
    else:
        c0 = room_w - col_gap - stamp_size
    return int(r0), int(c0)


def _build_stamp(required_colors: list[int], extra_colors: list[int], size: int) -> np.ndarray | None:
    palette = list(dict.fromkeys(int(c) for c in required_colors))
    target_color_count = 2 if len(palette) == 1 else int(_RNG.choice([2, 3]))
    while len(palette) < target_color_count:
        if not extra_colors:
            break
        pick = int(extra_colors.pop(int(_RNG.integers(0, len(extra_colors)))))
        if pick not in palette:
            palette.append(pick)
    if len(palette) < 2:
        return None

    for _ in range(80):
        offset = int(_RNG.integers(0, len(palette)))
        mode = int(_RNG.integers(0, 3))
        patch = np.empty((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                if mode == 0:
                    idx = (r + 2 * c + offset) % len(palette)
                elif mode == 1:
                    idx = (2 * r + c + offset) % len(palette)
                else:
                    idx = (r + c + (r % 2) + offset) % len(palette)
                patch[r, c] = palette[idx]

        positions = list(_RNG.permutation(size * size))
        for color in palette:
            repeats = 2 if color in required_colors else 1
            for _ in range(repeats):
                if not positions:
                    break
                patch.flat[int(positions.pop())] = int(color)

        if np.all(patch == patch[0, 0]):
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        if any(int(np.sum(patch == color)) == 0 for color in palette):
            continue
        if any(int(np.sum(patch == color)) == patch.size for color in required_colors):
            continue
        return patch
    return None


def _divider_color_for_side(room_r0: int, room_r1: int, room_c0: int, room_c1: int, side: str, row_colors: dict[int, int], col_colors: dict[int, int]) -> int:
    if side == "top":
        return int(row_colors[room_r0 - 1])
    if side == "bottom":
        return int(row_colors[room_r1 + 1])
    if side == "left":
        return int(col_colors[room_c0 - 1])
    if side == "right":
        return int(col_colors[room_c1 + 1])
    raise ValueError(f"unknown side: {side}")


def _candidate_specs(n_room_rows: int, n_room_cols: int, reserved: set[tuple[int, int]]) -> list[dict]:
    candidates: list[dict] = []
    for room_r in range(n_room_rows):
        for room_c in range(n_room_cols):
            room = (room_r, room_c)
            if room in reserved:
                continue
            for corner, sides in CORNER_SIDES.items():
                available = []
                for side in sides:
                    nr, nc = _neighbor(room, side)
                    if 0 <= nr < n_room_rows and 0 <= nc < n_room_cols:
                        available.append(side)
                for side in available:
                    target = _neighbor(room, side)
                    if target in reserved:
                        continue
                    candidates.append({"room": room, "corner": corner, "sides": (side,), "targets": (target,)})
                if len(available) == 2:
                    targets = tuple(_neighbor(room, side) for side in available)
                    if any(target in reserved for target in targets):
                        continue
                    candidates.append({"room": room, "corner": corner, "sides": tuple(available), "targets": targets})
    return candidates


def _gen_attempt() -> np.ndarray | None:
    n_room_rows = int(_RNG.integers(2, 4))
    n_room_cols = int(_RNG.integers(2, 4))
    room_h = int(_RNG.integers(7, 10))
    room_w = int(_RNG.integers(7, 10))

    h = n_room_rows * room_h + (n_room_rows - 1)
    w = n_room_cols * room_w + (n_room_cols - 1)
    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    sep_rows = [room_h * (idx + 1) + idx for idx in range(n_room_rows - 1)]
    sep_cols = [room_w * (idx + 1) + idx for idx in range(n_room_cols - 1)]

    all_line_colors = list(_RNG.choice(LINE_COLORS, size=len(sep_rows) + len(sep_cols), replace=False))
    row_colors = {sep_rows[idx]: int(all_line_colors[idx]) for idx in range(len(sep_rows))}
    col_colors = {sep_cols[idx]: int(all_line_colors[len(sep_rows) + idx]) for idx in range(len(sep_cols))}

    for row, color in row_colors.items():
        grid[int(row), :] = int(color)
    for col, color in col_colors.items():
        grid[:, int(col)] = int(color)

    room_rows = _room_bounds(h, sep_rows)
    room_cols = _room_bounds(w, sep_cols)
    desired_sources = int(_RNG.choice([1, 2]))
    reserved: set[tuple[int, int]] = set()
    sources: list[dict] = []

    for _ in range(8):
        candidates = _candidate_specs(n_room_rows, n_room_cols, reserved)
        if not candidates:
            break
        want_two_dots = bool(_RNG.integers(0, 2))
        preferred = [cand for cand in candidates if len(cand["sides"]) == (2 if want_two_dots else 1)]
        pool = preferred or candidates
        choice = dict(pool[int(_RNG.integers(0, len(pool)))])
        room = choice["room"]
        targets = choice["targets"]
        if room in reserved or any(target in reserved for target in targets):
            continue
        sources.append(choice)
        reserved.add(room)
        reserved.update(targets)
        if len(sources) == desired_sources:
            break

    if len(sources) != desired_sources:
        return None

    non_line_colors = [color for color in LINE_COLORS if color not in set(all_line_colors)]
    if len(non_line_colors) < 2:
        return None

    for spec in sources:
        room_r, room_c = spec["room"]
        corner = str(spec["corner"])
        sides = tuple(spec["sides"])
        room_r0, room_r1 = room_rows[room_r]
        room_c0, room_c1 = room_cols[room_c]
        row_gap = int(_RNG.integers(1, 3))
        col_gap = int(_RNG.integers(1, 3))
        stamp_size = int(_RNG.choice([3, 4]))
        stamp_r0_l, stamp_c0_l = _stamp_bbox(corner, room_h, room_w, stamp_size, row_gap, col_gap)
        stamp_r1_l = stamp_r0_l + stamp_size - 1
        stamp_c1_l = stamp_c0_l + stamp_size - 1
        if not (0 <= stamp_r0_l <= stamp_r1_l < room_h and 0 <= stamp_c0_l <= stamp_c1_l < room_w):
            return None

        required_colors = [
            _divider_color_for_side(room_r0, room_r1, room_c0, room_c1, side, row_colors, col_colors)
            for side in sides
        ]
        extras = [color for color in non_line_colors if color not in required_colors]
        if len(set(required_colors)) + len(extras) < 2:
            return None
        stamp = _build_stamp(required_colors, extras.copy(), stamp_size)
        if stamp is None:
            return None

        room = grid[room_r0 : room_r1 + 1, room_c0 : room_c1 + 1]
        if np.any(room[stamp_r0_l : stamp_r1_l + 1, stamp_c0_l : stamp_c1_l + 1] != BG):
            return None
        room[stamp_r0_l : stamp_r1_l + 1, stamp_c0_l : stamp_c1_l + 1] = stamp

        for side in sides:
            if side == "top":
                dot_r, dot_c = 0, stamp_c0_l
            elif side == "bottom":
                dot_r, dot_c = room_h - 1, stamp_c0_l
            elif side == "left":
                dot_r, dot_c = stamp_r0_l, 0
            else:
                dot_r, dot_c = stamp_r0_l, room_w - 1
            if room[dot_r, dot_c] != BG:
                return None
            room[dot_r, dot_c] = DOT

        for side, filter_color in zip(sides, required_colors):
            if int(np.sum(stamp == filter_color)) == 0:
                return None
            if int(np.sum(stamp != filter_color)) == 0:
                return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid divider-stamp puzzle")

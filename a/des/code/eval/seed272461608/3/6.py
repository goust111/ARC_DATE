import numpy as np
from collections import deque


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _runs_1d(values: np.ndarray) -> list[tuple[int, int, int, int]]:
    runs = []
    start = 0
    current = int(values[0])
    for idx in range(1, len(values)):
        value = int(values[idx])
        if value != current:
            runs.append((current, start, idx - 1, idx - start))
            start = idx
            current = value
    runs.append((current, start, len(values) - 1, len(values) - start))
    return runs


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_anchor(
    grid: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    orientation: str,
) -> str | None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    if orientation == "horizontal":
        left_blocked = c0 == 0 or np.all(grid[r0 : r1 + 1, c0 - 1] != bg)
        right_blocked = c1 == w - 1 or np.all(grid[r0 : r1 + 1, c1 + 1] != bg)
        if left_blocked == right_blocked:
            return None
        return "left" if left_blocked else "right"

    top_blocked = r0 == 0 or np.all(grid[r0 - 1, c0 : c1 + 1] != bg)
    bottom_blocked = r1 == h - 1 or np.all(grid[r1 + 1, c0 : c1 + 1] != bg)
    if top_blocked == bottom_blocked:
        return None
    return "top" if top_blocked else "bottom"


def _parse_candidate(grid: np.ndarray, side: str, bg: int) -> dict | None:
    h, w = grid.shape
    if side == "left":
        ruler_values = grid[:, 0]
        ruler_mask = np.zeros_like(grid, dtype=bool)
        ruler_mask[:, 0] = True
        expected_orientation = "horizontal"
    else:
        ruler_values = grid[0, :]
        ruler_mask = np.zeros_like(grid, dtype=bool)
        ruler_mask[0, :] = True
        expected_orientation = "vertical"

    runs = _runs_1d(ruler_values)
    if len(runs) < 2:
        return None

    colors = [int(color) for color, _, _, _ in runs]
    if bg in colors or len(colors) != len(set(colors)):
        return None

    color_to_len = {int(color): int(length) for color, _, _, length in runs}
    bars = []
    for color in colors:
        mask = (grid == color) & (~ruler_mask)
        for coords in _components(mask):
            r0, c0, r1, c1 = _bbox(coords)
            box_h = r1 - r0 + 1
            box_w = c1 - c0 + 1
            if len(coords) != box_h * box_w or box_h == box_w:
                return None
            orientation = "horizontal" if box_w > box_h else "vertical"
            if orientation != expected_orientation:
                return None
            anchor = _detect_anchor(grid, bg, (r0, c0, r1, c1), orientation)
            if anchor is None:
                return None
            bars.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "orientation": orientation,
                    "anchor": anchor,
                }
            )

    if len(bars) < 2:
        return None

    return {
        "side": side,
        "bg": int(bg),
        "ruler_mask": ruler_mask,
        "color_to_len": color_to_len,
        "bars": bars,
    }


def _select_scene(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    candidates = []
    for side in ("left", "top"):
        parsed = _parse_candidate(grid, side, bg)
        if parsed is not None:
            candidates.append(parsed)

    assert candidates, "no valid ruler detected"
    if len(candidates) == 1:
        return candidates[0]

    candidates.sort(key=lambda item: len(item["bars"]), reverse=True)
    assert len(candidates[0]["bars"]) != len(candidates[1]["bars"]), "ambiguous ruler side"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior stripe's length to the same-color ruler segment.

    The ruler is either the leftmost column or the top row. Each ruler color
    defines a target length. Every interior stripe with that color keeps its
    fixed end in place and is clipped or extended from the free end.
    """
    g = np.asarray(grid, dtype=int)
    scene = _select_scene(g)
    bg = int(scene["bg"])
    color_to_len = scene["color_to_len"]

    out = g.copy()
    for bar in scene["bars"]:
        color = int(bar["color"])
        target = int(color_to_len[color])
        r0, c0, r1, c1 = map(int, bar["bbox"])

        if bar["orientation"] == "horizontal":
            length = c1 - c0 + 1
            if bar["anchor"] == "left":
                if length > target:
                    out[r0 : r1 + 1, c0 + target : c1 + 1] = bg
                    length = target
                    c1 = c0 + target - 1
                while length < target:
                    nc = c1 + 1
                    if nc >= g.shape[1] or not np.all(g[r0 : r1 + 1, nc] == bg):
                        break
                    out[r0 : r1 + 1, nc] = color
                    c1 = nc
                    length += 1
            else:
                if length > target:
                    out[r0 : r1 + 1, c0 : c1 - target + 1] = bg
                    length = target
                    c0 = c1 - target + 1
                while length < target:
                    nc = c0 - 1
                    if nc < 0 or not np.all(g[r0 : r1 + 1, nc] == bg):
                        break
                    out[r0 : r1 + 1, nc] = color
                    c0 = nc
                    length += 1
        else:
            length = r1 - r0 + 1
            if bar["anchor"] == "top":
                if length > target:
                    out[r0 + target : r1 + 1, c0 : c1 + 1] = bg
                    length = target
                    r1 = r0 + target - 1
                while length < target:
                    nr = r1 + 1
                    if nr >= g.shape[0] or not np.all(g[nr, c0 : c1 + 1] == bg):
                        break
                    out[nr, c0 : c1 + 1] = color
                    r1 = nr
                    length += 1
            else:
                if length > target:
                    out[r0 : r1 - target + 1, c0 : c1 + 1] = bg
                    length = target
                    r0 = r1 - target + 1
                while length < target:
                    nr = r0 - 1
                    if nr < 0 or not np.all(g[nr, c0 : c1 + 1] == bg):
                        break
                    out[nr, c0 : c1 + 1] = color
                    r0 = nr
                    length += 1

    return out


_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))
_GEN_ATTEMPTS = 400
_BAR_ATTEMPTS = 120
_MIN_GAP = 1


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _sample_lengths(n: int) -> list[int]:
    return list(map(int, _RNG.choice(np.arange(3, 10), size=n, replace=False)))


def _place_bands(height: int, thicknesses: list[int]) -> list[tuple[int, int]] | None:
    for _ in range(_BAR_ATTEMPTS):
        bands = []
        ok = True
        for thickness in thicknesses:
            candidates = []
            for start in range(1, height - thickness + 1):
                end = start + thickness - 1
                separated = True
                for other_start, other_end in bands:
                    if not (end + _MIN_GAP < other_start or other_end + _MIN_GAP < start):
                        separated = False
                        break
                if separated:
                    candidates.append(start)
            if not candidates:
                ok = False
                break
            start = int(_RNG.choice(candidates))
            bands.append((start, start + thickness - 1))
        if ok:
            return bands
    return None


def _branch_list(n: int) -> list[str]:
    branches = ["clip", "extend_clear", "extend_blocked"]
    extras = ["clip", "extend_clear", "extend_blocked", "exact"]
    while len(branches) < n:
        branches.append(str(_RNG.choice(extras)))
    _RNG.shuffle(branches)
    return branches


def _place_horizontal_bar(
    width: int,
    rows: tuple[int, int],
    target_len: int,
    thickness: int,
    branch: str,
) -> dict | None:
    r0, r1 = rows
    for _ in range(_BAR_ATTEMPTS):
        anchor = str(_RNG.choice(["left", "right"]))
        spec = {"input_bbox": None, "output_bbox": None, "anchor_wall": None, "block_wall": None}

        if anchor == "right":
            use_wall = bool(_RNG.integers(0, 2))
            if use_wall:
                wall_col = _randint(max(3, width - 4), width - 2)
                fixed_end = wall_col - 1
                spec["anchor_wall"] = wall_col
            else:
                fixed_end = width - 1

            max_len = fixed_end - 1
            if target_len > max_len:
                continue

            if branch == "clip":
                max_delta = min(3, max_len - target_len)
                if max_delta < 1:
                    continue
                current_len = target_len + _randint(1, max_delta)
                final_len = target_len
                stop_extra = None
            elif branch == "exact":
                current_len = target_len
                final_len = target_len
                stop_extra = None
            elif branch == "extend_clear":
                min_len = max(thickness + 1, target_len - 3)
                if min_len > target_len - 1:
                    continue
                current_len = _randint(min_len, target_len - 1)
                final_len = target_len
                stop_extra = None
            else:
                min_len = max(thickness + 1, target_len - 4)
                if min_len > target_len - 2:
                    continue
                current_len = _randint(min_len, target_len - 2)
                max_stop = target_len - current_len - 1
                if max_stop < 1:
                    continue
                stop_extra = _randint(1, max_stop)
                final_len = current_len + stop_extra

            c1 = fixed_end
            c0 = c1 - current_len + 1
            final_c0 = c1 - final_len + 1
            if c0 < 2 or final_c0 < 2:
                continue

            if branch == "extend_blocked":
                block_col = c0 - stop_extra - 1
                if block_col < 1:
                    continue
                spec["block_wall"] = block_col

            spec["input_bbox"] = (r0, c0, r1, c1)
            spec["output_bbox"] = (r0, final_c0, r1, c1)
            return spec

        wall_col = _randint(1, min(3, width - 6))
        fixed_start = wall_col + 1
        max_len = width - 1 - fixed_start
        if target_len > max_len:
            continue

        if branch == "clip":
            max_delta = min(3, max_len - target_len)
            if max_delta < 1:
                continue
            current_len = target_len + _randint(1, max_delta)
            final_len = target_len
            stop_extra = None
        elif branch == "exact":
            current_len = target_len
            final_len = target_len
            stop_extra = None
        elif branch == "extend_clear":
            min_len = max(thickness + 1, target_len - 3)
            if min_len > target_len - 1:
                continue
            current_len = _randint(min_len, target_len - 1)
            final_len = target_len
            stop_extra = None
        else:
            min_len = max(thickness + 1, target_len - 4)
            if min_len > target_len - 2:
                continue
            current_len = _randint(min_len, target_len - 2)
            max_stop = target_len - current_len - 1
            if max_stop < 1:
                continue
            stop_extra = _randint(1, max_stop)
            final_len = current_len + stop_extra

        c0 = fixed_start
        c1 = c0 + current_len - 1
        final_c1 = c0 + final_len - 1
        if c1 > width - 2 or final_c1 > width - 2:
            continue

        spec["anchor_wall"] = wall_col
        if branch == "extend_blocked":
            block_col = c1 + stop_extra + 1
            if block_col > width - 1:
                continue
            spec["block_wall"] = block_col

        spec["input_bbox"] = (r0, c0, r1, c1)
        spec["output_bbox"] = (r0, c0, r1, final_c1)
        return spec

    return None


def _paint_vertical_wall(grid: np.ndarray, rows: tuple[int, int], col: int, color: int) -> None:
    r0, r1 = rows
    grid[r0 : r1 + 1, col] = color


def _render_base_scene() -> tuple[np.ndarray, np.ndarray] | None:
    n_bars = _randint(3, 5)
    ruler_lengths = _sample_lengths(n_bars)
    height = sum(ruler_lengths)
    thicknesses = [_randint(1, 2) for _ in range(n_bars)]
    if height < 1 + sum(thicknesses) + (n_bars - 1) * _MIN_GAP:
        return None

    bands = _place_bands(height, thicknesses)
    if bands is None:
        return None

    max_target = max(ruler_lengths)
    min_width = max(14, max_target + 7)
    max_width = max(min_width, max_target + 13)
    width = _randint(min_width, max_width)

    colors = list(map(int, _RNG.choice(_PALETTE, size=n_bars + 2, replace=False)))
    bg = colors[0]
    obstacle_color = colors[1]
    bar_colors = colors[2:]
    _RNG.shuffle(bar_colors)

    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()

    ruler_order = list(zip(list(bar_colors), ruler_lengths))
    _RNG.shuffle(ruler_order)
    cursor = 0
    target_by_color = {}
    for color, length in ruler_order:
        grid[cursor : cursor + length, 0] = color
        expected[cursor : cursor + length, 0] = color
        target_by_color[int(color)] = int(length)
        cursor += length

    branches = _branch_list(n_bars)
    bar_specs = []
    for color, rows, thickness, branch in zip(bar_colors, bands, thicknesses, branches):
        target_len = target_by_color[int(color)]
        spec = _place_horizontal_bar(width, rows, target_len, thickness, branch)
        if spec is None:
            return None
        spec["color"] = int(color)
        spec["rows"] = rows
        bar_specs.append(spec)

    seen_branches = set(branches)
    if not {"clip", "extend_clear", "extend_blocked"}.issubset(seen_branches):
        return None

    for spec in bar_specs:
        color = int(spec["color"])
        r0, c0, r1, c1 = map(int, spec["input_bbox"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

        if spec["anchor_wall"] is not None:
            _paint_vertical_wall(grid, spec["rows"], int(spec["anchor_wall"]), obstacle_color)
        if spec["block_wall"] is not None:
            _paint_vertical_wall(grid, spec["rows"], int(spec["block_wall"]), obstacle_color)

        expected[r0 : r1 + 1, c0 : c1 + 1] = bg
        fr0, fc0, fr1, fc1 = map(int, spec["output_bbox"])
        expected[fr0 : fr1 + 1, fc0 : fc1 + 1] = color
        if spec["anchor_wall"] is not None:
            _paint_vertical_wall(expected, spec["rows"], int(spec["anchor_wall"]), obstacle_color)
        if spec["block_wall"] is not None:
            _paint_vertical_wall(expected, spec["rows"], int(spec["block_wall"]), obstacle_color)

    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == bg)[0][0]])
    if np.any((values != bg) & (counts >= bg_count)):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        result = _render_base_scene()
        if result is None:
            continue

        grid, expected = result
        if bool(_RNG.integers(0, 2)):
            grid = grid.T
            expected = expected.T

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid ruler-stripe sample")

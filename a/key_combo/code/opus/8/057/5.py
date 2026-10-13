import random
from typing import Any

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

BAR_COLORS = (BLUE, RED, GREEN)
PATTERN_COLORS = (YELLOW, ORANGE, TEAL)
RECOLOR = {BLUE: None, RED: PINK, GREEN: MAROON}

MAX_ATTEMPTS = 400


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey separator column"
    sep = int(cols[0])
    assert 0 < sep < grid.shape[1] - 1, "separator must split the grid into non-empty halves"
    return sep


def _decode_bar(row: np.ndarray) -> tuple[int, int | None, int | None]:
    cols = np.flatnonzero(row != BG)
    if cols.size == 0:
        return 0, None, None

    c0, c1 = int(cols[0]), int(cols[-1])
    assert np.array_equal(cols, np.arange(c0, c1 + 1)), "left-side cue must be one contiguous bar"
    values = row[c0 : c1 + 1]
    assert np.unique(values).size == 1, "bar must be monochrome"

    color = int(values[0])
    length = int(c1 - c0 + 1)
    assert color in BAR_COLORS, "bar color must be blue, red, or green"
    assert 1 <= length <= 4, "bar length must be in 1..4"
    return length, color, c0


def _shift_right(row: np.ndarray, amount: int) -> np.ndarray:
    out = np.zeros_like(row)
    if amount < row.size:
        out[amount:] = row[: row.size - amount]
    return out


def _transform_row(row: np.ndarray, length: int, color: int | None) -> np.ndarray:
    shifted = _shift_right(row, length)
    if color is None or color == BLUE:
        return shifted
    out = shifted.copy()
    out[out != BG] = RECOLOR[color]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode each left-side bar into a row-specific shift and recolor for the right-side pattern.

    Input structure
    - A full-height grey vertical line splits the grid into a left cue area and a right pattern area.
    - In the left area, each row contains at most one contiguous horizontal bar on black background.
    - Cue bars use blue, red, or green; the right area contains the multicolor source pattern on black background.

    Rule
    - For each row, read the same-row left cue bar.
    - The bar length gives the rightward shift amount for that row of the right area; missing bars mean shift 0.
    - The bar color gives the recolor rule for non-black cells after shifting: blue keeps colors, red turns them pink, and green turns them maroon.
    - Output only the transformed right area, with the same height and right-half width as in the input.

    Key insight
    - Each bar is a two-channel code: its length controls geometry while its color independently controls recoloring, so both must be decoded row by row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    out = np.zeros_like(right)
    for r in range(g.shape[0]):
        length, color, _start = _decode_bar(left[r])
        out[r] = _transform_row(right[r], length, color)
    return out


def _make_template(width: int, start: int, length: int, variant: int, phase: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    colors = np.array(PATTERN_COLORS, dtype=int)

    if variant == 0:
        seq = colors[(phase + np.arange(length)) % len(colors)]
    elif variant == 1:
        cuts = sorted(random.sample(range(1, length), 2))
        spans = [0, cuts[0], cuts[1], length]
        seq = np.empty(length, dtype=int)
        order = [colors[(phase + i) % len(colors)] for i in range(3)]
        for i in range(3):
            seq[spans[i] : spans[i + 1]] = order[i]
    else:
        seq = np.empty(length, dtype=int)
        for i in range(length):
            idx = phase + min(i, length - 1 - i) + (i % 2)
            seq[i] = colors[idx % len(colors)]

    assert len(set(map(int, seq))) >= 2
    row[start : start + length] = seq
    return row


def _sample_template_set(right_w: int) -> dict[str, np.ndarray]:
    core_start = random.randint(0, min(2, right_w - 5))
    core_len = random.randint(5, min(7, right_w - core_start))
    templates = {}
    variants = random.sample([0, 1, 2], 3)
    phases = random.sample([0, 1, 2], 3)
    templates["shift"] = _make_template(right_w, core_start, core_len, variants[0], phases[0])
    templates["color"] = _make_template(right_w, core_start, core_len, variants[1], phases[1])
    templates["none"] = _make_template(right_w, core_start, core_len, variants[2], phases[2])
    templates["extra_a"] = _make_template(
        right_w,
        core_start,
        core_len,
        random.choice([0, 1, 2]),
        random.randint(0, 2),
    )
    templates["extra_b"] = _make_template(
        right_w,
        core_start,
        core_len,
        random.choice([0, 1, 2]),
        random.randint(0, 2),
    )
    return templates


def _wrong_transform(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    out = np.zeros_like(right)

    for r in range(g.shape[0]):
        length, color, start = _decode_bar(left[r])
        if mode == "ignore_color":
            out[r] = _shift_right(right[r], length)
        elif mode == "ignore_length":
            row = right[r].copy()
            if color in (RED, GREEN):
                row[row != BG] = RECOLOR[color]
            out[r] = row
        elif mode == "use_start":
            wrong_length = 0 if color is None else int(start + 1)
            out[r] = _transform_row(right[r], wrong_length, color)
        else:
            raise ValueError(mode)
    return out


def _validate_instance(grid: np.ndarray, meta: dict[str, Any]) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        h, w = g.shape
        if g.ndim != 2 or max(h, w) > 30:
            return False

        sep = _find_separator_col(g)
        if sep != meta["sep"]:
            return False
        left = g[:, :sep]
        right = g[:, sep + 1 :]
        if right.shape[1] != meta["right_w"]:
            return False

        decoded: list[tuple[int, int | None, int | None]] = []
        seen_colors = set()
        seen_lengths = set()
        start_length_mismatch = False
        for r in range(h):
            info = _decode_bar(left[r])
            decoded.append(info)
            length, color, start = info
            if color is not None:
                seen_colors.add(color)
                seen_lengths.add(length)
                if start is not None and start + 1 != length:
                    start_length_mismatch = True
        if seen_colors != set(BAR_COLORS) or len(seen_lengths) < 2 or not start_length_mismatch:
            return False

        out = main(g)
        if out.shape != right.shape:
            return False
        if np.array_equal(out, right):
            return False
        if np.array_equal(out, _wrong_transform(g, "ignore_color")):
            return False
        if np.array_equal(out, _wrong_transform(g, "ignore_length")):
            return False
        if np.array_equal(out, _wrong_transform(g, "use_start")):
            return False

        blue_a, blue_b = meta["blue_pair"]
        if not np.array_equal(right[blue_a], right[blue_b]):
            return False
        if decoded[blue_a][1] != BLUE or decoded[blue_b][1] != BLUE:
            return False
        if decoded[blue_a][0] == decoded[blue_b][0]:
            return False
        if not np.any(right[blue_a] != BG):
            return False
        if np.array_equal(out[blue_a], out[blue_b]):
            return False

        red_row, green_row = meta["color_pair"]
        if not np.array_equal(right[red_row], right[green_row]):
            return False
        if decoded[red_row][0] != decoded[green_row][0]:
            return False
        if decoded[red_row][1] != RED or decoded[green_row][1] != GREEN:
            return False
        if not np.any(out[red_row] == PINK) or not np.any(out[green_row] == MAROON):
            return False
        if np.array_equal(out[red_row], out[green_row]):
            return False

        none_row, paired_row = meta["none_pair"]
        if decoded[none_row][1] is not None or decoded[none_row][0] != 0:
            return False
        if decoded[paired_row][1] != BLUE or decoded[paired_row][0] == 0:
            return False
        if not np.array_equal(right[none_row], right[paired_row]):
            return False
        if np.array_equal(out[none_row], out[paired_row]):
            return False
        if not np.array_equal(out[none_row], right[none_row]):
            return False

        if meta["active_rows"] != sorted(meta["active_rows"]):
            return False
        if any(not np.any(right[r] != BG) for r in meta["active_rows"]):
            return False
        if any(np.any(right[r] != BG) for r in range(h) if r not in set(meta["active_rows"])):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    right_w = random.randint(8, 10)
    left_w = random.randint(5, 7)
    band_h = random.randint(6, 8)
    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 2)
    height = band_h + top_pad + bottom_pad
    if height > 12:
        return None

    active_rows = list(range(top_pad, top_pad + band_h))
    if len(active_rows) < 6:
        return None
    role_rows = random.sample(active_rows, 6)
    none_row, none_pair_row, blue_a, blue_b, red_row, green_row = role_rows

    lengths = random.sample([1, 2, 3, 4], 3)
    blue_lengths = lengths[:2]
    color_pair_length = lengths[2]
    none_pair_length = random.choice([v for v in [1, 2, 3, 4] if v != 0])

    templates = _sample_template_set(right_w)

    grid = np.zeros((height, left_w + 1 + right_w), dtype=int)
    sep = left_w
    grid[:, sep] = GREY

    row_plan: dict[int, tuple[str, int | None, int | None]] = {}
    row_plan[none_row] = ("none", None, None)
    row_plan[none_pair_row] = ("none", BLUE, none_pair_length)
    row_plan[blue_a] = ("shift", BLUE, blue_lengths[0])
    row_plan[blue_b] = ("shift", BLUE, blue_lengths[1])
    row_plan[red_row] = ("color", RED, color_pair_length)
    row_plan[green_row] = ("color", GREEN, color_pair_length)

    extra_templates = ["extra_a", "extra_b", "shift", "color"]
    for r in active_rows:
        if r in row_plan:
            continue
        template_name = random.choice(extra_templates)
        if random.random() < 0.25:
            row_plan[r] = (template_name, None, None)
        else:
            color = random.choice(BAR_COLORS)
            length = random.randint(1, 4)
            row_plan[r] = (template_name, color, length)

    for r in active_rows:
        template_name, color, length = row_plan[r]
        grid[r, sep + 1 :] = templates[template_name]
        if color is not None:
            max_start = left_w - length
            if max_start < 0:
                return None
            start = random.randint(0, max_start)
            grid[r, start : start + length] = color

    meta = {
        "sep": sep,
        "right_w": right_w,
        "blue_pair": tuple(sorted((blue_a, blue_b))),
        "color_pair": tuple(sorted((red_row, green_row))),
        "none_pair": (none_row, none_pair_row),
        "active_rows": sorted(active_rows),
    }
    if not _validate_instance(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bar-coded row-transform ARC input")

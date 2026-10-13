from __future__ import annotations

from dataclasses import dataclass
import numpy as np


BG = 0
BOX = 3
BAR_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
MIN_WIDTH = 18
MAX_WIDTH = 30
MIN_CAPACITY = 4
MAX_CAPACITY = 8


@dataclass(frozen=True)
class Bar:
    color: int
    c0: int
    c1: int

    @property
    def length(self) -> int:
        return int(self.c1 - self.c0 + 1)


@dataclass(frozen=True)
class ParsedBand:
    top: int
    mid: int
    bottom: int
    box_c0: int
    box_c1: int
    tab_side: str
    left_bars: tuple[Bar, ...]
    right_bars: tuple[Bar, ...]

    @property
    def capacity(self) -> int:
        return int(self.box_c1 - self.box_c0 - 1)


def _row_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return []
    blocks: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for value in rows[1:]:
        row = int(value)
        if row == prev + 1:
            prev = row
            continue
        blocks.append((start, prev))
        start = prev = row
    blocks.append((start, prev))
    return blocks


def _parse_bar_runs(row: np.ndarray, start: int, stop: int) -> tuple[Bar, ...]:
    bars: list[Bar] = []
    c = int(start)
    while c <= int(stop):
        if int(row[c]) in (BG, BOX):
            c += 1
            continue
        color = int(row[c])
        c0 = c
        while c <= int(stop) and int(row[c]) == color:
            c += 1
        c1 = c - 1
        bars.append(Bar(color=color, c0=c0, c1=c1))
    return tuple(bars)


def _parse_band(grid: np.ndarray, top: int, bottom: int) -> ParsedBand:
    assert bottom - top + 1 == 3, "each band must be exactly three rows tall"
    band = grid[top : bottom + 1]
    assert np.all((band[0] == BG) | (band[0] == BOX)), "top band row may only contain box pixels"
    assert np.all((band[2] == BG) | (band[2] == BOX)), "bottom band row may only contain box pixels"

    frame_cols = np.flatnonzero((band[0] == BOX) & (band[2] == BOX))
    assert frame_cols.size >= 3, "box frame must be at least three columns wide"
    box_c0 = int(frame_cols[0])
    box_c1 = int(frame_cols[-1])
    assert np.array_equal(frame_cols, np.arange(box_c0, box_c1 + 1)), "box top/bottom edges must be contiguous"
    assert np.all(band[0, box_c0 : box_c1 + 1] == BOX)
    assert np.all(band[2, box_c0 : box_c1 + 1] == BOX)

    mid = top + 1
    mid_row = band[1]
    assert int(mid_row[box_c0]) == BOX and int(mid_row[box_c1]) == BOX, "side walls must exist"
    if box_c1 - box_c0 > 1:
        assert np.all(mid_row[box_c0 + 1 : box_c1] == BG), "box cavity must be a single empty row"

    extras = [c for c in np.flatnonzero(mid_row == BOX) if c not in (box_c0, box_c1)]
    assert len(extras) == 1, "expected exactly one tab pixel"
    tab_col = int(extras[0])
    if tab_col == box_c0 - 1:
        tab_side = "left"
    else:
        assert tab_col == box_c1 + 1, "tab must touch a box side"
        tab_side = "right"

    left_bars = _parse_bar_runs(mid_row, 0, box_c0 - 1)
    right_bars = _parse_bar_runs(mid_row, box_c1 + 1, grid.shape[1] - 1)
    assert left_bars and right_bars, "each band must contain bars on both sides"

    if tab_side == "left":
        assert all(bar.c1 <= box_c0 - 2 for bar in left_bars), "left-tab bars must stay left of the tab"
    else:
        assert all(bar.c0 >= box_c1 + 2 for bar in right_bars), "right-tab bars must stay right of the tab"

    left_sorted = tuple(sorted(left_bars, key=lambda bar: -bar.c1))
    right_sorted = tuple(sorted(right_bars, key=lambda bar: bar.c0))
    return ParsedBand(
        top=top,
        mid=mid,
        bottom=bottom,
        box_c0=box_c0,
        box_c1=box_c1,
        tab_side=tab_side,
        left_bars=left_sorted,
        right_bars=right_sorted,
    )


def _parse_grid(grid: np.ndarray) -> list[ParsedBand]:
    nonempty = np.any(grid != BG, axis=1)
    blocks = _row_blocks(nonempty)
    assert blocks, "expected at least one band"
    return [_parse_band(grid, top, bottom) for top, bottom in blocks]


def _split_entrance(parsed: ParsedBand) -> tuple[tuple[Bar, ...], tuple[Bar, ...], tuple[Bar, ...]]:
    entrance = parsed.left_bars if parsed.tab_side == "left" else parsed.right_bars
    blocked = parsed.right_bars if parsed.tab_side == "left" else parsed.left_bars
    inside: list[Bar] = []
    overflow: list[Bar] = []
    remaining = parsed.capacity
    spilling = False
    for bar in entrance:
        if not spilling and bar.length <= remaining:
            inside.append(bar)
            remaining -= bar.length
        else:
            spilling = True
            overflow.append(bar)
    return tuple(inside), tuple(overflow), tuple(blocked)


def _paint_segment(row: np.ndarray, c0: int, c1: int, color: int) -> None:
    assert 0 <= c0 <= c1 < row.shape[0], "segment out of bounds"
    assert np.all(row[c0 : c1 + 1] == BG), "segments may only land on background"
    row[c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide horizontal bars into a tabbed hollow box, letting the tab side act as the only entrance and spilling excess bars outside.

    Input structure:
    - A black background is split into horizontal bands by full blank rows.
    - Each band contains one green hollow box whose cavity is a single black row, plus one extra green tab pixel on the left or right wall midpoint.
    - Colored foreground objects other than the box are single-row horizontal bars placed on both sides of the box.

    Rule:
    - All bars slide horizontally toward the box while the green box and its tab stay fixed.
    - Bars on the tab side are processed from nearest to farthest: each fitting bar enters the cavity and is packed tightly against the opposite inner wall; once one bar is too long for the remaining cavity, that bar and all later bars queue outside on the entrance side.
    - Bars on the non-tab side never enter; they queue tightly outside the plain wall, while background is restored where bars moved away.

    Key insight:
    - The lone tab pixel is the functional entrance, and the decisive branch is not bar length alone but whether the next bar still fits after earlier admitted bars have already consumed cavity capacity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= set((BG, BOX, *BAR_COLORS)), "unexpected colors in grid"

    bands = _parse_grid(g)
    out = g.copy()
    out[(out != BG) & (out != BOX)] = BG

    for parsed in bands:
        row = out[parsed.mid]
        inside, overflow, blocked = _split_entrance(parsed)

        if parsed.tab_side == "left":
            cursor = parsed.box_c1
            for bar in inside:
                start = cursor - bar.length
                _paint_segment(row, start, cursor - 1, bar.color)
                cursor = start

            cursor = parsed.box_c0 - 1
            for bar in overflow:
                start = cursor - bar.length
                _paint_segment(row, start, cursor - 1, bar.color)
                cursor = start

            cursor = parsed.box_c1 + 1
            for bar in blocked:
                end = cursor + bar.length - 1
                _paint_segment(row, cursor, end, bar.color)
                cursor = end + 1
        else:
            cursor = parsed.box_c0 + 1
            for bar in inside:
                end = cursor + bar.length - 1
                _paint_segment(row, cursor, end, bar.color)
                cursor = end + 1

            cursor = parsed.box_c1 + 2
            for bar in overflow:
                end = cursor + bar.length - 1
                _paint_segment(row, cursor, end, bar.color)
                cursor = end + 1

            cursor = parsed.box_c0
            for bar in blocked:
                start = cursor - bar.length
                _paint_segment(row, start, cursor - 1, bar.color)
                cursor = start

    return out


def _sample_lengths_with_sum_limit(rng: np.random.Generator, count: int, limit: int, *, at_least_one_shortfall: bool) -> list[int] | None:
    for _ in range(200):
        lengths = [int(rng.integers(1, min(5, limit + 1))) for _ in range(count)]
        total = sum(lengths)
        if total > limit:
            continue
        if at_least_one_shortfall and total == limit and count == 1:
            continue
        return lengths
    return None


def _build_entrance_lengths(
    rng: np.random.Generator,
    capacity: int,
    overflow_mode: bool,
    soft_overflow: bool,
) -> tuple[list[int], bool]:
    if not overflow_mode:
        count = int(rng.integers(1, 4))
        lengths = _sample_lengths_with_sum_limit(rng, count, capacity, at_least_one_shortfall=False)
        assert lengths is not None
        return lengths, False

    for _ in range(300):
        count = int(rng.integers(2, 5))
        inside_count = int(rng.integers(1, count))
        remaining = capacity
        inside: list[int] = []
        ok = True
        for idx in range(inside_count):
            slots_left = inside_count - idx - 1
            max_len = min(4, remaining - slots_left)
            if max_len < 1:
                ok = False
                break
            length = int(rng.integers(1, max_len + 1))
            inside.append(length)
            remaining -= length
        if not ok:
            continue

        if soft_overflow:
            low = remaining + 1
            high = min(4, capacity)
        else:
            low = max(remaining + 1, 1)
            high = min(6, capacity + 2)
        if low > high:
            continue

        trigger = int(rng.integers(low, high + 1))
        tail = [int(rng.integers(1, 5)) for _ in range(count - inside_count - 1)]
        return inside + [trigger] + tail, True
    raise RuntimeError("failed to build an overflow entrance sequence")


def _sample_side_spacing(rng: np.random.Generator, count: int) -> tuple[int, list[int]]:
    lead_gap = int(rng.integers(1, 4))
    between = [int(rng.integers(1, 4)) for _ in range(max(0, count - 1))]
    return lead_gap, between


def _side_extent(lengths: list[int], base_skip: int, lead_gap: int, between: list[int]) -> int:
    return int(base_skip + lead_gap + sum(lengths) + sum(between))


def _place_left_bars(c0: int, lengths: list[int], colors: list[int], *, base_skip: int, lead_gap: int, between: list[int]) -> list[dict]:
    right_end = c0 - 1 - base_skip - lead_gap
    bars: list[dict] = []
    for idx, (length, color) in enumerate(zip(lengths, colors, strict=True)):
        start = right_end - length + 1
        bars.append({"color": int(color), "length": int(length), "c0": int(start), "c1": int(right_end)})
        if idx < len(between):
            right_end = start - 1 - between[idx]
    return bars


def _place_right_bars(c1: int, lengths: list[int], colors: list[int], *, base_skip: int, lead_gap: int, between: list[int]) -> list[dict]:
    left_start = c1 + 1 + base_skip + lead_gap
    bars: list[dict] = []
    for idx, (length, color) in enumerate(zip(lengths, colors, strict=True)):
        end = left_start + length - 1
        bars.append({"color": int(color), "length": int(length), "c0": int(left_start), "c1": int(end)})
        if idx < len(between):
            left_start = end + 1 + between[idx]
    return bars


def _render_box(grid: np.ndarray, top: int, c0: int, c1: int, tab_side: str) -> None:
    grid[top, c0 : c1 + 1] = BOX
    grid[top + 2, c0 : c1 + 1] = BOX
    grid[top + 1, c0] = BOX
    grid[top + 1, c1] = BOX
    if tab_side == "left":
        grid[top + 1, c0 - 1] = BOX
    else:
        grid[top + 1, c1 + 1] = BOX


def _render_bars(grid: np.ndarray, row: int, bars: list[dict]) -> None:
    for bar in bars:
        assert np.all(grid[row, bar["c0"] : bar["c1"] + 1] == BG), "input bars must not overlap"
        grid[row, bar["c0"] : bar["c1"] + 1] = int(bar["color"])


def _simulate_band_meta(band: dict) -> tuple[list[dict], list[dict], list[dict]]:
    entrance_side = "left" if band["tab_side"] == "left" else "right"
    entrance = band[entrance_side]
    blocked = band["right" if entrance_side == "left" else "left"]
    inside: list[dict] = []
    overflow: list[dict] = []
    remaining = int(band["capacity"])
    spilling = False
    for bar in entrance:
        if not spilling and int(bar["length"]) <= remaining:
            inside.append(bar)
            remaining -= int(bar["length"])
        else:
            spilling = True
            overflow.append(bar)
    return inside, overflow, blocked


def _render_output_from_meta(case: dict) -> np.ndarray:
    out = np.zeros((case["height"], case["width"]), dtype=int)
    for band in case["bands"]:
        top = int(band["top"])
        row = top + 1
        c0 = int(band["box_c0"])
        c1 = int(band["box_c1"])
        _render_box(out, top, c0, c1, band["tab_side"])
        inside, overflow, blocked = _simulate_band_meta(band)

        if band["tab_side"] == "left":
            cursor = c1
            for bar in inside:
                start = cursor - int(bar["length"])
                out[row, start:cursor] = int(bar["color"])
                cursor = start

            cursor = c0 - 1
            for bar in overflow:
                start = cursor - int(bar["length"])
                out[row, start:cursor] = int(bar["color"])
                cursor = start

            cursor = c1 + 1
            for bar in blocked:
                end = cursor + int(bar["length"])
                out[row, cursor:end] = int(bar["color"])
                cursor = end
        else:
            cursor = c0 + 1
            for bar in inside:
                end = cursor + int(bar["length"])
                out[row, cursor:end] = int(bar["color"])
                cursor = end

            cursor = c1 + 2
            for bar in overflow:
                end = cursor + int(bar["length"])
                out[row, cursor:end] = int(bar["color"])
                cursor = end

            cursor = c0
            for bar in blocked:
                start = cursor - int(bar["length"])
                out[row, start:cursor] = int(bar["color"])
                cursor = start
    return out


def _build_band_spec(
    rng: np.random.Generator,
    *,
    tab_side: str,
    overflow_mode: bool,
    soft_overflow: bool,
) -> dict:
    capacity = int(rng.integers(MIN_CAPACITY, MAX_CAPACITY + 1))
    entrance_lengths, overflow_expected = _build_entrance_lengths(rng, capacity, overflow_mode, soft_overflow)
    blocked_count = int(rng.integers(1, 4))
    blocked_lengths = [int(rng.integers(1, 5)) for _ in range(blocked_count)]

    total_bar_count = len(entrance_lengths) + len(blocked_lengths)
    colors = list(map(int, rng.choice(np.array(BAR_COLORS), size=total_bar_count, replace=False)))
    entrance_colors = colors[: len(entrance_lengths)]
    blocked_colors = colors[len(entrance_lengths) :]

    left_is_entrance = tab_side == "left"
    left_lengths = entrance_lengths if left_is_entrance else blocked_lengths
    right_lengths = blocked_lengths if left_is_entrance else entrance_lengths
    left_colors = entrance_colors if left_is_entrance else blocked_colors
    right_colors = blocked_colors if left_is_entrance else entrance_colors

    left_lead, left_between = _sample_side_spacing(rng, len(left_lengths))
    right_lead, right_between = _sample_side_spacing(rng, len(right_lengths))
    left_base = 1 if left_is_entrance else 0
    right_base = 1 if not left_is_entrance else 0

    left_extent = _side_extent(left_lengths, left_base, left_lead, left_between)
    right_extent = _side_extent(right_lengths, right_base, right_lead, right_between)
    box_w = capacity + 2
    min_width = left_extent + box_w + right_extent

    return {
        "tab_side": tab_side,
        "capacity": capacity,
        "box_w": box_w,
        "left_lengths": left_lengths,
        "right_lengths": right_lengths,
        "left_colors": left_colors,
        "right_colors": right_colors,
        "left_lead": left_lead,
        "right_lead": right_lead,
        "left_between": left_between,
        "right_between": right_between,
        "left_base": left_base,
        "right_base": right_base,
        "left_extent": left_extent,
        "right_extent": right_extent,
        "min_width": min_width,
        "overflow_expected": overflow_expected,
        "soft_overflow": bool(soft_overflow and overflow_expected),
    }


def _validate_case(grid: np.ndarray, case: dict) -> bool:
    try:
        parsed_bands = _parse_grid(grid)
    except AssertionError:
        return False

    if len(parsed_bands) != len(case["bands"]):
        return False

    for parsed, band in zip(parsed_bands, case["bands"], strict=True):
        if parsed.tab_side != band["tab_side"]:
            return False
        if parsed.box_c0 != band["box_c0"] or parsed.box_c1 != band["box_c1"]:
            return False
        if [bar.length for bar in parsed.left_bars] != [int(bar["length"]) for bar in band["left"]]:
            return False
        if [bar.color for bar in parsed.left_bars] != [int(bar["color"]) for bar in band["left"]]:
            return False
        if [bar.length for bar in parsed.right_bars] != [int(bar["length"]) for bar in band["right"]]:
            return False
        if [bar.color for bar in parsed.right_bars] != [int(bar["color"]) for bar in band["right"]]:
            return False

    directions = {band["tab_side"] for band in case["bands"]}
    if len(case["bands"]) >= 2 and directions != {"left", "right"}:
        return False

    overflow_flags = [bool(band["overflow_expected"]) for band in case["bands"]]
    if not any(overflow_flags) or not any(not flag for flag in overflow_flags):
        return False
    if not any(bool(band["soft_overflow"]) for band in case["bands"] if band["overflow_expected"]):
        return False

    expected = _render_output_from_meta(case)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _build_case(rng: np.random.Generator) -> dict | None:
    band_count = int(rng.integers(2, 5))

    directions = ["left", "right"] + [str(rng.choice(np.array(["left", "right"]))) for _ in range(max(0, band_count - 2))]
    rng.shuffle(directions)

    overflow_modes = [True, False] + [bool(rng.integers(0, 2)) for _ in range(max(0, band_count - 2))]
    rng.shuffle(overflow_modes)

    overflow_indices = [idx for idx, value in enumerate(overflow_modes) if value]
    soft_index = int(rng.choice(np.array(overflow_indices)))

    raw_bands: list[dict] = []
    for idx in range(band_count):
        raw_bands.append(
            _build_band_spec(
                rng,
                tab_side=directions[idx],
                overflow_mode=overflow_modes[idx],
                soft_overflow=idx == soft_index,
            )
        )

    min_width = max(int(band["min_width"]) for band in raw_bands)
    if min_width > MAX_WIDTH:
        return None
    width = int(rng.integers(max(MIN_WIDTH, min_width), MAX_WIDTH + 1))

    top_pad = int(rng.integers(0, 3))
    bottom_pad = int(rng.integers(0, 3))
    separators = [int(rng.integers(1, 3)) for _ in range(max(0, band_count - 1))]
    height = top_pad + bottom_pad + 3 * band_count + sum(separators)
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    bands: list[dict] = []
    top = top_pad
    for idx, raw in enumerate(raw_bands):
        min_c0 = int(raw["left_extent"])
        max_c0 = int(width - raw["right_extent"] - raw["box_w"])
        if min_c0 > max_c0:
            return None
        box_c0 = int(rng.integers(min_c0, max_c0 + 1))
        box_c1 = box_c0 + int(raw["box_w"]) - 1

        left_bars = _place_left_bars(
            box_c0,
            raw["left_lengths"],
            raw["left_colors"],
            base_skip=int(raw["left_base"]),
            lead_gap=int(raw["left_lead"]),
            between=list(raw["left_between"]),
        )
        right_bars = _place_right_bars(
            box_c1,
            raw["right_lengths"],
            raw["right_colors"],
            base_skip=int(raw["right_base"]),
            lead_gap=int(raw["right_lead"]),
            between=list(raw["right_between"]),
        )

        _render_box(grid, top, box_c0, box_c1, str(raw["tab_side"]))
        _render_bars(grid, top + 1, left_bars + right_bars)

        band = {
            "top": top,
            "box_c0": box_c0,
            "box_c1": box_c1,
            "capacity": int(raw["capacity"]),
            "tab_side": str(raw["tab_side"]),
            "left": left_bars,
            "right": right_bars,
            "overflow_expected": bool(raw["overflow_expected"]),
            "soft_overflow": bool(raw["soft_overflow"]),
        }
        bands.append(band)
        if idx < len(separators):
            top += 3 + separators[idx]

    case = {"height": height, "width": width, "bands": bands}
    if not _validate_case(grid, case):
        return None
    case["input"] = grid
    return case


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid tabbed-box intake puzzle")

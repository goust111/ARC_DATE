from __future__ import annotations

import random
from collections import Counter

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> np.ndarray:
    h, w = patch.shape
    dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = patch
    return dst


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _is_solid_rectangle(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if int(comp["area"]) != area:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])))


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(int(a[0]), int(b[0])) <= min(int(a[1]), int(b[1]))


def _build_channels(rects: list[dict]) -> list[list[dict]]:
    n = len(rects)
    seen = [False] * n
    channels: list[list[dict]] = []

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        group_idx: list[int] = []
        while stack:
            idx = stack.pop()
            group_idx.append(idx)
            a = rects[idx]["bbox"][1], rects[idx]["bbox"][3]
            for nxt in range(n):
                if seen[nxt]:
                    continue
                b = rects[nxt]["bbox"][1], rects[nxt]["bbox"][3]
                if _intervals_overlap(a, b):
                    seen[nxt] = True
                    stack.append(nxt)

        group = [rects[idx] for idx in group_idx]
        group.sort(key=lambda comp: int(comp["bbox"][0]), reverse=True)
        for lower, upper in zip(group, group[1:]):
            lower_r0 = int(lower["bbox"][0])
            upper_r1 = int(upper["bbox"][2])
            assert lower_r0 > upper_r1, "channel rectangles must be vertically separated"
        channels.append(group)
    return channels


def _seed_match(channel: list[dict], pixel: tuple[int, int]) -> bool:
    bottom = channel[0]
    r0, c0, r1, c1 = bottom["bbox"]
    r, c = pixel
    return int(r) == int(r1) + 1 and int(c0) <= int(c) <= int(c1)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[list[dict]], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected foreground objects"

    rectangles: list[dict] = []
    other: list[dict] = []
    for comp in components:
        if int(comp["area"]) > 1 and _is_solid_rectangle(g, comp):
            rectangles.append(comp)
        else:
            other.append(comp)

    assert rectangles, "expected rectangle components"
    assert len(other) == 1 and int(other[0]["area"]) == 1, "expected exactly one seed pixel"
    seed = tuple(map(int, other[0]["coords"][0]))

    channels = _build_channels(rectangles)
    seed_channels = [idx for idx, channel in enumerate(channels) if _seed_match(channel, seed)]
    assert len(seed_channels) == 1, "seed must identify exactly one channel"
    return bg, channels, seed


def _compute_channel_moves(channel: list[dict]) -> list[int]:
    moves = [0] * len(channel)
    impulse = _bbox_h(channel[0]["bbox"])
    for idx, rect in enumerate(channel):
        if idx == len(channel) - 1:
            moves[idx] = int(impulse)
            break
        gap = int(rect["bbox"][0] - channel[idx + 1]["bbox"][2] - 1)
        assert gap > 0, "channel rectangles must not touch"
        if impulse <= gap:
            moves[idx] = int(impulse)
            break
        moves[idx] = int(gap)
        impulse -= gap
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """Move only the seeded stack by passing a height-sized impulse upward through its gaps.

    Input structure
    A black background contains several solid monochromatic rectangles arranged in vertical channels. Rectangles
    within one channel have overlapping horizontal spans, are ordered bottom-to-top, and are separated by blank
    rows. Exactly one channel has a single seed pixel directly below its bottom rectangle.

    Rule
    The bottom rectangle in the seeded channel receives impulse E equal to its own height. For a rectangle carrying
    E, let g be the blank-row gap to the nearest rectangle above in the same channel, or infinity if none exists.
    If E <= g, move that rectangle up by E rows and stop. If E > g, move it up by g rows to reach the next
    rectangle, then pass residual impulse E-g to that upper rectangle and continue. All rectangles in unseeded
    channels stay fixed, and the seed pixel remains in place.

    Key insight
    The motion is not “each rectangle moves by its own height”. Only the seeded channel changes, and every crossed
    gap subtracts exactly that many rows from the impulse before the next rectangle can react.
    """
    g = np.asarray(grid, dtype=int)
    bg, channels, seed = _parse_scene(g)
    seed_channel_idx = next(idx for idx, channel in enumerate(channels) if _seed_match(channel, seed))
    moves = _compute_channel_moves(channels[seed_channel_idx])

    out = g.copy()
    for rect, shift in zip(channels[seed_channel_idx], moves):
        if shift <= 0:
            continue
        coords = rect["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for rect, shift in zip(channels[seed_channel_idx], moves):
        if shift <= 0:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        patch = np.full((_bbox_h(rect["bbox"]), c1 - c0 + 1), int(rect["color"]), dtype=int)
        _paste(out, patch, r0 - shift, c0)
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_PALETTE = list(range(1, 10))
GEN_HEIGHT_RANGE = (2, 4)
GEN_CHANNEL_WIDTH_RANGE = (2, 5)
GEN_GAP_RANGE = (1, 4)
GEN_CHANNEL_COUNT_RANGE = (2, 4)
GEN_RECT_COUNT_RANGE = (1, 4)
GEN_HORIZONTAL_GAP_RANGE = (2, 3)
GEN_MARGIN_MIN = 1
GEN_EXTRA_CANVAS = 6
GEN_MAX_ATTEMPTS = 400
GEN_SEED_CASES = ("single", "stop_bottom", "stop_mid", "reach_top")
GEN_SEED_WEIGHTS = (1, 1, 2, 2)


def _rand_height() -> int:
    return random.randint(*GEN_HEIGHT_RANGE)


def _rand_gap() -> int:
    return random.randint(*GEN_GAP_RANGE)


def _make_channel_columns(count: int) -> tuple[list[tuple[int, int]], int]:
    channel_w = random.randint(*GEN_CHANNEL_WIDTH_RANGE)
    core_w = random.randint(1, min(3, channel_w))
    core_l = random.randint(0, channel_w - core_w)
    core_r = core_l + core_w - 1

    intervals = []
    for _ in range(count):
        c0 = random.randint(0, core_l)
        c1 = random.randint(core_r, channel_w - 1)
        intervals.append((int(c0), int(c1)))
    return intervals, int(channel_w)


def _relative_moves_from_rects(rects: list[dict]) -> list[int]:
    channel = []
    for rect in rects:
        channel.append(
            {
                "bbox": (int(rect["r0"]), int(rect["c0"]), int(rect["r1"]), int(rect["c1"])),
            }
        )
    return _compute_channel_moves(channel)


def _build_channel_spec(
    heights: list[int],
    gaps: list[int],
    *,
    with_seed: bool,
) -> dict:
    count = len(heights)
    intervals, channel_w = _make_channel_columns(count)
    colors = random.sample(GEN_PALETTE, k=min(len(GEN_PALETTE), count))
    if count > len(colors):
        colors.extend(random.choices(GEN_PALETTE, k=count - len(colors)))

    bottom_r1 = sum(heights) + sum(gaps) - 1
    rects: list[dict] = []
    prev_r0 = None
    for idx in range(count):
        h = int(heights[idx])
        if idx == 0:
            r1 = int(bottom_r1)
        else:
            r1 = int(prev_r0 - gaps[idx - 1] - 1)
        r0 = int(r1 - h + 1)
        c0, c1 = intervals[idx]
        rects.append(
            {
                "r0": r0,
                "c0": int(c0),
                "r1": r1,
                "c1": int(c1),
                "color": int(colors[idx]),
            }
        )
        prev_r0 = r0

    moves = _relative_moves_from_rects(rects) if with_seed else [0] * count
    top_clearance = max(0, -min(int(rect["r0"]) - int(shift) for rect, shift in zip(rects, moves)))

    seed = None
    full_height = int(bottom_r1 + 1)
    if with_seed:
        bottom = rects[0]
        seed_row = int(bottom["r1"] + 1)
        seed_col = random.randint(int(bottom["c0"]), int(bottom["c1"]))
        seed_choices = [color for color in GEN_PALETTE if color != int(bottom["color"])]
        seed = (seed_row, seed_col, int(random.choice(seed_choices)))
        full_height = int(seed_row + 1)

    return {
        "rects": rects,
        "moves": moves,
        "seed": seed,
        "width": channel_w,
        "full_height": full_height,
        "top_clearance": top_clearance,
    }


def _make_seed_profile(case: str) -> tuple[list[int], list[int]]:
    if case == "single":
        return [_rand_height()], []

    if case == "stop_bottom":
        count = random.randint(2, GEN_RECT_COUNT_RANGE[1])
        bottom_h = _rand_height()
        heights = [bottom_h] + [_rand_height() for _ in range(count - 1)]
        gaps = [random.randint(bottom_h, GEN_GAP_RANGE[1])]
        gaps.extend(_rand_gap() for _ in range(count - 2))
        return heights, gaps

    if case == "stop_mid":
        count = random.randint(3, GEN_RECT_COUNT_RANGE[1])
        stop_idx = random.randint(1, count - 2)
        bottom_h = random.randint(stop_idx + 1, GEN_HEIGHT_RANGE[1])
        heights = [bottom_h] + [_rand_height() for _ in range(count - 1)]
        residual = bottom_h
        gaps: list[int] = []
        for pass_idx in range(stop_idx):
            remaining_passes = stop_idx - pass_idx - 1
            gap = random.randint(1, residual - (remaining_passes + 1))
            gaps.append(gap)
            residual -= gap
        gaps.append(random.randint(residual, GEN_GAP_RANGE[1]))
        gaps.extend(_rand_gap() for _ in range(count - 2 - stop_idx))
        return heights, gaps

    if case == "reach_top":
        count = random.randint(2, GEN_RECT_COUNT_RANGE[1])
        bottom_h = random.randint(count, GEN_HEIGHT_RANGE[1])
        heights = [bottom_h] + [_rand_height() for _ in range(count - 1)]
        residual = bottom_h
        gaps = []
        for gap_idx in range(count - 1):
            remaining_gaps = count - gap_idx - 2
            gap = random.randint(1, residual - (remaining_gaps + 1))
            gaps.append(gap)
            residual -= gap
        return heights, gaps

    raise ValueError(f"unknown seed case: {case}")


def _classify_seed_case(spec: dict) -> str:
    moved = sum(1 for shift in spec["moves"] if int(shift) > 0)
    total = len(spec["rects"])
    if total == 1:
        return "single"
    if moved == 1:
        return "stop_bottom"
    if moved == total:
        return "reach_top"
    return "stop_mid"


def _make_nonseed_spec() -> dict:
    count = random.randint(*GEN_RECT_COUNT_RANGE)
    heights = [_rand_height() for _ in range(count)]
    gaps = [_rand_gap() for _ in range(count - 1)]
    return _build_channel_spec(heights, gaps, with_seed=False)


def _paint_channel(out: np.ndarray, spec: dict, row_offset: int, col_offset: int, moved: bool) -> None:
    for rect, shift in zip(spec["rects"], spec["moves"]):
        delta = int(shift) if moved else 0
        h = int(rect["r1"] - rect["r0"] + 1)
        w = int(rect["c1"] - rect["c0"] + 1)
        patch = np.full((h, w), int(rect["color"]), dtype=int)
        _paste(out, patch, int(row_offset + rect["r0"] - delta), int(col_offset + rect["c0"]))
    if spec["seed"] is not None:
        sr, sc, color = spec["seed"]
        out[int(row_offset + sr), int(col_offset + sc)] = int(color)


def _gen_attempt() -> np.ndarray | None:
    case = random.choices(GEN_SEED_CASES, weights=GEN_SEED_WEIGHTS, k=1)[0]
    seed_heights, seed_gaps = _make_seed_profile(case)
    seed_spec = _build_channel_spec(seed_heights, seed_gaps, with_seed=True)
    if _classify_seed_case(seed_spec) != case:
        return None

    total_channels = random.randint(*GEN_CHANNEL_COUNT_RANGE)
    specs = [seed_spec] + [_make_nonseed_spec() for _ in range(total_channels - 1)]
    random.shuffle(specs)

    between = [random.randint(*GEN_HORIZONTAL_GAP_RANGE) for _ in range(len(specs) - 1)]
    min_w = GEN_MARGIN_MIN * 2 + sum(int(spec["width"]) for spec in specs) + sum(between)
    min_h = max(int(spec["full_height"] + spec["top_clearance"]) for spec in specs)
    if min_w > 30 or min_h > 30:
        return None

    width = random.randint(min_w, min(30, min_w + GEN_EXTRA_CANVAS))
    height = random.randint(min_h, min(30, min_h + GEN_EXTRA_CANVAS))

    slot_count = len(specs) + 1
    extra_w = width - min_w
    slot_extra = [0] * slot_count
    for _ in range(extra_w):
        slot_extra[random.randrange(slot_count)] += 1

    col_offsets = []
    cursor = GEN_MARGIN_MIN + slot_extra[0]
    for idx, spec in enumerate(specs):
        col_offsets.append(int(cursor))
        cursor += int(spec["width"])
        if idx < len(specs) - 1:
            cursor += int(between[idx] + slot_extra[idx + 1])

    row_offsets = []
    for spec in specs:
        low = int(spec["top_clearance"])
        high = int(height - spec["full_height"])
        if low > high:
            return None
        row_offsets.append(random.randint(low, high))

    grid = np.full((height, width), GEN_BG, dtype=int)
    expected = np.full_like(grid, GEN_BG)
    for spec, row_offset, col_offset in zip(specs, row_offsets, col_offsets):
        _paint_channel(grid, spec, row_offset, col_offset, moved=False)
        _paint_channel(expected, spec, row_offset, col_offset, moved=True)

    if _get_bg(grid) != GEN_BG or _get_bg(expected) != GEN_BG:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid impulse-attrition puzzle input")

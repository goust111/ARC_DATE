import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

FRAME_COLOR = GREY
GROUND_COLOR = TEAL
BAR_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, MAROON)

GEN_ATTEMPTS = 400
GEN_FRAME_ROLES = ("seal", "top_mismatch", "side_gap", "bottom_gap")
GEN_FRAME_SIDE_GAPS = ("left", "right")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _frame_gap_cells(r0: int, c0: int, r1: int, c1: int, side: str, start: int, length: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(r1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, c0) for r in range(start, start + length)]
    if side == "right":
        return [(r, c1) for r in range(start, start + length)]
    raise ValueError(f"unknown side: {side}")


def _frame_cells(r0: int, c0: int, r1: int, c1: int, side: str, start: int, length: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    cells -= set(_frame_gap_cells(r0, c0, r1, c1, side, start, length))
    return cells


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for idx, coords in enumerate(_components(grid == FRAME_COLOR)):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("frame too small")

        perimeter = {(r0, c) for c in range(c0, c1 + 1)}
        perimeter |= {(r1, c) for c in range(c0, c1 + 1)}
        perimeter |= {(r, c0) for r in range(r0, r1 + 1)}
        perimeter |= {(r, c1) for r in range(r0, r1 + 1)}

        grey_cells = {tuple(map(int, rc)) for rc in coords.tolist()}
        gap = sorted(perimeter - grey_cells)
        if not gap:
            raise AssertionError("frame must have one perimeter gap")

        rows = {r for r, _ in gap}
        cols = {c for _, c in gap}
        if rows == {r0}:
            side = "top"
            values = sorted(c for _, c in gap)
            lo, hi = c0, c1
        elif rows == {r1}:
            side = "bottom"
            values = sorted(c for _, c in gap)
            lo, hi = c0, c1
        elif cols == {c0}:
            side = "left"
            values = sorted(r for r, _ in gap)
            lo, hi = r0, r1
        elif cols == {c1}:
            side = "right"
            values = sorted(r for r, _ in gap)
            lo, hi = r0, r1
        else:
            raise AssertionError("gap must lie on exactly one side")

        expected = list(range(values[0], values[-1] + 1))
        if values != expected or values[0] <= lo or values[-1] >= hi:
            raise AssertionError("gap must be a contiguous non-corner run")

        interior = np.array(
            [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)],
            dtype=int,
        )
        frames.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "gap_side": side,
                "gap_coords": np.array(gap, dtype=int),
                "gap_key": frozenset(gap),
                "interior": interior,
            }
        )
    return frames


def _parse_bars(grid: np.ndarray, bg: int, ground_color: int) -> list[dict]:
    mask = (grid != bg) & (grid != FRAME_COLOR) & (grid != ground_color)
    bars: list[dict] = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            raise AssertionError("bars must be monochrome")
        bars.append({"coords": coords.copy(), "color": int(next(iter(colors)))})
    return bars


def _simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    bg = _get_bg(g)
    assert bg == BLACK, "expected black background"

    ground_color = int(g[-1, 0])
    assert ground_color != bg and np.all(g[-1] == ground_color), "expected a solid ground line"

    frames = _parse_frames(g)
    bars = _parse_bars(g, bg, ground_color)

    out = np.full_like(g, bg)
    out[-1, :] = ground_color
    out[g == FRAME_COLOR] = FRAME_COLOR

    occupied = np.zeros((h, w), dtype=bool)
    occupied[-1, :] = True
    occupied[g == FRAME_COLOR] = True

    frame_by_gap = {
        frame["gap_key"]: frame for frame in frames if frame["gap_side"] == "top"
    }
    for frame in frames:
        if frame["gap_side"] == "top":
            gap = frame["gap_coords"]
            occupied[gap[:, 0], gap[:, 1]] = True

    sealed_ids: set[int] = set()
    active = [{"coords": bar["coords"].copy(), "color": bar["color"]} for bar in bars]

    for _ in range(h + 2):
        if not active:
            return out

        active.sort(key=lambda bar: int(bar["coords"][:, 0].max()), reverse=True)
        next_active: list[dict] = []

        for bar in active:
            coords = bar["coords"]
            color = bar["color"]
            down = coords + np.array([1, 0], dtype=int)
            if int(down[:, 0].max()) >= h:
                raise AssertionError("bar moved out of bounds")

            down_key = frozenset(map(tuple, down.tolist()))
            frame = frame_by_gap.get(down_key)
            if frame is not None and frame["id"] not in sealed_ids:
                sealed_ids.add(frame["id"])
                out[down[:, 0], down[:, 1]] = color
                occupied[down[:, 0], down[:, 1]] = True
                interior = frame["interior"]
                if interior.size:
                    out[interior[:, 0], interior[:, 1]] = color
                    occupied[interior[:, 0], interior[:, 1]] = True
                continue

            if np.any(occupied[down[:, 0], down[:, 1]]):
                out[coords[:, 0], coords[:, 1]] = color
                occupied[coords[:, 0], coords[:, 1]] = True
            else:
                next_active.append({"coords": down, "color": color})

        active = next_active

    raise AssertionError("simulation did not settle")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop every bar toward the ground line, sealing only exact upward gaps and filling the newly closed interiors.

    Input structure:
    - The background is black, and the bottom row is a solid ground line of one color.
    - Several gray one-cell-thick hollow rectangles float above the ground; each frame has exactly one contiguous perimeter gap.
    - Separate monochrome bar components are suspended above the scene.

    Rule:
    - All bars fall straight toward the ground line.
    - A falling bar stops when the next downward step would hit a frame, the ground, or an already settled bar.
    - If an unsealed top-side gap is met head-on and the bar's translated shape matches that gap exactly, the bar plugs the gap and the whole rectangle interior is filled with that bar's color.
    - Side-facing or downward gaps, and non-exact alignments above top gaps, do not trigger a fill.

    Key insight:
    - The missing border cell run behaves like a special slot: only a bar that reaches the exact top-gap footprint can replace the border and thereby create a closed region worth flood-filling.
    """
    return _simulate(grid)


def _bar_cells(row: int, c0: int, length: int) -> np.ndarray:
    return np.array([(row, c) for c in range(c0, c0 + length)], dtype=int)


def _interval_conflicts(used: set[tuple[int, int]], row: int, c0: int, length: int) -> bool:
    return any((row, c) in used for c in range(c0, c0 + length))


def _choose_mismatch_interval(frame: dict) -> tuple[int, int]:
    r0, c0, _r1, c1 = frame["bbox"]
    gap = frame["gap_coords"]
    gap_cols = sorted(int(c) for _r, c in gap)
    gap_set = set(gap_cols)
    top_span = list(range(c0, c1 + 1))
    options: list[tuple[int, int]] = []
    for length in range(2, c1 - c0 + 1):
        for start in range(c0, c1 - length + 2):
            cols = list(range(start, start + length))
            if not set(cols) & gap_set:
                continue
            if cols == gap_cols:
                continue
            if any(col in top_span and col not in gap_set for col in cols):
                options.append((start, length))
    if not options:
        raise ValueError("no mismatch interval")
    return random.choice(options)


def _random_frame_spec(h: int, w: int, role: str) -> dict:
    outer_h = random.randint(5, min(8, h - 7))
    outer_w = random.randint(6, min(10, w - 4))
    r0 = random.randint(3, h - outer_h - 4)
    c0 = random.randint(1, w - outer_w - 1)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    if role == "seal" or role == "top_mismatch":
        side = "top"
        gap_len = random.randint(2, outer_w - 3)
        start = random.randint(c0 + 1, c1 - gap_len)
    elif role == "side_gap":
        side = random.choice(GEN_FRAME_SIDE_GAPS)
        gap_len = random.randint(2, outer_h - 3)
        start = random.randint(r0 + 1, r1 - gap_len)
    else:
        side = "bottom"
        gap_len = random.randint(2, outer_w - 3)
        start = random.randint(c0 + 1, c1 - gap_len)

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "gap_start": start,
        "gap_len": gap_len,
        "role": role,
    }


def _frame_padding_cells(spec: dict, pad: int = 1) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    cells = set()
    for r in range(r0 - pad, r1 + pad + 1):
        for c in range(c0 - pad, c1 + pad + 1):
            cells.add((r, c))
    return cells


def _place_frames(h: int, w: int, roles: list[str]) -> list[dict] | None:
    placed: list[dict] = []
    padded_used: set[tuple[int, int]] = set()
    for role in roles:
        for _ in range(120):
            spec = _random_frame_spec(h, w, role)
            if any((r < 0 or r >= h or c < 0 or c >= w) for r, c in _frame_padding_cells(spec, pad=0)):
                continue
            if _frame_padding_cells(spec, pad=2) & padded_used:
                continue
            placed.append(spec)
            padded_used |= _frame_padding_cells(spec, pad=2)
            break
        else:
            return None
    return placed


def _build_static_grid(h: int, w: int, frames: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[-1, :] = GROUND_COLOR
    for spec in frames:
        r0, c0, r1, c1 = spec["bbox"]
        cells = _frame_cells(r0, c0, r1, c1, spec["gap_side"], spec["gap_start"], spec["gap_len"])
        for r, c in cells:
            grid[r, c] = FRAME_COLOR
    return grid


def _build_final_bars(frames: list[dict], h: int, w: int) -> list[dict] | None:
    bars: list[dict] = []
    used: set[tuple[int, int]] = set()
    used_cols_with_rows: set[tuple[int, int]] = set()
    colors = list(BAR_COLORS)
    random.shuffle(colors)
    color_idx = 0

    def next_color() -> int:
        nonlocal color_idx
        color = colors[color_idx % len(colors)]
        color_idx += 1
        return color

    for spec in frames:
        r0, c0, r1, c1 = spec["bbox"]
        role = spec["role"]
        if role == "seal":
            row = r0
            c_start = spec["gap_start"]
            length = spec["gap_len"]
            kind = "seal"
        elif role == "top_mismatch":
            if r0 < 1:
                return None
            row = r0 - 1
            c_start, length = _choose_mismatch_interval(
                {
                    "bbox": spec["bbox"],
                    "gap_coords": np.array(
                        _frame_gap_cells(r0, c0, r1, c1, "top", spec["gap_start"], spec["gap_len"]),
                        dtype=int,
                    ),
                }
            )
            kind = "rest"
        elif role == "side_gap":
            if r0 < 1:
                return None
            row = r0 - 1
            length = random.randint(2, max(2, min(5, c1 - c0 - 1)))
            c_start = random.randint(c0, c1 - length + 1)
            kind = "rest"
        else:
            if r0 < 1:
                return None
            row = r0 - 1
            length = random.randint(2, max(2, min(5, c1 - c0 - 1)))
            c_start = random.randint(c0, c1 - length + 1)
            kind = "rest"

        if c_start < 0 or c_start + length > w:
            return None
        if _interval_conflicts(used, row, c_start, length):
            return None
        for c in range(c_start, c_start + length):
            used.add((row, c))
            used_cols_with_rows.add((row, c))
        bars.append(
            {
                "final_row": row,
                "c0": c_start,
                "length": length,
                "color": next_color(),
                "kind": kind,
                "frame_bbox": spec["bbox"],
            }
        )

    if random.random() < 0.45:
        taken = _build_static_grid(h, w, frames) != BLACK
        for bar in bars:
            taken[bar["final_row"], bar["c0"] : bar["c0"] + bar["length"]] = True
        choices: list[tuple[int, int]] = []
        for length in range(2, 6):
            for c0 in range(0, w - length + 1):
                if np.any(taken[h - 2, c0 : c0 + length]) or np.any(taken[h - 3, c0 : c0 + length]):
                    continue
                choices.append((c0, length))
        if choices:
            c0, length = random.choice(choices)
            bars.append(
                {
                    "final_row": h - 2,
                    "c0": c0,
                    "length": length,
                    "color": next_color(),
                    "kind": "rest",
                    "frame_bbox": None,
                }
            )
            bars.append(
                {
                    "final_row": h - 3,
                    "c0": c0,
                    "length": length,
                    "color": next_color(),
                    "kind": "rest",
                    "frame_bbox": None,
                    "group": "stack",
                }
            )

    return bars


def _build_expected_output(static_grid: np.ndarray, frames: list[dict], bars: list[dict]) -> np.ndarray:
    out = static_grid.copy()
    fill_by_bbox = {
        bar["frame_bbox"]: bar["color"] for bar in bars if bar["kind"] == "seal" and bar["frame_bbox"] is not None
    }
    for spec in frames:
        fill_color = fill_by_bbox.get(spec["bbox"])
        if fill_color is None:
            continue
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    for bar in bars:
        row = bar["final_row"]
        c0 = bar["c0"]
        out[row, c0 : c0 + bar["length"]] = bar["color"]
    return out


def _assign_input_rows(static_grid: np.ndarray, bars: list[dict]) -> list[int] | None:
    order = list(range(len(bars)))
    order.sort(key=lambda i: (bars[i]["c0"], bars[i]["final_row"]))
    occupied = static_grid != BLACK
    input_rows = [-1] * len(bars)
    stack_lift: int | None = None

    for idx in order:
        bar = bars[idx]
        max_lift = min(6, bar["final_row"])
        candidates = list(range(1, max_lift + 1))
        random.shuffle(candidates)

        if bar.get("group") == "stack" and stack_lift is not None:
            candidates = [stack_lift]

        placed = False
        for lift in candidates:
            row = bar["final_row"] - lift
            cols = slice(bar["c0"], bar["c0"] + bar["length"])
            if np.any(occupied[row, cols]):
                continue
            if bar["kind"] == "seal":
                # Keep sealers visibly above the target frame.
                frame_r0 = bar["frame_bbox"][0]
                if row >= frame_r0 - 1:
                    continue
            input_rows[idx] = row
            occupied[row, cols] = True
            if bar.get("group") == "stack" or (
                idx + 1 < len(bars)
                and bars[idx + 1].get("group") == "stack"
                and bars[idx + 1]["c0"] == bar["c0"]
                and bars[idx + 1]["length"] == bar["length"]
            ):
                stack_lift = lift
            placed = True
            break
        if not placed:
            return None
    return input_rows


def _build_input(static_grid: np.ndarray, bars: list[dict], input_rows: list[int]) -> np.ndarray:
    inp = static_grid.copy()
    for bar, row in zip(bars, input_rows):
        inp[row, bar["c0"] : bar["c0"] + bar["length"]] = bar["color"]
    return inp


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(16, 26)
    w = random.randint(14, 26)

    frame_count = random.randint(2, 4)
    roles = ["seal"]
    remaining = frame_count - 1
    if remaining >= 1:
        roles.append(random.choice(GEN_FRAME_ROLES[1:]))
    for _ in range(frame_count - len(roles)):
        roles.append(random.choice(GEN_FRAME_ROLES))
    random.shuffle(roles)

    frames = _place_frames(h, w, roles)
    if frames is None:
        return None

    static_grid = _build_static_grid(h, w, frames)
    bars = _build_final_bars(frames, h, w)
    if bars is None:
        return None

    input_rows = _assign_input_rows(static_grid, bars)
    if input_rows is None:
        return None

    inp = _build_input(static_grid, bars, input_rows)
    expected = _build_expected_output(static_grid, frames, bars)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid falling-gap ARC input")

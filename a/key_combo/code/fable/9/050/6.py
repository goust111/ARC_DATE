import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return [int(v) for v in np.unique(grid) if int(v) != bg]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_solid_square(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side = r1 - r0 + 1
    return (
        side == c1 - c0 + 1
        and side % 2 == 1
        and int(comp["area"]) == side * side
        and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"]))
    )


def _square_box(shape: tuple[int, int], axis: int, anchor_low: bool, side: int, center: int) -> tuple[int, int, int, int]:
    h, w = shape
    half = side // 2
    if axis == 0:
        r0 = 0 if anchor_low else h - side
        c0 = center - half
    else:
        r0 = center - half
        c0 = 0 if anchor_low else w - side
    return int(r0), int(c0), int(r0 + side - 1), int(c0 + side - 1)


def _size_history(side: int, bar_count: int) -> tuple[list[int], int]:
    widths = []
    current = int(side)
    for _ in range(int(bar_count)):
        widths.append(current)
        current = max(1, current - 2)
    return widths, current


def _parse_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert comps, "expected foreground objects"

    square_candidates = [comp for comp in comps if _is_solid_square(g, comp)]
    edge_squares = []
    for comp in square_candidates:
        r0, c0, r1, c1 = comp["bbox"]
        touches = []
        if r0 == 0:
            touches.append(("top", 0, 1))
        if r1 == h - 1:
            touches.append(("bottom", 0, -1))
        if c0 == 0:
            touches.append(("left", 1, 1))
        if c1 == w - 1:
            touches.append(("right", 1, -1))
        if len(touches) == 1:
            name, axis, direction = touches[0]
            edge_squares.append((comp, name, axis, direction))
    assert len(edge_squares) == 1, "expected exactly one odd solid square touching exactly one border"

    square, touch_name, axis, direction = edge_squares[0]
    r0, c0, r1, c1 = square["bbox"]
    side = r1 - r0 + 1
    center = (c0 + c1) // 2 if axis == 0 else (r0 + r1) // 2
    anchor_low = direction == 1
    start_box = _square_box((h, w), axis, anchor_low, side, center)
    assert start_box == square["bbox"], "square must be flush with its touched border"

    bars = []
    for comp in comps:
        if comp is square:
            continue
        br0, bc0, br1, bc1 = comp["bbox"]
        if axis == 0:
            assert br0 == br1 and bc0 == 0 and bc1 == w - 1 and int(comp["area"]) == w, "expected full-width horizontal bars"
            pos = br0
        else:
            assert bc0 == bc1 and br0 == 0 and br1 == h - 1 and int(comp["area"]) == h, "expected full-height vertical bars"
            pos = bc0
        bars.append({"pos": int(pos), "color": int(comp["color"])})

    assert bars, "expected at least one thin bar"
    ordered = sorted(bars, key=lambda item: item["pos"], reverse=(direction < 0))
    positions = [item["pos"] for item in ordered]
    assert len(set(positions)) == len(positions), "bars must occupy distinct lines"
    for a, b in zip(sorted(positions), sorted(positions)[1:]):
        assert b - a >= 2, "bars must stay one cell thick and separated"

    if axis == 0:
        if direction > 0:
            assert all(pos > r1 for pos in positions), "bars must lie beyond the starting square"
        else:
            assert all(pos < r0 for pos in positions), "bars must lie beyond the starting square"
    else:
        if direction > 0:
            assert all(pos > c1 for pos in positions), "bars must lie beyond the starting square"
        else:
            assert all(pos < c0 for pos in positions), "bars must lie beyond the starting square"

    widths, final_side = _size_history(side, len(ordered))
    dest_box = _square_box((h, w), axis, not anchor_low, final_side, center)
    dr0, dc0, dr1, dc1 = dest_box
    if axis == 0:
        assert all(not (dr0 <= pos <= dr1) for pos in positions), "bars may not overlap the final square band"
    else:
        assert all(not (dc0 <= pos <= dc1) for pos in positions), "bars may not overlap the final square band"

    return {
        "grid": g,
        "bg": bg,
        "shape": (h, w),
        "axis": axis,
        "direction": direction,
        "touch": touch_name,
        "center": center,
        "square_color": int(square["color"]),
        "square_bbox": tuple(map(int, square["bbox"])),
        "side": side,
        "bars": ordered,
        "gap_widths": widths,
        "final_side": final_side,
        "dest_bbox": dest_box,
    }


def _apply_rule(spec: dict) -> np.ndarray:
    g = spec["grid"]
    bg = int(spec["bg"])
    axis = int(spec["axis"])
    center = int(spec["center"])
    out = g.copy()

    sr0, sc0, sr1, sc1 = spec["square_bbox"]
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

    for bar, width in zip(spec["bars"], spec["gap_widths"]):
        pos = int(bar["pos"])
        half = width // 2
        lo = center - half
        hi = center + half
        if axis == 0:
            out[pos, lo : hi + 1] = bg
        else:
            out[lo : hi + 1, pos] = bg

    dr0, dc0, dr1, dc1 = spec["dest_bbox"]
    assert np.all(out[dr0 : dr1 + 1, dc0 : dc1 + 1] == bg), "final square destination must be clear"
    out[dr0 : dr1 + 1, dc0 : dc1 + 1] = int(spec["square_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the edge-anchored solid square through the aligned thin bars, carving matching gaps and shrinking after each pass.

    Input structure:
    - The background is a single dominant color.
    - Exactly one monochrome odd-sized solid square touches exactly one grid border.
    - Several one-cell-thick full bars lie farther along the square's center row or center column, perpendicular to the motion axis.

    Rule:
    - Move the square straight away from its touched border until it becomes flush with the opposite border.
    - Each time the square passes a bar, erase from that bar a centered gap whose width equals the square's current side length.
    - After opening that gap, shrink the square by removing one outer layer on every side, reducing the side length by two; once the side length reaches one, it stays one.
    - Clear the starting square location, keep the bars in place with their carved gaps, and place the final square on the opposite border on the same center line.

    Key insight:
    - The ordered bars are a time axis: their gap widths must follow the square's size history as it marches across the grid.
    """
    return _apply_rule(_parse_puzzle(np.asarray(grid, dtype=int)))


def _render_input(spec: dict) -> np.ndarray:
    h, w = spec["shape"]
    bg = int(spec["bg"])
    grid = np.full((h, w), bg, dtype=int)
    bar_color = int(spec["bar_color"])
    if spec["axis"] == 0:
        for pos in spec["bars_all"]:
            grid[int(pos), :] = bar_color
    else:
        for pos in spec["bars_all"]:
            grid[:, int(pos)] = bar_color

    sr0, sc0, sr1, sc1 = spec["square_bbox"]
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = int(spec["square_color"])
    return grid


def _oracle_output(spec: dict) -> np.ndarray:
    inp = _render_input(spec)
    out = inp.copy()
    sr0, sc0, sr1, sc1 = spec["square_bbox"]
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = spec["bg"]
    for pos, width in zip(spec["bars_in_order"], spec["gap_widths"]):
        half = width // 2
        lo = spec["center"] - half
        hi = spec["center"] + half
        if spec["axis"] == 0:
            out[int(pos), lo : hi + 1] = spec["bg"]
        else:
            out[lo : hi + 1, int(pos)] = spec["bg"]
    dr0, dc0, dr1, dc1 = spec["dest_bbox"]
    out[dr0 : dr1 + 1, dc0 : dc1 + 1] = spec["square_color"]
    return out


def _sample_bar_count(side: int) -> int:
    threshold = (side - 1) // 2
    regime = random.choice(["before", "reach", "after"])
    if threshold == 1:
        if regime == "reach":
            return 1
        return random.randint(2, 4)
    if regime == "before":
        return random.randint(1, threshold - 1)
    if regime == "reach":
        return threshold
    return random.randint(threshold + 1, threshold + 3)


def _build_spec() -> dict:
    bg = 0
    axis = random.randint(0, 1)
    direction = random.choice((-1, 1))
    square_color, bar_color = random.sample(range(1, 10), 2)

    for _ in range(200):
        side = random.choice((3, 5, 7, 9))
        bar_count = _sample_bar_count(side)
        gap_widths, final_side = _size_history(side, bar_count)

        front_gap = random.randint(1, 3)
        between_gaps = [random.randint(1, 3) for _ in range(bar_count - 1)]
        back_gap = random.randint(1, 3)
        along = side + front_gap + bar_count + sum(between_gaps) + back_gap + final_side
        if along > 30:
            continue

        cross_min = max(side + 2, gap_widths[0] + 2)
        cross = random.randint(cross_min, 30)
        half = side // 2
        center = random.randint(half + 1, cross - half - 2)

        pos = side + front_gap
        bars_canonical = [pos]
        for gap in between_gaps:
            pos += 1 + gap
            bars_canonical.append(pos)

        if direction == 1:
            bars_in_order = bars_canonical
        else:
            bars_in_order = [along - 1 - pos for pos in bars_canonical]
        bars_all = sorted(bars_in_order)

        shape = (along, cross) if axis == 0 else (cross, along)
        anchor_low = direction == 1
        square_bbox = _square_box(shape, axis, anchor_low, side, center)
        dest_bbox = _square_box(shape, axis, not anchor_low, final_side, center)

        spec = {
            "bg": bg,
            "shape": shape,
            "axis": axis,
            "direction": direction,
            "center": center,
            "side": side,
            "final_side": final_side,
            "gap_widths": gap_widths,
            "square_color": square_color,
            "bar_color": bar_color,
            "bars_in_order": [int(x) for x in bars_in_order],
            "bars_all": [int(x) for x in bars_all],
            "square_bbox": square_bbox,
            "dest_bbox": dest_bbox,
        }

        inp = _render_input(spec)
        out = _oracle_output(spec)
        _validate_pair(spec, inp, out)
        return spec
    raise RuntimeError("failed to sample a valid puzzle specification")


def _validate_pair(spec: dict, inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert not np.array_equal(inp, out), "input and output must differ"

    parsed = _parse_puzzle(inp)
    assert parsed["axis"] == spec["axis"]
    assert parsed["direction"] == spec["direction"]
    assert parsed["center"] == spec["center"]
    assert parsed["side"] == spec["side"]
    assert parsed["final_side"] == spec["final_side"]
    assert [bar["pos"] for bar in parsed["bars"]] == spec["bars_in_order"]
    assert parsed["gap_widths"] == spec["gap_widths"]

    solved = main(inp)
    assert np.array_equal(solved, out), "solver and oracle disagree"

    bg = spec["bg"]
    dr0, dc0, dr1, dc1 = spec["dest_bbox"]
    square = out[dr0 : dr1 + 1, dc0 : dc1 + 1]
    assert square.shape == (spec["final_side"], spec["final_side"])
    assert np.all(square == spec["square_color"])

    sr0, sc0, sr1, sc1 = spec["square_bbox"]
    assert np.all(out[sr0 : sr1 + 1, sc0 : sc1 + 1] == bg), "starting square must be cleared"

    for pos, width in zip(spec["bars_in_order"], spec["gap_widths"]):
        half = width // 2
        lo = spec["center"] - half
        hi = spec["center"] + half
        if spec["axis"] == 0:
            line = out[int(pos), :]
            assert np.all(line[:lo] == spec["bar_color"])
            assert np.all(line[lo : hi + 1] == bg)
            assert np.all(line[hi + 1 :] == spec["bar_color"])
        else:
            line = out[:, int(pos)]
            assert np.all(line[:lo] == spec["bar_color"])
            assert np.all(line[lo : hi + 1] == bg)
            assert np.all(line[hi + 1 :] == spec["bar_color"])

    if len(spec["gap_widths"]) >= 2:
        for prev, nxt in zip(spec["gap_widths"], spec["gap_widths"][1:]):
            assert nxt == max(1, prev - 2), "gap widths must record the shrink history"


def generate_input() -> np.ndarray:
    for _ in range(300):
        spec = _build_spec()
        inp = _render_input(spec)
        out = _oracle_output(spec)
        try:
            _validate_pair(spec, inp, out)
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")

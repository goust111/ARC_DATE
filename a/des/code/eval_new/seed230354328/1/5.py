from __future__ import annotations

import numpy as np


BG = 0
OBSTACLE = 5
BASELINE = 9
GATE_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_MIN = 18
GRID_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
SEGMENT_LENGTH_RANGE = (2, 4)
CHANNEL_LENGTH_RANGE = (2, 8)
OBSTACLE_COUNT_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _line_info(coords: np.ndarray) -> dict[str, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    area = int(coords.shape[0])
    if c0 == c1 and area == int(r1 - r0 + 1):
        return {
            "orientation": "vertical",
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r1),
            "c1": int(c1),
            "length": area,
        }
    if r0 == r1 and area == int(c1 - c0 + 1):
        return {
            "orientation": "horizontal",
            "r0": int(r0),
            "c0": int(c0),
            "r1": int(r1),
            "c1": int(c1),
            "length": area,
        }
    return None


def _extract_gates(grid: np.ndarray) -> tuple[list[dict[str, int]], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2

    baseline_color = int(g[h - 1, 0])
    assert baseline_color != BG
    assert np.all(g[h - 1, :] == baseline_color), "bottom row must be the output baseline"

    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, OBSTACLE, baseline_color)]
    gates: list[dict[str, int]] = []
    for color in colors:
        comps = _components(g == int(color))
        assert len(comps) == 2, "each gate color must form exactly two segments"
        parts = [_line_info(comp) for comp in comps]
        assert parts[0] is not None and parts[1] is not None, "gate segments must be straight lines"
        a = parts[0]
        b = parts[1]
        assert a is not None and b is not None
        assert a["orientation"] == b["orientation"], "same-color gate segments must be parallel"
        assert a["length"] == b["length"], "same-color gate segments must have equal length"
        assert SEGMENT_LENGTH_RANGE[0] <= a["length"] <= SEGMENT_LENGTH_RANGE[1]

        if a["orientation"] == "vertical":
            assert a["r0"] == b["r0"] and a["r1"] == b["r1"], "vertical gate segments must face each other"
            left, right = sorted((a, b), key=lambda part: part["c0"])
            channel_len = int(right["c0"] - left["c0"] - 1)
            assert channel_len > 0
            channel = g[left["r0"] : left["r1"] + 1, left["c0"] + 1 : right["c0"]]
            start_col = int(left["c0"])
            first_row = int(left["r0"])
        else:
            assert a["c0"] == b["c0"] and a["c1"] == b["c1"], "horizontal gate segments must face each other"
            top, bottom = sorted((a, b), key=lambda part: part["r0"])
            channel_len = int(bottom["r0"] - top["r0"] - 1)
            assert channel_len > 0
            channel = g[top["r0"] + 1 : bottom["r0"], top["c0"] : top["c1"] + 1]
            start_col = int(top["c0"])
            first_row = int(top["r0"])

        obstacle_count = int(np.count_nonzero(channel == OBSTACLE))
        assert OBSTACLE_COUNT_RANGE[0] <= obstacle_count <= OBSTACLE_COUNT_RANGE[1]
        gates.append(
            {
                "color": int(color),
                "first_row": first_row,
                "start_col": start_col,
                "channel_len": channel_len,
                "band_height": obstacle_count + 1,
            }
        )

    assert gates, "expected at least one gate pair"
    gates.sort(key=lambda gate: (gate["first_row"], gate["start_col"]))
    return gates, baseline_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert paired gate segments into stacked horizontal bars whose size is encoded by the channel between them.

    Input structure:
    - The background is black, with a full-width maroon baseline on the bottom row.
    - Each non-baseline gate color appears as exactly two equal short line segments, either vertical or horizontal, facing each other across a rectangular channel.
    - Grey pixels may appear inside a channel as obstacles; different gate pairs use different colors.

    Rule:
    - For every gate color, identify the two matching segments and determine whether the channel runs horizontally or vertically.
    - The output bar for that color is horizontal: its length equals the gap between the two gate segments, and its height equals one plus the number of grey obstacle pixels inside that channel.
    - Stack these bars above the baseline, ordering them by the first row where each gate color appears; the highest gate in the input becomes the lowest bar, and each bar starts at the left segment's column or the upper segment's left column.

    Key insight:
    - The visible gate color only selects the bar color; the bar dimensions come from two different hidden cues in the channel itself: gap size for length and grey obstacle count for height.
    """
    g = np.asarray(grid, dtype=int)
    gates, baseline_color = _extract_gates(g)
    h, w = g.shape

    out = np.full((h, w), BG, dtype=int)
    out[h - 1, :] = baseline_color

    cursor = h - 2
    for gate in gates:
        top = cursor - gate["band_height"] + 1
        assert top >= 0, "stacked bars must fit above the baseline"
        c0 = gate["start_col"]
        c1 = c0 + gate["channel_len"]
        assert 0 <= c0 < c1 <= w
        out[top : cursor + 1, c0:c1] = gate["color"]
        cursor = top - 1

    return out


def _padded_bbox_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    return not bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _reserve_padded_bbox(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = occupied.shape
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _spec_bbox(spec: dict[str, int], r0: int, c0: int) -> tuple[int, int, int, int]:
    if spec["orientation"] == "vertical":
        return (r0, c0, r0 + spec["segment_len"] - 1, c0 + spec["channel_len"] + 1)
    return (r0, c0, r0 + spec["channel_len"] + 1, c0 + spec["segment_len"] - 1)


def _paint_gate(grid: np.ndarray, spec: dict[str, int], color: int, r0: int, c0: int, rng: np.random.Generator) -> None:
    if spec["orientation"] == "vertical":
        segment_len = spec["segment_len"]
        gap = spec["channel_len"]
        grid[r0 : r0 + segment_len, c0] = int(color)
        grid[r0 : r0 + segment_len, c0 + gap + 1] = int(color)
        channel_cells = [(r, c) for r in range(r0, r0 + segment_len) for c in range(c0 + 1, c0 + gap + 1)]
    else:
        segment_len = spec["segment_len"]
        gap = spec["channel_len"]
        grid[r0, c0 : c0 + segment_len] = int(color)
        grid[r0 + gap + 1, c0 : c0 + segment_len] = int(color)
        channel_cells = [(r, c) for r in range(r0 + 1, r0 + gap + 1) for c in range(c0, c0 + segment_len)]

    if spec["obstacles"] > 0:
        pick = rng.choice(len(channel_cells), size=spec["obstacles"], replace=False)
        for idx in np.atleast_1d(pick):
            rr, cc = channel_cells[int(idx)]
            grid[rr, cc] = OBSTACLE


def _expected_output(shape: tuple[int, int], baseline_color: int, placed: list[dict[str, int]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[h - 1, :] = int(baseline_color)
    cursor = h - 2
    for gate in sorted(placed, key=lambda gate: (gate["first_row"], gate["start_col"])):
        top = cursor - gate["band_height"] + 1
        if top < 0:
            raise AssertionError("expected bars do not fit")
        out[top : cursor + 1, gate["start_col"] : gate["start_col"] + gate["channel_len"]] = gate["color"]
        cursor = top - 1
    return out


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))

    specs: list[dict[str, int]] = []
    total_band_height = 0
    for _ in range(pair_count):
        orientation = "vertical" if int(rng.integers(0, 2)) == 0 else "horizontal"
        spec = {
            "orientation": orientation,
            "segment_len": int(rng.integers(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1)),
            "channel_len": int(rng.integers(CHANNEL_LENGTH_RANGE[0], CHANNEL_LENGTH_RANGE[1] + 1)),
            "obstacles": int(rng.integers(OBSTACLE_COUNT_RANGE[0], OBSTACLE_COUNT_RANGE[1] + 1)),
        }
        total_band_height += spec["obstacles"] + 1
        specs.append(spec)

    min_h = max(
        GRID_MIN,
        total_band_height + 2,
        max((spec["segment_len"] if spec["orientation"] == "vertical" else spec["channel_len"] + 2) for spec in specs) + 2,
    )
    min_w = max(
        GRID_MIN,
        max((spec["channel_len"] + 2 if spec["orientation"] == "vertical" else spec["segment_len"]) for spec in specs) + 2,
    )
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = int(rng.integers(min_h, GRID_MAX + 1))
    w = int(rng.integers(min_w, GRID_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = BASELINE

    colors = [int(color) for color in rng.choice(np.asarray(GATE_COLORS, dtype=int), size=pair_count, replace=False)]
    order = sorted(
        range(pair_count),
        key=lambda idx: -(
            (specs[idx]["segment_len"] * (specs[idx]["channel_len"] + 2))
            if specs[idx]["orientation"] == "vertical"
            else ((specs[idx]["channel_len"] + 2) * specs[idx]["segment_len"])
        ),
    )

    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 1, :] = True
    used_first_rows: set[int] = set()
    placements: list[tuple[int, int] | None] = [None] * pair_count

    for idx in order:
        spec = specs[idx]
        if spec["orientation"] == "vertical":
            box_h = spec["segment_len"]
            box_w = spec["channel_len"] + 2
            max_c0 = w - box_w
        else:
            box_h = spec["channel_len"] + 2
            box_w = spec["segment_len"]
            max_c0 = w - max(box_w, spec["channel_len"])

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(0, h - 1 - box_h + 1))
            c0 = int(rng.integers(0, max_c0 + 1))
            if r0 in used_first_rows:
                continue
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            if not _padded_bbox_free(occupied, r0, c0, r1, c1):
                continue
            placements[idx] = (r0, c0)
            _reserve_padded_bbox(occupied, r0, c0, r1, c1)
            used_first_rows.add(r0)
            placed = True
            break
        if not placed:
            return None

    placed_gates: list[dict[str, int]] = []
    for idx, (spec, color) in enumerate(zip(specs, colors)):
        pos = placements[idx]
        if pos is None:
            return None
        r0, c0 = pos
        _paint_gate(grid, spec, int(color), r0, c0, rng)
        placed_gates.append(
            {
                "color": int(color),
                "first_row": int(r0),
                "start_col": int(c0),
                "channel_len": int(spec["channel_len"]),
                "band_height": int(spec["obstacles"] + 1),
            }
        )

    expected = _expected_output(grid.shape, BASELINE, placed_gates)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")

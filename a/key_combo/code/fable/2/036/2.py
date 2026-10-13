from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_fable_2_036_2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
colors_present = _ARC.colors_present
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste


BG = Color.BLACK
GROUND = Color.GREY
MARKER = Color.RED
PILLAR_COLORS = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
GEN_MODES = ("pit_stop", "height_cut", "multi_relay")
GEN_RETRY_LIMIT = 400
_GEN_CALL_COUNT = 0


def _find_ground_row(grid: np.ndarray) -> int:
    grey_counts = np.sum(grid == GROUND, axis=1)
    rows = np.flatnonzero(grey_counts > 0)
    assert rows.size == 1, "expected exactly one grey ground row"
    ground_row = int(rows[0])
    allowed = {BG, GROUND}
    assert set(map(int, np.unique(grid[ground_row]))) <= allowed, "ground row may contain only grey and black"
    return ground_row


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (1, 2), "expected exactly one red push marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _pillar_components(grid: np.ndarray, ground_row: int) -> list[dict]:
    pillar_colors = colors_present(grid, exclude={BG, GROUND, MARKER})
    pillars = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=pillar_colors,
        return_format="dict",
    )
    assert pillars, "expected at least one pillar"

    parsed: list[dict] = []
    for comp in pillars:
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        assert bbox_w(comp["bbox"]) == 1, "pillars must be one cell wide"
        assert bbox_h(comp["bbox"]) == int(comp["area"]), "pillars must be solid vertical bars"
        assert r1 == ground_row - 1, "pillar feet must sit on the ground"
        assert np.array_equal(np.sort(coords[:, 0]), np.arange(r0, r1 + 1)), "pillar rows must be contiguous"
        assert np.all(coords[:, 1] == c0), "pillar must occupy a single column"
        parsed.append(
            {
                "color": int(comp["color"]),
                "col": int(c0),
                "height": int(comp["area"]),
                "top": int(r0),
                "coords": coords,
            }
        )
    parsed.sort(key=lambda pillar: pillar["col"])
    return parsed


def _topple_chain(
    base_grid: np.ndarray,
    ground_row: int,
    pits: set[int],
    pillars: list[dict],
    marker: tuple[int, int],
) -> np.ndarray:
    out = np.asarray(base_grid, dtype=int).copy()
    surface_row = ground_row - 1
    standing = {pillar["col"]: idx for idx, pillar in enumerate(pillars)}
    current_idx = 0

    while True:
        pillar = pillars[current_idx]
        standing.pop(pillar["col"], None)
        for r, c in pillar["coords"]:
            out[int(r), int(c)] = BG

        remaining = int(pillar["height"])
        col = int(pillar["col"])
        color = int(pillar["color"])
        out[surface_row, col] = color
        remaining -= 1

        next_idx = None
        cursor_col = col
        while remaining > 0:
            target_col = cursor_col + 1
            if target_col >= out.shape[1]:
                break
            if target_col in standing:
                next_idx = standing[target_col]
                break

            if target_col in pits:
                if remaining < 2:
                    break
                out[surface_row, target_col] = color
                out[ground_row, target_col] = color
                remaining -= 2
            else:
                out[surface_row, target_col] = color
                remaining -= 1
            cursor_col = target_col

        if next_idx is None:
            break
        current_idx = next_idx

    out[marker] = MARKER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Push the leftmost pillar rightward so each toppled pillar relays into the next unless pits consume too much length first.

    Input structure:
    - The background is black, with exactly one grey ground row near the bottom and some black pit cells carved into it.
    - Several separated one-cell-wide monochromatic pillars stand on intact ground columns above that row.
    - A single red push marker sits immediately left of the leftmost pillar on one of its body cells.

    Rule:
    - Clear the pushed pillar, then rotate its height into a rightward laid shape starting from its foot cell on the surface row.
    - Each normal ground column contributes one surface cell; each pit column contributes both a surface cell and a dipped cell inside the pit, so every pit consumes one extra unit of the pillar's length.
    - If the laid shape reaches the foot column of the next upright pillar before length runs out, stop there and continue the same process from that pillar.
    - If the current pillar runs out of length first, the relay ends and all farther pillars stay upright; the red marker is preserved.

    Key insight:
    - The decisive quantity is not raw height but height minus pit penalties along each gap, because pit columns visibly eat an extra cell of the toppled pillar before the chain can reach the next pillar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG, "expected black to be the dominant background"

    ground_row = _find_ground_row(g)
    pits = {int(c) for c in np.flatnonzero(g[ground_row] == BG)}
    marker = _find_marker(g)
    pillars = _pillar_components(g, ground_row)

    leftmost = pillars[0]
    assert marker[1] == leftmost["col"] - 1, "marker must be attached to the leftmost pillar"
    assert leftmost["top"] <= marker[0] <= ground_row - 1, "marker must sit beside the leftmost pillar body"
    assert all(pillar["col"] not in pits for pillar in pillars), "pillars cannot stand above pits"

    return _topple_chain(g, ground_row, pits, pillars, marker)


def _make_gap_pits(left_col: int, right_col: int, count: int) -> set[int]:
    available = list(range(left_col + 1, right_col))
    assert 0 <= count <= len(available)
    if count == 0:
        return set()
    start = random.randint(0, len(available) - count)
    return set(available[start : start + count])


def _make_layout(mode: str) -> dict:
    if mode == "pit_stop":
        n_pillars = random.randint(3, 5)
        gaps = [random.randint(3, 6)] + [random.randint(2, 5) for _ in range(n_pillars - 2)]
        pit_counts = [random.randint(1, min(3, gaps[0] - 1))] + [
            random.randint(0, min(2, gap - 1)) for gap in gaps[1:]
        ]
        heights = [gaps[0] + pit_counts[0]]
        expected_toppled = 1
        pit_sensitive_segment = 0
    elif mode == "height_cut":
        n_pillars = random.randint(3, 5)
        gaps = [random.randint(2, 5) for _ in range(n_pillars - 1)]
        pit_counts = [random.randint(0, min(2, gap - 1)) for gap in gaps]
        heights = [gaps[0] + pit_counts[0] + random.randint(1, 3), 1]
        expected_toppled = 2
        pit_sensitive_segment = None
    elif mode == "multi_relay":
        n_pillars = random.randint(4, 5)
        gaps = [random.randint(2, 4), random.randint(2, 4), random.randint(3, 6)]
        gaps.extend(random.randint(2, 5) for _ in range(n_pillars - 4))
        pit_counts = [
            random.randint(0, min(2, gaps[0] - 1)),
            random.randint(0, min(2, gaps[1] - 1)),
            random.randint(1, min(3, gaps[2] - 1)),
        ]
        pit_counts.extend(random.randint(0, min(2, gap - 1)) for gap in gaps[3:])
        heights = [
            gaps[0] + pit_counts[0] + random.randint(1, 2),
            gaps[1] + pit_counts[1] + random.randint(1, 2),
            gaps[2] + pit_counts[2],
        ]
        expected_toppled = 3
        pit_sensitive_segment = 2
    else:
        raise ValueError(f"unknown generation mode: {mode}")

    first_col = random.randint(2, 4)
    cols = [first_col]
    for gap in gaps:
        cols.append(cols[-1] + gap)

    max_required_height = max(heights)
    ground_min = max(7, max_required_height + 1)
    ground_max = min(16, max_required_height + 6)
    ground_row = random.randint(ground_min, ground_max)

    for _ in range(n_pillars - len(heights)):
        heights.append(random.randint(2, min(7, ground_row - 1)))

    colors = random.sample(PILLAR_COLORS, k=n_pillars)
    pits: set[int] = set()
    for idx, count in enumerate(pit_counts):
        pits |= _make_gap_pits(cols[idx], cols[idx + 1], count)
    if not pits:
        stop_segment = int(expected_toppled) - 1
        pits |= _make_gap_pits(cols[stop_segment], cols[stop_segment + 1], 1)

    last_col = cols[-1]
    width = min(30, last_col + random.randint(3, 5))
    assert width > last_col

    open_cols = [c for c in range(width) if c not in cols]
    if random.random() < 0.5 and open_cols:
        candidates = [c for c in open_cols if c < cols[0] - 1 or c > cols[-1]]
        if candidates:
            extra_width = random.randint(1, min(2, len(candidates)))
            candidates.sort()
            for start in range(len(candidates) - extra_width + 1):
                chunk = candidates[start : start + extra_width]
                if chunk[-1] - chunk[0] + 1 == extra_width:
                    pits |= set(chunk)
                    break

    height = min(30, ground_row + 1 + random.randint(1, 3))
    marker_row = ground_row - random.randint(1, heights[0])

    return {
        "shape": (height, width),
        "ground_row": ground_row,
        "pits": pits,
        "pillars": [
            {"col": col, "height": pillar_height, "color": color}
            for col, pillar_height, color in zip(cols, heights, colors)
        ],
        "marker_row": marker_row,
        "mode": mode,
        "expected_toppled": expected_toppled,
        "pit_sensitive_segment": pit_sensitive_segment,
    }


def _render_input(layout: dict) -> np.ndarray:
    height, width = layout["shape"]
    ground_row = int(layout["ground_row"])
    grid = np.zeros((height, width), dtype=int)
    grid[ground_row, :] = GROUND
    if layout["pits"]:
        pit_cols = np.array(sorted(layout["pits"]), dtype=int)
        grid[ground_row, pit_cols] = BG

    for pillar in layout["pillars"]:
        top = ground_row - int(pillar["height"])
        sprite = np.full((int(pillar["height"]), 1), int(pillar["color"]), dtype=int)
        paste(grid, sprite, top, int(pillar["col"]))

    first_col = int(layout["pillars"][0]["col"])
    grid[int(layout["marker_row"]), first_col - 1] = MARKER
    return grid


def _oracle_pillars(layout: dict) -> list[dict]:
    ground_row = int(layout["ground_row"])
    pillars = []
    for pillar in layout["pillars"]:
        top = ground_row - int(pillar["height"])
        coords = np.array([(r, int(pillar["col"])) for r in range(top, ground_row)], dtype=int)
        pillars.append(
            {
                "color": int(pillar["color"]),
                "col": int(pillar["col"]),
                "height": int(pillar["height"]),
                "top": int(top),
                "coords": coords,
            }
        )
    return pillars


def _oracle_output(layout: dict, grid: np.ndarray) -> np.ndarray:
    marker = (int(layout["marker_row"]), int(layout["pillars"][0]["col"]) - 1)
    pillars = _oracle_pillars(layout)
    return _topple_chain(grid, int(layout["ground_row"]), set(layout["pits"]), pillars, marker)


def _segment_cost(layout: dict, idx: int, *, include_pits: bool) -> int:
    left = int(layout["pillars"][idx]["col"])
    right = int(layout["pillars"][idx + 1]["col"])
    gap = right - left
    if not include_pits:
        return gap
    pit_penalty = sum(1 for c in range(left + 1, right) if c in layout["pits"])
    return gap + pit_penalty


def _abstract_chain_length(layout: dict, *, include_pits: bool) -> int:
    toppled = 1
    for idx in range(len(layout["pillars"]) - 1):
        height = int(layout["pillars"][idx]["height"])
        if height > _segment_cost(layout, idx, include_pits=include_pits):
            toppled += 1
        else:
            break
    return toppled


def _validate_layout(layout: dict, grid: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and isinstance(out, np.ndarray)
    assert grid.shape == tuple(layout["shape"])
    assert out.shape == grid.shape
    assert max(grid.shape) <= 30
    assert len(layout["pillars"]) > layout["expected_toppled"], "need untouched pillars after the chain stops"
    assert _abstract_chain_length(layout, include_pits=True) == int(layout["expected_toppled"])

    pit_sensitive_segment = layout["pit_sensitive_segment"]
    if pit_sensitive_segment is not None:
        idx = int(pit_sensitive_segment)
        height = int(layout["pillars"][idx]["height"])
        assert height > _segment_cost(layout, idx, include_pits=False), "without pits the chain would not differ"
        assert height <= _segment_cost(layout, idx, include_pits=True), "with pits the relay must stop here"
        assert _abstract_chain_length(layout, include_pits=False) > _abstract_chain_length(layout, include_pits=True)

    if layout["mode"] == "height_cut":
        assert int(layout["pillars"][1]["height"]) == 1

    ground_row = int(layout["ground_row"])
    first_col = int(layout["pillars"][0]["col"])
    assert grid[int(layout["marker_row"]), first_col - 1] == MARKER
    assert all(int(pillar["col"]) not in layout["pits"] for pillar in layout["pillars"])
    assert np.any(grid[ground_row] == BG), "need at least one visible pit"
    assert not np.array_equal(grid, out), "transformation must change the grid"


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    mode = GEN_MODES[_GEN_CALL_COUNT % len(GEN_MODES)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_RETRY_LIMIT):
        layout = _make_layout(mode)
        grid = _render_input(layout)
        expected = _oracle_output(layout, grid)
        actual = main(grid)
        _validate_layout(layout, grid, expected)
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError(f"failed to generate a valid input for mode {mode}")

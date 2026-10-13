from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
BASELINE = 7
FRAME_COLORS = (1, 2, 3, 4, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _distance_to_border(cell: tuple[int, int], border: Iterable[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - br) + abs(c - bc) for br, bc in border)


def _expanded_ring_distance(
    cell: tuple[int, int], bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r, c = cell
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr, dc


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, GREY, BASELINE)):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "frame color must appear"
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "frame outer size must be at least 5x5"
        border = _border_cells((r0, c0, r1, c1))
        assert len(border) == len(set(border))
        assert coords.shape[0] == len(border), "frame color must occupy exactly the perimeter"
        assert all(int(grid[r, c]) == color for r, c in border)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.size > 0 and np.all(inner == BG), "frame interior must be black"
        frames.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "border": border,
                "inner_h": h - 2,
                "inner_w": w - 2,
                "center_r2": r0 + r1,
            }
        )
    assert 2 <= len(frames) <= 3, "expected 2 or 3 frames"
    return sorted(frames, key=lambda frame: frame["bbox"])


def _associate_markers(grid: np.ndarray, frames: list[dict]) -> dict[int, int]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert markers, "expected grey markers"
    counts = {frame["color"]: 0 for frame in frames}
    for marker in markers:
        distances = [
            (frame["color"], _distance_to_border(marker, frame["border"])) for frame in frames
        ]
        best = min(dist for _color, dist in distances)
        owners = [color for color, dist in distances if dist == best]
        assert len(owners) == 1, "grey marker must have a unique nearest frame"
        counts[owners[0]] += 1
    for frame in frames:
        n = counts[frame["color"]]
        assert 1 <= n <= min(
            frame["inner_h"], 4
        ), "marker count must fit within the frame interior height and the 1..4 marker range"
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each frame's nearby grey markers, then partially fill it from the baseline-facing side.

    Input structure:
    - The grid has a black background and one full-width orange horizontal baseline.
    - Two or three differently colored hollow rectangular frames lie entirely above or below that baseline.
    - Grey single-pixel markers are scattered near the frames, with each marker uniquely closest
      in Manhattan distance to one frame border.

    Rule:
    - For each frame, assign every grey marker to the frame whose border is nearest in Manhattan distance,
      then let N be the number of markers assigned to that frame.
    - If the frame center is above the baseline, fill the frame interior from its bottom interior row upward
      for N rows; if the frame center is below the baseline, fill from its top interior row downward for N rows.
    - Fill those rows across the full interior width using the frame's border color, then erase all grey markers
      and erase the orange baseline to black while leaving frame borders and all other background unchanged.

    Key insight:
    - The grey markers only provide the fill height, while the baseline only provides the fill direction;
      both cues are required because the same count produces different outputs above versus below the baseline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    baseline_rows = np.where(np.all(g == BASELINE, axis=1))[0]
    assert baseline_rows.size == 1, "expected exactly one full-width orange baseline"
    baseline_r = int(baseline_rows[0])
    assert int(np.count_nonzero(g == BASELINE)) == w

    frames = _parse_frames(g)
    counts = _associate_markers(g, frames)

    out = g.copy()
    out[g == GREY] = BG
    out[baseline_r, :] = BG

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        n = counts[frame["color"]]
        if frame["center_r2"] < 2 * baseline_r:
            row_start = r1 - n
            row_stop = r1
        else:
            row_start = r0 + 1
            row_stop = r0 + 1 + n
        out[row_start:row_stop, c0 + 1 : c1] = frame["color"]

    return out


def _random_split(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _column_positions(widths: list[int], grid_w: int, min_gap: int = 2) -> list[int]:
    required = sum(widths) + min_gap * max(0, len(widths) - 1)
    assert required <= grid_w
    extra = grid_w - required
    slack = _random_split(extra, len(widths) + 1)
    positions = []
    c = slack[0]
    for idx, width in enumerate(widths):
        positions.append(c)
        c += width
        if idx + 1 < len(widths):
            c += min_gap + slack[idx + 1]
    return positions


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_expected(grid: np.ndarray, frame_specs: list[dict], baseline_r: int) -> np.ndarray:
    out = grid.copy()
    out[out == GREY] = BG
    out[baseline_r, :] = BG
    for frame in frame_specs:
        r0, c0, r1, c1 = frame["bbox"]
        n = frame["count"]
        color = frame["color"]
        if frame["side"] == "top":
            out[r1 - n : r1, c0 + 1 : c1] = color
        else:
            out[r0 + 1 : r0 + 1 + n, c0 + 1 : c1] = color
    return out


def _marker_candidates(
    shape: tuple[int, int],
    frame: dict,
    frames: list[dict],
    baseline_r: int,
    blocked: set[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = frame["bbox"]
    options: list[tuple[int, int]] = []
    for r in range(max(0, r0 - 2), min(h - 1, r1 + 2) + 1):
        for c in range(max(0, c0 - 2), min(w - 1, c1 + 2) + 1):
            cell = (r, c)
            if cell in blocked or cell in chosen or r == baseline_r:
                continue
            dr, dc = _expanded_ring_distance(cell, frame["bbox"])
            cheb = max(dr, dc)
            if cheb not in (1, 2):
                continue
            d_self = _distance_to_border(cell, frame["border"])
            if d_self < 2:
                continue
            distances = [_distance_to_border(cell, other["border"]) for other in frames]
            best = min(distances)
            if distances.count(best) != 1 or distances[frames.index(frame)] != best:
                continue
            if any(abs(r - mr) + abs(c - mc) == 1 for mr, mc in chosen):
                continue
            options.append(cell)
    random.shuffle(options)
    return options


def _choose_counts(frame_specs: list[dict]) -> None:
    top_frames = [frame for frame in frame_specs if frame["side"] == "top"]
    bottom_frames = [frame for frame in frame_specs if frame["side"] == "bottom"]

    for frame in frame_specs:
        frame["count"] = random.randint(1, min(frame["inner_h"] - 1, 4))

    full_candidates = [
        frame
        for frame in frame_specs
        if frame["inner_h"] <= 4
        and (
            (frame["side"] == "top" and len(top_frames) >= 2)
            or (frame["side"] == "bottom" and len(bottom_frames) >= 2)
        )
    ]
    if full_candidates and random.random() < 0.45:
        full_frame = random.choice(full_candidates)
        full_frame["count"] = full_frame["inner_h"]

    assert any(frame["count"] < frame["inner_h"] for frame in top_frames)
    assert any(frame["count"] < frame["inner_h"] for frame in bottom_frames)


def _sample_layout() -> tuple[np.ndarray, list[dict], int] | None:
    frame_count = random.choice([2, 3])
    if frame_count == 2:
        top_n, bottom_n = 1, 1
    else:
        top_n, bottom_n = random.choice([(1, 2), (2, 1)])

    top_sizes = [(random.randint(5, 7), random.randint(5, 7)) for _ in range(top_n)]
    bottom_sizes = [(random.randint(5, 7), random.randint(5, 7)) for _ in range(bottom_n)]

    max_top_h = max(h for h, _w in top_sizes)
    max_bottom_h = max(h for h, _w in bottom_sizes)
    min_h = max(15, max_top_h + 1 + max_bottom_h)
    max_h = min(20, min_h + 4)
    if min_h > max_h:
        return None
    grid_h = random.randint(min_h, max_h)

    top_required_w = sum(w for _h, w in top_sizes) + 2 * max(0, top_n - 1)
    bottom_required_w = sum(w for _h, w in bottom_sizes) + 2 * max(0, bottom_n - 1)
    min_w = max(14, top_required_w, bottom_required_w)
    max_w = min(22, min_w + 4)
    if min_w > max_w:
        return None
    grid_w = random.randint(min_w, max_w)

    extra_rows = grid_h - (max_top_h + 1 + max_bottom_h)
    top_slack = random.randint(0, extra_rows)
    baseline_r = max_top_h + top_slack

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[baseline_r, :] = BASELINE

    colors = random.sample(FRAME_COLORS, frame_count)
    frame_specs: list[dict] = []

    top_cols = _column_positions([w for _h, w in top_sizes], grid_w)
    bottom_cols = _column_positions([w for _h, w in bottom_sizes], grid_w)

    for idx, ((outer_h, outer_w), c0) in enumerate(zip(top_sizes, top_cols)):
        r0 = random.randint(0, baseline_r - outer_h)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        frame_specs.append(
            {
                "color": colors[idx],
                "bbox": bbox,
                "side": "top",
                "inner_h": outer_h - 2,
                "border": _border_cells(bbox),
            }
        )

    for idx, ((outer_h, outer_w), c0) in enumerate(zip(bottom_sizes, bottom_cols), start=top_n):
        r0 = random.randint(baseline_r + 1, grid_h - outer_h)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        frame_specs.append(
            {
                "color": colors[idx],
                "bbox": bbox,
                "side": "bottom",
                "inner_h": outer_h - 2,
                "border": _border_cells(bbox),
            }
        )

    for frame in frame_specs:
        _draw_frame(grid, frame["bbox"], frame["color"])

    _choose_counts(frame_specs)
    return grid, frame_specs, baseline_r


def _place_markers(grid: np.ndarray, frame_specs: list[dict], baseline_r: int) -> bool:
    blocked = {
        (int(r), int(c))
        for r, c in np.argwhere((grid != BG) | (np.arange(grid.shape[0])[:, None] == baseline_r))
    }
    chosen: set[tuple[int, int]] = set()
    for frame in sorted(frame_specs, key=lambda item: item["count"], reverse=True):
        for _ in range(frame["count"]):
            options = _marker_candidates(grid.shape, frame, frame_specs, baseline_r, blocked, chosen)
            if not options:
                return False
            cell = random.choice(options)
            chosen.add(cell)
            grid[cell] = GREY
    return True


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, frame_specs: list[dict]) -> bool:
    try:
        frames = _parse_frames(inp)
        counts = _associate_markers(inp, frames)
        if {frame["color"] for frame in frames} != {frame["color"] for frame in frame_specs}:
            return False
        if not np.array_equal(main(inp), expected):
            return False
        top_seen = False
        bottom_seen = False
        partial_seen = False
        for frame in frames:
            meta = next(spec for spec in frame_specs if spec["color"] == frame["color"])
            if counts[frame["color"]] != meta["count"]:
                return False
            if meta["side"] == "top":
                top_seen = True
            else:
                bottom_seen = True
            partial_seen |= meta["count"] < meta["inner_h"]
        return top_seen and bottom_seen and partial_seen
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_layout()
    if sampled is None:
        return None
    grid, frame_specs, baseline_r = sampled
    if not _place_markers(grid, frame_specs, baseline_r):
        return None
    expected = _build_expected(grid, frame_specs, baseline_r)
    if not _validate_semantics(grid, expected, frame_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-count baseline-fill puzzle")

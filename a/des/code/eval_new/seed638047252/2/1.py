from __future__ import annotations

from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 300

DISTRACTOR_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(cells)),
                }
            )
    return comps


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _find_components(grid == int(color)):
            comp["color"] = int(color)
            comps.append(comp)
    return comps


def _shape_key(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = int(coords[:, 0].max() - r0 + 1)
    w = int(coords[:, 1].max() - c0 + 1)
    return (h, w, rel)


def _run_segments(line_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, present in enumerate(bool(v) for v in line_mask):
        if present and start is None:
            start = int(i)
        if not present and start is not None:
            runs.append((int(start), int(i)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(line_mask))))
    return runs


def _gap_sort_key(gap: dict) -> tuple[int, int]:
    side_order = {"top": 0, "bottom": 1, "left": 2, "right": 3}
    if gap["side"] in ("top", "bottom"):
        pos = int(gap["coords"][0][1])
    else:
        pos = int(gap["coords"][0][0])
    return (int(side_order[gap["side"]]), pos)


def _find_gaps(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, r1, c1 = map(int, bbox)
    gaps: list[dict] = []

    top_runs = _run_segments(grid[r0, c0 + 1 : c1] == int(bg))
    for start, end in top_runs:
        coords = [(int(r0), int(c0 + 1 + k)) for k in range(start, end)]
        gaps.append({"side": "top", "coords": coords})

    bottom_runs = _run_segments(grid[r1, c0 + 1 : c1] == int(bg))
    for start, end in bottom_runs:
        coords = [(int(r1), int(c0 + 1 + k)) for k in range(start, end)]
        gaps.append({"side": "bottom", "coords": coords})

    left_runs = _run_segments(grid[r0 + 1 : r1, c0] == int(bg))
    for start, end in left_runs:
        coords = [(int(r0 + 1 + k), int(c0)) for k in range(start, end)]
        gaps.append({"side": "left", "coords": coords})

    right_runs = _run_segments(grid[r0 + 1 : r1, c1] == int(bg))
    for start, end in right_runs:
        coords = [(int(r0 + 1 + k), int(c1)) for k in range(start, end)]
        gaps.append({"side": "right", "coords": coords})

    return sorted(gaps, key=_gap_sort_key)


def _find_marker_for_gap(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    bbox: tuple[int, int, int, int],
    gap: dict,
) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    r0, c0, r1, c1 = map(int, bbox)
    side = str(gap["side"])
    coords = gap["coords"]

    if side == "top":
        span = [int(c) for _r, c in coords]
        candidate_groups = [
            [(int(r1 - 1), int(c)) for c in span],
            [(int(r1), int(c)) for c in span],
        ]
    elif side == "bottom":
        span = [int(c) for _r, c in coords]
        candidate_groups = [
            [(int(r0 + 1), int(c)) for c in span],
            [(int(r0), int(c)) for c in span],
        ]
    elif side == "left":
        span = [int(r) for r, _c in coords]
        candidate_groups = [
            [(int(r), int(c1 - 1)) for r in span],
            [(int(r), int(c1)) for r in span],
        ]
    else:
        span = [int(r) for r, _c in coords]
        candidate_groups = [
            [(int(r), int(c0 + 1)) for r in span],
            [(int(r), int(c0)) for r in span],
        ]

    for candidates in candidate_groups:
        hits = [
            (int(r), int(c))
            for r, c in candidates
            if 0 <= r < h and 0 <= c < w and int(grid[r, c]) not in (int(bg), int(frame_color))
        ]
        if len(hits) == 1:
            mr, mc = hits[0]
            return (int(mr), int(mc)), int(grid[mr, mc])
        if len(hits) > 1:
            raise AssertionError("expected a unique marker for each gap")

    raise AssertionError("missing opposite-side marker")


def _detect_frame_bbox(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    color_totals = sorted(
        (
            (int(color), int(np.sum(grid == int(color))))
            for color in np.unique(grid)
            if int(color) != int(bg)
        ),
        key=lambda item: -item[1],
    )

    for frame_color, _total in color_totals:
        mask = grid == int(frame_color)
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        row_candidates = [int(r) for r, count in enumerate(row_counts) if int(count) >= 4]
        col_candidates = [int(c) for c, count in enumerate(col_counts) if int(count) >= 4]
        best_bbox: tuple[int, int, int, int] | None = None
        best_score: tuple[int, int] | None = None

        for i, r0 in enumerate(row_candidates):
            for r1 in row_candidates[i + 1 :]:
                if int(r1 - r0 + 1) < 6:
                    continue
                for j, c0 in enumerate(col_candidates):
                    for c1 in col_candidates[j + 1 :]:
                        if int(c1 - c0 + 1) < 6:
                            continue

                        top = grid[r0, c0 : c1 + 1]
                        bottom = grid[r1, c0 : c1 + 1]
                        left = grid[r0 + 1 : r1, c0]
                        right = grid[r0 + 1 : r1, c1]
                        border_vals = np.concatenate([top, bottom, left, right])

                        frame_hits = int(np.sum(border_vals == int(frame_color)))
                        bg_hits = int(np.sum(border_vals == int(bg)))
                        other_hits = int(border_vals.size - frame_hits - bg_hits)
                        if other_hits != 0 or bg_hits < 2:
                            continue

                        side_hits = [
                            int(np.sum(top == int(frame_color))),
                            int(np.sum(bottom == int(frame_color))),
                            int(np.sum(left == int(frame_color))),
                            int(np.sum(right == int(frame_color))),
                        ]
                        if min(side_hits) <= 0:
                            continue

                        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                        if int(np.sum(interior == int(frame_color))) != 0:
                            continue

                        score = (frame_hits, int((r1 - r0 + 1) * (c1 - c0 + 1)))
                        if best_score is None or score > best_score:
                            best_score = score
                            best_bbox = (int(r0), int(c0), int(r1), int(c1))

        if best_bbox is not None:
            return int(frame_color), best_bbox

    raise AssertionError("failed to detect frame bbox")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor and dock exterior fragments into the matching gaps of a hollow frame.

    Input structure:
    - bg is black or the dominant background color.
    - One large monochrome hollow rectangular frame has wall thickness 1 and several straight border gaps.
    - Each gap faces a single colored marker placed just inside the frame near the opposite wall.
    - Outside the frame are several small monochrome fragments, all in the frame color; some match gap shapes and others are distractors.

    Rule:
    - Detect the large frame and read each continuous gap on its four border sides.
    - For every gap, find the exterior fragment whose exact binary shape matches that gap without rotating it.
    - Read the color of the marker directly opposite the gap inside the frame.
    - Move the matching fragment into the gap, recolor it to the marker color, and erase the marker to background.
    - Leave unmatched exterior fragments unchanged.

    Key insight:
    - The right fragment is determined purely by gap shape/orientation, while the final color comes from an independent opposite-wall marker.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    comps = _find_monochrome_components(g, bg)
    assert comps, "expected foreground objects"

    frame_color, frame_bbox = _detect_frame_bbox(g, bg)
    gaps = _find_gaps(g, frame_bbox, bg)
    assert gaps, "expected at least one frame gap"

    fr0, fc0, fr1, fc1 = map(int, frame_bbox)

    pieces_by_key: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = defaultdict(list)
    for comp in comps:
        if int(comp["color"]) != int(frame_color):
            continue
        touches_frame = any(
            fr0 <= int(r) <= fr1
            and fc0 <= int(c) <= fc1
            and (int(r) in (fr0, fr1) or int(c) in (fc0, fc1))
            for r, c in comp["coords"]
        )
        if touches_frame:
            continue
        pieces_by_key[_shape_key(comp["coords"])].append(comp)

    for bucket in pieces_by_key.values():
        bucket.sort(key=lambda comp: tuple(map(int, comp["bbox"])))

    gaps_by_key: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[dict]] = defaultdict(list)
    for gap in gaps:
        gaps_by_key[_shape_key(np.array(gap["coords"], dtype=int))].append(gap)

    matches: list[tuple[dict, dict]] = []
    for key, gap_group in gaps_by_key.items():
        gap_group = sorted(gap_group, key=_gap_sort_key)
        available = pieces_by_key.get(key, [])
        assert len(available) >= len(gap_group), "not enough matching fragments"
        for gap, piece in zip(gap_group, available[: len(gap_group)]):
            matches.append((gap, piece))

    for gap, piece in matches:
        marker_pos, marker_color = _find_marker_for_gap(g, bg, frame_color, frame_bbox, gap)
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)
        for r, c in gap["coords"]:
            out[int(r), int(c)] = int(marker_color)
        mr, mc = marker_pos
        out[int(mr), int(mc)] = int(bg)

    return out


def _side_requirement(lengths: list[int]) -> int:
    if not lengths:
        return 0
    return int(sum(lengths) + max(0, len(lengths) - 1) + 2)


def _distribute_lengths(lengths: list[int]) -> tuple[list[int], list[int]]:
    if not lengths:
        return [], []
    n = len(lengths)
    min_left = max(0, n - 2)
    max_left = min(2, n)
    left_count = int(RNG.integers(min_left, max_left + 1))
    left_choice = set(int(x) for x in RNG.choice(lengths, size=left_count, replace=False).tolist())
    a = [int(x) for x in lengths if int(x) in left_choice]
    b = [int(x) for x in lengths if int(x) not in left_choice]
    RNG.shuffle(a)
    RNG.shuffle(b)
    return a, b


def _place_side_segments(edge_len: int, lengths: list[int]) -> list[tuple[int, int]]:
    if not lengths:
        return []
    for _ in range(200):
        used = np.zeros(edge_len, dtype=bool)
        placed: list[tuple[int, int]] = []
        order = list(int(x) for x in lengths)
        RNG.shuffle(order)
        ok = True
        for length in order:
            candidates: list[int] = []
            for start in range(1, edge_len - length):
                end = int(start + length)
                if bool(used[start:end].any()):
                    continue
                candidates.append(int(start))
            if not candidates:
                ok = False
                break
            start = int(RNG.choice(candidates))
            end = int(start + length)
            used[max(0, start - 1) : min(edge_len, end + 1)] = True
            placed.append((int(start), int(end)))
        if ok:
            return sorted(placed)
    raise RuntimeError("failed to place side gaps")


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                blocked[int(rr), int(cc)] = True


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in np.argwhere(mask)]


def _place_piece(
    grid: np.ndarray,
    blocked: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color: int,
    mask: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

    for top in range(0, h - mh + 1):
        for left in range(0, w - mw + 1):
            cells = _mask_cells(mask, int(top), int(left))
            ok = True
            for r, c in cells:
                if fr0 <= r <= fr1 and fc0 <= c <= fc1:
                    ok = False
                    break
                if bool(blocked[r, c]):
                    ok = False
                    break
            if ok:
                candidates.append((int(top), int(left), cells))

    if not candidates:
        return None

    _top, _left, cells = candidates[int(RNG.integers(len(candidates)))]
    for r, c in cells:
        grid[int(r), int(c)] = int(color)
    _mark_blocked(blocked, cells)
    return cells


def _build_expected_output(inp: np.ndarray, matches: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for match in matches:
        for r, c in match["piece_cells"]:
            out[int(r), int(c)] = int(bg)
        for r, c in match["gap_cells"]:
            out[int(r), int(c)] = int(match["marker_color"])
        mr, mc = match["marker_pos"]
        out[int(mr), int(mc)] = int(bg)
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    frame_color = int(RNG.choice(np.arange(1, 10)))

    total_gaps = int(RNG.integers(2, 7))
    min_horiz = max(0, total_gaps - 3)
    max_horiz = min(3, total_gaps)
    horiz_count = int(RNG.integers(min_horiz, max_horiz + 1))
    vert_count = int(total_gaps - horiz_count)

    horiz_lengths = [int(x) for x in RNG.choice([2, 3, 4], size=horiz_count, replace=False).tolist()]
    vert_lengths = [int(x) for x in RNG.choice([2, 3, 4], size=vert_count, replace=False).tolist()]

    top_lengths, bottom_lengths = _distribute_lengths(horiz_lengths)
    left_lengths, right_lengths = _distribute_lengths(vert_lengths)

    min_frame_h = max(8, _side_requirement(left_lengths), _side_requirement(right_lengths)) + 2
    min_frame_w = max(8, _side_requirement(top_lengths), _side_requirement(bottom_lengths)) + 2
    frame_h = int(min_frame_h + RNG.integers(0, 5))
    frame_w = int(min_frame_w + RNG.integers(0, 7))

    if frame_h + 6 > 30 or frame_w + 6 > 30:
        return None

    grid_h = int(RNG.integers(max(18, frame_h + 6), 31))
    grid_w = int(RNG.integers(max(18, frame_w + 6), 31))
    r0 = int(RNG.integers(2, grid_h - frame_h - 1))
    c0 = int(RNG.integers(2, grid_w - frame_w - 1))
    r1 = int(r0 + frame_h - 1)
    c1 = int(c0 + frame_w - 1)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color

    try:
        top_segments = _place_side_segments(frame_w - 2, top_lengths)
        bottom_segments = _place_side_segments(frame_w - 2, bottom_lengths)
        left_segments = _place_side_segments(frame_h - 2, left_lengths)
        right_segments = _place_side_segments(frame_h - 2, right_lengths)
    except RuntimeError:
        return None

    gap_specs: list[dict] = []
    for start, end in top_segments:
        coords = [(int(r0), int(c0 + 1 + k)) for k in range(start, end)]
        gap_specs.append({"side": "top", "coords": coords})
    for start, end in bottom_segments:
        coords = [(int(r1), int(c0 + 1 + k)) for k in range(start, end)]
        gap_specs.append({"side": "bottom", "coords": coords})
    for start, end in left_segments:
        coords = [(int(r0 + 1 + k), int(c0)) for k in range(start, end)]
        gap_specs.append({"side": "left", "coords": coords})
    for start, end in right_segments:
        coords = [(int(r0 + 1 + k), int(c1)) for k in range(start, end)]
        gap_specs.append({"side": "right", "coords": coords})

    for gap in gap_specs:
        for r, c in gap["coords"]:
            grid[int(r), int(c)] = int(bg)

    colors_available = [int(v) for v in range(1, 10) if int(v) != int(frame_color)]
    if len(colors_available) < len(gap_specs):
        return None
    marker_colors = [int(x) for x in RNG.choice(colors_available, size=len(gap_specs), replace=False).tolist()]

    for gap, marker_color in zip(sorted(gap_specs, key=_gap_sort_key), marker_colors):
        side = str(gap["side"])
        coords = gap["coords"]
        offset = int(RNG.integers(len(coords)))
        if side == "top":
            mr, mc = int(r1 - 1), int(coords[offset][1])
        elif side == "bottom":
            mr, mc = int(r0 + 1), int(coords[offset][1])
        elif side == "left":
            mr, mc = int(coords[offset][0]), int(c1 - 1)
        else:
            mr, mc = int(coords[offset][0]), int(c0 + 1)
        grid[int(mr), int(mc)] = int(marker_color)
        gap["marker_pos"] = (int(mr), int(mc))
        gap["marker_color"] = int(marker_color)

    blocked = np.zeros_like(grid, dtype=bool)
    foreground_cells = [(int(r), int(c)) for r, c in np.argwhere(grid != int(bg))]
    _mark_blocked(blocked, foreground_cells)

    matches: list[dict] = []
    for gap in sorted(gap_specs, key=_gap_sort_key):
        mask = np.zeros(
            (
                int(max(r for r, _c in gap["coords"]) - min(r for r, _c in gap["coords"]) + 1),
                int(max(c for _r, c in gap["coords"]) - min(c for _r, c in gap["coords"]) + 1),
            ),
            dtype=bool,
        )
        base_r = int(min(r for r, _c in gap["coords"]))
        base_c = int(min(c for _r, c in gap["coords"]))
        for r, c in gap["coords"]:
            mask[int(r - base_r), int(c - base_c)] = True
        piece_cells = _place_piece(grid, blocked, (r0, c0, r1, c1), frame_color, mask)
        if piece_cells is None:
            return None
        matches.append(
            {
                "gap_cells": [(int(r), int(c)) for r, c in gap["coords"]],
                "piece_cells": piece_cells,
                "marker_pos": tuple(map(int, gap["marker_pos"])),
                "marker_color": int(gap["marker_color"]),
            }
        )

    distractor_count = int(RNG.integers(0, 4))
    distractor_order = [DISTRACTOR_SHAPES[int(i)] for i in RNG.integers(0, len(DISTRACTOR_SHAPES), size=distractor_count)]
    for mask in distractor_order:
        placed = _place_piece(grid, blocked, (r0, c0, r1, c1), frame_color, mask)
        if placed is None:
            return None

    expected = _build_expected_output(grid, matches, bg)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is None:
            continue
        inp, expected = result
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")

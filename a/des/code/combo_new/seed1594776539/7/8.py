import random

import numpy as np


BG = 0
GREY = 5
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 500
SPAN_TRIES = 180


def _band_bounds(center: int, thickness: int) -> tuple[int, int]:
    return int(center - (thickness - 1) // 2), int(center + thickness // 2)


def _legend_segments(row: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    c = 0
    while c < row.size:
        value = int(row[c])
        if value == GREY:
            start = c
            while c < row.size and int(row[c]) == GREY:
                c += 1
            segments.append((int(start), int(c - start)))
        else:
            assert value == BG, "legend row may only contain black gaps and grey segments"
            c += 1
    return segments


def _extract_pairs(grid: np.ndarray, legend_r: int) -> list[dict]:
    pairs: list[dict] = []
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, GREY)]
    assert 2 <= len(colors) <= 4, "expected 2 to 4 colored anchor pairs"

    for color in colors:
        coords = np.argwhere(grid == int(color))
        assert coords.shape == (2, 2), f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = (tuple(map(int, rc)) for rc in coords)
        assert r0 != legend_r and r1 != legend_r, "anchors may not lie on the legend row"
        assert r0 == r1 or c0 == c1, "matching anchors must share a row or a column"
        assert not (r0 == r1 and c0 == c1)
        distance = abs(r1 - r0) + abs(c1 - c0)
        assert distance > 0
        pairs.append(
            {
                "color": int(color),
                "distance": int(distance),
                "orient": "H" if r0 == r1 else "V",
                "a": (r0, c0),
                "b": (r1, c1),
            }
        )

    distances = [pair["distance"] for pair in pairs]
    assert len(set(distances)) == len(distances), "pair distances must be distinct"
    return pairs


def _paint_pair(out: np.ndarray, pair: dict) -> None:
    color = int(pair["color"])
    thickness = int(pair["thickness"])
    h, w = out.shape

    if pair["orient"] == "H":
        row = int(pair["a"][0])
        c0, c1 = sorted((int(pair["a"][1]), int(pair["b"][1])))
        r0, r1 = _band_bounds(row, thickness)
        r0 = max(0, r0)
        r1 = min(h - 1, r1)
        out[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    col = int(pair["a"][1])
    r0, r1 = sorted((int(pair["a"][0]), int(pair["b"][0])))
    c0, c1 = _band_bounds(col, thickness)
    c0 = max(0, c0)
    c1 = min(w - 1, c1)
    out[r0 : r1 + 1, c0 : c1 + 1] = color


def _render_output(grid: np.ndarray, pairs: list[dict], legend_r: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    legend_row = out[int(legend_r)].copy()
    for pair in sorted(pairs, key=lambda item: int(item["distance"]), reverse=True):
        _paint_pair(out, pair)
    out[int(legend_r)] = legend_row
    return out


def _pair_anchors(pair: dict) -> set[tuple[int, int]]:
    return {tuple(map(int, pair["a"])), tuple(map(int, pair["b"]))}


def _pair_footprint(pair: dict) -> set[tuple[int, int]]:
    thickness = int(pair["thickness"])
    cells: set[tuple[int, int]] = set()
    if pair["orient"] == "H":
        row = int(pair["row"])
        c0, c1 = sorted((int(pair["c0"]), int(pair["c1"])))
        r0, r1 = _band_bounds(row, thickness)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((int(r), int(c)))
        return cells

    col = int(pair["col"])
    r0, r1 = sorted((int(pair["r0"]), int(pair["r1"])))
    c0, c1 = _band_bounds(col, thickness)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cells.add((int(r), int(c)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored anchor pair with a thickness read from the sorted grey legend.

    Input structure:
    - The background is black.
    - Each non-black, non-grey color appears exactly twice as a pair of single-pixel anchors.
    - Every anchor pair is aligned either horizontally or vertically.
    - The bottom legend row contains separated grey segments whose lengths increase from left to right.

    Rule:
    - Read the grey legend segments from left to right and record their lengths.
    - Sort the colored anchor pairs by the distance between their two pixels.
    - Match the shortest pair to the shortest legend segment, the next shortest pair to the next segment, and so on.
    - For each pair, draw a straight horizontal or vertical line in that color between the anchors, including both endpoints, with thickness equal to its matched segment length.
    - Thickness expands symmetrically around the center line; when the thickness is even, the extra row or column is placed downward or rightward.
    - Draw the longest-distance pair first and the shortest-distance pair last so later lines overwrite earlier ones, while keeping the legend row unchanged.

    Key insight:
    - The puzzle combines three independent readings: color identifies the anchor pairing, distance order identifies the matching legend width, and overwrite order is the reverse of that distance ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_rows = [int(r) for r in range(g.shape[0]) if np.any(g[int(r)] == GREY)]
    assert grey_rows, "expected a grey legend row"
    legend_r = max(grey_rows)

    segments = _legend_segments(g[int(legend_r)])
    assert len(segments) >= 2, "legend must contain multiple grey segments"
    lengths = [int(length) for _start, length in segments]
    assert all(lengths[i] < lengths[i + 1] for i in range(len(lengths) - 1)), (
        "legend segment lengths must increase strictly left to right"
    )

    pairs = _extract_pairs(g, legend_r)
    assert len(pairs) == len(segments), "pair count must match legend segment count"

    ordered_pairs = sorted(pairs, key=lambda item: int(item["distance"]))
    for pair, (_start, length) in zip(ordered_pairs, segments):
        pair["thickness"] = int(length)

    return _render_output(g, ordered_pairs, legend_r)


def _random_split(total: int, parts: int) -> list[int]:
    buckets = [0] * int(parts)
    for _ in range(int(total)):
        buckets[random.randrange(parts)] += 1
    return buckets


def _allocate_bands(sizes: list[int], extent: int) -> list[int] | None:
    if not sizes:
        return []
    required = int(sum(sizes) + max(0, len(sizes) - 1))
    if required > extent:
        return None
    extras = _random_split(int(extent - required), len(sizes) + 1)
    starts: list[int] = []
    pos = int(extras[0])
    for i, size in enumerate(sizes):
        starts.append(int(pos))
        pos += int(size)
        if i < len(sizes) - 1:
            pos += 1 + int(extras[i + 1])
    return starts


def _place_centers(pairs: list[dict], content_h: int, width: int) -> bool:
    horiz_ids = [i for i, pair in enumerate(pairs) if pair["orient"] == "H"]
    vert_ids = [i for i, pair in enumerate(pairs) if pair["orient"] == "V"]
    random.shuffle(horiz_ids)
    random.shuffle(vert_ids)

    horiz_starts = _allocate_bands([int(pairs[i]["thickness"]) for i in horiz_ids], content_h)
    vert_starts = _allocate_bands([int(pairs[i]["thickness"]) for i in vert_ids], width)
    if horiz_starts is None or vert_starts is None:
        return False

    for idx, start in zip(horiz_ids, horiz_starts):
        thickness = int(pairs[idx]["thickness"])
        pairs[idx]["row"] = int(start + (thickness - 1) // 2)

    for idx, start in zip(vert_ids, vert_starts):
        thickness = int(pairs[idx]["thickness"])
        pairs[idx]["col"] = int(start + (thickness - 1) // 2)

    return True


def _anchor_clear(pairs: list[dict]) -> bool:
    footprints = [_pair_footprint(pair) for pair in pairs]
    anchors = [_pair_anchors(pair) for pair in pairs]
    for i, anchor_cells in enumerate(anchors):
        for j, footprint in enumerate(footprints):
            if i != j and anchor_cells & footprint:
                return False
    return True


def _overlap_metrics(pairs: list[dict]) -> tuple[int, int]:
    footprints = [_pair_footprint(pair) for pair in pairs]
    overlap_pairs = 0
    overlap_cells = 0
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            shared = footprints[i] & footprints[j]
            if shared:
                overlap_pairs += 1
                overlap_cells += len(shared)
    return int(overlap_pairs), int(overlap_cells)


def _sample_spans(
    base_pairs: list[dict],
    content_h: int,
    width: int,
    *,
    min_overlap_pairs: int,
    max_overlap_pairs: int,
) -> list[dict] | None:
    for _ in range(SPAN_TRIES):
        pairs = [dict(pair) for pair in base_pairs]
        ok = True
        for pair in pairs:
            if pair["orient"] == "H":
                max_start = int(width - pair["distance"] - 1)
                if max_start < 0:
                    ok = False
                    break
                pair["c0"] = random.randint(0, max_start)
                pair["c1"] = int(pair["c0"] + pair["distance"])
                pair["a"] = (int(pair["row"]), int(pair["c0"]))
                pair["b"] = (int(pair["row"]), int(pair["c1"]))
            else:
                max_start = int(content_h - pair["distance"] - 1)
                if max_start < 0:
                    ok = False
                    break
                pair["r0"] = random.randint(0, max_start)
                pair["r1"] = int(pair["r0"] + pair["distance"])
                pair["a"] = (int(pair["r0"]), int(pair["col"]))
                pair["b"] = (int(pair["r1"]), int(pair["col"]))
        if not ok or not _anchor_clear(pairs):
            continue

        overlap_pairs, overlap_cells = _overlap_metrics(pairs)
        if not (min_overlap_pairs <= overlap_pairs <= max_overlap_pairs):
            continue
        if overlap_cells > 28:
            continue
        return pairs
    return None


def _place_legend(width: int, lengths: list[int]) -> list[int] | None:
    required = int(sum(lengths) + max(0, len(lengths) - 1))
    if required > width:
        return None
    extras = _random_split(int(width - required), len(lengths) + 1)
    starts: list[int] = []
    pos = int(extras[0])
    for i, length in enumerate(lengths):
        starts.append(int(pos))
        pos += int(length)
        if i < len(lengths) - 1:
            pos += 1 + int(extras[i + 1])
    return starts


def _build_input(shape: tuple[int, int], pairs: list[dict], legend_starts: list[int], legend_lengths: list[int]) -> np.ndarray:
    h, w = shape
    legend_r = h - 1
    grid = np.full((h, w), BG, dtype=int)
    for pair in pairs:
        color = int(pair["color"])
        for r, c in _pair_anchors(pair):
            grid[int(r), int(c)] = color
    for start, length in zip(legend_starts, legend_lengths):
        grid[int(legend_r), int(start) : int(start + length)] = GREY
    return grid


def _sample_thicknesses(count: int) -> list[int]:
    options = list(range(1, 7))
    while True:
        lengths = sorted(random.sample(options, count))
        if any(length % 2 == 0 for length in lengths) and any(length % 2 == 1 for length in lengths):
            return [int(length) for length in lengths]


def _sample_distances(count: int) -> list[int]:
    return [int(value) for value in sorted(random.sample(range(3, 12), count))]


def _orientation_list(count: int) -> list[str]:
    h_count = random.randint(1, count - 1)
    orientations = ["H"] * h_count + ["V"] * (count - h_count)
    random.shuffle(orientations)
    return orientations


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    colors = random.sample(DRAW_COLORS, pair_count)
    thicknesses = _sample_thicknesses(pair_count)
    distances = _sample_distances(pair_count)
    orientations = _orientation_list(pair_count)

    pairs = [
        {
            "color": int(color),
            "distance": int(distance),
            "thickness": int(thickness),
            "orient": orient,
        }
        for color, distance, thickness, orient in zip(colors, distances, thicknesses, orientations)
    ]

    horiz_thicknesses = [int(pair["thickness"]) for pair in pairs if pair["orient"] == "H"]
    vert_thicknesses = [int(pair["thickness"]) for pair in pairs if pair["orient"] == "V"]
    horiz_distances = [int(pair["distance"]) for pair in pairs if pair["orient"] == "H"]
    vert_distances = [int(pair["distance"]) for pair in pairs if pair["orient"] == "V"]

    legend_need = int(sum(thicknesses) + pair_count + 1)
    min_content_h = max(
        int(sum(horiz_thicknesses) + max(0, len(horiz_thicknesses) - 1)),
        int(max(vert_distances, default=1) + 3),
        8,
    )
    min_width = max(
        int(sum(vert_thicknesses) + max(0, len(vert_thicknesses) - 1)),
        int(max(horiz_distances, default=1) + 3),
        int(legend_need),
    )

    if min_content_h > 29 or min_width > 30:
        return None

    content_h = random.randint(min_content_h, min(29, min_content_h + 6))
    width = random.randint(min_width, min(30, min_width + 8))
    if not _place_centers(pairs, content_h, width):
        return None

    want_overlap = random.random() < 0.6
    placed = _sample_spans(
        pairs,
        content_h,
        width,
        min_overlap_pairs=1 if want_overlap else 0,
        max_overlap_pairs=2 if want_overlap else 0,
    )
    if placed is None:
        placed = _sample_spans(pairs, content_h, width, min_overlap_pairs=0, max_overlap_pairs=2)
    if placed is None:
        return None

    legend_starts = _place_legend(width, thicknesses)
    if legend_starts is None:
        return None

    inp = _build_input((content_h + 1, width), placed, legend_starts, thicknesses)
    expected = _render_output(inp, [dict(pair) for pair in placed], content_h)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

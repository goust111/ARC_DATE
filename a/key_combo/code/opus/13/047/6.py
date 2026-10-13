from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 500


def _infer_divider_color(grid: np.ndarray) -> int:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    common = set(row_colors) & set(col_colors)
    assert common, "expected a divider color spanning full rows and columns"
    if GREY in common:
        return GREY
    return Counter(row_colors + col_colors).most_common(1)[0][0]


def _bands(dividers: list[int], limit: int) -> list[tuple[int, int]]:
    dividers = sorted(set(int(x) for x in dividers))
    assert len(dividers) >= 2
    bands: list[tuple[int, int]] = []
    for a, b in zip(dividers, dividers[1:]):
        if b - a > 1:
            bands.append((a + 1, b - 1))
    assert bands, "expected at least one non-divider band"
    assert dividers[0] == 0 and dividers[-1] == limit - 1, "generator uses a bordered divider grid"
    return bands


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    value = int(value)
    for idx, (lo, hi) in enumerate(bands):
        if lo <= value <= hi:
            return idx
    raise AssertionError(f"value {value} is not inside any cell band")


def _band_center(band: tuple[int, int]) -> int:
    lo, hi = map(int, band)
    assert (hi - lo + 1) % 2 == 1, "cell interior must have a unique center"
    return (lo + hi) // 2


def _path_positions(
    orientation: str,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    divider_rows: set[int],
    divider_cols: set[int],
    fixed_index: int,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    if orientation == "h":
        r = _band_center(row_bands[fixed_index])
        c0 = _band_center(col_bands[start])
        c1 = _band_center(col_bands[end])
        return [(r, c) for c in range(c0, c1 + 1) if c not in divider_cols]

    c = _band_center(col_bands[fixed_index])
    r0 = _band_center(row_bands[start])
    r1 = _band_center(row_bands[end])
    return [(r, c) for r in range(r0, r1 + 1) if r not in divider_rows]


def _segment_mid2(orientation: str, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]], fixed_index: int, start: int, end: int) -> tuple[int, int]:
    if orientation == "h":
        r = _band_center(row_bands[fixed_index])
        c0 = _band_center(col_bands[start])
        c1 = _band_center(col_bands[end])
        return 2 * r, c0 + c1
    c = _band_center(col_bands[fixed_index])
    r0 = _band_center(row_bands[start])
    r1 = _band_center(row_bands[end])
    return r0 + r1, 2 * c


def _analyze_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    divider = _infer_divider_color(g)
    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == divider)]
    divider_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == divider)]
    row_bands = _bands(divider_rows, g.shape[0])
    col_bands = _bands(divider_cols, g.shape[1])
    center_rows = [_band_center(band) for band in row_bands]
    center_cols = [_band_center(band) for band in col_bands]

    points_by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BLACK, divider):
                continue
            points_by_color.setdefault(color, []).append((r, c))

    assert len(points_by_color) >= 3, "expected at least three colored pairs"

    segments: list[dict] = []
    divider_row_set = set(divider_rows)
    divider_col_set = set(divider_cols)
    for color, pts in sorted(points_by_color.items()):
        assert len(pts) == 2, "each color must appear exactly twice"
        (r1, c1), (r2, c2) = sorted(pts)
        rr1 = _band_index(r1, row_bands)
        rr2 = _band_index(r2, row_bands)
        cc1 = _band_index(c1, col_bands)
        cc2 = _band_index(c2, col_bands)
        assert r1 == center_rows[rr1] and r2 == center_rows[rr2], "each marker must sit at a cell-row center"
        assert c1 == center_cols[cc1] and c2 == center_cols[cc2], "each marker must sit at a cell-column center"

        if rr1 == rr2:
            assert c1 != c2 and abs(cc1 - cc2) >= 2, "horizontal pairs need a gap of at least one cell"
            start, end = sorted((cc1, cc2))
            positions = _path_positions("h", row_bands, col_bands, divider_row_set, divider_col_set, rr1, start, end)
            mid2 = _segment_mid2("h", row_bands, col_bands, rr1, start, end)
            segment = {
                "color": color,
                "orientation": "h",
                "track": rr1,
                "start": start,
                "end": end,
                "positions": positions,
                "mid2": mid2,
            }
        else:
            assert cc1 == cc2 and abs(rr1 - rr2) >= 2, "vertical pairs need a gap of at least one cell"
            start, end = sorted((rr1, rr2))
            positions = _path_positions("v", row_bands, col_bands, divider_row_set, divider_col_set, cc1, start, end)
            mid2 = _segment_mid2("v", row_bands, col_bands, cc1, start, end)
            segment = {
                "color": color,
                "orientation": "v",
                "track": cc1,
                "start": start,
                "end": end,
                "positions": positions,
                "mid2": mid2,
            }

        endpoints = set(pts)
        for pos in positions:
            value = int(g[pos])
            if pos in endpoints:
                assert value == color
            else:
                assert value == BLACK, "segment corridors must be black before filling"
        segment["tail_start"] = (len(positions) + 1) // 2
        segments.append(segment)

    for seg in segments:
        candidates = []
        for other in segments:
            if other is seg or other["orientation"] != seg["orientation"]:
                continue
            if abs(other["track"] - seg["track"]) != 1:
                continue
            dist = abs(seg["mid2"][0] - other["mid2"][0]) + abs(seg["mid2"][1] - other["mid2"][1])
            candidates.append((dist, other))

        if candidates:
            best_dist = min(dist for dist, _other in candidates)
            best = [other for dist, other in candidates if dist == best_dist]
            seg["neighbor_count"] = len(candidates)
            seg["donor"] = sorted(best, key=lambda item: (item["track"], item["start"], item["end"], item["color"]))[0]
            seg["donor_tie"] = len(best) > 1
        else:
            seg["neighbor_count"] = 0
            seg["donor"] = None
            seg["donor_tie"] = False

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color markers across a divider grid, then recolor each line's back half from a divider-adjacent donor line.

    Input structure:
    - Black background with grey full rows and columns that partition the canvas into rectangular cells.
    - Some cell centers contain a single colored marker; every non-grey, non-black color appears exactly twice.
    - Each matching pair lies in one cell row or one cell column, with at least one empty cell between the two markers.

    Rule:
    - Identify the grey divider rows and columns and recover the cell lattice.
    - For each matching color pair, draw a straight horizontal or vertical connection between the two markers,
      filling only the black corridor cells and leaving grey divider pixels unchanged.
    - Order each line's own colored pixels from left to right or top to bottom, keep the first ceil(length/2)
      pixels in the original color, and recolor the remaining floor(length/2) pixels.
    - The recolor target is the original color of the nearest same-orientation line that lies across one shared
      grey divider band in an adjacent cell row or adjacent cell column; if no such neighbor exists, keep the line color.

    Key insight:
    - The grey lattice does two jobs at once: it constrains how identical markers are connected, and it also encodes
      which nearby line supplies the replacement color for the back half.
    """
    g = np.asarray(grid, dtype=int)
    segments = _analyze_segments(g)
    out = g.copy()

    for seg in segments:
        for r, c in seg["positions"]:
            out[r, c] = seg["color"]

    for seg in segments:
        donor_color = seg["color"] if seg["donor"] is None else seg["donor"]["color"]
        for r, c in seg["positions"][seg["tail_start"] :]:
            out[r, c] = donor_color

    return out


def _apply_flips(pairs: list[dict], row_count: int, col_count: int, flip_r: bool, flip_c: bool) -> list[dict]:
    transformed: list[dict] = []
    for pair in pairs:
        item = dict(pair)
        if pair["orientation"] == "h":
            if flip_r:
                item["track"] = row_count - 1 - pair["track"]
            if flip_c:
                item["start"] = col_count - 1 - pair["end"]
                item["end"] = col_count - 1 - pair["start"]
        else:
            if flip_c:
                item["track"] = col_count - 1 - pair["track"]
            if flip_r:
                item["start"] = row_count - 1 - pair["end"]
                item["end"] = row_count - 1 - pair["start"]
        transformed.append(item)
    return transformed


def _render_input(row_count: int, col_count: int, cell_h: int, cell_w: int, pairs: list[dict]) -> np.ndarray:
    height = row_count * cell_h + (row_count + 1)
    width = col_count * cell_w + (col_count + 1)
    grid = np.full((height, width), BLACK, dtype=int)
    for r in range(0, height, cell_h + 1):
        grid[r, :] = GREY
    for c in range(0, width, cell_w + 1):
        grid[:, c] = GREY

    row_centers = [1 + rr * (cell_h + 1) + cell_h // 2 for rr in range(row_count)]
    col_centers = [1 + cc * (cell_w + 1) + cell_w // 2 for cc in range(col_count)]
    for pair in pairs:
        color = pair["color"]
        if pair["orientation"] == "h":
            r = row_centers[pair["track"]]
            grid[r, col_centers[pair["start"]]] = color
            grid[r, col_centers[pair["end"]]] = color
        else:
            c = col_centers[pair["track"]]
            grid[row_centers[pair["start"]], c] = color
            grid[row_centers[pair["end"]], c] = color
    return grid


def _expected_from_pairs(row_count: int, col_count: int, cell_h: int, cell_w: int, pairs: list[dict]) -> tuple[np.ndarray, dict]:
    inp = _render_input(row_count, col_count, cell_h, cell_w, pairs)
    row_bands = [(1 + rr * (cell_h + 1), rr * (cell_h + 1) + cell_h) for rr in range(row_count)]
    col_bands = [(1 + cc * (cell_w + 1), cc * (cell_w + 1) + cell_w) for cc in range(col_count)]
    divider_rows = set(range(0, inp.shape[0], cell_h + 1))
    divider_cols = set(range(0, inp.shape[1], cell_w + 1))

    segments: list[dict] = []
    for pair in pairs:
        positions = _path_positions(
            pair["orientation"],
            row_bands,
            col_bands,
            divider_rows,
            divider_cols,
            pair["track"],
            pair["start"],
            pair["end"],
        )
        mid2 = _segment_mid2(pair["orientation"], row_bands, col_bands, pair["track"], pair["start"], pair["end"])
        segments.append(
            {
                "color": pair["color"],
                "orientation": pair["orientation"],
                "track": pair["track"],
                "start": pair["start"],
                "end": pair["end"],
                "positions": positions,
                "mid2": mid2,
                "tail_start": (len(positions) + 1) // 2,
            }
        )

    donor_map: dict[int, int] = {}
    multi_neighbor = 0
    no_neighbor = 0
    tie_count = 0
    for seg in segments:
        candidates = []
        for other in segments:
            if other is seg or other["orientation"] != seg["orientation"]:
                continue
            if abs(other["track"] - seg["track"]) != 1:
                continue
            dist = abs(seg["mid2"][0] - other["mid2"][0]) + abs(seg["mid2"][1] - other["mid2"][1])
            candidates.append((dist, other))

        if not candidates:
            donor_map[seg["color"]] = seg["color"]
            no_neighbor += 1
            continue

        if len(candidates) > 1:
            multi_neighbor += 1
        best_dist = min(dist for dist, _other in candidates)
        best = [other for dist, other in candidates if dist == best_dist]
        if len(best) > 1:
            tie_count += 1
        chosen = sorted(best, key=lambda item: (item["track"], item["start"], item["end"], item["color"]))[0]
        donor_map[seg["color"]] = chosen["color"]

    out = inp.copy()
    for seg in segments:
        for r, c in seg["positions"]:
            out[r, c] = seg["color"]
    for seg in segments:
        for r, c in seg["positions"][seg["tail_start"] :]:
            out[r, c] = donor_map[seg["color"]]

    meta = {
        "pair_count": len(segments),
        "multi_neighbor_segments": multi_neighbor,
        "no_neighbor_segments": no_neighbor,
        "tie_count": tie_count,
        "donor_map": donor_map,
    }
    return out, meta


def _blueprint(kind: str, add_extra: bool) -> tuple[int, int, list[dict]]:
    if kind == "adj_h_plus_v":
        pairs = [
            {"orientation": "h", "track": 0, "start": 0, "end": 2},
            {"orientation": "h", "track": 1, "start": 1, "end": 3},
            {"orientation": "v", "track": 0, "start": 1, "end": 3},
        ]
        if add_extra:
            pairs.append({"orientation": "h", "track": 3, "start": 1, "end": 3})
        return 4, 4, pairs

    if kind == "triple_h":
        pairs = [
            {"orientation": "h", "track": 0, "start": 0, "end": 3},
            {"orientation": "h", "track": 1, "start": 0, "end": 2},
            {"orientation": "h", "track": 2, "start": 1, "end": 3},
        ]
        return random.choice([3, 4]), 4, pairs

    if kind == "adj_v_plus_h":
        pairs = [
            {"orientation": "v", "track": 0, "start": 0, "end": 2},
            {"orientation": "v", "track": 1, "start": 1, "end": 3},
            {"orientation": "h", "track": 0, "start": 1, "end": 3},
        ]
        if add_extra:
            pairs.append({"orientation": "v", "track": 3, "start": 1, "end": 3})
        return 4, 4, pairs

    if kind == "triple_v":
        pairs = [
            {"orientation": "v", "track": 0, "start": 0, "end": 3},
            {"orientation": "v", "track": 1, "start": 0, "end": 2},
            {"orientation": "v", "track": 2, "start": 1, "end": 3},
        ]
        return 4, random.choice([3, 4]), pairs

    raise AssertionError(f"unknown blueprint kind: {kind}")


def _gen_attempt() -> np.ndarray | None:
    kind = random.choice(["adj_h_plus_v", "triple_h", "adj_v_plus_h", "triple_v"])
    add_extra = random.random() < 0.75
    row_count, col_count, pairs = _blueprint(kind, add_extra)
    if max(row_count, col_count) > 4:
        return None

    cell_h = random.choice([3, 5])
    cell_w = random.choice([3, 5])
    pairs = _apply_flips(
        pairs,
        row_count,
        col_count,
        flip_r=bool(random.getrandbits(1)),
        flip_c=bool(random.getrandbits(1)),
    )

    colors = random.sample(PALETTE, len(pairs))
    for pair, color in zip(pairs, colors):
        pair["color"] = color

    expected, meta = _expected_from_pairs(row_count, col_count, cell_h, cell_w, pairs)
    if meta["pair_count"] < 3 or meta["tie_count"] != 0:
        return None
    if kind.startswith("triple") and meta["multi_neighbor_segments"] == 0:
        return None
    if kind.startswith("adj") and meta["no_neighbor_segments"] == 0:
        return None
    if kind.startswith("adj") and add_extra and meta["no_neighbor_segments"] < 2:
        return None

    inp = _render_input(row_count, col_count, cell_h, cell_w, pairs)
    try:
        analyzed = _analyze_segments(inp)
    except AssertionError:
        return None

    if len(analyzed) != len(pairs):
        return None
    if not any(seg["donor"] is not None and seg["donor"]["color"] != seg["color"] for seg in analyzed):
        return None
    if any(seg["donor_tie"] for seg in analyzed):
        return None

    actual = main(inp)
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

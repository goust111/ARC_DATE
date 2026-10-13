from __future__ import annotations

import random
import numpy as np


BG_COLOR = 0
GREY_COLOR = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_OPTIONS = (1, 2, 3)
MODE_WEIGHTS = (4, 4, 2)  # horizontal-only, vertical-only, mixed
SEGMENT_COUNT_OPTIONS = (2, 3)
SEGMENT_MIN = 2
SEGMENT_MAX = 6
GAP_MIN = 1
GAP_MAX = 3
MAX_PATH_LEN = 14
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _run_lengths(line_mask: list[bool]) -> list[int]:
    lengths: list[int] = []
    run = 0
    for flag in line_mask:
        if flag:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    grey_cells = {tuple(map(int, pos)) for pos in np.argwhere(g == GREY_COLOR)}
    colors = [int(color) for color in np.unique(g) if int(color) not in (bg, GREY_COLOR)]

    objects: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        if len(rows) == 1 and len(cols) >= 2:
            row = rows[0]
            start = cols[0]
            end = cols[-1]
            runs = _run_lengths([int(g[row, c]) == color for c in range(start, end + 1)])
            left_marked = start > 0 and int(g[row, start - 1]) == GREY_COLOR
            right_marked = end + 1 < w and int(g[row, end + 1]) == GREY_COLOR
            assert left_marked ^ right_marked, "horizontal path must have exactly one grey endpoint marker"
            direction = -1 if left_marked else 1
            marker = (row, start - 1 if left_marked else end + 1)
            objects.append(
                {
                    "color": color,
                    "orientation": "h",
                    "lane": row,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "direction": direction,
                    "segments": runs,
                    "marker": marker,
                }
            )
        elif len(cols) == 1 and len(rows) >= 2:
            col = cols[0]
            start = rows[0]
            end = rows[-1]
            runs = _run_lengths([int(g[r, col]) == color for r in range(start, end + 1)])
            up_marked = start > 0 and int(g[start - 1, col]) == GREY_COLOR
            down_marked = end + 1 < h and int(g[end + 1, col]) == GREY_COLOR
            assert up_marked ^ down_marked, "vertical path must have exactly one grey endpoint marker"
            direction = -1 if up_marked else 1
            marker = (start - 1 if up_marked else end + 1, col)
            objects.append(
                {
                    "color": color,
                    "orientation": "v",
                    "lane": col,
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "direction": direction,
                    "segments": runs,
                    "marker": marker,
                }
            )
        else:
            raise AssertionError("each color must form one horizontal or vertical family of collinear segments")

    assert len(grey_cells) == len(objects), "grey marker count must match path count"
    assert {obj["marker"] for obj in objects} == grey_cells, "every grey marker must belong to exactly one path"
    for obj in objects:
        assert 2 <= len(obj["segments"]) <= 3, "each color must have 2-3 visible segments"
        assert all(SEGMENT_MIN <= length <= SEGMENT_MAX for length in obj["segments"]), "segment lengths out of range"
    return objects


def _paint_object(out: np.ndarray, obj: dict) -> None:
    color = int(obj["color"])
    start = int(obj["start"])
    end = int(obj["end"])
    length = int(obj["length"])
    direction = int(obj["direction"])

    if obj["orientation"] == "h":
        row = int(obj["lane"])
        out[row, start : end + 1] = color
        anchor = end if direction > 0 else start
        for step in range(1, length + 1):
            out[row, anchor + direction * step] = color
        tip_col = anchor + direction * length
        out[row - 1, tip_col] = color
        out[row + 1, tip_col] = color
    else:
        col = int(obj["lane"])
        out[start : end + 1, col] = color
        anchor = end if direction > 0 else start
        for step in range(1, length + 1):
            out[anchor + direction * step, col] = color
        tip_row = anchor + direction * length
        out[tip_row, col - 1] = color
        out[tip_row, col + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color collinear path fragments, then grow a same-length arrow from the grey-marked endpoint.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    objects = _extract_objects(g, bg)

    out = g.copy()
    out[out == GREY_COLOR] = bg
    for obj in objects:
        _paint_object(out, obj)
    return out


def _sample_segments(max_total: int, segment_count: int) -> tuple[list[int], list[int], int] | None:
    for _ in range(PLACE_ATTEMPTS):
        segments = [random.randint(SEGMENT_MIN, SEGMENT_MAX) for _ in range(segment_count)]
        gaps = [random.randint(GAP_MIN, GAP_MAX) for _ in range(segment_count - 1)]
        total = sum(segments) + sum(gaps)
        if total <= max_total:
            return segments, gaps, total
    return None


def _spec_input_cells(spec: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    pos = int(spec["start"])
    if spec["orientation"] == "h":
        row = int(spec["lane"])
        for idx, seg_len in enumerate(spec["segments"]):
            for col in range(pos, pos + seg_len):
                cells.add((row, col))
            pos += seg_len
            if idx + 1 < len(spec["segments"]):
                pos += int(spec["gaps"][idx])
        marker_col = spec["end"] + 1 if int(spec["direction"]) > 0 else spec["start"] - 1
        cells.add((row, int(marker_col)))
    else:
        col = int(spec["lane"])
        for idx, seg_len in enumerate(spec["segments"]):
            for row in range(pos, pos + seg_len):
                cells.add((row, col))
            pos += seg_len
            if idx + 1 < len(spec["segments"]):
                pos += int(spec["gaps"][idx])
        marker_row = spec["end"] + 1 if int(spec["direction"]) > 0 else spec["start"] - 1
        cells.add((int(marker_row), col))
    return cells


def _spec_output_cells(spec: dict) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    start = int(spec["start"])
    end = int(spec["end"])
    length = int(spec["length"])
    direction = int(spec["direction"])

    if spec["orientation"] == "h":
        row = int(spec["lane"])
        for col in range(start, end + 1):
            cells.add((row, col))
        anchor = end if direction > 0 else start
        for step in range(1, length + 1):
            cells.add((row, anchor + direction * step))
        tip_col = anchor + direction * length
        cells.add((row - 1, tip_col))
        cells.add((row + 1, tip_col))
    else:
        col = int(spec["lane"])
        for row in range(start, end + 1):
            cells.add((row, col))
        anchor = end if direction > 0 else start
        for step in range(1, length + 1):
            cells.add((anchor + direction * step, col))
        tip_row = anchor + direction * length
        cells.add((tip_row, col - 1))
        cells.add((tip_row, col + 1))
    return cells


def _padded_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for row, col in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = row + dr
                cc = col + dc
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((rr, cc))
    return padded


def _sample_spec(h: int, w: int, color: int, orientation: str, blocked: set[tuple[int, int]]) -> dict | None:
    axis = w if orientation == "h" else h
    max_total = min(MAX_PATH_LEN, axis // 2)
    if max_total < 5:
        return None

    for _ in range(PLACE_ATTEMPTS):
        segment_count = random.choice(SEGMENT_COUNT_OPTIONS)
        parts = _sample_segments(max_total=max_total, segment_count=segment_count)
        if parts is None:
            continue
        segments, gaps, total = parts
        direction = random.choice((-1, 1))

        if orientation == "h":
            if h < 3:
                return None
            row = random.randint(1, h - 2)
            if direction > 0:
                lo, hi = 0, w - 2 * total
            else:
                lo, hi = total, w - total
            if lo > hi:
                continue
            start = random.randint(lo, hi)
            spec = {
                "color": color,
                "orientation": "h",
                "lane": row,
                "start": start,
                "end": start + total - 1,
                "length": total,
                "direction": direction,
                "segments": segments,
                "gaps": gaps,
            }
        else:
            if w < 3:
                return None
            col = random.randint(1, w - 2)
            if direction > 0:
                lo, hi = 0, h - 2 * total
            else:
                lo, hi = total, h - total
            if lo > hi:
                continue
            start = random.randint(lo, hi)
            spec = {
                "color": color,
                "orientation": "v",
                "lane": col,
                "start": start,
                "end": start + total - 1,
                "length": total,
                "direction": direction,
                "segments": segments,
                "gaps": gaps,
            }

        cells = _spec_output_cells(spec) | _spec_input_cells(spec)
        if all(0 <= row < h and 0 <= col < w for row, col in cells) and not (cells & blocked):
            return spec
    return None


def _render_pair(h: int, w: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG_COLOR, dtype=int)
    out = np.full((h, w), BG_COLOR, dtype=int)
    for spec in specs:
        pos = int(spec["start"])
        if spec["orientation"] == "h":
            row = int(spec["lane"])
            for idx, seg_len in enumerate(spec["segments"]):
                inp[row, pos : pos + seg_len] = int(spec["color"])
                pos += seg_len
                if idx + 1 < len(spec["segments"]):
                    pos += int(spec["gaps"][idx])
            marker_col = spec["end"] + 1 if int(spec["direction"]) > 0 else spec["start"] - 1
            inp[row, int(marker_col)] = GREY_COLOR
        else:
            col = int(spec["lane"])
            for idx, seg_len in enumerate(spec["segments"]):
                inp[pos : pos + seg_len, col] = int(spec["color"])
                pos += seg_len
                if idx + 1 < len(spec["segments"]):
                    pos += int(spec["gaps"][idx])
            marker_row = spec["end"] + 1 if int(spec["direction"]) > 0 else spec["start"] - 1
            inp[int(marker_row), col] = GREY_COLOR
        _paint_object(out, spec)
    return inp, out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    mode = random.choices(("h", "v", "mixed"), weights=MODE_WEIGHTS, k=1)[0]
    object_count = random.choice(OBJECT_COUNT_OPTIONS)
    if mode == "mixed":
        object_count = max(2, object_count)
        orientations = ["h", "v"] + [random.choice(("h", "v")) for _ in range(object_count - 2)]
        random.shuffle(orientations)
    else:
        orientations = [mode] * object_count

    colors = random.sample(PATH_COLORS, object_count)
    blocked: set[tuple[int, int]] = set()
    specs: list[dict] = []
    for color, orientation in zip(colors, orientations):
        spec = _sample_spec(h, w, color, orientation, blocked)
        if spec is None:
            return None
        cells = _spec_output_cells(spec) | _spec_input_cells(spec)
        blocked |= _padded_cells(cells, h, w)
        specs.append(spec)

    inp, expected = _render_pair(h, w, specs)
    if not np.array_equal(main(inp), expected):
        return None

    parsed = _extract_objects(inp, BG_COLOR)
    if len(parsed) != object_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

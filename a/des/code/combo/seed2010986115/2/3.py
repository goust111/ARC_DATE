from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
MAROON = 9
INPUT_COLOR_POOL = [1, 2, 3, 4, 5, 6, 8]
SEGMENT_COUNT_CHOICES = (2, 3, 4)
COLOR_COUNT_CHOICES = (2, 3)
SEGMENT_LENGTH_RANGE = range(2, 7)
FORWARD_GAP_RANGE = (2, 4)
PERP_GAP_RANGE = (2, 4)
PATCH_SIZE_LIMIT = 14
PATCH_ATTEMPTS = 200
GEN_ATTEMPTS = 300


Coord = tuple[int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _segment_from_coords(coords: np.ndarray, color: int) -> dict[str, object]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    if r0 == r1:
        ordered = sorted(int(c) for c in cols)
        assert ordered == list(range(c0, c1 + 1)), "horizontal segment must be gapless"
        orientation = "h"
        start = (r0, c0)
        end = (r0, c1)
        length = c1 - c0 + 1
    elif c0 == c1:
        ordered = sorted(int(r) for r in rows)
        assert ordered == list(range(r0, r1 + 1)), "vertical segment must be gapless"
        orientation = "v"
        start = (r0, c0)
        end = (r1, c0)
        length = r1 - r0 + 1
    else:
        raise AssertionError("components must be straight line segments")

    assert length >= 2, "segments must be length >= 2"
    return {
        "color": int(color),
        "coords": {tuple(map(int, xy)) for xy in coords},
        "orientation": orientation,
        "start": start,
        "end": end,
        "length": length,
        "bbox": (r0, c0, r1, c1),
    }


def _line_cells(a: Coord, b: Coord) -> set[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        lo, hi = sorted((ac, bc))
        return {(ar, c) for c in range(lo, hi + 1)}
    if ac == bc:
        lo, hi = sorted((ar, br))
        return {(r, ac) for r in range(lo, hi + 1)}
    raise AssertionError("line must be orthogonal")


def _connection(source: dict[str, object], target: dict[str, object]) -> tuple[set[Coord], Coord]:
    sr, sc = source["end"]  # type: ignore[index]
    tr, tc = target["start"]  # type: ignore[index]
    sr, sc, tr, tc = int(sr), int(sc), int(tr), int(tc)

    if source["orientation"] == "h":
        assert tc > sc, "horizontal source must extend rightward toward the target column"
        assert tr != sr, "horizontal source connection must have a real bend"
        bend = (sr, tc)
    else:
        assert tr > sr, "vertical source must extend downward toward the target row"
        assert tc != sc, "vertical source connection must have a real bend"
        bend = (tr, sc)

    cells = _line_cells((sr, sc), bend) | _line_cells(bend, (tr, tc))
    assert bend != (sr, sc) and bend != (tr, tc), "bend must be distinct from both segment endpoints"
    return cells, bend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color line segments, sort each group by length from longest to shortest,
    then connect consecutive segments with same-color L paths whose bends are marked maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    segments_by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = _components_of_color(g, color)
        assert 2 <= len(comps) <= 4, "each color must appear as 2 to 4 segments"
        for comp in comps:
            segments_by_color[color].append(_segment_from_coords(comp, color))

    for color, segments in segments_by_color.items():
        ordered = sorted(
            segments,
            key=lambda seg: (
                -int(seg["length"]),
                int(seg["start"][0]),  # type: ignore[index]
                int(seg["start"][1]),  # type: ignore[index]
                0 if seg["orientation"] == "h" else 1,
            ),
        )
        lengths = [int(seg["length"]) for seg in ordered]
        assert len(lengths) == len(set(lengths)), "same-color segment lengths must be distinct"

        for source, target in zip(ordered, ordered[1:]):
            path_cells, bend = _connection(source, target)
            source_end = source["end"]  # type: ignore[index]
            target_start = target["start"]  # type: ignore[index]
            for cell in path_cells:
                if cell in {source_end, target_start}:
                    continue
                r, c = cell
                assert out[r, c] == BG, "paths may only pass through empty background"
                out[r, c] = int(color)
            br, bc = bend
            assert out[br, bc] == int(color), "bend must sit on the just-painted path"
            out[br, bc] = MAROON

    return out


def _segment_cells(start: Coord, orientation: str, length: int) -> set[Coord]:
    r, c = start
    if orientation == "h":
        return {(r, c + dc) for dc in range(length)}
    return {(r + dr, c) for dr in range(length)}


def _build_segment(start: Coord, orientation: str, length: int, color: int) -> dict[str, object]:
    cells = _segment_cells(start, orientation, length)
    if orientation == "h":
        end = (start[0], start[1] + length - 1)
    else:
        end = (start[0] + length - 1, start[1])
    return {
        "color": color,
        "coords": cells,
        "orientation": orientation,
        "start": start,
        "end": end,
        "length": length,
    }


def _touches_segment(candidate: set[Coord], existing: set[Coord]) -> bool:
    for r, c in candidate:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in existing:
                    return True
    return False


def _render_local_patch(
    color: int,
    segments: list[dict[str, object]],
    paths: list[tuple[set[Coord], Coord]],
) -> tuple[np.ndarray, np.ndarray]:
    output_cells: set[Coord] = set()
    for segment in segments:
        output_cells |= segment["coords"]  # type: ignore[operator]
    for path_cells, _bend in paths:
        output_cells |= path_cells

    rows = [r for r, _c in output_cells]
    cols = [c for _r, c in output_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    pad = random.randint(1, 2)
    h = r1 - r0 + 1 + 2 * pad
    w = c1 - c0 + 1 + 2 * pad
    if max(h, w) > PATCH_SIZE_LIMIT:
        raise ValueError("patch grew too large")

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    def shift(cell: Coord) -> Coord:
        return (cell[0] - r0 + pad, cell[1] - c0 + pad)

    for segment in segments:
        for cell in segment["coords"]:  # type: ignore[assignment]
            rr, cc = shift(cell)
            inp[rr, cc] = color
            out[rr, cc] = color

    for path_cells, bend in paths:
        for cell in path_cells:
            rr, cc = shift(cell)
            out[rr, cc] = color
        br, bc = shift(bend)
        out[br, bc] = MAROON

    return inp, out


def _generate_color_patch(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_ATTEMPTS):
        segment_count = random.choice(SEGMENT_COUNT_CHOICES)
        lengths = sorted(random.sample(list(SEGMENT_LENGTH_RANGE), segment_count), reverse=True)

        first = _build_segment((0, 0), random.choice(("h", "v")), lengths[0], color)
        segments = [first]
        paths: list[tuple[set[Coord], Coord]] = []
        occupied_output = set(first["coords"])  # type: ignore[arg-type]
        segment_cells = set(first["coords"])  # type: ignore[arg-type]

        ok = True
        for length in lengths[1:]:
            source = segments[-1]
            placed = False
            for _ in range(PATCH_ATTEMPTS):
                orientation = random.choice(("h", "v"))
                sr, sc = source["end"]  # type: ignore[index]
                sr, sc = int(sr), int(sc)
                forward_gap = random.randint(*FORWARD_GAP_RANGE)
                perp_gap = random.randint(*PERP_GAP_RANGE)
                sign = random.choice((-1, 1))

                if source["orientation"] == "h":
                    start = (sr + sign * perp_gap, sc + forward_gap)
                else:
                    start = (sr + forward_gap, sc + sign * perp_gap)

                target = _build_segment(start, orientation, length, color)
                try:
                    path_cells, bend = _connection(source, target)
                except AssertionError:
                    continue

                if target["coords"] & occupied_output:  # type: ignore[operator]
                    continue
                if _touches_segment(target["coords"], segment_cells):  # type: ignore[arg-type]
                    continue

                allowed_overlap = {source["end"], target["start"]}  # type: ignore[arg-type]
                if any(cell in occupied_output and cell not in allowed_overlap for cell in path_cells):
                    continue

                segments.append(target)
                paths.append((path_cells, bend))
                occupied_output |= target["coords"]  # type: ignore[operator]
                occupied_output |= path_cells
                segment_cells |= target["coords"]  # type: ignore[operator]
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        try:
            inp, out = _render_local_patch(color, segments, paths)
        except ValueError:
            continue

        try:
            if np.array_equal(main(inp), out):
                return inp, out
        except AssertionError:
            continue

    return None


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.nonzero(patch)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        canvas[top + rr, left + cc] = int(patch[rr, cc])


def _compose_scene(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    count = len(patches)
    order = list(range(count))
    random.shuffle(order)
    patches = [patches[i] for i in order]

    n_cols = random.randint(1, min(2, count))
    n_rows = (count + n_cols - 1) // n_cols

    row_slots: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for row in range(n_rows):
        row_slots.append(patches[row * n_cols : (row + 1) * n_cols])

    col_widths: list[int] = []
    for col in range(n_cols):
        col_patches = [row[col] for row in row_slots if col < len(row)]
        max_w = max(patch[0].shape[1] for patch in col_patches)
        col_widths.append(max_w + random.randint(0, 3))

    row_heights: list[int] = []
    for row in row_slots:
        max_h = max(patch[0].shape[0] for patch in row)
        row_heights.append(max_h + random.randint(0, 3))

    row_gaps = [random.randint(2, 4) for _ in range(max(0, n_rows - 1))]
    col_gaps = [random.randint(2, 4) for _ in range(max(0, n_cols - 1))]
    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    height = top_margin + sum(row_heights) + sum(row_gaps) + bottom_margin
    width = left_margin + sum(col_widths) + sum(col_gaps) + right_margin
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    row_origin = top_margin
    for row_idx, row in enumerate(row_slots):
        col_origin = left_margin
        for col_idx, (inp_patch, out_patch) in enumerate(row):
            max_r_off = row_heights[row_idx] - inp_patch.shape[0]
            max_c_off = col_widths[col_idx] - inp_patch.shape[1]
            r_off = random.randint(0, max_r_off)
            c_off = random.randint(0, max_c_off)
            _paste_patch(inp, inp_patch, row_origin + r_off, col_origin + c_off)
            _paste_patch(out, out_patch, row_origin + r_off, col_origin + c_off)
            col_origin += col_widths[col_idx]
            if col_idx + 1 < n_cols:
                col_origin += col_gaps[col_idx]
        row_origin += row_heights[row_idx]
        if row_idx + 1 < n_rows:
            row_origin += row_gaps[row_idx]

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        color_count = random.choice(COLOR_COUNT_CHOICES)
        colors = random.sample(INPUT_COLOR_POOL, color_count)
        patches: list[tuple[np.ndarray, np.ndarray]] = []
        failed = False
        for color in colors:
            patch = _generate_color_patch(color)
            if patch is None:
                failed = True
                break
            patches.append(patch)
        if failed:
            continue

        inp, out = _compose_scene(patches)
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid input")

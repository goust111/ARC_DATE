from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r0 for r0, _ in coords]
            cols = [c0 for _, c0 in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _endpoint_cells(coord_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    for r, c in coord_set:
        degree = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        assert degree <= 2, "path fragments must be non-branching"
        if degree == 1:
            endpoints.append((int(r), int(c)))
    endpoints.sort()
    return endpoints


def _fragment_span(fragment: dict) -> int:
    r0, c0, r1, c1 = fragment["bbox"]
    return int((r1 - r0) + (c1 - c0))


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int], dict[int, list[dict]]]:
    comps = _same_color_components(grid)
    fragments: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        if comp["area"] == 1:
            markers[comp["coords"][0]] = int(comp["color"])
            continue
        assert comp["area"] >= 3, "fragments must have length at least three"
        endpoints = _endpoint_cells(comp["coord_set"])
        assert len(endpoints) == 2, "each fragment must have exactly two endpoints"
        fragments.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "coord_set": comp["coord_set"],
                "area": int(comp["area"]),
                "bbox": comp["bbox"],
                "endpoints": endpoints,
            }
        )

    assert fragments, "expected at least one path fragment"
    assert markers, "expected endpoint markers"

    pairs: dict[int, list[dict]] = defaultdict(list)
    used_markers: set[tuple[int, int]] = set()
    for frag_id, fragment in enumerate(fragments):
        for endpoint in fragment["endpoints"]:
            adjacent = []
            for nr, nc in _neighbors4(*endpoint):
                if (nr, nc) in markers and int(markers[(nr, nc)]) != int(fragment["color"]):
                    adjacent.append((int(nr), int(nc)))
            assert len(adjacent) == 1, "each endpoint must have exactly one adjacent marker"
            marker_pos = adjacent[0]
            assert marker_pos not in used_markers, "marker pixels must belong to one endpoint only"
            used_markers.add(marker_pos)
            pairs[int(markers[marker_pos])].append(
                {
                    "fragment_id": int(frag_id),
                    "endpoint": endpoint,
                    "marker_pos": marker_pos,
                    "marker_color": int(markers[marker_pos]),
                }
            )

    assert len(used_markers) == len(markers), "all markers must be attached to endpoints"
    for marker_color, recs in pairs.items():
        assert len(recs) == 2, f"marker color {marker_color} must appear exactly twice"
        p1 = recs[0]["endpoint"]
        p2 = recs[1]["endpoint"]
        assert p1[0] == p2[0] or p1[1] == p2[1], "matched endpoints must share a row or column"

    return fragments, markers, pairs


def _connector_color(fragment_a: dict, fragment_b: dict) -> int:
    if int(fragment_a["area"]) > int(fragment_b["area"]):
        return int(fragment_a["color"])
    if int(fragment_b["area"]) > int(fragment_a["area"]):
        return int(fragment_b["color"])
    if int(fragment_a["color"]) == int(fragment_b["color"]):
        return int(fragment_a["color"])
    span_a = _fragment_span(fragment_a)
    span_b = _fragment_span(fragment_b)
    if span_a != span_b:
        return int(fragment_a["color"] if span_a > span_b else fragment_b["color"])
    return int(fragment_a["color"])


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        c_lo, c_hi = sorted((c1, c2))
        grid[r1, c_lo : c_hi + 1] = int(color)
        return
    assert c1 == c2
    r_lo, r_hi = sorted((r1, r2))
    grid[r_lo : r_hi + 1, c1] = int(color)


def _fill_enclosed_black(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(out[r, c]) == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(out[r, c]) == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BG and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    mask = (out == BG) & (~outside)
    out[mask] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect marker-paired path endpoints and fill any newly enclosed black regions with grey.

    Input structure:
    - The background is black.
    - Several one-pixel-wide monochrome path fragments are scattered on the grid.
    - Each fragment is a simple 4-connected path with two endpoints.
    - A single non-black marker pixel sits in a 4-neighbor black cell beside each endpoint, and each marker color appears exactly twice.

    Rule:
    - Treat equal-colored endpoint markers as instructions to connect those two endpoints.
    - Draw the missing horizontal or vertical segment between each matched endpoint pair.
    - Color each connector with the color of the larger of the two incident fragments.
    - Remove all marker pixels by turning them black, then flood-fill every black region that is completely surrounded by the finished paths with grey.

    Key insight:
    - The isolated colored pixels are not part of the paths; they are pairing labels that identify which endpoints must be linked before checking for enclosed regions.
    """
    g = np.asarray(grid, dtype=int)
    fragments, markers, pairs = _analyze_scene(g)

    out = np.array(g, dtype=int, copy=True)
    for recs in pairs.values():
        frag_a = fragments[int(recs[0]["fragment_id"])]
        frag_b = fragments[int(recs[1]["fragment_id"])]
        color = _connector_color(frag_a, frag_b)
        _draw_segment(out, recs[0]["endpoint"], recs[1]["endpoint"], color)

    for r, c in markers:
        out[r, c] = BG

    return _fill_enclosed_black(out)


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        r1, c1 = points[i]
        r2, c2 = points[i + 1]
        assert r1 == r2 or c1 == c2, "polyline segments must be axis-aligned"
        segment: list[tuple[int, int]] = []
        if r1 == r2:
            step = 1 if c2 >= c1 else -1
            for cc in range(c1, c2 + step, step):
                segment.append((int(r1), int(cc)))
        else:
            step = 1 if r2 >= r1 else -1
            for rr in range(r1, r2 + step, step):
                segment.append((int(rr), int(c1)))
        if i > 0:
            segment = segment[1:]
        cells.extend(segment)
    assert len(cells) == len(set(cells)), "fragment templates must be simple paths"
    return cells


def _horizontal_fragment(row: int, left: int, width: int, kind: str, bump_dir: int) -> list[tuple[int, int]]:
    right = left + width - 1
    if kind != "bump" or width < 4:
        return _polyline_cells([(row, left), (row, right)])
    c_a = random.randint(left + 1, right - 2)
    c_b = random.randint(c_a + 1, right - 1)
    return _polyline_cells(
        [
            (row, left),
            (row, c_a),
            (row + bump_dir, c_a),
            (row + bump_dir, c_b),
            (row, c_b),
            (row, right),
        ]
    )


def _take_color(available: list[int]) -> int:
    assert available, "ran out of colors while generating a scene"
    return int(available.pop())


def _build_component(available: list[int]) -> np.ndarray:
    width = random.randint(3, 6)
    gap = random.randint(3, 6)
    top_kind = random.choice(["line", "bump"])
    bottom_kind = random.choice(["line", "bump"])
    if width < 4:
        top_kind = "line"
        bottom_kind = "line"

    top_row = 1
    bottom_row = top_row + gap
    left = 1
    right = left + width - 1
    height = bottom_row + 1 + (1 if bottom_kind == "bump" else 0)
    width_total = right + 2

    top_cells = _horizontal_fragment(top_row, left, width, top_kind, -1)
    bottom_cells = _horizontal_fragment(bottom_row, left, width, bottom_kind, 1)

    use_two_path_colors = len(top_cells) != len(bottom_cells) and random.random() < 0.65
    top_color = _take_color(available)
    bottom_color = _take_color(available) if use_two_path_colors else top_color
    left_marker = _take_color(available)
    right_marker = _take_color(available)

    comp = np.zeros((height, width_total), dtype=int)
    for r, c in top_cells:
        comp[r, c] = int(top_color)
    for r, c in bottom_cells:
        comp[r, c] = int(bottom_color)

    comp[top_row, left - 1] = int(left_marker)
    comp[bottom_row, left - 1] = int(left_marker)
    comp[top_row, right + 1] = int(right_marker)
    comp[bottom_row, right + 1] = int(right_marker)

    return np.rot90(comp, random.randrange(4))


def _can_place(canvas: np.ndarray, comp: np.ndarray, top: int, left: int) -> bool:
    h, w = canvas.shape
    ch, cw = comp.shape
    if top < 0 or left < 0 or top + ch > h or left + cw > w:
        return False
    coords = np.argwhere(comp != BG)
    for rr0, cc0 in coords:
        rr = int(top + rr0)
        cc = int(left + cc0)
        if int(canvas[rr, cc]) != BG:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and int(canvas[nr, nc]) != BG:
                    return False
    return True


def _paste_component(canvas: np.ndarray, comp: np.ndarray, top: int, left: int) -> None:
    coords = np.argwhere(comp != BG)
    for rr0, cc0 in coords:
        canvas[top + int(rr0), left + int(cc0)] = int(comp[int(rr0), int(cc0)])


def _has_connector_gaps(grid: np.ndarray, pairs: dict[int, list[dict]]) -> bool:
    for recs in pairs.values():
        (r1, c1) = recs[0]["endpoint"]
        (r2, c2) = recs[1]["endpoint"]
        if r1 == r2:
            c_lo, c_hi = sorted((c1, c2))
            if not np.all(grid[r1, c_lo + 1 : c_hi] == BG):
                return False
        else:
            r_lo, r_hi = sorted((r1, r2))
            if not np.all(grid[r_lo + 1 : r_hi, c1] == BG):
                return False
    return True


def _generate_candidate() -> np.ndarray | None:
    n_components = random.choice([1, 2])
    available = random.sample(PALETTE, len(PALETTE))
    components: list[np.ndarray] = []
    try:
        for _ in range(n_components):
            components.append(_build_component(available))
    except AssertionError:
        return None

    min_h = max(max(comp.shape[0] for comp in components) + 2, 10)
    min_w = max(max(comp.shape[1] for comp in components) + 2, 10)
    if min_h > 30 or min_w > 30:
        return None

    for _attempt in range(160):
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        canvas = np.zeros((h, w), dtype=int)
        placed = True
        for comp in sorted(components, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            ch, cw = comp.shape
            row_lo = 1 if h - ch >= 2 else 0
            row_hi = h - ch - 1 if h - ch >= 2 else h - ch
            col_lo = 1 if w - cw >= 2 else 0
            col_hi = w - cw - 1 if w - cw >= 2 else w - cw
            slots = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
            random.shuffle(slots)
            spot = None
            for top, left in slots[:200]:
                if _can_place(canvas, comp, top, left):
                    spot = (int(top), int(left))
                    break
            if spot is None:
                placed = False
                break
            _paste_component(canvas, comp, spot[0], spot[1])
        if not placed:
            continue

        try:
            fragments, _markers, pairs = _analyze_scene(canvas)
        except AssertionError:
            continue

        if not all(3 <= int(fragment["area"]) <= 8 for fragment in fragments):
            continue
        if not _has_connector_gaps(canvas, pairs):
            continue

        out = main(canvas)
        if not np.any(out == GREY):
            continue
        return canvas

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

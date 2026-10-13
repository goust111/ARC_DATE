from __future__ import annotations

import random

import numpy as np


PALETTE = (1, 2, 3, 4, 7)
GEN_RETRIES = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_by_color(grid: np.ndarray) -> list[dict]:
    """Extract 4-connected monochrome components, following arc_utils (r, c) conventions."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
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
                }
            )
    return comps


def _parse_regions(grid: np.ndarray) -> list[dict]:
    """Recover the tiled rectangles; a marked region is a solid rectangle with one black hole."""
    g = np.asarray(grid, dtype=int)
    comps = _components_by_color(g)
    cover = np.zeros_like(g, dtype=bool)
    regions: list[dict] = []

    for comp in comps:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        ph, pw = patch.shape
        assert ph != pw, "regions must be non-square"

        present = set(map(int, np.unique(patch)))
        assert present.issubset({0, color}), "a region bbox may only contain its fill color and a marker"

        marker_cells = np.argwhere(patch == 0)
        if len(marker_cells) == 0:
            assert len(comp["coords"]) == ph * pw
            marker = None
        else:
            assert len(marker_cells) == 1, "each marked region must contain exactly one marker"
            assert len(comp["coords"]) == ph * pw - 1
            mr, mc = map(int, marker_cells[0])
            assert 0 < mr < ph - 1 and 0 < mc < pw - 1, "marker must be contained inside the region"
            marker = (r0 + mr, c0 + mc)

        bbox_mask = np.zeros_like(g, dtype=bool)
        bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert not np.any(cover & bbox_mask), "region bboxes must not overlap"
        cover |= bbox_mask

        regions.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "h": ph,
                "w": pw,
                "area": ph * pw,
                "marker": marker,
            }
        )

    assert np.all(cover), "rectangular regions must tile the full canvas without gaps"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each marker region into a full-grid ray, using region aspect ratio for direction and area priority for crossings.

    Input structure:
    - The grid is fully tiled by non-overlapping solid-color rectangular regions, with no black gaps.
    - Some regions contain exactly one black marker pixel strictly inside the rectangle.
    - Every marked region is non-square, so its width and height uniquely determine an orientation class.

    Rule:
    - Recover each rectangle from its colored component bbox; if it contains one black hole, it is a marked source region.
    - A marked wide region (width > height) emits a vertical ray along the marker column; a marked tall region (height > width) emits a horizontal ray along the marker row.
    - Draw all rays in increasing order of source-region area, using the source color, so larger source regions overwrite smaller ones at crossings and the marker itself is filled back in.

    Key insight:
    - The region geometry, not the marker location, encodes the ray direction; once rays span the whole canvas, intersections are resolved by comparing the source rectangle areas.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    marked = []

    for region in _parse_regions(g):
        marker = region["marker"]
        if marker is None:
            continue
        h = int(region["h"])
        w = int(region["w"])
        if w > h:
            axis = "v"
            line_index = int(marker[1])
        else:
            axis = "h"
            line_index = int(marker[0])
        marked.append({**region, "axis": axis, "line_index": line_index})

    for region in sorted(marked, key=lambda reg: (int(reg["area"]), tuple(reg["bbox"]))):
        color = int(region["color"])
        if region["axis"] == "v":
            out[:, int(region["line_index"])] = color
        else:
            out[int(region["line_index"]), :] = color
    return out


def _fill_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _candidate_layout() -> tuple[np.ndarray, dict] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)

    h_top = random.randint(5, min(10, h - 9))
    w_b = random.randint(3, min(h_top - 1, w - 9))
    w_a = w - w_b
    h_bottom = h - h_top

    w_c = random.randint(3, min(h_bottom - 1, w - 8))
    right_w = w - w_c
    h_d = random.randint(3, min(right_w - 1, h_bottom - 3))
    h_e = h_bottom - h_d

    rects = {
        "A": (0, 0, h_top - 1, w_a - 1),
        "B": (0, w_a, h_top - 1, w - 1),
        "C": (h_top, 0, h - 1, w_c - 1),
        "D": (h_top, w_c, h_top + h_d - 1, w - 1),
        "E": (h_top + h_d, w_c, h - 1, w - 1),
    }
    dims = {
        name: (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        for name, bbox in rects.items()
    }

    if not (
        dims["A"][1] > dims["A"][0]
        and dims["B"][0] > dims["B"][1]
        and dims["C"][0] > dims["C"][1]
        and dims["D"][1] > dims["D"][0]
    ):
        return None
    if any(min(hh, ww) < 3 or hh == ww for hh, ww in dims.values()):
        return None

    areas = {name: dims[name][0] * dims[name][1] for name in "ABCD"}
    if len(set(areas.values())) != 4:
        return None
    if not (areas["A"] > areas["B"] and areas["C"] > areas["D"]):
        return None

    colors = random.sample(PALETTE, k=5)
    grid = np.zeros((h, w), dtype=int)
    for name, color in zip(("A", "B", "C", "D", "E"), colors):
        _fill_rect(grid, rects[name], color)

    # Wide sources use separated columns; tall sources use separated rows.
    col_a = random.randint(1, min(w_c - 1, w_a - 2))
    col_d = random.randint(w_a + 1, w - 2)
    row_b = random.randint(1, h_top - 2)
    row_c = random.randint(h_top + h_d + 1, h - 2)

    markers = {
        "A": (random.randint(1, h_top - 2), col_a),
        "B": (row_b, random.randint(w_a + 1, w - 2)),
        "C": (row_c, random.randint(1, w_c - 2)),
        "D": (random.randint(h_top + 1, h_top + h_d - 2), col_d),
    }
    for r, c in markers.values():
        grid[r, c] = 0

    meta = {"rects": rects, "areas": areas, "markers": markers}
    return grid, meta


def _augment(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    return np.array(out, dtype=int, copy=True)


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        regions = _parse_regions(grid)
    except AssertionError:
        return False

    if len(regions) != 5:
        return False

    marked = [region for region in regions if region["marker"] is not None]
    if len(marked) != 4:
        return False
    if len({int(region["area"]) for region in marked}) != len(marked):
        return False

    wide = [region for region in marked if int(region["w"]) > int(region["h"])]
    tall = [region for region in marked if int(region["h"]) > int(region["w"])]
    if len(wide) < 2 or len(tall) < 2:
        return False

    if len({int(region["marker"][1]) for region in wide}) != len(wide):
        return False
    if len({int(region["marker"][0]) for region in tall}) != len(tall):
        return False

    wide_wins = any(int(w["area"]) > int(t["area"]) for w in wide for t in tall)
    tall_wins = any(int(t["area"]) > int(w["area"]) for w in wide for t in tall)
    if not (wide_wins and tall_wins):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    changed = int(np.count_nonzero(out != grid))
    if changed < min(grid.shape):
        return False

    saw_wide_winner = False
    saw_tall_winner = False
    for w_region in wide:
        for t_region in tall:
            rr = int(t_region["marker"][0])
            cc = int(w_region["marker"][1])
            if int(w_region["area"]) > int(t_region["area"]):
                expected = int(w_region["color"])
                saw_wide_winner = True
            else:
                expected = int(t_region["color"])
                saw_tall_winner = True
            if int(out[rr, cc]) != expected:
                return False
    return saw_wide_winner and saw_tall_winner


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _candidate_layout()
        if candidate is None:
            continue
        grid, _meta = candidate
        grid = _augment(grid)
        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid ray-area mosaic input")

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BACKGROUND = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
YELLOW = Color.YELLOW
PINK = Color.PINK
ORANGE = Color.ORANGE
PERIOD = 4

GEN_ATTEMPTS = 400
GEN_HEIGHT_RANGE = (18, 25)
GEN_WIDTH_RANGE = (18, 27)
GEN_TARGET_COUNT_RANGE = (4, 6)
GEN_TARGET_HEIGHT_RANGE = (4, 9)
GEN_QUAL_WIDTH_RANGE = (4, 9)
GEN_NONQUAL_WIDTH_RANGE = (3, 3)
GEN_MIN_QUALIFIED = 2
GEN_MIN_NONQUALIFIED = 1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _edge_thickness(pattern: np.ndarray, axis: int, from_start: bool, color: int) -> int:
    lines = pattern if axis == 0 else pattern.T
    seq = lines if from_start else lines[::-1]
    count = 0
    for line in seq:
        if np.all(line == color):
            count += 1
        else:
            break
    return count


def _resize_nn(block: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    src_h, src_w = block.shape
    row_idx = np.floor(np.arange(h) * src_h / h).astype(int)
    col_idx = np.floor(np.arange(w) * src_w / w).astype(int)
    return block[np.ix_(row_idx, col_idx)]


def _stretch_exemplar(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    outer = int(pattern[0, 0])
    top = _edge_thickness(pattern, axis=0, from_start=True, color=outer)
    bottom = _edge_thickness(pattern, axis=0, from_start=False, color=outer)
    left = _edge_thickness(pattern, axis=1, from_start=True, color=outer)
    right = _edge_thickness(pattern, axis=1, from_start=False, color=outer)

    target_h, target_w = map(int, shape)
    assert target_h > top + bottom and target_w > left + right, "target too small for fixed border"

    out = np.full((target_h, target_w), outer, dtype=int)
    inner = pattern[top : pattern.shape[0] - bottom, left : pattern.shape[1] - right]
    scaled = _resize_nn(inner, (target_h - top - bottom, target_w - left - right))
    out[top : target_h - bottom, left : target_w - right] = scaled
    return out


def _yellow_frame(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    patch = np.full((h, w), ORANGE, dtype=int)
    if h > 2 and w > 2:
        patch[1:-1, 1:-1] = YELLOW
    return patch


def _parse_scene(grid: np.ndarray):
    yellow_comps = find_components(grid, bg=BACKGROUND, connectivity=4, colors=[YELLOW])
    orange_comps = find_components(grid, bg=BACKGROUND, connectivity=4, colors=[ORANGE])
    markers: list[int] = []
    for comp in yellow_comps:
        bbox = comp["bbox"]
        assert comp["area"] == 1 and bbox[0] == 0, "yellow markers must be isolated top-row pixels"
        markers.append(int(bbox[1]))

    markers.sort()
    assert len(markers) >= 3, "expected periodic column markers"
    diffs = [b - a for a, b in zip(markers, markers[1:])]
    assert all(d == diffs[0] for d in diffs), "marker columns must have one period"

    exemplars = []
    targets = []
    for comp in orange_comps:
        bbox = comp["bbox"]
        patch = crop_bbox(grid, bbox)
        assert np.all(patch != BACKGROUND), "orange rectangle bbox must not contain background"
        if int(comp["area"]) == _bbox_area(bbox):
            assert len(np.unique(patch)) == 1, "solid orange targets must be monochromatic"
            targets.append((bbox, patch.shape))
        else:
            exemplars.append((bbox, patch))

    assert len(exemplars) == 1, "expected one multicolor exemplar rectangle"
    exemplar_bbox, exemplar = exemplars[0]
    outer = int(exemplar[0, 0])
    assert outer == ORANGE
    assert np.all(exemplar[0, :] == outer)
    assert np.all(exemplar[-1, :] == outer)
    assert np.all(exemplar[:, 0] == outer)
    assert np.all(exemplar[:, -1] == outer)

    return markers, exemplar_bbox, exemplar, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the multicolor exemplar into periodic-column targets, then recolor only the periodic interior hits.

    Input structure
    - The grid has a black background, one multicolor orange-bordered exemplar rectangle,
      several solid orange target rectangles, and yellow marker pixels on the top row.
    - The top-row yellow pixels mark equally spaced periodic columns.
    - The exemplar interior contains horizontal colored bands inside a solid orange border.

    Rule
    - Identify the unique multicolor rectangle as the exemplar and the top-row yellow single pixels as the periodic-column markers.
    - For each solid orange target rectangle, check whether its column span intersects any marked periodic column:
      intersecting targets receive an elastic stretch of the exemplar that keeps the orange border fixed and resizes only the interior;
      non-intersecting targets become orange frames with solid yellow interiors.
    - After stretching, in every intersecting target, replace the pixels lying on periodic columns by pink whenever those pixels are neither black background nor orange border.
    - Finally erase the top-row yellow marker pixels back to black and leave the exemplar unchanged.

    Key insight
    - The same periodic columns drive two distinct stages: they first decide which rectangles inherit the exemplar and then slice those stretched interiors into pink vertical accents.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    periodic_cols, _exemplar_bbox, exemplar, targets = _parse_scene(g)

    out = g.copy()
    out[0, periodic_cols] = BACKGROUND

    qualified: list[tuple[int, int, int, int]] = []
    for bbox, shape in targets:
        r0, c0, r1, c1 = bbox
        hits = [c for c in periodic_cols if c0 <= c <= c1]
        if hits:
            patch = _stretch_exemplar(exemplar, shape)
            paste(out, patch, r0, c0)
            qualified.append(bbox)
        else:
            paste(out, _yellow_frame(shape), r0, c0)

    for r0, c0, r1, c1 in qualified:
        for c in periodic_cols:
            if c0 <= c <= c1:
                col = out[r0 : r1 + 1, c]
                mask = (col != BACKGROUND) & (col != ORANGE)
                col[mask] = PINK
                out[r0 : r1 + 1, c] = col

    return out


def _make_exemplar() -> np.ndarray:
    inner_h = random.randint(2, 4)
    inner_w = random.randint(3, 5)
    split = random.randint(1, inner_h - 1)
    colors = [RED, BLUE]
    if random.random() < 0.5:
        colors.reverse()

    patch = np.full((inner_h + 2, inner_w + 2), ORANGE, dtype=int)
    patch[1 : 1 + split, 1:-1] = colors[0]
    patch[1 + split : -1, 1:-1] = colors[1]
    return patch


def _rect_patch(h: int, w: int) -> np.ndarray:
    return np.full((h, w), ORANGE, dtype=int)


def _has_inner_periodic(c0: int, w: int, cols: list[int]) -> bool:
    c1 = c0 + w - 1
    return any(c0 + 1 <= c <= c1 - 1 for c in cols)


def _can_place(grid: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    h0, w0 = grid.shape
    if r0 < 1 or c0 < 0 or r0 + h > h0 or c0 + w > w0:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h0, r0 + h + padding)
    cc1 = min(w0, c0 + w + padding)
    return np.all(grid[rr0:rr1, cc0:cc1] == BACKGROUND)


def _candidate_positions(grid: np.ndarray, h: int, w: int, predicate) -> list[tuple[int, int]]:
    out = []
    for r0 in range(1, grid.shape[0] - h + 1):
        for c0 in range(0, grid.shape[1] - w + 1):
            if _can_place(grid, r0, c0, h, w) and predicate(r0, c0):
                out.append((r0, c0))
    random.shuffle(out)
    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, predicate) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = _candidate_positions(grid, h, w, predicate)
    if not candidates:
        return None
    r0, c0 = candidates[0]
    paste(grid, patch, r0, c0)
    return r0, c0


def _scene_summary(grid: np.ndarray) -> dict:
    cols, exemplar_bbox, exemplar, targets = _parse_scene(grid)
    summary = {
        "periodic_cols": cols,
        "exemplar_bbox": exemplar_bbox,
        "exemplar": exemplar,
        "qualified": [],
        "nonqualified": [],
    }
    for bbox, shape in targets:
        r0, c0, r1, c1 = bbox
        hits = [c for c in cols if c0 <= c <= c1]
        item = {"bbox": bbox, "shape": shape, "hits": hits}
        if hits:
            summary["qualified"].append(item)
        else:
            summary["nonqualified"].append(item)
    return summary


def _validate_generated_scene(inp: np.ndarray, out: np.ndarray) -> None:
    summary = _scene_summary(inp)
    assert len(summary["qualified"]) >= GEN_MIN_QUALIFIED
    assert len(summary["nonqualified"]) >= GEN_MIN_NONQUALIFIED
    assert any(_has_inner_periodic(item["bbox"][1], item["shape"][1], summary["periodic_cols"]) for item in summary["qualified"])

    for c in summary["periodic_cols"]:
        assert int(out[0, c]) == BACKGROUND

    has_magenta = False
    for item in summary["qualified"]:
        r0, c0, r1, c1 = item["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        expected = _stretch_exemplar(summary["exemplar"], item["shape"])
        for c in item["hits"]:
            local_c = c - c0
            col = expected[:, local_c]
            mask = (col != BACKGROUND) & (col != ORANGE)
            col[mask] = PINK
            expected[:, local_c] = col
        assert np.array_equal(patch, expected)
        has_magenta |= bool(np.any(patch == PINK))

    for item in summary["nonqualified"]:
        r0, c0, r1, c1 = item["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        expected = _yellow_frame(item["shape"])
        assert np.array_equal(patch, expected)

    assert has_magenta, "at least one qualified target must visibly recolor on a periodic interior column"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(*GEN_HEIGHT_RANGE)
        w = random.randint(*GEN_WIDTH_RANGE)
        grid = np.zeros((h, w), dtype=int)
        periodic_cols = list(range(0, w, PERIOD))
        grid[0, periodic_cols] = YELLOW

        exemplar = _make_exemplar()
        ex_pos = _place_patch(grid, exemplar, lambda _r0, c0: c0 <= max(2, w // 5))
        if ex_pos is None:
            continue

        target_count = random.randint(*GEN_TARGET_COUNT_RANGE)
        nonqual_count = random.randint(GEN_MIN_NONQUALIFIED, min(2, target_count - GEN_MIN_QUALIFIED))
        qual_count = target_count - nonqual_count

        placed = []
        ok = True

        for _ in range(qual_count):
            rect_h = random.randint(*GEN_TARGET_HEIGHT_RANGE)
            rect_w = random.randint(*GEN_QUAL_WIDTH_RANGE)
            patch = _rect_patch(rect_h, rect_w)
            pos = _place_patch(
                grid,
                patch,
                lambda _r0, c0, rect_w=rect_w: _has_inner_periodic(c0, rect_w, periodic_cols),
            )
            if pos is None:
                ok = False
                break
            placed.append((pos[0], pos[1], rect_h, rect_w, True))

        if not ok:
            continue

        for _ in range(nonqual_count):
            rect_h = random.randint(*GEN_TARGET_HEIGHT_RANGE)
            rect_w = random.randint(*GEN_NONQUAL_WIDTH_RANGE)
            patch = _rect_patch(rect_h, rect_w)
            pos = _place_patch(
                grid,
                patch,
                lambda _r0, c0, rect_w=rect_w: not any(c0 <= c <= c0 + rect_w - 1 for c in periodic_cols),
            )
            if pos is None:
                ok = False
                break
            placed.append((pos[0], pos[1], rect_h, rect_w, False))

        if not ok:
            continue

        if not any(flag for *_rest, flag in placed) or all(flag for *_rest, flag in placed):
            continue

        out = main(grid)
        _validate_generated_scene(grid, out)
        return grid

    raise RuntimeError("failed to generate a valid elastic periodic-column scene")

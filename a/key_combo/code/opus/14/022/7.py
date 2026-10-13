from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
GRAY = 5
MARKER_COLORS = (1, 2, 3)
BLOCK_SIZE = 5
MAX_RADIUS = 3
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    target = int(color)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != target:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == target:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    blocks = []
    for comp in _components_of_color(grid, GRAY):
        r0, c0, r1, c1 = _bbox(comp)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert crop.shape == (BLOCK_SIZE, BLOCK_SIZE), "each gray object must occupy a 5x5 bbox"
        assert np.all(crop[0, :] == GRAY) and np.all(crop[-1, :] == GRAY)
        assert np.all(crop[:, 0] == GRAY) and np.all(crop[:, -1] == GRAY)
        assert np.all((crop == GRAY) | (crop == bg)), "gray blocks may only contain gray and background"
        holes = int(np.sum(crop == bg))
        assert 1 <= holes <= 3, "hole count must be 1..3"
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": (r0 + 2, c0 + 2),
                "holes": holes,
            }
        )
    assert blocks, "expected at least one gray block"
    return blocks


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    markers = []
    for r, c in np.argwhere((grid != bg) & (grid != GRAY)):
        color = int(grid[r, c])
        assert color in MARKER_COLORS, "unexpected marker color"
        markers.append((int(r), int(c), color))
    return markers


def _pair_markers(blocks: list[dict], markers: list[tuple[int, int, int]]) -> dict[int, tuple[int, int, int]]:
    assert len(markers) == len(blocks), "expected one marker per block"
    assigned: dict[int, tuple[int, int, int]] = {}
    for marker in markers:
        mr, mc, _color = marker
        distances = []
        for idx, block in enumerate(blocks):
            br, bc = block["center"]
            distances.append((idx, (mr - br) ** 2 + (mc - bc) ** 2))
        best = min(d for _idx, d in distances)
        winners = [idx for idx, d in distances if d == best]
        assert len(winners) == 1, "marker-to-block pairing must be unambiguous"
        idx = winners[0]
        assert idx not in assigned, "each block must receive exactly one marker"
        assigned[idx] = marker
    assert len(assigned) == len(blocks)
    return assigned


def _paint_ring(out: np.ndarray, bbox: tuple[int, int, int, int], radius: int, color: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    rr0, cc0 = r0 - radius, c0 - radius
    rr1, cc1 = r1 + radius, c1 + radius
    for c in range(cc0, cc1 + 1):
        if int(out[rr0, c]) == bg:
            out[rr0, c] = color
        if int(out[rr1, c]) == bg:
            out[rr1, c] = color
    for r in range(rr0 + 1, rr1):
        if int(out[r, cc0]) == bg:
            out[r, cc0] = color
        if int(out[r, cc1]) == bg:
            out[r, cc1] = color


def _expected_output_from_structure(grid: np.ndarray, blocks: list[dict], assignments: dict[int, tuple[int, int, int]]) -> np.ndarray:
    bg = _get_bg(grid)
    out = np.asarray(grid, dtype=int).copy()
    for mr, mc, _color in assignments.values():
        out[mr, mc] = bg
    for idx, block in enumerate(blocks):
        color = assignments[idx][2]
        for radius in range(1, int(block["holes"]) + 1):
            _paint_ring(out, block["bbox"], radius, color, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a color-coded square envelope around each holed gray 5x5 block.
    
    Input structure:
    - The background is black, with 3-4 disconnected gray 5x5 foreground blocks.
    - Every gray block has a solid gray outer border and 1-3 background hole pixels inside its 5x5 bounding box.
    - Near each block there is exactly one isolated blue, red, or green marker pixel, and nearest-center pairing is unique.
    
    Rule:
    - For each gray block, count the number of background pixels inside its 5x5 bounding box.
    - Pair each marker pixel with the gray block whose 5x5-box center is uniquely closest to that marker.
    - Around that block's 5x5 bounding box, paint one square ring per counted hole, using the paired marker's color on background cells only.
    - Erase all marker pixels to background and leave the original gray blocks unchanged.
    
    Key insight:
    - Envelope size is determined by the block's internal hole count, while envelope color is determined independently by the nearby marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    blocks = _extract_blocks(g, bg)
    markers = _extract_markers(g, bg)
    assignments = _pair_markers(blocks, markers)
    return _expected_output_from_structure(g, blocks, assignments)


_INTERIOR_CELLS = tuple((r, c) for r in range(1, 4) for c in range(1, 4))


def _hole_patterns(count: int) -> list[tuple[tuple[int, int], ...]]:
    patterns = []
    for combo in combinations(_INTERIOR_CELLS, count):
        ok = True
        for (r1, c1), (r2, c2) in combinations(combo, 2):
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                ok = False
                break
        if ok:
            patterns.append(combo)
    return patterns


_PATTERNS = {count: _hole_patterns(count) for count in (1, 2, 3)}


def _make_block(holes: int) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), GRAY, dtype=int)
    for r, c in random.choice(_PATTERNS[int(holes)]):
        block[r, c] = BG
    return block


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_block_origins(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    origins: list[tuple[int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []
    for _ in range(400):
        origins.clear()
        reserved.clear()
        ok = True
        order = list(range(count))
        random.shuffle(order)
        placed: dict[int, tuple[int, int]] = {}
        for idx in order:
            candidates = []
            for r0 in range(4, height - 8):
                for c0 in range(4, width - 8):
                    reserve = (r0 - 4, c0 - 4, r0 + 8, c0 + 8)
                    if any(_boxes_overlap(reserve, prev) for prev in reserved):
                        continue
                    candidates.append((r0, c0, reserve))
            if not candidates:
                ok = False
                break
            r0, c0, reserve = random.choice(candidates)
            placed[idx] = (r0, c0)
            reserved.append(reserve)
        if ok and len(placed) == count:
            return [placed[i] for i in range(count)]
    return None


def _marker_candidates(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    cells = set()
    for dist in (1, 2):
        for c in range(c0 - 1, c1 + 2):
            cells.add((r0 - dist, c))
            cells.add((r1 + dist, c))
        for r in range(r0 - 1, r1 + 2):
            cells.add((r, c0 - dist))
            cells.add((r, c1 + dist))
    return sorted((r, c) for r, c in cells if 0 <= r < h and 0 <= c < w)


def _sample_markers(blocks: list[dict], colors: list[int], shape: tuple[int, int]) -> list[tuple[int, int, int]] | None:
    block_boxes = [block["bbox"] for block in blocks]
    block_centers = [block["center"] for block in blocks]
    candidates_by_block = []
    for block in blocks:
        block_cells = []
        for r, c in _marker_candidates(block["bbox"], shape):
            inside_some_box = False
            for br0, bc0, br1, bc1 in block_boxes:
                if br0 <= r <= br1 and bc0 <= c <= bc1:
                    inside_some_box = True
                    break
            if not inside_some_box:
                block_cells.append((r, c))
        random.shuffle(block_cells)
        candidates_by_block.append(block_cells)

    for _ in range(300):
        taken: set[tuple[int, int]] = set()
        markers: list[tuple[int, int, int]] = []
        ok = True
        for idx, color in enumerate(colors):
            legal = []
            for r, c in candidates_by_block[idx]:
                if (r, c) in taken:
                    continue
                dists = [((r - br) ** 2 + (c - bc) ** 2) for br, bc in block_centers]
                if dists[idx] != min(dists):
                    continue
                if dists.count(dists[idx]) != 1:
                    continue
                legal.append((r, c))
            if not legal:
                ok = False
                break
            r, c = random.choice(legal)
            taken.add((r, c))
            markers.append((r, c, color))
        if not ok:
            continue
        assignments = _pair_markers(blocks, markers)
        if len(assignments) == len(blocks):
            return markers
    return None


def _build_expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    blocks = []
    markers = []
    for spec in specs:
        r0, c0 = spec["origin"]
        block = spec["block"]
        inp[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
        holes = int(np.sum(block == BG))
        bbox = (r0, c0, r0 + 4, c0 + 4)
        blocks.append({"bbox": bbox, "center": (r0 + 2, c0 + 2), "holes": holes})
        mr, mc = spec["marker"]
        inp[mr, mc] = spec["color"]
        markers.append((mr, mc, spec["color"]))
    assignments = _pair_markers(blocks, markers)
    out = _expected_output_from_structure(inp, blocks, assignments)
    return inp, out


def _validate_generated_case(inp: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    try:
        assert inp.ndim == 2 and max(inp.shape) <= 30
        assert expected.ndim == 2 and max(expected.shape) <= 30
        assert len(specs) in (3, 4)
        assert len({tuple(spec["origin"]) for spec in specs}) == len(specs)
        holes = [int(np.sum(spec["block"] == BG)) for spec in specs]
        assert set(holes) >= {1, 2, 3}
        if len(specs) == 3:
            assert sorted(holes) == [1, 2, 3]
        colors = [int(spec["color"]) for spec in specs]
        assert len(set(colors)) >= 2
        blocks = _extract_blocks(inp, BG)
        assert len(blocks) == len(specs)
        markers = _extract_markers(inp, BG)
        assert len(markers) == len(specs)
        assignments = _pair_markers(blocks, markers)
        rebuilt = _expected_output_from_structure(inp, blocks, assignments)
        assert np.array_equal(rebuilt, expected)
        assert np.array_equal(main(inp), expected)
        for idx, block in enumerate(blocks):
            r0, c0, r1, c1 = block["bbox"]
            color = assignments[idx][2]
            changed = rebuilt[r0 - block["holes"] : r1 + block["holes"] + 1, c0 - block["holes"] : c1 + block["holes"] + 1]
            assert np.any(changed == color), "each envelope must add visible pixels"
        return True
    except AssertionError:
        return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    height = random.randint(23, 29)
    width = random.randint(23, 29)
    count = 4 if min(height, width) >= 25 and random.random() < 0.4 else 3

    origins = _sample_block_origins(height, width, count)
    if origins is None:
        return None

    holes = [1, 2, 3]
    if count == 4:
        holes.append(random.choice((1, 2, 3)))
    random.shuffle(holes)

    colors = list(MARKER_COLORS)
    if count == 4:
        colors.append(random.choice(MARKER_COLORS))
    random.shuffle(colors)

    blocks = []
    specs: list[dict] = []
    for (r0, c0), hole_count, color in zip(origins, holes, colors):
        block = _make_block(hole_count)
        bbox = (r0, c0, r0 + 4, c0 + 4)
        blocks.append({"bbox": bbox, "center": (r0 + 2, c0 + 2), "holes": hole_count})
        specs.append({"origin": (r0, c0), "block": block, "color": color})

    markers = _sample_markers(blocks, colors, (height, width))
    if markers is None:
        return None

    for spec, marker in zip(specs, markers):
        spec["marker"] = marker[:2]

    inp, expected = _build_expected_from_specs((height, width), specs)
    if not _validate_generated_case(inp, expected, specs):
        return None
    return inp, expected, specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid puzzle instance")

from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
AUX_COLORS = (4, 6, 7, 8, 9)
CORE_COLORS = (BLUE, RED, GREEN)
GEN_ATTEMPTS = 1000


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    local = np.zeros((height, width), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.array_equal(local, _border_mask(height, width)):
        return r0, c0, r1, c1
    return None


def _find_frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components(grid == GREY)
    assert 2 <= len(comps) <= 3, "expected 2-3 grey frame components"
    bboxes = []
    for comp in comps:
        box = _frame_bbox(comp)
        assert box is not None, "grey components must all be 1-pixel hollow rectangles"
        bboxes.append(box)
    bboxes.sort(key=lambda b: (-(b[2] - b[0] + 1) * (b[3] - b[1] + 1), b[0], b[1]))
    for outer, inner in zip(bboxes, bboxes[1:]):
        assert inner[0] == outer[0] + 2
        assert inner[1] == outer[1] + 2
        assert inner[2] == outer[2] - 2
        assert inner[3] == outer[3] - 2
    return bboxes


def _extract_core(grid: np.ndarray, inner_bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = inner_bbox
    core = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert 3 <= core.shape[0] <= 5 and 3 <= core.shape[1] <= 5
    assert np.all(core != BG)
    assert np.all(core != GREY)
    return core


def _legend_mapping(grid: np.ndarray, outer_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    mask = (grid != BG) & (grid != GREY)
    r0, c0, r1, c1 = outer_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(mask)
    assert 1 <= len(comps) <= 2, "expected 1-2 legend components outside the frames"
    mapping: dict[int, int] = {}
    for comp in comps:
        assert len(comp) == 2, "each legend must have exactly two cells"
        rows = sorted({int(v) for v in comp[:, 0]})
        cols = sorted(int(v) for v in comp[:, 1])
        assert len(rows) == 1 and cols[1] == cols[0] + 1, "legends must be horizontal pairs"
        row = rows[0]
        left, right = cols
        src = int(grid[row, right])
        dst = int(grid[row, left])
        assert src not in mapping, "legend sources must be unique"
        mapping[src] = dst
    return mapping


def _iterated_recolor(pattern: np.ndarray, mapping: dict[int, int], steps: int) -> np.ndarray:
    lut = np.arange(16, dtype=int)
    for src, dst in mapping.items():
        lut[int(src)] = int(dst)
    out = np.asarray(pattern, dtype=int)
    for _ in range(int(steps)):
        out = lut[out]
    return out.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the innermost nested frame interior and recolor it by iterating the legend mapping once per frame layer.

    Input structure:
    - The grid has black background, 2-3 nested grey 1-pixel hollow rectangles, and a small non-black non-grey core pattern inside the innermost frame.
    - Each inner frame is inset by exactly one background gap from the previous frame, so the frame count encodes a depth.
    - Outside the frames, 1-2 horizontal two-cell legends appear; in each legend the right cell is a source color and the left cell is its replacement color.

    Rule:
    - Detect the nested grey hollow rectangles and count their depth N.
    - Merge all legends into one color-replacement function f, reading each pair as source->target from right to left.
    - Crop the interior of the innermost frame to obtain the core pattern.
    - Apply f to every core pixel exactly N times, leaving colors unchanged on iterations where f is undefined for them.

    Key insight:
    - The grey geometry is not part of the answer image; it encodes the iteration count that controls how many times the separate corner legend must be reapplied to the cropped inner pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frame_bboxes(g)
    mapping = _legend_mapping(g, frames[0])
    core = _extract_core(g, frames[-1])
    return _iterated_recolor(core, mapping, len(frames))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _sample_core(
    height: int,
    width: int,
    palette: tuple[int, ...],
    required: tuple[int, ...],
) -> np.ndarray | None:
    for _ in range(300):
        core = np.random.choice(np.asarray(palette, dtype=int), size=(height, width)).astype(int)
        if any(not np.any(core == color) for color in required):
            continue
        if len(np.unique(core)) < 2:
            continue
        if np.all(core == core[0, 0]):
            continue
        return core
    return None


def _legend_positions(shape: tuple[int, int], corner: str, layout: str) -> list[tuple[int, int]]:
    h, w = shape
    if layout == "same_row":
        if corner == "left":
            return [(h - 1, 0), (h - 1, 3)]
        return [(h - 1, w - 5), (h - 1, w - 2)]
    if corner == "left":
        return [(h - 3, 0), (h - 1, 0)]
    return [(h - 3, w - 2), (h - 1, w - 2)]


def _single_rule(mapping_pair: tuple[int, int]) -> dict[int, int]:
    src, dst = mapping_pair
    return {int(src): int(dst)}


def _validate_generated(
    grid: np.ndarray,
    *,
    layers: int,
    core: np.ndarray,
    rule_pairs: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    out = main(grid)
    frames = _find_frame_bboxes(grid)
    parsed_core = _extract_core(grid, frames[-1])
    parsed_mapping = _legend_mapping(grid, frames[0])
    expected_mapping = {src: dst for src, dst in rule_pairs}

    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert len(frames) == layers
    assert np.array_equal(parsed_core, core)
    assert parsed_mapping == expected_mapping

    step1 = _iterated_recolor(core, expected_mapping, 1)
    step2 = _iterated_recolor(core, expected_mapping, 2)
    step3 = _iterated_recolor(core, expected_mapping, 3)
    expected = step2 if layers == 2 else step3

    assert np.array_equal(out, expected)
    assert out.shape == core.shape
    assert not np.array_equal(out, core), "crop alone must not solve the task"
    assert len(np.unique(out)) >= 2, "output should preserve visible pattern structure"

    only_first = _iterated_recolor(core, _single_rule(rule_pairs[0]), layers)
    only_second = _iterated_recolor(core, _single_rule(rule_pairs[1]), layers)
    assert not np.array_equal(out, only_first), "first legend alone must be insufficient"
    assert not np.array_equal(out, only_second), "second legend alone must be insufficient"

    if layers == 2:
        assert np.array_equal(out, step2)
        assert not np.array_equal(out, step1), "two-layer examples must require a second iteration"
    else:
        assert np.array_equal(out, step3)
        assert not np.array_equal(out, step2), "three-layer examples must differ from two iterations"


def _gen_attempt() -> np.ndarray | None:
    layers = random.choice((2, 3))
    core_h = random.randint(3, 5)
    core_w = random.randint(3, 5)
    inner_h = core_h + 2
    inner_w = core_w + 2
    outer_h = inner_h + 4 * (layers - 1)
    outer_w = inner_w + 4 * (layers - 1)

    grid_h = random.randint(max(15, outer_h + 4), min(23, outer_h + 8))
    grid_w = random.randint(max(15, outer_w + 4), min(23, outer_w + 8))
    if max(grid_h, grid_w) > 30:
        return None

    bottom_reserve = 4
    max_r0 = grid_h - outer_h - bottom_reserve
    if max_r0 < 0:
        return None
    outer_r0 = random.randint(0, max_r0)
    outer_c0 = random.randint(0, grid_w - outer_w)

    if layers == 3:
        a, b = random.sample(CORE_COLORS, 2)
        extra = [c for c in CORE_COLORS if c not in (a, b)]
        used = [a, b]
        if extra and random.random() < 0.6:
            used.append(extra[0])
        core = _sample_core(core_h, core_w, tuple(used), (a, b))
        if core is None:
            return None
        mapping_pairs = ((a, b), (b, a))
    else:
        source = random.choice(CORE_COLORS)
        survivors = [c for c in CORE_COLORS if c != source]
        survivor_count = random.choice((1, 2))
        used = [source, *random.sample(survivors, survivor_count)]
        middle, final = random.sample(AUX_COLORS, 2)
        core = _sample_core(core_h, core_w, tuple(used), (source, used[1]))
        if core is None:
            return None
        mapping_pairs = ((source, middle), (middle, final))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for i in range(layers):
        frame_bbox = (
            outer_r0 + 2 * i,
            outer_c0 + 2 * i,
            outer_r0 + outer_h - 1 - 2 * i,
            outer_c0 + outer_w - 1 - 2 * i,
        )
        _draw_frame(grid, frame_bbox)

    inner_bbox = (
        outer_r0 + 2 * (layers - 1),
        outer_c0 + 2 * (layers - 1),
        outer_r0 + outer_h - 1 - 2 * (layers - 1),
        outer_c0 + outer_w - 1 - 2 * (layers - 1),
    )
    grid[inner_bbox[0] + 1 : inner_bbox[2], inner_bbox[1] + 1 : inner_bbox[3]] = core

    corner = random.choice(("left", "right"))
    layout = random.choice(("same_row", "stacked"))
    positions = _legend_positions(grid.shape, corner, layout)
    legend_specs = list(mapping_pairs)
    random.shuffle(legend_specs)
    for (src, dst), (r, c) in zip(legend_specs, positions):
        grid[r, c] = dst
        grid[r, c + 1] = src

    try:
        _validate_generated(grid, layers=layers, core=core, rule_pairs=mapping_pairs)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame recolor sample")

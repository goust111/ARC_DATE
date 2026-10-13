from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared_012_3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BG = Color.BLACK
FRAME = Color.RED
MARKER = Color.YELLOW
PALETTE_POOL = (Color.BLUE, Color.GREEN, Color.GREY, Color.ORANGE)
GRID_SIDE_RANGE = (20, 28)
FRAME_COUNT_OPTIONS = (3, 4)
COUNT_SETS = {
    3: [
        (0, 1, 4),
        (0, 2, 4),
        (0, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
    ],
    4: [
        (0, 1, 2, 4),
        (0, 1, 3, 4),
        (0, 2, 3, 4),
        (1, 2, 3, 4),
        (0, 1, 2, 3),
    ],
}
MAX_ATTEMPTS = 500


def _extract_palette(grid: np.ndarray) -> list[int]:
    palette_mask = ~np.isin(grid, [BG, FRAME, MARKER])
    coords = np.argwhere(palette_mask)
    assert len(coords) == 4, "expected exactly four palette pixels"
    items = sorted((int(r), int(c), int(grid[r, c])) for r, c in coords)
    rows = [r for r, _, _ in items]
    cols = [c for _, c, _ in items]
    assert len({color for _, _, color in items}) == 4, "palette colors must be distinct"
    assert all(rows[i + 1] == rows[i] + 1 for i in range(3)), "palette rows must be consecutive"
    assert all(cols[i + 1] == cols[i] + 1 for i in range(3)), "palette cols must be consecutive"
    assert len({rows[i] - cols[i] for i in range(4)}) == 1, "palette must lie on one down-right diagonal"
    return [color for _, _, color in items]


def _extract_frames(grid: np.ndarray) -> list[dict]:
    red_components = find_components(
        grid,
        bg=int(get_bg(grid)),
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    frames: list[dict] = []
    for comp in red_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        expected_area = 2 * (h + w) - 4
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        assert int(comp["area"]) == expected_area, "red object must be a 1-pixel hollow rectangle"
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in coords), "rectangle frame must match its bbox"
        assert h >= 7 and w >= 7, "frame interior must be at least 5x5"
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = [
            (int(r), int(c))
            for r, c in np.argwhere(interior == MARKER)
        ]
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": (h - 2, w - 2),
                "count": len(marker_coords),
                "marker_coords": [(r0 + 1 + r, c0 + 1 + c) for r, c in marker_coords],
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _paint_concentric(out: np.ndarray, bbox: tuple[int, int, int, int], palette: list[int], count: int) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if count == 0:
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        return
    assert 1 <= count <= len(palette)
    for rr in range(inner_h):
        for cc in range(inner_w):
            layer = min(rr, cc, inner_h - 1 - rr, inner_w - 1 - cc)
            out[r0 + 1 + rr, c0 + 1 + cc] = int(palette[min(layer, count - 1)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count yellow markers inside each red frame and use that count as the concentric fill depth.

    Input structure:
    - A black background contains 3-4 disjoint red hollow rectangles with interiors of at least 5x5.
    - Each frame interior contains zero or more yellow marker pixels whose positions are arbitrary.
    - Four distinct non-red/non-yellow colors appear as a short down-right diagonal palette in the top-left area.

    Rule:
    - Read the four palette colors in diagonal order from top-left toward bottom-right.
    - For each red frame, count the yellow markers strictly inside the frame.
    - Fill the frame interior with that many concentric rectangular layers: layer 1 uses palette[0], layer 2 uses palette[1], and so on; everything deeper than the last requested layer takes the last requested color.
    - Yellow markers are overwritten, while the red frames and the diagonal palette remain unchanged.

    Key insight:
    - The yellow pixels are count tokens, not shape templates: only their number matters, and that number selects how many inward color layers to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    palette = _extract_palette(g)
    frames = _extract_frames(g)
    assert 3 <= len(frames) <= 4

    out = g.copy()
    for frame in frames:
        _paint_concentric(out, frame["bbox"], palette, int(frame["count"]))
    return out


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _sample_counts(frame_count: int) -> list[int]:
    counts = list(random.choice(COUNT_SETS[frame_count]))
    random.shuffle(counts)
    return counts


def _sample_specs(counts: list[int]) -> list[dict] | None:
    for _ in range(80):
        specs = []
        for count in counts:
            min_inner = max(5, 2 * count - 1)
            max_inner = min(10, min_inner + 4)
            inner_h = random.randint(min_inner, max_inner)
            inner_w = random.randint(min_inner, max_inner)
            specs.append(
                {
                    "count": count,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": inner_h + 2,
                    "outer_w": inner_w + 2,
                }
            )

        areas = [spec["inner_h"] * spec["inner_w"] for spec in specs]
        if len(set((spec["inner_h"], spec["inner_w"]) for spec in specs)) < 2:
            continue
        if max(counts) < 3:
            continue
        if not any(
            areas[i] > areas[j] and counts[i] < counts[j]
            for i, j in itertools.permutations(range(len(specs)), 2)
        ):
            continue
        if not any(
            spec["count"] < 4 and min(spec["inner_h"], spec["inner_w"]) >= max(5, 2 * (spec["count"] + 1) - 1)
            for spec in specs
        ):
            continue
        return specs
    return None


def _place_frames(height: int, width: int, specs: list[dict], palette_start: tuple[int, int]) -> list[dict] | None:
    reserved = np.zeros((height, width), dtype=bool)
    pr, pc = palette_start
    palette_bbox = (0, 0, min(height - 1, pr + 5), min(width - 1, pc + 5))
    if not _reserve(reserved, palette_bbox, pad=1):
        return None

    placed: list[dict | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        placed_here = False
        for _ in range(500):
            r0 = random.randint(0, height - spec["outer_h"])
            c0 = random.randint(0, width - spec["outer_w"])
            bbox = (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1)
            if r0 <= pr + 4 and c0 <= pc + 4:
                continue
            if not _reserve(reserved, bbox, pad=1):
                continue
            placed[idx] = {**spec, "bbox": bbox}
            placed_here = True
            break
        if not placed_here:
            return None
    return [item for item in placed if item is not None]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _sample_marker_positions(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if count == 1:
        return [random.choice(cells)]

    mode = random.choice(("spread", "cluster"))
    if mode == "cluster":
        anchor_r = random.randint(0, inner_h - 1)
        anchor_c = random.randint(0, inner_w - 1)
        cells.sort(key=lambda pos: (abs(pos[0] - anchor_r) + abs(pos[1] - anchor_c), random.random()))
        return cells[:count]

    return random.sample(cells, count)


def _analyze_generated(grid: np.ndarray) -> tuple[list[int], list[dict]]:
    palette = _extract_palette(grid)
    frames = _extract_frames(grid)
    assert 3 <= len(frames) <= 4
    counts = [int(frame["count"]) for frame in frames]
    assert len(set(counts)) == len(counts), "marker counts must be unique"
    assert max(counts) <= len(palette)
    assert max(counts) >= 3
    return palette, frames


def _relocate_markers_variant(grid: np.ndarray, frame: dict) -> np.ndarray:
    count = int(frame["count"])
    if count == 0:
        return grid.copy()
    r0, c0, r1, c1 = frame["bbox"]
    inner_h, inner_w = frame["inner_shape"]
    all_cells = {(r0 + 1 + r, c0 + 1 + c) for r in range(inner_h) for c in range(inner_w)}
    current = set(frame["marker_coords"])
    choices = list(all_cells - current)
    random.shuffle(choices)
    new_markers = set(choices[:count])
    if len(new_markers) < count:
        return grid.copy()
    mutated = grid.copy()
    mutated[r0 + 1 : r1, c0 + 1 : c1] = BG
    for rr, cc in new_markers:
        mutated[rr, cc] = MARKER
    return mutated


def _increment_marker_variant(grid: np.ndarray, frame: dict) -> np.ndarray | None:
    count = int(frame["count"])
    if count >= 4:
        return None
    inner_h, inner_w = frame["inner_shape"]
    if min(inner_h, inner_w) < max(5, 2 * (count + 1) - 1):
        return None
    r0, c0, r1, c1 = frame["bbox"]
    current = set(frame["marker_coords"])
    candidates = [(r0 + 1 + r, c0 + 1 + c) for r in range(inner_h) for c in range(inner_w)]
    empties = [cell for cell in candidates if cell not in current]
    if not empties:
        return None
    rr, cc = random.choice(empties)
    mutated = grid.copy()
    mutated[rr, cc] = MARKER
    return mutated


def _validate_generated(grid: np.ndarray) -> None:
    palette, frames = _analyze_generated(grid)
    assert set(palette) == set(PALETTE_POOL)

    areas = [frame["inner_shape"][0] * frame["inner_shape"][1] for frame in frames]
    counts = [int(frame["count"]) for frame in frames]
    assert any(
        areas[i] > areas[j] and counts[i] < counts[j]
        for i, j in itertools.permutations(range(len(frames)), 2)
    ), "frame size must not determine fill depth"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "at least one frame should change after filling"

    moved_checked = False
    for frame in frames:
        variant = _relocate_markers_variant(grid, frame)
        if not np.array_equal(variant, grid):
            moved_checked = True
            assert np.array_equal(main(variant), out), "marker positions must be irrelevant"
            break
    assert moved_checked, "expected a positive-count frame for relocation validation"

    for frame in frames:
        variant = _increment_marker_variant(grid, frame)
        if variant is not None:
            assert not np.array_equal(main(variant), out), "changing a count must change the output"
            break


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    palette_start = (random.randint(0, 2), random.randint(0, 2))
    palette = list(PALETTE_POOL)
    random.shuffle(palette)
    pr, pc = palette_start
    for idx, color in enumerate(palette):
        grid[pr + idx, pc + idx] = color

    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    counts = _sample_counts(frame_count)
    specs = _sample_specs(counts)
    if specs is None:
        return None
    placed = _place_frames(height, width, specs, palette_start)
    if placed is None:
        return None

    for spec in placed:
        bbox = spec["bbox"]
        _draw_frame(grid, bbox)
        positions = _sample_marker_positions(spec["inner_h"], spec["inner_w"], spec["count"])
        r0, c0, _r1, _c1 = bbox
        for rr, cc in positions:
            grid[r0 + 1 + rr, c0 + 1 + cc] = MARKER

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

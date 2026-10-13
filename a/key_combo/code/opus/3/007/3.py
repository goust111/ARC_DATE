from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


BACKGROUND = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
MARKER_SIZE = 2
REGION_HEIGHT = 2

GEN_RETRY_LIMIT = 400
REGION_BUILD_RETRIES = 200
GAP_WIDTH_RANGE = (8, 12)
GRID_WIDTH_RANGE = (18, 27)
TOP_MARGIN_RANGE = (1, 4)
BOTTOM_MARGIN_RANGE = (1, 4)
ROW_GAP_RANGE = (1, 5)
NUM_REGION_CHOICES = (3, 4)

BLUE_TEMPLATES: dict[int, tuple[np.ndarray, ...]] = {
    2: (
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ),
    4: (
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    ),
    5: (
        np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
        np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=int),
    ),
    6: (
        np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    ),
}

ANCHOR_SPEC_PAIRS = (
    (((4,), 2), ((2, 2), 1)),
    (((5,), 2), ((3, 2), 1)),
    (((6,), 2), ((4, 2), 1)),
)

EXTRA_SPECS = (
    ((2,), 0),
    ((2,), 1),
    ((3,), 0),
    ((3,), 1),
    ((4,), 1),
    ((4,), 2),
    ((2, 3), 0),
    ((2, 3), 1),
    ((4, 2), 0),
    ((4, 2), 2),
    ((2, 2, 2), 0),
)


def _paired_marker_blocks(g: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    bg = int(get_bg(g))
    marker_components = find_components(g, bg=bg, colors=[RED], connectivity=4)
    blocks: list[tuple[int, int, int, int]] = []
    for comp in marker_components:
        box = comp["bbox"]
        assert int(comp["area"]) == 4 and bbox_h(box) == 2 and bbox_w(box) == 2
        blocks.append(box)

    assert len(blocks) in {6, 8}
    by_row_span: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for box in blocks:
        by_row_span.setdefault((int(box[0]), int(box[2])), []).append(box)

    pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    gap_widths = set()
    for row_span in sorted(by_row_span):
        row_blocks = sorted(by_row_span[row_span], key=lambda b: b[1])
        assert len(row_blocks) == 2
        left, right = row_blocks
        gap_width = int(right[1] - left[3] - 1)
        assert gap_width > 0
        gap_widths.add(gap_width)
        pairs.append((left, right))

    assert len(gap_widths) == 1
    return pairs


def _extract_regions(g: np.ndarray) -> list[np.ndarray]:
    regions = []
    widths = set()
    for left, right in _paired_marker_blocks(g):
        assert left[0] == right[0] and left[2] == right[2]
        region = g[left[0] : left[2] + 1, left[3] + 1 : right[1]]
        assert region.shape[0] == REGION_HEIGHT and region.shape[1] > 0
        widths.add(int(region.shape[1]))
        regions.append(region.copy())
    assert regions and len(widths) == 1
    return regions


def _components_of_color(region: np.ndarray, color: int) -> list[dict]:
    return find_components(region, bg=BACKGROUND, colors=[int(color)], connectivity=4)


def _count_blue_objects(region: np.ndarray) -> int:
    return len(_components_of_color(region, BLUE))


def _count_all_visible_components(region: np.ndarray) -> int:
    return len(_components_of_color(region, BLUE)) + len(_components_of_color(region, GREEN))


def _blue_pixel_count(region: np.ndarray) -> int:
    return int(np.count_nonzero(region == BLUE))


def _stack_by_rule(regions: list[np.ndarray], count_fn) -> np.ndarray:
    processed = []
    for region in regions:
        processed.append(np.fliplr(region) if count_fn(region) % 2 == 1 else region.copy())
    return np.vstack(processed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each marker-bounded interior strip, flip the strips with an odd number of blue objects, and stack them.

    Input structure:
    - The background is black.
    - Red pixels appear only as 2x2 blocks, forming 3-4 left/right pairs that share the same two-row span.
    - The rectangle strictly between each paired red blocks contains blue 4-connected objects and isolated green single-pixel distractors.

    Rule:
    - Pair the red 2x2 blocks by shared row span and read the pairs from top to bottom.
    - For each pair, crop the rectangle strictly between the markers, excluding the marker columns.
    - Count only blue 4-connected objects in that cropped region, treating green pixels as background for counting.
    - If the blue-object count is odd, horizontally flip the entire cropped region; otherwise keep it unchanged.
    - Vertically stack the processed cropped regions in top-to-bottom order and discard everything else.

    Key insight:
    - The red marker pairs define the extractable strips, while the green singletons are visible but non-counting noise; miscounting them changes the odd/even flip decision.
    """
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)
    return _stack_by_rule(regions, _count_blue_objects)


def _composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(map(int, rng.multinomial(int(total), [1.0 / parts] * parts)))


def _template_for_area(area: int, rng: np.random.Generator) -> np.ndarray:
    template = BLUE_TEMPLATES[int(area)][int(rng.integers(len(BLUE_TEMPLATES[int(area)])))]
    out = template.copy()
    if int(rng.integers(2)):
        out = np.fliplr(out)
    return out


def _paint_template(region: np.ndarray, template: np.ndarray, row0: int, col0: int) -> None:
    mask = template.astype(bool)
    h, w = template.shape
    view = region[row0 : row0 + h, col0 : col0 + w]
    view[mask] = BLUE


def _green_candidates(region: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    height, width = region.shape
    for r in range(height):
        for c in range(width):
            if int(region[r, c]) != BACKGROUND:
                continue
            blocked = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and int(region[nr, nc]) in {BLUE, GREEN}:
                    blocked = True
                    break
            if not blocked:
                candidates.append((r, c))
    return candidates


def _build_region(
    width: int,
    blue_areas: tuple[int, ...],
    green_count: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    for _ in range(REGION_BUILD_RETRIES):
        region = np.zeros((REGION_HEIGHT, width), dtype=int)
        order = list(map(int, blue_areas))
        rng.shuffle(order)
        templates = [_template_for_area(area, rng) for area in order]
        widths = [int(template.shape[1]) for template in templates]
        extra = int(width) - sum(widths) - max(0, len(templates) - 1)
        if extra < 0:
            continue

        gaps = _composition(extra, len(templates) + 1, rng)
        col = int(gaps[0])
        for i, template in enumerate(templates):
            row0 = 0 if template.shape[0] == REGION_HEIGHT else int(rng.integers(0, REGION_HEIGHT))
            _paint_template(region, template, row0, col)
            col += int(template.shape[1])
            if i + 1 < len(templates):
                col += 1 + int(gaps[i + 1])

        placed = 0
        while placed < int(green_count):
            candidates = _green_candidates(region)
            if not candidates:
                break
            idx = int(rng.integers(len(candidates)))
            r, c = candidates[idx]
            region[r, c] = GREEN
            placed += 1
        if placed != int(green_count):
            continue

        blue_components = _components_of_color(region, BLUE)
        if len(blue_components) != len(blue_areas):
            continue
        if not all(2 <= int(comp["area"]) <= 8 for comp in blue_components):
            continue

        green_components = _components_of_color(region, GREEN)
        if len(green_components) != int(green_count):
            continue
        if not all(int(comp["area"]) == 1 for comp in green_components):
            continue

        if np.array_equal(region, np.fliplr(region)):
            continue
        return region
    return None


def _row_starts(n_regions: int, rng: np.random.Generator) -> tuple[list[int], int]:
    top_margin = int(rng.integers(*TOP_MARGIN_RANGE))
    bottom_margin = int(rng.integers(*BOTTOM_MARGIN_RANGE))
    gaps = [int(rng.integers(*ROW_GAP_RANGE)) for _ in range(max(0, n_regions - 1))]
    starts = [top_margin]
    for gap in gaps:
        starts.append(starts[-1] + REGION_HEIGHT + gap)
    height = starts[-1] + REGION_HEIGHT + bottom_margin
    return starts, int(height)


def _choose_specs(n_regions: int, rng: np.random.Generator) -> list[tuple[tuple[int, ...], int]]:
    anchor_a, anchor_b = ANCHOR_SPEC_PAIRS[int(rng.integers(len(ANCHOR_SPEC_PAIRS)))]
    specs = [anchor_a, anchor_b]
    extras = list(EXTRA_SPECS)
    rng.shuffle(extras)
    specs.extend(extras[: n_regions - 2])
    rng.shuffle(specs)
    return [(tuple(map(int, blue_areas)), int(green_count)) for blue_areas, green_count in specs]


def _assemble_grid(
    regions: list[np.ndarray],
    gap_width: int,
    grid_width: int,
    row_starts: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros((row_starts[-1] + REGION_HEIGHT + 1, grid_width), dtype=int)
    max_left = int(grid_width - gap_width - 4)
    assert max_left >= 1
    for region, row0 in zip(regions, row_starts):
        left_col = int(rng.integers(1, max_left + 1))
        right_col = left_col + MARKER_SIZE + int(gap_width)
        grid[row0 : row0 + MARKER_SIZE, left_col : left_col + MARKER_SIZE] = RED
        grid[row0 : row0 + MARKER_SIZE, left_col + MARKER_SIZE : right_col] = region
        grid[row0 : row0 + MARKER_SIZE, right_col : right_col + MARKER_SIZE] = RED
    return grid


def _summaries(regions: list[np.ndarray]) -> list[dict[str, int | bool]]:
    data = []
    for region in regions:
        blue_count = _count_blue_objects(region)
        green_count = len(_components_of_color(region, GREEN))
        data.append(
            {
                "blue_count": int(blue_count),
                "green_count": int(green_count),
                "blue_area": _blue_pixel_count(region),
                "total_components": int(blue_count + green_count),
                "is_symmetric": bool(np.array_equal(region, np.fliplr(region))),
            }
        )
    return data


def _has_same_area_and_same_total_with_opposite_parity(regions: list[np.ndarray]) -> bool:
    summaries = _summaries(regions)
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            a = summaries[i]
            b = summaries[j]
            if (
                int(a["blue_area"]) == int(b["blue_area"])
                and int(a["total_components"]) == int(b["total_components"])
                and int(a["blue_count"]) % 2 != int(b["blue_count"]) % 2
            ):
                return True
    return False


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        regions = _extract_regions(grid)
    except Exception:
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if expected.shape[0] != REGION_HEIGHT * len(regions):
        return False

    summaries = _summaries(regions)
    if not any(int(item["blue_count"]) % 2 == 1 for item in summaries):
        return False
    if not any(int(item["blue_count"]) % 2 == 0 for item in summaries):
        return False
    if any(bool(item["is_symmetric"]) for item in summaries):
        return False
    if not _has_same_area_and_same_total_with_opposite_parity(regions):
        return False

    processed_keys = {
        tuple(_stack_by_rule([region], _count_blue_objects).ravel()) for region in regions
    }
    if len(processed_keys) != len(regions):
        return False

    if np.array_equal(expected, np.vstack(regions)):
        return False
    if np.array_equal(expected, _stack_by_rule(regions, _count_all_visible_components)):
        return False
    if np.array_equal(expected, _stack_by_rule(regions, _blue_pixel_count)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_regions = int(NUM_REGION_CHOICES[int(rng.integers(len(NUM_REGION_CHOICES)))])
    gap_width = int(rng.integers(*GAP_WIDTH_RANGE))
    grid_width = int(rng.integers(*GRID_WIDTH_RANGE))
    if grid_width - gap_width - 4 < 1:
        return None

    specs = _choose_specs(n_regions, rng)
    regions = []
    for blue_areas, green_count in specs:
        region = _build_region(gap_width, blue_areas, green_count, rng)
        if region is None:
            return None
        regions.append(region)

    row_starts, height = _row_starts(n_regions, rng)
    if height > 30:
        return None
    grid = np.zeros((height, grid_width), dtype=int)
    max_left = int(grid_width - gap_width - 4)
    if max_left < 1:
        return None

    for region, row0 in zip(regions, row_starts):
        left_col = int(rng.integers(1, max_left + 1))
        right_col = left_col + MARKER_SIZE + gap_width
        grid[row0 : row0 + MARKER_SIZE, left_col : left_col + MARKER_SIZE] = RED
        grid[row0 : row0 + MARKER_SIZE, left_col + MARKER_SIZE : right_col] = region
        grid[row0 : row0 + MARKER_SIZE, right_col : right_col + MARKER_SIZE] = RED

    expected = _stack_by_rule(regions, _count_blue_objects)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

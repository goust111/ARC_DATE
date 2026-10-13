from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

TOP_ROWS = 2
BOTTOM_ROWS = 3
GRID_HEIGHT = TOP_ROWS + BOTTOM_ROWS
PERIOD_CHOICES = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 800


def _shape(rows: list[str]) -> np.ndarray:
    return np.array([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=int)


SHAPES = [
    _shape(["1", "1", "1"]),
    _shape(["111"]),
    _shape(["10", "11"]),
    _shape(["01", "11"]),
    _shape(["11", "10"]),
    _shape(["11", "01"]),
    _shape(["11", "11"]),
    _shape(["111", "010"]),
    _shape(["010", "111"]),
    _shape(["110", "011"]),
    _shape(["011", "110"]),
    _shape(["10", "10", "11"]),
    _shape(["01", "01", "11"]),
    _shape(["11", "01", "01"]),
    _shape(["11", "10", "10"]),
    _shape(["010", "111", "010"]),
    _shape(["11", "11", "10"]),
    _shape(["11", "11", "01"]),
]


def _apply_variant(tile: np.ndarray, variant: str) -> np.ndarray:
    if variant == "id":
        return tile
    if variant == "ud":
        return np.flipud(tile)
    if variant == "lr":
        return np.fliplr(tile)
    raise ValueError(f"unknown variant {variant}")


def _matches_period(band: np.ndarray, period: int) -> bool:
    width = int(band.shape[1])
    if width % period != 0:
        return False
    base = band[:, :period]
    for start in range(0, width, period):
        block = band[:, start : start + period]
        if not any(np.array_equal(base, _apply_variant(block, variant)) for variant in ("id", "ud", "lr")):
            return False
    return True


def _detect_period(band: np.ndarray) -> int:
    width = int(band.shape[1])
    for period in range(1, width + 1):
        if _matches_period(band, period):
            return period
    raise AssertionError("expected at least one valid period")


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _selected_indices(count: int, period: int) -> list[int]:
    return list(range(0, count, period))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the top band's smallest mirror-aware period and use it to recolor every P-th lower object.

    Input structure:
    - The grid has five rows: a two-row grey/black horizontal pattern band on top
      and a three-row black field below.
    - The top band is built from equal-width repeated blocks; each block may match
      the leftmost block directly, upside down, or left-right mirrored.
    - The lower field contains several separated green 4-connected objects.

    Rule:
    - Find the smallest horizontal period P of the top two rows under the mirror
      rule above.
    - Enumerate the lower green objects from left to right by bbox left edge.
    - Recolor objects numbered 1, 1+P, 1+2P, ... from green to red; keep all
      other pixels unchanged.

    Key insight:
    - The top band is not itself transformed; it encodes a numeric stride P that
      controls which lower objects are recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and int(g.shape[0]) == GRID_HEIGHT, "expected a 5-row grid"

    period = _detect_period(g[:TOP_ROWS, :])
    lower = g[TOP_ROWS:, :]
    components = _find_color_components(lower, GREEN)
    components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))

    out = g.copy()
    for idx in _selected_indices(len(components), period):
        for r, c in components[idx]["coords"]:
            out[TOP_ROWS + r, c] = RED
    return out


def _sample_base_unit(period: int) -> np.ndarray | None:
    for _ in range(200):
        unit = np.full((TOP_ROWS, period), BLACK, dtype=int)
        grey_count = random.randint(2, 2 * period - 1)
        for pos in random.sample(range(2 * period), grey_count):
            unit[pos // period, pos % period] = GREY
        if np.array_equal(unit, np.flipud(unit)) or np.array_equal(unit, np.fliplr(unit)):
            continue
        return unit
    return None


def _build_top_band(period: int, width: int) -> tuple[np.ndarray, list[str]] | None:
    repeat_count = width // period
    require_reflection = repeat_count >= 3 and random.random() < 0.65
    for _ in range(300):
        base = _sample_base_unit(period)
        if base is None:
            continue
        variants = ["id"]
        for _block_idx in range(1, repeat_count):
            choices = ["id", "ud", "lr"] if require_reflection else ["id", "id", "ud", "lr"]
            variants.append(random.choice(choices))
        if require_reflection and all(variant == "id" for variant in variants[1:]):
            variants[random.randrange(1, repeat_count)] = random.choice(["ud", "lr"])

        blocks = [_apply_variant(base, variant) for variant in variants]
        band = np.concatenate(blocks, axis=1)
        if _detect_period(band) != period:
            continue
        if require_reflection and not any(not np.array_equal(block, base) for block in blocks[1:]):
            continue
        return band, variants
    return None


def _sample_shapes(count: int, width: int) -> list[np.ndarray] | None:
    weighted_pool = []
    for idx, shape in enumerate(SHAPES):
        weighted_pool.extend([idx] * (5 - shape.shape[1]))
    for _ in range(400):
        indices = [random.choice(weighted_pool) for _ in range(count)]
        if len(set(indices)) < 3:
            continue
        shapes = [SHAPES[idx] for idx in indices]
        min_needed = sum(shape.shape[1] for shape in shapes) + (count - 1)
        if min_needed <= width:
            return shapes
    return None


def _random_parts(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _build_bottom(width: int, count: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    shapes = _sample_shapes(count, width)
    if shapes is None:
        return None

    min_needed = sum(shape.shape[1] for shape in shapes) + (count - 1)
    extra = width - min_needed
    left_margin = random.randint(0, min(2, extra))
    extra -= left_margin
    right_margin = random.randint(0, min(2, extra))
    extra -= right_margin
    internal_extras = _random_parts(extra, count - 1) if count > 1 else []

    bottom = np.full((BOTTOM_ROWS, width), BLACK, dtype=int)
    cursor = left_margin
    for shape_idx, shape in enumerate(shapes):
        sh, sw = shape.shape
        top = random.randint(0, BOTTOM_ROWS - sh)
        patch_rows, patch_cols = np.where(shape == 1)
        bottom[top + patch_rows, cursor + patch_cols] = GREEN
        cursor += sw
        if shape_idx + 1 < count:
            cursor += 1 + internal_extras[shape_idx]
    cursor += right_margin
    assert cursor == width
    return bottom, shapes


def _expected_output(grid: np.ndarray, period: int) -> np.ndarray:
    out = grid.copy()
    lower = grid[TOP_ROWS:, :]
    components = _find_color_components(lower, GREEN)
    components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    for idx in _selected_indices(len(components), period):
        for r, c in components[idx]["coords"]:
            out[TOP_ROWS + r, c] = RED
    return out


def _validate_instance(grid: np.ndarray, period: int, variants: list[str]) -> bool:
    if grid.shape[0] != GRID_HEIGHT or max(grid.shape) > 30:
        return False

    band = grid[:TOP_ROWS, :]
    if _detect_period(band) != period:
        return False

    if any(variant != "id" for variant in variants[1:]):
        first = band[:, :period]
        changed = [
            not np.array_equal(first, band[:, start : start + period])
            for start in range(period, band.shape[1], period)
        ]
        if not any(changed):
            return False

    lower = grid[TOP_ROWS:, :]
    components = _find_color_components(lower, GREEN)
    components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    count = len(components)
    if not (6 <= count <= 10):
        return False
    if any(not (3 <= comp["area"] <= 5) for comp in components):
        return False
    if any(components[i]["bbox"][1] >= components[i + 1]["bbox"][1] for i in range(count - 1)):
        return False
    if any(components[i]["bbox"][3] + 1 >= components[i + 1]["bbox"][1] for i in range(count - 1)):
        return False

    selected = _selected_indices(count, period)
    if len(selected) < 2 or count - len(selected) < 2:
        return False
    for alt_period in PERIOD_CHOICES:
        if alt_period == period:
            continue
        if _selected_indices(count, alt_period) == selected:
            return False

    expected = _expected_output(grid, period)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    changed_pixels = int(np.sum(grid != expected))
    if changed_pixels <= 3 or changed_pixels >= int(np.sum(lower == GREEN)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        period = random.choice(PERIOD_CHOICES)
        candidate_widths = [width for width in range(18, 31) if width % period == 0]
        width = random.choice(candidate_widths)

        max_count = min(10, (width + 2) // 3)
        if max_count < 6:
            continue
        count = random.randint(6, max_count)

        top_result = _build_top_band(period, width)
        if top_result is None:
            continue
        top_band, variants = top_result

        bottom_result = _build_bottom(width, count)
        if bottom_result is None:
            continue
        bottom, _shapes = bottom_result

        grid = np.vstack([top_band, bottom])
        if _validate_instance(grid, period, variants):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

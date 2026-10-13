from __future__ import annotations

import importlib.util
import random
from collections import deque
from pathlib import Path

import numpy as np


def _load_arc_utils():
    """Best-effort absolute-path import so SourceFileLoader can still load this file directly."""
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_arc_utils_031_4", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    _neighbors4 = _ARC_UTILS.neighbors4
else:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        YELLOW = 4

    def _neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1


GEN_WALL_COLORS = (1, 5, 6, 7, 8, 9)
GEN_REGION_COUNTS = (3, 4, 5)
GEN_ORIENTATIONS = ("vertical", "horizontal")
GEN_SPAN_POOL = tuple(range(2, 8))
GEN_STRIPE_THICKNESS_RANGE = (4, 11)
GEN_EXTRA_MARGIN_MAX = 8
GEN_MAX_ATTEMPTS = 400


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _mask_components(mask: np.ndarray) -> list[dict]:
    """4-connected components on a boolean mask, keeping ARC's (r, c) coordinate convention."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                touches_border |= rr in (0, h - 1) or cc in (0, w - 1)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "touches_border": bool(touches_border),
                }
            )

    return components


def _wall_component_count(grid: np.ndarray, wall_color: int) -> int:
    if _ARC_UTILS is not None:
        return len(
            _ARC_UTILS.find_components(
                grid,
                bg=Color.BLACK,
                connectivity=4,
                monochromatic=True,
                colors=[wall_color],
            )
        )
    return len(_mask_components(grid == int(wall_color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood the exterior black area green, then rank enclosed black regions by area and color them alternately red and yellow.

    Input structure:
    - A black background contains a single 4-connected one-pixel-wide orthogonal wall object in one non-black color.
    - The wall partitions black cells into one border-connected exterior region and several fully enclosed regions.
    - Enclosed regions may appear in any spatial order; their areas are the relevant distinguishing feature.

    Rule:
    - Keep all non-black wall pixels unchanged.
    - Find every 4-connected black region.
    - Recolor the black region that touches the grid border to green.
    - Sort the remaining enclosed black regions by area from largest to smallest.
    - Fill even-ranked enclosed regions red and odd-ranked enclosed regions yellow.

    Key insight:
    - The alternating colors are determined by area rank, not by left-to-right or top-to-bottom position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    black_regions = _mask_components(g == Color.BLACK)
    enclosed: list[dict] = []

    for region in black_regions:
        if region["touches_border"]:
            _paint_coords(out, region["coords"], Color.GREEN)
        else:
            enclosed.append(region)

    enclosed.sort(key=lambda region: (-region["area"], region["bbox"][0], region["bbox"][1]))
    for rank, region in enumerate(enclosed):
        fill_color = Color.RED if rank % 2 == 0 else Color.YELLOW
        _paint_coords(out, region["coords"], fill_color)

    return out


def _is_monotonic(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _sample_spans(region_count: int) -> list[int] | None:
    for _ in range(100):
        spans = random.sample(GEN_SPAN_POOL, region_count)
        random.shuffle(spans)
        if not _is_monotonic(spans):
            return spans
    return None


def _sample_total_and_offset(frame_size: int) -> tuple[int, int]:
    extra_limit = min(GEN_EXTRA_MARGIN_MAX, 30 - frame_size)
    if extra_limit < 2:
        raise ValueError("frame too large")
    total = frame_size + random.randint(2, extra_limit)
    offset = random.randint(1, total - frame_size - 1)
    return total, offset


def _draw_partitioned_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    spans: list[int],
    orthogonal_size: int,
    wall_color: int,
    orientation: str,
) -> None:
    if orientation == "vertical":
        frame_h = orthogonal_size + 2
        frame_w = sum(spans) + len(spans) + 1
    else:
        frame_h = sum(spans) + len(spans) + 1
        frame_w = orthogonal_size + 2

    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color

    if orientation == "vertical":
        cursor = c0
        for span in spans[:-1]:
            cursor += span + 1
            grid[r0 : r1 + 1, cursor] = wall_color
    else:
        cursor = r0
        for span in spans[:-1]:
            cursor += span + 1
            grid[cursor, c0 : c1 + 1] = wall_color


def _validate_generated_grid(grid: np.ndarray, wall_color: int, region_count: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = {int(v) for v in np.unique(grid)}
    if colors - {Color.BLACK, int(wall_color)}:
        return False

    if _wall_component_count(grid, wall_color) != 1:
        return False

    black_regions = _mask_components(grid == Color.BLACK)
    enclosed = [region for region in black_regions if not region["touches_border"]]
    if len(enclosed) != region_count:
        return False

    areas = [region["area"] for region in enclosed]
    if len(set(areas)) != len(areas):
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    if not np.array_equal(out[grid == wall_color], grid[grid == wall_color]):
        return False

    out_colors = {int(v) for v in np.unique(out)}
    return {Color.GREEN, Color.RED, Color.YELLOW}.issubset(out_colors)


def _gen_attempt() -> np.ndarray | None:
    region_count = random.choice(GEN_REGION_COUNTS)
    orientation = random.choice(GEN_ORIENTATIONS)
    wall_color = random.choice(GEN_WALL_COLORS)
    spans = _sample_spans(region_count)
    if spans is None:
        return None

    orthogonal_size = random.randint(*GEN_STRIPE_THICKNESS_RANGE)
    if orientation == "vertical":
        frame_h = orthogonal_size + 2
        frame_w = sum(spans) + region_count + 1
    else:
        frame_h = sum(spans) + region_count + 1
        frame_w = orthogonal_size + 2

    if frame_h > 28 or frame_w > 28:
        return None

    total_h, r0 = _sample_total_and_offset(frame_h)
    total_w, c0 = _sample_total_and_offset(frame_w)

    grid = np.full((total_h, total_w), Color.BLACK, dtype=int)
    _draw_partitioned_frame(grid, r0, c0, spans, orthogonal_size, wall_color, orientation)

    if not _validate_generated_grid(grid, wall_color, region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flood-rank-alternation input")

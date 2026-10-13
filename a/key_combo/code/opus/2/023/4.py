from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import Color, colors_present, find_components, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        colors = set(map(int, np.unique(grid)))
        if exclude is not None:
            colors -= {int(color) for color in exclude}
        return sorted(colors)

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ) -> list[dict]:
        assert connectivity == 4
        assert monochromatic
        assert return_format == "dict"

        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        if colors is None:
            palette = colors_present(g, exclude={bg})
        else:
            palette = [int(color) for color in colors if int(color) != int(bg)]

        out: list[dict] = []
        for color in palette:
            mask = g == int(color)
            seen = np.zeros_like(mask, dtype=bool)
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):
                    if not mask[r, c] or seen[r, c]:
                        continue
                    stack = [(r, c)]
                    seen[r, c] = True
                    coords: list[tuple[int, int]] = []
                    while stack:
                        rr, cc = stack.pop()
                        coords.append((rr, cc))
                        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    coords_arr = np.array(coords, dtype=int)
                    r0, c0 = coords_arr.min(axis=0)
                    r1, c1 = coords_arr.max(axis=0)
                    out.append(
                        {
                            "color": int(color),
                            "coords": coords_arr,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "area": int(len(coords)),
                        }
                    )
        return out


BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

HEIGHT_RANGE = (16, 22)
WIDTH_RANGE = (16, 22)
MAX_GENERATION_ATTEMPTS = 500
MIN_BORDER = 1

BLOCK_PATCH = np.ones((2, 2), dtype=int)
PIXEL_PATCH = np.ones((1, 1), dtype=int)
L_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
]


def _is_full_block(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == 4 and int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 2


def _component_stats(grid: np.ndarray) -> dict[int, dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    for color in colors_present(grid, exclude={BG}):
        stats[int(color)] = {
            "blocks": 0,
            "components": 0,
            "pixels": 0,
            "fragments": [],
        }

    for component in find_components(
        np.asarray(grid, dtype=int),
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        color = int(component["color"])
        info = stats.setdefault(color, {"blocks": 0, "components": 0, "pixels": 0, "fragments": []})
        info["components"] = int(info["components"]) + 1
        info["pixels"] = int(info["pixels"]) + int(component["area"])
        if _is_full_block(component):
            info["blocks"] = int(info["blocks"]) + 1
        else:
            area = int(component["area"])
            r0, c0, r1, c1 = component["bbox"]
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if area == 1:
                cast(list[str], info["fragments"]).append("pixel")
            elif area == 3 and height == 2 and width == 2:
                cast(list[str], info["fragments"]).append("L")
            else:
                cast(list[str], info["fragments"]).append("other")
    return stats


def _draw_square_ring(out: np.ndarray, size: int, color: int) -> None:
    n = out.shape[0]
    inset = (n - int(size)) // 2
    lo = inset
    hi = n - inset - 1
    if lo == hi:
        out[lo, lo] = int(color)
        return
    out[lo, lo : hi + 1] = int(color)
    out[hi, lo : hi + 1] = int(color)
    out[lo : hi + 1, lo] = int(color)
    out[lo : hi + 1, hi] = int(color)


def _render_output_from_counts(counts: dict[int, int]) -> np.ndarray:
    layers = sorted(((int(count), int(color)) for color, count in counts.items()), reverse=True)
    side = max(2 * count + 1 for count, _color in layers)
    out = np.full((side, side), BG, dtype=int)
    for count, color in layers:
        _draw_square_ring(out, 2 * count + 1, color)
    return out


def _has_inversion(target: dict[int, int], proxy: dict[int, int]) -> bool:
    colors = list(target)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            ta, tb = int(target[color_a]), int(target[color_b])
            pa, pb = int(proxy[color_a]), int(proxy[color_b])
            if (ta < tb and pa >= pb) or (tb < ta and pb >= pa):
                return True
    return False


def _wrong_numeric_order(colors: list[int], block_counts: dict[int, int]) -> bool:
    correct = [color for color, _count in sorted(block_counts.items(), key=lambda item: item[1], reverse=True)]
    return correct == sorted(colors) or correct == sorted(colors, reverse=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each color's complete 2x2 blocks and rebuild those counts as nested centered square rings.

    Input structure:
    - The grid has black background with several non-background colors scattered as small monochrome 4-connected objects.
    - For each color, some objects are solid 2x2 blocks while other objects are distractor fragments such as isolated pixels or 2x2-minus-one L shapes.
    - Different objects are spatially separate, so each candidate block is identified within its own monochrome connected component.

    Rule:
    - Find every monochrome 4-connected component and keep only components whose bbox is exactly 2x2 and whose area is exactly 4.
    - Count how many such complete 2x2 blocks each color has; map count n to a centered square ring of side length 2n+1.
    - Create a square output whose side equals the largest ring size, then draw the colors' rings from largest count to smallest count as one-cell-thick concentric borders.

    Key insight:
    - The output geometry is not visible directly in the input: it is reconstructed by filtering complete 2x2 components, counting them per color, and converting those counts into ordered ring sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts: dict[int, int] = {}
    for component in find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        if _is_full_block(component):
            color = int(component["color"])
            counts[color] = counts.get(color, 0) + 1

    assert counts, "expected at least one complete 2x2 block"
    assert len(set(counts.values())) == len(counts), "complete-block counts must be unique"
    return _render_output_from_counts(counts)


def _make_patch(kind: str, color: int) -> np.ndarray:
    if kind == "block":
        return BLOCK_PATCH * int(color)
    if kind == "pixel":
        return PIXEL_PATCH * int(color)
    if kind == "L":
        return random.choice(L_SHAPES) * int(color)
    raise ValueError(f"unknown patch kind: {kind}")


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    height, width = grid.shape
    if r0 < MIN_BORDER or c0 < MIN_BORDER or r0 + h > height - MIN_BORDER or c0 + w > width - MIN_BORDER:
        return False
    if np.any((patch != BG) & (grid[r0 : r0 + h, c0 : c0 + w] != BG)):
        return False

    rr0 = max(0, r0 - 1)
    rr1 = min(height, r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(width, c0 + w + 1)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    candidates = [
        (r0, c0)
        for r0 in range(MIN_BORDER, grid.shape[0] - h - MIN_BORDER + 1)
        for c0 in range(MIN_BORDER, grid.shape[1] - w - MIN_BORDER + 1)
        if _can_place(grid, patch, r0, c0)
    ]
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    mask = patch != BG
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    return True


def _sample_block_counts() -> list[int]:
    roll = random.random()
    if roll < 0.35:
        return [1, 2, 3]
    if roll < 0.70:
        return list(random.choice(((1, 2, 4), (1, 3, 4), (2, 3, 4))))
    return [1, 2, 3, 4]


def _assign_colors_to_counts() -> list[dict[str, object]] | None:
    block_counts = _sample_block_counts()
    colors = random.sample(PALETTE, len(block_counts))
    shuffled_counts = block_counts[:]
    random.shuffle(shuffled_counts)
    infos = [{"color": int(color), "blocks": int(count), "fragments": []} for color, count in zip(colors, shuffled_counts)]
    count_map = {int(info["color"]): int(info["blocks"]) for info in infos}
    if _wrong_numeric_order(colors, count_map):
        return None
    return infos


def _assign_fragments(infos: list[dict[str, object]]) -> None:
    ordered = sorted(infos, key=lambda info: int(info["blocks"]))
    boost_idx = random.randrange(len(ordered) - 1)
    boost = ordered[boost_idx]
    compare = ordered[boost_idx + 1]
    boost["fragments"] = ["L", "L", "L"]
    compare["fragments"] = ["pixel"]

    for info in infos:
        if info["fragments"]:
            continue
        fragment_count = random.randint(1, 3)
        info["fragments"] = [random.choice(["pixel", "L"]) for _ in range(fragment_count)]


def _render_input(infos: list[dict[str, object]], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    objects: list[tuple[int, str]] = []
    for info in infos:
        color = int(info["color"])
        for _ in range(int(info["blocks"])):
            objects.append((color, "block"))
        for kind in info["fragments"]:
            objects.append((color, str(kind)))

    random.shuffle(objects)
    objects.sort(key=lambda item: 4 if item[1] == "block" else (3 if item[1] == "L" else 1), reverse=True)

    for color, kind in objects:
        if not _place_patch(grid, _make_patch(kind, color)):
            return None
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        stats = _component_stats(g)
        colors = colors_present(g, exclude={BG})
        if not (3 <= len(colors) <= 4):
            return False

        block_counts = {color: int(stats[color]["blocks"]) for color in colors}
        if not all(1 <= count <= 4 for count in block_counts.values()):
            return False
        if len(set(block_counts.values())) != len(block_counts):
            return False

        if _wrong_numeric_order(colors, block_counts):
            return False

        fragment_lists = {color: list(stats[color]["fragments"]) for color in colors}
        if not all(fragment_lists[color] for color in colors):
            return False

        flat_fragments = [fragment for fragments in fragment_lists.values() for fragment in fragments]
        if "pixel" not in flat_fragments or "L" not in flat_fragments:
            return False
        if any(fragment == "other" for fragment in flat_fragments):
            return False

        component_counts = {color: int(stats[color]["components"]) for color in colors}
        pixel_totals = {color: int(stats[color]["pixels"]) for color in colors}
        if not _has_inversion(block_counts, component_counts):
            return False
        if not _has_inversion(block_counts, pixel_totals):
            return False

        expected = _render_output_from_counts(block_counts)
        if not np.array_equal(main(g), expected):
            return False

        proxy_output = _render_output_from_counts(component_counts)
        if np.array_equal(proxy_output, expected):
            return False

        unique_colors = set(map(int, np.unique(expected)))
        unique_colors.discard(BG)
        if unique_colors != set(colors):
            return False

        max_blocks = max(block_counts.values())
        if expected.shape != (2 * max_blocks + 1, 2 * max_blocks + 1):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    infos = _assign_colors_to_counts()
    if infos is None:
        return None
    _assign_fragments(infos)

    total_objects = sum(int(info["blocks"]) + len(info["fragments"]) for info in infos)
    min_height = max(HEIGHT_RANGE[0], 12 + (total_objects // 3))
    min_width = max(WIDTH_RANGE[0], 12 + (total_objects // 3))
    if min_height > HEIGHT_RANGE[1] or min_width > WIDTH_RANGE[1]:
        return None

    height = random.randint(min_height, HEIGHT_RANGE[1])
    width = random.randint(min_width, WIDTH_RANGE[1])
    grid = _render_input(infos, height, width)
    if grid is None:
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

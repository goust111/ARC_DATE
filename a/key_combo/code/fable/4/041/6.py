from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


def _extract_target_and_mapping(grid: np.ndarray, bg: int) -> tuple[dict, dict[int, int]]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    target_candidates = []
    legend_candidates = []

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        area = int(comp["area"])
        if area == 2 and height == 1 and width == 2:
            legend_candidates.append(comp)
            continue
        if area == height * width and height >= 3 and width >= 3:
            target_candidates.append(comp)
            continue
        raise AssertionError("unexpected foreground component shape")

    assert len(target_candidates) == 1, "expected exactly one solid rectangular target"
    assert legend_candidates, "expected at least one legend domino"

    target = target_candidates[0]
    crop = crop_bbox(grid, target["bbox"])
    assert np.all(crop != bg), "target rectangle must be solid"

    mapping: dict[int, int] = {}
    for comp in legend_candidates:
        r0, c0, r1, c1 = comp["bbox"]
        assert r0 == r1 and c1 == c0 + 1, "legend must be a horizontal domino"
        replacement = int(grid[r0, c0])
        source = int(grid[r0, c0 + 1])
        assert replacement != source, "legend colors must differ"
        assert source not in mapping, "each source color must map uniquely"
        mapping[source] = replacement

    return target, mapping


def _resolve_color(color: int, mapping: dict[int, int], memo: dict[int, int]) -> int:
    color = int(color)
    if color in memo:
        return memo[color]

    seen: set[int] = set()
    trail: list[int] = []
    current = color
    while current in mapping:
        if current in memo:
            current = memo[current]
            break
        if current in seen:
            raise AssertionError("legend mapping must be acyclic")
        seen.add(current)
        trail.append(current)
        current = int(mapping[current])

    resolved = int(current)
    for node in trail:
        memo[int(node)] = resolved
    memo[color] = resolved
    return resolved


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the solid target rectangle and follow each legend substitution chain to its terminal color.

    Input structure:
    - A dominant background separates one solid rectangular multicolor target object
      from several isolated horizontal 1x2 legend dominos.
    - Each legend is read left-to-right as replacement color then source color, and
      some source colors are themselves used as replacement colors in other legends.

    Rule:
    - Find the solid rectangular target and output only its bounding-box crop.
    - Inside the crop, whenever a pixel color is a legend source, replace it by that
      legend's replacement and keep repeating until the color is no longer a source.
    - Colors with no outgoing legend stay unchanged; if a chain ends at the target's
      base color, those pixels simply become indistinguishable from the base.

    Key insight:
    - The legends define a composable color-substitution graph rather than a single
      one-step lookup table, so each source color must be resolved all the way to its
      terminal color before painting the crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    target, mapping = _extract_target_and_mapping(g, bg)
    crop = crop_bbox(g, target["bbox"])

    memo: dict[int, int] = {}
    out = crop.copy()
    for color in np.unique(crop):
        resolved = _resolve_color(int(color), mapping, memo)
        if resolved != int(color):
            out[crop == int(color)] = resolved
    return out


GEN_BG = 0
GEN_RETRIES = 400
GRID_HEIGHT_RANGE = (14, 26)
GRID_WIDTH_RANGE = (14, 26)
TARGET_HEIGHT_RANGE = (6, 10)
TARGET_WIDTH_RANGE = (6, 10)
LEGEND_COUNT_RANGE = (3, 5)
TARGET_COLOR_COUNT_RANGE = (2, 4)
PATTERN_SHAPES = (
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
)
PALETTE = tuple(range(1, 10))


def _build_mapping() -> tuple[int, dict[int, int], list[int]]:
    n_legends = random.randint(*LEGEND_COUNT_RANGE)
    colors = random.sample(PALETTE, k=n_legends + 2)

    base = int(colors[0])
    sources = [int(c) for c in colors[1 : 1 + n_legends]]
    new_terminal = int(colors[-1])

    chain_leaf = sources[0]
    recolor_leaf = sources[1]
    hub = sources[-1]
    extras = sources[2:-1]

    mapping: dict[int, int] = {
        chain_leaf: hub,
        recolor_leaf: new_terminal,
        hub: base,
    }

    for src in extras:
        mapping[int(src)] = int(random.choice([hub, new_terminal, base]))

    target_sources = [chain_leaf, recolor_leaf]
    optional = [src for src in [hub, *extras] if src not in target_sources]
    random.shuffle(optional)
    max_sources = min(len(sources), random.randint(*TARGET_COLOR_COUNT_RANGE))
    while len(target_sources) < max_sources and optional:
        target_sources.append(optional.pop())
    random.shuffle(target_sources)
    return base, mapping, target_sources


def _make_target_patch(
    height: int,
    width: int,
    base: int,
    target_sources: list[int],
) -> np.ndarray | None:
    target = np.full((height, width), int(base), dtype=int)
    pattern_canvas = np.zeros((height, width), dtype=int)
    pattern_area = 0

    for index, color in enumerate(target_sources):
        shapes = [shape for shape in PATTERN_SHAPES if shape[0] <= height and shape[1] <= width]
        random.shuffle(shapes)
        placed = False

        for patch_h, patch_w in shapes:
            patch = np.full((patch_h, patch_w), index + 1, dtype=int)
            try:
                r0, c0 = random_free_location(
                    pattern_canvas,
                    patch,
                    transparent=0,
                    border=0,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            paste(pattern_canvas, patch, r0, c0)
            pattern_area += patch_h * patch_w
            target[pattern_canvas == index + 1] = int(color)
            placed = True
            break

        if not placed:
            return None

    total_area = height * width
    base_fraction = (total_area - pattern_area) / total_area
    if not (0.45 <= base_fraction <= 0.82):
        return None
    return target


def _place_legends(canvas: np.ndarray, mapping: dict[int, int]) -> bool:
    items = list(mapping.items())
    random.shuffle(items)
    for source, replacement in items:
        legend = np.array([[int(replacement), int(source)]], dtype=int)
        try:
            r0, c0 = random_free_location(
                canvas,
                legend,
                transparent=GEN_BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(canvas, legend, r0, c0)
    return True


def _gen_attempt() -> np.ndarray | None:
    base, mapping, target_sources = _build_mapping()
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    target_h = random.randint(*TARGET_HEIGHT_RANGE)
    target_w = random.randint(*TARGET_WIDTH_RANGE)
    if target_h >= height - 2 or target_w >= width - 2:
        return None

    target = _make_target_patch(target_h, target_w, base, target_sources)
    if target is None:
        return None

    canvas = np.full((height, width), GEN_BG, dtype=int)
    try:
        r0, c0 = random_free_location(
            canvas,
            target,
            transparent=GEN_BG,
            border=1,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(canvas, target, r0, c0)

    if not _place_legends(canvas, mapping):
        return None

    try:
        out = main(canvas)
    except AssertionError:
        return None

    if out.shape != target.shape:
        return None
    if np.array_equal(out, target):
        return None
    if not np.any(out == base):
        return None
    if not np.any(out != base):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

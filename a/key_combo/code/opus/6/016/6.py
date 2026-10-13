from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import components_mask, neighbors4, paste  # noqa: E402


BG = 8
BLUE = 1
BASELINE = 7
SPRITE_COLORS = [2, 3, 4, 5, 6, 9]
MAX_GEN_ATTEMPTS = 400


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    assert np.all(grid[-1, :] == BASELINE), "expected a full bottom baseline"
    scene = grid[:-1, :]
    sprite_mask = scene != BG
    components = components_mask(sprite_mask, connectivity=4)

    sprites: list[dict] = []
    for coords in components:
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        blue_count = 0
        main_colors: set[int] = set()

        for r, c in coords:
            color = int(grid[r, c])
            sprite[int(r) - r0, int(c) - c0] = color
            if color == BLUE:
                blue_count += 1
            else:
                main_colors.add(color)

        assert len(main_colors) == 1, "each sprite must have one non-blue main color"
        main_color = int(next(iter(main_colors)))
        sprites.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "sprite": sprite,
                "area": int(len(coords)),
                "blue_count": int(blue_count),
                "main_color": main_color,
                "nonblue_area": int(len(coords) - blue_count),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
    return sprites


def _packed_output(
    shape: tuple[int, int],
    sprites: list[dict],
    *,
    sort_key,
    gap_mode: str,
    recolor_blue: bool,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[-1, :] = BASELINE
    baseline_row = h - 1
    col = 0
    ordered = sorted(sprites, key=sort_key)

    for idx, sprite_info in enumerate(ordered):
        sprite = sprite_info["sprite"].copy()
        if recolor_blue:
            sprite[sprite == BLUE] = int(sprite_info["main_color"])
        sh, sw = sprite.shape
        assert baseline_row - sh >= 0, "sprite must fit above the baseline"
        assert col + sw <= w, "packed sprite row exceeds grid width"
        paste(out, sprite, baseline_row - sh, col, transparent=BG)
        if idx < len(ordered) - 1:
            if gap_mode == "encoded":
                gap = max(1, int(sprite_info["blue_count"]))
            elif gap_mode == "unit":
                gap = 1
            else:
                raise ValueError(f"unknown gap_mode: {gap_mode}")
            col += sw + gap
        else:
            col += sw

    assert col <= w, "packed layout must stay within the canvas"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the sprites by total area, pack them above the baseline, and erase the blue encodings after use.

    Input structure:
    - The background is azure (8), and the bottom row is a full orange (7) baseline.
    - Above the baseline are 3-5 irregular 4-connected sprites; each sprite uses one main color plus optional blue (1) pixels.
    - Blue pixels are part of the sprite itself, so both total sprite area and blue-pixel count are visible in the input.

    Rule:
    - Find each sprite as a 4-connected component in the rows above the baseline.
    - For every sprite, count its total area including blue pixels and also count its blue pixels B.
    - Sort sprites by total area from smallest to largest.
    - Repack their bounding-box sprites from left to right with bottoms aligned to the row just above the baseline.
    - The first sprite starts at column 0; every next sprite starts after a horizontal gap equal to the previous sprite's blue count, except B=0 uses a gap of 1.
    - After packing, recolor every blue pixel inside each sprite to that sprite's main color; keep the baseline and background unchanged.

    Key insight:
    - Blue pixels have a double role: they increase the sortable area of a sprite and also encode the size of the gap that follows that sprite in the packed output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprites = _extract_sprites(g)
    assert 3 <= len(sprites) <= 5

    areas = [int(sprite["area"]) for sprite in sprites]
    blue_counts = [int(sprite["blue_count"]) for sprite in sprites]
    assert len(set(areas)) == len(areas), "sprite areas must be unique"
    assert len(set(blue_counts)) == len(blue_counts), "blue counts must be unique"
    assert all(0 <= b <= 3 for b in blue_counts), "blue counts must be in 0..3"

    return _packed_output(
        g.shape,
        sprites,
        sort_key=lambda item: item["area"],
        gap_mode="encoded",
        recolor_blue=True,
    )


def _random_connected_mask(area: int) -> np.ndarray | None:
    for _ in range(80):
        canvas = np.zeros((6, 6), dtype=bool)
        start = (random.randint(1, 4), random.randint(1, 4))
        cells = {start}
        canvas[start] = True

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < 6 and 0 <= nc < 6 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break

            weights = []
            for nr, nc in frontier:
                degree = sum((rr, cc) in cells for rr, cc in neighbors4(nr, nc))
                center_bias = 6 - abs(nr - 2.5) - abs(nc - 2.5)
                weights.append(max(1, 3 * degree + int(center_bias)))
            new_cell = random.choices(frontier, weights=weights, k=1)[0]
            cells.add(new_cell)
            canvas[new_cell] = True

        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        mask = canvas[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1].copy()
        mh, mw = mask.shape
        bbox_area = mh * mw
        if area < 3 or area > 8:
            continue
        if mh <= 1 or mw <= 1:
            continue
        if mh > 4 or mw > 5:
            continue
        if area == bbox_area:
            continue
        return mask
    return None


def _valid_blue_subset(mask: np.ndarray, subset: tuple[tuple[int, int], ...]) -> bool:
    blue_cells = set(subset)
    shape_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    for r, c in blue_cells:
        if not any(
            (nr, nc) in shape_cells and (nr, nc) not in blue_cells
            for nr, nc in neighbors4(r, c)
        ):
            return False
    return True


def _paint_sprite(mask: np.ndarray, main_color: int, blue_count: int) -> np.ndarray | None:
    sprite = np.full(mask.shape, BG, dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    blue_cells: set[tuple[int, int]] = set()

    if blue_count:
        candidates = list(combinations(coords, blue_count))
        random.shuffle(candidates)
        for subset in candidates:
            if _valid_blue_subset(mask, subset):
                blue_cells = set(subset)
                break
        if len(blue_cells) != blue_count:
            return None

    for r, c in coords:
        sprite[r, c] = BLUE if (r, c) in blue_cells else main_color
    return sprite


def _sample_blue_counts(count: int) -> list[int]:
    options = [
        combo
        for combo in combinations([0, 1, 2, 3], count)
        if 0 in combo and max(combo) >= 2
    ]
    weights = [5 if 3 in combo else 3 for combo in options]
    return list(random.choices(options, weights=weights, k=1)[0])


def _build_sprite_specs() -> list[dict] | None:
    count = random.choice([3, 4])
    blue_counts = _sample_blue_counts(count)
    random.shuffle(blue_counts)

    area_sets = [
        combo
        for combo in combinations(range(3, 9), count)
        if max(combo) - min(combo) <= 4
    ]
    random.shuffle(area_sets)
    chosen_areas: list[int] | None = None
    for area_set in area_sets:
        permuted = list(area_set)
        random.shuffle(permuted)
        if all(area >= blue_count + 2 for area, blue_count in zip(permuted, blue_counts)):
            chosen_areas = permuted
            break
    if chosen_areas is None:
        return None

    colors = random.sample(SPRITE_COLORS, count)
    specs: list[dict] = []
    for area, blue_count, color in zip(chosen_areas, blue_counts, colors):
        mask = _random_connected_mask(area)
        if mask is None:
            return None
        sprite = _paint_sprite(mask, color, blue_count)
        if sprite is None:
            return None
        specs.append(
            {
                "sprite": sprite,
                "area": int(area),
                "blue_count": int(blue_count),
                "main_color": int(color),
                "nonblue_area": int(area - blue_count),
                "bbox_area": int(mask.shape[0] * mask.shape[1]),
            }
        )

    if 0 not in blue_counts or max(blue_counts) < 2:
        return None

    ordered_by_area = sorted(specs, key=lambda item: item["area"])
    ordered_by_nonblue = sorted(specs, key=lambda item: item["nonblue_area"])
    ordered_by_bbox = sorted(specs, key=lambda item: item["bbox_area"])
    if [item["area"] for item in ordered_by_area] == [item["area"] for item in ordered_by_nonblue]:
        return None
    if [item["area"] for item in ordered_by_area] == [item["area"] for item in ordered_by_bbox]:
        return None

    used_gaps = [max(1, int(item["blue_count"])) for item in ordered_by_area[:-1]]
    if len(set(used_gaps)) < 2:
        return None
    return specs


def _required_width(sprites: list[dict]) -> int:
    ordered = sorted(sprites, key=lambda item: item["area"])
    total = sum(int(sprite["sprite"].shape[1]) for sprite in ordered)
    total += sum(max(1, int(sprite["blue_count"])) for sprite in ordered[:-1])
    return total


def _choose_canvas(sprites: list[dict]) -> tuple[int, int] | None:
    needed_w = _required_width(sprites)
    max_sprite_h = max(int(sprite["sprite"].shape[0]) for sprite in sprites)
    min_rows = max(15, max_sprite_h + 6)
    max_rows = min(22, min_rows + 5)
    min_cols = max(18, needed_w + 2)
    max_cols = min(28, needed_w + 8)
    if min_rows > max_rows or min_cols > max_cols:
        return None
    return random.randint(min_rows, max_rows), random.randint(min_cols, max_cols)


def _can_place(
    occupied: set[tuple[int, int]],
    sprite: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    cells = [(top + r, left + c) for r, c in np.argwhere(sprite != BG)]
    if not cells:
        return False
    if min(r for r, _ in cells) < 0 or min(c for _, c in cells) < 0:
        return False
    if max(r for r, _ in cells) >= h - 2 or max(c for _, c in cells) >= w:
        return False
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_input(rows: int, cols: int, sprites: list[dict]) -> np.ndarray | None:
    grid = np.full((rows, cols), BG, dtype=int)
    grid[-1, :] = BASELINE
    occupied: set[tuple[int, int]] = {(rows - 1, c) for c in range(cols)}
    shuffled = sprites[:]
    random.shuffle(shuffled)

    for sprite_info in shuffled:
        sprite = sprite_info["sprite"]
        sh, sw = sprite.shape
        candidates = [
            (r, c)
            for r in range(0, rows - 1 - sh + 1)
            for c in range(0, cols - sw + 1)
            if _can_place(occupied, sprite, r, c, (rows, cols))
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        paste(grid, sprite, top, left, transparent=BG)
        for r, c in np.argwhere(sprite != BG):
            occupied.add((top + int(r), left + int(c)))
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        sprites = _extract_sprites(inp)
        if not (3 <= len(sprites) <= 4):
            return False
        if len({sprite["area"] for sprite in sprites}) != len(sprites):
            return False
        if len({sprite["blue_count"] for sprite in sprites}) != len(sprites):
            return False
        if any(sprite["blue_count"] > 3 for sprite in sprites):
            return False
        if not any(sprite["blue_count"] == 0 for sprite in sprites):
            return False
        if not any(sprite["blue_count"] >= 2 for sprite in sprites):
            return False

        actual = _packed_output(
            inp.shape,
            sprites,
            sort_key=lambda item: item["area"],
            gap_mode="encoded",
            recolor_blue=True,
        )
        if not np.array_equal(actual, out):
            return False

        wrong_nonblue = _packed_output(
            inp.shape,
            sprites,
            sort_key=lambda item: item["nonblue_area"],
            gap_mode="encoded",
            recolor_blue=True,
        )
        wrong_bbox = _packed_output(
            inp.shape,
            sprites,
            sort_key=lambda item: item["bbox_area"],
            gap_mode="encoded",
            recolor_blue=True,
        )
        wrong_gap = _packed_output(
            inp.shape,
            sprites,
            sort_key=lambda item: item["area"],
            gap_mode="unit",
            recolor_blue=True,
        )
        wrong_keep_blue = _packed_output(
            inp.shape,
            sprites,
            sort_key=lambda item: item["area"],
            gap_mode="encoded",
            recolor_blue=False,
        )
        if np.array_equal(out, wrong_nonblue):
            return False
        if np.array_equal(out, wrong_bbox):
            return False
        if np.array_equal(out, wrong_gap):
            return False
        if np.array_equal(out, wrong_keep_blue):
            return False

        area_order = [sprite["area"] for sprite in sorted(sprites, key=lambda item: item["area"])]
        nonblue_order = [
            sprite["area"] for sprite in sorted(sprites, key=lambda item: item["nonblue_area"])
        ]
        if area_order == nonblue_order:
            return False

        sorted_sprites = sorted(sprites, key=lambda item: item["area"])
        used_gaps = [max(1, int(sprite["blue_count"])) for sprite in sorted_sprites[:-1]]
        if len(set(used_gaps)) < 2:
            return False

        if np.any(out[:-1, :] == BLUE):
            return False
        if not np.all(out[-1, :] == BASELINE):
            return False
        if np.array_equal(inp, out):
            return False
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    sprites = _build_sprite_specs()
    if sprites is None:
        return None
    canvas = _choose_canvas(sprites)
    if canvas is None:
        return None
    rows, cols = canvas
    out = _packed_output(
        (rows, cols),
        sprites,
        sort_key=lambda item: item["area"],
        gap_mode="encoded",
        recolor_blue=True,
    )
    inp = _place_input(rows, cols, sprites)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if not _semantic_checks(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

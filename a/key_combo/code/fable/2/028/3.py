from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_ROWS = (22, 30)
GRID_COLS = (20, 30)
SPRITE_HEIGHT_RANGE = (2, 5)
SPRITE_WIDTH_CHOICES = tuple(range(3, 9))
PLACEMENT_PADDING = 1
SHAPE_RETRIES = 80
GENERATION_RETRIES = 500
_GEN_CALL_COUNT = 0


def _int_unique_colors(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(set(map(int, np.unique(grid))) - {int(bg)})


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    sprites: list[dict] = []
    bars: list[dict] = []

    for color in _int_unique_colors(g, bg):
        coords = np.argwhere(g == color)
        assert coords.size > 0
        (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

        if r0 == r1:
            span = g[r0, c0 : c1 + 1]
            assert np.all((span == bg) | (span == color)), "row object must be monochrome with background holes"
            hole_cols = [idx for idx, value in enumerate(span) if int(value) == bg]
            if hole_cols:
                assert int(span[0]) == color and int(span[-1]) == color, "bar holes must be interior to the span"
                bars.append(
                    {
                        "color": int(color),
                        "row": int(r0),
                        "c0": int(c0),
                        "c1": int(c1),
                        "width": int(c1 - c0 + 1),
                        "hole_cols": tuple(hole_cols),
                    }
                )
                continue

        comps = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        assert len(comps) == 1, "each sprite color must form exactly one connected component"
        comp = comps[0]
        mask = crop_bbox(g == color, comp["bbox"])
        sprites.append(
            {
                "color": int(color),
                "bbox": tuple(map(int, comp["bbox"])),
                "mask": mask.astype(bool),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
            }
        )

    assert len(sprites) >= 2, "expected multiple sprites"
    assert bars, "expected at least one punched bar"

    width_to_sprite: dict[int, dict] = {}
    for sprite in sprites:
        width = int(sprite["width"])
        assert width not in width_to_sprite, "sprite widths must be unique"
        width_to_sprite[width] = sprite

    for bar in bars:
        assert int(bar["width"]) in width_to_sprite, "each bar must match exactly one sprite width"

    return bg, sprites, bars, width_to_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each punched horizontal bar with a recolored copy of the uniquely same-width sprite, clipped by the bar's hole columns.

    Input structure:
    - The grid has a black background, several monochrome sprites, and several one-cell-thick horizontal punched bars.
    - Each bar occupies one row, uses one non-sprite color, and may contain background holes inside its column span.
    - Every bar span width matches exactly one sprite width, and the bar lies either above or below that sprite.

    Rule:
    - Erase every punched bar.
    - For each bar, find the unique sprite whose bounding-box width equals the bar's full span width.
    - Stamp a full copy of that sprite at the bar's span, recolor it to the bar color, and delete every copied cell in the columns where the bar has holes.
    - If the bar is below its sprite, align the copy's top row to the bar row and expand downward; if the bar is above its sprite, align the copy's bottom row to the bar row and expand upward.
    - Keep the original sprites unchanged.

    Key insight:
    - The punched bar is a program card: its width selects the sprite, its color recolors the copy, its hole columns clip the copy, and its side relative to the source sprite sets the copy's vertical expansion direction.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, bars, width_to_sprite = _parse_scene(g)

    out = g.copy()
    for bar in bars:
        out[bar["row"], bar["c0"] : bar["c1"] + 1] = bg

    for bar in bars:
        sprite = width_to_sprite[int(bar["width"])]
        mask = sprite["mask"]
        h, w = mask.shape
        assert w == int(bar["width"])

        sprite_top, _sprite_left, sprite_bottom, _sprite_right = sprite["bbox"]
        if int(bar["row"]) < int(sprite_top):
            r0 = int(bar["row"]) - h + 1
        elif int(bar["row"]) > int(sprite_bottom):
            r0 = int(bar["row"])
        else:
            raise AssertionError("bar must be strictly above or below its matched sprite")
        c0 = int(bar["c0"])
        assert 0 <= r0 and r0 + h <= g.shape[0]
        assert 0 <= c0 and c0 + w <= g.shape[1]

        patch = np.full((h, w), bg, dtype=int)
        patch[mask] = int(bar["color"])
        patch[:, list(bar["hole_cols"])] = bg

        patch_mask = patch != bg
        region = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(region[patch_mask] != bg), "copy placement must not overwrite existing objects"
        paste(out, patch, r0, c0, transparent=bg, clip=False)

    return out


def _rect_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PADDING) -> bool:
    h_lim, w_lim = occupied.shape
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(h_lim, int(r0) + int(h) + pad)
    cc1 = min(w_lim, int(c0) + int(w) + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)] = True


def _sprite_mask(height: int, width: int) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        intervals: list[tuple[int, int]] = []
        anchor = random.randrange(height)
        for col in range(width):
            if col == 0:
                top = anchor - random.randint(0, min(1, anchor))
                bottom = anchor + random.randint(0, min(2, height - 1 - anchor))
            else:
                overlap = random.randint(intervals[-1][0], intervals[-1][1])
                top = overlap - random.randint(0, min(2, overlap))
                bottom = overlap + random.randint(0, min(2, height - 1 - overlap))
            intervals.append((int(top), int(bottom)))

        mask = np.zeros((height, width), dtype=bool)
        for col, (top, bottom) in enumerate(intervals):
            mask[top : bottom + 1, col] = True

        if mask.all():
            continue
        if np.count_nonzero(mask.sum(axis=0)) != width:
            continue
        if np.count_nonzero(mask.sum(axis=1)) < 2:
            continue
        if int(mask.sum()) <= width:
            continue
        return mask

    raise RuntimeError("failed to generate a non-degenerate sprite mask")


def _paint_sprite(grid: np.ndarray, sprite: dict) -> None:
    patch = np.full((sprite["height"], sprite["width"]), BG, dtype=int)
    patch[sprite["mask"]] = int(sprite["color"])
    paste(grid, patch, int(sprite["r0"]), int(sprite["c0"]), transparent=BG, clip=False)


def _paint_bar(grid: np.ndarray, bar: dict) -> None:
    line = np.full((1, int(bar["width"])), BG, dtype=int)
    keep_cols = [c for c in range(int(bar["width"])) if c not in set(bar["hole_cols"])]
    line[0, keep_cols] = int(bar["color"])
    paste(grid, line, int(bar["row"]), int(bar["c0"]), clip=False)


def _expected_output(shape: tuple[int, int], sprites: list[dict], bars: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    sprite_by_index = {int(sprite["index"]): sprite for sprite in sprites}

    for sprite in sprites:
        _paint_sprite(out, sprite)

    for bar in bars:
        sprite = sprite_by_index[int(bar["sprite_index"])]
        patch = np.full((sprite["height"], sprite["width"]), BG, dtype=int)
        patch[sprite["mask"]] = int(bar["color"])
        patch[:, list(bar["hole_cols"])] = BG
        paste(out, patch, int(bar["copy_r0"]), int(bar["c0"]), transparent=BG, clip=False)

    return out


def _render_input(shape: tuple[int, int], sprites: list[dict], bars: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(grid, sprite)
    for bar in bars:
        _paint_bar(grid, bar)
    return grid


def _vertical_gap(row: int, sprite: dict) -> int:
    top = int(sprite["r0"])
    bottom = int(sprite["r0"]) + int(sprite["height"]) - 1
    if row < top:
        return int(top - row)
    if row > bottom:
        return int(row - bottom)
    return 0


def _phase_parameters(phase: int) -> tuple[int, int, int]:
    if phase == 0:
        sprite_count = random.choice((3, 4))
        used_count = sprite_count - 1
        total_bars = min(5, used_count + random.choice((1, 2)))
    elif phase == 1:
        sprite_count = random.choice((3, 4))
        used_count = sprite_count
        total_bars = min(5, sprite_count + 1)
    else:
        sprite_count = 4
        used_count = 3
        total_bars = 3
    return int(sprite_count), int(used_count), int(total_bars)


def _bar_plan(sprite_count: int, used_count: int, total_bars: int, phase: int) -> tuple[list[int], list[str]]:
    used_indices = random.sample(range(sprite_count), used_count)
    assignments = list(used_indices)
    while len(assignments) < total_bars:
        assignments.append(random.choice(used_indices))
    random.shuffle(assignments)

    if phase == 1:
        bias_above = 0.7
    elif phase == 2:
        bias_above = 0.3
    else:
        bias_above = 0.5

    sides = ["above", "below"]
    for _ in range(total_bars - 2):
        sides.append("above" if random.random() < bias_above else "below")
    random.shuffle(sides)
    return assignments, sides


def _place_sprites(height: int, width: int, sprites: list[dict], bars_by_sprite: dict[int, list[dict]]) -> tuple[list[dict], np.ndarray] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed = [dict(sprite) for sprite in sprites]

    order = sorted(
        range(len(placed)),
        key=lambda idx: (
            int(bool(any(bar["side"] == "above" for bar in bars_by_sprite.get(idx, []))))
            + int(bool(any(bar["side"] == "below" for bar in bars_by_sprite.get(idx, [])))),
            placed[idx]["height"] * placed[idx]["width"],
        ),
        reverse=True,
    )

    for idx in order:
        sprite = placed[idx]
        h = int(sprite["height"])
        w = int(sprite["width"])
        needs_above = any(bar["side"] == "above" for bar in bars_by_sprite.get(idx, []))
        needs_below = any(bar["side"] == "below" for bar in bars_by_sprite.get(idx, []))

        row_min = h + PLACEMENT_PADDING if needs_above else 0
        row_max = (height - 2 * h - PLACEMENT_PADDING) if needs_below else (height - h)
        if row_max < row_min:
            return None

        candidates = [
            (r0, c0)
            for r0 in range(row_min, row_max + 1)
            for c0 in range(0, width - w + 1)
            if _rect_free(occupied, r0, c0, h, w)
        ]
        if not candidates:
            return None

        random.shuffle(candidates)
        r0, c0 = candidates[0]
        sprite["r0"] = int(r0)
        sprite["c0"] = int(c0)
        _mark_rect(occupied, r0, c0, h, w)

    return placed, occupied


def _place_bars(height: int, width: int, sprites: list[dict], bars: list[dict], occupied: np.ndarray) -> list[dict] | None:
    placed = [dict(bar) for bar in bars]
    order = sorted(
        range(len(placed)),
        key=lambda idx: sprites[int(placed[idx]["sprite_index"])]["height"] * sprites[int(placed[idx]["sprite_index"])]["width"],
        reverse=True,
    )

    shifted_alignment = False

    for idx in order:
        bar = placed[idx]
        sprite = sprites[int(bar["sprite_index"])]
        h = int(sprite["height"])
        w = int(sprite["width"])
        sprite_top = int(sprite["r0"])
        sprite_bottom = sprite_top + h - 1

        if bar["side"] == "above":
            row_values = range(0, sprite_top - PLACEMENT_PADDING - h + 1)
            if not list(row_values):
                return None
            candidates = [
                (copy_r0, c0)
                for copy_r0 in row_values
                for c0 in range(0, width - w + 1)
                if _rect_free(occupied, copy_r0, c0, h, w)
            ]
            if not candidates:
                return None
            better = [pos for pos in candidates if pos[1] != int(sprite["c0"])]
            random.shuffle(better)
            random.shuffle(candidates)
            copy_r0, c0 = (better[0] if better and not shifted_alignment else candidates[0])
            bar["copy_r0"] = int(copy_r0)
            bar["row"] = int(copy_r0 + h - 1)
        else:
            row_values = range(sprite_bottom + PLACEMENT_PADDING + 1, height - h + 1)
            if not list(row_values):
                return None
            candidates = [
                (copy_r0, c0)
                for copy_r0 in row_values
                for c0 in range(0, width - w + 1)
                if _rect_free(occupied, copy_r0, c0, h, w)
            ]
            if not candidates:
                return None
            better = [pos for pos in candidates if pos[1] != int(sprite["c0"])]
            random.shuffle(better)
            random.shuffle(candidates)
            copy_r0, c0 = (better[0] if better and not shifted_alignment else candidates[0])
            bar["copy_r0"] = int(copy_r0)
            bar["row"] = int(copy_r0)

        bar["c0"] = int(c0)
        bar["c1"] = int(c0 + w - 1)
        _mark_rect(occupied, int(bar["copy_r0"]), int(c0), h, w)
        if int(c0) != int(sprite["c0"]):
            shifted_alignment = True

    return placed


def _assign_holes(bars: list[dict], sprites: list[dict]) -> None:
    widest_bar = max(
        range(len(bars)),
        key=lambda idx: sprites[int(bars[idx]["sprite_index"])]["width"],
    )

    for idx, bar in enumerate(bars):
        width = int(sprites[int(bar["sprite_index"])]["width"])
        interior = list(range(1, width - 1))
        assert interior, "bars need an interior hole column"
        if idx == widest_bar and len(interior) >= 2:
            hole_count = random.randint(2, min(3, len(interior)))
        else:
            hole_count = random.randint(1, min(3, len(interior)))
        bar["width"] = int(width)
        bar["hole_cols"] = tuple(sorted(random.sample(interior, hole_count)))


def _spec_semantics_ok(inp: np.ndarray, out: np.ndarray, sprites: list[dict], bars: list[dict]) -> bool:
    if np.array_equal(inp, out):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if int((inp == BG).sum()) <= inp.size // 2:
        return False

    widths = {int(sprite["width"]) for sprite in sprites}
    if len(widths) != len(sprites):
        return False
    if {bar["side"] for bar in bars} != {"above", "below"}:
        return False
    if not any(len(bar["hole_cols"]) >= 2 for bar in bars):
        return False
    if not any(int(bar["c0"]) != int(sprites[int(bar["sprite_index"])]["c0"]) for bar in bars):
        return False

    counts_by_sprite = {int(sprite["index"]): 0 for sprite in sprites}
    for bar in bars:
        counts_by_sprite[int(bar["sprite_index"])] += 1
    if all(count == 1 for count in counts_by_sprite.values()):
        return False

    if not any(
        any(
            idx != int(bar["sprite_index"])
            and _vertical_gap(int(bar["row"]), sprite) < _vertical_gap(int(bar["row"]), sprites[int(bar["sprite_index"])])
            for idx, sprite in enumerate(sprites)
        )
        for bar in bars
    ):
        return False

    for bar in bars:
        sprite = sprites[int(bar["sprite_index"])]
        patch = np.full((sprite["height"], sprite["width"]), BG, dtype=int)
        patch[sprite["mask"]] = int(bar["color"])
        before = int(np.count_nonzero(patch))
        patch[:, list(bar["hole_cols"])] = BG
        after = int(np.count_nonzero(patch))
        if not (0 < after < before):
            return False

    return True


def _attempt(phase: int) -> np.ndarray | None:
    sprite_count, used_count, total_bars = _phase_parameters(phase)
    widths = random.sample(SPRITE_WIDTH_CHOICES, sprite_count)
    heights = [random.randint(*SPRITE_HEIGHT_RANGE) for _ in range(sprite_count)]

    object_colors = random.sample(COLOR_POOL, sprite_count + total_bars)
    sprite_colors = object_colors[:sprite_count]
    bar_colors = object_colors[sprite_count:]

    assignments, sides = _bar_plan(sprite_count, used_count, total_bars, phase)
    bars = [
        {"index": idx, "sprite_index": int(sprite_idx), "side": side, "color": int(color)}
        for idx, (sprite_idx, side, color) in enumerate(zip(assignments, sides, bar_colors))
    ]
    bars_by_sprite: dict[int, list[dict]] = {idx: [] for idx in range(sprite_count)}
    for bar in bars:
        bars_by_sprite[int(bar["sprite_index"])].append(bar)

    sprites = []
    for idx, (width, height, color) in enumerate(zip(widths, heights, sprite_colors)):
        sprites.append(
            {
                "index": int(idx),
                "color": int(color),
                "width": int(width),
                "height": int(height),
                "mask": _sprite_mask(int(height), int(width)),
            }
        )

    grid_h = random.randint(*GRID_ROWS)
    grid_w = random.randint(*GRID_COLS)

    placed_sprites_result = _place_sprites(grid_h, grid_w, sprites, bars_by_sprite)
    if placed_sprites_result is None:
        return None
    placed_sprites, occupied = placed_sprites_result

    placed_bars = _place_bars(grid_h, grid_w, placed_sprites, bars, occupied)
    if placed_bars is None:
        return None
    _assign_holes(placed_bars, placed_sprites)

    inp = _render_input((grid_h, grid_w), placed_sprites, placed_bars)
    expected = _expected_output((grid_h, grid_w), placed_sprites, placed_bars)

    try:
        solved = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if not _spec_semantics_ok(inp, expected, placed_sprites, placed_bars):
        return None

    parsed_bg, parsed_sprites, parsed_bars, _ = _parse_scene(inp)
    if parsed_bg != BG or len(parsed_sprites) != len(placed_sprites) or len(parsed_bars) != len(placed_bars):
        return None

    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    phase = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1

    for _ in range(GENERATION_RETRIES):
        grid = _attempt(phase)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid punched-program-bar ARC input")

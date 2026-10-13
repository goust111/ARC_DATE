from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, components_mask, crop_bbox, get_bg, in_bounds, neighbors4, paste


GRAY = Color.GRAY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _template_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    return (grid != int(bg)) & (grid != GRAY)


def _extract_template(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, tuple[int, int]]:
    comps = components_mask(_template_mask(grid, bg), connectivity=4)
    assert len(comps) == 1, "expected one non-gray template object"
    coords = comps[0]
    box = bbox_coords(coords)
    sprite = crop_bbox(grid, box)
    cols = tuple(colors_present(sprite, exclude={bg}))
    assert len(cols) == 2, "template must use exactly two colors"
    assert GRAY not in cols, "gray is reserved for indicator and counters"
    return coords, box, sprite, (int(cols[0]), int(cols[1]))


def _find_indicator(
    template_mask: np.ndarray,
    box: tuple[int, int, int, int],
    gray_coords: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    touching: list[tuple[int, int, tuple[int, int]]] = []

    for r_raw, c_raw in gray_coords:
        r = int(r_raw)
        c = int(c_raw)
        has_template_neighbor = any(
            in_bounds(template_mask.shape, nr, nc) and bool(template_mask[nr, nc])
            for nr, nc in neighbors4(r, c)
        )
        if not has_template_neighbor:
            continue

        dirs: list[tuple[int, int]] = []
        if r == r0 - 1 and c0 <= c <= c1:
            dirs.append((-1, 0))
        if r == r1 + 1 and c0 <= c <= c1:
            dirs.append((1, 0))
        if c == c0 - 1 and r0 <= r <= r1:
            dirs.append((0, -1))
        if c == c1 + 1 and r0 <= r <= r1:
            dirs.append((0, 1))

        assert len(dirs) == 1, "indicator must sit just outside one template bbox side"
        touching.append((r, c, dirs[0]))

    assert len(touching) == 1, "expected exactly one touching gray indicator"
    r, c, direction = touching[0]

    for r_raw, c_raw in gray_coords:
        rr = int(r_raw)
        cc = int(c_raw)
        if (rr, cc) == (r, c):
            continue
        assert not any(
            in_bounds(template_mask.shape, nr, nc) and bool(template_mask[nr, nc])
            for nr, nc in neighbors4(rr, cc)
        ), "counter dots must not touch the template"

    return (r, c), direction


def _swap_template_colors(sprite: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    c0, c1 = map(int, colors)
    swapped = sprite.copy()
    mask0 = swapped == c0
    mask1 = swapped == c1
    swapped[mask0] = c1
    swapped[mask1] = c0
    swapped[swapped == bg] = bg
    return swapped


def _step_size(sprite: np.ndarray, direction: tuple[int, int]) -> int:
    dr, dc = map(int, direction)
    return int(sprite.shape[0] if dr else sprite.shape[1])


def _build_output_from_params(
    shape: tuple[int, int],
    bg: int,
    sprite: np.ndarray,
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
    counter_count: int,
    colors: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, _r1, _c1 = map(int, box)
    paste(out, sprite, r0, c0, transparent=bg)

    swapped = _swap_template_colors(sprite, colors, bg)
    step = _step_size(sprite, direction)
    dr, dc = map(int, direction)
    for copy_idx in range(1, int(counter_count) + 1):
        patch = swapped if copy_idx % 2 == 1 else sprite
        paste(
            out,
            patch,
            r0 + dr * step * copy_idx,
            c0 + dc * step * copy_idx,
            transparent=bg,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile a two-color template in the indicator direction, using the free gray-dot count as the number of copies and swapping the two template colors on odd-numbered copies.

    Input structure:
    - The background is a single non-black color.
    - Exactly one connected template object uses two non-gray colors.
    - One gray pixel touches one side of the template bbox and indicates one of the four cardinal copy directions.
    - The remaining gray pixels are isolated counter dots away from the template and away from the copy strip.

    Rule:
    - Count the gray pixels that are not the touching indicator, then erase every gray pixel.
    - Keep the original template unchanged.
    - Paste that many additional template copies in the indicated direction, stepping by one template bbox width or height each time so the copies form a flush strip.
    - Copy 1, 3, 5, ... swaps the template's two colors, while copy 2, 4, 6, ... keeps the original coloring.

    Key insight:
    - Gray plays two roles at once: the touching gray pixel gives the strip direction, while the detached gray dots give the exact copy count; the copies also alternate between original and color-swapped palette states.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    bg = int(get_bg(g))
    gray_coords = np.argwhere(g == GRAY)
    assert gray_coords.size > 0, "expected at least one gray pixel"

    _template_coords, box, sprite, colors = _extract_template(g, bg)
    mask = _template_mask(g, bg)
    _indicator, direction = _find_indicator(mask, box, gray_coords)

    return _build_output_from_params(
        g.shape,
        bg,
        sprite,
        box,
        direction,
        int(gray_coords.shape[0] - 1),
        colors,
    )


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _bridge_positions(mask: np.ndarray, direction: tuple[int, int]) -> list[int]:
    dr, dc = map(int, direction)
    if dc:
        return [int(r) for r in range(mask.shape[0]) if bool(mask[r, 0]) and bool(mask[r, -1])]
    return [int(c) for c in range(mask.shape[1]) if bool(mask[0, c]) and bool(mask[-1, c])]


def _sample_mask(rng: np.random.Generator, h: int, w: int, direction: tuple[int, int]) -> np.ndarray | None:
    min_area = max(4, (h * w + 2) // 3)
    for _ in range(120):
        target = int(rng.integers(min_area, h * w + 1))
        mask = np.zeros((h, w), dtype=bool)
        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True

        if _touches_all_sides(mask) and _bridge_positions(mask, direction):
            return mask
    return None


def _sample_sprite(
    rng: np.random.Generator,
    bg: int,
    direction: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    h = int(rng.integers(2, 6))
    w = int(rng.integers(2, 6))
    mask = _sample_mask(rng, h, w, direction)
    if mask is None:
        return None

    colors = sorted(int(c) for c in rng.choice([c for c in PALETTE if c != bg], size=2, replace=False))
    coords = np.argwhere(mask)
    min_count = 2 if len(coords) >= 4 else 1
    count0 = int(rng.integers(min_count, len(coords) - min_count + 1))
    chosen = set(int(i) for i in rng.choice(len(coords), size=count0, replace=False))

    sprite = np.full((h, w), int(bg), dtype=int)
    for idx, (r, c) in enumerate(coords):
        sprite[int(r), int(c)] = int(colors[0] if idx in chosen else colors[1])
    return sprite, mask, (int(colors[0]), int(colors[1]))


def _choose_counter_count(rng: np.random.Generator, sprite: np.ndarray, direction: tuple[int, int]) -> int | None:
    step = _step_size(sprite, direction)
    max_copies = 30 // step - 1
    max_copies = min(max_copies, 5)
    if max_copies < 2:
        return None
    return int(rng.integers(2, max_copies + 1))


def _choose_canvas(
    rng: np.random.Generator,
    sprite: np.ndarray,
    counter_count: int,
    direction: tuple[int, int],
) -> tuple[int, int]:
    h, w = map(int, sprite.shape)
    dr, dc = map(int, direction)
    strip_h = h if dc else h * (counter_count + 1)
    strip_w = w * (counter_count + 1) if dc else w

    min_h = max(strip_h, h + (6 if dc else 0), 10)
    min_w = max(strip_w, w + (6 if dr else 0), 10)
    if min_h > 30 or min_w > 30:
        raise ValueError("canvas would exceed ARC limit")
    return int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))


def _place_strip(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sprite: np.ndarray,
    counter_count: int,
    direction: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    h, w = map(int, sprite.shape)
    H, W = map(int, shape)
    dr, dc = map(int, direction)
    strip_h = h if dc else h * (counter_count + 1)
    strip_w = w * (counter_count + 1) if dc else w

    strip_r0 = int(rng.integers(0, H - strip_h + 1))
    strip_c0 = int(rng.integers(0, W - strip_w + 1))

    if (dr, dc) == (0, 1):
        box = (strip_r0, strip_c0, strip_r0 + h - 1, strip_c0 + w - 1)
    elif (dr, dc) == (0, -1):
        box = (strip_r0, strip_c0 + counter_count * w, strip_r0 + h - 1, strip_c0 + strip_w - 1)
    elif (dr, dc) == (1, 0):
        box = (strip_r0, strip_c0, strip_r0 + h - 1, strip_c0 + w - 1)
    else:
        box = (strip_r0 + counter_count * h, strip_c0, strip_r0 + strip_h - 1, strip_c0 + w - 1)

    strip_box = (strip_r0, strip_c0, strip_r0 + strip_h - 1, strip_c0 + strip_w - 1)
    return box, strip_box


def _indicator_from_mask(
    rng: np.random.Generator,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    options = _bridge_positions(mask, direction)
    pick = int(options[int(rng.integers(0, len(options)))])
    dr, dc = map(int, direction)
    if (dr, dc) == (0, 1):
        return r0 + pick, c1 + 1
    if (dr, dc) == (0, -1):
        return r0 + pick, c0 - 1
    if (dr, dc) == (1, 0):
        return r1 + 1, c0 + pick
    return r0 - 1, c0 + pick


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _counter_dots(
    rng: np.random.Generator,
    shape: tuple[int, int],
    strip_box: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    H, W = map(int, shape)
    r0, c0, r1, c1 = map(int, strip_box)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(H - 1, r1 + 1)
    ec1 = min(W - 1, c1 + 1)

    candidates = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if not (er0 <= r <= er1 and ec0 <= c <= ec1)
    ]
    rng.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if any(_too_close(cell, other) for other in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    sprite: np.ndarray,
    box: tuple[int, int, int, int],
    strip_box: tuple[int, int, int, int],
    direction: tuple[int, int],
    counter_dots: list[tuple[int, int]],
    indicator: tuple[int, int],
    colors: tuple[int, int],
) -> bool:
    bg = int(get_bg(grid))
    parsed_coords, parsed_box, parsed_sprite, parsed_colors = _extract_template(grid, bg)
    del parsed_coords
    if parsed_box != tuple(map(int, box)):
        return False
    if not np.array_equal(parsed_sprite, sprite):
        return False
    if parsed_colors != tuple(map(int, colors)):
        return False

    gray_coords = np.argwhere(grid == GRAY)
    if gray_coords.shape[0] != len(counter_dots) + 1:
        return False
    parsed_indicator, parsed_direction = _find_indicator(_template_mask(grid, bg), box, gray_coords)
    if parsed_indicator != tuple(map(int, indicator)) or parsed_direction != tuple(map(int, direction)):
        return False

    r0, c0, r1, c1 = map(int, strip_box)
    er0 = max(0, r0 - 1)
    ec0 = max(0, c0 - 1)
    er1 = min(grid.shape[0] - 1, r1 + 1)
    ec1 = min(grid.shape[1] - 1, c1 + 1)
    for r, c in counter_dots:
        if er0 <= r <= er1 and ec0 <= c <= ec1:
            return False
        if _too_close((r, c), indicator):
            return False

    if not np.array_equal(main(grid), expected):
        return False
    return True


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice(PALETTE))
    direction = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]

    sprite_bundle = _sample_sprite(rng, bg, direction)
    if sprite_bundle is None:
        return None
    sprite, mask, colors = sprite_bundle

    counter_count = _choose_counter_count(rng, sprite, direction)
    if counter_count is None:
        return None

    try:
        shape = _choose_canvas(rng, sprite, counter_count, direction)
    except ValueError:
        return None

    box, strip_box = _place_strip(rng, shape, sprite, counter_count, direction)
    indicator = _indicator_from_mask(rng, mask, box, direction)
    counter_dots = _counter_dots(rng, shape, strip_box, counter_count)
    if counter_dots is None:
        return None

    H, W = map(int, shape)
    inp = np.full((H, W), int(bg), dtype=int)
    paste(inp, sprite, box[0], box[1], transparent=bg)
    inp[indicator] = GRAY
    for r, c in counter_dots:
        inp[int(r), int(c)] = GRAY

    expected = _build_output_from_params(shape, bg, sprite, box, direction, counter_count, colors)
    if not _semantic_checks(inp, expected, sprite, box, strip_box, direction, counter_dots, indicator, colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")

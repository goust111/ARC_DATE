from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MOTIF_COLORS = [BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
GRID_ROWS_RANGE = (18, 24)
GRID_COLS_RANGE = (18, 24)
BAR_COUNT_CHOICES = (2, 3)
BAR_LENGTH_CHOICES = (3, 4, 6)
MOTIF_SIDE_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_motif_cell(value: int, bg: int) -> bool:
    return int(value) not in (int(bg), GREY)


def _find_flag(
    g: np.ndarray,
    endpoint: tuple[int, int],
    preferred: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    h, w = g.shape
    pr, pc = preferred
    if 0 <= pr < h and 0 <= pc < w and int(g[pr, pc]) in (RED, GREEN):
        return (pr, pc), int(g[pr, pc])

    r, c = endpoint
    candidates: list[tuple[tuple[int, int], int]] = []
    for nr, nc in _neighbors4(r, c):
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in blocked:
            continue
        color = int(g[nr, nc])
        if color in (RED, GREEN):
            candidates.append(((nr, nc), color))
    assert len(candidates) == 1, f"expected one endpoint flag, got {candidates}"
    return candidates[0]


def _parse_vertical(g: np.ndarray, bg: int, box: tuple[int, int, int, int]) -> dict:
    h, w = g.shape
    r0, c0, r1, c1 = box
    assert c0 == c1 and r0 < r1
    c = c0
    assert c + 1 < w and _is_motif_cell(g[r1, c + 1], bg)

    motif_w = 0
    while c + 1 + motif_w < w and _is_motif_cell(g[r1, c + 1 + motif_w], bg):
        motif_w += 1
    assert motif_w > 0

    motif_h = 0
    while r1 - motif_h >= r0:
        strip = g[r1 - motif_h, c + 1 : c + 1 + motif_w]
        if not np.all([_is_motif_cell(v, bg) for v in strip]):
            break
        motif_h += 1
    length = r1 - r0 + 1
    assert motif_h > 0 and length % motif_h == 0

    tile = g[r1 - motif_h + 1 : r1 + 1, c + 1 : c + 1 + motif_w].copy()
    fill = np.tile(tile, (length // motif_h, 1))
    flag_pos, flag_color = _find_flag(g, (r0, c), (r0 - 1, c), {(r0 + 1, c)})
    return {
        "orientation": "v",
        "bbox": box,
        "length": length,
        "motif_h": motif_h,
        "motif_w": motif_w,
        "tile": tile,
        "fill": fill,
        "flag_pos": flag_pos,
        "flag_color": flag_color,
    }


def _parse_horizontal(g: np.ndarray, bg: int, box: tuple[int, int, int, int]) -> dict:
    h, w = g.shape
    r0, c0, r1, c1 = box
    assert r0 == r1 and c0 < c1
    r = r0
    assert r + 1 < h and _is_motif_cell(g[r + 1, c0], bg)

    motif_h = 0
    while r + 1 + motif_h < h and _is_motif_cell(g[r + 1 + motif_h, c0], bg):
        motif_h += 1
    assert motif_h > 0

    motif_w = 0
    while c0 + motif_w <= c1:
        strip = g[r + 1 : r + 1 + motif_h, c0 + motif_w]
        if not np.all([_is_motif_cell(v, bg) for v in strip]):
            break
        motif_w += 1
    length = c1 - c0 + 1
    assert motif_w > 0 and length % motif_w == 0

    tile = g[r + 1 : r + 1 + motif_h, c0 : c0 + motif_w].copy()
    fill = np.tile(tile, (1, length // motif_w))
    flag_pos, flag_color = _find_flag(g, (r, c1), (r, c1 + 1), {(r, c1 - 1)})
    return {
        "orientation": "h",
        "bbox": box,
        "length": length,
        "motif_h": motif_h,
        "motif_w": motif_w,
        "tile": tile,
        "fill": fill,
        "flag_pos": flag_pos,
        "flag_color": flag_color,
    }


def _parse_bars(g: np.ndarray) -> list[dict]:
    bg = _get_bg(g)
    bars: list[dict] = []
    for coords in _components(g == GREY):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        area = len(coords)
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), f"grey component is not a solid bar: {box}"
        if c0 == c1 and r0 < r1:
            bars.append(_parse_vertical(g, bg, box))
        elif r0 == r1 and c0 < c1:
            bars.append(_parse_horizontal(g, bg, box))
        else:
            raise AssertionError(f"grey component is not a straight bar: {box}")
    assert bars, "expected at least one ruler bar"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each motif beside its grey ruler, and use the far-end flag to decide whether a mirrored copy appears on the opposite side.

    Input structure:
    - The grid has a black background and contains 2-3 straight grey ruler bars.
    - Each vertical bar has a small colored rectangular motif attached at its lower-right end; each horizontal bar has one attached below its left end.
    - The ruler's opposite endpoint has a single red or green flag pixel touching that endpoint.

    Rule:
    - For every vertical bar, tile its attached motif to fill the full same-height rectangle immediately to the right of the bar; for every horizontal bar, tile its motif to fill the full same-width rectangle immediately below the bar.
    - If the opposite-end flag is red, mirror that tiled rectangle across the ruler axis and place the mirrored copy on the other side of the ruler.
    - If the flag is green, keep only the original one-sided tiling.
    - Erase every flag pixel to black while keeping the ruler bar and the original attached motif.

    Key insight:
    - The grey ruler encodes repeat extent, while the color of the single endpoint flag encodes whether the completed tiled result must also be mirrored across the ruler.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for bar in _parse_bars(g):
        (fr, fc), flag_color = bar["flag_pos"], bar["flag_color"]
        out[fr, fc] = BLACK
        r0, c0, r1, c1 = bar["bbox"]
        fill = bar["fill"]
        motif_h = bar["motif_h"]
        motif_w = bar["motif_w"]

        if bar["orientation"] == "v":
            out[r0 : r1 + 1, c0 + 1 : c0 + 1 + motif_w] = fill
            if flag_color == RED:
                assert c0 - motif_w >= 0
                out[r0 : r1 + 1, c0 - motif_w : c0] = np.fliplr(fill)
        else:
            out[r0 + 1 : r0 + 1 + motif_h, c0 : c1 + 1] = fill
            if flag_color == RED:
                assert r0 - motif_h >= 0
                out[r0 - motif_h : r0, c0 : c1 + 1] = np.flipud(fill)

    return out


def _random_tile(height: int, width: int) -> np.ndarray:
    area = height * width
    n_colors = random.randint(2, min(3, area))
    palette = random.sample(MOTIF_COLORS, n_colors)
    tile = np.array(random.choices(palette, k=area), dtype=int).reshape(height, width)
    for idx, color in zip(random.sample(range(area), n_colors), palette):
        tile.flat[idx] = color
    return tile


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _make_spec(orientation: str, flag_color: int) -> dict:
    length = random.choice(BAR_LENGTH_CHOICES)
    if orientation == "v":
        motif_h = random.choice([d for d in MOTIF_SIDE_CHOICES if length % d == 0])
        motif_w = random.choice(MOTIF_SIDE_CHOICES)
    else:
        motif_h = random.choice(MOTIF_SIDE_CHOICES)
        motif_w = random.choice([d for d in MOTIF_SIDE_CHOICES if length % d == 0])
    return {
        "orientation": orientation,
        "flag_color": int(flag_color),
        "length": int(length),
        "motif_h": int(motif_h),
        "motif_w": int(motif_w),
        "tile": _random_tile(int(motif_h), int(motif_w)),
    }


def _orientation_list(count: int) -> list[str]:
    if count == 2:
        items = ["v", "h"]
    else:
        items = random.choice([["v", "v", "h"], ["v", "h", "h"]]).copy()
    random.shuffle(items)
    return items


def _flag_color_list(count: int) -> list[int]:
    if count == 2:
        items = [RED, GREEN]
    else:
        items = [RED, GREEN, random.choice([RED, GREEN])]
    random.shuffle(items)
    return items


def _placement_masks(spec: dict, shape: tuple[int, int], r: int, c: int) -> dict | None:
    h, w = shape
    length = int(spec["length"])
    motif_h = int(spec["motif_h"])
    motif_w = int(spec["motif_w"])
    orientation = spec["orientation"]

    bar = np.zeros((h, w), dtype=bool)
    motif = np.zeros((h, w), dtype=bool)
    symmetric_fill = np.zeros((h, w), dtype=bool)
    input_nonbg = np.zeros((h, w), dtype=bool)

    if orientation == "v":
        r1 = r + length - 1
        if not (1 <= r and r1 < h and motif_w <= c and c + motif_w < w):
            return None
        bar[r : r1 + 1, c] = True
        symmetric_fill[r : r1 + 1, c - motif_w : c] = True
        symmetric_fill[r : r1 + 1, c + 1 : c + 1 + motif_w] = True
        motif[r1 - motif_h + 1 : r1 + 1, c + 1 : c + 1 + motif_w] = True
        flag = (r - 1, c)
    else:
        c1 = c + length - 1
        if not (motif_h <= r and r + motif_h < h and c1 + 1 < w):
            return None
        bar[r, c : c1 + 1] = True
        symmetric_fill[r - motif_h : r, c : c1 + 1] = True
        symmetric_fill[r + 1 : r + 1 + motif_h, c : c1 + 1] = True
        motif[r + 1 : r + 1 + motif_h, c : c + motif_w] = True
        flag = (r, c1 + 1)

    input_nonbg |= bar | motif
    input_nonbg[flag] = True
    reserved = _dilate8(bar | symmetric_fill | input_nonbg)
    return {
        "bar": bar,
        "motif": motif,
        "input_nonbg": input_nonbg,
        "reserved": reserved,
        "flag": flag,
    }


def _paint_spec(grid: np.ndarray, spec: dict, r: int, c: int) -> None:
    length = int(spec["length"])
    motif_h = int(spec["motif_h"])
    motif_w = int(spec["motif_w"])
    tile = spec["tile"]
    if spec["orientation"] == "v":
        r1 = r + length - 1
        grid[r : r1 + 1, c] = GREY
        grid[r1 - motif_h + 1 : r1 + 1, c + 1 : c + 1 + motif_w] = tile
        grid[r - 1, c] = int(spec["flag_color"])
    else:
        c1 = c + length - 1
        grid[r, c : c1 + 1] = GREY
        grid[r + 1 : r + 1 + motif_h, c : c + motif_w] = tile
        grid[r, c1 + 1] = int(spec["flag_color"])


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(
        specs,
        key=lambda s: (2 * int(s["motif_w"]) + 1) * int(s["length"])
        if s["orientation"] == "v"
        else (2 * int(s["motif_h"]) + 1) * int(s["length"]),
        reverse=True,
    )

    for spec in order:
        candidates: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                masks = _placement_masks(spec, shape, r, c)
                if masks is None:
                    continue
                if np.any(masks["reserved"] & occupied):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        masks = _placement_masks(spec, shape, r, c)
        assert masks is not None
        occupied |= masks["reserved"]
        _paint_spec(grid, spec, r, c)
    return grid


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        bars = _parse_bars(inp)
        if len(bars) not in (2, 3):
            return False
        if {bar["orientation"] for bar in bars} != {"v", "h"}:
            return False
        flag_colors = {bar["flag_color"] for bar in bars}
        if flag_colors != {RED, GREEN}:
            return False

        out = main(inp)
        bg = _get_bg(inp)
        if np.array_equal(inp, out):
            return False

        for bar in bars:
            r0, c0, r1, c1 = bar["bbox"]
            motif_h = bar["motif_h"]
            motif_w = bar["motif_w"]
            fill = bar["fill"]
            fr, fc = bar["flag_pos"]
            if int(out[fr, fc]) != BLACK:
                return False

            if bar["orientation"] == "v":
                if not np.array_equal(out[r0 : r1 + 1, c0 + 1 : c0 + 1 + motif_w], fill):
                    return False
                if not np.all(inp[r0 : r1 - motif_h + 1, c0 + 1 : c0 + 1 + motif_w] == bg):
                    return False
                left = out[r0 : r1 + 1, c0 - motif_w : c0]
                if bar["flag_color"] == RED:
                    if not np.array_equal(left, np.fliplr(fill)):
                        return False
                else:
                    if not np.all(left == bg):
                        return False
            else:
                if not np.array_equal(out[r0 + 1 : r0 + 1 + motif_h, c0 : c1 + 1], fill):
                    return False
                if not np.all(inp[r0 + 1 : r0 + 1 + motif_h, c0 + bar["motif_w"] : c1 + 1] == bg):
                    return False
                top = out[r0 - motif_h : r0, c0 : c1 + 1]
                if bar["flag_color"] == RED:
                    if not np.array_equal(top, np.flipud(fill)):
                        return False
                else:
                    if not np.all(top == bg):
                        return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = (
            random.randint(*GRID_ROWS_RANGE),
            random.randint(*GRID_COLS_RANGE),
        )
        count = random.choice(BAR_COUNT_CHOICES)
        specs = [
            _make_spec(orientation, flag_color)
            for orientation, flag_color in zip(_orientation_list(count), _flag_color_list(count))
        ]
        grid = _place_specs(shape, specs)
        if grid is not None and _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid ruler-mirror-tiling input")

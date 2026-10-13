from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.flat)))
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _sprite_patch(rel_cells: frozenset[Coord], shape: tuple[int, int], color: int, bg: int) -> np.ndarray:
    patch = np.full(shape, int(bg), dtype=int)
    for r, c in rel_cells:
        patch[r, c] = int(color)
    return patch


def _sprite_spec(panel: np.ndarray, color: int) -> dict:
    mask = panel == int(color)
    comps = components_mask(mask, connectivity=8)
    assert len(comps) == 1, "each sprite color must form exactly one component per panel"
    coords = comps[0]
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    rel = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return {
        "color": int(color),
        "coords": coords,
        "bbox": bbox,
        "rel": rel,
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
    }


def _bounce_axis(pos: int, delta: int, limit: int) -> int:
    if limit <= 0:
        return 0
    raw = int(pos) + int(delta)
    period = 2 * int(limit)
    raw %= period
    if raw < 0:
        raw += period
    if raw > limit:
        raw = period - raw
    return int(raw)


def _single_bounce_axis(pos: int, delta: int, limit: int) -> tuple[int, bool]:
    raw = int(pos) + int(delta)
    if 0 <= raw <= limit:
        return int(raw), False
    if raw < 0:
        return int(-raw), True
    return int(2 * limit - raw), True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate a third sprite frame from two stacked panels, reflecting any out-of-bounds motion off the walls.

    Input structure:
    - The grid has odd height and one full-width separator row at the center, splitting it into equal top and bottom panels.
    - Each panel contains the same set of uniquely colored monochrome sprites on one background color.
    - For each color, the bottom-panel sprite is a pure translation of the top-panel sprite and keeps the same shape.

    Rule:
    - Pair sprites across panels by color and recover each translation vector from the top-panel bbox position to the bottom-panel bbox position.
    - Starting from the bottom-panel position, translate each sprite once more by that same vector to predict the third frame.
    - If that extra step would push the sprite bbox past a panel wall by k cells on one axis, reflect that axis and place the bbox k cells from that wall; row and column reflections are handled independently.

    Key insight:
    - The two panels are consecutive animation frames, so the hidden object to infer is each sprite's velocity rather than a static difference; the wall reflection only appears in the unseen third frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    total_h, panel_w = g.shape
    assert total_h % 2 == 1, "input must split into two equal panels plus one separator row"

    sep_r = total_h // 2
    panel_h = sep_r
    assert np.all(g[sep_r] == g[sep_r, 0]), "expected a full-width separator row"
    sep_color = int(g[sep_r, 0])
    bg = get_bg(g)
    assert sep_color != bg, "separator row must be non-background"

    top = g[:sep_r, :]
    bottom = g[sep_r + 1 :, :]
    top_colors = set(colors_present(top, exclude={bg}))
    bottom_colors = set(colors_present(bottom, exclude={bg}))
    assert sep_color not in top_colors and sep_color not in bottom_colors, "separator color may only appear on the divider"
    assert top_colors == bottom_colors, "top and bottom panels must contain the same sprite colors"
    assert top_colors, "expected at least one sprite"

    out = np.full((panel_h, panel_w), bg, dtype=int)

    for color in sorted(top_colors):
        top_spec = _sprite_spec(top, color)
        bottom_spec = _sprite_spec(bottom, color)
        assert top_spec["rel"] == bottom_spec["rel"], "paired sprites must keep the same shape"

        tr0, tc0, _tr1, _tc1 = top_spec["bbox"]
        br0, bc0, _br1, _bc1 = bottom_spec["bbox"]
        dr = int(br0 - tr0)
        dc = int(bc0 - tc0)

        translated = {
            (int(r + dr), int(c + dc))
            for r, c in top_spec["coords"]
        }
        assert translated == {tuple(map(int, cell)) for cell in bottom_spec["coords"]}, "bottom sprite must be a pure translation of the top sprite"

        sprite_h, sprite_w = bottom_spec["shape"]
        row_limit = panel_h - sprite_h
        col_limit = panel_w - sprite_w
        out_r0 = _bounce_axis(br0, dr, row_limit)
        out_c0 = _bounce_axis(bc0, dc, col_limit)

        patch = _sprite_patch(bottom_spec["rel"], bottom_spec["shape"], color, bg)
        view = out[out_r0 : out_r0 + sprite_h, out_c0 : out_c0 + sprite_w]
        mask = patch != bg
        assert not np.any((view != bg) & mask), "predicted sprites may not overlap"
        paste(out, patch, out_r0, out_c0, transparent=bg)

    return out


GEN_MAX_ATTEMPTS = 400
SPRITE_ATTEMPTS = 250
SPRITE_COUNT = 4
SPRITE_BRANCHES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]
PANEL_H_RANGE = (9, 14)
PANEL_W_RANGE = (11, 16)
SHAPE_H_RANGE = (1, 4)
SHAPE_W_RANGE = (1, 4)
MAX_AREA = 5
SPEED_VALUES = (-3, -2, -1, 1, 2, 3)
PADDING = 1


def _rand_range(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _random_connected_mask() -> np.ndarray:
    for _ in range(100):
        h = _rand_range(SHAPE_H_RANGE)
        w = _rand_range(SHAPE_W_RANGE)
        if h == 1 and w == 1:
            continue
        area = random.randint(2, min(MAX_AREA, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier: list[Coord] = []
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        return mask
    raise RuntimeError("failed to sample a connected sprite mask")


def _axis_options(limit: int, want_bounce: bool) -> list[tuple[int, int, int, int, bool]]:
    options: list[tuple[int, int, int, int, bool]] = []
    for delta in SPEED_VALUES:
        for top0 in range(limit + 1):
            bottom = top0 + delta
            if not (0 <= bottom <= limit):
                continue
            raw = bottom + delta
            bounced = raw < 0 or raw > limit
            if bounced != want_bounce:
                continue
            final0, one_bounce = _single_bounce_axis(bottom, delta, limit)
            if final0 != _bounce_axis(bottom, delta, limit):
                continue
            options.append((top0, delta, bottom, final0, one_bounce))
    return options


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> set[Coord]:
    cells = set()
    for r, c in np.argwhere(mask):
        cells.add((int(r0 + r), int(c0 + c)))
    return cells


def _padded_cells(cells: set[Coord], shape: tuple[int, int], padding: int = PADDING) -> set[Coord]:
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _free_in(blocked: np.ndarray, cells: set[Coord]) -> bool:
    return all(not blocked[r, c] for r, c in cells)


def _mark(blocked: np.ndarray, cells: set[Coord]) -> None:
    for r, c in cells:
        blocked[r, c] = True


def _paint_cells(grid: np.ndarray, cells: set[Coord], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _sample_sprite_spec(
    mask: np.ndarray,
    panel_h: int,
    panel_w: int,
    row_bounce: bool,
    col_bounce: bool,
    blocked_top: np.ndarray,
    blocked_bottom: np.ndarray,
    blocked_out: np.ndarray,
) -> dict | None:
    sprite_h, sprite_w = mask.shape
    row_limit = panel_h - sprite_h
    col_limit = panel_w - sprite_w
    if row_limit < 0 or col_limit < 0:
        return None

    row_opts = _axis_options(row_limit, row_bounce)
    col_opts = _axis_options(col_limit, col_bounce)
    if not row_opts or not col_opts:
        return None

    for _ in range(SPRITE_ATTEMPTS):
        tr0, dr, br0, or0, row_flag = random.choice(row_opts)
        tc0, dc, bc0, oc0, col_flag = random.choice(col_opts)
        top_cells = _cells_from_mask(mask, tr0, tc0)
        bottom_cells = _cells_from_mask(mask, br0, bc0)
        out_cells = _cells_from_mask(mask, or0, oc0)
        if not _free_in(blocked_top, top_cells):
            continue
        if not _free_in(blocked_bottom, bottom_cells):
            continue
        if not _free_in(blocked_out, out_cells):
            continue
        return {
            "mask": mask,
            "rel": frozenset((int(r), int(c)) for r, c in np.argwhere(mask)),
            "shape": mask.shape,
            "top0": (tr0, tc0),
            "bottom0": (br0, bc0),
            "out0": (or0, oc0),
            "velocity": (dr, dc),
            "branch": (row_flag, col_flag),
            "top_cells": top_cells,
            "bottom_cells": bottom_cells,
            "out_cells": out_cells,
        }
    return None


def _build_expected(panel_h: int, panel_w: int, bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full((panel_h, panel_w), int(bg), dtype=int)
    for spec in specs:
        patch = _sprite_patch(spec["rel"], spec["shape"], spec["color"], bg)
        r0, c0 = spec["out0"]
        view = out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != int(bg)
        if np.any((view != int(bg)) & mask):
            raise AssertionError("expected output sprites overlap")
        paste(out, patch, r0, c0, transparent=bg)
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, specs: list[dict], bg: int, sep_color: int) -> bool:
    panel_h, panel_w = expected.shape
    if inp.shape != (2 * panel_h + 1, panel_w):
        return False
    if not np.all(inp[panel_h] == int(sep_color)):
        return False
    top = inp[:panel_h, :]
    bottom = inp[panel_h + 1 :, :]
    if get_bg(inp) != int(bg):
        return False

    for spec in specs:
        color = int(spec["color"])
        if {tuple(map(int, cell)) for cell in np.argwhere(top == color)} != spec["top_cells"]:
            return False
        if {tuple(map(int, cell)) for cell in np.argwhere(bottom == color)} != spec["bottom_cells"]:
            return False
        if {tuple(map(int, cell)) for cell in np.argwhere(expected == color)} != spec["out_cells"]:
            return False

        top_bbox = bbox_coords(np.array(sorted(spec["top_cells"]), dtype=int))
        bottom_bbox = bbox_coords(np.array(sorted(spec["bottom_cells"]), dtype=int))
        dr = int(bottom_bbox[0] - top_bbox[0])
        dc = int(bottom_bbox[1] - top_bbox[1])
        if (dr, dc) != tuple(spec["velocity"]):
            return False

        row_limit = panel_h - spec["shape"][0]
        col_limit = panel_w - spec["shape"][1]
        row_final, row_bounce = _single_bounce_axis(bottom_bbox[0], dr, row_limit)
        col_final, col_bounce = _single_bounce_axis(bottom_bbox[1], dc, col_limit)
        if (row_bounce, col_bounce) != tuple(spec["branch"]):
            return False
        if (row_final, col_final) != tuple(spec["out0"]):
            return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    panel_h = _rand_range(PANEL_H_RANGE)
    panel_w = _rand_range(PANEL_W_RANGE)

    palette = list(range(10))
    bg = random.choice(palette)
    other_colors = [color for color in palette if color != bg]
    random.shuffle(other_colors)
    sep_color = other_colors[0]
    sprite_colors = other_colors[1 : 1 + SPRITE_COUNT]

    top = np.full((panel_h, panel_w), int(bg), dtype=int)
    bottom = np.full((panel_h, panel_w), int(bg), dtype=int)
    blocked_top = np.zeros((panel_h, panel_w), dtype=bool)
    blocked_bottom = np.zeros((panel_h, panel_w), dtype=bool)
    blocked_out = np.zeros((panel_h, panel_w), dtype=bool)

    specs: list[dict] = []
    branch_order = list(SPRITE_BRANCHES)
    random.shuffle(branch_order)
    branch_order.sort(key=lambda pair: pair[0] + pair[1], reverse=True)

    for row_bounce, col_bounce in branch_order:
        placed = None
        for _ in range(SPRITE_ATTEMPTS):
            mask = _random_connected_mask()
            spec = _sample_sprite_spec(
                mask,
                panel_h,
                panel_w,
                row_bounce,
                col_bounce,
                blocked_top,
                blocked_bottom,
                blocked_out,
            )
            if spec is None:
                continue
            placed = spec
            break
        if placed is None:
            return None
        specs.append(placed)
        _mark(blocked_top, _padded_cells(placed["top_cells"], (panel_h, panel_w)))
        _mark(blocked_bottom, _padded_cells(placed["bottom_cells"], (panel_h, panel_w)))
        _mark(blocked_out, _padded_cells(placed["out_cells"], (panel_h, panel_w)))

    random.shuffle(sprite_colors)
    for color, spec in zip(sprite_colors, specs):
        spec["color"] = int(color)
        _paint_cells(top, spec["top_cells"], color)
        _paint_cells(bottom, spec["bottom_cells"], color)

    expected = _build_expected(panel_h, panel_w, bg, specs)

    inp = np.full((2 * panel_h + 1, panel_w), int(bg), dtype=int)
    inp[:panel_h, :] = top
    inp[panel_h, :] = int(sep_color)
    inp[panel_h + 1 :, :] = bottom

    if not _validate_semantics(inp, expected, specs, bg, sep_color):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-frame bounce sample")

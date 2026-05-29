# Combo variant of 6e453dd6 (index 28)
# combo_spec: D2/1+D25/0
# Name: 射线反射+右侧障碍物

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace:
#    Original 6e453dd6 slides each black sprite flush to the separator and then
#    paints a straight horizontal red strip on every trigger row. In this combo,
#    the left-side trigger logic stays the same, but the right side is no longer
#    empty: each emitting row faces two pre-placed 2-pixel reflectors. The ray
#    first moves right, sees an obstacle directly ahead, turns up/down one cell
#    before that obstacle, then later sees a second obstacle ahead and turns
#    right again. So one original straight strip becomes an orthogonal polyline.
# b) Complexity calibration:
#    The reflector vocabulary is intentionally tiny: only 2-pixel dominoes are
#    used, and there are only two local cases the solver must decode:
#    - vertical domino ahead of a horizontal ray => bend vertically
#    - horizontal domino ahead of a vertical ray => bend horizontally
#    This is visibly different from the base task, but still repetitive enough
#    for ARC examples to support human induction.
# c) Hidden assumptions reviewed:
#    - The old background detector assumed the outer border stayed background.
#      That breaks once right-side obstacles may touch border rows, so bg now
#      prefers the border majority but falls back to the global non-separator mode.
#    - The old ray step asserted the whole right half was blank before painting.
#      That is false here; only background cells are paintable and obstacles stay.
#    - Reflection direction must be readable from shape, not color. Therefore
#      each obstacle cell that a ray can hit is guaranteed by the generator to
#      have exactly one perpendicular obstacle neighbor, which encodes the bend.
# d) Placement legality:
#    The generator keeps sprites in disjoint row bands on the left, and keeps
#    each ray inside the emitter's own vertical corridor on the right. Reflector
#    cells never overlap sprites, other reflectors, or planned red cells.
# e) Visual observability / cleanliness:
#    Each emitter contributes one trigger row and one three-segment polyline,
#    so samples show repeated "straight -> bend -> straight" behavior. Obstacles
#    use a small non-red palette and are preserved in both input and output,
#    making cause and effect easy to inspect by eye.
# f) Non-degeneracy target:
#    Every generated emitter gets at least two reflections, so generated samples
#    cannot collapse back to the original "paint a full row to the right edge"
#    behavior on the same input.


def _components4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected one full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(1, h - 1):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))

    if border:
        vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
        winners = vals[counts == counts.max()]
        if len(winners) == 1:
            return int(winners[0])

    vals, counts = np.unique(g[g != sep_color], return_counts=True)
    assert len(vals) > 0, "expected non-separator cells"
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must have a unique mode, got {winners.tolist()}"
    return int(winners[0])


def _slide_left_sprites(g: np.ndarray, sep: int, bg: int, black: int = 0) -> np.ndarray:
    left = g[:, :sep].copy()
    comps = _components4(left == black)
    left[left == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left components must start left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep
            left[r, nc] = black
    return left


def _trigger_rows(left: np.ndarray, sep: int, bg: int) -> list[int]:
    h, _ = left.shape
    rows = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if not any(c == sep - 2 for _, c in pts):
            continue
        rows.extend(sorted({int(r) for r, _ in pts}))
    assert len(rows) == len(set(rows)), f"trigger rows must be unique, got {rows}"
    return rows


def _obstacle_mask(g: np.ndarray, bg: int, sep_color: int = 5, red: int = 2) -> np.ndarray:
    return (g != bg) & (g != sep_color) & (g != red)


def _turn_from_reflector(obstacles: np.ndarray, nr: int, nc: int, dr: int, dc: int) -> tuple[int, int]:
    h, w = obstacles.shape
    if dc != 0:
        extensions = []
        for rr in (nr - 1, nr + 1):
            if 0 <= rr < h and obstacles[rr, nc]:
                extensions.append(rr - nr)
        assert len(extensions) == 1, f"horizontal ray needs one vertical reflector neighbor at {(nr, nc)}"
        return int(-extensions[0]), 0

    extensions = []
    for cc in (nc - 1, nc + 1):
        if 0 <= cc < w and obstacles[nr, cc]:
            extensions.append(cc - nc)
    assert len(extensions) == 1, f"vertical ray needs one horizontal reflector neighbor at {(nr, nc)}"
    return 0, int(-extensions[0])


def _paint_reflecting_ray(
    out: np.ndarray,
    obstacles: np.ndarray,
    start_row: int,
    sep: int,
    bg: int,
    red: int = 2,
) -> None:
    h, w = out.shape
    cur_r, cur_c = int(start_row), int(sep)
    dr, dc = 0, 1
    seen = set()

    while True:
        state = (cur_r, cur_c, dr, dc)
        if state in seen:
            raise AssertionError(f"ray loop detected from row {start_row}")
        seen.add(state)

        nr, nc = cur_r + dr, cur_c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if nc <= sep:
            break

        if obstacles[nr, nc]:
            dr, dc = _turn_from_reflector(obstacles, nr, nc, dr, dc)
            continue

        cur_r, cur_c = nr, nc
        if out[cur_r, cur_c] == bg:
            out[cur_r, cur_c] = red
        else:
            assert out[cur_r, cur_c] == red, f"ray entered unexpected colored cell {(cur_r, cur_c)}"


def _straight_baseline_output(inp: np.ndarray) -> np.ndarray:
    sep = _find_separator_col(inp, sep_color=SEP_COLOR)
    left = _slide_left_sprites(inp, sep=sep, bg=BG, black=BLACK)
    out = inp.copy()
    out[:, :sep] = left
    for r in _trigger_rows(left, sep=sep, bg=BG):
        out[r, sep + 1 :] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the grey divider, then route red rays through right-side reflectors.

    Input structure:
    - A single full-height grey column splits the grid into a left sprite pane and a right obstacle pane.
    - Black shapes live on the left over one background color; the right pane already contains colored obstacle pixels.

    Rule:
    - Move every black connected component rightward until its right edge becomes flush with the separator.
    - Find enclosed background holes formed after the slide; any hole row touching the inner face near the separator emits a red ray.
    - On the right, a ray advances through background cells, but if the next cell is an obstacle it reflects and continues; obstacle pixels themselves stay unchanged.

    Key insight: The emitting rows are still determined entirely by the slid left-side holes, but the final red geometry is controlled by local obstacle shape on the right, so the output combines hole detection with path tracing.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_color=sep_color)

    out = g.copy()
    left = _slide_left_sprites(g, sep=sep, bg=bg, black=black)
    out[:, :sep] = left

    rows = _trigger_rows(left, sep=sep, bg=bg)
    obstacles = _obstacle_mask(g, bg=bg, sep_color=sep_color, red=red)
    obstacles[:, : sep + 1] = False

    for r in rows:
        _paint_reflecting_ray(out, obstacles, start_row=r, sep=sep, bg=bg, red=red)

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5
OBSTACLE_COLORS = [1, 3, 4, 7, 8, 9]

H_MIN, H_MAX = 18, 25
W_MIN, W_MAX = 13, 18
SEP_MIN, SEP_MAX = 6, 10
RIGHT_W_MIN, RIGHT_W_MAX = 6, 8

N_SPRITES_MIN, N_SPRITES_MAX = 2, 4
EMITTER_H_MIN, EMITTER_H_MAX = 5, 7
DISTRACTOR_H_MIN, DISTRACTOR_H_MAX = 2, 5
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

EXTRA_EMITTERS_PROB = 0.35
FLUSH_PROB = 0.3
MAX_RETRIES = 400


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n: int) -> int:
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters: int, n_total: int) -> int:
    return n_emitters * EMITTER_H_MIN + (n_total - n_emitters) * DISTRACTOR_H_MIN + (n_total + 1) * MIN_BAND_GAP


def _choose_grid_dims() -> tuple[int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    return h, sep + 1 + right_w, sep


def _choose_emitter_flags(h: int) -> list[bool]:
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(_min_emitters_for(n), n) <= h]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = n
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _sprite_bands(h: int, flags: list[bool]) -> list[tuple[int, int]] | None:
    min_heights = [EMITTER_H_MIN if is_emitter else DISTRACTOR_H_MIN for is_emitter in flags]
    max_heights = [EMITTER_H_MAX if is_emitter else DISTRACTOR_H_MAX for is_emitter in flags]
    base_gaps = [MIN_BAND_GAP] * (len(flags) + 1)
    remaining = h - sum(base_gaps) - sum(min_heights)
    if remaining < 0:
        return None

    heights = min_heights[:]
    extra_caps = [mx - mn for mx, mn in zip(max_heights, min_heights)]
    for _ in range(random.randint(0, min(sum(extra_caps), remaining))):
        choices = [i for i, cap in enumerate(extra_caps) if cap > 0]
        if not choices:
            break
        i = random.choice(choices)
        heights[i] += 1
        extra_caps[i] -= 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), len(base_gaps)))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _emitter_patch(sh: int, max_sw: int) -> tuple[np.ndarray, list[int]]:
    hole_w = 1 if max_sw < 5 else random.choice([1, 2])
    sw = random.randint(hole_w + 3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    hole_row = sh // 2
    top = hole_row - 1
    bottom = hole_row + 1
    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    patch[top, left_wall : right_wall + 1] = BLACK
    patch[bottom, left_wall : right_wall + 1] = BLACK
    patch[top : bottom + 1, left_wall] = BLACK
    patch[top : bottom + 1, right_wall] = BLACK

    for r in range(0, top):
        patch[r, right_wall] = BLACK
        if random.random() < 0.35:
            patch[r, max(left_wall, right_wall - 1) : right_wall + 1] = BLACK
    for r in range(bottom + 1, sh):
        patch[r, right_wall] = BLACK
        if random.random() < 0.35:
            patch[r, max(left_wall, right_wall - 1) : right_wall + 1] = BLACK

    return patch, [hole_row]


def _distractor_patch(sh: int, max_sw: int) -> np.ndarray:
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), BLACK, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, -1] = BLACK
    elif kind == "c_right":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, 0] = BLACK
    else:
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, :] = BLACK
        else:
            patch[-1, :] = BLACK
    return patch


def _render_output_left(
    h: int,
    w: int,
    sep: int,
    bands: list[tuple[int, int]],
    flags: list[bool],
) -> tuple[np.ndarray, list[dict]]:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        if is_emitter:
            patch, local_rows = _emitter_patch(sh, max_sw)
        else:
            patch = _distractor_patch(sh, max_sw)
            local_rows = []
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append(
            {
                "patch": patch,
                "r0": r0,
                "c0": c0,
                "is_emitter": is_emitter,
                "trigger_rows": [r0 + rr for rr in local_rows],
                "band": (r0, sh),
            }
        )
    return out, sprites


def _ray_cells(start_r: int, bend_c: int, end_r: int, sep: int, w: int) -> list[tuple[int, int]]:
    cells = [(start_r, c) for c in range(sep + 1, bend_c + 1)]
    step = 1 if end_r > start_r else -1
    cells.extend((r, bend_c) for r in range(start_r + step, end_r + step, step))
    cells.extend((end_r, c) for c in range(bend_c + 1, w))
    return cells


def _build_right_layout(h: int, w: int, sep: int, emitters: list[dict]) -> tuple[np.ndarray, set[tuple[int, int]], int]:
    right = np.full((h, w), BG, dtype=int)
    right[:, sep] = SEP_COLOR

    occupied_obstacles: set[tuple[int, int]] = set()
    occupied_rays: set[tuple[int, int]] = set()
    total_reflections = 0
    palette = random.sample(OBSTACLE_COLORS, k=random.randint(2, min(3, len(OBSTACLE_COLORS))))

    for info in emitters:
        start_r = info["trigger_rows"][0]
        band_r0, band_h = info["band"]
        up_avail = start_r - band_r0
        down_avail = band_r0 + band_h - 1 - start_r
        directions = []
        if up_avail >= 2:
            directions.append(-1)
        if down_avail >= 2:
            directions.append(1)
        assert directions, f"emitter row {start_r} lacks vertical reflector room"

        planned = None
        for _ in range(120):
            vdir = random.choice(directions)
            avail = up_avail if vdir == -1 else down_avail
            vlen = random.randint(1, min(2, avail - 1))
            bend_c = random.randint(sep + 2, w - 2)

            anchor1 = (start_r, bend_c + 1)
            ext1 = (start_r - vdir, bend_c + 1)
            end_r = start_r + vdir * vlen
            anchor2 = (end_r + vdir, bend_c)
            ext2 = (anchor2[0], bend_c - 1)

            obstacles = {anchor1, ext1, anchor2, ext2}
            ray = set(_ray_cells(start_r, bend_c, end_r, sep, w))

            if any(not (0 <= r < h and sep < c < w) for r, c in obstacles | ray):
                continue
            if obstacles & ray:
                continue
            if obstacles & occupied_obstacles:
                continue
            if obstacles & occupied_rays:
                continue
            if ray & occupied_obstacles:
                continue
            if ray & occupied_rays:
                continue

            planned = (obstacles, ray)
            break

        assert planned is not None, f"failed to place reflector path for emitter row {start_r}"
        obstacles, ray = planned
        color1 = random.choice(palette)
        color2 = random.choice(palette)
        for cell, color in ((anchor1, color1), (ext1, color1), (anchor2, color2), (ext2, color2)):
            right[cell] = color
        occupied_obstacles |= obstacles
        occupied_rays |= ray
        total_reflections += 2

    return right, occupied_rays, total_reflections


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[dict], right: np.ndarray) -> np.ndarray | None:
    inp = right.copy()

    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1
    for i, info in enumerate(sprites):
        patch = info["patch"]
        r0 = info["r0"]
        c0 = info["c0"]
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out_left, sprites = _render_output_left(h, w, sep, bands, flags)
    emitters = [info for info in sprites if info["is_emitter"]]
    if not emitters:
        return None

    right, ray_cells, reflection_count = _build_right_layout(h, w, sep, emitters)
    out = out_left.copy()
    out[:, sep + 1 :] = right[:, sep + 1 :]
    for r, c in ray_cells:
        out[r, c] = RED

    inp = _unslide_sprites(h, w, sep, sprites, right)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    meta = {
        "emitters": len(emitters),
        "reflections": reflection_count,
        "ray_cells": len(ray_cells),
        "straight_diff": int(np.count_nonzero(out != _straight_baseline_output(inp))),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid combo input for 6e453dd6")

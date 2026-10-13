from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, paste


WINDOW_NAMES = ("tl", "tr", "bl", "br")
OTHER_WINDOWS = ("tr", "bl", "br")
SPRITE_BG = -1
SEPARATOR_THICKNESS = 1


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, int]] = []
    for sep_r in range(1, h - 1):
        if sep_r * 2 + 1 != h:
            continue
        row = grid[sep_r, :]
        if not np.all(row == row[0]):
            continue
        color = int(row[0])
        for sep_c in range(1, w - 1):
            if sep_c * 2 + 1 != w:
                continue
            if np.all(grid[:, sep_c] == color):
                candidates.append((sep_r, sep_c, color))
    assert len(candidates) == 1, "expected a unique solid separator cross"
    return candidates[0]


def _split_quadrants(grid: np.ndarray, sep_r: int, sep_c: int) -> dict[str, np.ndarray]:
    return {
        "tl": grid[:sep_r, :sep_c].copy(),
        "tr": grid[:sep_r, sep_c + 1 :].copy(),
        "bl": grid[sep_r + 1 :, :sep_c].copy(),
        "br": grid[sep_r + 1 :, sep_c + 1 :].copy(),
    }


def _join_quadrants(quadrants: dict[str, np.ndarray], separator_color: int) -> np.ndarray:
    qh, qw = quadrants["tl"].shape
    out = np.full((2 * qh + 1, 2 * qw + 1), int(separator_color), dtype=int)
    out[:qh, :qw] = quadrants["tl"]
    out[:qh, qw + 1 :] = quadrants["tr"]
    out[qh + 1 :, :qw] = quadrants["bl"]
    out[qh + 1 :, qw + 1 :] = quadrants["br"]
    return out


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[slice, slice, slice, slice] | None:
    h, w = map(int, shape)
    a_r0 = max(0, dr)
    a_r1 = min(h, h + dr)
    a_c0 = max(0, dc)
    a_c1 = min(w, w + dc)
    if a_r0 >= a_r1 or a_c0 >= a_c1:
        return None
    b_r0 = a_r0 - dr
    b_r1 = a_r1 - dr
    b_c0 = a_c0 - dc
    b_c1 = a_c1 - dc
    return slice(a_r0, a_r1), slice(a_c0, a_c1), slice(b_r0, b_r1), slice(b_c0, b_c1)


def _exact_alignment_candidates(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_overlap: int,
) -> list[tuple[int, int, int]]:
    h, w = map(int, a.shape)
    candidates: list[tuple[int, int, int]] = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            slices = _overlap_slices(a.shape, dr, dc)
            if slices is None:
                continue
            a_rs, a_cs, b_rs, b_cs = slices
            sub_a = a[a_rs, a_cs]
            sub_b = b[b_rs, b_cs]
            overlap = int(sub_a.size)
            if overlap < min_overlap:
                continue
            if np.array_equal(sub_a, sub_b):
                candidates.append((overlap, dr, dc))
    return candidates


def _best_exact_offset(a: np.ndarray, b: np.ndarray, *, min_overlap: int) -> tuple[int, int]:
    candidates = _exact_alignment_candidates(a, b, min_overlap=min_overlap)
    assert candidates, "expected at least one exact texture alignment"
    best_overlap = max(overlap for overlap, _dr, _dc in candidates)
    best = [(dr, dc) for overlap, dr, dc in candidates if overlap == best_overlap]
    assert len(best) == 1, "expected a unique exact texture alignment"
    return best[0]


def _build_master_map(windows: dict[str, np.ndarray], offsets: dict[str, tuple[int, int]]) -> dict[tuple[int, int], int]:
    master: dict[tuple[int, int], int] = {}
    for name, patch in windows.items():
        off_r, off_c = offsets[name]
        for r in range(patch.shape[0]):
            for c in range(patch.shape[1]):
                pos = (off_r + r, off_c + c)
                value = int(patch[r, c])
                if pos in master:
                    assert master[pos] == value, "texture windows must agree where they overlap"
                else:
                    master[pos] = value
    return master


def _best_noisy_offset(
    patch: np.ndarray,
    master: dict[tuple[int, int], int],
    *,
    min_coverage: int,
) -> tuple[int, int]:
    h, w = map(int, patch.shape)
    candidates: list[tuple[tuple[int, int, int], int, int]] = []
    for off_r in range(-h + 1, h):
        for off_c in range(-w + 1, w):
            matches = 0
            mismatches = 0
            for r in range(h):
                for c in range(w):
                    key = (off_r + r, off_c + c)
                    if key not in master:
                        continue
                    if int(patch[r, c]) == master[key]:
                        matches += 1
                    else:
                        mismatches += 1
            coverage = matches + mismatches
            if coverage < min_coverage:
                continue
            score = (matches - mismatches, matches, coverage)
            candidates.append((score, off_r, off_c))
    assert candidates, "expected at least one left-window alignment candidate"
    best_score = max(score for score, _off_r, _off_c in candidates)
    best = [(off_r, off_c) for score, off_r, off_c in candidates if score == best_score]
    assert len(best) == 1, "expected a unique left-window alignment"
    return best[0]


def _recover_relative_offsets(quadrants: dict[str, np.ndarray]) -> dict[str, tuple[int, int]]:
    qh, qw = quadrants["tl"].shape
    min_overlap = max(6, (qh - 2) * (qw - 2) // 4)
    non_tl_offsets = {
        "tr": (0, 0),
        "bl": _best_exact_offset(quadrants["tr"], quadrants["bl"], min_overlap=min_overlap),
        "br": _best_exact_offset(quadrants["tr"], quadrants["br"], min_overlap=min_overlap),
    }
    expected_bl_to_br = (
        non_tl_offsets["br"][0] - non_tl_offsets["bl"][0],
        non_tl_offsets["br"][1] - non_tl_offsets["bl"][1],
    )
    assert _best_exact_offset(quadrants["bl"], quadrants["br"], min_overlap=min_overlap) == expected_bl_to_br

    master = _build_master_map(
        {name: quadrants[name] for name in OTHER_WINDOWS},
        non_tl_offsets,
    )
    tl_offset = _best_noisy_offset(
        quadrants["tl"],
        master,
        min_coverage=max(6, qh * qw // 3),
    )

    offsets = {"tl": (0, 0)}
    for name, offset in non_tl_offsets.items():
        offsets[name] = (offset[0] - tl_offset[0], offset[1] - tl_offset[1])
    return offsets


def _infer_sprite_mask(
    tl: np.ndarray,
    relative_offsets: dict[str, tuple[int, int]],
    other_quadrants: dict[str, np.ndarray],
) -> np.ndarray:
    h, w = map(int, tl.shape)
    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            expected: list[int] = []
            for name in OTHER_WINDOWS:
                dr, dc = relative_offsets[name]
                rr = r - dr
                cc = c - dc
                if 0 <= rr < h and 0 <= cc < w:
                    expected.append(int(other_quadrants[name][rr, cc]))
            if not expected:
                continue
            assert len(set(expected)) == 1, "texture consensus should be unique"
            if int(tl[r, c]) != expected[0]:
                mask[r, c] = True
    assert mask.any(), "expected a non-empty top-left decal"
    assert len(components_mask(mask, connectivity=4)) == 1, "decal should be one connected sprite"
    return mask


def _stamp_sprite_into_window(dst: np.ndarray, sprite_patch: np.ndarray, top: int, left: int) -> None:
    paste(dst, sprite_patch, top, left, transparent=SPRITE_BG, clip=True)


def main(grid: np.ndarray) -> np.ndarray:
    """Treat the four cross-separated quadrants as offset windows onto one texture, then project the top-left decal into the other windows.

    Input structure:
    - One solid row and one solid column of the same color form a separator cross and split the grid into four equal-sized quadrants.
    - Each quadrant is a viewport onto the same multicolor, non-periodic texture, but each viewport samples it at a different translation offset.
    - The top-left viewport alone also contains a small connected decal sprite in colors that are not part of the shared texture.

    Rule:
    - Align the three decal-free viewports exactly by their texture to recover a shared texture coordinate system.
    - Align the top-left viewport against that texture while tolerating the few decal pixels, then identify which top-left pixels disagree with the texture consensus.
    - Keep the separator cross, texture, and original top-left decal unchanged; for each other viewport, translate that same decal by the recovered viewport offset and draw only the portion that falls inside the window.

    Key insight:
    - The four quadrants are not separate patterns: they are four shifted views of one hidden base image, so the decal behaves like a fixed texture-space sticker whose position in every window is determined by the phase offset between viewports.
    """
    g = np.asarray(grid, dtype=int)
    sep_r, sep_c, separator_color = _find_separator_cross(g)
    quadrants = _split_quadrants(g, sep_r, sep_c)
    relative_offsets = _recover_relative_offsets(quadrants)
    sprite_mask = _infer_sprite_mask(
        quadrants["tl"],
        relative_offsets,
        {name: quadrants[name] for name in OTHER_WINDOWS},
    )

    sprite_patch = np.full_like(quadrants["tl"], SPRITE_BG)
    sprite_patch[sprite_mask] = quadrants["tl"][sprite_mask]
    sprite_coords = np.argwhere(sprite_mask)
    top_left_r = int(sprite_coords[:, 0].min())
    top_left_c = int(sprite_coords[:, 1].min())
    bottom_right_r = int(sprite_coords[:, 0].max())
    bottom_right_c = int(sprite_coords[:, 1].max())
    sprite_patch = sprite_patch[top_left_r : bottom_right_r + 1, top_left_c : bottom_right_c + 1]

    out_quadrants = {name: patch.copy() for name, patch in quadrants.items()}
    for name in OTHER_WINDOWS:
        dr, dc = relative_offsets[name]
        target_r = top_left_r - dr
        target_c = top_left_c - dc
        _stamp_sprite_into_window(out_quadrants[name], sprite_patch, target_r, target_c)

    return _join_quadrants(out_quadrants, separator_color)


# --------------- generator (o5x) ---------------

QUADRANT_HEIGHT_RANGE = (8, 11)
QUADRANT_WIDTH_RANGE = (8, 11)
TEXTURE_COLOR_COUNT = (3, 4)
SPRITE_COLOR_COUNT = (1, 2)
TEXTURE_RECTANGLES = (4, 8)
MAX_REL_SHIFT = 2
GEN_ATTEMPTS = 400
BRANCH_PATTERNS = [
    {"full": 1, "partial": 1, "none": 1},
    {"full": 2, "partial": 1, "none": 0},
    {"full": 1, "partial": 0, "none": 2},
    {"full": 0, "partial": 2, "none": 1},
    {"full": 3, "partial": 0, "none": 0},
]
BRANCH_WEIGHTS = [4, 4, 3, 4, 1]


def _sprite_shape_catalog() -> list[np.ndarray]:
    catalog: list[np.ndarray] = []
    dims = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for h, w in dims:
        cells = h * w
        for bits in range(1, 1 << cells):
            area = bits.bit_count()
            if not 2 <= area <= 5:
                continue
            mask = np.array([(bits >> i) & 1 for i in range(cells)], dtype=bool).reshape(h, w)
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if len(components_mask(mask, connectivity=4)) != 1:
                continue
            catalog.append(mask)
    unique: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for mask in catalog:
        key = tuple(tuple(int(v) for v in row) for row in mask)
        unique.setdefault(key, mask)
    return list(unique.values())


SPRITE_SHAPES = _sprite_shape_catalog()


def _choose_palette() -> tuple[list[int], int, list[int]]:
    texture_count = random.randint(*TEXTURE_COLOR_COUNT)
    sprite_count = random.randint(*SPRITE_COLOR_COUNT)
    colors = random.sample(range(10), texture_count + sprite_count + 1)
    texture_colors = colors[:texture_count]
    separator_color = colors[texture_count]
    sprite_colors = colors[texture_count + 1 :]
    return texture_colors, separator_color, sprite_colors


def _make_texture(shape: tuple[int, int], texture_colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    texture = np.array(
        [[random.choice(texture_colors) for _ in range(w)] for _ in range(h)],
        dtype=int,
    )
    for _ in range(random.randint(*TEXTURE_RECTANGLES)):
        color = random.choice(texture_colors)
        rect_h = random.randint(2, min(4, h))
        rect_w = random.randint(2, min(4, w))
        r0 = random.randint(0, h - rect_h)
        c0 = random.randint(0, w - rect_w)
        texture[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
    seen = set(map(int, texture.ravel()))
    for color in texture_colors:
        if color not in seen:
            rr = random.randint(0, h - 1)
            cc = random.randint(0, w - 1)
            texture[rr, cc] = color
    return texture


def _choose_relative_offsets() -> dict[str, tuple[int, int]] | None:
    points = [
        (dr, dc)
        for dr in range(-MAX_REL_SHIFT, MAX_REL_SHIFT + 1)
        for dc in range(-MAX_REL_SHIFT, MAX_REL_SHIFT + 1)
        if (dr, dc) != (0, 0)
    ]
    random.shuffle(points)
    for tr in points:
        for bl in points:
            if bl == tr:
                continue
            for br in points:
                if br in {tr, bl}:
                    continue
                all_pts = [(0, 0), tr, bl, br]
                rows = {pt[0] for pt in all_pts}
                cols = {pt[1] for pt in all_pts}
                if len(rows) == 1 or len(cols) == 1:
                    continue
                return {"tl": (0, 0), "tr": tr, "bl": bl, "br": br}
    return None


def _absolute_offsets(relative_offsets: dict[str, tuple[int, int]]) -> tuple[dict[str, tuple[int, int]], tuple[int, int]]:
    all_rows = [offset[0] for offset in relative_offsets.values()]
    all_cols = [offset[1] for offset in relative_offsets.values()]
    min_r = min(all_rows)
    min_c = min(all_cols)
    absolute = {
        name: (offset[0] - min_r, offset[1] - min_c)
        for name, offset in relative_offsets.items()
    }
    span = (max(all_rows) - min_r, max(all_cols) - min_c)
    return absolute, span


def _render_texture_windows(
    texture: np.ndarray,
    quadrant_shape: tuple[int, int],
    absolute_offsets: dict[str, tuple[int, int]],
) -> dict[str, np.ndarray]:
    qh, qw = map(int, quadrant_shape)
    windows = {}
    for name, (r0, c0) in absolute_offsets.items():
        windows[name] = texture[r0 : r0 + qh, c0 : c0 + qw].copy()
    return windows


def _make_sprite_patch(mask: np.ndarray, sprite_colors: list[int]) -> np.ndarray:
    patch = np.full(mask.shape, SPRITE_BG, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    colors = [random.choice(sprite_colors) for _ in coords]
    for (r, c), color in zip(coords, colors, strict=False):
        patch[r, c] = int(color)
    return patch


def _visibility_state(
    sprite_patch: np.ndarray,
    origin: tuple[int, int],
    quadrant_shape: tuple[int, int],
) -> str:
    qh, qw = map(int, quadrant_shape)
    coords = np.argwhere(sprite_patch != SPRITE_BG)
    visible = 0
    for r, c in coords:
        rr = int(origin[0] + r)
        cc = int(origin[1] + c)
        if 0 <= rr < qh and 0 <= cc < qw:
            visible += 1
    if visible == 0:
        return "none"
    if visible == len(coords):
        return "full"
    return "partial"


def _sprite_cells_seen_elsewhere(
    sprite_patch: np.ndarray,
    tl_origin: tuple[int, int],
    relative_offsets: dict[str, tuple[int, int]],
    quadrant_shape: tuple[int, int],
) -> bool:
    qh, qw = map(int, quadrant_shape)
    for r, c in np.argwhere(sprite_patch != SPRITE_BG):
        local_r = int(tl_origin[0] + r)
        local_c = int(tl_origin[1] + c)
        if not any(
            0 <= local_r - relative_offsets[name][0] < qh
            and 0 <= local_c - relative_offsets[name][1] < qw
            for name in OTHER_WINDOWS
        ):
            return False
    return True


def _find_sprite_origins(
    quadrant_shape: tuple[int, int],
    sprite_patch: np.ndarray,
    relative_offsets: dict[str, tuple[int, int]],
    wanted: Counter,
) -> list[tuple[int, int]]:
    qh, qw = map(int, quadrant_shape)
    sh, sw = map(int, sprite_patch.shape)
    origins: list[tuple[int, int]] = []
    for r0 in range(qh - sh + 1):
        for c0 in range(qw - sw + 1):
            tl_origin = (r0, c0)
            if not _sprite_cells_seen_elsewhere(sprite_patch, tl_origin, relative_offsets, quadrant_shape):
                continue
            states = []
            for name in OTHER_WINDOWS:
                dr, dc = relative_offsets[name]
                states.append(_visibility_state(sprite_patch, (r0 - dr, c0 - dc), quadrant_shape))
            if Counter(states) == wanted:
                origins.append(tl_origin)
    return origins


def _draw_case(
    base_windows: dict[str, np.ndarray],
    separator_color: int,
    sprite_patch: np.ndarray,
    tl_origin: tuple[int, int],
    relative_offsets: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    inp_quadrants = {name: patch.copy() for name, patch in base_windows.items()}
    out_quadrants = {name: patch.copy() for name, patch in base_windows.items()}

    _stamp_sprite_into_window(inp_quadrants["tl"], sprite_patch, tl_origin[0], tl_origin[1])
    _stamp_sprite_into_window(out_quadrants["tl"], sprite_patch, tl_origin[0], tl_origin[1])

    states: dict[str, str] = {}
    for name in OTHER_WINDOWS:
        dr, dc = relative_offsets[name]
        target = (tl_origin[0] - dr, tl_origin[1] - dc)
        states[name] = _visibility_state(sprite_patch, target, inp_quadrants[name].shape)
        _stamp_sprite_into_window(out_quadrants[name], sprite_patch, target[0], target[1])

    return _join_quadrants(inp_quadrants, separator_color), _join_quadrants(out_quadrants, separator_color), states


def _validate_truth(
    inp: np.ndarray,
    expected: np.ndarray,
    true_offsets: dict[str, tuple[int, int]],
    branch_counter: Counter,
) -> bool:
    try:
        sep_r, sep_c, _separator_color = _find_separator_cross(inp)
        quadrants = _split_quadrants(inp, sep_r, sep_c)
        recovered = _recover_relative_offsets(quadrants)
        if recovered != true_offsets:
            return False
        sprite_mask = _infer_sprite_mask(quadrants["tl"], recovered, {name: quadrants[name] for name in OTHER_WINDOWS})
        if not sprite_mask.any():
            return False
        if sum(branch_counter.values()) != len(OTHER_WINDOWS):
            return False
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(*QUADRANT_HEIGHT_RANGE)
    qw = random.randint(*QUADRANT_WIDTH_RANGE)
    quadrant_shape = (qh, qw)

    texture_colors, separator_color, sprite_colors = _choose_palette()
    relative_offsets = _choose_relative_offsets()
    if relative_offsets is None:
        return None
    absolute_offsets, span = _absolute_offsets(relative_offsets)
    texture_shape = (qh + span[0], qw + span[1])
    texture = _make_texture(texture_shape, texture_colors)
    base_windows = _render_texture_windows(texture, quadrant_shape, absolute_offsets)

    sprite_mask = random.choice(SPRITE_SHAPES)
    sprite_patch = _make_sprite_patch(sprite_mask, sprite_colors)
    pattern = Counter(random.choices(BRANCH_PATTERNS, weights=BRANCH_WEIGHTS, k=1)[0])
    origins = _find_sprite_origins(quadrant_shape, sprite_patch, relative_offsets, pattern)
    if not origins:
        return None
    tl_origin = random.choice(origins)

    inp, expected, states = _draw_case(base_windows, separator_color, sprite_patch, tl_origin, relative_offsets)
    branch_counter = Counter(states.values())
    if branch_counter != pattern:
        return None
    if not _validate_truth(inp, expected, relative_offsets, branch_counter):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid viewport-decal input")

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste
import random


CORNER_SIGN = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}


def _three_corner_squares(g: np.ndarray, marker_color: int, side: int | None = None) -> list[dict]:
    coords = [tuple(map(int, p)) for p in np.argwhere(g == marker_color)]
    coord_set = set(coords)
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    controls = []

    for r0 in rows:
        for r1 in rows:
            if r1 <= r0:
                continue
            square_side = r1 - r0
            if side is not None and square_side != side:
                continue
            for c0 in cols:
                c1 = c0 + square_side
                if c1 not in cols:
                    continue
                corners = {
                    "tl": (r0, c0),
                    "tr": (r0, c1),
                    "bl": (r1, c0),
                    "br": (r1, c1),
                }
                present = [name for name, pos in corners.items() if pos in coord_set]
                if len(present) != 3:
                    continue
                missing = next(name for name in corners if name not in present)
                controls.append(
                    {
                        "side": square_side,
                        "missing_kind": missing,
                        "missing": corners[missing],
                        "markers": tuple(corners[name] for name in present),
                    }
                )
    return controls


def _corner_matches_bbox(corner: str, point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if corner == "tl":
        return (r1, c1) == point
    if corner == "tr":
        return (r1, c0) == point
    if corner == "bl":
        return (r0, c1) == point
    if corner == "br":
        return (r0, c0) == point
    raise AssertionError(f"unknown corner: {corner}")


def _corner_border_motif(sprite: np.ndarray, sprite_color: int, bg: int, sign: tuple[int, int]) -> np.ndarray:
    mask = sprite == sprite_color
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return sprite

    anchor_r = h - 1 if sign[0] < 0 else 0
    anchor_c = w - 1 if sign[1] < 0 else 0
    opposite_r = 0 if anchor_r == h - 1 else h - 1
    opposite_c = 0 if anchor_c == w - 1 else w - 1

    motif = np.zeros_like(mask, dtype=bool)
    motif[anchor_r, :] = True
    motif[:, anchor_c] = True
    motif[opposite_r, opposite_c] = True
    return np.where(motif, sprite_color, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the existing foreground sprite into every three-marker corner frame, reflecting it to the frame's missing corner.

    Input structure
    - The background is the dominant color.
    - One non-background color forms singleton marker pixels; every three markers are three corners of a same-size square.
    - The other non-background color forms one sprite mask already placed at the missing corner of one marker square.

    Rule
    - Find the source marker square whose missing corner touches the corresponding corner of the sprite bbox.
    - For every marker square with the same side length, place a copy of that sprite outside the square at its missing corner.
    - Reflect the sprite vertically and/or horizontally when the target missing corner lies in a different quadrant from the source missing corner.
    - If the source bbox is a complete hollow square border, keep the corner-bracket motif selected by the source anchor.

    Key insight
    - The yellow three-corner squares are control frames: their absent fourth corner gives both the anchor point and the quadrant for the red sprite copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2

    singleton_colors = [
        color
        for color in colors
        if all(len(component) == 1 for component in components_mask(g == color, connectivity=8))
    ]
    assert len(singleton_colors) == 1
    marker_color = singleton_colors[0]
    sprite_colors = [color for color in colors if color != marker_color]
    assert len(sprite_colors) == 1
    sprite_color = sprite_colors[0]

    sprite_coords = np.argwhere(g == sprite_color)
    assert len(sprite_coords) > 0
    sprite_box = bbox_coords(sprite_coords)
    r0, c0, r1, c1 = sprite_box
    sprite = np.where(g[r0 : r1 + 1, c0 : c1 + 1] == sprite_color, sprite_color, bg)
    sprite_h, sprite_w = sprite.shape
    assert sprite_h == sprite_w

    source_controls = [
        control
        for control in _three_corner_squares(g, marker_color)
        if _corner_matches_bbox(control["missing_kind"], control["missing"], sprite_box)
    ]
    assert len(source_controls) == 1
    source = source_controls[0]

    controls = _three_corner_squares(g, marker_color, side=source["side"])
    marker_coords = [tuple(map(int, p)) for p in np.argwhere(g == marker_color)]
    used_markers = [marker for control in controls for marker in control["markers"]]
    assert sorted(used_markers) == sorted(marker_coords)

    out = g.copy()
    H, W = g.shape
    source_sign = CORNER_SIGN[source["missing_kind"]]
    sprite = _corner_border_motif(sprite, sprite_color, bg, source_sign)
    out[g == sprite_color] = bg

    for control in controls:
        target_sign = CORNER_SIGN[control["missing_kind"]]
        oriented = sprite.copy()
        if source_sign[0] != target_sign[0]:
            oriented = oriented[::-1, :]
        if source_sign[1] != target_sign[1]:
            oriented = oriented[:, ::-1]

        anchor_r, anchor_c = control["missing"]
        paste_r = anchor_r - sprite_h + 1 if target_sign[0] < 0 else anchor_r
        paste_c = anchor_c - sprite_w + 1 if target_sign[1] < 0 else anchor_c
        assert 0 <= paste_r and 0 <= paste_c
        assert paste_r + sprite_h <= H and paste_c + sprite_w <= W

        sprite_mask = oriented == sprite_color
        target = out[paste_r : paste_r + sprite_h, paste_c : paste_c + sprite_w]
        assert np.all((target[sprite_mask] == bg) | (target[sprite_mask] == sprite_color))
        target[sprite_mask] = sprite_color

    return out

# --------------- generator (o5x) ---------------

# Palette and grid geometry.
_GEN_BG, _GEN_MARKER_COLOR, _GEN_SPRITE_COLOR = 0, 4, 2
_GEN_GRID_SIDE_RANGE = (25, 30)
_GEN_FRAME_SIDE, _GEN_SPRITE_SIZES = 2, (2, 3)

# Structural counts and retry limits.
_GEN_FRAME_COUNT_RANGE, _GEN_MAX_ATTEMPTS = (4, 8), 300

# Sprite shape sampling.
_GEN_HOLLOW_ALWAYS_SIZE, _GEN_HOLLOW_BORDER_PROB = 2, 0.25
_GEN_IRREGULAR_MASK_SIZE = 3
_GEN_IRREGULAR_BASE_CELLS = ((2, 2), (1, 1))
_GEN_IRREGULAR_TOP_CHOICES = ((0, 0), (0, 1), (0, 2))
_GEN_IRREGULAR_LEFT_CHOICES = ((0, 0), (1, 0))
_GEN_IRREGULAR_EXCLUDED_CELL = (2, 0)
_GEN_IRREGULAR_FILL_RANGE = (4, 8)

# Placement separation around singleton marker pixels.
_GEN_MARKER_CLEARANCE = 1

_GEN_CORNER_SIGN = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}
_GEN_KINDS = ("tl", "tr", "bl", "br")


def _gen_square_corners(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + side),
        "bl": (r0 + side, c0),
        "br": (r0 + side, c0 + side),
    }


def _gen_orient_from_tl_anchor(mask: np.ndarray, sign: tuple[int, int]) -> np.ndarray:
    out = mask.copy()
    if sign[0] > 0:
        out = out[::-1, :]
    if sign[1] > 0:
        out = out[:, ::-1]
    return out


def _gen_corner_border_motif(sprite: np.ndarray, sprite_color: int, bg: int, sign: tuple[int, int]) -> np.ndarray:
    mask = sprite == sprite_color
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return sprite

    anchor_r = h - 1 if sign[0] < 0 else 0
    anchor_c = w - 1 if sign[1] < 0 else 0
    opposite_r = 0 if anchor_r == h - 1 else h - 1
    opposite_c = 0 if anchor_c == w - 1 else w - 1
    motif = np.zeros_like(mask, dtype=bool)
    motif[anchor_r, :] = True
    motif[:, anchor_c] = True
    motif[opposite_r, opposite_c] = True
    return np.where(motif, sprite_color, bg)


def _gen_hollow_source_mask(n: int) -> np.ndarray:
    mask = np.ones((n, n), dtype=bool)
    if n == _GEN_IRREGULAR_MASK_SIZE:
        center = n // 2
        mask[center, center] = False
    return mask


def _gen_irregular_source_mask() -> np.ndarray:
    cells = set(_GEN_IRREGULAR_BASE_CELLS)
    cells.add(random.choice(_GEN_IRREGULAR_TOP_CHOICES))
    cells.add(random.choice(_GEN_IRREGULAR_LEFT_CHOICES))
    allowed = [
        (r, c)
        for r in range(_GEN_IRREGULAR_MASK_SIZE)
        for c in range(_GEN_IRREGULAR_MASK_SIZE)
        if (r, c) != _GEN_IRREGULAR_EXCLUDED_CELL
    ]
    while len(cells) < _GEN_IRREGULAR_FILL_RANGE[0]:
        cells.add(random.choice(allowed))
    target = random.randint(*_GEN_IRREGULAR_FILL_RANGE)
    extras = [p for p in allowed if p not in cells]
    random.shuffle(extras)
    cells.update(extras[: target - len(cells)])

    mask = np.zeros((_GEN_IRREGULAR_MASK_SIZE, _GEN_IRREGULAR_MASK_SIZE), dtype=bool)
    rows, cols = zip(*cells)
    mask[rows, cols] = True
    assert len(components_mask(mask, connectivity=8)) == 1
    return mask


def _gen_source_sprite(n: int, source_sign: tuple[int, int], sprite_color: int, bg: int) -> np.ndarray:
    if n == _GEN_HOLLOW_ALWAYS_SIZE or random.random() < _GEN_HOLLOW_BORDER_PROB:
        mask = _gen_hollow_source_mask(n)
    else:
        mask = _gen_orient_from_tl_anchor(_gen_irregular_source_mask(), source_sign)
    return np.where(mask, sprite_color, bg)


def _gen_frame(kind: str, r0: int, c0: int, side: int, n: int) -> dict:
    corners = _gen_square_corners(r0, c0, side)
    ar, ac = corners[kind]
    sign = _GEN_CORNER_SIGN[kind]
    paste_r = ar - n + 1 if sign[0] < 0 else ar
    paste_c = ac - n + 1 if sign[1] < 0 else ac
    return {
        "kind": kind,
        "missing": (ar, ac),
        "markers": tuple(corners[k] for k in _GEN_KINDS if k != kind),
        "paste": (paste_r, paste_c),
    }


def _gen_candidate_frames(kind: str, h: int, side: int, n: int) -> list[dict]:
    frames = []
    for r0 in range(h - side):
        for c0 in range(h - side):
            frame = _gen_frame(kind, r0, c0, side, n)
            pr, pc = frame["paste"]
            if 0 <= pr and 0 <= pc and pr + n <= h and pc + n <= h:
                frames.append(frame)
    random.shuffle(frames)
    return frames


def _gen_marker_control_keys(marker_coords: set[tuple[int, int]]) -> set[tuple[int, str, tuple[int, int], tuple[tuple[int, int], ...]]]:
    rows = sorted({r for r, _c in marker_coords})
    cols = sorted({c for _r, c in marker_coords})
    keys = set()
    for r0 in rows:
        for r1 in rows:
            if r1 <= r0:
                continue
            side = r1 - r0
            for c0 in cols:
                c1 = c0 + side
                if c1 not in cols:
                    continue
                corners = _gen_square_corners(r0, c0, side)
                present = [k for k, p in corners.items() if p in marker_coords]
                if len(present) == 3:
                    missing = next(k for k in _GEN_KINDS if k not in present)
                    keys.add((side, missing, corners[missing], tuple(sorted(corners[k] for k in present))))
    return keys


def _gen_clean_controls(placed: list[dict], side: int) -> bool:
    markers = {p for frame in placed for p in frame["markers"]}
    actual = _gen_marker_control_keys(markers)
    intended = {
        (side, frame["kind"], frame["missing"], tuple(sorted(frame["markers"])))
        for frame in placed
    }
    return actual == intended


def _gen_can_place(
    frame: dict,
    placed: list[dict],
    paste_used: np.ndarray,
    marker_used: np.ndarray,
    marker_forbidden: np.ndarray,
    side: int,
    n: int,
) -> bool:
    pr, pc = frame["paste"]
    if paste_used[pr : pr + n, pc : pc + n].any():
        return False
    if marker_used[pr : pr + n, pc : pc + n].any():
        return False
    for r, c in frame["markers"]:
        if marker_forbidden[r, c] or paste_used[r, c]:
            return False
        if pr <= r < pr + n and pc <= c < pc + n:
            return False
    return _gen_clean_controls(placed + [frame], side)


def _gen_mark_frame(frame: dict, paste_used: np.ndarray, marker_used: np.ndarray, marker_forbidden: np.ndarray, n: int) -> None:
    h = paste_used.shape[0]
    pr, pc = frame["paste"]
    paste_used[pr : pr + n, pc : pc + n] = True
    pad = _GEN_MARKER_CLEARANCE
    for r, c in frame["markers"]:
        marker_used[r, c] = True
        marker_forbidden[max(0, r - pad) : min(h, r + pad + 1), max(0, c - pad) : min(h, c + pad + 1)] = True


def _gen_place_frames(h: int, kinds: list[str], side: int, n: int) -> list[dict] | None:
    paste_used = np.zeros((h, h), dtype=bool)
    marker_used = np.zeros((h, h), dtype=bool)
    marker_forbidden = np.zeros((h, h), dtype=bool)
    placed = []
    for kind in kinds:
        for frame in _gen_candidate_frames(kind, h, side, n):
            if _gen_can_place(frame, placed, paste_used, marker_used, marker_forbidden, side, n):
                placed.append(frame)
                _gen_mark_frame(frame, paste_used, marker_used, marker_forbidden, n)
                break
        else:
            return None
    return placed


def _gen_kind_sequence(k: int) -> list[str]:
    kinds = list(_GEN_KINDS) + [random.choice(_GEN_KINDS) for _ in range(k - len(_GEN_KINDS))]
    random.shuffle(kinds)
    return kinds


def _gen_sample_layout() -> tuple[int, int, list[str], int, tuple[int, int]]:
    h = random.randint(*_GEN_GRID_SIDE_RANGE)
    n = random.choice(_GEN_SPRITE_SIZES)
    k = random.randint(*_GEN_FRAME_COUNT_RANGE)
    kinds = _gen_kind_sequence(k)
    source_i = random.randrange(k)
    return h, n, kinds, source_i, _GEN_CORNER_SIGN[kinds[source_i]]


def _gen_build_output(h: int, frames: list[dict], sprite: np.ndarray, source_sign: tuple[int, int]) -> np.ndarray:
    out = np.zeros((h, h), dtype=int)
    for frame in frames:
        for r, c in frame["markers"]:
            out[r, c] = _GEN_MARKER_COLOR

    for frame in frames:
        oriented = sprite.copy()
        target_sign = _GEN_CORNER_SIGN[frame["kind"]]
        if source_sign[0] != target_sign[0]:
            oriented = oriented[::-1, :]
        if source_sign[1] != target_sign[1]:
            oriented = oriented[:, ::-1]
        pr, pc = frame["paste"]
        paste(out, oriented, pr, pc, transparent=_GEN_BG)
    return out


def _gen_build_input(out: np.ndarray, source_frame: dict, source_sprite: np.ndarray) -> np.ndarray:
    inp = out.copy()
    inp[inp == _GEN_SPRITE_COLOR] = _GEN_BG
    pr, pc = source_frame["paste"]
    paste(inp, source_sprite, pr, pc, transparent=_GEN_BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, n, kinds, source_i, source_sign = _gen_sample_layout()
    source_sprite = _gen_source_sprite(n, source_sign, _GEN_SPRITE_COLOR, _GEN_BG)
    output_sprite = _gen_corner_border_motif(source_sprite, _GEN_SPRITE_COLOR, _GEN_BG, source_sign)

    frames = _gen_place_frames(h, kinds, _GEN_FRAME_SIDE, n)
    if frames is None:
        return None

    out = _gen_build_output(h, frames, output_sprite, source_sign)
    inp = _gen_build_input(out, frames[source_i], source_sprite)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed for ac0c5833")

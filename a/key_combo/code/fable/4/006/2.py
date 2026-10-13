from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
YELLOW = 4
GREY = 5

KINDS = ("tl", "tr", "bl", "br")
CORNER_SIGN = {
    "tl": (-1, -1),
    "tr": (-1, 1),
    "bl": (1, -1),
    "br": (1, 1),
}


def _get_bg(g: np.ndarray) -> int:
    return int(Counter(map(int, g.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _three_corner_squares(g: np.ndarray, marker_color: int, side: int | None = None) -> list[dict]:
    coords = [tuple(map(int, p)) for p in np.argwhere(g == int(marker_color))]
    coord_set = set(coords)
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    controls: list[dict] = []

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
                missing = next(name for name in KINDS if name not in present)
                controls.append(
                    {
                        "side": int(square_side),
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


def _axis_info(axis_coords: np.ndarray) -> tuple[str, int]:
    pts = {tuple(map(int, p)) for p in axis_coords}
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]

    if len(set(rows)) == 1:
        r = rows[0]
        expected = {(r, c) for c in range(min(cols), max(cols) + 1)}
        assert pts == expected, "horizontal grey axis must be contiguous"
        return "h", int(r)

    if len(set(cols)) == 1:
        c = cols[0]
        expected = {(r, c) for r in range(min(rows), max(rows) + 1)}
        assert pts == expected, "vertical grey axis must be contiguous"
        return "v", int(c)

    diffs = {r - c for r, c in pts}
    if len(diffs) == 1:
        k = next(iter(diffs))
        expected = {(r, r - k) for r in range(min(rows), max(rows) + 1)}
        assert pts == expected, "down-right diagonal grey axis must be contiguous"
        return "diag_main", int(k)

    sums = {r + c for r, c in pts}
    if len(sums) == 1:
        s = next(iter(sums))
        expected = {(r, s - r) for r in range(min(rows), max(rows) + 1)}
        assert pts == expected, "up-right diagonal grey axis must be contiguous"
        return "diag_anti", int(s)

    raise AssertionError("grey axis must be one straight row, column, or 45-degree diagonal")


def _signed_distance(pos: tuple[int, int], axis: tuple[str, int]) -> int:
    r, c = pos
    kind, value = axis
    if kind == "h":
        return int(r - value)
    if kind == "v":
        return int(c - value)
    if kind == "diag_main":
        return int(r - c - value)
    if kind == "diag_anti":
        return int(r + c - value)
    raise AssertionError(f"unknown axis kind: {kind}")


def _reflect_point(pos: tuple[int, int], axis: tuple[str, int]) -> tuple[int, int]:
    r, c = pos
    kind, value = axis
    if kind == "h":
        return 2 * value - r, c
    if kind == "v":
        return r, 2 * value - c
    if kind == "diag_main":
        return c + value, r - value
    if kind == "diag_anti":
        return value - c, value - r
    raise AssertionError(f"unknown axis kind: {kind}")


def _reconstruct_sprite(g: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    axis_coords = np.argwhere(g == GREY)
    red_coords = np.argwhere(g == RED)
    assert axis_coords.size > 0, "expected a grey symmetry axis"
    assert red_coords.size > 0, "expected a red half-sprite"

    axis = _axis_info(axis_coords)
    distances = [_signed_distance(tuple(map(int, p)), axis) for p in red_coords]
    assert all(d != 0 for d in distances), "red half-sprite should not lie on the grey axis"
    assert len({1 if d > 0 else -1 for d in distances}) == 1, "all red cells should be on one side of the axis"

    reflected = np.array([_reflect_point(tuple(map(int, p)), axis) for p in red_coords], dtype=int)
    H, W = g.shape
    assert np.all((0 <= reflected[:, 0]) & (reflected[:, 0] < H))
    assert np.all((0 <= reflected[:, 1]) & (reflected[:, 1] < W))

    all_box_coords = np.vstack([red_coords, reflected, axis_coords])
    box = _bbox_coords(all_box_coords)
    r0, c0, r1, c1 = box
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for rr, cc in np.vstack([red_coords, reflected]):
        sprite[int(rr) - r0, int(cc) - c0] = RED
    return sprite, box


def _orient_sprite(sprite: np.ndarray, source_kind: str, target_kind: str) -> np.ndarray:
    source_sign = CORNER_SIGN[source_kind]
    target_sign = CORNER_SIGN[target_kind]
    out = sprite.copy()
    if source_sign[0] != target_sign[0]:
        out = out[::-1, :]
    if source_sign[1] != target_sign[1]:
        out = out[:, ::-1]
    return out


def _paste_origin(missing: tuple[int, int], sign: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = missing
    return (r - h + 1 if sign[0] < 0 else r, c - w + 1 if sign[1] < 0 else c)


def _stamp_sprite(out: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> None:
    h, w = sprite.shape
    H, W = out.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W, "sprite stamp out of bounds"
    mask = sprite == RED
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(target[mask] == YELLOW), "sprite should not overwrite yellow frame corners"
    target[mask] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the half-drawn sprite by grey-axis reflection and distribute it to every missing-corner frame.

    Input structure:
    - The dominant color is the background; yellow singleton pixels form several equal square frames with exactly one missing corner each.
    - One red half-sprite is drawn on only one side of a contiguous grey symmetry-axis segment.
    - The grey axis and red half-sprite occupy the outside missing-corner position of exactly one yellow frame.

    Rule:
    - Reflect the red half-sprite across the grey axis to obtain the full sprite, then erase the grey axis.
    - Identify the source frame by the full sprite bbox touching that frame's missing corner.
    - Stamp the full sprite at every equal-size frame's missing corner, flipping vertically and/or horizontally when the target missing quadrant differs from the source quadrant.
    - Preserve every yellow corner marker.

    Key insight:
    - The full sprite never appears in the input: the grey axis supplies the missing half, while each three-corner frame supplies both an anchor point and the flip orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    sprite, sprite_box = _reconstruct_sprite(g, bg)
    controls = _three_corner_squares(g, YELLOW)
    source_controls = [
        control
        for control in controls
        if _corner_matches_bbox(control["missing_kind"], control["missing"], sprite_box)
    ]
    assert len(source_controls) == 1, "expected one source frame touching the reconstructed sprite"
    source = source_controls[0]

    controls = _three_corner_squares(g, YELLOW, side=source["side"])
    marker_coords = [tuple(map(int, p)) for p in np.argwhere(g == YELLOW)]
    used_markers = [marker for control in controls for marker in control["markers"]]
    assert sorted(used_markers) == sorted(marker_coords), "yellow markers must form only the intended equal-size frames"

    out = g.copy()
    out[g == RED] = bg
    out[g == GREY] = bg

    h, w = sprite.shape
    source_kind = source["missing_kind"]
    for control in controls:
        target_kind = control["missing_kind"]
        oriented = _orient_sprite(sprite, source_kind, target_kind)
        pr, pc = _paste_origin(control["missing"], CORNER_SIGN[target_kind], h, w)
        _stamp_sprite(out, oriented, pr, pc)
    return out


# ---------------- generator ----------------

GEN_RETRIES = 500
GEN_BACKGROUNDS = (0, 1, 3, 6, 7, 8, 9)
GEN_AXIS_KINDS = ("diag_main", "diag_anti")
GEN_SPRITE_SIZES = (5, 6, 7)
GEN_FRAME_SIDES = (3, 4, 5)
GEN_GRID_H_RANGE = (24, 30)
GEN_GRID_W_RANGE = (24, 30)
GEN_FRAME_COUNT_RANGE = (4, 7)
GEN_MARKER_CLEARANCE = 1


def _gen_square_corners(r0: int, c0: int, side: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c0 + side),
        "bl": (r0 + side, c0),
        "br": (r0 + side, c0 + side),
    }


def _gen_frame(kind: str, r0: int, c0: int, side: int, n: int) -> dict:
    corners = _gen_square_corners(r0, c0, side)
    missing = corners[kind]
    pr, pc = _paste_origin(missing, CORNER_SIGN[kind], n, n)
    return {
        "kind": kind,
        "missing": missing,
        "markers": tuple(corners[k] for k in KINDS if k != kind),
        "paste": (pr, pc),
    }


def _gen_candidate_frames(kind: str, H: int, W: int, side: int, n: int) -> list[dict]:
    frames = []
    for r0 in range(H - side):
        for c0 in range(W - side):
            frame = _gen_frame(kind, r0, c0, side, n)
            pr, pc = frame["paste"]
            if 0 <= pr and 0 <= pc and pr + n <= H and pc + n <= W:
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
                    missing = next(k for k in KINDS if k not in present)
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
    return _gen_clean_controls(placed + [frame], side)


def _gen_mark_frame(frame: dict, paste_used: np.ndarray, marker_used: np.ndarray, marker_forbidden: np.ndarray, n: int) -> None:
    H, W = paste_used.shape
    pr, pc = frame["paste"]
    paste_used[pr : pr + n, pc : pc + n] = True
    pad = GEN_MARKER_CLEARANCE
    for r, c in frame["markers"]:
        marker_used[r, c] = True
        marker_forbidden[max(0, r - pad) : min(H, r + pad + 1), max(0, c - pad) : min(W, c + pad + 1)] = True


def _gen_place_frames(H: int, W: int, kinds: list[str], side: int, n: int) -> list[dict] | None:
    paste_used = np.zeros((H, W), dtype=bool)
    marker_used = np.zeros((H, W), dtype=bool)
    marker_forbidden = np.zeros((H, W), dtype=bool)
    placed: list[dict] = []
    for kind in kinds:
        for frame in _gen_candidate_frames(kind, H, W, side, n):
            if _gen_can_place(frame, placed, paste_used, marker_used, marker_forbidden, side, n):
                placed.append(frame)
                _gen_mark_frame(frame, paste_used, marker_used, marker_forbidden, n)
                break
        else:
            return None
    return placed


def _gen_axis_cells(n: int, axis_kind: str) -> set[tuple[int, int]]:
    if axis_kind == "diag_main":
        return {(i, i) for i in range(n)}
    if axis_kind == "diag_anti":
        return {(i, n - 1 - i) for i in range(n)}
    raise AssertionError(f"unknown axis kind: {axis_kind}")


def _gen_mirror_cell(r: int, c: int, n: int, axis_kind: str) -> tuple[int, int]:
    if axis_kind == "diag_main":
        return c, r
    if axis_kind == "diag_anti":
        return n - 1 - c, n - 1 - r
    raise AssertionError(f"unknown axis kind: {axis_kind}")


def _gen_negative_side_cells(n: int, axis_kind: str) -> list[tuple[int, int]]:
    cells = []
    for r in range(n):
        for c in range(n):
            if axis_kind == "diag_main" and r - c < 0:
                cells.append((r, c))
            elif axis_kind == "diag_anti" and r + c - (n - 1) < 0:
                cells.append((r, c))
    return cells


def _gen_seed_half(n: int, axis_kind: str) -> set[tuple[int, int]]:
    if axis_kind == "diag_main":
        return {(i, i + 1) for i in range(n - 1)}
    return {(i, n - 2 - i) for i in range(n - 1)}


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    count = 0
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
    return count


def _gen_sprite_masks(n: int, axis_kind: str) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    side_cells = _gen_negative_side_cells(n, axis_kind)
    seed = _gen_seed_half(n, axis_kind)
    choices = [p for p in side_cells if p not in seed]

    for _ in range(200):
        half = set(seed)
        random.shuffle(choices)
        extra_count = random.randint(2, min(7, len(choices)))
        half.update(choices[:extra_count])
        full = set(half)
        full.update(_gen_mirror_cell(r, c, n, axis_kind) for r, c in half)

        mask = np.zeros((n, n), dtype=bool)
        for r, c in full:
            mask[r, c] = True
        orientations = {
            mask.tobytes(),
            mask[::-1, :].tobytes(),
            mask[:, ::-1].tobytes(),
            mask[::-1, ::-1].tobytes(),
        }
        if len(orientations) != 4:
            continue
        if _count_components(mask) != 1:
            continue
        if random.random() < 0.5:
            half = {_gen_mirror_cell(r, c, n, axis_kind) for r, c in half}
        return mask, half, _gen_axis_cells(n, axis_kind)

    raise RuntimeError("failed to generate a non-degenerate symmetric sprite")


def _gen_sprite_from_mask(mask: np.ndarray, bg: int) -> np.ndarray:
    return np.where(mask, RED, int(bg)).astype(int)


def _gen_source_patch(n: int, half: set[tuple[int, int]], axis_cells: set[tuple[int, int]], bg: int) -> np.ndarray:
    patch = np.full((n, n), int(bg), dtype=int)
    for r, c in axis_cells:
        patch[r, c] = GREY
    for r, c in half:
        patch[r, c] = RED
    return patch


def _gen_paste_non_bg(out: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != int(bg)
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(target[mask] == YELLOW)
    target[mask] = patch[mask]


def _gen_build_output(H: int, W: int, bg: int, frames: list[dict], full_sprite: np.ndarray, source_kind: str) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    for frame in frames:
        for r, c in frame["markers"]:
            out[r, c] = YELLOW
    for frame in frames:
        oriented = _orient_sprite(full_sprite, source_kind, frame["kind"])
        pr, pc = frame["paste"]
        _stamp_sprite(out, oriented, pr, pc)
    return out


def _gen_build_input(out: np.ndarray, source_frame: dict, source_patch: np.ndarray, bg: int) -> np.ndarray:
    inp = out.copy()
    inp[inp == RED] = int(bg)
    pr, pc = source_frame["paste"]
    _gen_paste_non_bg(inp, source_patch, pr, pc, bg)
    return inp


def _gen_kind_sequence(k: int) -> list[str]:
    kinds = list(KINDS) + [random.choice(KINDS) for _ in range(k - len(KINDS))]
    random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_GRID_H_RANGE)
    W = random.randint(*GEN_GRID_W_RANGE)
    bg = random.choice(GEN_BACKGROUNDS)
    n = random.choice(GEN_SPRITE_SIZES)
    side = random.choice(GEN_FRAME_SIDES)
    frame_count = random.randint(*GEN_FRAME_COUNT_RANGE)
    kinds = _gen_kind_sequence(frame_count)
    source_i = random.randrange(frame_count)
    source_kind = kinds[source_i]
    axis_kind = random.choice(GEN_AXIS_KINDS)

    try:
        full_mask, half_cells, axis_cells = _gen_sprite_masks(n, axis_kind)
    except RuntimeError:
        return None
    full_sprite = _gen_sprite_from_mask(full_mask, bg)
    source_patch = _gen_source_patch(n, half_cells, axis_cells, bg)

    frames = _gen_place_frames(H, W, kinds, side, n)
    if frames is None:
        return None

    out = _gen_build_output(H, W, bg, frames, full_sprite, source_kind)
    inp = _gen_build_input(out, frames[source_i], source_patch, bg)

    if not set(KINDS).issubset({frame["kind"] for frame in frames}):
        return None
    if np.count_nonzero(inp == RED) >= np.count_nonzero(out == RED):
        return None
    if np.any(out == GREY):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-sprite corner-frame input")

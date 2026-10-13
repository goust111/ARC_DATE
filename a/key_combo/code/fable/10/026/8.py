from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, get_bg


MASK_COLOR = Color.GREY
BACKGROUND_COLOR = Color.BLACK
PALETTE = tuple(color for color in Color.NOT_BLACK if color != MASK_COLOR)
GEN_ATTEMPTS = 400


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, tuple[int, int, int, int]]:
    mask = grid != bg
    box = bbox(mask)
    assert box is not None, "expected a foreground pattern"
    r0, c0, r1, c1 = map(int, box)

    separators: list[tuple[str, int]] = []
    for r in range(r0 + 1, r1):
        if np.all(grid[r, c0 : c1 + 1] == bg):
            if np.any(grid[r0:r, c0 : c1 + 1] != bg) and np.any(grid[r + 1 : r1 + 1, c0 : c1 + 1] != bg):
                separators.append(("row", r))
    for c in range(c0 + 1, c1):
        if np.all(grid[r0 : r1 + 1, c] == bg):
            if np.any(grid[r0 : r1 + 1, c0:c] != bg) and np.any(grid[r0 : r1 + 1, c + 1 : c1 + 1] != bg):
                separators.append(("col", c))

    assert len(separators) == 1, "expected exactly one blank interior separator"
    return separators[0][0], int(separators[0][1]), box


def _rot180_partner(
    r: int,
    c: int,
    axis: str,
    separator: int,
    box: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    if axis == "row":
        return 2 * separator - int(r), c0 + c1 - int(c)
    return r0 + r1 - int(r), 2 * separator - int(c)


def _mirror_partner(r: int, c: int, axis: str, separator: int) -> tuple[int, int]:
    if axis == "row":
        return 2 * separator - int(r), int(c)
    return int(r), 2 * separator - int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """Restore gray mask pixels from their 180-degree counterparts around the blank interior hinge line.

    Input structure:
    - A multicolor foreground pattern sits on a background and has one bounding box.
    - Strictly inside that bounding box, exactly one whole row or one whole column is blank.
    - Cells on opposite sides of that blank line match by 180-degree rotation about the line midpoint, with some cells replaced by gray masks.

    Rule:
    - Find the foreground bounding box and its unique blank interior row or column.
    - Treat the midpoint of that blank line segment as the 180-degree rotation center.
    - Replace each gray mask with the color at its rotated counterpart; keep every other cell, including the blank line, unchanged.

    Key insight:
    - The empty interior line marks a hinge center, not a mirror axis: the correct correspondence is point symmetry through a half-turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    axis, separator, box = _find_separator(g, bg)
    r0, c0, r1, c1 = box

    out = g.copy()
    mask_coords = np.argwhere(g == MASK_COLOR)
    assert len(mask_coords) > 0, "expected gray mask pixels"

    for rr, cc in mask_coords:
        r = int(rr)
        c = int(cc)
        if not (r0 <= r <= r1 and c0 <= c <= c1):
            continue
        pr, pc = _rot180_partner(r, c, axis, separator, box)
        assert r0 <= pr <= r1 and c0 <= pc <= c1, "rotated partner must stay inside the pattern bbox"
        value = int(g[pr, pc])
        assert value not in {bg, MASK_COLOR}, "every mask must have a visible rotated partner"
        out[r, c] = value

    return out


def _random_half(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    half = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    occupied = np.zeros((rows, cols), dtype=bool)

    for r in range(rows):
        occupied[r, int(rng.integers(cols))] = True
    for c in range(cols):
        occupied[int(rng.integers(rows)), c] = True

    occupied |= rng.random((rows, cols)) < rng.uniform(0.22, 0.42)
    if occupied.mean() < 0.35:
        extras = rng.random((rows, cols)) < 0.2
        occupied |= extras
    if occupied.mean() > 0.78:
        return _random_half(rng, rows, cols)

    palette_size = int(rng.integers(3, min(6, len(PALETTE)) + 1))
    palette = rng.choice(PALETTE, size=palette_size, replace=False)
    values = rng.choice(palette, size=int(occupied.sum()), replace=True)
    half[occupied] = values
    if len(set(map(int, half[occupied]))) < 2:
        return _random_half(rng, rows, cols)
    return half


def _row_local_output(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    half_h = int(rng.integers(2, 6))
    width = int(rng.integers(5, 10))
    top = _random_half(rng, half_h, width)
    if np.array_equal(top, top[:, ::-1]):
        raise ValueError("row half accidentally mirror-symmetric")

    local = np.full((2 * half_h + 1, width), BACKGROUND_COLOR, dtype=int)
    orbits: list[dict] = []
    for r, c in np.argwhere(top != BACKGROUND_COLOR):
        tr = int(r)
        tc = int(c)
        br = 2 * half_h - tr
        bc = width - 1 - tc
        color = int(top[tr, tc])
        local[tr, tc] = color
        local[br, bc] = color
        orbits.append({"a": (tr, tc), "b": (br, bc), "color": color})
    return local, orbits


def _col_local_output(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    height = int(rng.integers(5, 10))
    half_w = int(rng.integers(2, 6))
    left = _random_half(rng, height, half_w)
    if np.array_equal(left, left[::-1, :]):
        raise ValueError("column half accidentally mirror-symmetric")

    local = np.full((height, 2 * half_w + 1), BACKGROUND_COLOR, dtype=int)
    orbits: list[dict] = []
    for r, c in np.argwhere(left != BACKGROUND_COLOR):
        lr = int(r)
        lc = int(c)
        rr = height - 1 - lr
        rc = 2 * half_w - lc
        color = int(left[lr, lc])
        local[lr, lc] = color
        local[rr, rc] = color
        orbits.append({"a": (lr, lc), "b": (rr, rc), "color": color})
    return local, orbits


def _pick_masks(
    rng: np.random.Generator,
    local_output: np.ndarray,
    axis: str,
    orbits: list[dict],
) -> np.ndarray:
    local_input = local_output.copy()
    h, w = local_output.shape
    separator = h // 2 if axis == "row" else w // 2
    box = (0, 0, h - 1, w - 1)

    options = []
    if any(True for orbit in orbits):
        options.append("a_only")
        options.append("b_only")
    if len(orbits) >= 2:
        options.append("both")
    mode = str(rng.choice(options))

    max_masks = min(8, len(orbits))
    target = int(rng.integers(2, max_masks + 1))

    def decoy_fails(cell: tuple[int, int], expect: int) -> bool:
        mr, mc = _mirror_partner(cell[0], cell[1], axis, separator)
        if not (0 <= mr < h and 0 <= mc < w):
            return True
        return int(local_output[mr, mc]) != int(expect)

    selectable = {"a": [], "b": []}
    decoyable = {"a": [], "b": []}
    for idx, orbit in enumerate(orbits):
        for side in ("a", "b"):
            cell = orbit[side]
            selectable[side].append(idx)
            if decoy_fails(cell, orbit["color"]):
                decoyable[side].append(idx)

    if mode == "both":
        na = int(rng.integers(1, target))
        nb = target - na
    elif mode == "a_only":
        na, nb = target, 0
    else:
        na, nb = 0, target

    if na > len(selectable["a"]) or nb > len(selectable["b"]):
        raise ValueError("not enough mask candidates")

    chosen_a: set[int] = set()
    chosen_b: set[int] = set()

    if na:
        pool = selectable["a"][:]
        rng.shuffle(pool)
        chosen_a = set(pool[:na])
    if nb:
        pool = selectable["b"][:]
        rng.shuffle(pool)
        chosen_b = set(pool[:nb])

    if not any(idx in chosen_a for idx in decoyable["a"]) and not any(idx in chosen_b for idx in decoyable["b"]):
        repair_side = None
        repair_idx = None
        if na and decoyable["a"]:
            repair_side = "a"
            repair_idx = int(rng.choice(decoyable["a"]))
        elif nb and decoyable["b"]:
            repair_side = "b"
            repair_idx = int(rng.choice(decoyable["b"]))
        elif mode != "a_only" and decoyable["a"]:
            repair_side = "a"
            repair_idx = int(rng.choice(decoyable["a"]))
            chosen_a.add(repair_idx)
            if len(chosen_a) + len(chosen_b) > max_masks:
                chosen_b.pop()
        elif decoyable["b"]:
            repair_side = "b"
            repair_idx = int(rng.choice(decoyable["b"]))
            chosen_b.add(repair_idx)
            if len(chosen_a) + len(chosen_b) > max_masks:
                chosen_a.pop()
        else:
            raise ValueError("no mask position defeats the mirror decoy")

        if repair_side == "a" and repair_idx is not None:
            if not chosen_a and na == 0:
                chosen_a.add(repair_idx)
            else:
                chosen_a.add(repair_idx)
                while len(chosen_a) > max(na, 1):
                    victim = next(iter(chosen_a - {repair_idx}))
                    chosen_a.remove(victim)
        if repair_side == "b" and repair_idx is not None:
            if not chosen_b and nb == 0:
                chosen_b.add(repair_idx)
            else:
                chosen_b.add(repair_idx)
                while len(chosen_b) > max(nb, 1):
                    victim = next(iter(chosen_b - {repair_idx}))
                    chosen_b.remove(victim)

    if len(chosen_a) + len(chosen_b) < 2:
        raise ValueError("need at least two masked cells")

    for idx in chosen_a:
        r, c = orbits[idx]["a"]
        local_input[r, c] = MASK_COLOR
    for idx in chosen_b:
        r, c = orbits[idx]["b"]
        local_input[r, c] = MASK_COLOR

    mask_visible = (local_input != BACKGROUND_COLOR) & (local_input != MASK_COLOR)
    if axis == "row":
        for r in range(h):
            if r == separator:
                continue
            if not np.any(mask_visible[r]):
                raise ValueError("masking hid an entire non-separator row")
        for c in range(w):
            if not np.any(mask_visible[:, c]):
                raise ValueError("masking hid an entire column")
    else:
        for c in range(w):
            if c == separator:
                continue
            if not np.any(mask_visible[:, c]):
                raise ValueError("masking hid an entire non-separator column")
        for r in range(h):
            if not np.any(mask_visible[r]):
                raise ValueError("masking hid an entire row")

    for rr, cc in np.argwhere(local_input == MASK_COLOR):
        r = int(rr)
        c = int(cc)
        pr, pc = _rot180_partner(r, c, axis, separator, box)
        if int(local_input[pr, pc]) in {BACKGROUND_COLOR, MASK_COLOR}:
            raise ValueError("masked cell lost its visible rotated partner")

    return local_input


def _embed_pair(
    rng: np.random.Generator,
    local_in: np.ndarray,
    local_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assert local_in.shape == local_out.shape
    lh, lw = local_in.shape
    extra_h = int(rng.integers(2, 8))
    extra_w = int(rng.integers(2, 8))
    H = lh + extra_h
    W = lw + extra_w
    if H > 30 or W > 30:
        raise ValueError("canvas too large")

    top = int(rng.integers(1, H - lh))
    left = int(rng.integers(1, W - lw))
    inp = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    out = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    inp[top : top + lh, left : left + lw] = local_in
    out[top : top + lh, left : left + lw] = local_out
    return inp, out


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    assert inp.shape == out.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert out.ndim == 2 and max(out.shape) <= 30
    assert np.any(inp == MASK_COLOR), "expected at least one gray mask"

    bg = int(get_bg(inp))
    axis, separator, box = _find_separator(inp, bg)
    r0, c0, r1, c1 = box
    solved = main(inp)
    assert np.array_equal(solved, out), "solver output must match constructed output"

    mask_coords = np.argwhere(inp == MASK_COLOR)
    decoy_failure = False
    for rr, cc in mask_coords:
        r = int(rr)
        c = int(cc)
        assert r0 <= r <= r1 and c0 <= c <= c1, "masks belong inside the pattern bbox"
        pr, pc = _rot180_partner(r, c, axis, separator, box)
        assert int(inp[pr, pc]) not in {bg, MASK_COLOR}
        assert int(out[r, c]) == int(inp[pr, pc])

        mr, mc = _mirror_partner(r, c, axis, separator)
        if not (r0 <= mr <= r1 and c0 <= mc <= c1):
            decoy_failure = True
        elif int(inp[mr, mc]) != int(out[r, c]):
            decoy_failure = True

    assert decoy_failure, "at least one mask must reject the mirror shortcut"

    visible = (inp != bg) & (inp != MASK_COLOR)
    for rr, cc in np.argwhere(visible):
        r = int(rr)
        c = int(cc)
        pr, pc = _rot180_partner(r, c, axis, separator, box)
        partner_value = int(inp[pr, pc])
        assert partner_value in {int(inp[r, c]), MASK_COLOR}, "visible cells must agree with 180-degree symmetry"

    if axis == "row":
        for r in range(r0, r1 + 1):
            if r == separator:
                assert np.all(inp[r, c0 : c1 + 1] == bg)
            else:
                assert np.any(visible[r, c0 : c1 + 1])
    else:
        for c in range(c0, c1 + 1):
            if c == separator:
                assert np.all(inp[r0 : r1 + 1, c] == bg)
            else:
                assert np.any(visible[r0 : r1 + 1, c])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    axis = str(rng.choice(["row", "col"]))
    try:
        if axis == "row":
            local_out, orbits = _row_local_output(rng)
        else:
            local_out, orbits = _col_local_output(rng)
        local_in = _pick_masks(rng, local_out, axis, orbits)
        inp, out = _embed_pair(rng, local_in, local_out)
    except ValueError:
        return None

    try:
        in_box = bbox(inp != BACKGROUND_COLOR)
        out_box = bbox(out != BACKGROUND_COLOR)
        assert in_box == out_box, "input/output bbox must stay aligned"
        _validate_instance(inp, out)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hinge-rotation sample")

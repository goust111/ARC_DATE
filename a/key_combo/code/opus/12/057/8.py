import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

PATTERN_COLORS = (BLUE, RED, GREEN, YELLOW)
MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_coord(r: int, c: int, shape: tuple[int, int], axis: str) -> tuple[int, int]:
    h, w = shape
    if axis == "vertical":
        return r, w - 1 - c
    if axis == "horizontal":
        return h - 1 - r, c
    raise ValueError(axis)


def _analyze_bar(coords: np.ndarray, shape: tuple[int, int]) -> dict:
    h, w = shape
    r0, c0, r1, c1 = _bbox(coords)
    area = int(coords.shape[0])
    if c0 == c1 and area == (r1 - r0 + 1) and c0 in {0, w - 1} and area >= 3:
        return {"kind": "vertical", "edge": c0, "r0": r0, "r1": r1, "c0": c0, "c1": c1}
    if r0 == r1 and area == (c1 - c0 + 1) and r0 in {0, h - 1} and area >= 3:
        return {"kind": "horizontal", "edge": r0, "r0": r0, "r1": r1, "c0": c0, "c1": c1}
    raise AssertionError("maroon markers must be straight edge bars of length >= 3")


def _infer_axis(grid: np.ndarray) -> str:
    h, w = grid.shape
    comps = _components(grid == MAROON)
    assert len(comps) == 2, "expected exactly two maroon edge bars"
    bars = [_analyze_bar(comp, grid.shape) for comp in comps]

    if all(bar["kind"] == "vertical" for bar in bars):
        left, right = sorted(bars, key=lambda bar: bar["c0"])
        assert left["c0"] == 0 and right["c0"] == w - 1, "vertical bars must be on opposite side edges"
        assert left["r0"] == h - 1 - right["r1"] and left["r1"] == h - 1 - right["r0"], (
            "side-edge bar row ranges must be symmetric about the horizontal center line"
        )
        return "vertical"

    if all(bar["kind"] == "horizontal" for bar in bars):
        top, bottom = sorted(bars, key=lambda bar: bar["r0"])
        assert top["r0"] == 0 and bottom["r0"] == h - 1, "horizontal bars must be on opposite top/bottom edges"
        assert top["c0"] == w - 1 - bottom["c1"] and top["c1"] == w - 1 - bottom["c0"], (
            "top/bottom bar column ranges must be symmetric about the vertical center line"
        )
        return "horizontal"

    raise AssertionError("maroon bars must both be vertical or both be horizontal")


def _reconstruct_with_axis(grid: np.ndarray, axis: str) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    grey_coords = np.argwhere(out == GREY)
    assert grey_coords.size > 0, "expected one grey occlusion rectangle"
    grey_comps = _components(out == GREY)
    assert len(grey_comps) == 1, "expected exactly one grey component"
    r0, c0, r1, c1 = _bbox(grey_comps[0])
    assert (r1 - r0 + 1) * (c1 - c0 + 1) == int(grey_comps[0].shape[0]), "grey region must be a solid rectangle"

    for r, c in grey_coords:
        mr, mc = _mirror_coord(int(r), int(c), out.shape, axis)
        src = int(out[mr, mc])
        assert src not in {GREY, MAROON}, "each grey pixel must map to a visible non-grey, non-maroon source"
        out[r, c] = src
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the grey occluded region by inferring the mirror axis from the symmetric maroon edge bars.

    Input structure:
    - The grid has black background, one multicolor foreground pattern using 2-3 colors from blue/red/green/yellow, and one solid grey rectangle covering part of that pattern.
    - The foreground pattern itself is mirror-symmetric about exactly one center axis: either left-right symmetry about the vertical center axis or up-down symmetry about the horizontal center axis.
    - Two maroon bars lie on opposite edges as metadata: either vertical bars on the left/right edges or horizontal bars on the top/bottom edges.

    Rule:
    - First locate the two maroon edge bars and inspect their orientation.
    - Left/right maroon bars whose row ranges are symmetric about the horizontal center line indicate that the hidden pattern axis is vertical; top/bottom maroon bars whose column ranges are symmetric about the vertical center line indicate that the hidden pattern axis is horizontal.
    - Using that inferred axis, replace every grey cell by the color at its mirror coordinate while leaving the visible pattern and the maroon bars unchanged.

    Key insight:
    - The maroon edge bars are not part of the picture to restore; their own geometric symmetry encodes which mirror axis must be used, and choosing the wrong axis reconstructs a different region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    axis = _infer_axis(g)
    return _reconstruct_with_axis(g, axis)


def _choose_bar_span(total: int, length: int) -> tuple[int, int]:
    starts = list(range(1, total - length))
    random.shuffle(starts)
    for start in starts:
        mate = total - start - length
        if mate != start:
            return start, start + length - 1
    raise RuntimeError("failed to place asymmetric edge-bar span")


def _paint_source_rect(source: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    source[r0 : r1 + 1, c0 : c1 + 1] = color


def _make_symmetric_pattern(h: int, w: int, axis: str, colors: list[int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if axis == "vertical":
        source = np.zeros((h, w), dtype=int)
        row_max = h - 3
        col_max = w // 2 - 1
        assert row_max >= 2 and col_max >= 3
        anchor_h = random.randint(2, min(4, h - 5))
        anchor_w = random.randint(2, min(4, max(2, col_max - 1)))
        anchor_r0 = random.randint(2, h - anchor_h - 3)
        anchor_c0 = random.randint(2, col_max - anchor_w + 1)
        anchor = (anchor_r0, anchor_c0, anchor_r0 + anchor_h - 1, anchor_c0 + anchor_w - 1)
        _paint_source_rect(source, anchor, colors[0])

        extra_count = random.randint(4, 7)
        color_cycle = list(colors[1:]) + [random.choice(colors) for _ in range(extra_count)]
        random.shuffle(color_cycle)
        for color in color_cycle:
            for _ in range(30):
                rh = random.randint(1, min(3, h - 4))
                rw = random.randint(1, min(3, max(1, col_max - 1)))
                r0 = random.randint(2, h - rh - 3)
                c0 = random.randint(2, col_max - rw + 1)
                patch = source[r0 : r0 + rh, c0 : c0 + rw]
                if np.any(patch != BLACK):
                    continue
                _paint_source_rect(source, (r0, c0, r0 + rh - 1, c0 + rw - 1), color)
                break

        pattern = np.zeros((h, w), dtype=int)
        for r, c in np.argwhere(source != BLACK):
            color = int(source[r, c])
            mr, mc = _mirror_coord(int(r), int(c), (h, w), "vertical")
            pattern[r, c] = color
            pattern[mr, mc] = color
        return pattern, anchor

    source = np.zeros((h, w), dtype=int)
    row_max = h // 2 - 1
    col_max = w - 3
    assert row_max >= 3 and col_max >= 2
    anchor_h = random.randint(2, min(4, max(2, row_max - 1)))
    anchor_w = random.randint(2, min(4, w - 5))
    anchor_r0 = random.randint(2, row_max - anchor_h + 1)
    anchor_c0 = random.randint(2, w - anchor_w - 3)
    anchor = (anchor_r0, anchor_c0, anchor_r0 + anchor_h - 1, anchor_c0 + anchor_w - 1)
    _paint_source_rect(source, anchor, colors[0])

    extra_count = random.randint(4, 7)
    color_cycle = list(colors[1:]) + [random.choice(colors) for _ in range(extra_count)]
    random.shuffle(color_cycle)
    for color in color_cycle:
        for _ in range(30):
            rh = random.randint(1, min(3, max(1, row_max - 1)))
            rw = random.randint(1, min(3, w - 4))
            r0 = random.randint(2, row_max - rh + 1)
            c0 = random.randint(2, w - rw - 3)
            patch = source[r0 : r0 + rh, c0 : c0 + rw]
            if np.any(patch != BLACK):
                continue
            _paint_source_rect(source, (r0, c0, r0 + rh - 1, c0 + rw - 1), color)
            break

    pattern = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(source != BLACK):
        color = int(source[r, c])
        mr, mc = _mirror_coord(int(r), int(c), (h, w), "horizontal")
        pattern[r, c] = color
        pattern[mr, mc] = color
    return pattern, anchor


def _add_bars(grid: np.ndarray, axis: str) -> None:
    h, w = grid.shape
    if axis == "vertical":
        length = random.randint(3, min(6, h - 2))
        r0, r1 = _choose_bar_span(h, length)
        rr0 = h - 1 - r1
        rr1 = h - 1 - r0
        grid[r0 : r1 + 1, 0] = MAROON
        grid[rr0 : rr1 + 1, w - 1] = MAROON
        return

    length = random.randint(3, min(6, w - 2))
    c0, c1 = _choose_bar_span(w, length)
    cc0 = w - 1 - c1
    cc1 = w - 1 - c0
    grid[0, c0 : c1 + 1] = MAROON
    grid[h - 1, cc0 : cc1 + 1] = MAROON


def _make_occluder(anchor: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = anchor
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    occ_h = random.randint(2, height)
    occ_w = random.randint(2, width)
    occ_r0 = random.randint(r0, r1 - occ_h + 1)
    occ_c0 = random.randint(c0, c1 - occ_w + 1)
    return occ_r0, occ_c0, occ_r0 + occ_h - 1, occ_c0 + occ_w - 1


def _rect_on_one_side(rect: tuple[int, int, int, int], shape: tuple[int, int], axis: str) -> bool:
    r0, c0, r1, c1 = rect
    h, w = shape
    if axis == "vertical":
        center_left = (w - 1) / 2.0
        return c1 < center_left or c0 > center_left
    center_top = (h - 1) / 2.0
    return r1 < center_top or r0 > center_top


def _axis_symmetric(pattern: np.ndarray, axis: str) -> bool:
    if axis == "vertical":
        return bool(np.array_equal(pattern, np.fliplr(pattern)))
    return bool(np.array_equal(pattern, np.flipud(pattern)))


def _wrong_axis(axis: str) -> str:
    return "horizontal" if axis == "vertical" else "vertical"


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    axis = random.choice(("vertical", "horizontal"))
    colors = random.sample(PATTERN_COLORS, random.choice((2, 3)))

    pattern, anchor = _make_symmetric_pattern(h, w, axis, colors)
    if len(set(map(int, pattern[pattern != BLACK].tolist()))) < 2:
        raise RuntimeError("pattern used too few colors")
    if not _axis_symmetric(pattern, axis):
        raise RuntimeError("pattern lost intended symmetry")
    if _axis_symmetric(pattern, _wrong_axis(axis)):
        raise RuntimeError("pattern accidentally has both symmetries")

    clean = pattern.copy()
    _add_bars(clean, axis)

    occ = _make_occluder(anchor)
    if random.random() < 0.5:
        or0, oc0, or1, oc1 = occ
        mr0, mc0 = _mirror_coord(or0, oc0, clean.shape, axis)
        mr1, mc1 = _mirror_coord(or1, oc1, clean.shape, axis)
        occ = (min(mr0, mr1), min(mc0, mc1), max(mr0, mr1), max(mc0, mc1))

    assert _rect_on_one_side(occ, clean.shape, axis)
    r0, c0, r1, c1 = occ
    input_grid = clean.copy()
    input_grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    return input_grid, clean


def _validate_sample(inp: np.ndarray, out: np.ndarray) -> bool:
    axis = _infer_axis(inp)
    wrong = _wrong_axis(axis)

    grey_comps = _components(inp == GREY)
    if len(grey_comps) != 1:
        return False
    occ = _bbox(grey_comps[0])
    if not _rect_on_one_side(occ, inp.shape, axis):
        return False

    for r, c in grey_comps[0]:
        mr, mc = _mirror_coord(int(r), int(c), inp.shape, axis)
        if int(inp[mr, mc]) in {GREY, MAROON}:
            return False

    pattern_only = out.copy()
    pattern_only[pattern_only == MAROON] = BLACK
    if not _axis_symmetric(pattern_only, axis):
        return False
    if _axis_symmetric(pattern_only, wrong):
        return False

    recovered = _reconstruct_with_axis(inp, axis)
    if not np.array_equal(recovered, out):
        return False

    try:
        wrong_recovered = _reconstruct_with_axis(inp, wrong)
    except AssertionError:
        wrong_recovered = None
    if wrong_recovered is not None and np.array_equal(wrong_recovered, out):
        return False

    if np.array_equal(inp, out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        try:
            inp, out = _build_sample()
        except RuntimeError:
            continue
        if _validate_sample(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid sample for puzzle 8")

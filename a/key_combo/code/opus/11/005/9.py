from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
CORNER_TO_DELTA = {
    0: (0, 0),  # top-left
    1: (0, 1),  # top-right
    2: (1, 0),  # bottom-left
    3: (1, 1),  # bottom-right
}
MAX_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return []
    cells = {tuple(map(int, rc)) for rc in coords}
    comps: list[np.ndarray] = []
    while cells:
        start = cells.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cells:
                    cells.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 7 or w < 7:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub == int(color)
    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    if np.array_equal(mask, perimeter):
        return r0, c0, r1, c1
    return None


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _decode_palette(grid: np.ndarray) -> dict[int, int]:
    H, W = grid.shape
    if H < 2:
        raise AssertionError("grid too short for palette")
    band = grid[H - 2 : H, :]
    mapping: dict[int, int] = {}
    covered = np.zeros_like(band, dtype=bool)
    c = 0
    while c < W - 1:
        window = band[:, c : c + 2]
        vals = set(map(int, window.flat))
        nontrivial = vals - {BG}
        if GREY in nontrivial:
            color_vals = nontrivial - {GREY}
            if len(color_vals) == 1 and int((window == GREY).sum()) == 1:
                color = next(iter(color_vals))
                if int((window == color).sum()) == 3:
                    gr, gc = map(int, np.argwhere(window == GREY)[0])
                    mapping[int(color)] = 2 * gr + gc
                    covered[:, c : c + 2] |= (window != BG)
                    c += 2
                    continue
        c += 1

    assert 3 <= len(mapping) <= 4, "expected 3-4 palette swatches"
    for rr in range(2):
        for cc in range(W):
            if int(band[rr, cc]) != BG:
                assert covered[rr, cc], "palette band contains unexpected foreground pixels"
    return mapping


def _find_frames(grid: np.ndarray, frame_colors: set[int]) -> dict[int, tuple[int, int, int, int]]:
    frames: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(frame_colors):
        candidates = []
        for comp in _components_of_color(grid, color):
            box = _is_rectangular_frame(grid, color, comp)
            if box is not None:
                candidates.append(box)
        assert len(candidates) == 1, "expected exactly one frame per palette color"
        frames[int(color)] = candidates[0]
    return frames


def _read_cross_center(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = interior.shape
    mask = interior == GREY
    assert int(mask.sum()) == ih + iw - 1, "cross must be one full interior row plus one full interior column"
    rows = [r for r in range(ih) if bool(mask[r, :].all())]
    cols = [c for c in range(iw) if bool(mask[:, c].all())]
    assert len(rows) == 1 and len(cols) == 1, "cross must have one center row and one center column"
    rr = int(rows[0])
    cc = int(cols[0])
    expected = np.zeros_like(mask, dtype=bool)
    expected[rr, :] = True
    expected[:, cc] = True
    assert np.array_equal(mask, expected), "interior grey pixels must form a clean cross"
    return rr, cc


def _draw_cross(grid: np.ndarray, box: tuple[int, int, int, int], rel_center: tuple[int, int]) -> None:
    r0, c0, r1, c1 = box
    rr, cc = rel_center
    ir = r0 + 1 + int(rr)
    ic = c0 + 1 + int(cc)
    grid[ir, c0 + 1 : c1] = GREY
    grid[r0 + 1 : r1, ic] = GREY


def _target_center(box: tuple[int, int, int, int], corner_code: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    dr, dc = CORNER_TO_DELTA[int(corner_code)]
    return (0 if dr == 0 else ih - 1, 0 if dc == 0 else iw - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the palette row to decide which corner each framed cross must move to.

    Input structure:
    - The background is black.
    - The bottom two rows form a palette made of 3-4 separate 2x2 swatches; each swatch has three frame-color cells and one grey cell.
    - Above the palette are 3-4 non-touching one-pixel-thick hollow rectangular frames in those same colors.
    - Each frame contains one grey cross occupying exactly one full interior row and one full interior column.

    Rule:
    - Read each palette swatch: the shared non-grey color identifies a frame color, and the grey cell position inside the 2x2 swatch encodes one target corner.
    - For every frame, erase its existing interior grey cross.
    - Redraw the cross so its center lands on the corresponding interior corner pixel of that frame while keeping the frame border and the palette unchanged.

    Key insight:
    - The palette does not map color to another color; it maps frame color to a spatial corner, so the solution must combine palette decoding with frame geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert _get_bg(g) == BG, "expected black background"

    mapping = _decode_palette(g)
    frames = _find_frames(g, set(mapping))
    assert set(frames) == set(mapping), "palette colors and frame colors must match exactly"

    out = g.copy()
    for color, box in frames.items():
        _read_cross_center(g, box)
        r0, c0, r1, c1 = box
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == GREY] = BG
        _draw_cross(out, box, _target_center(box, mapping[color]))
    return out


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _slot_boxes(main_h: int, W: int) -> list[tuple[int, int, int, int]]:
    row_gap = 2
    col_gap = 2
    split_r = (main_h - row_gap) // 2
    split_c = (W - col_gap) // 2
    top = (1, 1, split_r - 1, split_c - 1)
    tr = (1, split_c + col_gap, split_r - 1, W - 2)
    bl = (split_r + row_gap, 1, main_h - 2, split_c - 1)
    br = (split_r + row_gap, split_c + col_gap, main_h - 2, W - 2)
    slots = [top, tr, bl, br]
    for r0, c0, r1, c1 in slots:
        assert r1 - r0 + 1 >= 7 and c1 - c0 + 1 >= 7
    return slots


def _sample_frame_in_slot(rng: random.Random, slot: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    sr0, sc0, sr1, sc1 = slot
    max_h = min(10, sr1 - sr0 + 1)
    max_w = min(10, sc1 - sc0 + 1)
    h = rng.randint(7, max_h)
    w = rng.randint(7, max_w)
    r0 = rng.randint(sr0, sr1 - h + 1)
    c0 = rng.randint(sc0, sc1 - w + 1)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _palette_positions(W: int, n: int, rng: random.Random) -> list[int]:
    gaps = [rng.randint(1, 3) for _ in range(n - 1)]
    total = 2 * n + sum(gaps)
    while total > W:
        idx = max(range(len(gaps)), key=lambda i: gaps[i])
        gaps[idx] -= 1
        total = 2 * n + sum(gaps)
    start = rng.randint(0, W - total)
    cols = []
    c = start
    for i in range(n):
        cols.append(c)
        c += 2
        if i < n - 1:
            c += gaps[i]
    return cols


def _render_case(
    H: int,
    W: int,
    sep_gap: int,
    records: list[dict],
    mapping: dict[int, int],
    palette_order: list[int],
    palette_cols: list[int],
    *,
    use_targets: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for rec in records:
        _draw_frame(grid, rec["box"], rec["color"])
        center = rec["target"] if use_targets else rec["input_center"]
        _draw_cross(grid, rec["box"], center)
    pr0 = H - 2
    for color, c0 in zip(palette_order, palette_cols):
        grid[pr0 : pr0 + 2, c0 : c0 + 2] = color
        code = mapping[color]
        dr, dc = CORNER_TO_DELTA[code]
        grid[pr0 + dr, c0 + dc] = GREY
    if sep_gap:
        assert not grid[H - 3, :].any()
    return grid


def _validate_generated_case(inp: np.ndarray, out: np.ndarray, records: list[dict], mapping: dict[int, int]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False
    if np.array_equal(inp, out):
        return False
    if len(records) != len(mapping):
        return False
    if len({rec["target_code"] for rec in records}) < min(3, len(records)):
        return False
    for rec in records:
        if rec["color"] not in mapping or mapping[rec["color"]] != rec["target_code"]:
            return False
        if rec["input_center"] == rec["target"]:
            return False
        in_r, in_c = rec["input_center"]
        tg_r, tg_c = rec["target"]
        if in_r == tg_r or in_c == tg_c:
            return False
        r0, c0, r1, c1 = rec["box"]
        if not (0 <= r0 < r1 < inp.shape[0] - 2):
            return False
        if not (0 <= c0 < c1 < inp.shape[1]):
            return False
        if _read_cross_center(inp, rec["box"]) != rec["input_center"]:
            return False
        if _read_cross_center(out, rec["box"]) != rec["target"]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    H = rng.randint(21, 25)
    W = rng.randint(21, 25)
    sep_gap = rng.randint(0, 1)
    main_h = H - 2 - sep_gap
    if main_h < 18:
        return None

    slots = _slot_boxes(main_h, W)
    n_frames = rng.randint(3, 4)
    used_slots = rng.sample(range(4), n_frames)
    colors = rng.sample(FRAME_COLORS, n_frames)
    target_codes = rng.sample([0, 1, 2, 3], n_frames)

    records: list[dict] = []
    for slot_idx, color, target_code in zip(used_slots, colors, target_codes):
        box = _sample_frame_in_slot(rng, slots[slot_idx])
        r0, c0, r1, c1 = box
        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        input_center = (rng.randint(1, ih - 2), rng.randint(1, iw - 2))
        target = _target_center(box, target_code)
        records.append(
            {
                "color": color,
                "box": box,
                "input_center": input_center,
                "target": target,
                "target_code": target_code,
            }
        )

    mapping = {rec["color"]: rec["target_code"] for rec in records}
    palette_order = rng.sample(colors, len(colors))
    palette_cols = _palette_positions(W, len(colors), rng)
    inp = _render_case(H, W, sep_gap, records, mapping, palette_order, palette_cols, use_targets=False)
    out = _render_case(H, W, sep_gap, records, mapping, palette_order, palette_cols, use_targets=True)

    if _validate_generated_case(inp, out, records, mapping):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")

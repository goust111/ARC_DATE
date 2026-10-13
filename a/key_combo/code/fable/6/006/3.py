from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_fable_6_006_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w


PALETTE = tuple(range(10))
MAX_GEN_ATTEMPTS = 400


def _body_component(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    large = [comp for comp in comps if int(comp["area"]) > 1]
    assert len(large) == 1, "expected exactly one non-singleton body component"
    return large[0]


def _body_orientation(body: dict) -> str:
    h = int(bbox_h(body["bbox"]))
    w = int(bbox_w(body["bbox"]))
    assert h != w, "body must be a non-square straight bar"
    return "horizontal" if w > h else "vertical"


def _other_singletons(grid: np.ndarray, bg: int, body: dict) -> set[tuple[int, int]]:
    temp = np.array(grid, copy=True)
    coords = body["coords"]
    temp[coords[:, 0], coords[:, 1]] = int(bg)
    singles: set[tuple[int, int]] = set()
    for comp in find_components(temp, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        assert int(comp["area"]) == 1, "all non-body foreground objects must be isolated pixels"
        r, c = map(int, comp["coords"][0])
        singles.add((r, c))
    return singles


def _notches(grid: np.ndarray, bg: int, body: dict) -> list[dict]:
    r0, c0, r1, c1 = map(int, body["bbox"])
    body_color = int(body["color"])
    out: list[dict] = []
    orientation = _body_orientation(body)

    if orientation == "horizontal":
        for c in range(c0 + 1, c1):
            if int(grid[r0, c]) == int(bg) and int(grid[r0 + 1, c]) == body_color:
                out.append(
                    {
                        "side": "top",
                        "notch": (r0, c),
                        "ray_step": (-1, 0),
                        "scan_start": (r0 - 1, c),
                        "landing": (r1 + 1, c),
                    }
                )
            if int(grid[r1, c]) == int(bg) and int(grid[r1 - 1, c]) == body_color:
                out.append(
                    {
                        "side": "bottom",
                        "notch": (r1, c),
                        "ray_step": (1, 0),
                        "scan_start": (r1 + 1, c),
                        "landing": (r0 - 1, c),
                    }
                )
    else:
        for r in range(r0 + 1, r1):
            if int(grid[r, c0]) == int(bg) and int(grid[r, c0 + 1]) == body_color:
                out.append(
                    {
                        "side": "left",
                        "notch": (r, c0),
                        "ray_step": (0, -1),
                        "scan_start": (r, c0 - 1),
                        "landing": (r, c1 + 1),
                    }
                )
            if int(grid[r, c1]) == int(bg) and int(grid[r, c1 - 1]) == body_color:
                out.append(
                    {
                        "side": "right",
                        "notch": (r, c1),
                        "ray_step": (0, 1),
                        "scan_start": (r, c1 + 1),
                        "landing": (r, c0 - 1),
                    }
                )

    assert out, "expected at least one notch"
    return out


def _scan_capture(
    grid: np.ndarray,
    singletons: set[tuple[int, int]],
    start: tuple[int, int],
    step: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = map(int, start)
    dr, dc = map(int, step)
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in singletons:
            return (r, c)
        r += dr
        c += dc
    return None


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    body = _body_component(g, bg)
    body_color = int(body["color"])
    singles = _other_singletons(g, bg, body)
    out = g.copy()

    for notch in _notches(g, bg, body):
        capture = _scan_capture(g, singles, notch["scan_start"], notch["ray_step"])
        if capture is None:
            continue
        rr, cc = capture
        lr, lc = notch["landing"]
        nr, nc = notch["notch"]
        color = int(g[rr, cc])
        out[rr, cc] = int(bg)
        out[nr, nc] = body_color
        out[lr, lc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each body-edge notch with the body color and push the nearest aligned outside pixel through the bar to the opposite side.

    Input structure:
    - The background is the dominant color.
    - One large monochrome straight bar sits near the center; its boundary contains several one-pixel-wide, one-pixel-deep notches.
    - Outside the bar are isolated colored single pixels, including aligned targets and distractors.

    Rule:
    - Determine the bar's long-axis orientation from its bounding box, so only the two long-facing boundary sides can host notches.
    - For each notch, look outward along the notch opening row or column and capture the nearest isolated single pixel on that ray.
    - Erase the captured pixel, fill the notch with the bar color, and place one pixel of the captured color one cell outside the opposite body side on the same row or column.
    - Pixels not lying on any notch ray, and farther pixels hidden behind a nearer captured one on the same ray, remain unchanged.

    Key insight:
    - The colored pixel does not recolor the notch; it passes through the whole body, so the notch heals in body color while the payload reappears as an opposite-side tab on the same line.
    """
    return _transform(grid)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _outside_body_halo(pos: tuple[int, int], body_bbox: tuple[int, int, int, int]) -> bool:
    r, c = map(int, pos)
    r0, c0, r1, c1 = map(int, body_bbox)
    return not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)


def _pick_spaced(values: list[int], count: int, gap: int) -> list[int] | None:
    values = list(values)
    random.shuffle(values)
    picked: list[int] = []
    for v in values:
        if all(abs(int(v) - int(u)) >= int(gap) for u in picked):
            picked.append(int(v))
            if len(picked) == int(count):
                return sorted(picked)
    return None


def _place_single(
    grid: np.ndarray,
    pos: tuple[int, int],
    color: int,
    occupied: set[tuple[int, int]],
    body_bbox: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    r, c = map(int, pos)
    if not (0 <= r < h and 0 <= c < w):
        return False
    if pos in forbidden:
        return False
    if not _outside_body_halo(pos, body_bbox):
        return False
    for other in occupied:
        if _chebyshev(pos, other) <= 1:
            return False
    grid[r, c] = int(color)
    occupied.add(pos)
    return True


def _color_choices(bg: int, body_color: int) -> list[int]:
    return [int(c) for c in PALETTE if int(c) not in {int(bg), int(body_color)}]


def _random_colors(bg: int, body_color: int, count: int) -> list[int]:
    choices = _color_choices(bg, body_color)
    return [int(random.choice(choices)) for _ in range(int(count))]


def _make_horizontal_scene() -> tuple[np.ndarray, dict] | None:
    thickness = random.randint(3, 5)
    length = random.randint(10, 16)
    top_space = random.randint(6, 9)
    bottom_space = random.randint(6, 9)
    left_space = random.randint(3, 6)
    right_space = random.randint(3, 6)
    h = top_space + thickness + bottom_space
    w = left_space + length + right_space
    if max(h, w) > 30:
        return None

    bg, body_color = random.sample(PALETTE, 2)
    grid = np.full((h, w), int(bg), dtype=int)
    r0 = top_space
    c0 = left_space
    r1 = r0 + thickness - 1
    c1 = c0 + length - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(body_color)
    body_bbox = (r0, c0, r1, c1)

    total_notches = random.randint(3, 5)
    cols = _pick_spaced(list(range(c0 + 1, c1)), total_notches, 2)
    if cols is None:
        return None
    random.shuffle(cols)
    top_count = random.randint(1, total_notches - 1)
    top_cols = sorted(cols[:top_count])
    bottom_cols = sorted(cols[top_count:])
    if not bottom_cols:
        return None

    for c in top_cols:
        grid[r0, c] = int(bg)
    for c in bottom_cols:
        grid[r1, c] = int(bg)

    occupied: set[tuple[int, int]] = set()
    forbidden = {(r1 + 1, c) for c in top_cols} | {(r0 - 1, c) for c in bottom_cols}
    records: list[dict] = []

    far_side = random.choice(["top", "bottom"])
    side_cols = {"top": top_cols, "bottom": bottom_cols}
    if not side_cols[far_side]:
        far_side = "bottom" if far_side == "top" else "top"
    if not side_cols[far_side]:
        return None
    far_col = random.choice(side_cols[far_side])

    for side, cols_on_side in (("top", top_cols), ("bottom", bottom_cols)):
        colors = _random_colors(bg, body_color, len(cols_on_side))
        for c, color in zip(cols_on_side, colors):
            if side == "top":
                near_candidates = list(range(r0 - 2, -1, -1))
                if c == far_col and side == far_side:
                    near_candidates = [r for r in near_candidates if r >= 2]
                random.shuffle(near_candidates)
                capture = None
                for rr in near_candidates:
                    if _place_single(grid, (rr, c), color, occupied, body_bbox, forbidden):
                        capture = (rr, c)
                        break
                if capture is None:
                    return None
                farther = None
                if c == far_col and side == far_side:
                    far_candidates = list(range(capture[0] - 2, -1, -1))
                    random.shuffle(far_candidates)
                    for rr in far_candidates:
                        far_color = random.choice(_color_choices(bg, body_color))
                        if _place_single(grid, (rr, c), far_color, occupied, body_bbox, forbidden):
                            farther = (rr, c, int(far_color))
                            break
                    if farther is None:
                        return None
                records.append(
                    {
                        "side": side,
                        "notch": (r0, c),
                        "capture": capture,
                        "landing": (r1 + 1, c),
                        "color": int(color),
                        "farther": farther,
                    }
                )
            else:
                near_candidates = list(range(r1 + 2, h))
                if c == far_col and side == far_side:
                    near_candidates = [r for r in near_candidates if r <= h - 3]
                random.shuffle(near_candidates)
                capture = None
                for rr in near_candidates:
                    if _place_single(grid, (rr, c), color, occupied, body_bbox, forbidden):
                        capture = (rr, c)
                        break
                if capture is None:
                    return None
                farther = None
                if c == far_col and side == far_side:
                    far_candidates = list(range(capture[0] + 2, h))
                    random.shuffle(far_candidates)
                    for rr in far_candidates:
                        far_color = random.choice(_color_choices(bg, body_color))
                        if _place_single(grid, (rr, c), far_color, occupied, body_bbox, forbidden):
                            farther = (rr, c, int(far_color))
                            break
                    if farther is None:
                        return None
                records.append(
                    {
                        "side": side,
                        "notch": (r1, c),
                        "capture": capture,
                        "landing": (r0 - 1, c),
                        "color": int(color),
                        "farther": farther,
                    }
                )

    no_notch_cols = [c for c in range(c0 + 1, c1) if c not in set(top_cols) | set(bottom_cols)]
    if not no_notch_cols:
        return None
    aligned_col = random.choice(no_notch_cols)
    aligned_side = random.choice(["top", "bottom"])
    aligned_color = random.choice(_color_choices(bg, body_color))
    if aligned_side == "top":
        aligned_rows = list(range(r0 - 2, -1, -1))
    else:
        aligned_rows = list(range(r1 + 2, h))
    random.shuffle(aligned_rows)
    aligned_pos = None
    for rr in aligned_rows:
        if _place_single(grid, (rr, aligned_col), aligned_color, occupied, body_bbox, forbidden):
            aligned_pos = (rr, aligned_col, int(aligned_color))
            break
    if aligned_pos is None:
        return None

    off_axis_color = random.choice(_color_choices(bg, body_color))
    off_positions = []
    for r in range(h):
        for c in list(range(0, c0 - 1)) + list(range(c1 + 2, w)):
            if c in top_cols or c in bottom_cols:
                continue
            off_positions.append((r, c))
    random.shuffle(off_positions)
    off_axis = None
    for pos in off_positions:
        if _place_single(grid, pos, off_axis_color, occupied, body_bbox, forbidden):
            off_axis = (pos[0], pos[1], int(off_axis_color))
            break
    if off_axis is None:
        return None

    return (
        grid,
        {
            "bg": int(bg),
            "body_color": int(body_color),
            "body_bbox": body_bbox,
            "orientation": "horizontal",
            "captures": records,
            "aligned_no_notch": aligned_pos,
            "off_axis": off_axis,
        },
    )


def _make_vertical_scene() -> tuple[np.ndarray, dict] | None:
    thickness = random.randint(3, 5)
    length = random.randint(10, 16)
    top_space = random.randint(3, 6)
    bottom_space = random.randint(3, 6)
    left_space = random.randint(6, 9)
    right_space = random.randint(6, 9)
    h = top_space + length + bottom_space
    w = left_space + thickness + right_space
    if max(h, w) > 30:
        return None

    bg, body_color = random.sample(PALETTE, 2)
    grid = np.full((h, w), int(bg), dtype=int)
    r0 = top_space
    c0 = left_space
    r1 = r0 + length - 1
    c1 = c0 + thickness - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(body_color)
    body_bbox = (r0, c0, r1, c1)

    total_notches = random.randint(3, 5)
    rows = _pick_spaced(list(range(r0 + 1, r1)), total_notches, 2)
    if rows is None:
        return None
    random.shuffle(rows)
    left_count = random.randint(1, total_notches - 1)
    left_rows = sorted(rows[:left_count])
    right_rows = sorted(rows[left_count:])
    if not right_rows:
        return None

    for r in left_rows:
        grid[r, c0] = int(bg)
    for r in right_rows:
        grid[r, c1] = int(bg)

    occupied: set[tuple[int, int]] = set()
    forbidden = {(r, c1 + 1) for r in left_rows} | {(r, c0 - 1) for r in right_rows}
    records: list[dict] = []

    far_side = random.choice(["left", "right"])
    side_rows = {"left": left_rows, "right": right_rows}
    if not side_rows[far_side]:
        far_side = "right" if far_side == "left" else "left"
    if not side_rows[far_side]:
        return None
    far_row = random.choice(side_rows[far_side])

    for side, rows_on_side in (("left", left_rows), ("right", right_rows)):
        colors = _random_colors(bg, body_color, len(rows_on_side))
        for r, color in zip(rows_on_side, colors):
            if side == "left":
                near_candidates = list(range(c0 - 2, -1, -1))
                if r == far_row and side == far_side:
                    near_candidates = [c for c in near_candidates if c >= 2]
                random.shuffle(near_candidates)
                capture = None
                for cc in near_candidates:
                    if _place_single(grid, (r, cc), color, occupied, body_bbox, forbidden):
                        capture = (r, cc)
                        break
                if capture is None:
                    return None
                farther = None
                if r == far_row and side == far_side:
                    far_candidates = list(range(capture[1] - 2, -1, -1))
                    random.shuffle(far_candidates)
                    for cc in far_candidates:
                        far_color = random.choice(_color_choices(bg, body_color))
                        if _place_single(grid, (r, cc), far_color, occupied, body_bbox, forbidden):
                            farther = (r, cc, int(far_color))
                            break
                    if farther is None:
                        return None
                records.append(
                    {
                        "side": side,
                        "notch": (r, c0),
                        "capture": capture,
                        "landing": (r, c1 + 1),
                        "color": int(color),
                        "farther": farther,
                    }
                )
            else:
                near_candidates = list(range(c1 + 2, w))
                if r == far_row and side == far_side:
                    near_candidates = [c for c in near_candidates if c <= w - 3]
                random.shuffle(near_candidates)
                capture = None
                for cc in near_candidates:
                    if _place_single(grid, (r, cc), color, occupied, body_bbox, forbidden):
                        capture = (r, cc)
                        break
                if capture is None:
                    return None
                farther = None
                if r == far_row and side == far_side:
                    far_candidates = list(range(capture[1] + 2, w))
                    random.shuffle(far_candidates)
                    for cc in far_candidates:
                        far_color = random.choice(_color_choices(bg, body_color))
                        if _place_single(grid, (r, cc), far_color, occupied, body_bbox, forbidden):
                            farther = (r, cc, int(far_color))
                            break
                    if farther is None:
                        return None
                records.append(
                    {
                        "side": side,
                        "notch": (r, c1),
                        "capture": capture,
                        "landing": (r, c0 - 1),
                        "color": int(color),
                        "farther": farther,
                    }
                )

    no_notch_rows = [r for r in range(r0 + 1, r1) if r not in set(left_rows) | set(right_rows)]
    if not no_notch_rows:
        return None
    aligned_row = random.choice(no_notch_rows)
    aligned_side = random.choice(["left", "right"])
    aligned_color = random.choice(_color_choices(bg, body_color))
    if aligned_side == "left":
        aligned_cols = list(range(c0 - 2, -1, -1))
    else:
        aligned_cols = list(range(c1 + 2, w))
    random.shuffle(aligned_cols)
    aligned_pos = None
    for cc in aligned_cols:
        if _place_single(grid, (aligned_row, cc), aligned_color, occupied, body_bbox, forbidden):
            aligned_pos = (aligned_row, cc, int(aligned_color))
            break
    if aligned_pos is None:
        return None

    off_axis_color = random.choice(_color_choices(bg, body_color))
    off_positions = []
    for r in list(range(0, r0 - 1)) + list(range(r1 + 2, h)):
        for c in range(w):
            if r in left_rows or r in right_rows:
                continue
            off_positions.append((r, c))
    random.shuffle(off_positions)
    off_axis = None
    for pos in off_positions:
        if _place_single(grid, pos, off_axis_color, occupied, body_bbox, forbidden):
            off_axis = (pos[0], pos[1], int(off_axis_color))
            break
    if off_axis is None:
        return None

    return (
        grid,
        {
            "bg": int(bg),
            "body_color": int(body_color),
            "body_bbox": body_bbox,
            "orientation": "vertical",
            "captures": records,
            "aligned_no_notch": aligned_pos,
            "off_axis": off_axis,
        },
    )


def _expected_from_spec(inp: np.ndarray, spec: dict) -> np.ndarray:
    out = np.array(inp, copy=True)
    bg = int(spec["bg"])
    body_color = int(spec["body_color"])
    for rec in spec["captures"]:
        rr, cc = map(int, rec["capture"])
        nr, nc = map(int, rec["notch"])
        lr, lc = map(int, rec["landing"])
        out[rr, cc] = int(bg)
        out[nr, nc] = int(body_color)
        out[lr, lc] = int(rec["color"])
    return out


def _validate_scene(inp: np.ndarray, spec: dict) -> bool:
    out = main(inp)
    expected = _expected_from_spec(inp, spec)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(inp, out):
        return False

    bg = int(spec["bg"])
    body_color = int(spec["body_color"])
    for rec in spec["captures"]:
        rr, cc = map(int, rec["capture"])
        nr, nc = map(int, rec["notch"])
        lr, lc = map(int, rec["landing"])
        if int(inp[rr, cc]) != int(rec["color"]) or int(out[rr, cc]) != bg:
            return False
        if int(inp[nr, nc]) != bg or int(out[nr, nc]) != body_color:
            return False
        if int(out[lr, lc]) != int(rec["color"]):
            return False
        farther = rec["farther"]
        if farther is not None:
            fr, fc, fcol = farther
            if int(inp[fr, fc]) != int(fcol) or int(out[fr, fc]) != int(fcol):
                return False

    ar, ac, acol = spec["aligned_no_notch"]
    if int(inp[ar, ac]) != int(acol) or int(out[ar, ac]) != int(acol):
        return False

    dr, dc, dcol = spec["off_axis"]
    if int(inp[dr, dc]) != int(dcol) or int(out[dr, dc]) != int(dcol):
        return False

    body = _body_component(inp, bg)
    if _body_orientation(body) != spec["orientation"]:
        return False
    if len(spec["captures"]) < 3:
        return False
    if not any(rec["farther"] is not None for rec in spec["captures"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        builder = random.choice([_make_horizontal_scene, _make_vertical_scene])
        built = builder()
        if built is None:
            continue
        inp, spec = built
        if _validate_scene(inp, spec):
            return inp
    raise RuntimeError("failed to generate a valid notch-throughput sample")

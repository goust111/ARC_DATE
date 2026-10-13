from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3, 4)
MARKER = 5
MARKER_TO_SHIFT = {
    "tl": (1, 1),
    "tr": (1, -1),
    "bl": (-1, 1),
    "br": (-1, -1),
}
SHIFT_TO_MARKER = {v: k for k, v in MARKER_TO_SHIFT.items()}
MARKER_COORD = {
    "tl": lambda h, w: (0, 0),
    "tr": lambda h, w: (0, w - 1),
    "bl": lambda h, w: (h - 1, 0),
    "br": lambda h, w: (h - 1, w - 1),
}
SHAPES_BY_AREA: dict[int, list[tuple[int, int]]] = {}
for _h in range(2, 6):
    for _w in range(2, 6):
        SHAPES_BY_AREA.setdefault(_h * _w, []).append((_h, _w))


def _rect_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    r0 = max(ar, br)
    c0 = max(ac, bc)
    r1 = min(ar + ah, br + bh)
    c1 = min(ac + aw, bc + bw)
    if r0 >= r1 or c0 >= c1:
        return None
    return (r0, c0, r1 - r0, c1 - c0)


def _rect_area(rect: tuple[int, int, int, int] | None) -> int:
    if rect is None:
        return 0
    return rect[2] * rect[3]


def _frame_outer(frame: dict) -> tuple[int, int, int, int]:
    return (frame["r"] - 1, frame["c"] - 1, frame["h"] + 2, frame["w"] + 2)


def _blocks(frame: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
    sr, sc = frame["shift"]
    base = (frame["r"], frame["c"], frame["h"], frame["w"])
    moved = (frame["r"] + sr * frame["h"], frame["c"] + sc * frame["w"], frame["h"], frame["w"])
    return [("base", base), ("shifted", moved)]


def _boxes_with_gap(outer_a: tuple[int, int, int, int], outer_b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar, ac, ah, aw = outer_a
    br, bc, bh, bw = outer_b
    a_r0, a_r1 = ar, ar + ah - 1
    a_c0, a_c1 = ac, ac + aw - 1
    b_r0, b_r1 = br, br + bh - 1
    b_c0, b_c1 = bc, bc + bw - 1
    return a_r1 + gap < b_r0 or b_r1 + gap < a_r0 or a_c1 + gap < b_c0 or b_c1 + gap < a_c0


def _sorted_frames(frames: list[dict], descending: bool = True) -> list[dict]:
    keyed = sorted(frames, key=lambda f: (f["area"], f["r"], f["c"], f["h"], f["w"]))
    return list(reversed(keyed)) if descending else keyed


def _compose(
    frames: list[dict],
    shape: tuple[int, int],
    *,
    descending: bool = True,
    include_base: bool = True,
    include_shifted: bool = True,
) -> tuple[np.ndarray, dict[int, dict[str, int]]]:
    out = np.zeros(shape, dtype=int)
    stats = {
        i: {"base": 0, "shifted": 0, "total": 0}
        for i in range(len(frames))
    }
    order = _sorted_frames(frames, descending=descending)
    index_of = {id(frame): idx for idx, frame in enumerate(frames)}
    for frame in order:
        idx = index_of[id(frame)]
        for kind, rect in _blocks(frame):
            if (kind == "base" and not include_base) or (kind == "shifted" and not include_shifted):
                continue
            r, c, h, w = rect
            view = out[r : r + h, c : c + w]
            mask = view == BG
            gained = int(mask.sum())
            if gained:
                view[mask] = frame["color"]
            stats[idx][kind] += gained
            stats[idx]["total"] += gained
    return out, stats


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {0, 1, 2, 3, 4, 5}, "unexpected colors present"

    frames: list[dict] = []
    for color in FRAME_COLORS:
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        oh, ow = patch.shape
        assert oh >= 4 and ow >= 4, "frame must have at least a 2x2 interior"

        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == color), "frame border must be monochromatic"
        assert not np.any(patch[~border] == color), "frame color may only appear on the border"

        inner = patch[1:-1, 1:-1]
        marker = np.argwhere(inner == MARKER)
        assert marker.shape == (1, 2), "expected exactly one grey marker per frame"
        mr, mc = map(int, marker[0])
        h, w = inner.shape
        corner_to_shift = {
            (0, 0): (1, 1),
            (0, w - 1): (1, -1),
            (h - 1, 0): (-1, 1),
            (h - 1, w - 1): (-1, -1),
        }
        assert (mr, mc) in corner_to_shift, "marker must sit on an interior corner"
        rest = inner.copy()
        rest[mr, mc] = BG
        assert np.all(rest == BG), "frame interior must otherwise be black"

        frames.append(
            {
                "color": color,
                "r": r0 + 1,
                "c": c0 + 1,
                "h": h,
                "w": w,
                "area": h * w,
                "marker": (mr, mc),
                "shift": corner_to_shift[(mr, mc)],
            }
        )

    assert 2 <= len(frames) <= 4, "expected 2 to 4 hollow frames"
    areas = [f["area"] for f in frames]
    assert len(set(areas)) == len(areas), "frame areas must be distinct to avoid ambiguous priority ties"
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            assert _boxes_with_gap(_frame_outer(a), _frame_outer(b), gap=2), "frames must be separated in input"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each hollow frame, fill its interior twice, and resolve diagonal-copy conflicts by larger-area priority.

    Input structure:
    - The grid has a black background with 2 to 4 single-pixel-thick hollow rectangular frames.
    - Each frame uses a distinct color from blue/red/green/yellow and contains one grey marker on an interior corner.
    - Apart from the frame borders and the grey markers, the frame interiors are black and the frames are mutually separated.

    Rule:
    - For each frame, identify its strict interior rectangle and the marker corner.
    - Erase the whole input drawing to black, then create one filled block in the original interior position.
    - Create a second filled block by translating that interior by its own height and width toward the corner opposite the marker.
    - Process frames from larger interior area to smaller; later frames may only paint still-black cells, so larger frames keep contested pixels.

    Key insight:
    - The frame determines the rectangular payload, the marker determines which diagonal copy is made, and overlaps are settled only by largest-first non-overwriting composition.
    """
    frames = _extract_frames(np.asarray(grid, dtype=int))
    out, _stats = _compose(frames, np.asarray(grid).shape, descending=True)
    return out


def _candidate_positions(
    H: int,
    W: int,
    h: int,
    w: int,
    rng: random.Random,
    *,
    row_pref: str | None = None,
    col_pref: str | None = None,
) -> list[tuple[int, int]]:
    r_lo, r_hi = h, H - 2 * h
    c_lo, c_hi = w, W - 2 * w
    assert r_lo <= r_hi and c_lo <= c_hi

    rows = list(range(r_lo, r_hi + 1))
    cols = list(range(c_lo, c_hi + 1))
    mid_r = (r_lo + r_hi) // 2
    mid_c = (c_lo + c_hi) // 2

    if row_pref == "upper":
        pref = [r for r in rows if r <= mid_r]
        rows = pref or rows
    elif row_pref == "lower":
        pref = [r for r in rows if r >= mid_r]
        rows = pref or rows

    if col_pref == "left":
        pref = [c for c in cols if c <= mid_c]
        cols = pref or cols
    elif col_pref == "right":
        pref = [c for c in cols if c >= mid_c]
        cols = pref or cols

    pts = [(r, c) for r in rows for c in cols]
    rng.shuffle(pts)
    return pts


def _make_frame(color: int, h: int, w: int, r: int, c: int, shift: tuple[int, int]) -> dict:
    marker_name = SHIFT_TO_MARKER[shift]
    mr, mc = MARKER_COORD[marker_name](h, w)
    return {
        "color": color,
        "h": h,
        "w": w,
        "r": r,
        "c": c,
        "area": h * w,
        "marker": (mr, mc),
        "shift": shift,
    }


def _render_input(H: int, W: int, frames: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for frame in frames:
        r0, c0 = frame["r"] - 1, frame["c"] - 1
        r1, c1 = frame["r"] + frame["h"], frame["c"] + frame["w"]
        color = frame["color"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        mr, mc = frame["marker"]
        grid[frame["r"] + mr, frame["c"] + mc] = MARKER
    return grid


def _force_conflict_pair(
    H: int,
    W: int,
    specs: list[tuple[int, int, int]],
    rng: random.Random,
) -> list[dict] | None:
    family = rng.choice(("backslash", "slash"))
    if family == "backslash":
        shifts = [(1, 1), (-1, -1)]
        prefs = [("upper", "left"), ("lower", "right")]
    else:
        shifts = [(1, -1), (-1, 1)]
        prefs = [("upper", "right"), ("lower", "left")]

    first_area, first_h, first_w = specs[0]
    second_area, second_h, second_w = specs[1]
    first_candidates = _candidate_positions(H, W, first_h, first_w, rng, row_pref=prefs[0][0], col_pref=prefs[0][1])
    second_candidates = _candidate_positions(H, W, second_h, second_w, rng, row_pref=prefs[1][0], col_pref=prefs[1][1])

    for r0, c0 in first_candidates[:80]:
        frame0 = _make_frame(-1, first_h, first_w, r0, c0, shifts[0])
        for r1, c1 in second_candidates[:80]:
            frame1 = _make_frame(-1, second_h, second_w, r1, c1, shifts[1])
            if not _boxes_with_gap(_frame_outer(frame0), _frame_outer(frame1), gap=2):
                continue
            if _rect_area(_rect_intersection(_blocks(frame0)[1][1], _blocks(frame1)[1][1])) == 0:
                continue
            return [frame0, frame1]
    return None


def _try_place_extra(
    H: int,
    W: int,
    h: int,
    w: int,
    used: list[dict],
    rng: random.Random,
) -> dict | None:
    shift = rng.choice(list(SHIFT_TO_MARKER))
    row_pref = rng.choice((None, "upper", "lower"))
    col_pref = rng.choice((None, "left", "right"))
    for r, c in _candidate_positions(H, W, h, w, rng, row_pref=row_pref, col_pref=col_pref):
        frame = _make_frame(-1, h, w, r, c, shift)
        if all(_boxes_with_gap(_frame_outer(frame), _frame_outer(other), gap=2) for other in used):
            return frame
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False

    overlaps = []
    for i, a in enumerate(frames):
        for j, b in enumerate(frames[i + 1 :], start=i + 1):
            for _kind_a, rect_a in _blocks(a):
                for _kind_b, rect_b in _blocks(b):
                    inter = _rect_intersection(rect_a, rect_b)
                    if inter is not None:
                        overlaps.append((i, j, inter))
    if not overlaps:
        return False

    distinct_dirs = {frame["shift"] for frame in frames}
    if len(distinct_dirs) < 2:
        return False

    composed, stats = _compose(frames, grid.shape, descending=True)
    if not np.array_equal(out, composed):
        return False

    if any(stats[i]["total"] == 0 for i in range(len(frames))):
        return False
    if sum(stats[i]["shifted"] > 0 for i in range(len(frames))) < 2:
        return False

    conflict_cells = sum(_rect_area(inter) for _, _, inter in overlaps)
    colored_cells = int(np.count_nonzero(out))
    if conflict_cells <= 0 or conflict_cells * 3 > colored_cells:
        return False

    smaller_first, _ = _compose(frames, grid.shape, descending=False)
    if np.array_equal(out, smaller_first):
        return False

    base_only, _ = _compose(frames, grid.shape, descending=True, include_shifted=False)
    shifted_only, _ = _compose(frames, grid.shape, descending=True, include_base=False)
    if np.array_equal(out, base_only) or np.array_equal(out, shifted_only):
        return False

    marker_sensitive = False
    for i, frame in enumerate(frames):
        for alt_shift in SHIFT_TO_MARKER:
            if alt_shift == frame["shift"]:
                continue
            altered = [dict(f) for f in frames]
            altered[i]["shift"] = alt_shift
            altered[i]["marker"] = MARKER_COORD[SHIFT_TO_MARKER[alt_shift]](frame["h"], frame["w"])
            alt_out, _ = _compose(altered, grid.shape, descending=True)
            if not np.array_equal(out, alt_out):
                marker_sensitive = True
                break
        if marker_sensitive:
            break
    return marker_sensitive


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(18, 22)
    W = rng.randint(18, 22)
    count = rng.randint(2, 4)

    areas = rng.sample(sorted(SHAPES_BY_AREA), count)
    areas.sort(reverse=True)
    specs = [(area, *rng.choice(SHAPES_BY_AREA[area])) for area in areas]

    frames = _force_conflict_pair(H, W, specs, rng)
    if frames is None:
        return None

    for area, h, w in specs[2:]:
        extra = _try_place_extra(H, W, h, w, frames, rng)
        if extra is None:
            return None
        frames.append(extra)

    colors = rng.sample(list(FRAME_COLORS), count)
    for frame, color in zip(frames, colors):
        frame["color"] = color

    rng.shuffle(frames)
    grid = _render_input(H, W, frames)
    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")

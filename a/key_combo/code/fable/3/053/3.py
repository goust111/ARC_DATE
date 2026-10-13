from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SIDE_TOP = "top"
SIDE_BOTTOM = "bottom"
SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDES = (SIDE_TOP, SIDE_BOTTOM, SIDE_LEFT, SIDE_RIGHT)

GEN_RETRIES = 400
PALETTE = list(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(bg)


def _is_hollow_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h, w = _bbox_hw(comp["bbox"])
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(comp["color"])
    border = np.ones((h, w), dtype=bool)
    border[1:-1, 1:-1] = False
    if not np.all(patch[border] == color):
        return False
    return bool(np.all(patch[1:-1, 1:-1] == bg))


def _is_solid_rectangle(comp: dict) -> bool:
    h, w = _bbox_hw(comp["bbox"])
    return int(comp["area"]) == h * w


def _classify_side(rect_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    rr0, rc0, rr1, rc1 = rect_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    hits = []
    if rr1 < fr0:
        hits.append(SIDE_TOP)
    if rr0 > fr1:
        hits.append(SIDE_BOTTOM)
    if rc1 < fc0:
        hits.append(SIDE_LEFT)
    if rc0 > fc1:
        hits.append(SIDE_RIGHT)
    assert len(hits) == 1, "each solid rectangle must lie on exactly one side of the frame"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide all aligned solid rectangles toward the hollow frame and let only the exact fit enter its cavity.

    Input structure:
    - A dominant background canvas contains one hollow axis-aligned rectangular frame of thickness 1.
    - Several other foreground objects are solid monochrome rectangles placed wholly outside the frame.
    - Exactly one solid rectangle has the same height and width as the frame's inner cavity.

    Rule:
    - Identify the hollow frame and every solid rectangle outside it.
    - Move each solid rectangle straight toward the frame from its side while keeping its orthogonal coordinate fixed.
    - The unique rectangle whose size exactly matches the cavity fills the cavity; every other rectangle stops in the adjacent exterior position against the frame wall.

    Key insight:
    - Entry is decided by exact size agreement with the cavity, not by which rectangle started closest to the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _find_components(g, bg)

    frames = [comp for comp in comps if _is_hollow_frame(g, comp, bg)]
    assert len(frames) == 1, "expected exactly one hollow frame"
    frame = frames[0]
    frame_bbox = frame["bbox"]
    frame_color = int(frame["color"])
    fr0, fc0, fr1, fc1 = frame_bbox
    cavity_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    cavity_h, cavity_w = _bbox_hw(cavity_bbox)

    rectangles = [comp for comp in comps if comp is not frame]
    assert rectangles, "expected at least one solid rectangle outside the frame"
    assert all(_is_solid_rectangle(comp) for comp in rectangles), "all non-frame objects must be solid rectangles"

    exact = [comp for comp in rectangles if _bbox_hw(comp["bbox"]) == (cavity_h, cavity_w)]
    assert len(exact) == 1, "expected exactly one rectangle to exactly match the cavity"

    out = np.full_like(g, bg)
    _paint_frame(out, frame_bbox, frame_color, bg)

    for comp in rectangles:
        color = int(comp["color"])
        rect_bbox = comp["bbox"]
        rect_h, rect_w = _bbox_hw(rect_bbox)
        side = _classify_side(rect_bbox, frame_bbox)

        if (rect_h, rect_w) == (cavity_h, cavity_w):
            dest = cavity_bbox
        elif side == SIDE_TOP:
            dest = (fr0 - rect_h, rect_bbox[1], fr0 - 1, rect_bbox[3])
        elif side == SIDE_BOTTOM:
            dest = (fr1 + 1, rect_bbox[1], fr1 + rect_h, rect_bbox[3])
        elif side == SIDE_LEFT:
            dest = (rect_bbox[0], fc0 - rect_w, rect_bbox[2], fc0 - 1)
        else:
            dest = (rect_bbox[0], fc1 + 1, rect_bbox[2], fc1 + rect_w)

        r0, c0, r1, c1 = dest
        assert 0 <= r0 <= r1 < g.shape[0] and 0 <= c0 <= c1 < g.shape[1]
        _paint_rect(out, dest, color)

    return out


def _transpose_side_choices(cavity_h: int, cavity_w: int) -> tuple[str, ...]:
    return (SIDE_TOP, SIDE_BOTTOM) if cavity_h < cavity_w else (SIDE_LEFT, SIDE_RIGHT)


def _random_extra_shape(side: str, cavity_h: int, cavity_w: int) -> tuple[int, int]:
    max_long = min(7, max(cavity_h, cavity_w) + 1)
    for _ in range(100):
        if side in (SIDE_TOP, SIDE_BOTTOM):
            h = random.randint(2, max_long)
            w = random.randint(2, cavity_w)
        else:
            h = random.randint(2, cavity_h)
            w = random.randint(2, max_long)
        if (h, w) not in {(cavity_h, cavity_w), (cavity_w, cavity_h)}:
            return h, w
    raise RuntimeError("failed to sample extra rectangle dimensions")


def _random_side_offsets(specs: list[dict], cavity_h: int, cavity_w: int) -> None:
    for spec in specs:
        if spec["side"] in (SIDE_TOP, SIDE_BOTTOM):
            spec["offset"] = random.randint(0, cavity_w - spec["w"])
        else:
            spec["offset"] = random.randint(0, cavity_h - spec["h"])


def _assign_unique_gaps(specs: list[dict], fit_is_nearest: bool) -> None:
    exact = next(spec for spec in specs if spec["role"] == "exact")
    nonfits = [spec for spec in specs if spec["role"] != "exact"]
    gap_pool = [2, 3, 4, 5, 6]
    if fit_is_nearest:
        exact["gap"] = 1
        picks = random.sample(gap_pool, len(nonfits))
        for spec, gap in zip(nonfits, picks):
            spec["gap"] = gap
        return

    weighted = [spec for spec in nonfits if spec["role"] == "transpose"]
    weighted += nonfits
    nearest = random.choice(weighted)
    nearest["gap"] = 1
    others = [spec for spec in specs if spec is not nearest]
    picks = random.sample(gap_pool, len(others))
    for spec, gap in zip(others, picks):
        spec["gap"] = gap


def _build_specs(cavity_h: int, cavity_w: int) -> list[dict] | None:
    exact_side = random.choice(SIDES)
    transpose_choices = [side for side in _transpose_side_choices(cavity_h, cavity_w) if side != exact_side]
    if not transpose_choices:
        return None
    transpose_side = random.choice(transpose_choices)

    remaining = [side for side in SIDES if side not in {exact_side, transpose_side}]
    extra_count = random.choice([1, 2])
    extra_count = min(extra_count, len(remaining))
    extra_sides = random.sample(remaining, extra_count)

    colors = random.sample(PALETTE, 2 + extra_count + 2)
    bg = colors[0]
    frame_color = colors[1]
    object_colors = colors[2:]

    specs = [
        {
            "role": "exact",
            "side": exact_side,
            "h": cavity_h,
            "w": cavity_w,
            "color": object_colors[0],
        },
        {
            "role": "transpose",
            "side": transpose_side,
            "h": cavity_w,
            "w": cavity_h,
            "color": object_colors[1],
        },
    ]

    for idx, side in enumerate(extra_sides, start=2):
        h, w = _random_extra_shape(side, cavity_h, cavity_w)
        specs.append({"role": f"extra_{idx}", "side": side, "h": h, "w": w, "color": object_colors[idx]})

    _random_side_offsets(specs, cavity_h, cavity_w)
    _assign_unique_gaps(specs, fit_is_nearest=bool(random.getrandbits(1)))
    return [{"bg": bg, "frame_color": frame_color, **spec} for spec in specs]


def _extent_by_side(specs: list[dict]) -> dict[str, int]:
    extents = {side: 0 for side in SIDES}
    for spec in specs:
        if spec["side"] in (SIDE_TOP, SIDE_BOTTOM):
            extents[spec["side"]] = spec["gap"] + spec["h"]
        else:
            extents[spec["side"]] = spec["gap"] + spec["w"]
    return extents


def _source_bbox(
    spec: dict,
    frame_bbox: tuple[int, int, int, int],
    cavity_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    cr0, cc0, _cr1, _cc1 = cavity_bbox
    h = int(spec["h"])
    w = int(spec["w"])
    gap = int(spec["gap"])
    off = int(spec["offset"])
    side = spec["side"]

    if side == SIDE_TOP:
        r0 = fr0 - gap - h
        c0 = cc0 + off
    elif side == SIDE_BOTTOM:
        r0 = fr1 + gap + 1
        c0 = cc0 + off
    elif side == SIDE_LEFT:
        r0 = cr0 + off
        c0 = fc0 - gap - w
    else:
        r0 = cr0 + off
        c0 = fc1 + gap + 1
    return (r0, c0, r0 + h - 1, c0 + w - 1)


def _dest_bbox(
    spec: dict,
    frame_bbox: tuple[int, int, int, int],
    cavity_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    cr0, cc0, cr1, cc1 = cavity_bbox
    h = int(spec["h"])
    w = int(spec["w"])
    off = int(spec["offset"])
    side = spec["side"]

    if spec["role"] == "exact":
        return cavity_bbox
    if side == SIDE_TOP:
        c0 = cc0 + off
        return (fr0 - h, c0, fr0 - 1, c0 + w - 1)
    if side == SIDE_BOTTOM:
        c0 = cc0 + off
        return (fr1 + 1, c0, fr1 + h, c0 + w - 1)
    if side == SIDE_LEFT:
        r0 = cr0 + off
        return (r0, fc0 - w, r0 + h - 1, fc0 - 1)
    r0 = cr0 + off
    return (r0, fc1 + 1, r0 + h - 1, fc1 + w)


def _build_instance(specs: list[dict], cavity_h: int, cavity_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    bg = int(specs[0]["bg"])
    frame_color = int(specs[0]["frame_color"])
    frame_h = cavity_h + 2
    frame_w = cavity_w + 2
    ext = _extent_by_side(specs)

    min_h = ext[SIDE_TOP] + frame_h + ext[SIDE_BOTTOM] + 2
    min_w = ext[SIDE_LEFT] + frame_w + ext[SIDE_RIGHT] + 2
    if min_h > 30 or min_w > 30:
        return None

    slack_h = 30 - min_h
    slack_w = 30 - min_w
    extra_top = random.randint(0, slack_h)
    extra_bottom = random.randint(0, slack_h - extra_top)
    extra_left = random.randint(0, slack_w)
    extra_right = random.randint(0, slack_w - extra_left)

    height = min_h + extra_top + extra_bottom
    width = min_w + extra_left + extra_right
    fr0 = 1 + extra_top + ext[SIDE_TOP]
    fc0 = 1 + extra_left + ext[SIDE_LEFT]
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)
    cavity_bbox = (fr0 + 1, fc0 + 1, fr0 + cavity_h, fc0 + cavity_w)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    _paint_frame(inp, frame_bbox, frame_color, bg)
    _paint_frame(out, frame_bbox, frame_color, bg)

    for spec in specs:
        src = _source_bbox(spec, frame_bbox, cavity_bbox)
        dst = _dest_bbox(spec, frame_bbox, cavity_bbox)
        sr0, sc0, sr1, sc1 = src
        dr0, dc0, dr1, dc1 = dst
        if not (0 <= sr0 <= sr1 < height and 0 <= sc0 <= sc1 < width):
            return None
        if not (0 <= dr0 <= dr1 < height and 0 <= dc0 <= dc1 < width):
            return None
        if np.any(inp[sr0 : sr1 + 1, sc0 : sc1 + 1] != bg):
            return None
        if spec["role"] != "exact" and np.any(out[dr0 : dr1 + 1, dc0 : dc1 + 1] != bg):
            return None
        _paint_rect(inp, src, int(spec["color"]))
        _paint_rect(out, dst, int(spec["color"]))

    if _get_bg(inp) != bg:
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    cavity_choices = [(h, w) for h in range(3, 7) for w in range(3, 7) if h != w]
    cavity_h, cavity_w = random.choice(cavity_choices)
    specs = _build_specs(cavity_h, cavity_w)
    if specs is None:
        return None
    built = _build_instance(specs, cavity_h, cavity_w)
    if built is None:
        return None
    inp, expected = built
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

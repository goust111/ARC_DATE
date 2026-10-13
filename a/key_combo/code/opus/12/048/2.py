from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_GRID_RANGE = (22, 29)
GEN_NUM_PAIRS_OPTIONS = (2, 3)
GEN_RETRY_LIMIT = 400
GEN_RECT_OPTIONS = [(2, 2), (2, 3), (2, 4)]
GEN_CAVITY_SPECS = (
    {"cavity": (2, 5), "area": 10},
    {"cavity": (3, 4), "area": 12},
    {"cavity": (3, 5), "area": 15},
    {"cavity": (4, 4), "area": 16},
)
GEN_SIDE_PATTERNS = {
    2: [
        ("top", "bottom"),
        ("left", "right"),
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    ],
    3: [
        ("top", "left", "right"),
        ("bottom", "left", "right"),
        ("top", "bottom", "left"),
        ("top", "bottom", "right"),
    ],
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    best: tuple[int, int, int, int, int] | None = None
    ties = 0
    for top in range(h):
        full_cols = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            full_cols &= mask[bottom]
            rect_h = bottom - top + 1
            if rect_h < 2:
                continue
            c = 0
            while c < w:
                while c < w and not full_cols[c]:
                    c += 1
                start = c
                while c < w and full_cols[c]:
                    c += 1
                end = c - 1
                rect_w = end - start + 1
                if rect_w < 2:
                    continue
                cand = (rect_h * rect_w, top, start, bottom, end)
                if best is None or cand[0] > best[0]:
                    best = cand
                    ties = 1
                elif cand[0] == best[0]:
                    ties += 1
    assert best is not None, "expected a solid rectangle of height>=2 and width>=2"
    assert ties == 1, "largest solid rectangle must be unique"
    _area, top, left, bottom, right = best
    return top, left, bottom, right


def _extract_frames(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    frames: list[dict] = []
    for comp in _connected_components(grid == FRAME):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        assert outer_h >= 4 and outer_w >= 4, "frame must contain a >=2x2 cavity"
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub[0, :] == FRAME)
        assert np.all(sub[-1, :] == FRAME)
        assert np.all(sub[:, 0] == FRAME)
        assert np.all(sub[:, -1] == FRAME)
        cavity = sub[1:-1, 1:-1]
        assert cavity.size > 0 and not np.any(cavity == FRAME)
        assert np.all(cavity == BG), "input cavities must be empty"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "cavity_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "cavity_shape": cavity.shape,
                "cavity_area": int(cavity.size),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    assert 2 <= len(frames) <= 3, "expected 2 or 3 border frames"
    areas = [frame["cavity_area"] for frame in frames]
    assert len(set(areas)) == len(areas), "cavity areas must be unique"
    return frames


def _validate_noise_isolation(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    rect_mask = np.zeros_like(mask, dtype=bool)
    rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    noise = np.argwhere(mask & ~rect_mask)
    for r, c in noise:
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                assert not rect_mask[nr, nc], "noise may not touch its rectangle by 4-connectivity"
                assert not (mask[nr, nc] and not rect_mask[nr, nc]), "noise pixels must be isolated"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored rectangle into the border frame whose cavity area equals that color's noise count.

    Input structure:
    - The background is black, and 2 or 3 grey rectangular border frames touch the grid edge.
    - Each grey frame is a one-cell-thick rectangle with an empty black cavity of unique area.
    - Away from the frames, each non-grey foreground color forms one solid axis-aligned rectangle plus isolated same-color noise pixels.

    Rule:
    - For each foreground color, find its unique largest solid rectangle with height and width at least 2.
    - Count the remaining same-color pixels; that noise count determines which frame it matches because it equals exactly one cavity area.
    - Erase the original rectangle and all noise, keep the grey frames, and redraw the rectangle with its original size and color at the matched cavity's top-left corner.

    Key insight:
    - The scattered same-color noise is not disposable clutter: its count is the routing code that selects the destination cavity, whereas rectangle area alone is intentionally uninformative.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    frame_by_area = {frame["cavity_area"]: frame for frame in frames}

    colors = [int(color) for color in np.unique(g) if int(color) not in (BG, FRAME)]
    assert len(colors) == len(frames), "foreground colors must pair one-to-one with frames"

    rectangles: list[dict] = []
    for color in colors:
        mask = g == color
        rect = _largest_solid_rectangle(mask)
        _validate_noise_isolation(mask, rect)
        r0, c0, r1, c1 = rect
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        rect_area = rect_h * rect_w
        noise_count = int(mask.sum()) - rect_area
        assert noise_count in frame_by_area, "noise count must match a cavity area"
        rectangles.append(
            {
                "color": color,
                "shape": (rect_h, rect_w),
                "area": rect_area,
                "noise_count": noise_count,
            }
        )

    noise_counts = [item["noise_count"] for item in rectangles]
    assert len(set(noise_counts)) == len(noise_counts), "noise counts must be unique"

    out = np.full_like(g, BG)
    out[g == FRAME] = FRAME
    for item in rectangles:
        cavity = frame_by_area[item["noise_count"]]["cavity_bbox"]
        cav_h = cavity[2] - cavity[0] + 1
        cav_w = cavity[3] - cavity[1] + 1
        rect_h, rect_w = item["shape"]
        assert rect_h <= cav_h and rect_w <= cav_w
        out[cavity[0] : cavity[0] + rect_h, cavity[1] : cavity[1] + rect_w] = item["color"]
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _frame_bbox(side: str, cavity_shape: tuple[int, int], h: int, w: int) -> tuple[int, int, int, int] | None:
    cav_h, cav_w = cavity_shape
    outer_h = cav_h + 2
    outer_w = cav_w + 2
    for _ in range(80):
        if side == "top":
            r0 = 0
            c0 = random.randint(0, w - outer_w)
        elif side == "bottom":
            r0 = h - outer_h
            c0 = random.randint(0, w - outer_w)
        elif side == "left":
            r0 = random.randint(0, h - outer_h)
            c0 = 0
        else:
            r0 = random.randint(0, h - outer_h)
            c0 = w - outer_w
        return (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = FRAME
    grid[r0 + 1 : r1, c0 + 1 : c1] = BG


def _find_frame_layout(pair_specs: list[dict], h: int, w: int) -> list[dict] | None:
    placed: list[dict] = []
    for spec in pair_specs:
        bbox = None
        for _ in range(100):
            cand = _frame_bbox(spec["side"], spec["cavity"], h, w)
            if cand is None:
                continue
            if any(_boxes_overlap(cand, other["bbox"], margin=1) for other in placed):
                continue
            bbox = cand
            break
        if bbox is None:
            return None
        placed.append(
            {
                **spec,
                "bbox": bbox,
                "cavity_bbox": (bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1),
            }
        )
    return placed


def _place_rectangle(
    grid: np.ndarray,
    occ: np.ndarray,
    rect_shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    rect_h, rect_w = rect_shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - rect_h):
        for c0 in range(1, w - rect_w):
            if occ[r0 : r0 + rect_h, c0 : c0 + rect_w].any():
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + rect_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + rect_w + 1)
            if occ[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_noise_points(
    grid: np.ndarray,
    occ: np.ndarray,
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    chosen: list[tuple[int, int]] = []
    blocked = occ.copy()
    for _ in range(count):
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if blocked[r, c]:
                    continue
                if any(0 <= nr < h and 0 <= nc < w and blocked[nr, nc] for nr, nc in _neighbors4(r, c)):
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        chosen.append((r, c))
        blocked[r, c] = True
    return chosen


def _validate_semantics(inp: np.ndarray, expected_out: np.ndarray) -> None:
    frames = _extract_frames(inp)
    colors = [int(color) for color in np.unique(inp) if int(color) not in (BG, FRAME)]
    assert len(colors) == len(frames)

    cavity_areas = sorted(frame["cavity_area"] for frame in frames)
    rect_areas = []
    noise_counts = []
    for color in colors:
        mask = inp == color
        rect = _largest_solid_rectangle(mask)
        _validate_noise_isolation(mask, rect)
        r0, c0, r1, c1 = rect
        rect_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rect_areas.append(rect_area)
        noise_counts.append(int(mask.sum()) - rect_area)
    assert len(set(noise_counts)) == len(noise_counts)
    assert sorted(noise_counts) == cavity_areas
    assert set(rect_areas).isdisjoint(cavity_areas), "rectangle-area matching must not solve the task"
    assert np.array_equal(main(inp), expected_out)


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    num_pairs = random.choice(GEN_NUM_PAIRS_OPTIONS)

    side_pattern = list(random.choice(GEN_SIDE_PATTERNS[num_pairs]))
    random.shuffle(side_pattern)
    cavity_specs = random.sample(list(GEN_CAVITY_SPECS), k=num_pairs)
    rect_shapes = random.sample(list(GEN_RECT_OPTIONS), k=num_pairs)
    colors = random.sample(PALETTE, k=num_pairs)

    pair_specs: list[dict] = []
    for side, cavity_spec, rect_shape, color in zip(side_pattern, cavity_specs, rect_shapes, colors):
        pair_specs.append(
            {
                "side": side,
                "cavity": cavity_spec["cavity"],
                "cavity_area": cavity_spec["area"],
                "rect_shape": rect_shape,
                "color": color,
            }
        )

    frames = _find_frame_layout(pair_specs, h, w)
    if frames is None:
        return None

    output = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _draw_frame(output, frame["bbox"])
        cav = frame["cavity_bbox"]
        rect_h, rect_w = frame["rect_shape"]
        output[cav[0] : cav[0] + rect_h, cav[1] : cav[1] + rect_w] = frame["color"]

    inp = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame["bbox"])

    occ = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        occ[r0 : r1 + 1, c0 : c1 + 1] = True
    random.shuffle(frames)
    for frame in frames:
        color = frame["color"]
        rect_shape = frame["rect_shape"]
        rect_pos = _place_rectangle(inp, occ, rect_shape)
        if rect_pos is None:
            return None
        r0, c0 = rect_pos
        rect_h, rect_w = rect_shape
        inp[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
        occ[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

        noise = _place_noise_points(inp, occ, frame["cavity_area"])
        if noise is None:
            return None
        for rr, cc in noise:
            inp[rr, cc] = color
            occ[rr, cc] = True

    _validate_semantics(inp, output)
    return inp, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_instance()
        if built is not None:
            inp, _out = built
            return inp
    raise AssertionError("failed to generate a valid input")

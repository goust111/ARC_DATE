from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
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
                coords.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            bbox = _component_bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": len(coords),
                }
            )

    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if (h - 2) % 2 != 0 or (w - 2) % 2 != 0:
        return False
    if comp["area"] != 2 * h + 2 * w - 4:
        return False

    for r, c in comp["coords"]:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    return True


def _is_filled_rect(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)


def _quadrant_slices(frame_bbox: tuple[int, int, int, int]) -> list[tuple[slice, slice]]:
    r0, c0, r1, c1 = frame_bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    qh = inner_h // 2
    qw = inner_w // 2
    rs = [r0 + 1, r0 + 1 + qh, r1]
    cs = [c0 + 1, c0 + 1 + qw, c1]
    return [
        (slice(rs[0], rs[1]), slice(cs[0], cs[1])),
        (slice(rs[0], rs[1]), slice(cs[1], cs[2])),
        (slice(rs[1], rs[2]), slice(cs[0], cs[1])),
        (slice(rs[1], rs[2]), slice(cs[1], cs[2])),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing quadrant in each framed 2x2 layout and fill it with the
    outside sprite keyed by the frame color, recolored to the frame's fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components_by_color(g, bg)

    frames = [comp for comp in comps if _is_rect_frame(comp)]
    sprites = [comp for comp in comps if (not _is_rect_frame(comp)) and _is_filled_rect(comp)]

    assert frames, "expected at least one frame"
    frame_colors = [int(comp["color"]) for comp in frames]
    assert len(frame_colors) == len(set(frame_colors)), "frame colors must be unique"

    sprites_by_color: dict[int, list[dict]] = {}
    for sprite in sprites:
        sprites_by_color.setdefault(int(sprite["color"]), []).append(sprite)

    out = g.copy()

    for frame in frames:
        frame_color = int(frame["color"])
        matches = sprites_by_color.get(frame_color, [])
        assert len(matches) == 1, "each frame color must match exactly one outside sprite"
        sprite = matches[0]

        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = {int(x) for x in np.unique(inner)} - {bg}
        assert len(inner_colors) == 1, "frame interior must contain exactly one fill color"
        fill_color = inner_colors.pop()

        quadrants = _quadrant_slices(frame["bbox"])
        missing_idx = None
        for q_idx, (rs, cs) in enumerate(quadrants):
            block = g[rs, cs]
            if np.all(block == bg):
                assert missing_idx is None, "frame must have exactly one missing quadrant"
                missing_idx = q_idx
            else:
                assert np.all(block == fill_color), "non-missing quadrants must be uniformly filled"
        assert missing_idx is not None, "frame must have one empty quadrant"

        sprite_h = sprite["bbox"][2] - sprite["bbox"][0] + 1
        sprite_w = sprite["bbox"][3] - sprite["bbox"][1] + 1
        target_rs, target_cs = quadrants[missing_idx]
        target_h = target_rs.stop - target_rs.start
        target_w = target_cs.stop - target_cs.start
        assert (sprite_h, sprite_w) == (target_h, target_w), "matched sprite must fit the missing quadrant"

        out[target_rs, target_cs] = fill_color

    for frame_color, sprite_list in sprites_by_color.items():
        if frame_color not in frame_colors:
            continue
        sprite = sprite_list[0]
        for r, c in sprite["coords"]:
            out[r, c] = bg

    return out


MAX_ATTEMPTS = 400
GRID_MIN = 16
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
QUAD_SIZE_RANGE = (1, 4)
PLACEMENT_GAP = 1


def _reserve_rect(mask: np.ndarray, top: int, left: int, h: int, w: int, gap: int) -> None:
    rr0 = max(0, top - gap)
    rr1 = min(mask.shape[0], top + h + gap)
    cc0 = max(0, left - gap)
    cc1 = min(mask.shape[1], left + w + gap)
    mask[rr0:rr1, cc0:cc1] = True


def _find_positions(mask: np.ndarray, h: int, w: int, gap: int) -> list[tuple[int, int]]:
    gh, gw = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(gh - h + 1):
        for c in range(gw - w + 1):
            rr0 = max(0, r - gap)
            rr1 = min(gh, r + h + gap)
            cc0 = max(0, c - gap)
            cc1 = min(gw, c + w + gap)
            if not mask[rr0:rr1, cc0:cc1].any():
                out.append((r, c))
    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    bottom = top + h - 1
    right = left + w - 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color


def _paint_quadrants(
    grid: np.ndarray,
    top: int,
    left: int,
    qh: int,
    qw: int,
    fill_color: int,
    missing_idx: int,
) -> None:
    quadrants = [
        (slice(top + 1, top + 1 + qh), slice(left + 1, left + 1 + qw)),
        (slice(top + 1, top + 1 + qh), slice(left + 1 + qw, left + 1 + 2 * qw)),
        (slice(top + 1 + qh, top + 1 + 2 * qh), slice(left + 1, left + 1 + qw)),
        (slice(top + 1 + qh, top + 1 + 2 * qh), slice(left + 1 + qw, left + 1 + 2 * qw)),
    ]
    for idx, (rs, cs) in enumerate(quadrants):
        if idx != missing_idx:
            grid[rs, cs] = fill_color


def _sample_grid_size(frame_specs: list[dict]) -> tuple[int, int]:
    reserve_area = 0
    max_h = GRID_MIN
    max_w = GRID_MIN
    for spec in frame_specs:
        frame_h = 2 * spec["qh"] + 2
        frame_w = 2 * spec["qw"] + 2
        sprite_h = spec["qh"]
        sprite_w = spec["qw"]
        reserve_area += (frame_h + 2) * (frame_w + 2)
        reserve_area += (sprite_h + 2) * (sprite_w + 2)
        max_h = max(max_h, frame_h + 2)
        max_w = max(max_w, frame_w + 2)

    side = int(np.ceil(np.sqrt(reserve_area * 1.2)))
    h_min = min(GRID_MAX, max(max_h, side - 2))
    w_min = min(GRID_MAX, max(max_w, side + 1))
    if h_min > GRID_MAX or w_min > GRID_MAX:
        raise ValueError("sampled case cannot fit in max grid size")
    h = random.randint(h_min, GRID_MAX)
    w = random.randint(w_min, GRID_MAX)
    return h, w


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    num_frames = random.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    border_colors = palette[1 : 1 + num_frames]
    fill_colors = palette[1 + num_frames : 1 + 2 * num_frames]

    frame_specs: list[dict] = []
    for i in range(num_frames):
        qh = random.randint(QUAD_SIZE_RANGE[0], QUAD_SIZE_RANGE[1])
        qw = random.randint(QUAD_SIZE_RANGE[0], QUAD_SIZE_RANGE[1])
        frame_specs.append(
            {
                "border_color": int(border_colors[i]),
                "fill_color": int(fill_colors[i]),
                "qh": int(qh),
                "qw": int(qw),
                "missing_idx": random.randint(0, 3),
            }
        )

    h, w = _sample_grid_size(frame_specs)
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    occupied = np.zeros((h, w), dtype=bool)

    frame_order = sorted(range(num_frames), key=lambda i: (frame_specs[i]["qh"] * frame_specs[i]["qw"]), reverse=True)
    for i in frame_order:
        spec = frame_specs[i]
        frame_h = 2 * spec["qh"] + 2
        frame_w = 2 * spec["qw"] + 2
        positions = _find_positions(occupied, frame_h, frame_w, PLACEMENT_GAP)
        if not positions:
            raise ValueError("failed to place frame")
        top, left = random.choice(positions)
        spec["frame_top"] = top
        spec["frame_left"] = left
        _reserve_rect(occupied, top, left, frame_h, frame_w, PLACEMENT_GAP)

        _draw_frame(inp, top, left, frame_h, frame_w, spec["border_color"])
        _draw_frame(out, top, left, frame_h, frame_w, spec["border_color"])
        _paint_quadrants(inp, top, left, spec["qh"], spec["qw"], spec["fill_color"], spec["missing_idx"])
        _paint_quadrants(out, top, left, spec["qh"], spec["qw"], spec["fill_color"], -1)

    sprite_order = list(range(num_frames))
    random.shuffle(sprite_order)
    for i in sprite_order:
        spec = frame_specs[i]
        sprite_h = spec["qh"]
        sprite_w = spec["qw"]
        positions = _find_positions(occupied, sprite_h, sprite_w, PLACEMENT_GAP)
        if not positions:
            raise ValueError("failed to place sprite")
        top, left = random.choice(positions)
        spec["sprite_top"] = top
        spec["sprite_left"] = left
        _reserve_rect(occupied, top, left, sprite_h, sprite_w, PLACEMENT_GAP)
        inp[top : top + sprite_h, left : left + sprite_w] = spec["border_color"]

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            inp, out = _make_case()
        except ValueError:
            continue

        bg = _get_bg(inp)
        counts = Counter(map(int, inp.ravel()))
        if counts[bg] <= max(v for k, v in counts.items() if k != bg):
            continue

        try:
            got = main(inp)
        except AssertionError:
            continue

        if np.array_equal(got, out):
            return inp

    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")

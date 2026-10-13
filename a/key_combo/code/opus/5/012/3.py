from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = sorted(set(map(int, np.asarray(grid).ravel())))
    if exclude is None:
        return cols
    return [c for c in cols if c not in exclude]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    if colors is None:
        colors = colors_present(g, exclude={int(bg)})
    out = []
    for color in colors:
        for coords in components_mask(g == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _scaled_index(i: int, src_len: int, dst_len: int) -> int:
    if src_len == 1:
        return int((dst_len - 1) // 2)
    return int(np.floor((int(i) * (dst_len - 1) / (src_len - 1)) + 0.5))


def _draw_hv_line(stroke: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    c_lo, c_hi = sorted((int(c0), int(c1)))
    stroke[int(r0), c_lo : c_hi + 1] = True
    r_lo, r_hi = sorted((int(r0), int(r1)))
    stroke[r_lo : r_hi + 1, int(c1)] = True


def _fill_enclosed(stroke: np.ndarray) -> np.ndarray:
    open_space = ~np.asarray(stroke, dtype=bool)
    h, w = open_space.shape
    padded = np.pad(open_space, 1, constant_values=True)
    seen = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    ph, pw = padded.shape
    for r in range(ph):
        for c in (0, pw - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(pw):
        for r in (0, ph - 1):
            if padded[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    holes = open_space & ~seen[1 : h + 1, 1 : w + 1]
    return np.asarray(stroke, dtype=bool) | holes


def _stretch_stats(sprite: np.ndarray, target_shape: tuple[int, int]) -> dict:
    sprite = np.asarray(sprite, dtype=bool)
    src_h, src_w = sprite.shape
    dst_h, dst_w = map(int, target_shape)
    stroke = np.zeros((dst_h, dst_w), dtype=bool)
    mapped: dict[tuple[int, int], tuple[int, int]] = {}
    for sr, sc in np.argwhere(sprite):
        tr = _scaled_index(int(sr), src_h, dst_h)
        tc = _scaled_index(int(sc), src_w, dst_w)
        mapped[(int(sr), int(sc))] = (tr, tc)
        stroke[tr, tc] = True
    point_count = int(stroke.sum())
    for sr, sc in np.argwhere(sprite):
        sr = int(sr)
        sc = int(sc)
        for nr, nc in ((sr + 1, sc), (sr, sc + 1)):
            if 0 <= nr < src_h and 0 <= nc < src_w and sprite[nr, nc]:
                r0, c0 = mapped[(sr, sc)]
                r1, c1 = mapped[(nr, nc)]
                _draw_hv_line(stroke, r0, c0, r1, c1)
    filled = _fill_enclosed(stroke)
    return {
        "points": {k: v for k, v in mapped.items()},
        "stroke": stroke,
        "filled": filled,
        "point_count": point_count,
        "stroke_count": int(stroke.sum()),
        "fill_count": int(filled.sum()),
    }


def _find_separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == 6)]
    assert len(rows) == 1, "expected a unique full-width magenta separator row"
    return int(rows[0])


def _parse_legend(legend: np.ndarray) -> list[dict]:
    g = np.asarray(legend, dtype=int)
    assert g.shape[0] >= 4, "legend must contain sprites plus marker row"
    marker_row = g.shape[0] - 1
    sprite_comps = find_components(g, bg=0, colors=[5])
    sprite_comps.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    markers = [(c, int(g[marker_row, c])) for c in range(g.shape[1]) if int(g[marker_row, c]) in {1, 2, 3}]
    assert len(sprite_comps) == len(markers) and 2 <= len(markers) <= 3
    pairs = []
    for comp, (mc, color) in zip(sprite_comps, sorted(markers)):
        r0, c0, r1, c1 = comp["bbox"]
        assert r1 == marker_row - 1, "each sprite must sit directly above the marker row"
        assert c0 <= mc <= c1, "marker must lie under its sprite bbox"
        sprite_mask = crop_bbox(g == 5, comp["bbox"])
        pairs.append({"mark_color": int(color), "mask": sprite_mask, "bbox": comp["bbox"]})
    return pairs


def _find_frames(work: np.ndarray) -> list[dict]:
    g = np.asarray(work, dtype=int)
    frames = []
    for color in colors_present(g, exclude={0}):
        if color not in {1, 2, 3}:
            continue
        mask = g == int(color)
        bbox = _rect_frame_bbox(mask)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0 and np.all(interior == 0), "frame interior must be empty in input"
        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "interior_shape": (r1 - r0 - 1, c1 - c0 - 1),
            }
        )
    frames.sort(key=lambda frame: frame["color"])
    assert 2 <= len(frames) <= 3, "expected 2 or 3 hollow rectangles"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each legend sprite into the same-colored hollow rectangle and fill any enclosed regions.

    Input structure:
    - Black background surrounds two zones split by one full-width magenta row.
    - Above the separator, several small gray 4-connected sprites sit in the legend, each with one red/blue/green marker pixel directly below it.
    - Below the separator, the work area contains 2 or 3 non-touching hollow rectangles whose one-pixel borders are red, blue, or green and whose interiors are empty.

    Rule:
    - Identify the gray legend sprites and pair them left-to-right with the colored marker pixels beneath them.
    - Match each legend sprite to the unique work-area rectangle whose border color equals that marker color.
    - Map the legend sprite bbox to the rectangle interior bbox, linearly rounding each mapped pixel to the nearest cell.
    - For every 4-neighbor adjacency in the original sprite, draw a Manhattan path in the rectangle interior using horizontal then vertical segments, then fill any enclosed background hole created by those strokes.
    - Remove the legend and separator by returning only the transformed work area.

    Key insight:
    - The colored marker picks which frame receives each sprite, while the gray sprite supplies a stretchable connectivity blueprint rather than a bitmap copy.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)
    legend = g[:sep]
    work = g[sep + 1 :].copy()
    legend_pairs = _parse_legend(legend)
    frames = _find_frames(work)
    frame_by_color = {frame["color"]: frame for frame in frames}
    assert {pair["mark_color"] for pair in legend_pairs} == set(frame_by_color), "marker colors and frame colors must match"

    out = work.copy()
    for pair in legend_pairs:
        color = pair["mark_color"]
        frame = frame_by_color[color]
        r0, c0, r1, c1 = frame["interior_bbox"]
        stretched = _stretch_stats(pair["mask"], frame["interior_shape"])["filled"]
        out[r0 : r1 + 1, c0 : c1 + 1][stretched] = int(color)
    return out


SPRITE_LIBRARY = [
    {"name": "block", "mask": np.array([[1, 1], [1, 1]], dtype=bool), "closed": True},
    {"name": "ell", "mask": np.array([[1, 0], [1, 0], [1, 1]], dtype=bool), "closed": False},
    {"name": "tee", "mask": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool), "closed": False},
    {"name": "zig", "mask": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool), "closed": False},
    {"name": "cee", "mask": np.array([[1, 1], [1, 0], [1, 1]], dtype=bool), "closed": False},
    {"name": "pee", "mask": np.array([[1, 1], [1, 1], [1, 0]], dtype=bool), "closed": True},
]
GEN_RETRIES = 400
MARKER_COLORS = (1, 2, 3)


def _tight_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    return crop_bbox(np.asarray(mask, dtype=bool), bbox_coords(coords))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=int))


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _tight_bool(out)


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _select_sprites(n: int) -> list[dict]:
    closed = [item for item in SPRITE_LIBRARY if item["closed"]]
    open_ = [item for item in SPRITE_LIBRARY if not item["closed"]]
    picked = []
    if n >= 2:
        picked.append(random.choice(closed))
        picked.append(random.choice(open_))
    while len(picked) < n:
        candidate = random.choice(SPRITE_LIBRARY)
        if candidate["name"] not in {item["name"] for item in picked}:
            picked.append(candidate)
    result = []
    seen = set()
    for item in picked:
        for _ in range(20):
            mask = _transform_mask(item["mask"])
            key = _mask_key(mask)
            if key not in seen:
                seen.add(key)
                result.append({"name": item["name"], "mask": mask, "closed": item["closed"]})
                break
        else:
            result.append({"name": item["name"], "mask": _tight_bool(item["mask"]), "closed": item["closed"]})
    return result


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _place_rectangles(work_h: int, work_w: int, specs: list[dict]) -> list[dict] | None:
    order = specs[:]
    random.shuffle(order)
    occupied = np.zeros((work_h, work_w), dtype=bool)
    placed = []
    for spec in order:
        outer_h = spec["interior_shape"][0] + 2
        outer_w = spec["interior_shape"][1] + 2
        choices = []
        for r0 in range(work_h - outer_h + 1):
            for c0 in range(work_w - outer_w + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(work_h, r0 + outer_h + 1)
                cc1 = min(work_w, c0 + outer_w + 1)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        placed.append(
            {
                "color": spec["color"],
                "bbox": bbox,
                "interior_bbox": (r0 + 1, c0 + 1, r0 + outer_h - 2, c0 + outer_w - 2),
                "interior_shape": spec["interior_shape"],
            }
        )
        occupied[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
    return placed


def _legend_layout(width: int, sprites: list[dict], colors: list[int]) -> tuple[list[dict], np.ndarray] | None:
    sprite_widths = [int(item["mask"].shape[1]) for item in sprites]
    min_total = sum(sprite_widths) + (len(sprites) - 1)
    if min_total > width:
        return None
    slack = width - min_total
    gaps = _random_composition(slack, len(sprites) + 1)
    x = gaps[0]
    layout = []
    legend = np.zeros((4, width), dtype=int)
    for idx, (sprite, color) in enumerate(zip(sprites, colors)):
        mask = sprite["mask"]
        h, w = mask.shape
        top = 3 - h
        if top < 0:
            return None
        legend[top : top + h, x : x + w][mask] = 5
        cols_with_pixels = np.where(mask.any(axis=0))[0]
        marker_c = x + int(random.choice(list(cols_with_pixels)))
        legend[3, marker_c] = int(color)
        layout.append({"color": int(color), "mask": mask, "x": int(x), "marker_c": int(marker_c)})
        x += w + 1 + gaps[idx + 1]
    return layout, legend


def _build_grid() -> np.ndarray | None:
    n = random.choice([2, 3])
    sprites = _select_sprites(n)
    colors = random.sample(list(MARKER_COLORS), n)

    rect_specs = []
    for sprite, color in zip(sprites, colors):
        sh, sw = sprite["mask"].shape
        for _ in range(20):
            ih = random.randint(max(3, sh), 6)
            iw = random.randint(max(3, sw), 6)
            if ih > sh or iw > sw:
                rect_specs.append({"color": int(color), "interior_shape": (ih, iw), "closed": sprite["closed"], "mask": sprite["mask"]})
                break
        else:
            return None

    min_legend_w = sum(mask["mask"].shape[1] for mask in rect_specs) + n - 1
    work_w = random.randint(max(12, min_legend_w), 20)
    work_h = random.randint(8, 14)
    placed = _place_rectangles(work_h, work_w, rect_specs)
    if placed is None:
        return None

    x_sorted_colors = [item["color"] for item in sorted(placed, key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0]))]
    if n > 1 and x_sorted_colors == colors:
        return None

    legend_data = _legend_layout(work_w, sprites, colors)
    if legend_data is None:
        return None
    _layout, legend = legend_data

    grid = np.zeros((5 + work_h, work_w), dtype=int)
    grid[:4] = legend
    grid[4, :] = 6
    for item in placed:
        _draw_frame(grid[5:], item["bbox"], item["color"])
    return grid


def _candidate_ok(grid: np.ndarray) -> bool:
    try:
        sep = _find_separator_row(grid)
        legend_pairs = _parse_legend(grid[:sep])
        frames = _find_frames(grid[sep + 1 :])
        if len(legend_pairs) != len(frames):
            return False
        raw_work = grid[sep + 1 :]
        out = main(grid)
        if np.array_equal(out, raw_work):
            return False
        frame_by_color = {frame["color"]: frame for frame in frames}
        needs_line = False
        needs_fill = False
        for pair in legend_pairs:
            stats = _stretch_stats(pair["mask"], frame_by_color[pair["mark_color"]]["interior_shape"])
            needs_line |= stats["stroke_count"] > stats["point_count"]
            needs_fill |= stats["fill_count"] > stats["stroke_count"]
        if not needs_line:
            return False
        if any(pair["mask"].sum() >= 4 and pair["mask"].shape == (2, 2) for pair in legend_pairs) and not needs_fill:
            return False
        inner_before = sum(np.count_nonzero(raw_work[frame["bbox"][0] + 1 : frame["bbox"][2], frame["bbox"][1] + 1 : frame["bbox"][3]]) for frame in frames)
        inner_after = sum(np.count_nonzero(out[frame["bbox"][0] + 1 : frame["bbox"][2], frame["bbox"][1] + 1 : frame["bbox"][3]]) for frame in frames)
        if inner_after <= inner_before:
            return False
    except Exception:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_grid()
        if grid is not None and _candidate_ok(grid):
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

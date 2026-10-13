from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 300
FRAME_COUNT_RANGE = (3, 5)
INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (4, 10)
FRAME_GAP_CHOICES = (2, 3)
FRAME_MARGIN_RANGE = (1, 3)
PALETTE_SEP_RANGE = (2, 4)
PALETTE_GAP_RANGE = (1, 3)
OUTER_EXTRA_MARGIN_RANGE = (1, 4)


def _boundary_bg(g: np.ndarray) -> int:
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _mono_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_from_points(pts)
            comps.append(
                {
                    "color": color,
                    "coords": np.array(pts, dtype=int),
                    "bbox": bbox,
                    "area": int(len(pts)),
                }
            )
    return comps


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == h - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == w - 1)
    )
    return bool(np.all(on_border))


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        int(outer[0]) < int(inner[0])
        and int(outer[1]) < int(inner[1])
        and int(outer[2]) > int(inner[2])
        and int(outer[3]) > int(inner[3])
    )


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (
        int(a[2]) < int(b[0])
        or int(b[2]) < int(a[0])
        or int(a[3]) < int(b[1])
        or int(b[3]) < int(a[1])
    )


def _frame_chain(g: np.ndarray, bg: int) -> list[dict]:
    frames = [comp for comp in _mono_components(g, bg) if _is_rect_frame(comp)]
    frames.sort(key=lambda comp: _bbox_area(comp["bbox"]), reverse=True)
    assert len(frames) >= 3, "expected at least three nested rectangular frames"
    chain = [frames[0]]
    for comp in frames[1:]:
        if _contains(chain[-1]["bbox"], comp["bbox"]):
            chain.append(comp)
    assert len(chain) >= 3, "expected a full nested frame chain"
    return chain


def _palette_map(
    g: np.ndarray,
    bg: int,
    outer_bbox: tuple[int, int, int, int],
) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for pts in _mask_components(g != bg):
        bbox = _bbox_from_points([tuple(map(int, p)) for p in pts])
        if _intersects(bbox, outer_bbox):
            continue
        r0, c0, r1, c1 = bbox
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        h, w = sub.shape
        if h not in (2, 3) or w != h + 1:
            continue
        marker = int(sub[0, 0])
        if np.any(sub[:, 0] != marker):
            continue
        pattern = np.asarray(sub[:, 1:], dtype=int).copy()
        vals = np.unique(pattern)
        if vals.size < 2:
            continue
        mapping[marker] = pattern
    return mapping


def _tile_pattern(
    out: np.ndarray,
    mask: np.ndarray,
    pattern: np.ndarray,
    anchor_r: int,
    anchor_c: int,
) -> None:
    ph, pw = pattern.shape
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols, strict=False):
        out[int(r), int(c)] = int(pattern[(int(r) - anchor_r) % ph, (int(c) - anchor_c) % pw])


def _fill_nested_regions(
    crop: np.ndarray,
    frames: list[dict],
    palette: dict[int, np.ndarray],
    crop_origin: tuple[int, int],
) -> np.ndarray:
    out = crop.copy()
    base_r, base_c = crop_origin
    rel_boxes = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        rel_boxes.append((int(r0 - base_r), int(c0 - base_c), int(r1 - base_r), int(c1 - base_c)))

    for i, frame in enumerate(frames[:-1]):
        color = int(frame["color"])
        assert color in palette, "missing palette entry for an outer frame color"
        r0, c0, r1, c1 = rel_boxes[i]
        ir0, ic0, ir1, ic1 = rel_boxes[i + 1]
        mask = np.zeros_like(out, dtype=bool)
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
        mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = False
        _tile_pattern(out, mask, palette[color], r0 + 1, c0 + 1)

    inner = frames[-1]
    color = int(inner["color"])
    assert color in palette, "missing palette entry for the innermost frame color"
    r0, c0, r1, c1 = rel_boxes[-1]
    mask = np.zeros_like(out, dtype=bool)
    mask[r0 + 1 : r1, c0 + 1 : c1] = True
    _tile_pattern(out, mask, palette[color], r0 + 1, c0 + 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _boundary_bg(g)
    frames = _frame_chain(g, bg)
    outer_bbox = frames[0]["bbox"]
    palette = _palette_map(g, bg, outer_bbox)
    needed = {int(frame["color"]) for frame in frames}
    assert needed.issubset(set(palette)), "palette must cover every frame color"

    r0, c0, r1, c1 = outer_bbox
    crop = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    return _fill_nested_regions(crop, frames, palette, (int(r0), int(c0)))


def _pattern_mask(size: int, kind: str) -> np.ndarray:
    rr, cc = np.indices((size, size))
    if kind == "checker":
        return (rr + cc) % 2 == 0
    if kind == "vstripe":
        return cc % 2 == 0
    if kind == "hstripe":
        return rr % 2 == 0
    if kind == "upper_tri":
        return rr <= cc
    if kind == "cross":
        mid = size // 2
        return (rr == mid) | (cc == mid)
    if kind == "x":
        return (rr == cc) | (rr + cc == size - 1)
    raise ValueError(f"unknown pattern kind: {kind}")


def _make_pattern(size: int, marker: int) -> np.ndarray:
    kinds_by_size = {
        2: ("checker", "vstripe", "hstripe", "upper_tri"),
        3: ("checker", "vstripe", "hstripe", "upper_tri", "cross", "x"),
    }
    colors = [c for c in range(1, 10) if c != marker]
    a, b = random.sample(colors, 2)
    mask = _pattern_mask(size, random.choice(kinds_by_size[size]))
    return np.where(mask, a, b).astype(int)


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _spec_to_case(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    frame_boxes_rel = spec["frame_boxes_rel"]
    frame_colors = spec["frame_colors"]
    palette_items = spec["palette_items"]
    frame_top = int(spec["frame_top"])
    frame_left = int(spec["frame_left"])
    palette_top = int(spec["palette_top"])
    palette_left = int(spec["palette_left"])
    height = int(spec["height"])
    width = int(spec["width"])

    inp = np.full((height, width), BG, dtype=int)
    frame_boxes_abs = []
    for bbox, color in zip(frame_boxes_rel, frame_colors, strict=False):
        r0, c0, r1, c1 = bbox
        abs_bbox = (r0 + frame_top, c0 + frame_left, r1 + frame_top, c1 + frame_left)
        frame_boxes_abs.append(abs_bbox)
        _draw_frame(inp, abs_bbox, int(color))

    for item in palette_items:
        pattern = item["pattern"]
        marker = int(item["marker"])
        top = int(item["top"])
        left = int(item["left"])
        patch = np.empty((pattern.shape[0], pattern.shape[1] + 1), dtype=int)
        patch[:, 0] = marker
        patch[:, 1:] = pattern
        inp[top : top + patch.shape[0], left : left + patch.shape[1]] = patch

    crop_h = frame_boxes_rel[0][2] - frame_boxes_rel[0][0] + 1
    crop_w = frame_boxes_rel[0][3] - frame_boxes_rel[0][1] + 1
    crop = inp[frame_top : frame_top + crop_h, frame_left : frame_left + crop_w].copy()
    frame_dicts = [
        {"color": int(color), "bbox": bbox}
        for color, bbox in zip(frame_colors, frame_boxes_abs, strict=False)
    ]
    palette = {int(item["marker"]): np.asarray(item["pattern"], dtype=int) for item in palette_items}
    out = _fill_nested_regions(crop, frame_dicts, palette, (frame_top, frame_left))
    return inp, out


def _build_spec() -> dict:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    gaps = [random.choice(FRAME_GAP_CHOICES) for _ in range(n_frames - 1)]
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)
    total_offset = sum(gap + 1 for gap in gaps)
    outer_h = inner_h + 2 + 2 * total_offset
    outer_w = inner_w + 2 + 2 * total_offset

    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + gap + 1)
    frame_boxes_rel = [
        (off, off, outer_h - 1 - off, outer_w - 1 - off)
        for off in offsets
    ]

    frame_colors = random.sample(range(1, 10), n_frames)
    pattern_specs = []
    for color in frame_colors:
        size = random.choice((2, 3))
        pattern_specs.append({"marker": int(color), "pattern": _make_pattern(size, int(color))})

    palette_items = random.sample(pattern_specs, len(pattern_specs))
    palette_gaps = [random.randint(*PALETTE_GAP_RANGE) for _ in range(len(palette_items) - 1)]
    palette_width = max(item["pattern"].shape[1] + 1 for item in palette_items)
    palette_height = sum(item["pattern"].shape[0] for item in palette_items) + sum(palette_gaps)

    frame_top = random.randint(*FRAME_MARGIN_RANGE)
    frame_left = random.randint(*FRAME_MARGIN_RANGE)
    palette_sep = random.randint(*PALETTE_SEP_RANGE)
    palette_left = frame_left + outer_w + palette_sep

    extra_bottom = random.randint(*OUTER_EXTRA_MARGIN_RANGE)
    height = max(
        frame_top + outer_h + extra_bottom,
        palette_height + extra_bottom + random.randint(*FRAME_MARGIN_RANGE),
    )
    palette_top_max = height - palette_height - 1
    palette_top = random.randint(1, max(1, palette_top_max))
    width = palette_left + palette_width + random.randint(*OUTER_EXTRA_MARGIN_RANGE)

    y = palette_top
    placed_items = []
    for i, item in enumerate(palette_items):
        placed_items.append(
            {
                "marker": int(item["marker"]),
                "pattern": np.asarray(item["pattern"], dtype=int),
                "top": int(y),
                "left": int(palette_left),
            }
        )
        y += item["pattern"].shape[0]
        if i + 1 < len(palette_items):
            y += palette_gaps[i]

    return {
        "height": int(height),
        "width": int(width),
        "frame_top": int(frame_top),
        "frame_left": int(frame_left),
        "palette_top": int(palette_top),
        "palette_left": int(palette_left),
        "frame_boxes_rel": frame_boxes_rel,
        "frame_colors": frame_colors,
        "palette_items": placed_items,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        spec = _build_spec()
        inp, expected = _spec_to_case(spec)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC instance")

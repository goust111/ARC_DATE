from __future__ import annotations

import numpy as np


BG = 0
SLOT_COLOR = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _is_slot_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == SLOT_COLOR):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == SLOT_COLOR):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == SLOT_COLOR):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == SLOT_COLOR):
        return False
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return False
    return True


def _segment_info(comp: dict) -> tuple[str, int]:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h == 1 and w >= 2 and int(comp["area"]) == w:
        return ("h", w)
    if w == 1 and h >= 2 and int(comp["area"]) == h:
        return ("v", h)
    raise AssertionError("foreground component is not a valid line segment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color line segments and encode each group as a centered rectangle in a grey slot.

    Input structure:
    - Black background.
    - Several horizontal or vertical monochrome line segments.
    - A bottom row of grey hollow slots.

    Rule:
    - Group segments by color; same-color segments share length and orientation.
    - If segments are horizontal, create an N x L solid rectangle.
    - If segments are vertical, create an L x N solid rectangle.
    - Sort groups by segment length and place them left-to-right into the slots.
    - Remove the original segments while preserving the grey slot borders.
    """
    g = np.asarray(grid, dtype=int)

    slot_comps = [comp for comp in _components_of_color(g, SLOT_COLOR) if _is_slot_frame(g, comp)]
    assert slot_comps, "expected at least one grey slot"
    slot_comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))

    groups: dict[int, list[dict]] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BG, SLOT_COLOR):
            continue
        comps = _components_of_color(g, color)
        assert comps, "foreground color must have at least one component"
        for comp in comps:
            orient, length = _segment_info(comp)
            groups.setdefault(color, []).append(
                {
                    "comp": comp,
                    "orient": orient,
                    "length": int(length),
                }
            )

    assert groups, "expected at least one encoded color group"
    assert len(groups) == len(slot_comps), "slot count must match the number of colors"

    items: list[dict] = []
    for color, segs in groups.items():
        orient = segs[0]["orient"]
        length = int(segs[0]["length"])
        assert all(seg["orient"] == orient for seg in segs)
        assert all(int(seg["length"]) == length for seg in segs)
        count = len(segs)
        anchor = min(
            (int(seg["comp"]["bbox"][0]), int(seg["comp"]["bbox"][1]))
            for seg in segs
        )
        rect_h, rect_w = (count, length) if orient == "h" else (length, count)
        items.append(
            {
                "color": int(color),
                "length": int(length),
                "anchor": anchor,
                "rect_h": int(rect_h),
                "rect_w": int(rect_w),
                "segs": segs,
            }
        )

    items.sort(key=lambda item: (item["length"], item["anchor"][0], item["anchor"][1]))
    out = g.copy()

    for item in items:
        for seg in item["segs"]:
            for r, c in seg["comp"]["coords"]:
                out[int(r), int(c)] = BG

    for item, slot in zip(items, slot_comps):
        r0, c0, r1, c1 = slot["bbox"]
        inner_h = int(r1 - r0 - 1)
        inner_w = int(c1 - c0 - 1)
        rect_h = int(item["rect_h"])
        rect_w = int(item["rect_w"])
        assert rect_h <= inner_h and rect_w <= inner_w, "slot interior is too small"
        off_r = (inner_h - rect_h) // 2
        off_c = (inner_w - rect_w) // 2
        rr = int(r0 + 1 + off_r)
        cc = int(c0 + 1 + off_c)
        out[rr : rr + rect_h, cc : cc + rect_w] = int(item["color"])

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    top = max(0, r0 - 1)
    left = max(0, c0 - 1)
    bottom = min(occupied.shape[0], r0 + h + 1)
    right = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[top:bottom, left:right].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    top = max(0, r0 - 1)
    left = max(0, c0 - 1)
    bottom = min(occupied.shape[0], r0 + h + 1)
    right = min(occupied.shape[1], c0 + w + 1)
    occupied[top:bottom, left:right] = True


def _sample_group_specs(rng: np.random.Generator) -> list[dict]:
    num_groups = int(rng.integers(2, 5))
    lengths = rng.choice(np.arange(2, 8), size=num_groups, replace=False)
    colors = rng.choice(np.array(PALETTE, dtype=int), size=num_groups, replace=False)

    specs: list[dict] = []
    for color, length in zip(colors.tolist(), lengths.tolist()):
        orient = "h" if bool(rng.integers(0, 2)) else "v"
        count = int(rng.integers(1, 5))
        rect_h, rect_w = (count, length) if orient == "h" else (length, count)
        specs.append(
            {
                "color": int(color),
                "length": int(length),
                "orient": orient,
                "count": int(count),
                "rect_h": int(rect_h),
                "rect_w": int(rect_w),
            }
        )
    return specs


def _slot_layout(
    specs: list[dict], rng: np.random.Generator
) -> tuple[int, int, int, int, int, int, int]:
    max_rect_h = max(int(spec["rect_h"]) for spec in specs)
    max_rect_w = max(int(spec["rect_w"]) for spec in specs)
    inner_h = max_rect_h + int(rng.integers(0, 3))
    inner_w = max_rect_w + int(rng.integers(0, 3))
    slot_h = inner_h + 2
    slot_w = inner_w + 2
    gap = int(rng.integers(2, 5))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 5))
    extra_w = int(rng.integers(0, 8))
    width = left + len(specs) * slot_w + (len(specs) - 1) * gap + right + extra_w
    return inner_h, inner_w, slot_h, slot_w, gap, left, width


def _place_segments(
    grid: np.ndarray,
    specs: list[dict],
    top_h: int,
    rng: np.random.Generator,
) -> bool:
    occupied = np.zeros((top_h, grid.shape[1]), dtype=bool)
    segments: list[dict] = []
    for spec in specs:
        for _ in range(int(spec["count"])):
            segments.append(
                {
                    "color": int(spec["color"]),
                    "length": int(spec["length"]),
                    "orient": spec["orient"],
                }
            )

    rng.shuffle(segments)
    segments.sort(key=lambda seg: (seg["orient"] == "v", seg["length"]), reverse=True)

    for seg in segments:
        if seg["orient"] == "h":
            h, w = 1, int(seg["length"])
        else:
            h, w = int(seg["length"]), 1
        if h > top_h or w > grid.shape[1]:
            return False

        placed = False
        for _ in range(250):
            r0 = int(rng.integers(0, top_h - h + 1))
            c0 = int(rng.integers(0, grid.shape[1] - w + 1))
            if not _can_place(occupied, r0, c0, h, w):
                continue
            _mark_occupied(occupied, r0, c0, h, w)
            if seg["orient"] == "h":
                grid[r0, c0 : c0 + w] = int(seg["color"])
            else:
                grid[r0 : r0 + h, c0] = int(seg["color"])
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(inp: np.ndarray, specs: list[dict], slots: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.full(inp.shape, BG, dtype=int)
    for r0, c0, h, w in slots:
        _draw_frame(out, int(r0), int(c0), int(h), int(w), SLOT_COLOR)

    ordered = sorted(specs, key=lambda spec: int(spec["length"]))
    for spec, (r0, c0, h, w) in zip(ordered, slots):
        inner_h = h - 2
        inner_w = w - 2
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        rr = int(r0 + 1 + (inner_h - rect_h) // 2)
        cc = int(c0 + 1 + (inner_w - rect_w) // 2)
        out[rr : rr + rect_h, cc : cc + rect_w] = int(spec["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _sample_group_specs(rng)
    _inner_h, _inner_w, slot_h, slot_w, slot_gap, left, width = _slot_layout(specs, rng)

    top_h = max(
        max(int(spec["length"]) for spec in specs) + 2,
        6 + sum(int(spec["count"]) for spec in specs) // 2,
        8,
    ) + int(rng.integers(0, 6))
    slot_gap_v = int(rng.integers(1, 3))
    bottom_margin = int(rng.integers(0, 2))
    height = top_h + slot_gap_v + slot_h + bottom_margin
    slot_top = top_h + slot_gap_v

    grid = np.full((height, width), BG, dtype=int)

    slots: list[tuple[int, int, int, int]] = []
    c = int(left)
    for _ in specs:
        _draw_frame(grid, slot_top, c, slot_h, slot_w, SLOT_COLOR)
        slots.append((slot_top, c, slot_h, slot_w))
        c += slot_w + slot_gap

    if not _place_segments(grid, specs, top_h, rng):
        return None

    expected = _expected_output(grid, specs, slots)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

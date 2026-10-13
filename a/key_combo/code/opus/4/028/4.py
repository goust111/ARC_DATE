from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("up", "down", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _draw_hollow_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _rect_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_hollow_rectangle(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[1:-1, 1:-1] != BG):
        return False
    expected = np.full((h, w), BG, dtype=int)
    expected[_rect_mask(h, w)] = color
    return np.array_equal(sub, expected)


def _marker_direction(box: tuple[int, int, int, int], marker: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = box
    mr, mc = marker
    if mr < r0 and c0 <= mc <= c1:
        return "up"
    if mr > r1 and c0 <= mc <= c1:
        return "down"
    if mc < c0 and r0 <= mr <= r1:
        return "left"
    if mc > c1 and r0 <= mr <= r1:
        return "right"
    return None


def _box_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _extend_box(box: tuple[int, int, int, int], direction: str, dist: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    if direction == "up":
        return r0 - dist, c0, r1, c1
    if direction == "down":
        return r0, c0, r1 + dist, c1
    if direction == "left":
        return r0, c0 - dist, r1, c1
    return r0, c0, r1, c1 + dist


def _boundary_allowance(shape: tuple[int, int], box: tuple[int, int, int, int], direction: str) -> int:
    h, w = shape
    r0, c0, r1, c1 = box
    if direction == "up":
        return r0
    if direction == "down":
        return h - 1 - r1
    if direction == "left":
        return c0
    return w - 1 - c1


def _overlap_allowance(
    box: tuple[int, int, int, int],
    direction: str,
    other_boxes: list[tuple[int, int, int, int]],
) -> int:
    r0, c0, r1, c1 = box
    best = 10**9
    for or0, oc0, or1, oc1 in other_boxes:
        if direction in ("left", "right"):
            if max(r0, or0) > min(r1, or1):
                continue
            if direction == "left" and oc1 < c0:
                best = min(best, c0 - oc1 - 1)
            if direction == "right" and oc0 > c1:
                best = min(best, oc0 - c1 - 1)
        else:
            if max(c0, oc0) > min(c1, oc1):
                continue
            if direction == "up" and or1 < r0:
                best = min(best, r0 - or1 - 1)
            if direction == "down" and or0 > r1:
                best = min(best, or0 - r1 - 1)
    return best if best != 10**9 else 10**9


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rectangles = []
    sprites_by_color: dict[int, list[dict]] = {}
    for color in sorted({int(v) for v in g.ravel()} - {bg, MARKER}):
        for coords in _components_mask(g == color, connectivity=8):
            box = _bbox_coords(coords)
            if _is_hollow_rectangle(g, color, coords):
                rectangles.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": box,
                    }
                )
            else:
                r0, c0, r1, c1 = box
                area = int(coords.shape[0])
                if (r1 - r0 + 1, c1 - c0 + 1) == (3, 3) and 4 <= area <= 6:
                    sprites_by_color.setdefault(color, []).append(
                        {"coords": coords, "bbox": box, "area": area}
                    )

    rectangles.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    colors = [item["color"] for item in rectangles]
    assert len(colors) == len(set(colors)) and 2 <= len(colors) <= 3

    marker_components = _components_mask(g == MARKER, connectivity=8)
    markers = [tuple(map(int, comp[0])) for comp in marker_components]
    assert len(markers) == len(rectangles)
    assert all(comp.shape[0] == 1 for comp in marker_components)

    used_markers: set[tuple[int, int]] = set()
    for rect in rectangles:
        candidates = []
        for marker in markers:
            direction = _marker_direction(rect["bbox"], marker)
            if direction is not None:
                candidates.append((marker, direction))
        assert len(candidates) == 1
        marker, direction = candidates[0]
        assert marker not in used_markers
        used_markers.add(marker)
        rect["marker"] = marker
        rect["direction"] = direction
        rect["sprite_count"] = len(sprites_by_color.get(rect["color"], []))
        assert rect["sprite_count"] >= 1
    return {"bg": bg, "rectangles": rectangles, "sprites_by_color": sprites_by_color}


def _solve(grid: np.ndarray, *, return_debug: bool = False):
    info = _analyze_grid(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    boxes = [tuple(rect["bbox"]) for rect in info["rectangles"]]

    for r, c in np.argwhere(g == MARKER):
        out[int(r), int(c)] = BG
    for rect in info["rectangles"]:
        out[tuple(rect["coords"].T)] = BG

    final_boxes: list[tuple[int, int, int, int]] = boxes[:]
    debug_rects: list[dict] = []
    for idx, rect in enumerate(info["rectangles"]):
        desired = int(rect["sprite_count"])
        boundary = _boundary_allowance(g.shape, boxes[idx], rect["direction"])
        others = [final_boxes[j] for j in range(len(final_boxes)) if j != idx]
        no_overlap = _overlap_allowance(boxes[idx], rect["direction"], others)
        dist = max(0, min(desired, boundary, no_overlap))
        final_box = _extend_box(boxes[idx], rect["direction"], dist)
        for j, other in enumerate(final_boxes):
            if j == idx:
                continue
            assert not _box_intersects(final_box, other)
        final_boxes[idx] = final_box
        debug_rects.append(
            {
                "color": rect["color"],
                "bbox": boxes[idx],
                "final_bbox": final_box,
                "direction": rect["direction"],
                "marker": rect["marker"],
                "sprite_count": desired,
                "stretch": dist,
                "boundary_allowance": boundary,
                "overlap_allowance": no_overlap,
            }
        )

    for rect, final_box in zip(info["rectangles"], final_boxes):
        _draw_hollow_rect(out, final_box, rect["color"])

    if return_debug:
        return out, {"rectangles": debug_rects, "sprites_by_color": info["sprites_by_color"]}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each hollow rectangle by the count of same-colored sprites in the marker direction.

    Input structure:
    - Background is black, with 2-3 colored hollow rectangles whose border colors are all distinct.
    - Each rectangle has exactly one gray marker pixel outside it, aligned above, below, left, or right of that rectangle.
    - The remaining foreground objects are small monochrome 3x3 sprites; each sprite color matches one rectangle border color.

    Rule:
    - For each rectangle, count all 8-connected 3x3 sprites of the same color.
    - Use the rectangle's gray marker to choose the stretch direction: up, down, left, or right.
    - Move that one side outward by exactly the sprite count, clipping at the grid boundary and shrinking further if needed to avoid overlapping any rectangle's current occupied box, processing rectangles from left to right then top to bottom.
    - Remove every gray marker and redraw the resulting hollow rectangles; the small sprites stay unchanged.

    Key insight:
    - Color links two unrelated object types: same-colored sprites provide a numeric parameter, and the gray marker converts that count into a directed rectangle stretch.
    """
    return _solve(grid, return_debug=False)


def _all_sprite_masks() -> list[np.ndarray]:
    masks = []
    seen: set[bytes] = set()
    for bits in range(1 << 9):
        area = bits.bit_count()
        if not (4 <= area <= 6):
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for k in range(9):
            if (bits >> k) & 1:
                mask[k // 3, k % 3] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        comps = _components_mask(mask, connectivity=8)
        if len(comps) != 1:
            continue
        key = mask.tobytes()
        if key not in seen:
            seen.add(key)
            masks.append(mask)
    return masks


SPRITE_MASKS = _all_sprite_masks()


def _box_clear(blocked: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_box(blocked: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _marker_from_spec(spec: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = spec["bbox"]
    gap = spec["marker_gap"]
    if spec["direction"] == "up":
        return r0 - gap, c0 + spec["marker_offset"]
    if spec["direction"] == "down":
        return r1 + gap, c0 + spec["marker_offset"]
    if spec["direction"] == "left":
        return r0 + spec["marker_offset"], c0 - gap
    return r0 + spec["marker_offset"], c1 + gap


def _reserve_marker_line(blocked: np.ndarray, spec: dict) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = spec["bbox"]
    mr, mc = _marker_from_spec(spec)
    line_cells = []
    if spec["direction"] in ("up", "down"):
        step = -1 if spec["direction"] == "up" else 1
        for rr in range(r0 + step, mr + step, step):
            line_cells.append((rr, mc))
    else:
        step = -1 if spec["direction"] == "left" else 1
        row = mr
        for cc in range(c0 + step if step < 0 else c1 + step, mc + step, step):
            line_cells.append((row, cc))
    for rr, cc in line_cells:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if blocked[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)].any():
            return False
    for rr, cc in line_cells:
        blocked[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = True
    return True


def _reserve_spec(blocked: np.ndarray, spec: dict) -> bool:
    if not _reserve_marker_line(blocked, spec):
        return False
    _mark_box(blocked, spec["bbox"], pad=1)
    mr, mc = spec["marker"]
    blocked[max(0, mr - 1) : min(blocked.shape[0], mr + 2), max(0, mc - 1) : min(blocked.shape[1], mc + 2)] = True
    return True


def _simulate_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict]:
    ordered = sorted(specs, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    current = [tuple(spec["bbox"]) for spec in ordered]
    out: list[dict] = []
    for idx, spec in enumerate(ordered):
        boundary = _boundary_allowance(shape, current[idx], spec["direction"])
        others = [current[j] for j in range(len(current)) if j != idx]
        overlap = _overlap_allowance(current[idx], spec["direction"], others)
        dist = max(0, min(spec["count"], boundary, overlap))
        final_box = _extend_box(current[idx], spec["direction"], dist)
        current[idx] = final_box
        clone = dict(spec)
        clone["final_bbox"] = final_box
        clone["stretch"] = dist
        clone["boundary_allowance"] = boundary
        clone["overlap_allowance"] = overlap
        out.append(clone)
    return out


def _markers_are_unique(specs: list[dict]) -> bool:
    markers = [tuple(spec["marker"]) for spec in specs]
    boxes = [tuple(spec["bbox"]) for spec in specs]
    for box in boxes:
        if sum(_marker_direction(box, marker) is not None for marker in markers) != 1:
            return False
    for marker in markers:
        if sum(_marker_direction(box, marker) is not None for box in boxes) != 1:
            return False
    return True


def _try_random_rect(
    shape: tuple[int, int],
    blocked: np.ndarray,
    existing_specs: list[dict],
    color: int,
    count: int,
) -> dict | None:
    h, w = shape
    for _ in range(200):
        rh = random.randint(4, 6)
        rw = random.randint(4, 6)
        direction = random.choice(DIRS)
        marker_gap = random.randint(3, 8)
        if direction == "up":
            r0 = random.randint(marker_gap, h - rh - 1)
            c0 = random.randint(0, w - rw)
            marker_offset = random.randint(0, rw - 1)
        elif direction == "down":
            r0 = random.randint(0, h - rh - marker_gap - 1)
            c0 = random.randint(0, w - rw)
            marker_offset = random.randint(0, rw - 1)
        elif direction == "left":
            r0 = random.randint(0, h - rh)
            c0 = random.randint(marker_gap, w - rw - 1)
            marker_offset = random.randint(0, rh - 1)
        else:
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw - marker_gap - 1)
            marker_offset = random.randint(0, rh - 1)
        box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        spec = {
            "color": color,
            "bbox": box,
            "direction": direction,
            "count": count,
            "marker_gap": marker_gap,
            "marker_offset": marker_offset,
        }
        marker = _marker_from_spec(spec)
        if not _box_clear(blocked, box, pad=1):
            continue
        if not _box_clear(blocked, (marker[0], marker[1], marker[0], marker[1]), pad=1):
            continue
        shadow = blocked.copy()
        if not _reserve_marker_line(shadow, spec):
            continue
        spec["marker"] = marker
        if not _markers_are_unique(existing_specs + [spec]):
            continue
        return spec
    return None


def _place_overlap_pair(shape: tuple[int, int], blocked: np.ndarray, colors: list[int]) -> list[dict] | None:
    h, w = shape
    for _ in range(200):
        ah = random.randint(4, 6)
        aw = random.randint(4, 6)
        bh = random.randint(4, 6)
        bw = random.randint(4, 6)
        overlap_rows = random.randint(1, min(2, ah - 2, bh - 1))
        ra = random.randint(0, h - max(ah, bh + ah - overlap_rows))
        rb = ra + ah - overlap_rows
        if rb + bh > h:
            continue
        ca_max = w - (aw + 3 + bw)
        if ca_max < 0:
            continue
        ca = random.randint(0, ca_max)
        cb = ca + aw + 3
        spec_a = {
            "color": colors[0],
            "bbox": (ra, ca, ra + ah - 1, ca + aw - 1),
            "direction": "right",
            "count": 4,
            "marker_gap": 3,
            "marker_offset": 0,
        }
        spec_b = {
            "color": colors[1],
            "bbox": (rb, cb, rb + bh - 1, cb + bw - 1),
            "direction": random.choice(["up", "down", "right"]),
            "count": random.randint(1, 4),
            "marker_gap": random.randint(3, 6),
            "marker_offset": 0,
        }
        # Keep the pair overlapping in rows but keep A's marker row unique to A.
        spec_b["marker_offset"] = 0 if spec_b["direction"] in ("left", "right") else random.randint(0, bw - 1)
        if spec_b["direction"] == "up":
            if rb < spec_b["marker_gap"]:
                continue
        elif spec_b["direction"] == "down":
            if h - 1 - (rb + bh - 1) < spec_b["marker_gap"]:
                continue
        elif spec_b["direction"] == "right":
            if w - 1 - (cb + bw - 1) < spec_b["marker_gap"]:
                continue
            spec_b["marker_offset"] = random.randint(0, bh - 1)
        boxes = [spec_a["bbox"], spec_b["bbox"]]
        if any(not _box_clear(blocked, box, pad=1) for box in boxes):
            continue
        ma = _marker_from_spec(spec_a)
        mb = _marker_from_spec(spec_b)
        if not _box_clear(blocked, (ma[0], ma[1], ma[0], ma[1]), pad=1):
            continue
        if not _box_clear(blocked, (mb[0], mb[1], mb[0], mb[1]), pad=1):
            continue
        shadow = blocked.copy()
        ok = True
        for spec in (spec_a, spec_b):
            if not _reserve_marker_line(shadow, spec):
                ok = False
                break
        if not ok:
            continue
        spec_a["marker"] = ma
        spec_b["marker"] = mb
        if not _markers_are_unique([spec_a, spec_b]):
            continue
        sim = _simulate_specs(shape, [spec_a, spec_b])
        if sim[0]["stretch"] != 3:
            continue
        if not (sim[0]["stretch"] < min(sim[0]["count"], sim[0]["boundary_allowance"])):
            continue
        return [spec_a, spec_b]
    return None


def _place_edge_rect(
    shape: tuple[int, int],
    blocked: np.ndarray,
    existing_specs: list[dict],
    color: int,
) -> dict | None:
    h, w = shape
    for direction in random.sample(DIRS, k=4):
        for _ in range(100):
            rh = random.randint(4, 6)
            rw = random.randint(4, 6)
            if direction == "up":
                if h - rh - 1 < 3:
                    continue
                r0 = 3
                c0 = random.randint(0, w - rw)
                marker_offset = random.randint(0, rw - 1)
            elif direction == "down":
                if h - rh - 1 < 3:
                    continue
                r0 = h - rh - 3
                c0 = random.randint(0, w - rw)
                marker_offset = random.randint(0, rw - 1)
            elif direction == "left":
                if w - rw - 1 < 3:
                    continue
                r0 = random.randint(0, h - rh)
                c0 = 3
                marker_offset = random.randint(0, rh - 1)
            else:
                if w - rw - 1 < 3:
                    continue
                r0 = random.randint(0, h - rh)
                c0 = w - rw - 3
                marker_offset = random.randint(0, rh - 1)
            spec = {
                "color": color,
                "bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1),
                "direction": direction,
                "count": 4,
                "marker_gap": 3,
                "marker_offset": marker_offset,
            }
            marker = _marker_from_spec(spec)
            if not _box_clear(blocked, spec["bbox"], pad=1):
                continue
            if not _box_clear(blocked, (marker[0], marker[1], marker[0], marker[1]), pad=1):
                continue
            shadow = blocked.copy()
            if not _reserve_marker_line(shadow, spec):
                continue
            spec["marker"] = marker
            if not _markers_are_unique(existing_specs + [spec]):
                continue
            return spec
    return None


def _render_input(shape: tuple[int, int], specs: list[dict], sprite_masks: dict[int, list[np.ndarray]]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    ordered_specs = sorted(specs, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for spec in ordered_specs:
        if not _reserve_marker_line(blocked, spec):
            return None
    for spec in ordered_specs:
        _draw_hollow_rect(grid, spec["bbox"], spec["color"])
        _mark_box(blocked, spec["bbox"], pad=1)
    for spec in ordered_specs:
        mr, mc = spec["marker"]
        grid[mr, mc] = MARKER
        blocked[max(0, mr - 1) : min(h, mr + 2), max(0, mc - 1) : min(w, mc + 2)] = True

    for color, masks in sprite_masks.items():
        for mask in masks:
            candidates = []
            for r0 in range(h - 2):
                for c0 in range(w - 2):
                    box = (r0, c0, r0 + 2, c0 + 2)
                    if not _box_clear(blocked, box, pad=1):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                return None
            r0, c0 = random.choice(candidates)
            grid[r0 : r0 + 3, c0 : c0 + 3][mask] = color
            _mark_box(blocked, (r0, c0, r0 + 2, c0 + 2), pad=1)
    return grid


def _random_sprite_multiset(count: int) -> list[np.ndarray]:
    chosen = []
    pool = SPRITE_MASKS[:]
    random.shuffle(pool)
    for i in range(count):
        if i == 0 or random.random() < 0.6:
            chosen.append(pool.pop().copy())
        else:
            chosen.append(random.choice(chosen).copy())
    random.shuffle(chosen)
    return chosen


def _validate_semantics(inp: np.ndarray, specs: list[dict], force_overlap: bool, force_edge: bool) -> bool:
    out, debug = _solve(inp, return_debug=True)
    expected = _simulate_specs(inp.shape, specs)
    got = {item["color"]: item for item in debug["rectangles"]}
    want = {item["color"]: item for item in expected}
    if set(got) != set(want):
        return False
    overlap_hit = False
    edge_hit = False
    for color, spec in want.items():
        rect = got[color]
        if rect["direction"] != spec["direction"]:
            return False
        if tuple(rect["marker"]) != tuple(_marker_from_spec(spec)):
            return False
        if rect["sprite_count"] != spec["count"]:
            return False
        if tuple(rect["final_bbox"]) != tuple(spec["final_bbox"]):
            return False
        if rect["stretch"] != spec["stretch"]:
            return False
        if rect["stretch"] < min(spec["count"], spec["boundary_allowance"]):
            overlap_hit = True
        if spec["boundary_allowance"] < spec["count"] and rect["stretch"] == spec["boundary_allowance"]:
            edge_hit = True

    if force_overlap and not overlap_hit:
        return False
    if force_edge and not edge_hit:
        return False

    if np.any(inp == MARKER) and np.any(out == MARKER):
        return False
    for spec in expected:
        r0, c0, r1, c1 = spec["final_bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        expected_sub = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        expected_sub[_rect_mask(expected_sub.shape[0], expected_sub.shape[1])] = spec["color"]
        if not np.array_equal(sub, expected_sub):
            return False
    return not np.array_equal(inp, out)


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(18, 24), random.randint(18, 24))
    force_overlap = random.random() < 0.35
    force_edge = random.random() < 0.35
    n_rects = 3 if (force_overlap or force_edge or random.random() < 0.5) else 2
    colors = random.sample(PALETTE, n_rects)

    counts = [random.randint(1, 4) for _ in range(n_rects)]
    if len(set(counts)) == 1:
        counts[random.randrange(n_rects)] = 1 + (counts[0] % 4)

    blocked = np.zeros(shape, dtype=bool)
    specs: list[dict] = []
    remaining_colors = colors[:]
    remaining_counts = counts[:]

    if force_overlap:
        pair = _place_overlap_pair(shape, blocked, remaining_colors[:2])
        if pair is None:
            return None
        for spec in pair:
            if not _reserve_spec(blocked, spec):
                return None
        specs.extend(pair)
        remaining_colors = remaining_colors[2:]
        remaining_counts = remaining_counts[2:]

    if force_edge and remaining_colors:
        spec = _place_edge_rect(shape, blocked, specs, remaining_colors[0])
        if spec is None:
            return None
        spec["count"] = 4
        if not _reserve_spec(blocked, spec):
            return None
        specs.append(spec)
        remaining_colors = remaining_colors[1:]
        remaining_counts = remaining_counts[1:]

    for color, count in zip(remaining_colors, remaining_counts):
        spec = _try_random_rect(shape, blocked, specs, color, count)
        if spec is None:
            return None
        if not _reserve_spec(blocked, spec):
            return None
        specs.append(spec)

    if len(specs) != n_rects:
        return None
    if not _markers_are_unique(specs):
        return None

    by_color = {spec["color"]: spec for spec in specs}
    for color, count in zip(colors, counts):
        if color in by_color and "count" in by_color[color]:
            by_color[color]["count"] = by_color[color].get("count", count)
    if len(set(spec["count"] for spec in specs)) == 1:
        return None

    sprite_masks = {spec["color"]: _random_sprite_multiset(spec["count"]) for spec in specs}
    grid = _render_input(shape, specs, sprite_masks)
    if grid is None:
        return None
    if not _validate_semantics(grid, specs, force_overlap, force_edge):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

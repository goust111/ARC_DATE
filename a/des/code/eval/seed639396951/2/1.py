import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rect_frame(grid: np.ndarray, color: int, comp: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if comp.shape[0] != expected:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    perim = np.zeros((h, w), dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    return bool(np.all(sub[perim] == color) and np.all(sub[~perim] != color))


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    best_color = None
    best_frames: list[dict] = []
    best_score = (-1, -1)
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        frames = []
        total_cells = 0
        for comp in _components(grid == color):
            if not _is_rect_frame(grid, color, comp):
                continue
            frames.append({"bbox": _bbox(comp), "color": color})
            total_cells += int(comp.shape[0])
        if len(frames) < 2:
            continue
        score = (len(frames), total_cells)
        if score > best_score:
            best_score = score
            best_color = color
            best_frames = frames
    if best_color is None:
        raise ValueError("failed to identify rectangular frame color")
    best_frames.sort(key=lambda item: item["bbox"])
    return int(best_color), best_frames


def _interior_items(grid: np.ndarray, frame: dict, bg: int, frame_color: int) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    sub = grid[r0 + 1 : r1, c0 + 1 : c1]
    items = []
    for dr, dc in np.argwhere((sub != bg) & (sub != frame_color)):
        rr = r0 + 1 + int(dr)
        cc = c0 + 1 + int(dc)
        items.append((rr, cc, int(grid[rr, cc])))
    return items


def _pattern_from_items(items: list[tuple[int, int, int]]) -> dict:
    rows = [r for r, _, _ in items]
    cols = [c for _, c, _ in items]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    pixels = [(r - r0, c - c0, color) for r, c, color in items]
    return {"shape": (r1 - r0 + 1, c1 - c0 + 1), "pixels": pixels}


def _collect_side_markers(
    grid: np.ndarray, source_bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> dict[int, str]:
    h, w = grid.shape
    r0, c0, r1, c1 = source_bbox
    markers: dict[int, str] = {}

    def scan(side: str, cells: list[tuple[int, int]]) -> None:
        found = []
        for rr, cc in cells:
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            val = int(grid[rr, cc])
            if val == bg or val == frame_color:
                continue
            found.append((val, rr, cc))
        if len(found) > 1:
            raise ValueError(f"multiple side markers on {side}")
        if found:
            color = found[0][0]
            if color in markers:
                raise ValueError("duplicate marker colors on different sides")
            markers[color] = side

    scan("top", [(r0 - 1, c) for c in range(c0, c1 + 1)])
    scan("bottom", [(r1 + 1, c) for c in range(c0, c1 + 1)])
    scan("left", [(r, c0 - 1) for r in range(r0, r1 + 1)])
    scan("right", [(r, c1 + 1) for r in range(r0, r1 + 1)])
    return markers


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frames = _find_frames(g, bg)

    source_frame = None
    source_items = None
    anchors: dict[int, dict] = {}
    for frame in frames:
        items = _interior_items(g, frame, bg, frame_color)
        if len(items) > 1:
            if source_frame is not None:
                raise ValueError("multiple source frames detected")
            source_frame = frame
            source_items = items
        elif len(items) == 1:
            rr, cc, color = items[0]
            if color in anchors:
                raise ValueError("anchor colors must be unique")
            anchors[color] = {"frame": frame, "anchor": (rr, cc)}

    if source_frame is None or source_items is None:
        raise ValueError("missing source frame")
    if not anchors:
        raise ValueError("missing anchor frames")

    markers = _collect_side_markers(g, source_frame["bbox"], bg, frame_color)
    if not markers:
        raise ValueError("missing side markers")

    pattern = _pattern_from_items(source_items)
    return {
        "bg": bg,
        "frame_color": frame_color,
        "frames": frames,
        "source": source_frame,
        "anchors": anchors,
        "markers": markers,
        "pattern": pattern,
    }


def _stamp_origin(anchor: tuple[int, int], shape: tuple[int, int], side: str) -> tuple[int, int]:
    ar, ac = anchor
    ph, pw = shape
    center_r = (ph - 1) // 2
    center_c = (pw - 1) // 2
    if side == "top":
        return ar, ac - center_c
    if side == "bottom":
        return ar - (ph - 1), ac - center_c
    if side == "left":
        return ar - center_r, ac
    if side == "right":
        return ar - center_r, ac - (pw - 1)
    raise ValueError(f"unknown side {side}")


def _stamp_into_frame(
    out: np.ndarray,
    pattern: dict,
    target_bbox: tuple[int, int, int, int],
    anchor: tuple[int, int],
    side: str,
) -> None:
    r0, c0, r1, c1 = target_bbox
    start_r, start_c = _stamp_origin(anchor, pattern["shape"], side)
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    for pr, pc, color in pattern["pixels"]:
        rr = start_r + pr
        cc = start_c + pc
        if inner_r0 <= rr <= inner_r1 and inner_c0 <= cc <= inner_c1:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the source stamp from the unique multi-pixel frame interior, then stamp
    it into anchor frames whose colors are referenced by source-side markers.

    The marker color selects the target frame. The marker side on the source frame
    selects how the stamp aligns to that target anchor:
    - top/bottom marker: matching stamp edge row aligns to the anchor row, stamp is
      horizontally centered on the anchor column.
    - left/right marker: matching stamp edge column aligns to the anchor column,
      stamp is vertically centered on the anchor row.

    Stamp pixels outside the target frame interior are clipped.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)

    out = g.copy()
    pattern = scene["pattern"]
    for color, side in scene["markers"].items():
        target = scene["anchors"].get(color)
        if target is None:
            continue
        _stamp_into_frame(out, pattern, target["frame"]["bbox"], target["anchor"], side)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_pattern(rng: random.Random, colors: list[int]) -> np.ndarray:
    for _ in range(80):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        target_area = rng.randint(max(3, (h * w) // 3), max(3, h * w - 1))
        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        cells = [seed]
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True
            cells.append((nr, nc))
        if mask.sum() < 3:
            continue
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue

        pattern = np.full(mask.shape, -1, dtype=int)
        coords = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        if len(colors) == 1 or rng.random() < 0.45:
            pattern[mask] = colors[0]
            return pattern

        axis = rng.choice([0, 1])
        values = sorted({pt[axis] for pt in coords})
        if len(values) < 2:
            pattern[mask] = colors[0]
            return pattern
        cut = rng.choice(values[:-1])
        accent = {pt for pt in coords if pt[axis] > cut}
        if not accent or len(accent) == len(coords):
            ordered = sorted(coords)
            accent = set(ordered[len(ordered) // 2 :])
        for pt in coords:
            pattern[pt] = colors[1] if pt in accent else colors[0]
        return pattern
    raise RuntimeError("failed to build a pattern")


def _fits_with_padding(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int
) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_rectangles(
    rng: random.Random, shape: tuple[int, int], specs: list[dict]
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda idx: (0 if specs[idx]["kind"] == "source" else 1, -(specs[idx]["h"] * specs[idx]["w"])),
    )
    for idx in order:
        spec = specs[idx]
        rh, rw, pad = spec["h"], spec["w"], spec["pad"]
        if rh > h - 2 or rw > w - 2:
            return None
        placed = False
        for _ in range(300):
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            if not _fits_with_padding(occupied, r0, c0, rh, rw, pad):
                continue
            _mark_occupied(occupied, r0, c0, rh, rw, pad)
            placements[idx] = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            placed = True
            break
        if not placed:
            return None
    return [item for item in placements if item is not None]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _side_strip_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 - 1, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1 + 1) for r in range(r0, r1 + 1)]
    raise ValueError(f"unknown side {side}")


def _full_fit_positions(inner_h: int, inner_w: int, pattern: np.ndarray, side: str) -> tuple[list, list]:
    ph, pw = pattern.shape
    fit = []
    crop = []
    for ar in range(inner_h):
        for ac in range(inner_w):
            sr, sc = _stamp_origin((ar, ac), (ph, pw), side)
            inside = sr >= 0 and sc >= 0 and sr + ph <= inner_h and sc + pw <= inner_w
            (fit if inside else crop).append((ar, ac))
    return fit, crop


def _pick_anchor(
    rng: random.Random, inner_h: int, inner_w: int, pattern: np.ndarray, side: str | None, want_crop: bool
) -> tuple[int, int]:
    if side is None:
        return rng.randrange(inner_h), rng.randrange(inner_w)
    fit, crop = _full_fit_positions(inner_h, inner_w, pattern, side)
    if want_crop and crop:
        return rng.choice(crop)
    if not want_crop and fit:
        return rng.choice(fit)
    all_pos = fit + crop
    return rng.choice(all_pos)


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    bg_count = counts.pop(bg)
    return all(bg_count > count for count in counts.values())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    used_count = rng.randint(2, 4)
    extra_targets = rng.randint(0, 1)
    total_targets = used_count + extra_targets

    bg = rng.randrange(10)
    remaining = [c for c in range(10) if c != bg]
    frame_color = rng.choice(remaining)
    remaining.remove(frame_color)

    anchor_colors = rng.sample(remaining, total_targets)
    remaining = [c for c in remaining if c not in anchor_colors]
    pattern_color_count = 1 if rng.random() < 0.45 else 2
    pattern_colors = rng.sample(remaining, pattern_color_count)
    pattern = _random_pattern(rng, pattern_colors)
    ph, pw = pattern.shape

    margins = [rng.randint(1, 2) for _ in range(4)]
    src_inner_h = ph + margins[0] + margins[1]
    src_inner_w = pw + margins[2] + margins[3]
    source_spec = {"kind": "source", "h": src_inner_h + 2, "w": src_inner_w + 2, "pad": 2}

    target_specs = []
    for idx in range(total_targets):
        inner_h = rng.randint(max(3, ph - 2), ph + 3)
        inner_w = rng.randint(max(3, pw - 2), pw + 3)
        if idx == 0 and rng.random() < 0.6:
            inner_h = rng.randint(3, max(3, ph))
        if idx == 1 and rng.random() < 0.6:
            inner_w = rng.randint(3, max(3, pw))
        target_specs.append({"kind": "target", "h": inner_h + 2, "w": inner_w + 2, "pad": 1})

    frame_cells = (2 * source_spec["h"] + 2 * source_spec["w"] - 4) + sum(
        2 * spec["h"] + 2 * spec["w"] - 4 for spec in target_specs
    )
    other_cells = int((pattern != -1).sum()) + total_targets + used_count
    min_area = max(frame_cells * 2 + other_cells + 40, 22 * 22)
    base = int(np.ceil(np.sqrt(min_area)))
    grid_h = base + rng.randint(0, 5)
    grid_w = base + rng.randint(0, 5)
    if rng.random() < 0.5:
        grid_h += rng.randint(0, 3)
    else:
        grid_w += rng.randint(0, 3)

    specs = [source_spec] + target_specs
    placements = _place_rectangles(rng, (grid_h, grid_w), specs)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for bbox in placements:
        _draw_frame(grid, bbox, frame_color)

    source_bbox = placements[0]
    sr0, sc0, sr1, sc1 = source_bbox
    src_row = sr0 + 1 + rng.randint(0, src_inner_h - ph)
    src_col = sc0 + 1 + rng.randint(0, src_inner_w - pw)
    for pr, pc in np.argwhere(pattern != -1):
        grid[src_row + pr, src_col + pc] = int(pattern[pr, pc])

    chosen_sides = rng.sample(list(SIDES), used_count)
    rng.shuffle(anchor_colors)
    side_by_color = {anchor_colors[idx]: chosen_sides[idx] for idx in range(used_count)}

    for idx, bbox in enumerate(placements[1:]):
        r0, c0, r1, c1 = bbox
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        color = anchor_colors[idx]
        side = side_by_color.get(color)
        want_crop = idx % 2 == 0
        ar, ac = _pick_anchor(rng, inner_h, inner_w, pattern, side, want_crop)
        grid[r0 + 1 + ar, c0 + 1 + ac] = color

    for color, side in side_by_color.items():
        cells = _side_strip_cells(source_bbox, side)
        rr, cc = rng.choice(cells)
        grid[rr, cc] = color

    if not _bg_is_dominant(grid, bg):
        return None

    try:
        scene = _parse_scene(grid)
        if len(scene["anchors"]) != total_targets:
            return None
        if len(scene["markers"]) != used_count:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")

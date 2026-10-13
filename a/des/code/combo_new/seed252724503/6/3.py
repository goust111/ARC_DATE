from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GRAY = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
SLOT_COUNT = 3
SLOT_OUTER = 9
SLOT_INNER = SLOT_OUTER - 2
SLOT_STEP = SLOT_OUTER + 1
LOWER_HEIGHT = SLOT_OUTER + 2
MAX_SIZE = 30


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask[:0, :0]
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _count_holes(mask: np.ndarray) -> int:
    bg_mask = ~mask
    holes = 0
    for comp in _components(bg_mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == mask.shape[0] - 1 or c1 == mask.shape[1] - 1:
            continue
        holes += 1
    return holes


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    actual = {(int(r), int(c)) for r, c in coords}
    return actual == expected


def _extract_slot_component(coords: np.ndarray) -> tuple[dict[str, int], int] | None:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = sorted({int(r) for r, _c in pts})
    r0 = rows[0]
    top_cols = sorted(int(c) for r, c in pts if int(r) == r0)
    if not top_cols or top_cols != list(range(top_cols[0], top_cols[-1] + 1)) or len(top_cols) < 3:
        return None
    c0, c1 = top_cols[0], top_cols[-1]

    r1 = None
    for candidate in rows[2:]:
        if not all((candidate, c) in pts for c in range(c0, c1 + 1)):
            continue
        if not all((r, c0) in pts and (r, c1) in pts for r in range(r0, candidate + 1)):
            continue
        r1 = int(candidate)
    if r1 is None:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    if not border.issubset(pts):
        return None
    if any((r, c) in pts for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
        return None

    extras = [(r, c) for r, c in pts if (r, c) not in border]
    if any(r <= r1 or c < c0 or c > c1 or r > r1 + 3 for r, c in extras):
        return None
    if not extras:
        return {"r0": r0, "c0": c0, "r1": r1, "c1": c1}, 0

    extra_r0 = min(r for r, _c in extras)
    extra_c0 = min(c for _r, c in extras)
    extra_r1 = max(r for r, _c in extras)
    extra_c1 = max(c for _r, c in extras)
    extra_mask = np.zeros((extra_r1 - extra_r0 + 1, extra_c1 - extra_c0 + 1), dtype=bool)
    for r, c in extras:
        extra_mask[r - extra_r0, c - extra_c0] = True
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1}, len(_components(extra_mask))


def _resize_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if mask.shape == (out_h, out_w):
        return mask.copy()
    src_h, src_w = mask.shape
    row_idx = np.clip(
        np.round((np.arange(out_h) + 0.5) * src_h / out_h - 0.5).astype(int),
        0,
        src_h - 1,
    )
    col_idx = np.clip(
        np.round((np.arange(out_w) + 0.5) * src_w / out_w - 0.5).astype(int),
        0,
        src_w - 1,
    )
    return mask[row_idx[:, None], col_idx[None, :]]


def _fit_thumbnail(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = mask.shape
    scale = min(target_h / src_h, target_w / src_w)
    out_h = max(1, min(target_h, int(round(src_h * scale))))
    out_w = max(1, min(target_w, int(round(src_w * scale))))
    scaled = _resize_mask(mask, out_h, out_w)
    canvas = np.zeros((target_h, target_w), dtype=bool)
    r0 = (target_h - out_h) // 2
    c0 = (target_w - out_w) // 2
    canvas[r0 : r0 + out_h, c0 : c0 + out_w] = scaled
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each upper object into the lower frame whose dot-count matches that object's hole count.

    Input structure:
    - Background is black, and a full-width grey horizontal line splits the grid into an upper object area and a lower slot area.
    - The upper area contains several disjoint single-color 4-connected objects; each object may enclose 0, 1, 2, or 3 black holes.
    - The lower area contains empty grey rectangular frames, each labeled by the number of grey dots directly underneath it.

    Rule:
    - For every upper object, count the number of enclosed black background regions fully surrounded by that object.
    - Find the lower frame whose dot-count equals that hole count.
    - Shrink or enlarge the object's cropped shape to fit inside that frame while preserving aspect ratio, and draw the thumbnail in the object's original color.
    - If several objects have the same hole count, preserve their left-to-right order when filling the corresponding equally labeled frames.

    Key insight:
    - The deciding feature is not color or size but topology: the grey dots encode a slot number, and that number is exactly the object's count of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    separator_rows = [r for r in range(h) if len(set(map(int, g[r, :]))) == 1 and int(g[r, 0]) != BG]
    assert separator_rows, "expected a solid non-background separator row"
    sep_r = int(separator_rows[0])
    gray = int(g[sep_r, 0])

    upper = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]

    object_comps = _components((upper != BG) & (upper != gray))
    objects_by_holes: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in object_comps:
        colors = np.unique(upper[comp[:, 0], comp[:, 1]])
        assert len(colors) == 1, "upper objects must be monochrome"
        color = int(colors[0])
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        holes = _count_holes(mask)
        center_x = float(comp[:, 1].mean())
        objects_by_holes[holes].append({"color": color, "mask": mask, "x": center_x})

    for hole_count in objects_by_holes:
        objects_by_holes[hole_count].sort(key=lambda item: float(item["x"]))

    gray_comps = _components(lower == gray)
    frames: list[dict[str, int]] = []
    dot_comps: list[np.ndarray] = []
    for comp in gray_comps:
        global_comp = comp.copy()
        global_comp[:, 0] += sep_r + 1
        parsed = _extract_slot_component(global_comp)
        if parsed is not None:
            frame, label = parsed
            frame["label"] = label
            frames.append(frame)
        elif _is_rect_frame(global_comp):
            r0, c0, r1, c1 = _bbox(global_comp)
            frames.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "label": 0})
        else:
            dot_comps.append(global_comp)

    frames.sort(key=lambda fr: fr["c0"])
    frame_dots: list[list[np.ndarray]] = [[] for _ in frames]
    for comp in dot_comps:
        cx = float(comp[:, 1].mean())
        cy = float(comp[:, 0].mean())
        candidates = []
        for idx, frame in enumerate(frames):
            if frame["r1"] < cy <= frame["r1"] + 3 and frame["c0"] <= cx <= frame["c1"]:
                candidates.append((abs(cy - frame["r1"]), abs(cx - (frame["c0"] + frame["c1"]) / 2), idx))
        if not candidates:
            continue
        _, _, best_idx = min(candidates)
        frame_dots[best_idx].append(comp)

    slots_by_holes: dict[int, list[dict[str, int]]] = defaultdict(list)
    for frame, dots in zip(frames, frame_dots):
        slots_by_holes[int(frame["label"]) + len(dots)].append(frame)

    out = g.copy()
    for hole_count in range(4):
        objs = objects_by_holes.get(hole_count, [])
        slots = sorted(slots_by_holes.get(hole_count, []), key=lambda fr: fr["c0"])
        assert len(objs) == len(slots), f"mismatched count for hole class {hole_count}"
        for obj, slot in zip(objs, slots):
            thumb = _fit_thumbnail(np.asarray(obj["mask"], dtype=bool), slot["r1"] - slot["r0"] - 1, slot["c1"] - slot["c0"] - 1)
            rr0 = slot["r0"] + 1
            cc0 = slot["c0"] + 1
            out[rr0 : rr0 + thumb.shape[0], cc0 : cc0 + thumb.shape[1]][thumb] = int(obj["color"])

    return out


def _decorate_mask(mask: np.ndarray, rng: random.Random, max_tabs: int = 3) -> np.ndarray:
    pad = 2
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    for _ in range(rng.randint(0, max_tabs)):
        coords = np.argwhere(canvas)
        r0, c0, r1, c1 = _bbox(coords)
        options = []
        if r0 > 0:
            options.append("top")
        if r1 < canvas.shape[0] - 1:
            options.append("bottom")
        if c0 > 0:
            options.append("left")
        if c1 < canvas.shape[1] - 1:
            options.append("right")
        if not options:
            break
        side = rng.choice(options)
        if side in ("top", "bottom"):
            length = rng.randint(1, min(3, c1 - c0 + 1))
            start = rng.randint(c0, c1 - length + 1)
            row = r0 - 1 if side == "top" else r1 + 1
            canvas[row, start : start + length] = True
        else:
            length = rng.randint(1, min(3, r1 - r0 + 1))
            start = rng.randint(r0, r1 - length + 1)
            col = c0 - 1 if side == "left" else c1 + 1
            canvas[start : start + length, col] = True
    for _ in range(rng.randint(0, 3)):
        k = rng.randint(0, 3)
        rotated = np.rot90(canvas, k)
        if max(rotated.shape) <= 14:
            canvas = rotated
    if rng.random() < 0.5:
        canvas = np.fliplr(canvas)
    if rng.random() < 0.5:
        canvas = np.flipud(canvas)
    return _crop_mask(canvas)


def _make_holey_mask(holes: int, rng: random.Random) -> np.ndarray:
    assert 1 <= holes <= 3
    if holes == 1:
        base = np.ones((3, 3), dtype=bool)
        base[1, 1] = False
    elif rng.random() < 0.5:
        base = np.ones((3, 2 * holes + 1), dtype=bool)
        base[1, 1::2] = False
    else:
        base = np.ones((2 * holes + 1, 3), dtype=bool)
        base[1::2, 1] = False
    return _decorate_mask(base, rng, max_tabs=3)


def _make_zero_hole_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        mask = np.zeros((h, w), dtype=bool)
        sr = rng.randrange(h)
        sc = rng.randrange(w)
        mask[sr, sc] = True
        target = rng.randint(max(7, (h * w) // 3), max(8, (2 * h * w) // 3))
        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True
            if rng.random() < 0.2:
                extra_frontier = [
                    (er, ec)
                    for er, ec in _neighbors4(nr, nc)
                    if 0 <= er < h and 0 <= ec < w and not mask[er, ec]
                ]
                if extra_frontier:
                    er, ec = rng.choice(extra_frontier)
                    mask[er, ec] = True
        mask = _crop_mask(mask)
        if mask.size == 0 or int(mask.sum()) < 7 or min(mask.shape) < 2:
            continue
        if len(_components(mask)) != 1 or _count_holes(mask) != 0:
            continue
        return _decorate_mask(mask, rng, max_tabs=2)
    raise RuntimeError("failed to build zero-hole mask")


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return mask.copy()
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _random_gap_allocation(extra: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted([0, extra] + [rng.randint(0, extra) for _ in range(parts - 1)])
    return [cuts[i + 1] - cuts[i] for i in range(parts)]


def _make_slot_labels(rng: random.Random) -> list[int]:
    if rng.random() < 0.5:
        return rng.sample(range(4), 3)
    repeated = rng.randrange(4)
    other = rng.choice([v for v in range(4) if v != repeated])
    labels = [repeated, repeated, other]
    rng.shuffle(labels)
    return labels


def _draw_frame(grid: np.ndarray, top: int, left: int) -> None:
    grid[top, left : left + SLOT_OUTER] = GRAY
    grid[top + SLOT_OUTER - 1, left : left + SLOT_OUTER] = GRAY
    grid[top : top + SLOT_OUTER, left] = GRAY
    grid[top : top + SLOT_OUTER, left + SLOT_OUTER - 1] = GRAY


def _dot_columns(left: int, count: int) -> list[int]:
    center = left + SLOT_OUTER // 2
    if count == 0:
        return []
    if count == 1:
        return [center]
    if count == 2:
        return [center - 1, center + 1]
    return [center - 2, center, center + 2]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.choice([29, 30])
    slot_strip = SLOT_COUNT * SLOT_OUTER + (SLOT_COUNT - 1)
    slot_left = rng.randint(0, width - slot_strip)

    labels = _make_slot_labels(rng)
    colors = rng.sample(OBJECT_COLORS, SLOT_COUNT)

    masks = []
    for holes in labels:
        if holes == 0:
            masks.append(_make_zero_hole_mask(rng))
        else:
            masks.append(_make_holey_mask(holes, rng))

    scales = [1, 1, 1]
    if rng.random() < 0.5:
        idx = rng.randrange(SLOT_COUNT)
        trial = scales.copy()
        trial[idx] = 2
        widths = [m.shape[1] * s for m, s in zip(masks, trial)]
        if sum(widths) + 2 * (SLOT_COUNT - 1) <= width:
            scales = trial

    scaled_masks = [_scale_mask(mask, scale) for mask, scale in zip(masks, scales)]
    obj_widths = [mask.shape[1] for mask in scaled_masks]
    obj_heights = [mask.shape[0] for mask in scaled_masks]
    max_obj_h = max(obj_heights)

    min_gap = 2
    total_min_width = sum(obj_widths) + min_gap * (SLOT_COUNT - 1)
    if total_min_width > width or max_obj_h > 18:
        return None
    upper_h = max_obj_h + rng.randint(0, 18 - max_obj_h)
    total_h = upper_h + 1 + LOWER_HEIGHT
    if total_h > MAX_SIZE:
        return None

    extras = _random_gap_allocation(width - total_min_width, SLOT_COUNT + 1, rng)
    x_positions = []
    cursor = extras[0]
    for idx, obj_w in enumerate(obj_widths):
        x_positions.append(cursor)
        cursor += obj_w
        if idx < SLOT_COUNT - 1:
            cursor += min_gap + extras[idx + 1]

    y_positions = [rng.randint(0, upper_h - obj_h) for obj_h in obj_heights]

    grid = np.full((total_h, width), BG, dtype=int)
    sep_r = upper_h
    grid[sep_r, :] = GRAY

    for mask, color, top, left in zip(scaled_masks, colors, y_positions, x_positions):
        h, w = mask.shape
        region = grid[top : top + h, left : left + w]
        if np.any(region[mask] != BG):
            return None
        region[mask] = color

    frame_top = sep_r + 1
    dot_row = frame_top + SLOT_OUTER
    sorted_slot_labels = sorted(labels)
    for idx, label in enumerate(sorted_slot_labels):
        left = slot_left + idx * SLOT_STEP
        _draw_frame(grid, frame_top, left)
        for dc in _dot_columns(left, label):
            grid[dot_row, dc] = GRAY

    if len(_components((grid[:sep_r, :] != BG) & (grid[:sep_r, :] != GRAY))) != SLOT_COUNT:
        return None
    detected_holes = []
    for color in colors:
        comps = _components(grid[:sep_r, :] == color)
        if len(comps) != 1:
            return None
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        detected_holes.append(_count_holes(mask))
    detected_holes.sort()
    if detected_holes != sorted(labels):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

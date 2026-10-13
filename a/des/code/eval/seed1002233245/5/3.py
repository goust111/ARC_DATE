import random
from collections import Counter, deque

import numpy as np


BG = 8
SEP = 9
GREY = 5
COLOR_POOL = (0, 1, 2, 3, 4, 6, 7)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _resize_nearest(pattern: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = pattern.shape
    row_pos = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    col_pos = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    row_idx = np.clip(np.rint(row_pos), 0, in_h - 1).astype(int)
    col_idx = np.clip(np.rint(col_pos), 0, in_w - 1).astype(int)
    return pattern[row_idx[:, None], col_idx[None, :]]


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, f"expected unique vertical separator, got {candidates}"
    return candidates[0]


def _extract_templates(left: np.ndarray, bg: int, grey: int) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for color in sorted(int(v) for v in np.unique(left) if int(v) not in (bg, grey)):
        comps = _components(left == color)
        assert len(comps) == 1, f"expected one template component for color {color}"
        r0, c0, r1, c1 = _bbox(comps[0])
        if c1 + 1 < left.shape[1]:
            assert int(left[r0, c1 + 1]) == grey, "template should have a grey tag beside its top-right corner"
        sub = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in comps[0]:
            sub[r - r0, c - c0] = color
        templates[color] = sub
    assert templates, "no templates found"
    return templates


def _is_valid_frame(
    zone: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    border: int,
    bg: int,
    label_r: int,
    label_c: int,
) -> bool:
    if r0 < 0 or c0 < 0 or r1 >= zone.shape[0] or c1 >= zone.shape[1]:
        return False
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if (label_r, label_c) != (r0 + 1, c0 + 1):
        return False
    sub = zone[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == border):
        return False
    if not np.all(sub[-1, :] == border):
        return False
    if not np.all(sub[:, 0] == border):
        return False
    if not np.all(sub[:, -1] == border):
        return False
    inside = sub[1:-1, 1:-1].copy()
    inside[0, 0] = bg
    return bool(np.all(inside == bg))


def _frame_bbox_from_label(zone: np.ndarray, r: int, c: int, border: int, bg: int) -> tuple[int, int, int, int]:
    r0, c0 = r - 1, c - 1
    assert r0 >= 0 and c0 >= 0 and int(zone[r0, c0]) == border

    cand_rows = [rr for rr in range(r + 1, zone.shape[0]) if int(zone[rr, c0]) == border and int(zone[rr, c]) == border]
    cand_cols = [cc for cc in range(c + 1, zone.shape[1]) if int(zone[r0, cc]) == border and int(zone[r, cc]) == border]

    valid = []
    for r1 in cand_rows:
        for c1 in cand_cols:
            if _is_valid_frame(zone, r0, c0, r1, c1, border, bg, r, c):
                valid.append((r1, c1))
    assert len(valid) == 1, f"expected unique frame for label at {(r, c)}, got {valid}"
    r1, c1 = valid[0]
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the template area and sample area separated by a vertical line, then
    match each sample-frame tag color to the template of the same color and
    scale that template into the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = Counter(map(int, g.reshape(-1))).most_common(1)[0][0]
    sep_col, _sep_color = _find_separator(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    non_bg_right = [int(v) for v in np.unique(right) if int(v) != bg]
    assert non_bg_right, "sample zone must contain non-background colors"
    border = max(non_bg_right, key=lambda color: int(np.count_nonzero(right == color)))

    templates = _extract_templates(left, bg, border)

    out = right.copy()
    label_points = np.argwhere((right != bg) & (right != border))
    assert len(label_points) > 0, "no frame labels found"

    for r, c in label_points:
        color = int(right[r, c])
        assert color in templates, f"missing template for label color {color}"
        fr0, fc0, fr1, fc1 = _frame_bbox_from_label(right, int(r), int(c), border, bg)
        scaled = _resize_nearest(templates[color], fr1 - fr0 - 1, fc1 - fc0 - 1)
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = scaled

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _random_template_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    mask = np.ones((h, w), dtype=bool)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    max_remove = rng.randint(0, max(0, h * w // 3))
    removed = 0
    for r, c in cells:
        if removed >= max_remove:
            break
        trial = mask.copy()
        trial[r, c] = False
        if not np.any(trial):
            continue
        if np.any(trial.sum(axis=0) == 0) or np.any(trial.sum(axis=1) == 0):
            continue
        if not _mask_connected(trial):
            continue
        mask = trial
        removed += 1
    return mask


def _group_indices(n: int, groups: int, rng: random.Random) -> list[list[int]]:
    ids = list(range(n))
    rng.shuffle(ids)
    out = [[] for _ in range(groups)]
    for i, idx in enumerate(ids):
        out[i % groups].append(idx)
    for group in out:
        rng.shuffle(group)
    return out


def _build_left_zone(templates: list[dict], rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(1, min(3, len(templates)))
    groups = _group_indices(len(templates), row_count, rng)

    row_specs = []
    max_row_w = 0
    for group in groups:
        row_h = max(templates[idx]["pattern"].shape[0] for idx in group)
        gaps = [rng.randint(2, 4) for _ in range(max(0, len(group) - 1))]
        row_w = 0
        placements = []
        for pos, idx in enumerate(group):
            if pos > 0:
                row_w += gaps[pos - 1]
            pat = templates[idx]["pattern"]
            th, tw = pat.shape
            yoff = rng.randint(0, row_h - th)
            placements.append((idx, yoff, row_w))
            row_w += tw + 1
        row_specs.append((row_h, row_w, placements))
        max_row_w = max(max_row_w, row_w)

    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    left_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 3)
    row_gaps = [rng.randint(1, 3) for _ in range(max(0, len(row_specs) - 1))]

    zone_h = top_margin + bottom_margin + sum(spec[0] for spec in row_specs) + sum(row_gaps)
    zone_w = left_margin + right_margin + max_row_w
    if zone_w > 18:
        return None

    zone = np.full((zone_h, zone_w), BG, dtype=int)
    y = top_margin
    for row_idx, (row_h, row_w, placements) in enumerate(row_specs):
        x0 = left_margin + rng.randint(0, zone_w - left_margin - right_margin - row_w)
        for idx, yoff, xoff in placements:
            pattern = templates[idx]["pattern"]
            color = templates[idx]["color"]
            th, tw = pattern.shape
            r0 = y + yoff
            c0 = x0 + xoff
            zone[r0 : r0 + th, c0 : c0 + tw][pattern != BG] = color
            zone[r0, c0 + tw] = GREY
        y += row_h
        if row_idx < len(row_gaps):
            y += row_gaps[row_idx]
    return zone


def _build_right_zone(templates: list[dict], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = rng.randint(2, 5)
    frames = []
    used_colors = set()
    for _ in range(frame_count):
        template = rng.choice(templates)
        scale = rng.choice((1, 2, 3))
        ph, pw = template["pattern"].shape
        frames.append(
            {
                "color": template["color"],
                "pattern": template["pattern"],
                "ih": ph * scale,
                "iw": pw * scale,
                "fh": ph * scale + 2,
                "fw": pw * scale + 2,
            }
        )
        used_colors.add(template["color"])

    if len(used_colors) < min(2, len(templates)):
        frames[0]["color"] = templates[0]["color"]
        frames[0]["pattern"] = templates[0]["pattern"]
        if len(frames) > 1:
            frames[1]["color"] = templates[1]["color"]
            frames[1]["pattern"] = templates[1]["pattern"]

    if not any(frame["ih"] > frame["pattern"].shape[0] or frame["iw"] > frame["pattern"].shape[1] for frame in frames):
        frame = rng.choice(frames)
        ph, pw = frame["pattern"].shape
        frame["ih"] = ph * 2
        frame["iw"] = pw * 2
        frame["fh"] = frame["ih"] + 2
        frame["fw"] = frame["iw"] + 2

    band_count = rng.randint(1, min(3, frame_count))
    groups = _group_indices(frame_count, band_count, rng)

    band_specs = []
    max_band_w = 0
    for group in groups:
        band_h = max(frames[idx]["fh"] for idx in group)
        gaps = [rng.choice((0, 1, 2)) for _ in range(max(0, len(group) - 1))]
        band_w = 0
        placements = []
        for pos, idx in enumerate(group):
            if pos > 0:
                band_w += gaps[pos - 1]
            fh = frames[idx]["fh"]
            mode = rng.choice(("top", "bottom", "random"))
            if band_h == fh:
                yoff = 0
            elif mode == "top":
                yoff = 0
            elif mode == "bottom":
                yoff = band_h - fh
            else:
                yoff = rng.randint(0, band_h - fh)
            placements.append((idx, yoff, band_w))
            band_w += frames[idx]["fw"]
        band_specs.append((band_h, band_w, placements))
        max_band_w = max(max_band_w, band_w)

    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    band_gaps = [rng.randint(1, 3) for _ in range(max(0, len(band_specs) - 1))]
    extra_w = rng.randint(0, 3)

    zone_h = top_margin + bottom_margin + sum(spec[0] for spec in band_specs) + sum(band_gaps)
    zone_w = left_margin + right_margin + max_band_w + extra_w
    if zone_h > 24 or zone_w > 32:
        return None

    inp = np.full((zone_h, zone_w), BG, dtype=int)
    out = np.full((zone_h, zone_w), BG, dtype=int)
    y = top_margin
    for band_idx, (band_h, band_w, placements) in enumerate(band_specs):
        x0 = left_margin + rng.randint(0, zone_w - left_margin - right_margin - band_w)
        for idx, yoff, xoff in placements:
            frame = frames[idx]
            r0 = y + yoff
            c0 = x0 + xoff
            fh, fw = frame["fh"], frame["fw"]
            color = frame["color"]
            inp[r0, c0 : c0 + fw] = GREY
            inp[r0 + fh - 1, c0 : c0 + fw] = GREY
            inp[r0 : r0 + fh, c0] = GREY
            inp[r0 : r0 + fh, c0 + fw - 1] = GREY
            inp[r0 + 1, c0 + 1] = color

            out[r0 : r0 + fh, c0 : c0 + fw] = inp[r0 : r0 + fh, c0 : c0 + fw]
            out[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = _resize_nearest(
                frame["pattern"], frame["ih"], frame["iw"]
            )
        y += band_h
        if band_idx < len(band_gaps):
            y += band_gaps[band_idx]
    return inp, out


def _embed(zone: np.ndarray, target_h: int, rng: random.Random) -> tuple[np.ndarray, int]:
    top = rng.randint(0, target_h - zone.shape[0])
    canvas = np.full((target_h, zone.shape[1]), BG, dtype=int)
    canvas[top : top + zone.shape[0], :] = zone
    return canvas, top


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(2, 4)
    colors = list(COLOR_POOL)
    rng.shuffle(colors)

    templates = []
    for color in colors[:template_count]:
        mask = _random_template_mask(rng)
        pattern = np.full(mask.shape, BG, dtype=int)
        pattern[mask] = color
        templates.append({"color": color, "pattern": pattern})

    left_zone = _build_left_zone(templates, rng)
    right_pair = _build_right_zone(templates, rng)
    if left_zone is None or right_pair is None:
        return None
    right_zone, expected_right = right_pair

    total_h = max(left_zone.shape[0], right_zone.shape[0]) + rng.randint(0, 3)
    left_canvas, _left_top = _embed(left_zone, total_h, rng)
    right_canvas, _right_top = _embed(right_zone, total_h, rng)
    expected_canvas, _expected_top = _embed(expected_right, total_h, rng)

    # Keep sample-zone placement identical in input and output.
    if _right_top != _expected_top:
        expected_canvas = np.full((total_h, right_zone.shape[1]), BG, dtype=int)
        expected_canvas[_right_top : _right_top + expected_right.shape[0], :] = expected_right

    grid = np.full((total_h, left_canvas.shape[1] + 1 + right_canvas.shape[1]), BG, dtype=int)
    grid[:, : left_canvas.shape[1]] = left_canvas
    grid[:, left_canvas.shape[1]] = SEP
    grid[:, left_canvas.shape[1] + 1 :] = right_canvas

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected_canvas):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

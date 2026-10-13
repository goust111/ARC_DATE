from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


CANVAS_COLOR = 0
BAR_COLOR = 5
PAGE_BG_POOL = (8, 9)
PANEL_BG_POOL = (4, 6, 7)
SHAPE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_SLOT = 5
TOP_GUTTER = 5
LEFT_GUTTER = 5
PANEL_GAP = 2
PANEL_COLS = 2
PANEL_PAD = 1
SLOT_GAP = 1

MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize_shape(coords: np.ndarray) -> np.ndarray:
    mask = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return _crop_mask(mask)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return mask.shape, tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _d4_variants(mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    transforms = [
        ("id", lambda x: x),
        ("rot90", lambda x: np.rot90(x, 1)),
        ("rot180", lambda x: np.rot90(x, 2)),
        ("rot270", lambda x: np.rot90(x, 3)),
        ("flip_lr", lambda x: np.fliplr(x)),
        ("flip_ud", lambda x: np.flipud(x)),
        ("flip_diag", lambda x: x.T),
        ("flip_anti", lambda x: np.fliplr(np.rot90(x, 1))),
    ]
    seen = set()
    out = []
    for name, fn in transforms:
        variant = _crop_mask(fn(mask))
        key = _mask_key(variant)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, variant))
    return out


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _row_sig(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.sum(axis=1))


def _col_sig(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.sum(axis=0))


def _all_rows_cols_used(mask: np.ndarray) -> bool:
    return bool(mask.size) and bool(np.all(mask.any(axis=1))) and bool(np.all(mask.any(axis=0)))


def _shape_orbit_key(mask: np.ndarray) -> tuple:
    keys = [_mask_key(variant) for _name, variant in _d4_variants(mask)]
    return min(keys)


def _extract_problem(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    page_bg = _get_bg(g)

    canvas_mask = g == CANVAS_COLOR
    canvas_coords = np.argwhere(canvas_mask)
    assert canvas_coords.size > 0
    canvas_bbox = _bbox_coords(canvas_coords)
    r0, c0, r1, c1 = canvas_bbox
    canvas = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(canvas == CANVAS_COLOR)

    row_counts = []
    for rr in range(r0, r1 + 1):
        count = 0
        cc = c0 - 1
        while cc >= 0 and int(g[rr, cc]) == BAR_COLOR:
            count += 1
            cc -= 1
        row_counts.append(count)

    col_counts = []
    for cc in range(c0, c1 + 1):
        count = 0
        rr = r0 - 1
        while rr >= 0 and int(g[rr, cc]) == BAR_COLOR:
            count += 1
            rr -= 1
        col_counts.append(count)

    row_support = [i for i, v in enumerate(row_counts) if v > 0]
    col_support = [i for i, v in enumerate(col_counts) if v > 0]
    assert row_support and col_support
    target_top, target_bottom = row_support[0], row_support[-1]
    target_left, target_right = col_support[0], col_support[-1]
    target_row_sig = tuple(row_counts[target_top : target_bottom + 1])
    target_col_sig = tuple(col_counts[target_left : target_right + 1])
    assert all(v > 0 for v in target_row_sig)
    assert all(v > 0 for v in target_col_sig)

    panel_mask = (g != page_bg) & (g != CANVAS_COLOR) & (g != BAR_COLOR)
    panel_coords = np.argwhere(panel_mask)
    assert panel_coords.size > 0
    pr0, pc0, pr1, pc1 = _bbox_coords(panel_coords)
    panel = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    panel_bg = int(Counter(map(int, panel.ravel())).most_common(1)[0][0])

    candidates = []
    for color in sorted(int(c) for c in np.unique(panel) if int(c) != panel_bg):
        comps = _component_coords(panel == color, connectivity=4)
        assert len(comps) == 1
        mask = _normalize_shape(comps[0])
        assert _all_rows_cols_used(mask)
        candidates.append({"color": color, "mask": mask})
    assert len(candidates) >= 2

    return {
        "grid": g,
        "page_bg": page_bg,
        "canvas_bbox": canvas_bbox,
        "row_counts": tuple(row_counts),
        "col_counts": tuple(col_counts),
        "target_bbox": (target_top, target_left, target_bottom, target_right),
        "target_row_sig": target_row_sig,
        "target_col_sig": target_col_sig,
        "panel_bbox": (pr0, pc0, pr1, pc1),
        "panel_bg": panel_bg,
        "candidates": candidates,
    }


def _match_records(candidates: list[dict], row_sig: tuple[int, ...], col_sig: tuple[int, ...]) -> list[dict]:
    matches = []
    for idx, cand in enumerate(candidates):
        for name, variant in _d4_variants(cand["mask"]):
            if _row_sig(variant) == row_sig and _col_sig(variant) == col_sig:
                matches.append(
                    {
                        "candidate_index": idx,
                        "color": cand["color"],
                        "transform": name,
                        "mask": variant,
                    }
                )
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the only candidate shape-and-pose that matches the two projection shadows and paint it back onto the canvas.

    Input structure:
    - A solid black rectangle is the empty canvas.
    - Immediately above the canvas, grey vertical bars encode the hidden shape's column counts; immediately left of the canvas, grey horizontal bars encode its row counts; zero-count lines have no bar.
    - Elsewhere, a separated legend panel shows several single-color candidate shapes on a uniform panel background.

    Rule:
    - Treat every legend shape as movable up to rotation and mirror in the D4 family.
    - Find the unique candidate pose whose row projection and column projection both match the two bar sets.
    - The non-zero bar supports mark the hidden shape's bounding-box rows and columns inside the canvas; draw the matched pose there using that candidate's own color.
    - Leave the projection bars and the legend panel unchanged.

    Key insight:
    - The actual target never appears in the input; only its row and column shadows are visible, and both shadows together are needed to isolate a single candidate pose.
    """
    info = _extract_problem(grid)
    matches = _match_records(info["candidates"], info["target_row_sig"], info["target_col_sig"])
    assert len(matches) == 1
    match = matches[0]

    out = np.array(grid, dtype=int, copy=True)
    canvas_r0, canvas_c0, _canvas_r1, _canvas_c1 = info["canvas_bbox"]
    top, left, _bottom, _right = info["target_bbox"]
    mask = match["mask"]
    color = int(match["color"])
    h, w = mask.shape
    rr = canvas_r0 + top
    cc = canvas_c0 + left
    out[rr : rr + h, cc : cc + w][mask] = color
    return out


def _random_connected_shape(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(2, MAX_SLOT)
    w = rng.randint(2, MAX_SLOT)
    max_area = min(h * w - 1, 10)
    min_area = max(h, w, 4)
    if min_area > max_area:
        return None
    target = rng.randint(min_area, max_area)

    cells = {(rng.randrange(h), rng.randrange(w))}
    for _ in range(target * 20):
        if len(cells) >= target:
            break
        frontier = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        frontier.sort(key=lambda rc: (rc[0] not in {0, h - 1}, rc[1] not in {0, w - 1}, rng.random()))
        cells.add(frontier[0])
    if len(cells) != target:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _crop_mask(mask)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return None
    if not _all_rows_cols_used(mask):
        return None
    if np.all(mask):
        return None
    if len(_component_coords(mask, connectivity=4)) != 1:
        return None
    variants = _d4_variants(mask)
    if len(variants) < 3:
        return None
    if len(set(_row_sig(v) for _n, v in variants)) < 2:
        return None
    if len(set(_col_sig(v) for _n, v in variants)) < 2:
        return None
    return mask


def _candidate_suite(rng: random.Random) -> list[dict] | None:
    page_bg = rng.choice(PAGE_BG_POOL)
    panel_choices = [c for c in PANEL_BG_POOL if c != page_bg]
    panel_bg = rng.choice(panel_choices)

    count = rng.randint(3, 5)
    colors = [c for c in SHAPE_COLOR_POOL if c not in {CANVAS_COLOR, BAR_COLOR, page_bg, panel_bg}]
    if len(colors) < count:
        return None
    rng.shuffle(colors)

    shapes = []
    orbit_keys = set()
    for color in colors[:count]:
        picked = None
        for _ in range(80):
            mask = _random_connected_shape(rng)
            if mask is None:
                continue
            key = _shape_orbit_key(mask)
            if key in orbit_keys:
                continue
            orbit_keys.add(key)
            picked = {"color": color, "mask": mask}
            break
        if picked is None:
            return None
        shapes.append(picked)
    return [{"page_bg": page_bg, "panel_bg": panel_bg, **entry} for entry in shapes]


def _admissible_matches(candidates: list[dict]) -> list[dict]:
    records = []
    for idx, cand in enumerate(candidates):
        for name, variant in _d4_variants(cand["mask"]):
            records.append(
                {
                    "candidate_index": idx,
                    "color": cand["color"],
                    "transform": name,
                    "mask": variant,
                    "row_sig": _row_sig(variant),
                    "col_sig": _col_sig(variant),
                }
            )

    admissible = []
    for rec in records:
        both = [r for r in records if r["row_sig"] == rec["row_sig"] and r["col_sig"] == rec["col_sig"]]
        row_only = [r for r in records if r["row_sig"] == rec["row_sig"]]
        col_only = [r for r in records if r["col_sig"] == rec["col_sig"]]
        if len(both) != 1:
            continue
        if len(row_only) < 2 or len(col_only) < 2:
            continue
        admissible.append(rec)
    return admissible


def _choose_target_record(rng: random.Random, admissible: list[dict]) -> dict:
    mirror_labels = {"flip_lr", "flip_ud", "flip_diag", "flip_anti"}
    mirror = [r for r in admissible if r["transform"] in mirror_labels]
    rotated = [r for r in admissible if r["transform"] not in mirror_labels and r["transform"] != "id"]
    identity = [r for r in admissible if r["transform"] == "id"]
    if mirror and rng.random() < 0.55:
        return rng.choice(mirror)
    if rotated and rng.random() < 0.75:
        return rng.choice(rotated)
    return rng.choice(admissible if not identity else admissible if rng.random() < 0.7 else identity)


def _panel_dims(count: int) -> tuple[int, int]:
    rows = (count + PANEL_COLS - 1) // PANEL_COLS
    width = PANEL_PAD * 2 + PANEL_COLS * MAX_SLOT + (PANEL_COLS - 1) * SLOT_GAP
    height = PANEL_PAD * 2 + rows * MAX_SLOT + (rows - 1) * SLOT_GAP
    return height, width


def _paint_panel(grid: np.ndarray, top: int, left: int, panel_bg: int, candidates: list[dict], rng: random.Random) -> None:
    panel_h, panel_w = _panel_dims(len(candidates))
    grid[top : top + panel_h, left : left + panel_w] = panel_bg
    for idx, cand in enumerate(candidates):
        slot_r = idx // PANEL_COLS
        slot_c = idx % PANEL_COLS
        slot_top = top + PANEL_PAD + slot_r * (MAX_SLOT + SLOT_GAP)
        slot_left = left + PANEL_PAD + slot_c * (MAX_SLOT + SLOT_GAP)
        mask = cand["mask"]
        h, w = mask.shape
        off_r = rng.randint(0, MAX_SLOT - h)
        off_c = rng.randint(0, MAX_SLOT - w)
        patch = grid[slot_top + off_r : slot_top + off_r + h, slot_left + off_c : slot_left + off_c + w]
        patch[mask] = cand["color"]


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = _candidate_suite(rng)
    if candidates is None:
        return None
    page_bg = int(candidates[0]["page_bg"])
    panel_bg = int(candidates[0]["panel_bg"])
    admissible = _admissible_matches(candidates)
    if not admissible:
        return None
    target = _choose_target_record(rng, admissible)
    mask = target["mask"]
    row_sig = target["row_sig"]
    col_sig = target["col_sig"]
    target_h, target_w = mask.shape

    canvas_h = rng.randint(max(7, target_h + 2), 10)
    canvas_w = rng.randint(max(7, target_w + 2), 9)
    top = rng.randint(0, canvas_h - target_h)
    left = rng.randint(0, canvas_w - target_w)

    row_counts = [0] * canvas_h
    col_counts = [0] * canvas_w
    for i, v in enumerate(row_sig):
        row_counts[top + i] = int(v)
    for j, v in enumerate(col_sig):
        col_counts[left + j] = int(v)

    panel_h, panel_w = _panel_dims(len(candidates))
    canvas_r0 = TOP_GUTTER + 1
    canvas_c0 = LEFT_GUTTER + 1
    panel_left = canvas_c0 + canvas_w + PANEL_GAP
    panel_top = rng.randint(1, 3)
    height = max(canvas_r0 + canvas_h + 1, panel_top + panel_h + 1)
    width = panel_left + panel_w + 1
    if max(height, width) > 30:
        return None

    inp = np.full((height, width), page_bg, dtype=int)
    inp[canvas_r0 : canvas_r0 + canvas_h, canvas_c0 : canvas_c0 + canvas_w] = CANVAS_COLOR

    for j, count in enumerate(col_counts):
        if count:
            inp[canvas_r0 - count : canvas_r0, canvas_c0 + j] = BAR_COLOR
    for i, count in enumerate(row_counts):
        if count:
            inp[canvas_r0 + i, canvas_c0 - count : canvas_c0] = BAR_COLOR

    _paint_panel(inp, panel_top, panel_left, panel_bg, candidates, rng)

    out = inp.copy()
    rr = canvas_r0 + top
    cc = canvas_c0 + left
    out[rr : rr + target_h, cc : cc + target_w][mask] = int(target["color"])

    if not _semantic_check(inp, out):
        return None
    return inp, out


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        info = _extract_problem(inp)
    except Exception:
        return False

    orbit_keys = [_shape_orbit_key(c["mask"]) for c in info["candidates"]]
    if len(set(orbit_keys)) != len(orbit_keys):
        return False
    if not all(_all_rows_cols_used(c["mask"]) for c in info["candidates"]):
        return False

    records = []
    for idx, cand in enumerate(info["candidates"]):
        for name, variant in _d4_variants(cand["mask"]):
            records.append((idx, name, variant, _row_sig(variant), _col_sig(variant)))

    row_sig = info["target_row_sig"]
    col_sig = info["target_col_sig"]
    both = [(idx, name) for idx, name, _mask, rs, cs in records if rs == row_sig and cs == col_sig]
    row_only = [(idx, name) for idx, name, _mask, rs, _cs in records if rs == row_sig]
    col_only = [(idx, name) for idx, name, _mask, _rs, cs in records if cs == col_sig]
    if len(both) != 1 or len(row_only) < 2 or len(col_only) < 2:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    diff = (expected != inp)
    changed = np.argwhere(diff)
    if changed.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(changed)
    canvas_r0, canvas_c0, _canvas_r1, _canvas_c1 = info["canvas_bbox"]
    top, left, bottom, right = info["target_bbox"]
    if (r0, c0, r1, c1) != (canvas_r0 + top, canvas_c0 + left, canvas_r0 + bottom, canvas_c0 + right):
        return False
    if np.any(inp[diff] != CANVAS_COLOR):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = random.Random()
        built = _build_instance(rng)
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")

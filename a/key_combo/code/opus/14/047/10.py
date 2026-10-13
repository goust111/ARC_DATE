from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SCALES = (2, 3, 4)
CORNERS = ("tl", "tr", "bl", "br")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale(template: np.ndarray, k: int) -> np.ndarray:
    return np.kron(template, np.ones((k, k), dtype=int))


def _neighbors8(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    if not mask[seed]:
        return out
    q = deque([seed])
    out[seed] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c, (h, w)):
            if mask[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    seed = tuple(map(int, np.argwhere(mask)[0]))
    return bool(np.array_equal(mask, _component_from_seed(mask, seed)))


def _corner_seed(corner: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 1
    if corner == "bl":
        return h - 1, 0
    return h - 1, w - 1


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1, 0)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1, 0)
    return row_gap, col_gap


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    nonzero = g != BG

    corner_hits = []
    for corner in CORNERS:
        seed = _corner_seed(corner, g.shape)
        if nonzero[seed]:
            corner_hits.append((corner, seed))
    assert len(corner_hits) == 1, "expected exactly one corner-anchored template"

    _corner, seed = corner_hits[0]
    template_mask = _component_from_seed(nonzero, seed)
    template_box = _bbox(template_mask)
    assert template_box is not None
    template = _crop_bbox(g, template_box)

    remaining = g.copy()
    remaining[template_mask] = BG
    partial_box = _bbox(remaining != BG)
    assert partial_box is not None, "expected one partial enlarged region"
    partial_crop = _crop_bbox(remaining, partial_box)
    return template_mask, template_box, template, remaining, partial_box, partial_crop


def _candidate_infos(template: np.ndarray, partial_crop: np.ndarray) -> list[dict]:
    th, tw = template.shape
    infos: list[dict] = []
    for k in SCALES:
        full = _scale(template, k)
        for m in range(2, th):
            prefix = full[: m * k, :]
            box = _bbox(prefix != BG)
            if box is None:
                continue
            crop = _crop_bbox(prefix, box)
            if crop.shape == partial_crop.shape and np.array_equal(crop, partial_crop):
                infos.append(
                    {
                        "k": k,
                        "m": m,
                        "full": full,
                        "prefix": prefix,
                        "prefix_box": box,
                    }
                )
    return infos


def _valid_candidates(grid: np.ndarray, template_mask: np.ndarray, partial_box: tuple[int, int, int, int], template: np.ndarray) -> list[dict]:
    _r0, _c0, _r1, _c1 = partial_box
    remaining = np.asarray(grid, dtype=int).copy()
    remaining[template_mask] = BG
    partial_crop = _crop_bbox(remaining, partial_box)
    valid = []
    for info in _candidate_infos(template, partial_crop):
        pr0, pc0, _pr1, _pc1 = info["prefix_box"]
        ar = partial_box[0] - pr0
        ac = partial_box[1] - pc0
        fh, fw = info["full"].shape
        if ar < 0 or ac < 0 or ar + fh > grid.shape[0] or ac + fw > grid.shape[1]:
            continue
        expected = np.zeros_like(grid)
        _paste(expected, info["prefix"], ar, ac)
        if np.array_equal(expected, remaining):
            out = dict(info)
            out["anchor"] = (ar, ac)
            valid.append(out)
    return valid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden block scale from the visible enlarged prefix, then complete the missing lower part of the template copy.

    Input structure:
    - A black background contains one small multicolor template touching exactly one grid corner.
    - Elsewhere, a larger region shows only the top prefix of an integer-scaled copy of that same template.
    - The lower part of the enlarged copy is missing and remains black in the input.

    Rule:
    - Extract the corner template and the separate enlarged partial region.
    - Try integer scales 2, 3, and 4, enlarge the whole template, and compare the enlarged top-prefix crop against the visible partial region.
    - Once the unique scale and prefix height are identified, place the full enlarged template at that location.
    - Keep the small corner template unchanged and preserve all already visible enlarged pixels.

    Key insight:
    - The enlarged object is not fully visible and may hide blank leading or trailing columns, so the scale must be inferred by matching the cropped visible prefix against candidate scaled templates rather than by a simple size ratio.
    """
    g = np.asarray(grid, dtype=int)
    template_mask, _template_box, template, _remaining, partial_box, _partial_crop = _scene(g)
    candidates = _valid_candidates(g, template_mask, partial_box, template)
    assert len(candidates) == 1, "expected a unique scale/prefix match"

    chosen = candidates[0]
    full = chosen["full"]
    ar, ac = chosen["anchor"]

    out = g.copy()
    region = out[ar : ar + full.shape[0], ac : ac + full.shape[1]]
    conflict = (region != BG) & (full != BG) & (region != full)
    assert not conflict.any(), "visible prefix must agree with the completed template"
    region[full != BG] = full[full != BG]
    return out


def _random_walk_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    while (r, c) != (tr, tc):
        opts = []
        sr = 0 if tr == r else (1 if tr > r else -1)
        sc = 0 if tc == c else (1 if tc > c else -1)
        if sr != 0 and sc != 0:
            opts.append((r + sr, c + sc))
        if sr != 0:
            opts.append((r + sr, c))
        if sc != 0:
            opts.append((r, c + sc))
        r, c = random.choice(opts)
        path.append((r, c))
    return path


def _boundary_points(h: int, w: int, m: int, corner: str) -> list[tuple[int, int]]:
    seed = _corner_seed(corner, (h, w))
    late_rows = list(range(m, h))
    assert late_rows

    points = {seed}
    if corner not in ("tl", "tr"):
        top_cols = [c for c in range(w) if (0, c) != seed]
        points.add((0, random.choice(top_cols)))
    if corner not in ("bl", "br"):
        bottom_cols = [c for c in range(w) if (h - 1, c) != seed]
        points.add((h - 1, random.choice(bottom_cols)))
    if corner not in ("tl", "bl"):
        points.add((random.choice(late_rows), 0))
    if corner not in ("tr", "br"):
        points.add((random.choice(late_rows), w - 1))

    return list(points)


def _grow_mask(h: int, w: int, m: int, corner: str) -> np.ndarray | None:
    area = h * w
    target = random.randint(max(7, int(np.ceil(0.42 * area))), max(8, int(np.floor(0.7 * area))))
    points = _boundary_points(h, w, m, corner)
    mask = np.zeros((h, w), dtype=bool)

    order = points[:]
    random.shuffle(order)
    head = order[0]
    mask[head] = True
    for pt in order[1:]:
        for cell in _random_walk_path(head, pt):
            mask[cell] = True
        head = pt

    frontier = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors8(int(r), int(c), (h, w)):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nr, nc in _neighbors8(cell[0], cell[1], (h, w)):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _colorize(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(3, 4)
    colors = random.sample(PALETTE, color_count)
    assignments = colors + [random.choice(colors) for _ in range(len(coords) - color_count)]
    random.shuffle(assignments)
    out = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, assignments):
        out[r, c] = color
    return out


def _template_ok(template: np.ndarray, m: int, corner: str) -> bool:
    mask = template != BG
    h, w = template.shape
    if not _is_connected(mask):
        return False
    if not all(mask[r].any() for r in range(h)):
        return False
    if len(set(int(v) for v in template[mask])) < 3:
        return False
    if not mask[_corner_seed(corner, (h, w))]:
        return False
    if _bbox(mask) != (0, 0, h - 1, w - 1):
        return False
    prefix_box = _bbox(mask[:m, :])
    if prefix_box is None:
        return False
    if prefix_box[3] - prefix_box[1] + 1 >= w:
        return False
    if np.unique(template[0, template[0] != BG]).size < 2:
        return False
    if not mask[m:, :].any():
        return False
    return True


def _make_template(h: int, w: int, m: int, corner: str) -> np.ndarray | None:
    for _ in range(120):
        mask = _grow_mask(h, w, m, corner)
        if mask is None:
            continue
        template = _colorize(mask)
        if _template_ok(template, m, corner):
            return template
    return None


def _place_template(grid: np.ndarray, template: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    h, w = template.shape
    gh, gw = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, gw - w
    elif corner == "bl":
        r0, c0 = gh - h, 0
    else:
        r0, c0 = gh - h, gw - w
    _paste(grid, template, r0, c0)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _sample_full_anchor(
    grid_shape: tuple[int, int],
    full_shape: tuple[int, int],
    template_box: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    gh, gw = grid_shape
    fh, fw = full_shape
    choices = []
    for r0 in range(1, gh - fh):
        for c0 in range(1, gw - fw):
            full_box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            row_gap, col_gap = _bbox_gap(template_box, full_box)
            if row_gap >= 2 or col_gap >= 3:
                choices.append((r0, c0))
    if not choices:
        return None
    return random.choice(choices)


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        template_mask, template_box, template, remaining, partial_box, partial_crop = _scene(grid)
        candidates = _valid_candidates(grid, template_mask, partial_box, template)
        if len(candidates) != 1:
            return False
        chosen = candidates[0]
        full = chosen["full"]
        if partial_crop.shape[1] >= full.shape[1]:
            return False
        if int((remaining != BG).sum()) >= int((main(grid) != BG).sum()):
            pass
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if out.shape != grid.shape or max(out.shape) > 30:
            return False
        if not np.array_equal(out[template_mask], grid[template_mask]):
            return False
        return True
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(4, 6)
    k = random.choice(SCALES)
    m = random.randint(2, h - 1)
    corner = random.choice(CORNERS)

    template = _make_template(h, w, m, corner)
    if template is None:
        return None

    full = _scale(template, k)
    prefix = full[: m * k, :]
    prefix_box = _bbox(prefix != BG)
    if prefix_box is None:
        return None
    prefix_crop = _crop_bbox(prefix, prefix_box)
    if prefix_crop.shape[0] < 2 * k or prefix_crop.shape[1] < 2:
        return None
    if len(_candidate_infos(template, prefix_crop)) != 1:
        return None

    min_h = max(16, h + full.shape[0] + 3)
    min_w = max(16, w + full.shape[1] + 3)
    if min_h > 30 or min_w > 30:
        return None
    gh = random.randint(min_h, 30)
    gw = random.randint(min_w, 30)

    grid = np.zeros((gh, gw), dtype=int)
    template_box = _place_template(grid, template, corner)
    anchor = _sample_full_anchor((gh, gw), full.shape, template_box)
    if anchor is None:
        return None
    _paste(grid, prefix, anchor[0], anchor[1])

    nonzero_corners = sum(int(grid[_corner_seed(c, grid.shape)] != BG) for c in CORNERS)
    if nonzero_corners != 1:
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = list(range(1, 10))

PANEL_H_RANGE = (6, 8)
PANEL_W_RANGE = (6, 9)
SHADOW_MIN_CELLS = 4
SHADOW_MAX_CELLS = 8

CANDIDATE_COUNT_CHOICES = (4, 5)
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 30)
TOP_MARGIN_CHOICES = (0, 1)
BOTTOM_GAP_RANGE = (2, 4)
PLACEMENT_PAD = 1

GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = box
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return np.count_nonzero(mask) > 0 and len(_component_coords(mask)) == 1


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _component_coords(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask = np.asarray(mask, dtype=bool)
    return tuple(map(int, mask.shape)), mask.tobytes()


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    mask = _trim_mask(mask)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            candidate = _trim_mask(candidate)
            sig = _mask_signature(candidate)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(candidate)
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [_mask_signature(v) for v in _orientations(mask)]
    return min(variants)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _render_mask(mask: np.ndarray, color: int, bg: int = BG) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _paste_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = int(color)


def _find_top_separator_and_panels(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], tuple[int, int, int, int], int]:
    candidates = []
    for comp in _find_monochrome_components(grid, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if c0 != c1 or r1 - r0 + 1 < 3:
            continue
        if c0 == 0 or c1 == grid.shape[1] - 1:
            continue

        sep_color = int(comp["color"])
        band = grid[r0 : r1 + 1, :]
        left = band[:, :c0]
        right = band[:, c1 + 1 :]
        if left.size == 0 or right.size == 0:
            continue

        left_mask = (left != bg) & (left != sep_color)
        right_mask = (right != bg) & (right != sep_color)
        left_box = _bbox(left_mask)
        right_box = _bbox(right_mask)
        if left_box is None or right_box is None:
            continue

        h = r1 - r0 + 1
        if left_box[0] != 0 or left_box[2] != h - 1:
            continue
        if right_box[0] != 0 or right_box[2] != h - 1:
            continue
        if left_box[3] != c0 - 1:
            continue
        if right_box[1] != 0:
            continue

        left_h = left_box[2] - left_box[0] + 1
        left_w = left_box[3] - left_box[1] + 1
        right_h = right_box[2] - right_box[0] + 1
        right_w = right_box[3] - right_box[1] + 1
        if (left_h, left_w) != (right_h, right_w):
            continue

        left_bbox = (r0, int(left_box[1]), r1, int(left_box[3]))
        right_bbox = (r0, c1 + 1 + int(right_box[1]), r1, c1 + 1 + int(right_box[3]))
        candidates.append((h, left_w, left_bbox, right_bbox, sep_color))

    assert candidates, "failed to find the two top panels"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, left_bbox, right_bbox, sep_color = candidates[0]
    return left_bbox[2], left_bbox, right_bbox, sep_color


def _candidate_patch_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    return grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the XOR-only shadow between the two top binary panels and output the unique matching candidate piece.

    Input structure:
    - The upper part contains two equal-size binary panels separated by one vertical monochrome line.
    - Each panel uses background plus a single foreground color, and the lower part contains separated monochrome candidate pieces.

    Rule:
    - Detect the two upper panels, convert them to boolean masks, and compute the cells that are foreground in exactly one panel.
    - Tight-crop that XOR shadow shape.
    - Among the candidate pieces below, find the unique piece whose tight-cropped shape matches the shadow under rotation or reflection.
    - Return that piece's own bbox crop, preserving its original orientation and color.

    Key insight:
    - The answer shape is never drawn directly in the top panels; it appears only as the panel-wise XOR residue, and the lower pieces must be matched to that hidden boolean shape up to D4 symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    top_bottom, left_bbox, right_bbox, _sep_color = _find_top_separator_and_panels(g, bg)
    lr0, lc0, lr1, lc1 = left_bbox
    rr0, rc0, rr1, rc1 = right_bbox

    left_mask = g[lr0 : lr1 + 1, lc0 : lc1 + 1] != bg
    right_mask = g[rr0 : rr1 + 1, rc0 : rc1 + 1] != bg
    assert left_mask.shape == right_mask.shape, "panel sizes must match"

    shadow_full = np.logical_xor(left_mask, right_mask)
    assert np.any(shadow_full), "xor shadow must be non-empty"
    shadow = _trim_mask(shadow_full)
    assert _is_connected(shadow), "xor shadow must be connected"

    variants = {_mask_signature(v) for v in _orientations(shadow)}

    matches = []
    for comp in _find_monochrome_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        if r0 <= top_bottom:
            continue
        piece_mask = _candidate_patch_mask(g, comp)
        if _mask_signature(piece_mask) in variants:
            matches.append(comp)

    assert len(matches) == 1, f"expected exactly one matching candidate, found {len(matches)}"
    r0, c0, r1, c1 = matches[0]["bbox"]
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _plus_intersection_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    row = random.randint(1, h - 2)
    col = random.randint(1, w - 2)
    mask[row, :] = True
    mask[:, col] = True
    return mask


def _grow_connected_subset(allowed: np.ndarray, min_cells: int, max_cells: int) -> np.ndarray | None:
    allowed = np.asarray(allowed, dtype=bool)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(coords) < min_cells:
        return None
    target_hi = min(int(max_cells), len(coords))
    if min_cells > target_hi:
        return None

    for _ in range(200):
        target = random.randint(int(min_cells), int(target_hi))
        start = random.choice(coords)
        chosen = {start}
        frontier = {p for p in _neighbors4(*start) if 0 <= p[0] < allowed.shape[0] and 0 <= p[1] < allowed.shape[1] and allowed[p]}
        while len(chosen) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in chosen:
                continue
            chosen.add(cell)
            for nxt in _neighbors4(*cell):
                if 0 <= nxt[0] < allowed.shape[0] and 0 <= nxt[1] < allowed.shape[1] and allowed[nxt] and nxt not in chosen:
                    frontier.add(nxt)
        if len(chosen) != target:
            continue

        mask = np.zeros_like(allowed)
        for r, c in chosen:
            mask[r, c] = True
        trimmed = _trim_mask(mask)
        if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
            continue
        if len(_orientations(trimmed)) < 8:
            continue
        return mask
    return None


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return _trim_mask(out)


def _near_miss_mask(base: np.ndarray) -> np.ndarray | None:
    base = _trim_mask(base)
    for _ in range(200):
        mask = np.pad(base, 1, constant_values=False)
        if random.random() < 0.5:
            frontier = []
            for r, c in np.argwhere(~mask):
                rr = int(r)
                cc = int(c)
                if any(
                    0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]
                    for nr, nc in _neighbors4(rr, cc)
                ):
                    frontier.append((rr, cc))
            if not frontier:
                continue
            rr, cc = random.choice(frontier)
            mask[rr, cc] = True
        else:
            removable = []
            for r, c in np.argwhere(mask):
                trial = mask.copy()
                trial[int(r), int(c)] = False
                if not np.any(trial):
                    continue
                trial = _trim_mask(trial)
                if _is_connected(trial):
                    removable.append((int(r), int(c)))
            if not removable:
                continue
            rr, cc = random.choice(removable)
            mask[rr, cc] = False

        mask = _trim_mask(mask)
        if not _is_connected(mask):
            continue
        if _mask_key(mask) == _mask_key(base):
            continue
        return _random_dihedral(mask)
    return None


def _random_distractor_mask(max_h: int, max_w: int, banned: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(3, max_h)
        w = random.randint(3, max_w)
        mask = _grow_connected_subset(np.ones((h, w), dtype=bool), 3, min(8, h * w - 1))
        if mask is None:
            continue
        mask = _random_dihedral(mask)
        if _mask_key(mask) in banned:
            continue
        return mask
    return None


def _build_candidate_masks(
    shadow_full: np.ndarray,
    intersection: np.ndarray,
    left_panel: np.ndarray,
    right_panel: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    shadow = _trim_mask(shadow_full)
    target = _random_dihedral(shadow)
    near = _near_miss_mask(shadow)
    if near is None:
        raise RuntimeError("failed to build near-miss distractor")

    desired = random.choice(CANDIDATE_COUNT_CHOICES)
    result = [("target", target), ("near_miss", near)]
    used = {_mask_key(target), _mask_key(near)}

    union_panel = intersection | shadow_full
    pool = [
        ("intersection", _trim_mask(intersection)),
        ("left_panel", _trim_mask(left_panel)),
        ("right_panel", _trim_mask(right_panel)),
        ("union_panel", _trim_mask(union_panel)),
    ]
    random.shuffle(pool)
    for name, mask in pool:
        key = _mask_key(mask)
        if key in used:
            continue
        result.append((name, mask))
        used.add(key)
        if len(result) >= desired:
            return result

    while len(result) < desired:
        extra = _random_distractor_mask(intersection.shape[0], intersection.shape[1], used)
        if extra is None:
            break
        key = _mask_key(extra)
        result.append(("extra", extra))
        used.add(key)
    return result


def _find_free_bbox_position(
    occupied: np.ndarray,
    patch_h: int,
    patch_w: int,
    *,
    row_min: int,
    pad: int,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    options = []
    for r in range(row_min, h - patch_h + 1):
        for c in range(0, w - patch_w + 1):
            rr0 = max(0, r - pad)
            rr1 = min(h, r + patch_h + pad)
            cc0 = max(0, c - pad)
            cc1 = min(w, c + patch_w + pad)
            if not np.any(occupied[rr0:rr1, cc0:cc1]):
                options.append((r, c))
    if not options:
        return None
    return random.choice(options)


def _build_input_and_expected() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    intersection = _plus_intersection_mask(panel_h, panel_w)

    halo = _dilate4(intersection) & ~intersection
    shadow_full = _grow_connected_subset(
        halo,
        SHADOW_MIN_CELLS,
        min(SHADOW_MAX_CELLS, int(np.count_nonzero(halo))),
    )
    if shadow_full is None:
        return None

    shadow_trim = _trim_mask(shadow_full)
    if shadow_trim.shape == intersection.shape:
        return None

    shadow_coords = [tuple(map(int, rc)) for rc in np.argwhere(shadow_full)]
    left_extra = np.zeros_like(shadow_full)
    right_extra = np.zeros_like(shadow_full)
    for _ in range(50):
        left_extra.fill(False)
        right_extra.fill(False)
        for cell in shadow_coords:
            if random.random() < 0.5:
                left_extra[cell] = True
            else:
                right_extra[cell] = True
        if np.any(left_extra) and np.any(right_extra):
            break
    else:
        return None

    left_panel = intersection | left_extra
    right_panel = intersection | right_extra

    candidates = _build_candidate_masks(shadow_full, intersection, left_panel, right_panel)
    shadow_variants = {_mask_signature(v) for v in _orientations(shadow_trim)}
    if sum(1 for _name, mask in candidates if _mask_signature(mask) in shadow_variants) != 1:
        return None

    needed_colors = 3 + len(candidates)
    if needed_colors > len(PALETTE):
        return None
    colors = random.sample(PALETTE, needed_colors)
    sep_color, left_color, right_color = colors[:3]
    piece_colors = colors[3:]

    top_margin = random.choice(TOP_MARGIN_CHOICES)
    gap = random.randint(*BOTTOM_GAP_RANGE)
    top_width = panel_w * 2 + 1
    target_row_min = top_margin + panel_h + gap

    candidate_sizes = [mask.shape for _name, mask in candidates]
    min_grid_h = max(GRID_H_RANGE[0], target_row_min + max(h for h, _w in candidate_sizes) + 1)
    min_grid_w = max(GRID_W_RANGE[0], top_width + 2, max(w for _h, w in candidate_sizes) + 2)
    if min_grid_h > GRID_H_RANGE[1] or min_grid_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_grid_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_grid_w, GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    left_col = random.randint(0, grid_w - top_width)
    sep_col = left_col + panel_w
    right_col = sep_col + 1

    _paste_mask(grid, left_panel, left_color, top_margin, left_col)
    _paste_mask(grid, right_panel, right_color, top_margin, right_col)
    grid[top_margin : top_margin + panel_h, sep_col] = sep_color
    occupied[: target_row_min, :] = True

    expected = None
    pieces = list(zip(candidates, piece_colors))
    pieces.sort(key=lambda item: item[0][1].shape[0] * item[0][1].shape[1], reverse=True)
    for (name, mask), color in pieces:
        pos = _find_free_bbox_position(
            occupied,
            int(mask.shape[0]),
            int(mask.shape[1]),
            row_min=target_row_min,
            pad=PLACEMENT_PAD,
        )
        if pos is None:
            return None
        r0, c0 = pos
        _paste_mask(grid, mask, color, r0, c0)
        rr0 = max(0, r0 - PLACEMENT_PAD)
        rr1 = min(grid_h, r0 + mask.shape[0] + PLACEMENT_PAD)
        cc0 = max(0, c0 - PLACEMENT_PAD)
        cc1 = min(grid_w, c0 + mask.shape[1] + PLACEMENT_PAD)
        occupied[rr0:rr1, cc0:cc1] = True
        if name == "target":
            expected = _render_mask(mask, color)

    if expected is None:
        return None
    if get_bg(grid) != BG:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        built = _build_input_and_expected()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")

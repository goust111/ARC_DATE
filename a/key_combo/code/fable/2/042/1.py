from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1]).copy()


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    bbox = _bbox_from_mask(mask)
    assert bbox is not None, "expected a nonempty mask"
    return _crop_bbox(mask.astype(bool), bbox).astype(bool)


def _component_patch(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    patch[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
    return patch


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            bbox = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox,
                    "area": int(arr.shape[0]),
                    "mask": _component_patch(arr),
                }
            )
    return comps


def _reflect_to_tl(grid: np.ndarray, corner: str) -> np.ndarray:
    if corner == "TL":
        return np.asarray(grid)
    if corner == "TR":
        return np.fliplr(grid)
    if corner == "BL":
        return np.flipud(grid)
    assert corner == "BR"
    return np.flipud(np.fliplr(grid))


def _extract_corner_subgrid(
    grid: np.ndarray, corner: str, height: int, width: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, w - width
    elif corner == "BL":
        r0, c0 = h - height, 0
    else:
        r0, c0 = h - height, w - width
    bbox = (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))
    return _crop_bbox(grid, bbox), bbox


def _unique_mode(values: np.ndarray) -> int | None:
    counts = Counter(map(int, np.asarray(values).ravel()))
    ranked = counts.most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0])


def _is_unit_separated(runs: list[tuple[int, int]], total: int) -> tuple[bool, int]:
    if len(runs) < 2:
        return False, 0
    size = runs[0][1] - runs[0][0] + 1
    if size < 2 or runs[0][0] != 1 or runs[-1][1] != total - 2:
        return False, 0
    for r0, r1 in runs:
        if r1 - r0 + 1 != size:
            return False, 0
    for (a0, a1), (b0, _b1) in zip(runs, runs[1:]):
        if b0 != a1 + 2:
            return False, 0
    return True, int(size)


def _summarize_lattice(
    subgrid: np.ndarray, bg: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    row_bg = np.all(subgrid == bg, axis=1)
    col_bg = np.all(subgrid == bg, axis=0)
    row_runs = _runs(np.flatnonzero(~row_bg).tolist())
    col_runs = _runs(np.flatnonzero(~col_bg).tolist())
    row_ok, _block_h = _is_unit_separated(row_runs, subgrid.shape[0])
    col_ok, _block_w = _is_unit_separated(col_runs, subgrid.shape[1])
    if not row_ok or not col_ok:
        return None

    thumb = np.zeros((len(row_runs), len(col_runs)), dtype=int)
    for rr, (r0, r1) in enumerate(row_runs):
        for cc, (c0, c1) in enumerate(col_runs):
            block = subgrid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(block == bg):
                return None
            colors = np.unique(block)
            if colors.size != 2:
                return None
            mode = _unique_mode(block)
            if mode is None:
                return None
            thumb[rr, cc] = int(mode)

    majors = sorted(set(map(int, thumb.ravel())))
    if len(majors) != 2:
        return None
    counts = Counter(map(int, thumb.ravel()))
    if counts[majors[0]] == counts[majors[1]]:
        return None
    return thumb, row_runs, col_runs


def _find_lattice(grid: np.ndarray, bg: int) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []
    for corner in ("TL", "TR", "BL", "BR"):
        view = _reflect_to_tl(g, corner)
        for height in range(7, h + 1):
            for width in range(7, w + 1):
                sub = view[:height, :width]
                summary = _summarize_lattice(sub, bg)
                if summary is None:
                    continue
                actual_sub, bbox = _extract_corner_subgrid(g, corner, height, width)
                actual_summary = _summarize_lattice(actual_sub, bg)
                if actual_summary is None:
                    continue
                thumb, row_runs, col_runs = actual_summary
                candidates.append(
                    {
                        "corner": corner,
                        "bbox": bbox,
                        "height": height,
                        "width": width,
                        "thumb": thumb,
                        "row_runs": row_runs,
                        "col_runs": col_runs,
                        "area": int(height * width),
                    }
                )
    assert candidates, "expected one corner lattice"
    max_area = max(c["area"] for c in candidates)
    best = [c for c in candidates if c["area"] == max_area]
    assert len(best) == 1, "expected a unique maximal lattice"
    return best[0]


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    lattice = _find_lattice(g, bg)
    thumb = lattice["thumb"]
    counts = Counter(map(int, thumb.ravel()))
    minority_color, minority_count = min(counts.items(), key=lambda item: item[1])
    majority_color, majority_count = max(counts.items(), key=lambda item: item[1])
    assert minority_count < majority_count, "expected a strict minority majority-color"

    mask = thumb == int(minority_color)
    target_mask = _normalize_mask(mask)

    outside = g.copy()
    r0, c0, r1, c1 = lattice["bbox"]
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sprites = _find_components(outside, bg)
    assert sprites, "expected sprites outside the lattice"

    exact: list[dict] = []
    shape_matches: list[dict] = []
    color_matches: list[dict] = []
    for comp in sprites:
        same_shape = np.array_equal(comp["mask"], target_mask)
        same_color = int(comp["color"]) == int(minority_color)
        if same_shape:
            shape_matches.append(comp)
        if same_color:
            color_matches.append(comp)
        if same_shape and same_color:
            exact.append(comp)

    assert len(exact) == 1, "expected a unique sprite matching both shape and color"
    return {
        "bg": bg,
        "lattice": lattice,
        "thumbnail": thumb,
        "minority_color": int(minority_color),
        "majority_color": int(majority_color),
        "target_mask": target_mask,
        "sprites": sprites,
        "shape_matches": shape_matches,
        "color_matches": color_matches,
        "exact_match": exact[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the corner lattice by block majorities, then retrieve the only sprite matching the revealed minority shape and color.

    Input structure:
    - One grid corner contains a background-separated lattice with a one-cell outer border and one-cell separator lines.
    - Each lattice block is filled by exactly two colors and has a unique majority color.
    - After compression, the lattice thumbnail uses exactly two majority colors, with one appearing strictly fewer times.
    - Outside the lattice, several 4-connected monochrome sprites are scattered on the background.

    Rule:
    - Detect the corner lattice and compress each block to its majority color to obtain a coarse thumbnail.
    - Take the less frequent thumbnail color and normalize the binary mask formed by its cells.
    - Among the outside sprites, find the unique sprite whose normalized mask equals that thumbnail mask and whose own color equals the less frequent thumbnail color.
    - Crop that sprite to its bounding box and return it unchanged.

    Key insight:
    - Neither retrieval key is drawn explicitly: both the target shape and the target color emerge only after majority-vote compression of the noisy lattice.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    bbox = analysis["exact_match"]["bbox"]
    return _crop_bbox(np.asarray(grid, dtype=int), bbox)


def _polyomino_patch(
    area: int,
    *,
    max_h: int,
    max_w: int,
    exact_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        min_r, min_c = min(rs), min(cs)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h > max_h or w > max_w:
            continue
        if exact_shape is not None and (h, w) != exact_shape:
            continue
        patch = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            patch[r, c] = True
        return patch
    return None


def _choose_target_mask(m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    max_area = min(6, (m * n - 1) // 2)
    assert max_area >= 2
    for _ in range(300):
        area = random.randint(2, max_area)
        patch = _polyomino_patch(area, max_h=m, max_w=n)
        if patch is None:
            continue
        ph, pw = patch.shape
        top = random.randint(0, m - ph)
        left = random.randint(0, n - pw)
        mask = np.zeros((m, n), dtype=bool)
        mask[top : top + ph, left : left + pw] = patch
        if int(mask.sum()) * 2 >= m * n:
            continue
        return mask, patch
    raise AssertionError("failed to sample a connected minority mask")


def _sample_distinct_colors(bg: int, count: int) -> list[int]:
    return random.sample([c for c in PALETTE if c != bg], count)


def _render_lattice(
    mask: np.ndarray,
    *,
    block_h: int,
    block_w: int,
    bg: int,
    key_color: int,
    other_color: int,
) -> np.ndarray:
    m, n = mask.shape
    lattice = np.full((m * block_h + (m + 1), n * block_w + (n + 1)), bg, dtype=int)
    other_choices = [c for c in PALETTE if c not in {bg}]
    for br in range(m):
        for bc in range(n):
            majority = key_color if mask[br, bc] else other_color
            noise_palette = [c for c in other_choices if c != majority]
            noise_color = random.choice(noise_palette)
            block = np.full((block_h, block_w), majority, dtype=int)
            area = block_h * block_w
            max_noise = (area - 1) // 2
            noise_count = random.randint(1, max_noise)
            cells = random.sample(range(area), noise_count)
            block.reshape(-1)[cells] = noise_color
            r0 = 1 + br * (block_h + 1)
            c0 = 1 + bc * (block_w + 1)
            lattice[r0 : r0 + block_h, c0 : c0 + block_w] = block
    return lattice


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in DIR8:
        src_r0 = max(0, -dr)
        src_r1 = mask.shape[0] - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = mask.shape[1] - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, dc)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _patch_positions(
    patch: np.ndarray,
    shape: tuple[int, int],
    occupied: np.ndarray,
    blocked: np.ndarray,
    *,
    row_range: range | None = None,
    col_range: range | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch.shape
    rows = row_range if row_range is not None else range(0, h - ph + 1)
    cols = col_range if col_range is not None else range(0, w - pw + 1)
    coords = np.argwhere(patch)
    out: list[tuple[int, int]] = []
    for top in rows:
        for left in cols:
            if top < 0 or left < 0 or top + ph > h or left + pw > w:
                continue
            rr = coords[:, 0] + int(top)
            cc = coords[:, 1] + int(left)
            if np.any(blocked[rr, cc]) or np.any(occupied[rr, cc]):
                continue
            out.append((int(top), int(left)))
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(patch)
    rr = coords[:, 0] + int(top)
    cc = coords[:, 1] + int(left)
    grid[rr, cc] = int(color)
    return np.stack([rr, cc], axis=1)


def _place_sprites(
    grid: np.ndarray,
    lattice_bbox: tuple[int, int, int, int],
    specs: list[dict],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = lattice_bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    placed: list[dict] = []
    for spec in specs:
        patch = spec["patch"]
        mode = spec["mode"]
        positions: list[tuple[int, int]]
        if mode == "random":
            positions = _patch_positions(patch, grid.shape, occupied, blocked)
        elif mode == "vside":
            if c0 == 0:
                fixed_left = c1 + 2
            else:
                fixed_left = c0 - patch.shape[1] - 1
            top_lo = max(0, r0 - patch.shape[0] + 1)
            top_hi = min(h - patch.shape[0], r1)
            positions = _patch_positions(
                patch,
                grid.shape,
                occupied,
                blocked,
                row_range=range(top_lo, top_hi + 1),
                col_range=range(fixed_left, fixed_left + 1),
            )
        elif mode == "hside":
            if r0 == 0:
                fixed_top = r1 + 2
            else:
                fixed_top = r0 - patch.shape[0] - 1
            left_lo = max(0, c0 - patch.shape[1] + 1)
            left_hi = min(w - patch.shape[1], c1)
            positions = _patch_positions(
                patch,
                grid.shape,
                occupied,
                blocked,
                row_range=range(fixed_top, fixed_top + 1),
                col_range=range(left_lo, left_hi + 1),
            )
        else:
            raise ValueError(f"unknown placement mode: {mode}")

        if not positions:
            return False
        top, left = random.choice(positions)
        coords = _paint_patch(grid, patch, top, left, spec["color"])
        sprite_mask = np.zeros_like(occupied)
        sprite_mask[coords[:, 0], coords[:, 1]] = True
        occupied |= sprite_mask
        blocked |= _dilate8(sprite_mask)
        placed.append({"spec": spec, "coords": coords})

    return True


def _different_patch(target: np.ndarray) -> np.ndarray | None:
    target_area = int(target.sum())
    th, tw = target.shape
    modes = ["same_area", "same_bbox", "free"]
    random.shuffle(modes)
    for mode in modes:
        for _ in range(120):
            if mode == "same_area":
                patch = _polyomino_patch(target_area, max_h=max(th + 1, 4), max_w=max(tw + 1, 4))
            elif mode == "same_bbox":
                patch = _polyomino_patch(
                    random.randint(2, max(target_area + 1, 4)),
                    max_h=th,
                    max_w=tw,
                    exact_shape=(th, tw),
                )
            else:
                area = random.randint(2, max(4, target_area + 2))
                patch = _polyomino_patch(area, max_h=max(th + 1, 4), max_w=max(tw + 1, 4))
            if patch is None or np.array_equal(patch, target):
                continue
            return patch
    return None


def _build_instance() -> np.ndarray | None:
    corner = random.choice(("TL", "TR", "BL", "BR"))
    bg = random.choice(PALETTE)
    key_color, other_color, shape_decoy_color, neutral_color = _sample_distinct_colors(bg, 4)

    m = random.randint(3, 5)
    n = random.randint(3, 5)
    block_h = random.randint(2, 3)
    block_w = random.randint(2, 3)
    mask, target_patch = _choose_target_mask(m, n)
    color_decoy_patch = _different_patch(target_patch)
    if color_decoy_patch is None:
        return None

    lattice = _render_lattice(
        mask,
        block_h=block_h,
        block_w=block_w,
        bg=bg,
        key_color=key_color,
        other_color=other_color,
    )
    lh, lw = lattice.shape

    extra_h = random.randint(max(6, color_decoy_patch.shape[0] + 4), 9)
    extra_w = random.randint(max(6, target_patch.shape[1] + 4), 9)
    if lh + extra_h > 30 or lw + extra_w > 30:
        return None
    h = lh + extra_h
    w = lw + extra_w

    if corner in {"TL", "TR"}:
        r0 = 0
    else:
        r0 = extra_h
    if corner in {"TL", "BL"}:
        c0 = 0
    else:
        c0 = extra_w
    r1 = r0 + lh - 1
    c1 = c0 + lw - 1

    grid = np.full((h, w), bg, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = lattice

    alt_shape_color = random.choice([other_color, shape_decoy_color])
    specs = [
        {"patch": target_patch, "color": alt_shape_color, "mode": "vside", "role": "shape_decoy"},
        {"patch": color_decoy_patch, "color": key_color, "mode": "hside", "role": "color_decoy"},
        {"patch": target_patch, "color": key_color, "mode": "random", "role": "correct"},
    ]

    if random.random() < 0.6:
        extra_shape_color = random.choice([neutral_color, other_color, shape_decoy_color])
        if extra_shape_color != key_color:
            specs.append({"patch": target_patch, "color": extra_shape_color, "mode": "random", "role": "shape_decoy"})

    if random.random() < 0.6:
        extra_patch = _different_patch(target_patch)
        if extra_patch is None:
            return None
        specs.append({"patch": extra_patch, "color": key_color, "mode": "random", "role": "color_decoy"})

    for _ in range(random.randint(1, 2)):
        area = random.randint(2, 5)
        patch = _polyomino_patch(area, max_h=4, max_w=4)
        if patch is None:
            return None
        if np.array_equal(patch, target_patch):
            continue
        color = random.choice([c for c in PALETTE if c not in {bg}])
        if color == key_color and np.array_equal(patch, target_patch):
            continue
        specs.append({"patch": patch, "color": color, "mode": "random", "role": "neutral"})

    random.shuffle(specs)
    specs.sort(key=lambda spec: {"vside": 0, "hside": 1, "random": 2}[spec["mode"]])
    if not _place_sprites(grid, (r0, c0, r1, c1), specs):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return False

    thumb = analysis["thumbnail"]
    if thumb.shape[0] < 2 or thumb.shape[1] < 2:
        return False

    if len(analysis["shape_matches"]) < 2:
        return False
    if len(analysis["color_matches"]) < 2:
        return False

    exact = analysis["exact_match"]
    out = main(grid)
    if not np.array_equal(out, _crop_bbox(grid, exact["bbox"])):
        return False

    # Ensure the same-shape and same-color distractors are genuine single-key traps.
    target_mask = analysis["target_mask"]
    exact_color = int(analysis["minority_color"])
    has_shape_wrong_color = any(
        np.array_equal(comp["mask"], target_mask) and int(comp["color"]) != exact_color
        for comp in analysis["sprites"]
    )
    has_color_wrong_shape = any(
        int(comp["color"]) == exact_color and not np.array_equal(comp["mask"], target_mask)
        for comp in analysis["sprites"]
    )
    if not has_shape_wrong_color or not has_color_wrong_shape:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_instance()
        if grid is not None and _validate_instance(grid):
            return grid
    raise AssertionError("failed to generate a valid double-key lattice instance")

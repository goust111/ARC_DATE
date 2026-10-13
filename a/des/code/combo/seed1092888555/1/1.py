from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    dirs = DIR4 if connectivity == 4 else DIR8
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if not bool(mask[sr, sc]) or bool(seen[sr, sc]):
                continue
            q = deque([(int(sr), int(sc))])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_grid_border(pts: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(
            (pts[:, 0] == 0)
            | (pts[:, 0] == h - 1)
            | (pts[:, 1] == 0)
            | (pts[:, 1] == w - 1)
        )
    )


def _mask_from_points(points: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    out[points[:, 0], points[:, 1]] = True
    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_transform(arr: np.ndarray, flip_lr: bool, rot90: int) -> np.ndarray:
    out = np.fliplr(arr) if flip_lr else arr.copy()
    return np.rot90(out, rot90)


def _transform_variants(mask: np.ndarray) -> list[tuple[np.ndarray, bool, int]]:
    out: list[tuple[np.ndarray, bool, int]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip_lr in (False, True):
        for rot90 in range(4):
            cur = _apply_transform(mask, flip_lr, rot90)
            key = (cur.shape, cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((cur, flip_lr, rot90))
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((m.shape, m.tobytes()) for m, _, _ in _transform_variants(mask))


def _shape_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not bool(mask[nr, nc]):
                out[r, c] = True
                break
    return out


def _frame_from_hole_mask(hole_mask: np.ndarray) -> np.ndarray:
    h, w = hole_mask.shape
    out = np.zeros_like(hole_mask, dtype=bool)
    for r, c in np.argwhere(hole_mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out & (~hole_mask)


def _paste_mask(dst: np.ndarray, patch: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    h, w = mask.shape
    sub = dst[top : top + h, left : left + w]
    sub[mask] = patch[mask]


def _find_holes(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    holes: list[dict] = []
    for comp in _components(grid == bg, connectivity=4):
        if _touches_grid_border(comp, h, w):
            continue
        pts_set = {(int(r), int(c)) for r, c in comp}
        neighbor_colors: set[int] = set()
        ok = True
        for r, c in comp:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pts_set:
                    neighbor_colors.add(int(grid[nr, nc]))
                    if int(grid[nr, nc]) != GREY:
                        ok = False
        if not ok:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        local = comp - np.array([[r0, c0]], dtype=int)
        mask = _mask_from_points(local, r1 - r0 + 1, c1 - c0 + 1)
        holes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
            }
        )
    return holes


def _find_templates(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    out: list[dict] = []
    mask = (grid != bg) & (grid != GREY)
    for comp in _components(mask, connectivity=8):
        if not _touches_grid_border(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        local = comp - np.array([[r0, c0]], dtype=int)
        occ = _mask_from_points(local, r1 - r0 + 1, c1 - c0 + 1)
        patch = np.zeros_like(occ, dtype=int)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        patch[occ] = sub[occ]
        colors = sorted(int(x) for x in np.unique(patch[occ]))
        if len(colors) != 2:
            continue
        out.append(
            {
                "bbox": (r0, c0, r1, c1),
                "points": comp,
                "mask": occ,
                "patch": patch,
            }
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = BG if int((g == BG).sum()) >= int((g != BG).sum()) else int(np.bincount(g.ravel()).argmax())
    out = g.copy()

    holes = _find_holes(g, bg)
    templates = _find_templates(g, bg)
    used: set[int] = set()

    for hole in holes:
        matches: list[tuple[int, np.ndarray]] = []
        hole_mask = hole["mask"]
        for idx, tpl in enumerate(templates):
            if idx in used:
                continue
            for transformed_mask, flip_lr, rot90 in _transform_variants(tpl["mask"]):
                if transformed_mask.shape != hole_mask.shape:
                    continue
                if np.array_equal(transformed_mask, hole_mask):
                    transformed_patch = _apply_transform(tpl["patch"], flip_lr, rot90)
                    matches.append((idx, transformed_patch))
        assert len(matches) == 1, "each hole must match exactly one template"
        idx, patch = matches[0]
        used.add(idx)
        r0, c0, _r1, _c1 = hole["bbox"]
        _paste_mask(out, patch, r0, c0, hole_mask)

    for tpl in templates:
        out[tpl["points"][:, 0], tpl["points"][:, 1]] = bg

    return out


def _shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
        [[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]],
    ]
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        interior = mask & (~_shape_boundary(mask))
        if int(interior.sum()) == 0:
            continue
        if len(_transform_variants(mask)) < 4:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
    return out


SHAPES = _shape_library()


def _build_template_patch(shape: np.ndarray, frame_color: int, fill_color: int) -> np.ndarray:
    patch = np.full(shape.shape, BG, dtype=int)
    boundary = _shape_boundary(shape)
    patch[shape] = fill_color
    patch[boundary] = frame_color
    return patch


def _choose_color_pair(rng: random.Random) -> tuple[int, int]:
    frame_color, fill_color = rng.sample(PALETTE, 2)
    return int(frame_color), int(fill_color)


def _random_rows(rng: random.Random, n: int) -> list[int]:
    if n == 1:
        return [1]
    if n == 2:
        return rng.choice([[2], [1, 1]])  # type: ignore[arg-type]
    if n == 3:
        return rng.choice([[3], [2, 1], [1, 2]])  # type: ignore[arg-type]
    return rng.choice([[2, 2], [3, 1], [1, 3]])  # type: ignore[arg-type]


def _layout_region(rng: random.Random, holes: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray]]]:
    holes = holes[:]
    rng.shuffle(holes)
    row_sizes = list(_random_rows(rng, len(holes)))
    rows: list[list[np.ndarray]] = []
    idx = 0
    for size in row_sizes:
        rows.append(holes[idx : idx + size])
        idx += size

    margin = 1
    gap = 2
    row_heights = [max(h.shape[0] for h in row) for row in rows]
    row_widths = [sum(h.shape[1] for h in row) + gap * (len(row) - 1) for row in rows]
    inner_h = sum(row_heights) + gap * (len(rows) - 1)
    inner_w = max(row_widths)
    region_h = inner_h + 2 * margin
    region_w = inner_w + 2 * margin

    region = np.full((region_h, region_w), GREY, dtype=int)
    placements: list[tuple[int, int, np.ndarray]] = []
    row_top = margin
    for row, row_h, row_w in zip(rows, row_heights, row_widths, strict=False):
        extra_x = int(rng.randint(0, max(0, inner_w - row_w)))
        col_left = margin + extra_x
        for hole in row:
            max_drop = row_h - hole.shape[0]
            drop = int(rng.randint(0, max_drop))
            r0 = row_top + drop
            c0 = col_left
            region[r0 : r0 + hole.shape[0], c0 : c0 + hole.shape[1]][hole] = BG
            placements.append((r0, c0, hole))
            col_left += hole.shape[1] + gap
        row_top += row_h + gap
    return region, placements


def _can_place_bbox(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = occ.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _place_regions(
    rng: random.Random,
    grid: np.ndarray,
    occ: np.ndarray,
    regions: list[np.ndarray],
) -> list[tuple[int, int]]:
    H, W = grid.shape
    placed: list[tuple[int, int]] = []
    for region in regions:
        rh, rw = region.shape
        success = False
        for _ in range(400):
            top = rng.randint(2, H - rh - 3)
            left = rng.randint(2, W - rw - 3)
            if not _can_place_bbox(occ, top, left, rh, rw, pad=2):
                continue
            sub = region != BG
            grid[top : top + rh, left : left + rw][sub] = region[sub]
            occ[top : top + rh, left : left + rw] |= region != BG
            placed.append((top, left))
            success = True
            break
        if not success:
            raise RuntimeError("failed to place region")
    return placed


def _place_templates(
    rng: random.Random,
    grid: np.ndarray,
    occ: np.ndarray,
    patches: list[np.ndarray],
) -> list[tuple[int, int]]:
    H, W = grid.shape
    placed: list[tuple[int, int]] = []
    for patch in patches:
        ph, pw = patch.shape
        mask = patch != BG
        candidates: list[tuple[int, int]] = []
        for left in range(1, W - pw - 1):
            candidates.append((0, left))
            candidates.append((H - ph, left))
        for top in range(1, H - ph - 1):
            candidates.append((top, 0))
            candidates.append((top, W - pw))
        rng.shuffle(candidates)
        success = False
        for top, left in candidates:
            if not _can_place_bbox(occ, top, left, ph, pw, pad=1):
                continue
            sub = grid[top : top + ph, left : left + pw]
            if bool(np.any((sub != BG) & mask)):
                continue
            sub[mask] = patch[mask]
            occ[top : top + ph, left : left + pw] |= mask
            placed.append((top, left))
            success = True
            break
        if not success:
            raise RuntimeError("failed to place template")
    return placed


def _expected_output(
    inp: np.ndarray,
    hole_infos: list[tuple[tuple[int, int], np.ndarray]],
    tpl_infos: list[tuple[tuple[int, int], np.ndarray, np.ndarray]],
) -> np.ndarray:
    out = inp.copy()
    used = [False] * len(tpl_infos)
    for (hole_pos, hole_mask) in hole_infos:
        matches: list[tuple[int, np.ndarray]] = []
        for idx, (_tpl_pos, tpl_mask, tpl_patch) in enumerate(tpl_infos):
            if used[idx]:
                continue
            for oriented_mask, flip_lr, rot90 in _transform_variants(tpl_mask):
                if oriented_mask.shape != hole_mask.shape:
                    continue
                if np.array_equal(oriented_mask, hole_mask):
                    matches.append((idx, _apply_transform(tpl_patch, flip_lr, rot90)))
        if len(matches) != 1:
            raise RuntimeError("generator created ambiguous match")
        idx, oriented_patch = matches[0]
        used[idx] = True
        r0, c0 = hole_pos
        _paste_mask(out, oriented_patch, r0, c0, hole_mask)
    for (tpl_pos, tpl_mask, _tpl_patch) in tpl_infos:
        r0, c0 = tpl_pos
        h, w = tpl_mask.shape
        out[r0 : r0 + h, c0 : c0 + w][tpl_mask] = BG
    return out


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        total_holes = rng.randint(3, min(5, len(SHAPES)))
        chosen_shapes = rng.sample(SHAPES, total_holes)
        holes_oriented = [
            _apply_transform(shape, bool(rng.getrandbits(1)), rng.randrange(4))
            for shape in chosen_shapes
        ]

        region_count = rng.randint(1, min(3, total_holes))
        split = [1] * region_count
        for _i in range(total_holes - region_count):
            split[rng.randrange(region_count)] += 1
        rng.shuffle(split)

        regions: list[np.ndarray] = []
        region_holes: list[list[tuple[int, int, np.ndarray]]] = []
        idx = 0
        try:
            for count in split:
                region, placements = _layout_region(rng, holes_oriented[idx : idx + count])
                regions.append(region)
                region_holes.append(placements)
                idx += count
        except Exception:
            continue

        all_template_patches: list[np.ndarray] = []
        hole_infos: list[tuple[tuple[int, int], np.ndarray]] = []
        tpl_local_infos: list[tuple[np.ndarray, np.ndarray]] = []
        for shape in chosen_shapes:
            frame_color, fill_color = _choose_color_pair(rng)
            base_patch = _build_template_patch(shape, frame_color, fill_color)
            tpl_patch = _apply_transform(base_patch, bool(rng.getrandbits(1)), rng.randrange(4))
            tpl_mask = tpl_patch != BG
            all_template_patches.append(tpl_patch)
            tpl_local_infos.append((tpl_mask, tpl_patch))

        H = rng.randint(24, 32)
        W = rng.randint(24, 34)
        grid = np.full((H, W), BG, dtype=int)
        occ = np.zeros((H, W), dtype=bool)

        try:
            region_positions = _place_regions(rng, grid, occ, regions)
            template_positions = _place_templates(rng, grid, occ, all_template_patches)
        except Exception:
            continue

        abs_holes: list[tuple[tuple[int, int], np.ndarray]] = []
        for (rt, ct), placements in zip(region_positions, region_holes, strict=False):
            for hr, hc, hmask in placements:
                abs_holes.append(((rt + hr, ct + hc), hmask))

        tpl_infos: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
        for pos, (mask, patch) in zip(template_positions, tpl_local_infos, strict=False):
            tpl_infos.append((pos, mask, patch))

        try:
            expected = _expected_output(grid, abs_holes, tpl_infos)
        except Exception:
            continue

        try:
            actual = main(grid)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if len(_find_holes(grid, BG)) != total_holes:
            continue
        if len(_find_templates(grid, BG)) != total_holes:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")

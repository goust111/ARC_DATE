from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

SPRITE_COLORS = (BLUE, RED, GREEN, YELLOW)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALL_DIRS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
GEN_ATTEMPTS = 400


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


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    dirs = ORTHO_DIRS if connectivity == 4 else ALL_DIRS
    out = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == PINK)]
    assert len(rows) == 1, "expected exactly one full magenta separator row"
    sep = rows[0]
    assert sep == grid.shape[0] - sep - 1, "separator must split equal-height halves"
    return int(sep)


def _extract_sprites(top: np.ndarray) -> list[dict]:
    sprites: list[dict] = []
    for coords in _components(top != BLACK, connectivity=8):
        rr = coords[:, 0]
        cc = coords[:, 1]
        box = (int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max()))
        sprite = _crop_bbox(top, box)
        grey_coords = np.argwhere(sprite == GREY)
        assert len(grey_coords) == 1, "each sprite must have exactly one grey anchor"
        colors = sorted(int(v) for v in np.unique(sprite) if v not in {BLACK, GREY})
        assert len(colors) == 1, "each sprite must contain exactly one main color"
        sprites.append(
            {
                "bbox": box,
                "left_col": box[1],
                "color": colors[0],
                "sprite": sprite,
                "anchor": tuple(map(int, grey_coords[0])),
            }
        )
    assert 3 <= len(sprites) <= 4, "expected 3 or 4 upper sprites"
    return sprites


def _valid_anchor_positions(sprites: list[dict], top_h: int, width: int) -> list[tuple[int, int]]:
    row_lo = max(sprite["anchor"][0] for sprite in sprites)
    row_hi = min(top_h - 1 - (sprite["sprite"].shape[0] - 1 - sprite["anchor"][0]) for sprite in sprites)
    col_lo = max(sprite["anchor"][1] for sprite in sprites)
    col_hi = min(width - 1 - (sprite["sprite"].shape[1] - 1 - sprite["anchor"][1]) for sprite in sprites)
    if row_lo > row_hi or col_lo > col_hi:
        return []
    return [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]


def _build_composite(
    sprites: list[dict], anchor: tuple[int, int], top_shape: tuple[int, int]
) -> tuple[np.ndarray, int]:
    top_h, width = top_shape
    out = np.zeros((top_h, width), dtype=int)
    conflicts = 0
    ar, ac = anchor
    for sprite in sorted(sprites, key=lambda item: item["left_col"]):
        patch = sprite["sprite"]
        sr, sc = sprite["anchor"]
        for pr, pc in np.argwhere(patch != BLACK):
            tr = ar + int(pr) - sr
            tc = ac + int(pc) - sc
            assert 0 <= tr < top_h and 0 <= tc < width
            if out[tr, tc] != BLACK and (tr, tc) != anchor:
                conflicts += 1
            out[tr, tc] = int(patch[pr, pc])
    out[ar, ac] = BLACK
    return out, conflicts


def _mirror_to_bottom(composite: np.ndarray) -> np.ndarray:
    return composite[::-1].copy()


def _mismatch_count(lower: np.ndarray, reference: np.ndarray) -> int:
    return int(np.count_nonzero(lower != reference))


def _best_reference(
    sprites: list[dict], lower: np.ndarray, top_shape: tuple[int, int]
) -> tuple[tuple[int, int], np.ndarray, np.ndarray, int, int]:
    candidates = []
    for anchor in _valid_anchor_positions(sprites, *top_shape):
        composite, conflicts = _build_composite(sprites, anchor, top_shape)
        reference = _mirror_to_bottom(composite)
        mismatch = _mismatch_count(lower, reference)
        candidates.append((mismatch, anchor, composite, reference, conflicts))
    assert candidates, "no valid anchor-alignment position"
    candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    best = candidates[0]
    gap = None
    if len(candidates) > 1:
        gap = candidates[1][0] - best[0]
    return best[1], best[2], best[3], best[0], (999 if gap is None else int(gap))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the upper sprites by their gray anchors, mirror the composite, and erase lower-half pixels that disagree with that mirrored reference.

    Input structure:
    - A black background is split by one full magenta separator row into equal-height upper and lower halves.
    - The upper half contains 3 or 4 separated sprites; each sprite has one gray anchor pixel and pixels of exactly one main color.
    - The lower half contains a noisy version of one mirrored composite built from those upper sprites.

    Rule:
    - Crop every upper sprite to its bounding box.
    - Find the unique anchor-coincidence placement whose mirrored composite best matches the lower half.
    - Overlay sprites at that common anchor, resolving non-black conflicts by the sprites' left-to-right order in the upper half, then force the shared anchor cell to black.
    - Mirror that composite across the separator row and compare it pixel-by-pixel with the lower half; every lower pixel that disagrees with the mirrored reference becomes black.

    Key insight:
    - The upper sprites are registration fragments: only after anchor alignment and separator mirroring do they define the reference mask that reveals which lower pixels are genuine and which are errors to erase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep = _separator_row(g)
    top = g[:sep]
    lower = g[sep + 1 :]
    sprites = _extract_sprites(top)
    _anchor, _composite, reference, _mismatch, _gap = _best_reference(sprites, lower, top.shape)

    out = g.copy()
    out[sep + 1 :] = np.where(lower == reference, lower, BLACK)
    return out


def _path_between(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    if row_first:
        step = 1 if tr >= r else -1
        for nr in range(r + step, tr + step, step):
            cells.append((nr, c))
        step = 1 if tc >= c else -1
        for nc in range(c + step, tc + step, step):
            cells.append((tr, nc))
    else:
        step = 1 if tc >= c else -1
        for nc in range(c + step, tc + step, step):
            cells.append((r, nc))
        step = 1 if tr >= r else -1
        for nr in range(r + step, tr + step, step):
            cells.append((nr, tc))
    return cells


def _make_sprite(color: int, h: int, w: int, anchor: tuple[int, int]) -> np.ndarray | None:
    for _ in range(60):
        cells = {anchor}
        edge_targets = [
            (0, random.randint(0, w - 1)),
            (h - 1, random.randint(0, w - 1)),
            (random.randint(0, h - 1), 0),
            (random.randint(0, h - 1), w - 1),
        ]
        for target in edge_targets:
            for cell in _path_between(anchor, target, row_first=bool(random.getrandbits(1))):
                cells.add(cell)
        extra_budget = random.randint(0, 2)
        frontier = []
        for r, c in tuple(cells):
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        random.shuffle(frontier)
        for cell in frontier[:extra_budget]:
            cells.add(cell)

        arr = np.zeros((h, w), dtype=int)
        for r, c in cells:
            arr[r, c] = color
        arr[anchor] = GREY

        box = _bbox(arr != BLACK)
        if box != (0, 0, h - 1, w - 1):
            continue
        if len(_components(arr != BLACK, connectivity=4)) != 1:
            continue
        if len(cells) < 5 or len(cells) >= h * w:
            continue
        return arr
    return None


def _can_place(occupied: np.ndarray, sprite: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    for sr, sc in np.argwhere(sprite != BLACK):
        tr = r0 + int(sr)
        tc = c0 + int(sc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _place_upper_sprites(sprites: list[np.ndarray], top_h: int, width: int) -> list[tuple[int, int]] | None:
    for _ in range(250):
        occupied = np.zeros((top_h, width), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(sprites)
        ok = True
        order = list(range(len(sprites)))
        random.shuffle(order)
        for idx in order:
            sprite = sprites[idx]
            sh, sw = sprite.shape
            positions = [
                (r, c)
                for r in range(top_h - sh + 1)
                for c in range(width - sw + 1)
                if _can_place(occupied, sprite, r, c)
            ]
            if not positions:
                ok = False
                break
            r0, c0 = random.choice(positions)
            placements[idx] = (r0, c0)
            for sr, sc in np.argwhere(sprite != BLACK):
                occupied[r0 + int(sr), c0 + int(sc)] = True
        if not ok:
            continue
        if any(pos is None for pos in placements):
            continue
        placed = [pos for pos in placements if pos is not None]
        if len({c for _, c in placed}) != len(placed):
            continue
        return placed
    return None


def _inject_errors(reference: np.ndarray, colors: tuple[int, ...]) -> tuple[np.ndarray, dict[str, int]]:
    lower = reference.copy()
    ref_nonblack = [tuple(map(int, rc)) for rc in np.argwhere(reference != BLACK)]
    ref_box = _bbox(reference != BLACK)
    assert ref_box is not None
    r0, c0, r1, c1 = ref_box
    near_black = [
        (r, c)
        for r in range(max(0, r0 - 1), min(reference.shape[0], r1 + 2))
        for c in range(max(0, c0 - 1), min(reference.shape[1], c1 + 2))
        if reference[r, c] == BLACK
    ]
    all_black = [tuple(map(int, rc)) for rc in np.argwhere(reference == BLACK)]
    extra_pool = near_black if near_black else all_black

    error_total = random.randint(2, 4)
    visible_type = random.choice(["extra", "wrong"])
    type_plan = [visible_type]
    if random.random() < 0.7 and ref_nonblack:
        type_plan.append("missing")
    while len(type_plan) < error_total:
        candidates = ["extra", "wrong", "missing"]
        choice = random.choice(candidates)
        if choice == "extra" and not extra_pool:
            continue
        if choice in {"wrong", "missing"} and not ref_nonblack:
            continue
        type_plan.append(choice)

    changed: set[tuple[int, int]] = set()
    counts = {"extra": 0, "wrong": 0, "missing": 0}
    for kind in type_plan:
        if kind == "extra":
            pool = [cell for cell in extra_pool if cell not in changed]
            if not pool:
                continue
            cell = random.choice(pool)
            lower[cell] = random.choice(colors)
        elif kind == "missing":
            pool = [cell for cell in ref_nonblack if cell not in changed]
            if not pool:
                continue
            cell = random.choice(pool)
            lower[cell] = BLACK
        else:
            pool = [cell for cell in ref_nonblack if cell not in changed and reference[cell] in colors]
            if not pool:
                continue
            cell = random.choice(pool)
            lower[cell] = random.choice([color for color in colors if color != reference[cell]])
        changed.add(cell)
        counts[kind] += 1

    assert 2 <= len(changed) <= 4
    assert counts["extra"] + counts["wrong"] >= 1
    return lower, counts


def _validate_instance(
    grid: np.ndarray,
    anchor: tuple[int, int],
    reference: np.ndarray,
    error_counts: dict[str, int],
) -> bool:
    try:
        if grid.ndim != 2 or max(grid.shape) > 30:
            return False
        sep = _separator_row(grid)
        if sep < 7 or sep > 10:
            return False
        top = grid[:sep]
        lower = grid[sep + 1 :]
        if np.any(lower == GREY):
            return False

        sprites = _extract_sprites(top)
        best_anchor, _composite, best_reference, mismatch, gap = _best_reference(sprites, lower, top.shape)
        if best_anchor != anchor:
            return False
        if mismatch != sum(error_counts.values()):
            return False
        if gap < 2:
            return False
        if not np.array_equal(best_reference, reference):
            return False

        visible_colors = sorted(int(v) for v in np.unique(reference) if v != BLACK)
        if len(visible_colors) < 2 or np.count_nonzero(reference) < 6:
            return False

        expected = grid.copy()
        expected[sep + 1 :] = np.where(lower == reference, lower, BLACK)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(actual, grid):
            return False
        if not np.any((lower != reference) & (lower != BLACK)):
            return False
        return True
    except Exception:
        return False


def _generate_case() -> np.ndarray | None:
    top_h = random.randint(7, 10)
    width = random.randint(13, 18)
    sprite_count = random.randint(3, 4)
    chosen_colors = tuple(random.sample(SPRITE_COLORS, sprite_count))

    sprites: list[np.ndarray] = []
    used_anchor_shapes: set[tuple[int, int, int, int]] = set()
    for color in chosen_colors:
        made = None
        for _ in range(80):
            h = random.randint(3, 4)
            w = random.randint(3, 5)
            anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
            key = (anchor[0], anchor[1], h, w)
            if key in used_anchor_shapes:
                continue
            sprite = _make_sprite(color, h, w, anchor)
            if sprite is None:
                continue
            used_anchor_shapes.add(key)
            made = sprite
            break
        if made is None:
            return None
        sprites.append(made)

    placements = _place_upper_sprites(sprites, top_h, width)
    if placements is None:
        return None

    upper = np.zeros((top_h, width), dtype=int)
    sprite_infos: list[dict] = []
    for sprite, (r0, c0) in zip(sprites, placements):
        sh, sw = sprite.shape
        mask = sprite != BLACK
        upper[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]
        sprite_infos.append(
            {
                "bbox": (r0, c0, r0 + sh - 1, c0 + sw - 1),
                "left_col": c0,
                "color": int(next(v for v in np.unique(sprite) if v not in {BLACK, GREY})),
                "sprite": sprite,
                "anchor": tuple(map(int, np.argwhere(sprite == GREY)[0])),
            }
        )

    anchors = _valid_anchor_positions(sprite_infos, top_h, width)
    if not anchors:
        return None
    need_overlap = random.random() < 0.7
    random.shuffle(anchors)
    chosen_anchor = None
    composite = None
    reference = None
    conflict_count = 0
    for anchor in anchors:
        candidate, conflicts = _build_composite(sprite_infos, anchor, (top_h, width))
        visible_colors = sorted(int(v) for v in np.unique(candidate) if v != BLACK)
        if len(visible_colors) < 2:
            continue
        if need_overlap and conflicts == 0:
            continue
        chosen_anchor = anchor
        composite = candidate
        reference = _mirror_to_bottom(candidate)
        conflict_count = conflicts
        break
    if chosen_anchor is None or composite is None or reference is None:
        return None
    if need_overlap and conflict_count == 0:
        return None

    lower, error_counts = _inject_errors(reference, chosen_colors)

    grid = np.zeros((top_h * 2 + 1, width), dtype=int)
    grid[:top_h] = upper
    grid[top_h, :] = PINK
    grid[top_h + 1 :] = lower

    if not _validate_instance(grid, chosen_anchor, reference, error_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-overlay mirror-erase puzzle")

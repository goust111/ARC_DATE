from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
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


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords.sort()
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _is_hole_free(mask: np.ndarray) -> bool:
    inv = ~mask
    h, w = inv.shape
    for comp in _components(inv):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            return False
    return True


def _find_cavities(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    cavities: list[dict] = []
    for coords in _components(grid == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        coord_set = set(coords)
        boundary: set[tuple[int, int]] = set()
        boundary_colors = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in coord_set:
                    continue
                val = int(grid[nr, nc])
                if val != BLACK:
                    boundary.add((nr, nc))
                    boundary_colors.add(val)
        if boundary_colors == {GREY}:
            cavities.append(
                {
                    "coords": coords,
                    "coord_set": coord_set,
                    "boundary_bbox": _bbox(sorted(boundary)),
                }
            )
    cavities.sort(key=lambda item: item["coords"][0])
    return cavities


def _singleton_labels(grid: np.ndarray) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        for comp in _components(grid == color):
            if len(comp) == 1:
                lookup[comp[0]] = color
    return lookup


def _cavity_labels(grid: np.ndarray, cavities: list[dict]) -> dict[int, dict]:
    singletons = _singleton_labels(grid)
    mapping: dict[int, dict] = {}
    for cavity in cavities:
        br0, bc0, _, _ = cavity["boundary_bbox"]
        candidates: list[tuple[tuple[int, int], int]] = []
        for rr, cc in ((br0 - 1, bc0 - 1), (br0 - 1, bc0), (br0, bc0 - 1)):
            if (rr, cc) in singletons:
                candidates.append(((rr, cc), singletons[(rr, cc)]))
        if not candidates:
            for rr in range(max(0, br0 - 1), min(grid.shape[0], br0 + 1)):
                for cc in range(max(0, bc0 - 1), min(grid.shape[1], bc0 + 1)):
                    if (rr, cc) in singletons:
                        candidates.append(((rr, cc), singletons[(rr, cc)]))
        assert len(candidates) == 1, "each cavity must have one nearby singleton label"
        (lr, lc), color = candidates[0]
        assert color not in mapping, "label colors must be unique"
        mapping[color] = {"label": (lr, lc), "cavity": cavity}
    return mapping


def _fragments_for_color(
    grid: np.ndarray, color: int, label_coord: tuple[int, int]
) -> list[dict]:
    mask = grid == color
    lr, lc = label_coord
    mask = mask.copy()
    mask[lr, lc] = False
    fragments = []
    for coords in _components(mask):
        r0, c0, _, _ = _bbox(coords)
        rel = sorted((r - r0, c - c0) for r, c in coords)
        fragments.append({"coords": coords, "rel": rel})
    fragments.sort(key=lambda item: item["coords"][0])
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack scattered colored fragments back into the matching labeled grey cavities.

    Input structure:
    - The background is black.
    - Grey one-pixel outlines enclose black cavity regions.
    - Each outline has one colored singleton label just outside its top-left corner.
    - Elsewhere, solid 4-connected fragments use those same label colors.

    Rule:
    - Find every enclosed black cavity whose boundary is grey and pair it with its nearby colored label.
    - For each label color, collect all same-colored fragments in top-to-bottom, then left-to-right source order.
    - Erase those fragments, then greedily place them into the matching cavity so each fragment's own top-left bbox corner lands on the earliest still-empty cavity cell in scan order.

    Key insight:
    - The singleton label identifies which fragments belong to which cavity, and the fill order is determined entirely by scan-order anchoring plus translation only.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    cavities = _find_cavities(g)
    labels = _cavity_labels(g, cavities)
    assert labels, "expected at least one labeled cavity"

    for color, info in labels.items():
        fragments = _fragments_for_color(g, color, info["label"])
        assert fragments, "every label color must have at least one fragment"
        cavity_coords = info["cavity"]["coords"]
        cavity_set = info["cavity"]["coord_set"]
        filled: set[tuple[int, int]] = set()

        for frag in fragments:
            for r, c in frag["coords"]:
                out[r, c] = BLACK

        for frag in fragments:
            anchor = next((pt for pt in cavity_coords if pt not in filled), None)
            assert anchor is not None, "fragments cannot exceed cavity size"
            targets = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in frag["rel"]]
            assert all(pt in cavity_set for pt in targets), "fragment must stay inside cavity"
            assert not any(pt in filled for pt in targets), "fragments must not overlap in cavity"
            for r, c in targets:
                out[r, c] = color
                filled.add((r, c))

        assert filled == cavity_set, "fragments must fill the cavity exactly"

    return out


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
GROUP_COUNT_CHOICES = [1, 2, 3]
GROUP_COUNT_WEIGHTS = [2, 3, 2]
OUTLINE_PLACE_TRIES = 80
FRAGMENT_PICK_LIMIT = 24
GEN_RETRIES = 400


def _random_cavity_mask(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        shape_kind = rng.choice(("rect", "step", "stair", "taper"))
        if shape_kind == "rect":
            h = rng.randint(2, 4)
            w = rng.randint(2, 5)
            mask = np.ones((h, w), dtype=bool)
        elif shape_kind == "step":
            h = rng.randint(3, 5)
            w = rng.randint(3, 6)
            split = rng.randint(1, h - 1)
            shift = rng.randint(1, w - 2)
            mask = np.zeros((h, w), dtype=bool)
            mask[:split, :] = True
            mask[split:, shift:] = True
        elif shape_kind == "stair":
            h = rng.randint(3, 5)
            w = rng.randint(3, 6)
            starts = [0]
            for _ in range(1, h):
                starts.append(min(starts[-1] + rng.choice((0, 0, 1)), w - 2))
            if max(starts) == 0:
                starts[-1] = 1
            mask = np.zeros((h, w), dtype=bool)
            for r, start in enumerate(starts):
                mask[r, start:] = True
        else:
            h = rng.randint(3, 5)
            w = rng.randint(4, 6)
            left = 0
            right = w - 1
            mask = np.zeros((h, w), dtype=bool)
            for r in range(h):
                if r > 0:
                    left = min(left + rng.choice((0, 0, 1)), w - 2)
                    right = max(right - rng.choice((0, 1)), left + 1)
                mask[r, left : right + 1] = True

        if rng.random() < 0.5:
            mask = mask.T
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)

        mask = _trim_mask(mask)
        comps = _components(mask)
        area = int(mask.sum())
        if len(comps) == 1 and _is_hole_free(mask) and 4 <= area <= 20:
            return mask

    raise RuntimeError("failed to sample a valid cavity mask")


def _grow_piece(
    remaining: set[tuple[int, int]],
    anchor: tuple[int, int],
    desired: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    piece = {anchor}
    while len(piece) < desired:
        frontier: set[tuple[int, int]] = set()
        for r, c in piece:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                cand = (nr, nc)
                if cand not in remaining or cand in piece:
                    continue
                if nr < anchor[0] or nc < anchor[1]:
                    continue
                frontier.add(cand)
        if not frontier:
            break
        frontier_list = sorted(frontier)
        limit = min(len(frontier_list), 4)
        piece.add(rng.choice(frontier_list[:limit]))
    return piece


def _partition_cavity(mask: np.ndarray, rng: random.Random) -> list[list[tuple[int, int]]]:
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    pieces: list[list[tuple[int, int]]] = []
    target_count = rng.randint(2, min(4, len(remaining)))

    while remaining:
        anchor = min(remaining)
        max_size = min(4, len(remaining))
        if not pieces and len(remaining) > 1:
            max_size = min(max_size, len(remaining) - 1)
        if len(pieces) < target_count - 1:
            max_size = min(max_size, len(remaining) - (target_count - len(pieces) - 1))
        size_choices = list(range(1, max_size + 1))
        weights = [1 if s == 1 else 3 if s in (2, 3) else 2 for s in size_choices]
        desired = rng.choices(size_choices, weights=weights, k=1)[0]
        piece = _grow_piece(remaining, anchor, desired, rng)
        remaining -= piece
        ar, ac = anchor
        pieces.append(sorted((r - ar, c - ac) for r, c in piece))

    assert len(pieces) >= 2
    return pieces


def _outline_patch(cavity_mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    ch, cw = cavity_mask.shape
    patch = np.zeros((ch + 4, cw + 4), dtype=int)
    cavity = np.zeros_like(patch, dtype=bool)
    cavity[2 : 2 + ch, 2 : 2 + cw] = cavity_mask
    shell = _dilate8(cavity) & ~cavity
    patch[shell] = GREY
    patch[0, 0] = color
    return patch, cavity


def _clear_rect(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not blocked[r0:r1, c0:c1].any()


def _mark_rect(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _place_outline(
    grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    h, w = grid.shape
    for _ in range(OUTLINE_PLACE_TRIES):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        if not _clear_rect(blocked, top, left, ph, pw, pad=1):
            continue
        vals = np.argwhere(patch != BLACK)
        for r, c in vals:
            grid[top + int(r), left + int(c)] = int(patch[r, c])
        _mark_rect(blocked, top, left, ph, pw, pad=1)
        return top, left
    return None


def _candidate_positions(
    blocked: np.ndarray,
    piece: list[tuple[int, int]],
    after: tuple[int, int],
) -> list[tuple[int, int]]:
    height = 1 + max(r for r, _ in piece)
    width = 1 + max(c for _, c in piece)
    h, w = blocked.shape
    out: list[tuple[int, int]] = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            if (top, left) <= after:
                continue
            if _clear_rect(blocked, top, left, height, width, pad=1):
                out.append((top, left))
    return out


def _place_pieces_for_group(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    pieces: list[list[tuple[int, int]]],
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    current_blocked = blocked.copy()
    placements: list[tuple[int, int]] = []
    after = (-1, -1)
    for idx, piece in enumerate(pieces):
        valid = _candidate_positions(current_blocked, piece, after)
        if not valid:
            return None
        keep = max(1, min(len(valid), max(6, len(valid) // 2)))
        pool = valid[:keep]
        if len(pool) > FRAGMENT_PICK_LIMIT:
            pool = rng.sample(pool, FRAGMENT_PICK_LIMIT)
        top, left = rng.choice(pool)
        placements.append((top, left))
        height = 1 + max(r for r, _ in piece)
        width = 1 + max(c for _, c in piece)
        _mark_rect(current_blocked, top, left, height, width, pad=1)
        after = (top, left)

    coords_by_piece: list[list[tuple[int, int]]] = []
    for piece, (top, left) in zip(pieces, placements):
        coords = [(top + r, left + c) for r, c in piece]
        coords_by_piece.append(coords)
        for r, c in coords:
            grid[r, c] = color

    blocked[:, :] = current_blocked
    return coords_by_piece


def _replay_partition(
    cavity_coords: list[tuple[int, int]],
    pieces: list[list[tuple[int, int]]],
) -> set[tuple[int, int]]:
    cavity_set = set(cavity_coords)
    filled: set[tuple[int, int]] = set()
    for piece in pieces:
        anchor = next(pt for pt in cavity_coords if pt not in filled)
        targets = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in piece}
        assert targets <= cavity_set
        assert not (targets & filled)
        filled |= targets
    return filled


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    group_count = rng.choices(GROUP_COUNT_CHOICES, weights=GROUP_COUNT_WEIGHTS, k=1)[0]
    colors = rng.sample(COLORS, group_count)

    groups = []
    for color in colors:
        cavity_mask = _random_cavity_mask(rng)
        pieces = _partition_cavity(cavity_mask, rng)
        patch, cavity_local = _outline_patch(cavity_mask, color)
        placed = _place_outline(grid, blocked, patch, rng)
        if placed is None:
            return None
        top, left = placed
        cavity_coords = [
            (top + int(r), left + int(c))
            for r, c in np.argwhere(cavity_local)
        ]
        cavity_coords.sort()
        assert _replay_partition(cavity_coords, pieces) == set(cavity_coords)
        groups.append(
            {
                "color": color,
                "pieces": pieces,
                "cavity_coords": cavity_coords,
            }
        )

    groups.sort(key=lambda item: -len(item["cavity_coords"]))
    for group in groups:
        coords_by_piece = _place_pieces_for_group(
            grid, blocked, group["color"], group["pieces"], rng
        )
        if coords_by_piece is None:
            return None
        group["piece_coords"] = coords_by_piece

    expected = grid.copy()
    for group in groups:
        for piece_coords in group["piece_coords"]:
            for r, c in piece_coords:
                expected[r, c] = BLACK
        for r, c in group["cavity_coords"]:
            expected[r, c] = group["color"]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle instance")

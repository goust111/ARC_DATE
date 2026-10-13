from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


BG = 0
ARC_MAX = 30
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

# Generator constraints.
MIN_PIECE_AREA = 3
RECT_H_RANGE = (3, 7)  # inclusive low, exclusive high
RECT_W_RANGE = (3, 9)
MAX_RECT_AREA = 24
GROUP_RETRIES = 120
SAMPLE_RETRIES = 400
PLACEMENT_RETRIES = 300


def _choice(seq):
    return seq[int(RNG.integers(len(seq)))]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in row) for row in mask.astype(int))


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in coords]
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _piece_from_coords(coords: Iterable[tuple[int, int]]) -> dict[str, object]:
    mask = _normalize_coords(coords)
    return {
        "mask": mask,
        "coords": _coords_from_mask(mask),
        "area": int(mask.sum()),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
    }


def _find_color_components(grid: np.ndarray) -> dict[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            groups[color].append(_piece_from_coords(coords))
    return groups


def _candidate_rectangle_dims(total_area: int, pieces: list[dict[str, object]]) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(1, int(total_area ** 0.5) + 1):
        if total_area % h:
            continue
        for H, W in ((h, total_area // h), (total_area // h, h)):
            dim = (int(H), int(W))
            if dim in dims:
                continue
            if any(int(piece["h"]) > dim[0] or int(piece["w"]) > dim[1] for piece in pieces):
                continue
            dims.append(dim)
    dims.sort(key=lambda hw: (max(hw), min(hw)))
    return dims


def _placement_mask(piece: dict[str, object], H: int, W: int, top: int, left: int) -> int:
    mask = 0
    for r, c in piece["coords"]:
        idx = (int(top) + int(r)) * int(W) + (int(left) + int(c))
        mask |= 1 << idx
    return mask


def _count_rectangle_assemblies(
    pieces: list[dict[str, object]],
    H: int,
    W: int,
    limit: int = 2,
) -> int:
    if sum(int(piece["area"]) for piece in pieces) != int(H * W):
        return 0

    candidates: list[list[int]] = []
    for piece in pieces:
        piece_h = int(piece["h"])
        piece_w = int(piece["w"])
        masks = []
        for top in range(int(H - piece_h + 1)):
            for left in range(int(W - piece_w + 1)):
                masks.append(_placement_mask(piece, H, W, top, left))
        if not masks:
            return 0
        candidates.append(masks)

    order = sorted(range(len(pieces)), key=lambda idx: (len(candidates[idx]), -int(pieces[idx]["area"])))
    found = 0

    def dfs(pos: int, occupied: int) -> None:
        nonlocal found
        if found >= limit:
            return
        if pos == len(order):
            found += 1
            return
        piece_idx = order[pos]
        for placed_mask in candidates[piece_idx]:
            if placed_mask & occupied:
                continue
            dfs(pos + 1, occupied | placed_mask)
            if found >= limit:
                return

    dfs(0, 0)
    return found


def _reconstruct_rectangle(pieces: list[dict[str, object]]) -> tuple[int, int]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    solution_dims: tuple[int, int] | None = None
    solution_count = 0

    for H, W in _candidate_rectangle_dims(total_area, pieces):
        count = _count_rectangle_assemblies(pieces, H, W, limit=2)
        if count == 0:
            continue
        if solution_dims is None:
            solution_dims = (int(H), int(W))
        solution_count += count
        if solution_count > 1:
            break

    assert solution_dims is not None, "same-color pieces must reassemble into a rectangle"
    assert solution_count == 1, "same-color pieces must have a unique rectangle assembly"
    return solution_dims


def _pack_rectangles(rectangles: list[dict[str, int]]) -> np.ndarray:
    rows = []
    for piece_count in (2, 3, 4):
        row = [rect for rect in rectangles if int(rect["count"]) == piece_count]
        if not row:
            continue
        row.sort(key=lambda rect: (int(rect["w"]), int(rect["h"]), int(rect["color"])))
        row_h = max(int(rect["h"]) for rect in row)
        row_w = sum(int(rect["w"]) for rect in row) + len(row) - 1
        rows.append((row, row_h, row_w))

    out_h = sum(row_h for _row, row_h, _row_w in rows) + max(0, len(rows) - 1)
    out_w = max((row_w for _row, _row_h, row_w in rows), default=0)
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor_r = 0
    for row_index, (row, _row_h, _row_w) in enumerate(rows):
        cursor_c = 0
        for rect in row:
            h = int(rect["h"])
            w = int(rect["w"])
            out[cursor_r : cursor_r + h, cursor_c : cursor_c + w] = int(rect["color"])
            cursor_c += w + 1
        cursor_r += max(int(rect["h"]) for rect in row)
        if row_index != len(rows) - 1:
            cursor_r += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored exploded fragments into rectangles, then sort the rectangles by fragment count into rows.

    Input structure:
    - The background is black.
    - Several monochrome 4-connected fragments are scattered across the grid.
    - Fragments of the same color belong to one original object and can be translated, without rotation or reflection, to form one solid rectangle.
    - Each color group contains between 2 and 4 fragments.

    Rule:
    - Group fragments by color and, for each group, find the unique translation-only arrangement whose union is a filled rectangle.
    - Replace each group by that reconstructed solid rectangle in the output.
    - Put 2-fragment rectangles in the first row, 3-fragment rectangles in the second row, and 4-fragment rectangles in the third row.
    - Within each row, sort rectangles by width from small to large and separate neighbors by one black column; separate used rows by one black row.

    Key insight:
    - The important intermediate result is not the fragment shapes alone but the number of fragments needed to rebuild each rectangle; that reconstructed piece count determines the final row assignment.
    """
    g = np.asarray(grid, dtype=int)
    groups = _find_color_components(g)
    assert groups

    rectangles: list[dict[str, int]] = []
    for color, pieces in groups.items():
        piece_count = len(pieces)
        assert 2 <= piece_count <= 4
        rect_h, rect_w = _reconstruct_rectangle(pieces)
        rectangles.append(
            {
                "color": int(color),
                "count": int(piece_count),
                "h": int(rect_h),
                "w": int(rect_w),
            }
        )

    return _pack_rectangles(rectangles)


def _sample_piece_counts() -> list[int]:
    counts = [2, 3, 4]
    extra_groups = int(RNG.integers(1, 3))
    for _ in range(extra_groups):
        counts.append(int(_choice([2, 3, 4])))
    RNG.shuffle(counts)
    return counts


def _sample_piece_sizes(area: int, piece_count: int) -> list[int]:
    sizes = [MIN_PIECE_AREA] * piece_count
    extra = area - MIN_PIECE_AREA * piece_count
    for _ in range(extra):
        sizes[int(RNG.integers(piece_count))] += 1
    RNG.shuffle(sizes)
    return sizes


def _sample_seeds(h: int, w: int, piece_count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    RNG.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - prev_r) + abs(cell[1] - prev_c) >= 2 for prev_r, prev_c in seeds):
            seeds.append(cell)
            if len(seeds) == piece_count:
                return seeds
    return cells[:piece_count]


def _partition_rectangle(h: int, w: int, piece_count: int) -> list[dict[str, object]] | None:
    area = int(h * w)
    if area < MIN_PIECE_AREA * piece_count:
        return None

    for _ in range(GROUP_RETRIES):
        targets = _sample_piece_sizes(area, piece_count)
        labels = np.full((h, w), -1, dtype=int)
        sizes = [0] * piece_count
        frontiers = [set() for _ in range(piece_count)]

        for idx, (r, c) in enumerate(_sample_seeds(h, w, piece_count)):
            labels[int(r), int(c)] = idx
            sizes[idx] = 1

        for idx in range(piece_count):
            r, c = np.argwhere(labels == idx)[0]
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) == -1:
                    frontiers[idx].add((nr, nc))

        assigned = piece_count
        success = True
        while assigned < area:
            growable = [idx for idx in range(piece_count) if sizes[idx] < targets[idx] and frontiers[idx]]
            if not growable:
                success = False
                break
            weights = np.array([targets[idx] - sizes[idx] for idx in growable], dtype=float)
            weights /= weights.sum()
            grow_idx = int(growable[int(RNG.choice(len(growable), p=weights))])
            cell = _choice(list(frontiers[grow_idx]))
            r, c = int(cell[0]), int(cell[1])
            if int(labels[r, c]) != -1:
                frontiers[grow_idx].discard((r, c))
                continue

            labels[r, c] = grow_idx
            sizes[grow_idx] += 1
            assigned += 1
            for frontier in frontiers:
                frontier.discard((r, c))
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) == -1:
                    frontiers[grow_idx].add((nr, nc))

        if not success:
            continue

        pieces = []
        for idx in range(piece_count):
            coords = [(int(r), int(c)) for r, c in np.argwhere(labels == idx)]
            piece = _piece_from_coords(coords)
            pieces.append(piece)

        if len({_mask_signature(piece["mask"]) for piece in pieces}) != piece_count:
            continue
        if not any(int(piece["area"]) < int(piece["h"]) * int(piece["w"]) for piece in pieces):
            continue
        try:
            recon_h, recon_w = _reconstruct_rectangle(pieces)
        except AssertionError:
            continue
        if (int(recon_h), int(recon_w)) != (int(h), int(w)):
            continue
        return pieces

    return None


def _sample_rect_dims(piece_count: int) -> tuple[int, int]:
    while True:
        h = int(RNG.integers(*RECT_H_RANGE))
        w = int(RNG.integers(*RECT_W_RANGE))
        area = h * w
        if area < MIN_PIECE_AREA * piece_count or area > MAX_RECT_AREA:
            continue
        return h, w


def _build_group(piece_count: int, color: int) -> dict[str, object] | None:
    for _ in range(GROUP_RETRIES):
        rect_h, rect_w = _sample_rect_dims(piece_count)
        pieces = _partition_rectangle(rect_h, rect_w, piece_count)
        if pieces is None:
            continue
        return {
            "color": int(color),
            "count": int(piece_count),
            "h": int(rect_h),
            "w": int(rect_w),
            "pieces": pieces,
        }
    return None


def _group_rectangles(groups: list[dict[str, object]]) -> list[dict[str, int]]:
    return [
        {
            "color": int(group["color"]),
            "count": int(group["count"]),
            "h": int(group["h"]),
            "w": int(group["w"]),
        }
        for group in groups
    ]


def _packed_output_fits(groups: list[dict[str, object]]) -> bool:
    rectangles = _group_rectangles(groups)
    for piece_count in (2, 3, 4):
        widths = [int(rect["w"]) for rect in rectangles if int(rect["count"]) == piece_count]
        if len(widths) != len(set(widths)):
            return False
    out = _pack_rectangles(rectangles)
    return out.size > 0 and max(out.shape) <= ARC_MAX


def _sprite_from_piece(piece: dict[str, object], color: int) -> np.ndarray:
    sprite = np.full((int(piece["h"]), int(piece["w"])), BG, dtype=int)
    sprite[piece["mask"]] = int(color)
    return sprite


def _place_sprites(groups: list[dict[str, object]]) -> np.ndarray | None:
    canvas = np.full((ARC_MAX, ARC_MAX), BG, dtype=int)
    blocked = np.zeros((ARC_MAX, ARC_MAX), dtype=bool)
    sprites = []
    for group in groups:
        color = int(group["color"])
        for piece in group["pieces"]:
            sprite = _sprite_from_piece(piece, color)
            sprites.append(sprite)

    sprites.sort(key=lambda arr: ((arr.shape[0] + 2) * (arr.shape[1] + 2), arr.shape[0] * arr.shape[1]), reverse=True)

    for sprite in sprites:
        sh, sw = map(int, sprite.shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(RNG.integers(0, ARC_MAX - sh + 1))
            left = int(RNG.integers(0, ARC_MAX - sw + 1))
            pad_top = max(0, top - 1)
            pad_left = max(0, left - 1)
            pad_bottom = min(ARC_MAX, top + sh + 1)
            pad_right = min(ARC_MAX, left + sw + 1)
            if blocked[pad_top:pad_bottom, pad_left:pad_right].any():
                continue
            view = canvas[top : top + sh, left : left + sw]
            view[sprite != BG] = sprite[sprite != BG]
            blocked[pad_top:pad_bottom, pad_left:pad_right] = True
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    crop_top = int(RNG.integers(0, int(r0) + 1))
    crop_left = int(RNG.integers(0, int(c0) + 1))
    crop_bottom = int(RNG.integers(int(r1), ARC_MAX))
    crop_right = int(RNG.integers(int(c1), ARC_MAX))
    return canvas[crop_top : crop_bottom + 1, crop_left : crop_right + 1]


def _generate_attempt() -> np.ndarray | None:
    piece_counts = _sample_piece_counts()
    colors = RNG.choice(np.array(PALETTE, dtype=int), size=len(piece_counts), replace=False).tolist()
    groups = []
    for piece_count, color in zip(piece_counts, colors):
        group = _build_group(int(piece_count), int(color))
        if group is None:
            return None
        groups.append(group)

    dims = [(int(group["h"]), int(group["w"])) for group in groups]
    if len(dims) != len(set(dims)):
        return None
    if not _packed_output_fits(groups):
        return None

    grid = _place_sprites(groups)
    if grid is None or max(grid.shape) > ARC_MAX:
        return None

    expected = _pack_rectangles(_group_rectangles(groups))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SAMPLE_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid exploded-fragment sample")

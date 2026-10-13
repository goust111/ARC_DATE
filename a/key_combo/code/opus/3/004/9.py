from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, paste


BG = 0
SEP = 5
PLACEHOLDER = 7
SPRITE_SIZE = 5
PALETTE = [1, 2, 3, 4, 6, 8, 9]
MAX_GEN_ATTEMPTS = 400


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [int(r) for r in range(grid.shape[0]) if bool((grid[r] == SEP).all())]
    assert len(rows) == 1, "expected exactly one full gray separator row"
    return rows[0]


def _crop_component(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_sprites(upper: np.ndarray) -> list[dict]:
    comps = find_components(
        upper,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    sprites = []
    for comp in comps:
        patch = _crop_component(upper, comp["bbox"])
        if patch.shape != (SPRITE_SIZE, SPRITE_SIZE):
            continue
        mask = patch != BG
        colors = sorted({int(v) for v in patch[mask]})
        if 2 <= len(colors) <= 3 and 10 <= int(mask.sum()) <= 18:
            sprites.append(
                {
                    "bbox": tuple(map(int, comp["bbox"])),
                    "patch": patch,
                    "mask": mask,
                }
            )
    sprites.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert 3 <= len(sprites) <= 4, "expected 3 or 4 upper sprites"
    return sprites


def _edge_rows(mask: np.ndarray, side: str) -> set[int]:
    if side == "left":
        return {int(r) for r in np.where(mask[:, 0])[0]}
    if side == "right":
        return {int(r) for r in np.where(mask[:, -1])[0]}
    raise ValueError(f"bad side: {side}")


def _connects(left_sprite: np.ndarray, right_sprite: np.ndarray) -> bool:
    return bool(_edge_rows(left_sprite, "right") & _edge_rows(right_sprite, "left"))


def _find_unique_chain(masks: list[np.ndarray]) -> tuple[int, ...]:
    perms = []
    for perm in itertools.permutations(range(len(masks))):
        if all(_connects(masks[perm[i]], masks[perm[i + 1]]) for i in range(len(perm) - 1)):
            perms.append(tuple(int(v) for v in perm))
    assert len(perms) == 1, "edge matching must define a unique chain"
    return perms[0]


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _placement_bits(mask: np.ndarray, r0: int, c0: int, width: int) -> int:
    bits = 0
    for r, c in _mask_coords(mask):
        idx = (int(r0) + int(r)) * width + int(c0) + int(c)
        bits |= 1 << idx
    return bits


def _orange_bits(mask: np.ndarray) -> int:
    bits = 0
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        bits |= 1 << (int(r) * w + int(c))
    return bits


def _candidate_positions(lower_mask: np.ndarray, sprite_mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = lower_mask.shape
    sh, sw = sprite_mask.shape
    coords = _mask_coords(sprite_mask)
    candidates: list[tuple[int, int, int]] = []
    for r0 in range(h - sh + 1):
        rr = coords[:, 0] + int(r0)
        for c0 in range(w - sw + 1):
            cc = coords[:, 1] + int(c0)
            if bool(lower_mask[rr, cc].all()):
                candidates.append((int(r0), int(c0), _placement_bits(sprite_mask, r0, c0, w)))
    candidates.sort()
    return candidates


def _recover_placeholder_positions(
    lower_mask: np.ndarray,
    sprite_masks: list[np.ndarray],
    *,
    limit: int = 2,
) -> tuple[list[tuple[int, int]], int]:
    orange = _orange_bits(lower_mask)
    candidates = [_candidate_positions(lower_mask, mask) for mask in sprite_masks]
    assert all(candidates), "each placeholder shape must have at least one candidate placement"

    order = sorted(range(len(sprite_masks)), key=lambda idx: (len(candidates[idx]), idx))
    suffix_possible = [0] * (len(order) + 1)
    for pos in range(len(order) - 1, -1, -1):
        bits = 0
        for _r0, _c0, cand_bits in candidates[order[pos]]:
            bits |= cand_bits
        suffix_possible[pos] = suffix_possible[pos + 1] | bits

    placements: list[tuple[int, int] | None] = [None] * len(sprite_masks)
    solutions: list[list[tuple[int, int]]] = []

    def backtrack(pos: int, covered: int) -> None:
        if len(solutions) >= limit:
            return
        if pos == len(order):
            if covered == orange:
                solutions.append([tuple(map(int, pos)) for pos in placements])
            return
        if (covered | suffix_possible[pos]) != orange:
            return

        sprite_idx = order[pos]
        for r0, c0, bits in candidates[sprite_idx]:
            new_covered = covered | bits
            if (new_covered | suffix_possible[pos + 1]) != orange:
                continue
            placements[sprite_idx] = (int(r0), int(c0))
            backtrack(pos + 1, new_covered)
            placements[sprite_idx] = None

    backtrack(0, 0)
    assert solutions, "placeholder recovery failed"
    return solutions[0], len(solutions)


def _stamp_lower(
    lower_shape: tuple[int, int],
    sprites: list[dict],
    placements: list[tuple[int, int]],
    order: tuple[int, ...],
) -> np.ndarray:
    out = np.zeros(lower_shape, dtype=int)
    for idx in order:
        r0, c0 = placements[idx]
        paste(out, sprites[idx]["patch"], r0, c0, mask=sprites[idx]["mask"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp upper sprites onto overlapping lower placeholders using edge-chain order as the occlusion priority.

    Input structure:
    - A black background grid is split into upper and lower regions by one full gray separator row.
    - The upper region contains 3 or 4 multicolor 5x5 sprites, each a single 4-connected foreground object.
    - The lower region contains orange placeholder pixels whose hidden 5x5 masks match the upper sprites, with some placeholders overlapping in the same visible orange union.

    Rule:
    - Recover the upper sprites and order them into the unique left-to-right edge-contact chain by matching right-edge foreground rows to left-edge foreground rows.
    - Recover the hidden lower placeholders by finding one placement of each sprite mask whose union is exactly the visible orange region.
    - Match each lower placeholder to the upper sprite with the same normalized mask.
    - Stamp the matched sprites onto the lower region from chain head to chain tail, so later stamps overwrite earlier ones on overlap cells.
    - Keep the upper region and the gray separator row unchanged.

    Key insight:
    - The edge-contact chain does not rearrange sprites in space; it provides the drawing priority that resolves which sprite stays visible on overlapping placeholder pixels.
    """
    g = np.asarray(grid, dtype=int)
    sep_r = _find_separator_row(g)
    upper = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]

    sprites = _extract_sprites(upper)
    order = _find_unique_chain([sprite["mask"] for sprite in sprites])
    placements, count = _recover_placeholder_positions(lower == PLACEHOLDER, [sprite["mask"] for sprite in sprites], limit=2)
    assert count == 1, "placeholder placement must be unique"

    out = g.copy()
    out[sep_r + 1 :, :] = _stamp_lower(lower.shape, sprites, placements, order)
    return out


def _manhattan_path(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, start)
    gr, gc = map(int, goal)
    path = [(r, c)]
    while (r, c) != (gr, gc):
        moves = []
        if r < gr:
            moves.append((r + 1, c))
        elif r > gr:
            moves.append((r - 1, c))
        if c < gc:
            moves.append((r, c + 1))
        elif c > gc:
            moves.append((r, c - 1))
        r, c = random.choice(moves)
        path.append((int(r), int(c)))
    return path


def _allowed_boundary_cell(r: int, c: int, left_row: int, right_row: int) -> bool:
    if c == 0:
        return r == left_row
    if c == SPRITE_SIZE - 1:
        return r == right_row
    return True


def _grow_sprite_mask(left_row: int, right_row: int) -> np.ndarray | None:
    mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    hub_r = random.randint(1, SPRITE_SIZE - 2)

    for c in range(0, 2):
        mask[int(left_row), int(c)] = True
    for r in range(min(int(left_row), hub_r), max(int(left_row), hub_r) + 1):
        mask[int(r), 1] = True
    for c in range(1, SPRITE_SIZE - 1):
        mask[hub_r, int(c)] = True
    for r in range(min(hub_r, int(right_row)), max(hub_r, int(right_row)) + 1):
        mask[int(r), SPRITE_SIZE - 2] = True
    for c in range(SPRITE_SIZE - 2, SPRITE_SIZE):
        mask[int(right_row), int(c)] = True

    if 0 not in {left_row, right_row}:
        for r in range(0, hub_r + 1):
            mask[int(r), 2] = True
    if SPRITE_SIZE - 1 not in {left_row, right_row}:
        for r in range(hub_r, SPRITE_SIZE):
            mask[int(r), 2] = True

    target = random.randint(max(10, int(mask.sum())), 18)
    while int(mask.sum()) < target:
        frontier = []
        for r, c in zip(*np.where(mask)):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < SPRITE_SIZE and 0 <= nc < SPRITE_SIZE):
                    continue
                if mask[nr, nc]:
                    continue
                if not _allowed_boundary_cell(nr, nc, left_row, right_row):
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True

    if int(mask.sum()) < 10 or int(mask.sum()) > 18:
        return None
    if _edge_rows(mask, "left") != {int(left_row)}:
        return None
    if _edge_rows(mask, "right") != {int(right_row)}:
        return None
    if not mask[0].any() or not mask[-1].any():
        return None
    return mask


def _colorize_mask(mask: np.ndarray, primary: int, secondaries: list[int]) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    for r, c in coords:
        patch[r, c] = int(primary)

    alt_colors = [int(c) for c in secondaries]
    random.shuffle(coords)
    splits = [max(2, len(coords) // 4)]
    if len(alt_colors) == 2:
        splits.append(max(1, len(coords) // 7))
    cursor = 0
    for color, count in zip(alt_colors, splits):
        block = coords[cursor : cursor + count]
        cursor += count
        for r, c in block:
            patch[r, c] = int(color)
    return patch


def _sample_sprite_patches(n: int) -> list[np.ndarray] | None:
    rows = random.sample(range(SPRITE_SIZE), SPRITE_SIZE)
    link_rows = rows[: n - 1]
    head_row = rows[n - 1]
    tail_row = rows[n]
    primaries = random.sample(PALETTE, n)
    remaining = [color for color in PALETTE if color not in primaries]

    patches = []
    seen_masks = set()
    for idx in range(n):
        left_row = head_row if idx == 0 else link_rows[idx - 1]
        right_row = tail_row if idx == n - 1 else link_rows[idx]
        mask = _grow_sprite_mask(left_row, right_row)
        if mask is None:
            return None
        key = mask.tobytes()
        if key in seen_masks:
            return None
        seen_masks.add(key)
        secondary_count = random.choice((1, 2))
        sec_pool = [color for color in remaining if color != primaries[idx]]
        if len(sec_pool) < secondary_count:
            sec_pool = [color for color in PALETTE if color != primaries[idx]]
        secondaries = random.sample(sec_pool, secondary_count)
        patches.append(_colorize_mask(mask, primaries[idx], secondaries))

    try:
        chain = _find_unique_chain([patch != BG for patch in patches])
    except AssertionError:
        return None
    if chain != tuple(range(n)):
        return None
    return patches


def _rectangles_overlap(a: tuple[int, int], b: tuple[int, int], *, size: int = SPRITE_SIZE, gap: int = 1) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + size + gap <= br
        or br + size + gap <= ar
        or ac + size + gap <= bc
        or bc + size + gap <= ac
    )


def _sample_upper_positions(n: int, upper_h: int, width: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    for _ in range(200):
        positions.clear()
        ok = True
        for _idx in range(n):
            placed = False
            for _ in range(200):
                r0 = random.randint(0, upper_h - SPRITE_SIZE)
                c0 = random.randint(0, width - SPRITE_SIZE)
                if any(_rectangles_overlap((r0, c0), prev) for prev in positions):
                    continue
                positions.append((int(r0), int(c0)))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return positions.copy()
    return None


def _col_major_order(positions: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted(range(len(positions)), key=lambda idx: (positions[idx][1], positions[idx][0])))


def _row_major_order(positions: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted(range(len(positions)), key=lambda idx: (positions[idx][0], positions[idx][1])))


def _mask_overlap_count(
    mask_a: np.ndarray,
    pos_a: tuple[int, int],
    mask_b: np.ndarray,
    pos_b: tuple[int, int],
) -> int:
    coords_a = {(int(pos_a[0] + r), int(pos_a[1] + c)) for r, c in np.argwhere(mask_a)}
    coords_b = {(int(pos_b[0] + r), int(pos_b[1] + c)) for r, c in np.argwhere(mask_b)}
    return len(coords_a & coords_b)


def _possible_overlap_offsets(mask_a: np.ndarray, mask_b: np.ndarray) -> list[tuple[int, int]]:
    offsets = []
    for dr in range(-(SPRITE_SIZE - 1), SPRITE_SIZE):
        for dc in range(-(SPRITE_SIZE - 1), SPRITE_SIZE):
            if dr == 0 and dc == 0:
                continue
            count = _mask_overlap_count(mask_a, (0, 0), mask_b, (dr, dc))
            if 1 <= count <= 4:
                offsets.append((int(dr), int(dc)))
    random.shuffle(offsets)
    return offsets


def _sample_placeholder_positions(masks: list[np.ndarray], lower_h: int, width: int) -> list[tuple[int, int]] | None:
    n = len(masks)
    pair = random.sample(range(n), 2)
    a, b = pair
    offsets = _possible_overlap_offsets(masks[a], masks[b])
    if not offsets:
        return None

    for dr, dc in offsets:
        r_lo = max(0, -dr)
        r_hi = min(lower_h - SPRITE_SIZE, lower_h - SPRITE_SIZE - dr)
        c_lo = max(0, -dc)
        c_hi = min(width - SPRITE_SIZE, width - SPRITE_SIZE - dc)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        for _ in range(120):
            positions: list[tuple[int, int] | None] = [None] * n
            ra = random.randint(r_lo, r_hi)
            ca = random.randint(c_lo, c_hi)
            positions[a] = (int(ra), int(ca))
            positions[b] = (int(ra + dr), int(ca + dc))

            ok = True
            for idx in range(n):
                if positions[idx] is not None:
                    continue
                placed = False
                for _ in range(160):
                    r0 = random.randint(0, lower_h - SPRITE_SIZE)
                    c0 = random.randint(0, width - SPRITE_SIZE)
                    here = (int(r0), int(c0))
                    overlaps = [
                        _mask_overlap_count(masks[idx], here, masks[j], positions[j])
                        for j in range(n)
                        if positions[j] is not None
                    ]
                    if any(v > 0 for v in overlaps):
                        continue
                    positions[idx] = here
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if not ok:
                continue
            realized = [(int(r), int(c)) for r, c in positions if r is not None]
            if len(realized) != n:
                continue
            return realized
    return None


def _render_union_mask(
    shape: tuple[int, int],
    masks: list[np.ndarray],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for mask, (r0, c0) in zip(masks, positions):
        rr, cc = np.where(mask)
        union[rr + int(r0), cc + int(c0)] = True
    return union


def _exclusive_counts(masks: list[np.ndarray], positions: list[tuple[int, int]], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    cover = np.zeros((h, w), dtype=int)
    placed_cells = []
    for mask, (r0, c0) in zip(masks, positions):
        cell_mask = np.zeros((h, w), dtype=bool)
        rr, cc = np.where(mask)
        cell_mask[rr + int(r0), cc + int(c0)] = True
        placed_cells.append(cell_mask)
        cover[cell_mask] += 1
    return [int((cell_mask & (cover == 1)).sum()) for cell_mask in placed_cells]


def _render_input_grid(
    patches: list[np.ndarray],
    upper_positions: list[tuple[int, int]],
    positions: list[tuple[int, int]],
    *,
    upper_h: int,
    lower_h: int,
    width: int,
) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((upper_h + 1 + lower_h, width), dtype=int)
    grid[upper_h, :] = SEP

    sprites = []
    for patch, (r0, c0) in zip(patches, upper_positions):
        paste(grid[:upper_h, :], patch, r0, c0, transparent=BG)
        sprites.append({"patch": patch, "mask": patch != BG, "upper_pos": (int(r0), int(c0))})

    row_sorted = sorted(range(len(sprites)), key=lambda idx: sprites[idx]["upper_pos"])
    sprites_sorted = [sprites[idx] for idx in row_sorted]
    sorted_positions = [positions[idx] for idx in row_sorted]

    lower_union = _render_union_mask((lower_h, width), [sprite["mask"] for sprite in sprites_sorted], sorted_positions)
    grid[upper_h + 1 :, :][lower_union] = PLACEHOLDER

    for new_idx, sprite in enumerate(sprites_sorted):
        sprite["logical_idx"] = int(new_idx)
    return grid, sprites_sorted


def _semantic_checks(
    grid: np.ndarray,
    sprites: list[dict],
    placements: list[tuple[int, int]],
) -> bool:
    sep_r = _find_separator_row(grid)
    lower_mask = grid[sep_r + 1 :, :] == PLACEHOLDER
    masks = [sprite["mask"] for sprite in sprites]

    try:
        order = _find_unique_chain(masks)
    except AssertionError:
        return False

    try:
        recovered, count = _recover_placeholder_positions(lower_mask, masks, limit=2)
    except AssertionError:
        return False

    if count != 1:
        return False
    if recovered != placements:
        return False

    overlaps = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            count_ij = _mask_overlap_count(masks[i], placements[i], masks[j], placements[j])
            if count_ij > 0:
                overlaps.append((i, j, count_ij))
    if not overlaps or any(count_ij > 4 for _i, _j, count_ij in overlaps):
        return False

    if min(_exclusive_counts(masks, placements, lower_mask.shape)) < 3:
        return False

    actual = _stamp_lower(lower_mask.shape, sprites, placements, order)
    reverse = _stamp_lower(lower_mask.shape, sprites, placements, tuple(reversed(order)))
    if np.array_equal(actual, reverse):
        return False

    upper_positions = [sprite["upper_pos"] for sprite in sprites]
    row_order = _row_major_order(upper_positions)
    col_order = _col_major_order(upper_positions)
    placeholder_order = _row_major_order(placements)

    if order in (row_order, col_order, placeholder_order):
        return False
    if np.array_equal(actual, _stamp_lower(lower_mask.shape, sprites, placements, row_order)):
        return False
    if np.array_equal(actual, _stamp_lower(lower_mask.shape, sprites, placements, placeholder_order)):
        return False

    if np.any(actual == PLACEHOLDER):
        return False

    return bool(np.array_equal(main(grid), np.vstack([grid[: sep_r + 1, :], actual])))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        n = random.randint(3, 4)
        width = random.randint(max(18, n * 5 + 2), 26)
        upper_h = random.randint(10, 13)
        lower_h = random.randint(9, 12)
        if upper_h + 1 + lower_h > 30:
            continue

        patches = _sample_sprite_patches(n)
        if patches is None:
            continue

        upper_positions = _sample_upper_positions(n, upper_h, width)
        if upper_positions is None:
            continue

        positions = _sample_placeholder_positions([patch != BG for patch in patches], lower_h, width)
        if positions is None:
            continue

        grid, sprites = _render_input_grid(
            patches,
            upper_positions,
            positions,
            upper_h=upper_h,
            lower_h=lower_h,
            width=width,
        )
        sorted_positions = [positions[idx] for idx in sorted(range(n), key=lambda idx: upper_positions[idx])]

        if not _semantic_checks(grid, sprites, sorted_positions):
            continue
        return grid

    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")

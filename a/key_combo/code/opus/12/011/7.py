from __future__ import annotations

from collections import Counter
import random

import numpy as np


GEN_BG = 0
GEN_CONTROL = 1
GEN_SPRITE = 2
GEN_OCCLUDER = 7

GEN_K_CHOICES = (3, 4)
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _infer_k(size: int) -> int:
    for k in range(2, 6):
        if k * (k + 1) - 1 == int(size):
            return int(k)
    raise AssertionError(f"unsupported lattice size: {size}")


def _slot_starts(k: int) -> list[int]:
    return [i * (k + 1) for i in range(k)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corners(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    return {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }


def _distance_sum(shape: tuple[int, int], corner: str, r: int, c: int) -> int:
    h, w = shape
    if corner == "TL":
        return int(r + c)
    if corner == "TR":
        return int(r + (w - 1 - c))
    if corner == "BL":
        return int((h - 1 - r) + c)
    assert corner == "BR"
    return int((h - 1 - r) + (w - 1 - c))


def _paint_triangle(grid: np.ndarray, corner: str, side: int, color: int) -> None:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if _distance_sum((h, w), corner, r, c) < int(side):
                grid[r, c] = int(color)


def _has_same_color_neighbor(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color):
                return True
    return False


def _render_lattice(k: int, sprite_mask: np.ndarray, bg: int, sprite_color: int) -> np.ndarray:
    size = k * (k + 1) - 1
    out = np.full((size, size), int(bg), dtype=int)
    starts = _slot_starts(k)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            if not bool(sprite_mask[br, bc]):
                continue
            block = out[rs : rs + k, cs : cs + k]
            block[sprite_mask] = int(sprite_color)
    return out


def _infer_sprite_mask(
    grid: np.ndarray,
    starts: list[int],
    k: int,
    sprite_color: int,
    occluder_color: int,
) -> np.ndarray | None:
    visible_masks = []
    empty_mask = np.zeros((k, k), dtype=bool)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            block = grid[rs : rs + k, cs : cs + k]
            if np.any(block == int(occluder_color)):
                continue
            color_mask = block == int(sprite_color)
            if np.any(color_mask):
                visible_masks.append(color_mask)
    if not visible_masks:
        return None

    sprite_mask = visible_masks[0]
    if not np.any(sprite_mask):
        return None
    for mask in visible_masks[1:]:
        if not np.array_equal(mask, sprite_mask):
            return None

    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            block = grid[rs : rs + k, cs : cs + k]
            if np.any(block == int(occluder_color)):
                continue
            expected = sprite_mask if bool(sprite_mask[br, bc]) else empty_mask
            if not np.array_equal(block == int(sprite_color), expected):
                return None
    return sprite_mask.copy()


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "grid must be square"
    k = _infer_k(h)
    bg = _get_bg(g)
    starts = _slot_starts(k)

    non_bg_colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    solid_rects = []
    for color in non_bg_colors:
        mask = g == int(color)
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        spans_rows = r0 == 0 and r1 == h - 1
        spans_cols = c0 == 0 and c1 == w - 1
        if int(mask.sum()) == area and spans_rows ^ spans_cols:
            solid_rects.append((int(color), box))
    assert len(solid_rects) == 1, "expected one full-row or full-column occluder"
    occluder_color, (r0, c0, r1, c1) = solid_rects[0]

    hides_slot_row = c0 == 0 and c1 == w - 1
    stripe_size = (r1 - r0 + 1) if hides_slot_row else (c1 - c0 + 1)
    assert int(stripe_size) == int(k), "occluder stripe width must equal slot size"
    hidden_start = int(r0 if hides_slot_row else c0)
    assert hidden_start in starts, "occluder must align to slot starts"
    hidden_idx = starts.index(hidden_start)

    remaining_colors = [int(color) for color in non_bg_colors if int(color) != int(occluder_color)]
    assert len(remaining_colors) == 2, "expected sprite color and control color"

    candidates = []
    for sprite_color in remaining_colors:
        sprite_mask = _infer_sprite_mask(g, starts, k, sprite_color, occluder_color)
        if sprite_mask is None:
            continue
        control_color = [c for c in remaining_colors if int(c) != int(sprite_color)][0]
        candidates.append((int(sprite_color), int(control_color), sprite_mask))
    assert len(candidates) == 1, "expected a unique self-similar sprite color"
    sprite_color, control_color, sprite_mask = candidates[0]

    marked_corners = [
        (name, rc)
        for name, rc in _corners(g.shape).items()
        if int(g[rc]) == int(control_color)
    ]
    assert len(marked_corners) == 1, "expected exactly one control-colored corner marker"
    corner, corner_rc = marked_corners[0]

    control_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == int(control_color))]
    assert corner_rc in control_coords
    assert 2 <= len(control_coords) <= k, "control count must determine triangle side"
    assert all(
        not _has_same_color_neighbor(g, int(r), int(c), int(control_color))
        for r, c in control_coords
    ), "control pixels must stay isolated"

    return {
        "bg": int(bg),
        "k": int(k),
        "starts": starts,
        "corner": corner,
        "corner_rc": tuple(map(int, corner_rc)),
        "control_color": int(control_color),
        "control_count": int(len(control_coords)),
        "control_coords": control_coords,
        "sprite_color": int(sprite_color),
        "sprite_mask": sprite_mask.copy(),
        "occluder_color": int(occluder_color),
        "hides_slot_row": bool(hides_slot_row),
        "hidden_idx": int(hidden_idx),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the self-similar sprite lattice, erase the control dots, and replace them with a corner triangle whose side is the dot count.

    Input structure:
    - The grid is square and partitioned into a k by k lattice of k by k slots separated by 1-cell background lines.
    - One foreground color is a repeated sprite: a slot contains that k by k sprite exactly when the corresponding cell of the sprite itself is filled.
    - A second foreground color is a solid occluder strip covering one entire slot row or one entire slot column.
    - A third foreground color forms isolated control pixels on background cells, with exactly one control pixel placed in a grid corner.

    Rule:
    - Count all control pixels first; that count N is the side length of the final triangle, and the marked corner gives the triangle orientation.
    - Remove the occluder and reconstruct the full lattice by using the sprite's own foreground mask as the coarse slot-occupancy mask.
    - Erase every control pixel, then paint a solid right triangle of the control color in the marked corner, filling all cells whose two corner-edge distances sum to less than N.
    - The triangle is drawn last and overwrites both restored sprite cells and background if they overlap.

    Key insight:
    - The visible sprite is simultaneously the repeated local motif and the global k by k occupancy map, while the scattered corner-marked dots encode independent geometric information that must be extracted before those dots are removed.
    """
    info = _extract_structure(grid)
    out = _render_lattice(
        info["k"],
        info["sprite_mask"],
        info["bg"],
        info["sprite_color"],
    )
    _paint_triangle(out, info["corner"], info["control_count"], info["control_color"])
    return out


def _valid_configs(mask: np.ndarray) -> list[tuple[str, str, int]]:
    k = mask.shape[0]
    configs: list[tuple[str, str, int]] = []
    corner_locs = {
        "TL": (0, 0),
        "TR": (0, k - 1),
        "BL": (k - 1, 0),
        "BR": (k - 1, k - 1),
    }
    for corner, (cr, cc) in corner_locs.items():
        if bool(mask[cr, cc]):
            continue
        forbidden_rows = {0} if corner.startswith("T") else {k - 1}
        forbidden_cols = {0} if corner.endswith("L") else {k - 1}
        for idx in range(k):
            if idx not in forbidden_rows and bool(mask[idx, :].any()) and bool(np.delete(mask, idx, axis=0).any()):
                configs.append((corner, "row", int(idx)))
            if idx not in forbidden_cols and bool(mask[:, idx].any()) and bool(np.delete(mask, idx, axis=1).any()):
                configs.append((corner, "col", int(idx)))
    return configs


def _sample_sprite_mask(k: int) -> tuple[np.ndarray, tuple[str, str, int]] | None:
    total = k * k
    min_count = max(3, k)
    max_count = total - 2
    for _ in range(200):
        count = random.randint(min_count, max_count)
        flat = np.zeros(total, dtype=bool)
        flat[random.sample(range(total), count)] = True
        mask = flat.reshape(k, k)
        if int(mask.sum()) >= total:
            continue
        if int(mask.sum()) < 2:
            continue
        if not bool((~mask).sum()):
            continue
        configs = _valid_configs(mask)
        if not configs:
            continue
        return mask, random.choice(configs)
    return None


def _apply_occluder(grid: np.ndarray, k: int, axis: str, idx: int, color: int) -> None:
    start = _slot_starts(k)[idx]
    if axis == "row":
        grid[start : start + k, :] = int(color)
    else:
        grid[:, start : start + k] = int(color)


def _neighbor_coords(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((int(rr), int(cc)))
    return out


def _blue_pool(grid: np.ndarray, chosen: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    sep_rows = {r for r in range(h) if r % (_infer_k(h) + 1) == _infer_k(h)}
    sep_cols = {c for c in range(w) if c % (_infer_k(w) + 1) == _infer_k(w)}
    slot_cells = []
    separator_cells = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GEN_BG:
                continue
            if any((nr, nc) in chosen for nr, nc in _neighbor_coords(r, c, grid.shape)):
                continue
            target = separator_cells if r in sep_rows or c in sep_cols else slot_cells
            target.append((int(r), int(c)))
    random.shuffle(slot_cells)
    random.shuffle(separator_cells)
    return slot_cells + separator_cells


def _place_controls(grid: np.ndarray, corner_rc: tuple[int, int], count: int) -> bool:
    chosen = {tuple(map(int, corner_rc))}
    grid[corner_rc] = GEN_CONTROL
    for _ in range(count - 1):
        pool = _blue_pool(grid, chosen)
        if not pool:
            return False
        r, c = pool[0]
        grid[r, c] = GEN_CONTROL
        chosen.add((int(r), int(c)))
    return True


def _corner_rc(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    return _corners(shape)[corner]


def _expected_output(mask: np.ndarray, corner: str, side: int) -> np.ndarray:
    out = _render_lattice(mask.shape[0], mask, GEN_BG, GEN_SPRITE)
    _paint_triangle(out, corner, side, GEN_CONTROL)
    return out


def _validate_generated(inp: np.ndarray, expected: np.ndarray, corner: str, axis: str, idx: int) -> bool:
    try:
        info = _extract_structure(inp)
    except Exception:
        return False
    if info["corner"] != corner or info["hidden_idx"] != idx or info["hides_slot_row"] != (axis == "row"):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    restored = _render_lattice(info["k"], info["sprite_mask"], info["bg"], info["sprite_color"])
    if np.array_equal(restored, inp):
        return False
    if np.array_equal(restored, expected):
        return False
    hidden = inp == GEN_OCCLUDER
    if not np.any(hidden & (restored == GEN_SPRITE)):
        return False
    return True


def _generate_one() -> np.ndarray | None:
    k = random.choice(GEN_K_CHOICES)
    sampled = _sample_sprite_mask(k)
    if sampled is None:
        return None
    sprite_mask, (corner, axis, idx) = sampled
    side = random.randint(2, k)

    base = _render_lattice(k, sprite_mask, GEN_BG, GEN_SPRITE)
    expected = _expected_output(sprite_mask, corner, side)
    inp = base.copy()
    _apply_occluder(inp, k, axis, idx, GEN_OCCLUDER)

    if not _place_controls(inp, _corner_rc(inp.shape, corner), side):
        return None

    # Keep the input semantically clean and branch-relevant.
    if np.count_nonzero(inp == GEN_CONTROL) != side:
        return None
    if int(inp[_corner_rc(inp.shape, corner)]) != GEN_CONTROL:
        return None
    if not _validate_generated(inp, expected, corner, axis, idx):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

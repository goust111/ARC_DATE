from __future__ import annotations

import random
from typing import Iterable

import numpy as np


PALETTE = tuple(range(1, 10))
LAYOUTS = ((2, 2), (2, 3), (3, 2))
PANEL_H_RANGE = (7, 8)
PANEL_W_RANGE = (7, 8)
SEP_RANGE = (1, 2)
GENERATION_RETRIES = 500


def _bands(indices: np.ndarray) -> list[tuple[int, int]]:
    values = [int(v) for v in indices]
    if not values:
        return []
    out: list[tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1, "expected one separator color shared by full rows and full columns"
    return next(iter(candidates))


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "shape mask must be non-empty"
    origin = coords.min(axis=0)
    shifted = coords - origin
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _mask_from_signature(signature: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in signature]
    cols = [c for _, c in signature]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in signature:
        mask[int(r), int(c)] = True
    return mask


def _coords_set(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _masks_too_close(a: np.ndarray, b: np.ndarray) -> bool:
    b_coords = _coords_set(b)
    for r, c in np.argwhere(a):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in b_coords:
                    return True
    return False


def _place_shape(panel_h: int, panel_w: int, shape: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.zeros((panel_h, panel_w), dtype=bool)
    h, w = shape.shape
    out[top : top + h, left : left + w] = shape
    return out


def _all_placements(panel_h: int, panel_w: int, shape: np.ndarray) -> list[np.ndarray]:
    h, w = shape.shape
    placements = []
    for top in range(1, panel_h - h):
        for left in range(1, panel_w - w):
            placements.append(_place_shape(panel_h, panel_w, shape, top, left))
    return placements


def _partition(total: int, parts: int) -> list[int]:
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _grow_component(mask: np.ndarray, target_size: int, seed: tuple[int, int]) -> bool:
    h, w = mask.shape
    cells = {seed}
    mask[seed] = True
    while len(cells) < target_size:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.add((rr, cc))
        if not frontier:
            return False
        nxt = random.choice(tuple(frontier))
        mask[nxt] = True
        cells.add(nxt)
    return True


def _sample_shape(max_h: int, max_w: int) -> np.ndarray | None:
    box_h = random.randint(2, min(4, max_h))
    box_w = random.randint(2, min(4, max_w))
    max_area = min(box_h * box_w - 1, 7)
    if max_area < 3:
        return None
    area = random.randint(3, max_area)
    n_comp = 1 if area < 5 or random.random() < 0.7 else 2
    mask = np.zeros((box_h, box_w), dtype=bool)

    cells = [(r, c) for r in range(box_h) for c in range(box_w)]
    random.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for cell in cells:
        if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in seeds):
            seeds.append(cell)
            if len(seeds) == n_comp:
                break
    if len(seeds) != n_comp:
        return None

    for size, seed in zip(_partition(area, n_comp), seeds):
        if not _grow_component(mask, size, seed):
            return None

    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    if mask.all():
        return None
    return mask


def _sample_unique_shapes(count: int, panel_h: int, panel_w: int) -> list[tuple[tuple[int, int], ...]] | None:
    signatures: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    attempts = 0
    while len(signatures) < count and attempts < 400:
        attempts += 1
        shape = _sample_shape(panel_h - 2, panel_w - 2)
        if shape is None:
            continue
        signature = _normalize_mask(shape)
        if signature in seen:
            continue
        seen.add(signature)
        signatures.append(signature)
    return signatures if len(signatures) == count else None


def _panel_positions(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _is_nonlocal(src: int, dst: int, positions: list[tuple[int, int]]) -> bool:
    sr, sc = positions[src]
    tr, tc = positions[dst]
    return sr != tr and sc != tc


def _sample_derangement(count: int, positions: list[tuple[int, int]]) -> list[int] | None:
    values = list(range(count))
    for _ in range(300):
        perm = values[:]
        random.shuffle(perm)
        if any(i == perm[i] for i in values):
            continue
        if not any(_is_nonlocal(i, perm[i], positions) for i in values):
            continue
        return perm
    return None


def _paint_key(panel: np.ndarray, key_mask: np.ndarray, colors: list[int]) -> None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(key_mask)]
    offset = random.randrange(len(colors))
    for idx, (r, c) in enumerate(coords):
        panel[r, c] = int(colors[(idx + offset + r + 2 * c) % len(colors)])


def _extract_panels(grid: np.ndarray) -> tuple[int, list[dict]]:
    sep = _separator_color(grid)
    row_bands = _bands(np.where(~np.all(grid == sep, axis=1))[0])
    col_bands = _bands(np.where(~np.all(grid == sep, axis=0))[0])
    assert row_bands and col_bands, "expected separated panel bands"

    panel_h = row_bands[0][1] - row_bands[0][0] + 1
    panel_w = col_bands[0][1] - col_bands[0][0] + 1
    panels = []
    index = 0
    for pr, (r0, r1) in enumerate(row_bands):
        assert r1 - r0 + 1 == panel_h
        for pc, (c0, c1) in enumerate(col_bands):
            assert c1 - c0 + 1 == panel_w
            panel = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(panel == sep), "separator color cannot appear inside panels"
            base = int(panel[0, 0])
            assert base != 0, "panel base color must be non-black"
            assert np.all(panel[0, :] == base)
            assert np.all(panel[-1, :] == base)
            assert np.all(panel[:, 0] == base)
            assert np.all(panel[:, -1] == base)
            hole_mask = panel == 0
            key_mask = (panel != base) & (panel != 0)
            assert hole_mask.any() and key_mask.any(), "each panel needs one hole mask and one key mask"
            panels.append(
                {
                    "index": index,
                    "panel_rc": (pr, pc),
                    "bbox": (r0, c0, r1, c1),
                    "base": base,
                    "hole_mask": hole_mask,
                    "key_mask": key_mask,
                    "hole_sig": _normalize_mask(hole_mask),
                    "key_sig": _normalize_mask(key_mask),
                }
            )
            index += 1
    return sep, panels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each key motif into the other panel whose black hole has the same shape, then repaint it with the key panel's base color.

    Input structure:
    - Full separator-color rows and columns split the canvas into a rectangular array of same-sized panels.
    - Every panel has a border-colored base, one black hole shape, and one non-black key motif made of distractor colors.
    - A key's shape matches exactly one other panel's hole under translation only; no key matches its own panel's hole.

    Rule:
    - Identify each panel's base color from its border, its hole from the black cells, and its key from all remaining non-base cells.
    - For every key, find the different panel whose hole shape is translation-congruent to that key shape.
    - Clear every panel back to its own base color, then fill each target hole at its original location using the source key panel's base color.

    Key insight:
    - Key colors and panel adjacency are both distractions: only the key mask chooses the destination, and only the source panel's base color chooses the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    sep, panels = _extract_panels(g)
    hole_to_panel: dict[tuple[tuple[int, int], ...], int] = {}
    for panel in panels:
        assert panel["hole_sig"] not in hole_to_panel, "hole shapes must be unique"
        hole_to_panel[panel["hole_sig"]] = int(panel["index"])

    targets: dict[int, int] = {}
    used_targets: set[int] = set()
    for panel in panels:
        key_sig = panel["key_sig"]
        assert key_sig in hole_to_panel, "every key must match some hole"
        target = hole_to_panel[key_sig]
        assert target != panel["index"], "a key cannot target its own panel"
        assert target not in used_targets, "hole targets must be one-to-one"
        used_targets.add(target)
        targets[int(panel["index"])] = int(target)
    assert len(targets) == len(panels)

    out = np.full_like(g, sep)
    for panel in panels:
        r0, c0, r1, c1 = panel["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(panel["base"])

    for panel in panels:
        target = panels[targets[int(panel["index"])]]
        r0, c0, r1, c1 = target["bbox"]
        target_view = out[r0 : r1 + 1, c0 : c1 + 1]
        target_view[target["hole_mask"]] = int(panel["base"])
    return out


def _build_expected_output(
    shape_signatures: list[tuple[tuple[int, int], ...]],
    placements: list[np.ndarray],
    bases: list[int],
    perm: list[int],
    n_rows: int,
    n_cols: int,
    panel_h: int,
    panel_w: int,
    sep_color: int,
    sep_h: int,
    sep_w: int,
) -> np.ndarray:
    height = n_rows * panel_h + (n_rows - 1) * sep_h
    width = n_cols * panel_w + (n_cols - 1) * sep_w
    out = np.full((height, width), sep_color, dtype=int)
    incoming = {perm[src]: src for src in range(len(perm))}
    for idx in range(len(shape_signatures)):
        pr, pc = divmod(idx, n_cols)
        r0 = pr * (panel_h + sep_h)
        c0 = pc * (panel_w + sep_w)
        panel = np.full((panel_h, panel_w), bases[idx], dtype=int)
        panel[placements[idx]] = bases[incoming[idx]]
        out[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return out


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    perm: list[int],
    positions: list[tuple[int, int]],
) -> bool:
    if np.array_equal(inp, out):
        return False
    if not any(_is_nonlocal(i, perm[i], positions) for i in range(len(perm))):
        return False
    sep, panels = _extract_panels(inp)
    if len(panels) != len(perm):
        return False
    if any(panel["hole_sig"] == panel["key_sig"] for panel in panels):
        return False
    if len({panel["hole_sig"] for panel in panels}) != len(panels):
        return False
    if len({panel["base"] for panel in panels}) != len(panels):
        return False
    solved = main(inp)
    if not np.array_equal(solved, out):
        return False

    solved_sep = _separator_color(solved)
    if sep != solved_sep:
        return False
    row_bands = _bands(np.where(~np.all(solved == solved_sep, axis=1))[0])
    col_bands = _bands(np.where(~np.all(solved == solved_sep, axis=0))[0])
    if len(row_bands) * len(col_bands) != len(panels):
        return False

    incoming = {perm[src]: src for src in range(len(perm))}
    output_sigs = set()
    for panel in panels:
        idx = int(panel["index"])
        r0, c0, r1, c1 = panel["bbox"]
        solved_panel = solved[r0 : r1 + 1, c0 : c1 + 1]
        base = int(panel["base"])
        if np.any(solved_panel == 0):
            return False
        if not (
            np.all(solved_panel[0, :] == base)
            and np.all(solved_panel[-1, :] == base)
            and np.all(solved_panel[:, 0] == base)
            and np.all(solved_panel[:, -1] == base)
        ):
            return False
        foreign_mask = solved_panel != base
        if not foreign_mask.any():
            return False
        sig = _normalize_mask(foreign_mask)
        if sig != panel["hole_sig"]:
            return False
        foreign_colors = set(map(int, solved_panel[foreign_mask]))
        if foreign_colors != {int(panels[incoming[idx]]["base"])}:
            return False
        output_sigs.add((sig, next(iter(foreign_colors))))
    if len(output_sigs) != len(panels):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUTS)
    count = n_rows * n_cols
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    sep_h = random.randint(*SEP_RANGE)
    sep_w = random.randint(*SEP_RANGE)
    height = n_rows * panel_h + (n_rows - 1) * sep_h
    width = n_cols * panel_w + (n_cols - 1) * sep_w
    if max(height, width) > 30:
        return None

    sep_color = random.choice(PALETTE)
    base_pool = [color for color in PALETTE if color != sep_color]
    if len(base_pool) < count:
        return None
    bases = random.sample(base_pool, k=count)
    key_palette = [color for color in base_pool if color not in bases]
    if not key_palette:
        return None

    positions = _panel_positions(n_rows, n_cols)
    perm = _sample_derangement(count, positions)
    if perm is None:
        return None

    shape_signatures = _sample_unique_shapes(count, panel_h, panel_w)
    if shape_signatures is None:
        return None

    panels: list[np.ndarray] = []
    hole_placements: list[np.ndarray] = []
    for idx in range(count):
        hole_shape = _mask_from_signature(shape_signatures[idx])
        key_shape = _mask_from_signature(shape_signatures[perm[idx]])
        hole_options = _all_placements(panel_h, panel_w, hole_shape)
        key_options = _all_placements(panel_h, panel_w, key_shape)
        random.shuffle(hole_options)
        random.shuffle(key_options)

        placed_hole = placed_key = None
        for hole_mask in hole_options:
            for key_mask in key_options:
                if np.any(hole_mask & key_mask):
                    continue
                if _masks_too_close(hole_mask, key_mask):
                    continue
                placed_hole, placed_key = hole_mask, key_mask
                break
            if placed_hole is not None:
                break
        if placed_hole is None or placed_key is None:
            return None

        panel = np.full((panel_h, panel_w), bases[idx], dtype=int)
        panel[placed_hole] = 0
        key_color_count = random.randint(1, min(3, len(key_palette)))
        key_colors = random.sample(key_palette, k=key_color_count)
        _paint_key(panel, placed_key, key_colors)
        panels.append(panel)
        hole_placements.append(placed_hole)

    inp = np.full((height, width), sep_color, dtype=int)
    for idx, panel in enumerate(panels):
        pr, pc = divmod(idx, n_cols)
        r0 = pr * (panel_h + sep_h)
        c0 = pc * (panel_w + sep_w)
        inp[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    out = _build_expected_output(
        shape_signatures=shape_signatures,
        placements=hole_placements,
        bases=bases,
        perm=perm,
        n_rows=n_rows,
        n_cols=n_cols,
        panel_h=panel_h,
        panel_w=panel_w,
        sep_color=sep_color,
        sep_h=sep_h,
        sep_w=sep_w,
    )
    if not _validate_semantics(inp, out, perm, positions):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-key input")

from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BRACKET = 5
REFERENCE_COLORS = (1, 2, 3)
MOVER_COLORS = (4, 6, 7)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_bool(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _tight_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(_mask_key(rotated))
        variants.append(_mask_key(np.fliplr(rotated)))
    return min(variants)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


def _components(grid: np.ndarray, bg: int, *, color: int | None = None) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            if int(grid[r, c]) == int(bg):
                continue
            if color is not None and int(grid[r, c]) != int(color):
                continue
            target = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == int(bg):
                        continue
                    if color is not None:
                        if int(grid[nr, nc]) != int(color):
                            continue
                    elif int(grid[nr, nc]) != target:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(target),
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _can_place_opaque(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    H, W = grid.shape
    for r, c in np.argwhere(patch != int(bg)):
        rr = int(r0) + int(r)
        cc = int(c0) + int(c)
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if int(grid[rr, cc]) != int(bg):
            return False
    return True


def _is_top_bracket(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if int(comp["color"]) != BRACKET or r0 != 0 or r1 - r0 + 1 != 2 or c1 - c0 + 1 < 5:
        return False
    cells = {tuple(map(int, xy)) for xy in comp["coords"]}
    expected = {(0, c) for c in range(c0, c1 + 1)} | {(1, c0), (1, c1)}
    return cells == expected


def _extract_structure(grid: np.ndarray) -> tuple[int, list[dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    brackets = sorted(
        [comp for comp in _components(g, bg, color=BRACKET) if _is_top_bracket(comp)],
        key=lambda comp: int(comp["bbox"][1]),
    )
    assert len(brackets) == 3, "expected three top-edge grey brackets"

    colored = [
        comp
        for comp in _components(g, bg)
        if int(comp["color"]) != BRACKET
    ]

    references: list[dict] = []
    ref_ids: set[int] = set()
    for bracket in brackets:
        _r0, c0, _r1, c1 = bracket["bbox"]
        inner_left, inner_right = c0 + 1, c1 - 1
        matches = []
        for idx, comp in enumerate(colored):
            rr0, cc0, _rr1, cc1 = comp["bbox"]
            if rr0 < 1:
                continue
            if not any(int(r) == 1 for r, _c in comp["coords"]):
                continue
            if cc0 < inner_left or cc1 > inner_right:
                continue
            matches.append((idx, comp))
        assert len(matches) == 1, "each bracket must contain one attached reference shape"
        idx, ref = matches[0]
        references.append(ref)
        ref_ids.add(idx)

    movers = [comp for idx, comp in enumerate(colored) if idx not in ref_ids]
    assert len(references) == 3 and len(movers) == 3, "expected three references and three movers"
    return bg, brackets, references, movers


def _settled_output(grid: np.ndarray, placements: list[tuple[dict, dict]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for _bracket, ref in placements:
        out[ref["coords"][:, 0], ref["coords"][:, 1]] = int(bg)
    for _bracket, mover in placements:
        out[mover["coords"][:, 0], mover["coords"][:, 1]] = int(bg)

    current_top = g.shape[0]
    for bracket, mover in placements:
        _r0, c0, _r1, c1 = bracket["bbox"]
        patch = _tight_patch(g, mover["coords"], bg)
        inner_width = c1 - c0 - 1
        mover_width = patch.shape[1]
        assert (inner_width - mover_width) % 2 == 0, "horizontal centering must be unique"
        dest_c = c0 + 1 + (inner_width - mover_width) // 2
        dest_r = current_top - patch.shape[0]
        assert dest_r >= 0, "stack must fit inside the grid"
        _paste(out, patch, dest_r, dest_c, bg)
        current_top = dest_r

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Route each movable object to its shape-equivalent bracket lane, then stack them at the bottom.

    Input structure:
    - Black background surrounds exactly three grey top-edge brackets that open downward.
    - Each bracket contains one small colored reference shape attached to the inside of its top bar.
    - Three separate movable objects appear lower in the grid; their colors differ from the reference colors.

    Rule:
    - Crop every reference and movable object to its tight boolean mask and compare shapes up to rotation/reflection.
    - Match each movable object to the unique bracket whose reference is dihedrally equivalent.
    - Erase the movable objects from their original positions, then process matched pairs from left bracket to right bracket.
    - For each one, center the movable object's bbox horizontally on that bracket's interior span and let it fall so its bottom row reaches the grid bottom or sits exactly one row above the current top of the already placed stack.
    - Erase the reference shapes while keeping the grey brackets unchanged.

    Key insight:
    - The bracket contents are geometric address tags: colors are irrelevant, so the solver must normalize masks and use rotation/reflection equivalence before applying the global left-to-right stacking order.
    """
    g = np.asarray(grid, dtype=int)
    bg, brackets, references, movers = _extract_structure(g)

    ref_by_sig = {_shape_signature(_tight_bool(ref["coords"])): (bracket, ref) for bracket, ref in zip(brackets, references)}
    assert len(ref_by_sig) == 3, "reference shapes must be pairwise distinct under dihedral equivalence"

    placements = []
    for mover in movers:
        sig = _shape_signature(_tight_bool(mover["coords"]))
        assert sig in ref_by_sig, "every mover must match a reference by shape"
        placements.append((ref_by_sig[sig][0], ref_by_sig[sig][1], mover))

    assert len({int(item[0]["bbox"][1]) for item in placements}) == 3, "matching must be one-to-one"
    ordered = [(bracket, mover) for bracket, _ref, mover in sorted(placements, key=lambda item: int(item[0]["bbox"][1]))]

    return _settled_output(g, [(bracket, mover) for bracket, mover in ordered])


SHAPE_LIBRARY = [
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),  # L4
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),    # T4
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),    # S4
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),  # P5
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),    # U5
]


def _shape_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _patch_from_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _all_non_bg_cells(patch: np.ndarray, r0: int, c0: int, bg: int) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(patch != int(bg)):
        cells.add((int(r0) + int(r), int(c0) + int(c)))
    return cells


def _can_place_with_gap(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    H, W = grid.shape
    cells = _all_non_bg_cells(patch, r0, c0, bg)
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells and int(grid[nr, nc]) != int(bg):
                return False
    return True


def _place_with_gap(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    assert _can_place_with_gap(grid, patch, r0, c0, bg)
    _paste(grid, patch, r0, c0, bg)


def _choose_shapes() -> list[np.ndarray]:
    for _ in range(200):
        shapes = random.sample(SHAPE_LIBRARY, 3)
        areas = [_shape_area(shape) for shape in shapes]
        if max(Counter(areas).values()) >= 2:
            sigs = {_shape_signature(shape) for shape in shapes}
            if len(sigs) == 3:
                return [shape.copy() for shape in shapes]
    raise RuntimeError("failed to choose three distinct canonical shapes")


def _choose_variants(shape: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    variants = _dihedral_variants(shape)
    options = []
    for inner_width in (3, 4, 5):
        fitting_refs = [variant for variant in variants if variant.shape[1] <= inner_width]
        fitting_movers = [
            variant
            for variant in variants
            if variant.shape[1] <= inner_width and (inner_width - variant.shape[1]) % 2 == 0
        ]
        if fitting_refs and fitting_movers:
            options.append((inner_width, fitting_refs, fitting_movers))
    assert options
    inner_width, refs, movers = random.choice(options)
    ref_variant = random.choice(refs).copy()
    mover_variant = random.choice(movers).copy()
    return int(inner_width), ref_variant, mover_variant


def _layout_brackets(inner_widths: list[int]) -> tuple[int, list[int], list[int]]:
    total_widths = [width + 2 for width in inner_widths]
    for _ in range(400):
        gaps = [random.randint(1, 3), random.randint(1, 3)]
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        base = left_margin + sum(total_widths) + sum(gaps) + right_margin
        if base > 28:
            continue
        W = random.randint(base, min(30, base + 2))
        extra = W - base
        left_margin += random.randint(0, extra)
        starts = [left_margin]
        starts.append(starts[0] + total_widths[0] + gaps[0])
        starts.append(starts[1] + total_widths[1] + gaps[1])
        if starts[2] + total_widths[2] + right_margin > W:
            continue
        return W, starts, total_widths
    raise RuntimeError("failed to place three brackets")


def _metadata_output(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in records:
        out[record["ref_cells"][:, 0], record["ref_cells"][:, 1]] = BG
        out[record["mover_cells"][:, 0], record["mover_cells"][:, 1]] = BG
    current_top = inp.shape[0]
    for record in sorted(records, key=lambda rec: int(rec["bracket_start"])):
        patch = record["mover_patch"]
        dest_r = current_top - patch.shape[0]
        dest_c = record["bracket_start"] + 1 + (record["inner_width"] - patch.shape[1]) // 2
        _paste(out, patch, dest_r, dest_c, BG)
        current_top = dest_r
    return out


def _naive_bottom_output(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in records:
        out[record["ref_cells"][:, 0], record["ref_cells"][:, 1]] = BG
        out[record["mover_cells"][:, 0], record["mover_cells"][:, 1]] = BG
    for record in sorted(records, key=lambda rec: int(rec["bracket_start"])):
        patch = record["mover_patch"]
        dest_r = inp.shape[0] - patch.shape[0]
        dest_c = record["bracket_start"] + 1 + (record["inner_width"] - patch.shape[1]) // 2
        _paste(out, patch, dest_r, dest_c, BG)
    return out


def _reverse_order_output(inp: np.ndarray, records: list[dict]) -> np.ndarray:
    out = inp.copy()
    for record in records:
        out[record["ref_cells"][:, 0], record["ref_cells"][:, 1]] = BG
        out[record["mover_cells"][:, 0], record["mover_cells"][:, 1]] = BG
    current_top = inp.shape[0]
    for record in sorted(records, key=lambda rec: int(rec["bracket_start"]), reverse=True):
        patch = record["mover_patch"]
        dest_r = current_top - patch.shape[0]
        dest_c = record["bracket_start"] + 1 + (record["inner_width"] - patch.shape[1]) // 2
        _paste(out, patch, dest_r, dest_c, BG)
        current_top = dest_r
    return out


def _generate_attempt() -> np.ndarray | None:
    shapes = _choose_shapes()
    ref_colors = list(REFERENCE_COLORS)
    mover_colors = list(MOVER_COLORS)
    random.shuffle(ref_colors)
    random.shuffle(mover_colors)

    records = []
    inner_widths = []
    needs_rotated_pair = False
    ref_bottom = 1
    total_stack_height = 0

    for idx, shape in enumerate(shapes):
        inner_width, ref_variant, mover_variant = _choose_variants(shape)
        ref_patch = _patch_from_mask(ref_variant, ref_colors[idx], BG)
        mover_patch = _patch_from_mask(mover_variant, mover_colors[idx], BG)
        inner_widths.append(inner_width)
        total_stack_height += mover_patch.shape[0]
        ref_bottom = max(ref_bottom, 1 + ref_patch.shape[0] - 1)
        if ref_patch.shape != mover_patch.shape or not np.array_equal(ref_patch != BG, mover_patch != BG):
            needs_rotated_pair = True
        records.append(
            {
                "shape_signature": _shape_signature(shape),
                "canonical_area": _shape_area(shape),
                "inner_width": inner_width,
                "ref_patch": ref_patch,
                "mover_patch": mover_patch,
            }
        )

    if not needs_rotated_pair:
        return None

    W, starts, total_widths = _layout_brackets(inner_widths)
    gap_rows = random.randint(3, 5)
    min_stack_clearance = ref_bottom + 2
    H_min = max(15, total_stack_height + min_stack_clearance)
    H = random.randint(H_min, min(24, H_min + 4))
    if H > 30 or W > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)

    for record, start, total_width in zip(records, starts, total_widths):
        grid[0, start : start + total_width] = BRACKET
        grid[1, start] = BRACKET
        grid[1, start + total_width - 1] = BRACKET

        inner_left = start + 1
        slack = record["inner_width"] - record["ref_patch"].shape[1]
        ref_c = inner_left + (0 if slack == 0 else random.randint(0, slack))
        ref_r = 1
        if not _can_place_opaque(grid, record["ref_patch"], ref_r, ref_c, BG):
            return None
        _paste(grid, record["ref_patch"], ref_r, ref_c, BG)
        record["bracket_start"] = int(start)
        record["ref_top_left"] = (int(ref_r), int(ref_c))
        record["ref_cells"] = np.array(sorted(_all_non_bg_cells(record["ref_patch"], ref_r, ref_c, BG)), dtype=int)

    mover_top_min = ref_bottom + gap_rows + 1
    if mover_top_min >= H - 1:
        return None

    mover_order = list(range(3))
    random.shuffle(mover_order)
    for pos, idx in enumerate(mover_order):
        record = records[idx]
        patch = record["mover_patch"]
        placed = False
        row_lo = mover_top_min
        row_hi = H - patch.shape[0]
        col_hi = W - patch.shape[1]
        if row_lo > row_hi or col_hi < 0:
            return None
        row_choices = [row_lo] if pos == 0 else list(range(row_lo, row_hi + 1))
        random.shuffle(row_choices)
        for r0 in row_choices[: min(len(row_choices), 40)]:
            for _ in range(8):
                c0 = random.randint(0, col_hi)
                if not _can_place_with_gap(grid, patch, r0, c0, BG):
                    continue
                _place_with_gap(grid, patch, r0, c0, BG)
                record["mover_top_left"] = (int(r0), int(c0))
                record["mover_cells"] = np.array(sorted(_all_non_bg_cells(patch, r0, c0, BG)), dtype=int)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    top_rows = [int(record["mover_cells"][:, 0].min()) for record in records]
    if min(top_rows) - ref_bottom - 1 != gap_rows:
        return None

    expected = _metadata_output(grid, records)
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(expected, _naive_bottom_output(grid, records)):
        return None
    if np.array_equal(expected, _reverse_order_output(grid, records)):
        return None

    bg, brackets, references, movers = _extract_structure(grid)
    del bg
    if len({_shape_signature(_tight_bool(ref["coords"])) for ref in references}) != 3:
        return None
    if Counter(record["canonical_area"] for record in records).most_common(1)[0][1] < 2:
        return None

    raw_diff_count = 0
    for record in records:
        if not np.array_equal(record["ref_patch"] != BG, record["mover_patch"] != BG):
            raw_diff_count += 1
    if raw_diff_count == 0:
        return None

    mover_sigs = {_shape_signature(_tight_bool(comp["coords"])) for comp in movers}
    ref_sigs = {_shape_signature(_tight_bool(comp["coords"])) for comp in references}
    if mover_sigs != ref_sigs:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

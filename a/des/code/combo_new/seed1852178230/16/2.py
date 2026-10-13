from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


@dataclass
class Component:
    index: int
    color: int
    coords: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    mask: np.ndarray
    h: int
    w: int
    left_gap: np.ndarray
    right_gap: np.ndarray
    top_gap: np.ndarray
    bottom_gap: np.ndarray
    cells: frozenset[tuple[int, int]]


@dataclass
class Candidate:
    receiver: int
    donor: int
    side: str
    depth: int
    final_top: int
    final_left: int
    receiver_cells: frozenset[tuple[int, int]]
    donor_cells: frozenset[tuple[int, int]]
    union_cells: frozenset[tuple[int, int]]
    move_distance: int


def find_components(grid: np.ndarray, bg: int) -> list[Component]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            row_has = np.any(mask, axis=1)
            col_has = np.any(mask, axis=0)
            assert np.all(row_has) and np.all(col_has), "bbox should touch every row and column"

            left_gap = np.array([int(np.flatnonzero(mask[row])[0]) for row in range(mask.shape[0])], dtype=int)
            right_gap = np.array(
                [int(mask.shape[1] - 1 - np.flatnonzero(mask[row])[-1]) for row in range(mask.shape[0])],
                dtype=int,
            )
            top_gap = np.array([int(np.flatnonzero(mask[:, col])[0]) for col in range(mask.shape[1])], dtype=int)
            bottom_gap = np.array(
                [int(mask.shape[0] - 1 - np.flatnonzero(mask[:, col])[-1]) for col in range(mask.shape[1])],
                dtype=int,
            )

            comps.append(
                Component(
                    index=len(comps),
                    color=color,
                    coords=np.array(coords, dtype=int),
                    bbox=bbox,
                    area=len(coords),
                    mask=mask,
                    h=mask.shape[0],
                    w=mask.shape[1],
                    left_gap=left_gap,
                    right_gap=right_gap,
                    top_gap=top_gap,
                    bottom_gap=bottom_gap,
                    cells=frozenset(coords),
                )
            )
    return comps


def _mask_from_side_profile(side: str, height: int, width: int, profile: np.ndarray) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if side == "left":
        for r, gap in enumerate(profile):
            mask[r, int(gap) :] = True
    elif side == "right":
        for r, gap in enumerate(profile):
            mask[r, : width - int(gap)] = True
    elif side == "top":
        for c, gap in enumerate(profile):
            mask[int(gap) :, c] = True
    elif side == "bottom":
        for c, gap in enumerate(profile):
            mask[: height - int(gap), c] = True
    else:
        raise ValueError(f"unknown side: {side}")
    return mask


def _translate_cells(coords: np.ndarray, top: int, left: int, origin_bbox: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = origin_bbox
    dr = int(top - r0)
    dc = int(left - c0)
    return frozenset((int(r + dr), int(c + dc)) for r, c in coords)


def _constant_sum(a: np.ndarray, b: np.ndarray) -> int | None:
    total = a + b
    if total.size == 0 or not np.all(total == total[0]):
        return None
    depth = int(total[0])
    if depth <= 0:
        return None
    return depth


def _build_candidate(
    receiver: Component,
    donor: Component,
    side: str,
    depth: int,
    grid_shape: tuple[int, int],
) -> Candidate | None:
    h, w = grid_shape
    rr0, rc0, _rr1, _rc1 = receiver.bbox
    dr0, dc0, _dr1, _dc1 = donor.bbox

    if side == "left":
        final_top = rr0
        final_left = rc0 - (receiver.w - depth - 1)
        if not (dc0 < final_left <= rc0):
            return None
    elif side == "right":
        final_top = rr0
        final_left = rc0 + (receiver.w - depth - 1)
        if not (rc0 <= final_left < dc0):
            return None
    elif side == "top":
        final_top = rr0 - (receiver.h - depth - 1)
        final_left = rc0
        if not (dr0 < final_top <= rr0):
            return None
    elif side == "bottom":
        final_top = rr0 + (receiver.h - depth - 1)
        final_left = rc0
        if not (rr0 <= final_top < dr0):
            return None
    else:
        raise ValueError(f"unknown side: {side}")

    if final_top < 0 or final_left < 0 or final_top + donor.h > h or final_left + donor.w > w:
        return None

    moved = _translate_cells(donor.coords, final_top, final_left, donor.bbox)
    overlap = receiver.cells & moved
    if not overlap:
        return None

    return Candidate(
        receiver=receiver.index,
        donor=donor.index,
        side=side,
        depth=depth,
        final_top=int(final_top),
        final_left=int(final_left),
        receiver_cells=receiver.cells,
        donor_cells=moved,
        union_cells=receiver.cells | moved,
        move_distance=abs(int(final_top - dr0)) + abs(int(final_left - dc0)),
    )


def _generate_candidates(components: list[Component], grid_shape: tuple[int, int]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for receiver in components:
        for donor in components:
            if receiver.index == donor.index:
                continue
            if receiver.h != donor.h or receiver.w != donor.w:
                continue

            rr0, rc0, _rr1, _rc1 = receiver.bbox
            dr0, dc0, _dr1, _dc1 = donor.bbox

            if rr0 == dr0:
                depth = _constant_sum(receiver.left_gap, donor.right_gap)
                if depth is not None:
                    cand = _build_candidate(receiver, donor, "left", depth, grid_shape)
                    if cand is not None:
                        candidates.append(cand)

                depth = _constant_sum(receiver.right_gap, donor.left_gap)
                if depth is not None:
                    cand = _build_candidate(receiver, donor, "right", depth, grid_shape)
                    if cand is not None:
                        candidates.append(cand)

            if rc0 == dc0:
                depth = _constant_sum(receiver.top_gap, donor.bottom_gap)
                if depth is not None:
                    cand = _build_candidate(receiver, donor, "top", depth, grid_shape)
                    if cand is not None:
                        candidates.append(cand)

                depth = _constant_sum(receiver.bottom_gap, donor.top_gap)
                if depth is not None:
                    cand = _build_candidate(receiver, donor, "bottom", depth, grid_shape)
                    if cand is not None:
                        candidates.append(cand)

    candidates.sort(
        key=lambda c: (
            c.receiver,
            c.donor,
            c.side,
            c.final_top,
            c.final_left,
            c.move_distance,
        )
    )
    return candidates


def _match_candidates(components: list[Component], candidates: list[Candidate]) -> list[Candidate]:
    if len(components) % 2 != 0:
        raise AssertionError("expected an even number of pieces")

    by_index: dict[int, list[int]] = defaultdict(list)
    for idx, cand in enumerate(candidates):
        by_index[cand.receiver].append(idx)
        by_index[cand.donor].append(idx)

    remaining_all = frozenset(comp.index for comp in components)

    def search(remaining: frozenset[int], occupied: frozenset[tuple[int, int]]) -> list[int] | None:
        if not remaining:
            return []

        pivot = min(remaining)
        for cand_idx in by_index[pivot]:
            cand = candidates[cand_idx]
            other = cand.donor if cand.receiver == pivot else cand.receiver
            if other not in remaining:
                continue
            if cand.union_cells & occupied:
                continue
            nxt = search(remaining - {pivot, other}, occupied | cand.union_cells)
            if nxt is not None:
                return [cand_idx] + nxt
        return None

    match = search(remaining_all, frozenset())
    if match is None:
        raise AssertionError("could not find a full set of complementary pairs")
    return [candidates[idx] for idx in match]


def _assemble(grid: np.ndarray, components: list[Component], matched: list[Candidate]) -> np.ndarray:
    out = np.full_like(np.asarray(grid, dtype=int), get_bg(grid))
    comp_by_index = {comp.index: comp for comp in components}

    for cand in matched:
        receiver = comp_by_index[cand.receiver]
        donor = comp_by_index[cand.donor]
        seam_color = receiver.color if receiver.area < donor.area else donor.color
        overlap = cand.receiver_cells & cand.donor_cells

        for r, c in cand.receiver_cells:
            out[r, c] = receiver.color
        for r, c in cand.donor_cells - overlap:
            out[r, c] = donor.color
        for r, c in overlap:
            out[r, c] = seam_color

    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg)
    assert components, "expected at least one foreground piece"
    dims = {(comp.h, comp.w) for comp in components}
    assert len(dims) == 1, "all pieces must share the same bbox dimensions"
    candidates = _generate_candidates(components, g.shape)
    matched = _match_candidates(components, candidates)
    return _assemble(g, components, matched)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock complementary jagged-edged pieces into one another and color the seam with the smaller-area piece.

    Input structure:
    - The background is the dominant color.
    - Several disconnected monochrome pieces share one common bbox size.
    - Each piece is a near-rectangular block whose silhouette differs from its bbox on one docking side, producing a jagged notch or its complementary tab profile.
    - Matching pieces are already aligned on the perpendicular axis, so they can move only left, right, up, or down.

    Rule:
    - Find complementary pairs by comparing facing edge-gap profiles: a receiver notch and a donor tab pair up when their row-wise or column-wise gaps add to one constant depth.
    - Keep the notched receiver fixed.
    - Translate the complementary donor along that side until its profile interlocks with the receiver, creating a one-cell-thick overlap seam.
    - Repaint overlap cells with the color of the smaller visible input piece in that pair; keep the donor color on donor-only cells, the receiver color on receiver-only cells, and leave the background unchanged.

    Key insight:
    - The pieces are matched by geometric complementarity of their jagged boundary profiles, and the overlap is not resolved by draw order: the seam color is determined by comparing the two input piece areas.
    """
    return _solve(np.asarray(grid, dtype=int))


# --------------- generator ---------------

MASTER_SIZE = 30
PAIR_COUNT_RANGE = (2, 4)
PIECE_HEIGHT_RANGE = (4, 8)
PIECE_WIDTH_RANGE = (4, 8)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 300
COLORS = tuple(range(10))
SIDES = ("left", "right", "top", "bottom")


def _expanded(cells: frozenset[tuple[int, int]], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return frozenset(out)


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> frozenset[tuple[int, int]]:
    rows, cols = np.where(mask)
    return frozenset((int(top + r), int(left + c)) for r, c in zip(rows, cols))


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def _paint_pair_output(
    grid: np.ndarray,
    receiver_mask: np.ndarray,
    donor_mask: np.ndarray,
    receiver_top: int,
    receiver_left: int,
    donor_top: int,
    donor_left: int,
    receiver_color: int,
    donor_color: int,
) -> None:
    receiver_cells = _cells_from_mask(receiver_mask, receiver_top, receiver_left)
    donor_cells = _cells_from_mask(donor_mask, donor_top, donor_left)
    receiver_area = int(receiver_mask.sum())
    donor_area = int(donor_mask.sum())
    seam_color = receiver_color if receiver_area < donor_area else donor_color
    overlap = receiver_cells & donor_cells

    for r, c in receiver_cells:
        grid[r, c] = receiver_color
    for r, c in donor_cells - overlap:
        grid[r, c] = donor_color
    for r, c in overlap:
        grid[r, c] = seam_color


def _random_profile(length: int, depth: int, receiver_smaller: bool) -> np.ndarray:
    threshold = length * depth / 2.0
    for _ in range(200):
        profile = np.array([random.randint(0, depth) for _ in range(length)], dtype=int)
        if profile.max() == 0 or np.all(profile == profile[0]):
            continue
        if len(set(int(x) for x in profile)) < 3:
            continue
        if np.count_nonzero(np.diff(profile)) < 2:
            continue
        total = int(profile.sum())
        if total == threshold:
            continue
        if receiver_smaller and not (total > threshold):
            continue
        if (not receiver_smaller) and not (total < threshold):
            continue
        return profile
    raise RuntimeError("failed to sample a suitable profile")


def _receiver_and_donor_masks(side: str, height: int, width: int, depth: int, receiver_smaller: bool) -> tuple[np.ndarray, np.ndarray]:
    profile_len = height if side in ("left", "right") else width
    profile = _random_profile(profile_len, depth, receiver_smaller)
    receiver_mask = _mask_from_side_profile(side, height, width, profile)
    donor_side = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}[side]
    donor_mask = _mask_from_side_profile(donor_side, height, width, depth - profile)
    return receiver_mask, donor_mask


def _try_place_pair(
    side: str,
    depth: int,
    receiver_mask: np.ndarray,
    donor_mask: np.ndarray,
    input_blocked: frozenset[tuple[int, int]],
    output_blocked: frozenset[tuple[int, int]],
) -> dict | None:
    height, width = receiver_mask.shape

    if side in ("left", "right"):
        shift = width - depth - 1
        if shift <= 0:
            return None
    else:
        shift = height - depth - 1
        if shift <= 0:
            return None

    for _ in range(PLACEMENT_RETRIES):
        if side == "left":
            rr = random.randint(0, MASTER_SIZE - height)
            rc = random.randint(shift + width + 1, MASTER_SIZE - width)
            final_top, final_left = rr, rc - shift
            init_top = rr
            init_left = random.randint(0, final_left - width - 1)
        elif side == "right":
            rr = random.randint(0, MASTER_SIZE - height)
            max_rc = MASTER_SIZE - (shift + 2 * width + 1)
            if max_rc < 0:
                return None
            rc = random.randint(0, max_rc)
            final_top, final_left = rr, rc + shift
            init_top = rr
            init_left = random.randint(final_left + width + 1, MASTER_SIZE - width)
        elif side == "top":
            rc = random.randint(0, MASTER_SIZE - width)
            rr = random.randint(shift + height + 1, MASTER_SIZE - height)
            final_top, final_left = rr - shift, rc
            init_left = rc
            init_top = random.randint(0, final_top - height - 1)
        else:
            rc = random.randint(0, MASTER_SIZE - width)
            max_rr = MASTER_SIZE - (shift + 2 * height + 1)
            if max_rr < 0:
                return None
            rr = random.randint(0, max_rr)
            final_top, final_left = rr + shift, rc
            init_left = rc
            init_top = random.randint(final_top + height + 1, MASTER_SIZE - height)

        receiver_cells = _cells_from_mask(receiver_mask, rr, rc)
        donor_input_cells = _cells_from_mask(donor_mask, init_top, init_left)
        pair_input = receiver_cells | donor_input_cells
        if pair_input & input_blocked:
            continue

        donor_output_cells = _cells_from_mask(donor_mask, final_top, final_left)
        pair_output = receiver_cells | donor_output_cells
        if pair_output & output_blocked:
            continue

        return {
            "receiver_top": rr,
            "receiver_left": rc,
            "donor_input_top": init_top,
            "donor_input_left": init_left,
            "donor_output_top": final_top,
            "donor_output_left": final_left,
            "receiver_cells": receiver_cells,
            "donor_input_cells": donor_input_cells,
            "donor_output_cells": donor_output_cells,
            "input_blocked": _expanded(pair_input, (MASTER_SIZE, MASTER_SIZE)),
            "output_blocked": _expanded(pair_output, (MASTER_SIZE, MASTER_SIZE)),
        }
    return None


def _crop_pair(input_grid: np.ndarray, output_grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    occ = np.argwhere((input_grid != bg) | (output_grid != bg))
    r0, c0 = occ.min(axis=0)
    r1, c1 = occ.max(axis=0)

    top_margin = random.randint(0, int(r0))
    left_margin = random.randint(0, int(c0))
    bottom_margin = random.randint(0, int(MASTER_SIZE - 1 - r1))
    right_margin = random.randint(0, int(MASTER_SIZE - 1 - c1))

    r0 -= top_margin
    c0 -= left_margin
    r1 += bottom_margin
    c1 += right_margin

    return input_grid[r0 : r1 + 1, c0 : c1 + 1].copy(), output_grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(COLORS)
    available_colors = [color for color in COLORS if color != bg]
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    if pair_count * 2 > len(available_colors):
        return None

    height = random.randint(*PIECE_HEIGHT_RANGE)
    width = random.randint(*PIECE_WIDTH_RANGE)
    if min(height, width) < 4:
        return None

    input_grid = np.full((MASTER_SIZE, MASTER_SIZE), bg, dtype=int)
    output_grid = np.full((MASTER_SIZE, MASTER_SIZE), bg, dtype=int)
    input_blocked: frozenset[tuple[int, int]] = frozenset()
    output_blocked: frozenset[tuple[int, int]] = frozenset()

    colors = random.sample(available_colors, pair_count * 2)
    sides = random.choices(SIDES, k=pair_count)
    receiver_smaller_flags = [random.choice([True, False]) for _ in range(pair_count)]

    for pair_idx in range(pair_count):
        side = sides[pair_idx]
        receiver_smaller = receiver_smaller_flags[pair_idx]
        max_depth = (width - 2) if side in ("left", "right") else (height - 2)
        if max_depth < 2:
            return None
        depth = random.randint(2, max_depth)
        receiver_mask, donor_mask = _receiver_and_donor_masks(side, height, width, depth, receiver_smaller)

        placement = _try_place_pair(side, depth, receiver_mask, donor_mask, input_blocked, output_blocked)
        if placement is None:
            return None

        receiver_color = colors[2 * pair_idx]
        donor_color = colors[2 * pair_idx + 1]

        _paint_mask(input_grid, receiver_mask, placement["receiver_top"], placement["receiver_left"], receiver_color)
        _paint_mask(input_grid, donor_mask, placement["donor_input_top"], placement["donor_input_left"], donor_color)
        _paint_pair_output(
            output_grid,
            receiver_mask,
            donor_mask,
            placement["receiver_top"],
            placement["receiver_left"],
            placement["donor_output_top"],
            placement["donor_output_left"],
            receiver_color,
            donor_color,
        )

        input_blocked = input_blocked | placement["input_blocked"]
        output_blocked = output_blocked | placement["output_blocked"]

    cropped_inp, cropped_out = _crop_pair(input_grid, output_grid, bg)
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if Counter(cropped_inp.ravel()).most_common(1)[0][0] != bg:
        return None
    if np.count_nonzero(cropped_inp != bg) == 0:
        return None

    try:
        solved = main(cropped_inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, cropped_out):
        return None
    return cropped_inp, cropped_out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid docking-pieces input")

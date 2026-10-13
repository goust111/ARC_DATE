from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 300
PLACE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_dicts(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_coords(coord_arr)
            comp_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            comp_mask[coord_arr[:, 0] - r0, coord_arr[:, 1] - c0] = True
            comps.append(
                {
                    "coords": coord_arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": comp_mask,
                    "area": int(coord_arr.shape[0]),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    return comps


def _row_projection(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.sum(axis=1))


def _find_divider(grid: np.ndarray) -> int:
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GRAY)]
    assert len(divider_cols) == 1, "expected exactly one full-height gray divider"
    return int(divider_cols[0])


def _extract_histogram(grid: np.ndarray, divider_c: int) -> tuple[int, tuple[int, ...], np.ndarray]:
    left_red = grid[:, :divider_c] == RED
    rows = np.flatnonzero(left_red.any(axis=1))
    assert len(rows) > 0, "expected at least one red histogram row"
    assert int(rows[-1] - rows[0] + 1) == len(rows), "histogram rows must be contiguous"

    lengths: list[int] = []
    bar_mask = np.zeros_like(grid, dtype=bool)
    for r in range(int(rows[0]), int(rows[-1]) + 1):
        cols = np.flatnonzero(left_red[r])
        assert len(cols) > 0
        assert int(cols[-1]) == divider_c - 1, "bars must touch the divider"
        assert len(cols) == int(cols[-1] - cols[0] + 1), "each histogram row must be a solid run"
        lengths.append(int(len(cols)))
        bar_mask[r, cols] = True

    return int(rows[0]), tuple(lengths), bar_mask


def _solve_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_c = _find_divider(g)
    band_top, lengths, bar_mask = _extract_histogram(g, divider_c)

    blue_mask = g == BLUE
    blue_mask[:, : divider_c + 1] = False
    comps = _component_dicts(blue_mask)
    assert comps, "expected blue candidate shapes on the right"

    matches: list[tuple[int, int, np.ndarray]] = []
    for idx, comp in enumerate(comps):
        for rot in range(4):
            rotated = np.rot90(comp["mask"], rot)
            if _row_projection(rotated) == lengths:
                matches.append((idx, rot, rotated))

    assert len(matches) == 1, "expected exactly one matching shape+rotation"
    match_idx, match_rot, rotated_mask = matches[0]
    match_comp = comps[match_idx]

    dest_r0 = band_top
    dest_c0 = divider_c + 1
    dest_h, dest_w = rotated_mask.shape
    assert dest_r0 + dest_h <= g.shape[0]
    assert dest_c0 + dest_w <= g.shape[1]

    return {
        "divider_c": divider_c,
        "band_top": band_top,
        "lengths": lengths,
        "bar_mask": bar_mask,
        "components": comps,
        "match_idx": match_idx,
        "match_rot": match_rot,
        "match_comp": match_comp,
        "rotated_mask": rotated_mask,
        "dest_r0": dest_r0,
        "dest_c0": dest_c0,
    }


def _build_output(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[info["bar_mask"]] = BLACK

    for r, c in info["match_comp"]["coords"]:
        out[int(r), int(c)] = BLACK

    rotated = info["rotated_mask"]
    r0 = int(info["dest_r0"])
    c0 = int(info["dest_c0"])
    h, w = rotated.shape
    window = out[r0 : r0 + h, c0 : c0 + w]
    assert not np.any(window[rotated] != BLACK), "destination beside the divider must be empty"
    window[rotated] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Claim the histogram's owner shape, rotate it to the matching shadow, and move it beside the divider.

    Input structure:
    - A full-height gray vertical divider splits the black grid into a left histogram zone and a right shape zone.
    - On the left, red horizontal bars touch the divider and occupy one contiguous row band.
    - On the right, several blue connected shapes appear in arbitrary rotations; exactly one shape has one rotation whose per-row pixel counts match the red bar lengths.

    Rule:
    - Read the red histogram as a top-to-bottom sequence of bar lengths.
    - For every blue shape, test all four rotations by cropping its tight bbox and comparing the row-count sequence.
    - Remove the left histogram and the matched blue shape from its original location, then paste that shape in the matching rotation immediately to the divider's right, aligned to the histogram's row band, and recolor it red.

    Key insight:
    - The left bars are the missing shape's horizontal shadow, so identity and orientation are both recovered from the unique row-projection match.
    """
    info = _solve_info(grid)
    return _build_output(grid, info)


def _frontier(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return list(frontier)


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _canonical_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for rot in range(4):
        turned = np.rot90(mask, rot)
        variants.append((turned.shape, turned.tobytes()))
    return min(variants)


def _random_shape(
    *,
    exact_height: int | None = None,
    exact_area: int | None = None,
    min_height: int = 3,
    max_height: int = 6,
    max_width: int = 6,
    min_area: int = 5,
    max_area: int = 12,
    allow_rectangle: bool = False,
) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        height = int(exact_height if exact_height is not None else random.randint(min_height, max_height))
        if exact_area is None:
            low = max(min_area, height + 1)
            high = min(max_area, height * max_width - (0 if allow_rectangle else 1))
            if low > high:
                continue
            area = random.randint(low, high)
        else:
            area = int(exact_area)
            if area < height:
                continue
            if area > height * max_width:
                continue

        min_width = max(2, math.ceil(area / height))
        if min_width > max_width:
            continue
        width = random.randint(min_width, max_width)
        if area > height * width:
            continue

        cells = {(0, random.randrange(width))}
        prev_c = next(iter(cells))[1]
        for r in range(1, height):
            prev_c = min(max(prev_c + random.choice((-1, 0, 1)), 0), width - 1)
            cells.add((r, prev_c))

        frontier = _frontier(cells, height, width)
        while len(cells) < area and frontier:
            cell = random.choice(frontier)
            cells.add(cell)
            frontier = _frontier(cells, height, width)

        if len(cells) != area:
            continue

        mask = _mask_from_cells(cells)
        if exact_height is not None and mask.shape[0] != exact_height:
            continue
        if mask.shape[1] < 2:
            continue
        if not allow_rectangle and np.all(mask):
            continue
        return mask
    return None


def _generate_target_shape() -> tuple[np.ndarray, int, tuple[int, ...]]:
    for _ in range(GEN_ATTEMPTS):
        mask = _random_shape()
        if mask is None:
            continue
        projections = [_row_projection(np.rot90(mask, rot)) for rot in range(4)]
        unique_rots = [rot for rot in range(4) if projections.count(projections[rot]) == 1]
        if not unique_rots:
            continue
        match_rot = random.choice(unique_rots)
        lengths = projections[match_rot]
        if len(lengths) < 3:
            continue
        if len(set(lengths)) < 2:
            continue
        if max(lengths) < 2:
            continue
        return mask, int(match_rot), lengths
    raise RuntimeError("failed to generate a target shape with a unique projection")


def _generate_distractor(
    target_seq: tuple[int, ...],
    target_area: int,
    target_height: int,
    used_shapes: set[tuple[tuple[int, int], bytes]],
    *,
    require_same_area: bool = False,
    require_same_height: bool = False,
) -> np.ndarray | None:
    for _ in range(GEN_ATTEMPTS):
        mask = _random_shape(
            exact_height=target_height if require_same_height else None,
            exact_area=target_area if require_same_area else None,
            min_height=2,
            max_height=6,
            max_width=6,
            min_area=4,
            max_area=12,
            allow_rectangle=False,
        )
        if mask is None:
            continue
        if any(_row_projection(np.rot90(mask, rot)) == target_seq for rot in range(4)):
            continue
        key = _canonical_mask_key(mask)
        if key in used_shapes:
            continue
        used_shapes.add(key)
        return mask
    return None


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = blocked.shape
    if r0 < 1 or c0 < 1 or r0 + h >= hh or c0 + w >= ww:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = blocked.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_records(
    height: int,
    width: int,
    divider_c: int,
    dest_r0: int,
    dest_c0: int,
    dest_h: int,
    dest_w: int,
    records: list[dict],
) -> dict[str, tuple[int, int]] | None:
    for _ in range(PLACE_ATTEMPTS):
        blocked = np.zeros((height, width), dtype=bool)
        blocked[:, : divider_c + 1] = True
        _mark_blocked(blocked, dest_r0, dest_c0, dest_h, dest_w, pad=1)

        positions: dict[str, tuple[int, int]] = {}
        ordered = sorted(records, key=lambda rec: rec["mask"].shape[0] * rec["mask"].shape[1], reverse=True)
        ok = True
        for rec in ordered:
            h, w = rec["mask"].shape
            candidates = [
                (r0, c0)
                for r0 in range(1, height - h)
                for c0 in range(divider_c + 2, width - w)
                if _can_place(blocked, r0, c0, h, w, pad=1)
            ]
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            positions[rec["id"]] = pos
            _mark_blocked(blocked, pos[0], pos[1], h, w, pad=1)
        if ok:
            return positions
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[mask] = int(color)


def _cells_at(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        target_base, match_rot, lengths = _generate_target_shape()
        target_oriented = np.rot90(target_base, match_rot)
        band_h, dest_w = target_oriented.shape
        target_area = int(target_base.sum())
        hist_w = max(lengths)

        display_rot = random.choice([rot for rot in range(4) if rot != match_rot]) if random.random() < 0.75 else random.randrange(4)
        target_display = np.rot90(target_base, display_rot)
        needed_rot = (match_rot - display_rot) % 4

        used_shapes = {_canonical_mask_key(target_base)}

        same_area = _generate_distractor(lengths, target_area, band_h, used_shapes, require_same_area=True)
        same_height = _generate_distractor(lengths, target_area, band_h, used_shapes, require_same_height=True)
        if same_area is None or same_height is None:
            continue

        distractors = [same_area, same_height]
        extra_count = random.randint(0, 2)
        for _extra in range(extra_count):
            extra = _generate_distractor(lengths, target_area, band_h, used_shapes)
            if extra is None:
                break
            distractors.append(extra)

        records = [{"id": "target", "mask": target_display}]
        for idx, mask in enumerate(distractors):
            records.append({"id": f"d{idx}", "mask": np.rot90(mask, random.randrange(4))})

        min_h = max(14, band_h + 4, max(rec["mask"].shape[0] for rec in records) + 4)
        min_w = max(
            18,
            hist_w + 1 + dest_w + 8,
            hist_w + 1 + max(rec["mask"].shape[1] for rec in records) + 5,
        )
        if min_h > 30 or min_w > 30:
            continue

        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        divider_c = hist_w + random.randint(0, min(3, width - min_w))
        if divider_c + 1 + dest_w >= width:
            continue

        band_top_max = height - band_h - 2
        if band_top_max < 1:
            continue
        band_top = random.randint(1, band_top_max)
        dest_r0 = band_top
        dest_c0 = divider_c + 1

        positions = _place_records(height, width, divider_c, dest_r0, dest_c0, band_h, dest_w, records)
        if positions is None:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        grid[:, divider_c] = GRAY

        bar_mask = np.zeros_like(grid, dtype=bool)
        for offset, length in enumerate(lengths):
            r = band_top + offset
            c0 = divider_c - int(length)
            grid[r, c0:divider_c] = RED
            bar_mask[r, c0:divider_c] = True

        for rec in records:
            r0, c0 = positions[rec["id"]]
            _paint_mask(grid, rec["mask"], r0, c0, BLUE)

        expected = grid.copy()
        expected[bar_mask] = BLACK
        target_r0, target_c0 = positions["target"]
        expected[target_r0 : target_r0 + target_display.shape[0], target_c0 : target_c0 + target_display.shape[1]][target_display] = BLACK
        _paint_mask(expected, target_oriented, dest_r0, dest_c0, RED)

        try:
            info = _solve_info(grid)
        except AssertionError:
            continue

        source_cells = _cells_at(target_display, target_r0, target_c0)
        matched_cells = {tuple(map(int, cell)) for cell in info["match_comp"]["coords"]}
        if matched_cells != source_cells:
            continue
        if int(info["match_rot"]) != int(needed_rot):
            continue

        produced = _build_output(grid, info)
        if not np.array_equal(produced, expected):
            continue
        if not np.array_equal(main(grid), expected):
            continue

        red_comps = _component_dicts(expected == RED)
        if len(red_comps) != 1 or red_comps[0]["area"] != target_area:
            continue
        if np.any(expected[:, :divider_c] == RED):
            continue

        return grid

    raise RuntimeError("generate_input: failed to construct a valid puzzle instance")

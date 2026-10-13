from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ROTATION_COLORS = {0: 1, 1: 2, 2: 3, 3: 4, "mirror": 5}
INPUT_COLORS = (6, 7, 8, 9)
MAX_GENERATE_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 120
MAX_STAMP_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _component_masks(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _rot_cw(mask: np.ndarray, k: int) -> np.ndarray:
    return _tight_crop(np.rot90(mask, k=(-int(k)) % 4))


def _all_rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [_rot_cw(mask, k) for k in range(4)]


def _all_mirrors(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_crop(np.fliplr(mask))
    return [_rot_cw(base, k) for k in range(4)]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in _tight_crop(mask).astype(int))


def _transform_catalog(stamp: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    for k, mask in enumerate(_all_rotations(stamp)):
        variants.append({"family": int(k), "mask": mask, "name": f"rot{k}"})
    for k, mask in enumerate(_all_mirrors(stamp)):
        variants.append({"family": "mirror", "mask": mask, "name": f"mirror{k}"})
    return variants


def _find_corner_stamp(components: list[dict], shape: tuple[int, int]) -> dict:
    h, w = shape
    candidates = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_corner:
            candidates.append(comp)
    assert len(candidates) == 1, "expected exactly one corner reference stamp"
    return candidates[0]


def _enumerate_placements(mask: np.ndarray, stamp: np.ndarray) -> tuple[list[dict], int]:
    mask = np.asarray(mask, dtype=bool)
    stamp = np.asarray(stamp, dtype=bool)
    h, w = mask.shape
    piece_area = int(stamp.sum())
    assert piece_area > 0
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert len(cells) % piece_area == 0

    placements: list[dict] = []
    seen: set[tuple[object, frozenset[tuple[int, int]]]] = set()
    for variant in _transform_catalog(stamp):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(variant["mask"])]
        vh, vw = variant["mask"].shape
        for r0 in range(h - vh + 1):
            for c0 in range(w - vw + 1):
                placed = frozenset((r0 + rr, c0 + cc) for rr, cc in coords)
                if placed.issubset(cells):
                    key = (variant["name"], placed)
                    if key in seen:
                        continue
                    seen.add(key)
                    placements.append(
                        {
                            "family": variant["family"],
                            "name": variant["name"],
                            "cells": placed,
                        }
                    )
    return placements, piece_area


def _solve_cover(mask: np.ndarray, stamp: np.ndarray, max_solutions: int = 2) -> list[list[dict]]:
    placements, piece_area = _enumerate_placements(mask, stamp)
    all_cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    target_pieces = len(all_cells) // piece_area

    cell_to_ids: dict[tuple[int, int], list[int]] = {cell: [] for cell in all_cells}
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_ids[cell].append(idx)

    solutions: list[list[dict]] = []

    def search(uncovered: set[tuple[int, int]], chosen_ids: list[int]) -> None:
        if len(solutions) >= max_solutions:
            return
        if not uncovered:
            if len(chosen_ids) == target_pieces:
                solutions.append([placements[idx] for idx in chosen_ids])
            return

        remaining_slots = target_pieces - len(chosen_ids)
        if remaining_slots <= 0 or len(uncovered) != remaining_slots * piece_area:
            return

        best_ids: list[int] | None = None
        for cell in uncovered:
            fitting = [idx for idx in cell_to_ids[cell] if placements[idx]["cells"].issubset(uncovered)]
            if not fitting:
                return
            if best_ids is None or len(fitting) < len(best_ids):
                best_ids = fitting
                if len(best_ids) == 1:
                    break

        assert best_ids is not None
        best_ids = sorted(best_ids, key=lambda idx: (str(placements[idx]["family"]), sorted(placements[idx]["cells"])))
        for idx in best_ids:
            chosen_ids.append(idx)
            search(uncovered - set(placements[idx]["cells"]), chosen_ids)
            chosen_ids.pop()

    search(set(all_cells), [])
    return solutions


def _decompose_mask(mask: np.ndarray, stamp: np.ndarray) -> list[dict]:
    solutions = _solve_cover(mask, stamp, max_solutions=2)
    assert len(solutions) == 1, "expected a unique exact cover"
    solution = solutions[0]
    families = [placement["family"] for placement in solution]
    mirror_count = sum(family == "mirror" for family in families)
    rotation_families = {int(family) for family in families if family != "mirror"}
    assert mirror_count == 1, "expected exactly one mirrored impostor"
    assert rotation_families == {0, 1, 2, 3}, "expected all four rotation orientations"
    return solution


def _solve_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict], tuple[int, int, int, int], int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 1, "expected a single foreground color in the input"
    fg = colors[0]

    components = _component_masks(g, fg)
    stamp_component = _find_corner_stamp(components, g.shape)
    stamp_mask = _coords_to_mask(stamp_component["coords"])

    mask_grid = (g == fg).copy()
    mask_grid[stamp_component["coords"][:, 0], stamp_component["coords"][:, 1]] = False
    mask_coords = np.argwhere(mask_grid)
    assert mask_coords.size > 0, "expected a non-empty tiled mask away from the corner stamp"
    mask_bbox = _bbox_coords(mask_coords)
    mask_crop = mask_grid[mask_bbox[0] : mask_bbox[2] + 1, mask_bbox[1] : mask_bbox[3] + 1].copy()

    placements = _decompose_mask(mask_crop, stamp_mask)

    out = g.copy()
    for placement in placements:
        color = ROTATION_COLORS[placement["family"]]
        for rr, cc in placement["cells"]:
            out[mask_bbox[0] + rr, mask_bbox[1] + cc] = color

    return out, placements, mask_bbox, fg, stamp_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each translated copy of the corner stamp by its orientation and mark the lone mirrored copy in gray.

    Input structure:
    - One corner contains an isolated single-color stamp with no rotational or mirror self-symmetry.
    - All remaining foreground cells share that same color and form a mask that is exactly coverable by translated copies of the stamp.
    - The cover uses copies in the stamp's 0°, 90°, 180°, and 270° clockwise rotations, plus exactly one reflected copy.

    Rule:
    - Treat the corner stamp as the orientation reference and leave it unchanged.
    - Uniquely decompose the other foreground mask into non-overlapping translated copies of the same stamp, allowing rotations and reflections.
    - Recolor copies matching the reference, its clockwise 90°, 180°, and 270° rotations to blue, red, green, and yellow respectively, and recolor the one reflection-only impostor to gray.

    Key insight:
    - The foreground mask is not made of different pieces: it is one asymmetric stamp repeated under orientation changes, so exact-cover decomposition plus orientation classification reveals the single mirror fake.
    """
    out, _placements, _mask_bbox, _fg, _stamp_mask = _solve_grid(grid)
    return out


def _random_stamp() -> np.ndarray:
    for _ in range(MAX_STAMP_ATTEMPTS):
        h = random.randint(2, 4)
        w = random.randint(3, 4)
        area = random.randint(4, min(6, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_crop(mask)
        if mask.sum() != area or min(mask.shape) < 2:
            continue
        if int(mask.sum()) == mask.size:
            continue
        keys = {_shape_key(m) for m in _all_rotations(mask) + _all_mirrors(mask)}
        if len(keys) != 8:
            continue
        return mask
    raise AssertionError("failed to sample an asymmetric stamp")


def _touch_count(piece: frozenset[tuple[int, int]], occupied: set[tuple[int, int]]) -> int:
    total = 0
    for r, c in piece:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in occupied:
                total += 1
    return total


def _set_bbox(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _build_layout(piece_specs: list[dict], bbox_limit: int = 18) -> list[dict] | None:
    placed: list[dict] = []
    first_cells = frozenset(tuple(map(int, rc)) for rc in np.argwhere(piece_specs[0]["mask"]))
    occupied: set[tuple[int, int]] = set(first_cells)
    placed.append({"family": piece_specs[0]["family"], "cells": first_cells})

    for spec in piece_specs[1:]:
        rel_cells = [tuple(map(int, rc)) for rc in np.argwhere(spec["mask"])]
        candidates: dict[frozenset[tuple[int, int]], int] = {}

        for anchor_r, anchor_c in list(occupied):
            for dr, dc in DIR4:
                target = (anchor_r + dr, anchor_c + dc)
                if target in occupied:
                    continue
                for rr, cc in rel_cells:
                    shift_r = target[0] - rr
                    shift_c = target[1] - cc
                    piece = frozenset((shift_r + pr, shift_c + pc) for pr, pc in rel_cells)
                    if piece & occupied:
                        continue
                    contacts = _touch_count(piece, occupied)
                    if contacts == 0:
                        continue
                    merged = occupied | set(piece)
                    r0, c0, r1, c1 = _set_bbox(merged)
                    if r1 - r0 + 1 > bbox_limit or c1 - c0 + 1 > bbox_limit:
                        continue
                    candidates[piece] = max(candidates.get(piece, 0), contacts)

        if not candidates:
            return None

        candidate_items = list(candidates.items())
        random.shuffle(candidate_items)
        candidate_items.sort(key=lambda item: (abs(item[1] - 2), random.random()))
        chosen_piece = random.choice(candidate_items[: min(12, len(candidate_items))])[0]
        occupied |= set(chosen_piece)
        placed.append({"family": spec["family"], "cells": chosen_piece})

    return placed


def _crop_layout(layout: list[dict]) -> tuple[np.ndarray, list[dict]]:
    all_cells = set()
    for piece in layout:
        all_cells |= set(piece["cells"])
    r0, c0, r1, c1 = _set_bbox(all_cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted: list[dict] = []
    for piece in layout:
        cells = frozenset((r - r0, c - c0) for r, c in piece["cells"])
        for rr, cc in cells:
            mask[rr, cc] = True
        shifted.append({"family": piece["family"], "cells": cells})
    return mask, shifted


def _make_layout(stamp: np.ndarray) -> tuple[np.ndarray, list[dict]] | None:
    extras = random.choices([0, 1, 2], weights=[4, 3, 2], k=1)[0]
    mirror_k = random.randrange(4)
    specs = [{"family": k, "mask": _rot_cw(stamp, k)} for k in range(4)]
    specs.append({"family": "mirror", "mask": _all_mirrors(stamp)[mirror_k]})
    for _ in range(extras):
        k = random.randrange(4)
        specs.append({"family": k, "mask": _rot_cw(stamp, k)})
    random.shuffle(specs)

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        layout = _build_layout(specs)
        if layout is None:
            continue
        union_mask, shifted = _crop_layout(layout)
        if _count_components(union_mask) != 1:
            continue
        solutions = _solve_cover(union_mask, stamp, max_solutions=2)
        if len(solutions) != 1:
            continue
        families = [piece["family"] for piece in solutions[0]]
        if sum(family == "mirror" for family in families) != 1:
            continue
        if {int(family) for family in families if family != "mirror"} != {0, 1, 2, 3}:
            continue
        return union_mask, shifted
    return None


def _stamp_cells_at_corner(stamp: np.ndarray, corner: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    sh, sw = stamp.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - sw
    elif corner == "bl":
        r0, c0 = h - sh, 0
    else:
        r0, c0 = h - sh, w - sw
    return {(r0 + r, c0 + c) for r, c in np.argwhere(stamp)}


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in list(cells):
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.add((nr, nc))
    return out


def _place_patch(
    stamp: np.ndarray,
    union_mask: np.ndarray,
    corner: str,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    sh, sw = stamp.shape
    ph, pw = union_mask.shape
    min_h = max(sh + 4, ph + 3, max(sh, ph) + 3)
    min_w = max(sw + 4, pw + 3, max(sw, pw) + 3)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(120):
        h = random.randint(min_h, min(30, max(min_h, sh + ph + 8)))
        w = random.randint(min_w, min(30, max(min_w, sw + pw + 8)))
        stamp_cells = _stamp_cells_at_corner(stamp, corner, (h, w))
        blocked = _expanded(stamp_cells, (h, w))

        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                patch_cells = {(r0 + r, c0 + c) for r, c in np.argwhere(union_mask)}
                if patch_cells & blocked:
                    continue
                candidates.append((r0, c0))
        if candidates:
            return (h, w), next(iter(candidates)), random.choice(candidates)
    return None


def _render_instance(
    stamp: np.ndarray,
    union_mask: np.ndarray,
    placements: list[dict],
    fg: int,
    corner: str,
) -> tuple[np.ndarray, np.ndarray]:
    placed = _place_patch(stamp, union_mask, corner)
    assert placed is not None
    (h, w), _unused, (patch_r0, patch_c0) = placed
    grid = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    stamp_cells = _stamp_cells_at_corner(stamp, corner, (h, w))
    for r, c in stamp_cells:
        grid[r, c] = fg
        out[r, c] = fg

    for r, c in np.argwhere(union_mask):
        grid[patch_r0 + r, patch_c0 + c] = fg

    out[grid == fg] = fg
    for placement in placements:
        color = ROTATION_COLORS[placement["family"]]
        for rr, cc in placement["cells"]:
            out[patch_r0 + rr, patch_c0 + cc] = color

    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        stamp = _random_stamp()
        built = _make_layout(stamp)
        if built is None:
            continue
        union_mask, placements = built
        fg = random.choice(INPUT_COLORS)
        corner = random.choice(["tl", "tr", "bl", "br"])
        grid, expected = _render_instance(stamp, union_mask, placements, fg, corner)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue
        return grid
    raise AssertionError("failed to generate a valid puzzle instance")

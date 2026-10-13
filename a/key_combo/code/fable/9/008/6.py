from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in FOUR_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components_from_mask(grid == color):
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    shifted = coords - coords.min(axis=0)
    order = np.lexsort((shifted[:, 1], shifted[:, 0]))
    return shifted[order]


def _shape_key(coords: np.ndarray) -> tuple[Coord, ...]:
    norm = _normalize_coords(coords)
    return tuple((int(r), int(c)) for r, c in norm)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    norm = _normalize_coords(coords)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[norm[:, 0], norm[:, 1]] = True
    return mask


def _internal_bg_components(crop: np.ndarray, bg: int) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for comp in _components_from_mask(crop == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == crop.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == crop.shape[1] - 1):
            continue
        holes.append(comp)
    return holes


def _valid_piece_placements(hole_mask: np.ndarray, piece_coords: np.ndarray) -> list[frozenset[Coord]]:
    piece = _normalize_coords(piece_coords)
    ph = int(piece[:, 0].max()) + 1
    pw = int(piece[:, 1].max()) + 1
    hh, hw = hole_mask.shape
    placements: set[frozenset[Coord]] = set()
    if ph > hh or pw > hw:
        return []
    for dr in range(hh - ph + 1):
        for dc in range(hw - pw + 1):
            shifted = piece + np.array([dr, dc], dtype=int)
            if np.all(hole_mask[shifted[:, 0], shifted[:, 1]]):
                placements.add(frozenset((int(r), int(c)) for r, c in shifted))
    return sorted(placements, key=lambda cells: (len(cells), tuple(sorted(cells))))


def _enumerate_pair_solutions(
    hole_mask: np.ndarray, fragments: list[dict]
) -> tuple[list[list[frozenset[Coord]]], list[tuple[int, int, frozenset[Coord], frozenset[Coord]]], list[int]]:
    hole_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(hole_mask))
    hole_area = len(hole_cells)
    placements = [_valid_piece_placements(hole_mask, frag["coords"]) for frag in fragments]
    single_matches = [idx for idx, ps in enumerate(placements) if hole_cells in ps]

    solutions: list[tuple[int, int, frozenset[Coord], frozenset[Coord]]] = []
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            if fragments[i]["area"] + fragments[j]["area"] != hole_area:
                continue
            for first in placements[i]:
                for second in placements[j]:
                    if first.isdisjoint(second) and first | second == hole_cells:
                        solutions.append((i, j, first, second))
    return placements, solutions, single_matches


def _subject_and_hole(grid: np.ndarray, bg: int) -> tuple[dict, np.ndarray]:
    candidates: list[tuple[dict, np.ndarray]] = []
    for comp in _find_components(grid, bg):
        holes = _internal_bg_components(_crop_bbox(grid, comp["bbox"]), bg)
        if len(holes) == 1:
            candidates.append((comp, holes[0]))
    assert len(candidates) == 1, "expected exactly one monochrome subject with one connected hole"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the unique hole-bearing subject with the only fragment pair that translation-fits it.

    Input structure:
    - A black background contains one large monochrome subject whose bounding box encloses one
      connected black hole.
    - Several smaller monochrome fragments lie outside the subject, each in a distinct color.
    - Exactly one pair of fragments can, by translation only and without overlap, cover the hole.

    Rule:
    - Identify the unique subject and crop its bounding box.
    - Search all fragment pairs and all translation-only placements inside the hole's own
      coordinate frame.
    - Keep the only non-overlapping pair placement whose union exactly matches the hole, and fill
      each covered hole cell with that fragment's color.

    Key insight:
    - No single fragment matches the hole; the answer is determined only at the level of a
      complementary two-piece assembly, while every distractor fails that whole-hole test.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    subject, hole_local = _subject_and_hole(g, bg)
    fragments = []
    for comp in _find_components(g, bg):
        if np.array_equal(comp["coords"], subject["coords"]):
            continue
        fragments.append(
            {
                "color": comp["color"],
                "coords": _normalize_coords(comp["coords"]),
                "area": comp["area"],
            }
        )

    hole_bbox = _bbox_from_coords(hole_local)
    hole_mask = _coords_to_mask(hole_local)
    _, solutions, single_matches = _enumerate_pair_solutions(hole_mask, fragments)
    assert not single_matches, "no single fragment may solve the hole"
    assert len(solutions) == 1, "expected a unique complementary fragment pair"

    out = _crop_bbox(g, subject["bbox"])
    _, _, first_cells, second_cells = solutions[0]
    first_idx, second_idx = solutions[0][0], solutions[0][1]
    hr0, hc0, _, _ = hole_bbox
    for r, c in first_cells:
        out[hr0 + r, hc0 + c] = fragments[first_idx]["color"]
    for r, c in second_cells:
        out[hr0 + r, hc0 + c] = fragments[second_idx]["color"]
    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    return len(_components_from_mask(mask.astype(bool))) == 1


def _mask_has_internal_hole(mask: np.ndarray) -> bool:
    return bool(_internal_bg_components(mask.astype(int), 0))


def _random_connected_mask(area: int, max_side: int = 5) -> np.ndarray:
    for _ in range(250):
        h = random.randint(1, min(max_side, area))
        w = random.randint(1, min(max_side, area))
        if h * w < area:
            continue
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in FOUR_DIRS:
            nr, nc = start[0] + dr, start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in FOUR_DIRS:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        mask = _coords_to_mask(coords)
        if mask.shape[0] > max_side or mask.shape[1] > max_side:
            continue
        if not _mask_is_connected(mask) or _mask_has_internal_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to generate connected mask")


def _place_two_true_pieces() -> tuple[np.ndarray, list[np.ndarray]]:
    for _ in range(300):
        area1 = random.randint(2, 5)
        area2 = random.randint(2, 6)
        piece1 = _random_connected_mask(area1, max_side=4)
        piece2 = _random_connected_mask(area2, max_side=4)
        if _shape_key(np.argwhere(piece1)) == _shape_key(np.argwhere(piece2)):
            continue

        box_h = random.randint(max(piece1.shape[0], piece2.shape[0]), 6)
        box_w = random.randint(max(piece1.shape[1], piece2.shape[1]), 6)
        for _ in range(120):
            placed = np.zeros((box_h, box_w), dtype=int)
            r1 = random.randint(0, box_h - piece1.shape[0])
            c1 = random.randint(0, box_w - piece1.shape[1])
            if np.any(placed[r1 : r1 + piece1.shape[0], c1 : c1 + piece1.shape[1]] & piece1):
                continue
            placed[r1 : r1 + piece1.shape[0], c1 : c1 + piece1.shape[1]] |= piece1.astype(int)

            mask2_region = np.zeros_like(placed, dtype=bool)
            for r2 in range(box_h - piece2.shape[0] + 1):
                for c2 in range(box_w - piece2.shape[1] + 1):
                    view = placed[r2 : r2 + piece2.shape[0], c2 : c2 + piece2.shape[1]]
                    if np.any(view & piece2):
                        continue
                    mask2_region[r2, c2] = True
            candidates = np.argwhere(mask2_region)
            random.shuffle(candidates.tolist())
            for idx in np.random.permutation(len(candidates)):
                r2, c2 = map(int, candidates[idx])
                union = placed.copy()
                union[r2 : r2 + piece2.shape[0], c2 : c2 + piece2.shape[1]] |= piece2.astype(int)
                if not _mask_is_connected(union.astype(bool)):
                    continue
                union_coords = np.argwhere(union)
                hole_mask = _coords_to_mask(union_coords)
                if not 6 <= int(hole_mask.sum()) <= 12:
                    continue
                if hole_mask.shape[0] < 3 or hole_mask.shape[1] < 3:
                    continue
                frag_defs = [
                    {"coords": np.argwhere(piece1), "area": int(piece1.sum()), "color": 1},
                    {"coords": np.argwhere(piece2), "area": int(piece2.sum()), "color": 2},
                ]
                _, solutions, singles = _enumerate_pair_solutions(hole_mask, frag_defs)
                if singles or len(solutions) != 1:
                    continue
                return hole_mask, [piece1.astype(bool), piece2.astype(bool)]
    raise RuntimeError("failed to construct a unique two-piece hole")


def _bbox_separated(a: BBox, b: BBox, pad: int) -> bool:
    return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]


def _place_patch(grid: np.ndarray, patch: np.ndarray, boxes: list[BBox], bg: int, pad: int) -> BBox | None:
    h, w = patch.shape
    positions = [(r, c) for r in range(grid.shape[0] - h + 1) for c in range(grid.shape[1] - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        bbox = (r, c, r + h - 1, c + w - 1)
        if any(not _bbox_separated(bbox, other, pad) for other in boxes):
            continue
        view = grid[r : r + h, c : c + w]
        if np.any((patch != bg) & (view != bg)):
            continue
        view[patch != bg] = patch[patch != bg]
        return bbox
    return None


def _build_subject(hole_mask: np.ndarray, subject_color: int) -> tuple[np.ndarray, BBox]:
    mt, mb = random.randint(1, 3), random.randint(1, 3)
    ml, mr = random.randint(1, 3), random.randint(1, 3)
    out = np.full((hole_mask.shape[0] + mt + mb, hole_mask.shape[1] + ml + mr), subject_color, dtype=int)
    hole_bbox = (mt, ml, mt + hole_mask.shape[0] - 1, ml + hole_mask.shape[1] - 1)
    hr0, hc0, hr1, hc1 = hole_bbox
    hole_view = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
    hole_view[hole_mask] = BG
    return out, hole_bbox


def _expected_output(subject_patch: np.ndarray, hole_bbox: BBox, hole_mask: np.ndarray, fragments: list[dict]) -> np.ndarray:
    _, solutions, singles = _enumerate_pair_solutions(hole_mask, fragments)
    assert not singles and len(solutions) == 1 and solutions[0][0:2] == (0, 1)
    out = subject_patch.copy()
    hr0, hc0, _, _ = hole_bbox
    i, j, first_cells, second_cells = solutions[0]
    for r, c in first_cells:
        out[hr0 + r, hc0 + c] = fragments[i]["color"]
    for r, c in second_cells:
        out[hr0 + r, hc0 + c] = fragments[j]["color"]
    return out


def _fragment_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _generate_distractors(hole_mask: np.ndarray, true_fragments: list[dict]) -> list[dict]:
    used_keys = {_shape_key(frag["coords"]) for frag in true_fragments}
    hole_area = int(hole_mask.sum())
    distractors: list[dict] = []

    for _ in range(80):
        area = random.randint(1, hole_area - 1)
        mask = _random_connected_mask(area, max_side=5)
        key = _shape_key(np.argwhere(mask))
        if key in used_keys:
            continue
        if _valid_piece_placements(hole_mask, np.argwhere(mask)):
            distractors.append({"coords": np.argwhere(mask), "area": area, "key": key})
            used_keys.add(key)
            break
    if not distractors:
        raise RuntimeError("failed to create subset-fit distractor")

    target_area = hole_area - distractors[0]["area"]
    for _ in range(120):
        area = target_area if target_area >= 1 else random.randint(1, hole_area - 1)
        mask = _random_connected_mask(area, max_side=5)
        key = _shape_key(np.argwhere(mask))
        if key in used_keys:
            continue
        distractors.append({"coords": np.argwhere(mask), "area": area, "key": key})
        used_keys.add(key)
        break
    if len(distractors) < 2:
        raise RuntimeError("failed to create area-sum distractor")

    while len(distractors) < random.randint(2, 4):
        area = random.randint(1, hole_area - 1)
        mask = _random_connected_mask(area, max_side=5)
        key = _shape_key(np.argwhere(mask))
        if key in used_keys:
            continue
        distractors.append({"coords": np.argwhere(mask), "area": area, "key": key})
        used_keys.add(key)

    return distractors


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> None:
    bg = _get_bg(grid)
    subject, hole_local = _subject_and_hole(grid, bg)
    hole_mask = _coords_to_mask(hole_local)
    fragments = []
    for comp in _find_components(grid, bg):
        if np.array_equal(comp["coords"], subject["coords"]):
            continue
        fragments.append(
            {
                "color": comp["color"],
                "coords": _normalize_coords(comp["coords"]),
                "area": comp["area"],
            }
        )
    placements, solutions, singles = _enumerate_pair_solutions(hole_mask, fragments)
    assert not singles
    assert len(solutions) == 1
    assert solutions[0][0:2] == (0, 1)
    distractor_range = range(2, len(fragments))
    assert any(len(placements[idx]) > 0 for idx in distractor_range)
    hole_area = int(hole_mask.sum())
    assert any(
        fragments[i]["area"] + fragments[j]["area"] == hole_area
        and (i, j) != (0, 1)
        for i in range(len(fragments))
        for j in range(i + 1, len(fragments))
    )
    assert np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        hole_mask, true_masks = _place_two_true_pieces()
        true_fragments = [
            {"coords": np.argwhere(true_masks[0]), "area": int(true_masks[0].sum())},
            {"coords": np.argwhere(true_masks[1]), "area": int(true_masks[1].sum())},
        ]
        distractors = _generate_distractors(hole_mask, true_fragments)
        colors = random.sample(range(1, 10), k=1 + len(true_fragments) + len(distractors))
        subject_color = colors[0]
        subject_patch, hole_bbox = _build_subject(hole_mask, subject_color)
        for frag, color in zip(true_fragments, colors[1:3]):
            frag["color"] = color
        all_fragments = true_fragments + [
            {"color": color, "coords": d["coords"], "area": d["area"]}
            for color, d in zip(colors[3:], distractors)
        ]
        _, solutions, singles = _enumerate_pair_solutions(hole_mask, all_fragments)
        if singles or len(solutions) != 1 or solutions[0][0:2] != (0, 1):
            continue
        expected = _expected_output(subject_patch, hole_bbox, hole_mask, all_fragments)

        grid = np.zeros((random.randint(20, 28), random.randint(20, 28)), dtype=int)
        boxes: list[BBox] = []
        subject_bbox = _place_patch(grid, subject_patch, boxes, BG, pad=1)
        if subject_bbox is None:
            continue
        boxes.append(subject_bbox)

        ok = True
        for frag in all_fragments:
            patch = _fragment_patch(_coords_to_mask(frag["coords"]), frag["color"])
            bbox = _place_patch(grid, patch, boxes, BG, pad=1)
            if bbox is None:
                ok = False
                break
            boxes.append(bbox)
        if not ok:
            continue

        try:
            _semantic_checks(grid, expected)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")

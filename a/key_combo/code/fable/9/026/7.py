from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _tight_component_from_grid(
    grid: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask, r0, c0 = _tight_mask(coords)
    color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return mask, patch, r0, c0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for rot in range(4):
        base = np.rot90(mask, rot)
        for cand in (base, np.fliplr(base)):
            key = (cand.tobytes(), tuple(map(int, cand.shape)))
            if key in seen:
                continue
            seen.add(key)
            variants.append(cand)
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    return min((cand.tobytes(), tuple(map(int, cand.shape))) for cand in _dihedral_variants(mask))


def _variant_count(mask: np.ndarray) -> int:
    return len(_dihedral_variants(mask))


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    if mask is None:
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        return
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _render_output(
    grid: np.ndarray, bg: int, holes: list[dict], pieces: list[dict], assignment: dict[int, int]
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for piece in pieces:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = bg
    for hole_idx, piece_idx in assignment.items():
        color = int(pieces[piece_idx]["color"])
        hole_coords = holes[hole_idx]["coords_abs"]
        out[hole_coords[:, 0], hole_coords[:, 1]] = color
    return out


def _solve_outputs(grid: np.ndarray) -> tuple[list[np.ndarray], dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _color_components(g, bg)
    assert comps, "expected a frame and outside pieces"

    frame = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = frame["bbox"]
    frame_color = int(frame["color"])

    frame_mask, _frame_patch, _r0, _c0 = _tight_component_from_grid(g, frame["coords"])
    assert np.all(_frame_patch[frame_mask] == frame_color), "frame must be monochrome"

    sub = _crop_bbox(g, frame["bbox"])
    bg_mask = sub == bg
    holes: list[dict] = []
    for coords in _components(bg_mask):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == sub.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == sub.shape[1] - 1)
        ):
            continue
        hole_mask, hr0, hc0 = _tight_mask(coords)
        coords_abs = coords + np.array([fr0, fc0], dtype=int)
        holes.append(
            {
                "coords_rel": coords,
                "coords_abs": coords_abs,
                "mask": hole_mask,
                "bbox_abs": (fr0 + hr0, fc0 + hc0, fr0 + hr0 + hole_mask.shape[0] - 1, fc0 + hc0 + hole_mask.shape[1] - 1),
                "area": int(coords.shape[0]),
                "shape_key": _canonical_key(hole_mask),
            }
        )
    holes.sort(key=lambda hole: hole["bbox_abs"])
    assert holes, "expected enclosed black holes"

    pieces = [comp for comp in comps if comp is not frame]
    pieces.sort(key=lambda piece: piece["bbox"])
    assert len(pieces) == len(holes), "piece count must equal hole count"

    for piece in pieces:
        mask, patch, pr0, pc0 = _tight_component_from_grid(g, piece["coords"])
        piece["mask"] = mask
        piece["patch"] = patch
        piece["top_left"] = (pr0, pc0)
        piece["shape_key"] = _canonical_key(mask)
        piece["area"] = int(mask.sum())

    hole_candidates: list[list[int]] = []
    piece_candidate_counts = [0 for _ in pieces]
    for hole in holes:
        cand: list[int] = []
        for piece_idx, piece in enumerate(pieces):
            if piece["area"] != hole["area"]:
                continue
            if piece["shape_key"] != hole["shape_key"]:
                continue
            cand.append(piece_idx)
            piece_candidate_counts[piece_idx] += 1
        assert cand, "every hole must have at least one matching piece"
        hole_candidates.append(cand)

    unique_outputs: dict[tuple[bytes, tuple[int, int]], np.ndarray] = {}
    identity_matchings = 0

    def backtrack(assignment: dict[int, int], used_pieces: set[int]) -> None:
        nonlocal identity_matchings
        if len(assignment) == len(holes):
            identity_matchings += 1
            rendered = _render_output(g, bg, holes, pieces, assignment)
            key = (rendered.tobytes(), tuple(map(int, rendered.shape)))
            unique_outputs.setdefault(key, rendered)
            return

        remaining_holes = [idx for idx in range(len(holes)) if idx not in assignment]
        hole_idx = min(
            remaining_holes,
            key=lambda idx: (
                sum(1 for piece_idx in hole_candidates[idx] if piece_idx not in used_pieces),
                holes[idx]["bbox_abs"],
            ),
        )
        available = [piece_idx for piece_idx in hole_candidates[hole_idx] if piece_idx not in used_pieces]
        assert available, "partial assignment became impossible"
        for piece_idx in available:
            assignment[hole_idx] = piece_idx
            used_pieces.add(piece_idx)
            backtrack(assignment, used_pieces)
            used_pieces.remove(piece_idx)
            del assignment[hole_idx]

    backtrack({}, set())
    assert unique_outputs, "expected at least one perfect matching"

    meta = {
        "bg": bg,
        "frame": frame,
        "holes": holes,
        "pieces": pieces,
        "hole_candidates": hole_candidates,
        "piece_candidate_counts": piece_candidate_counts,
        "identity_matchings": identity_matchings,
        "unique_output_count": len(unique_outputs),
    }
    return list(unique_outputs.values()), meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place the scattered pieces into the frame's black holes using the only globally consistent assignment.

    Input structure:
    - The largest monochrome component is a frame that encloses several black hole regions.
    - Outside the frame sit the same number of monochrome puzzle pieces, separated from the frame and from each other.
    - A piece may match multiple holes under rotation or reflection, but all holes must be filled exactly once.

    Rule:
    - Extract the enclosed black hole components from inside the frame.
    - Compare each outside piece with each hole by shape up to rotation and mirror.
    - Among all perfect one-piece-per-hole assignments, keep the unique rendered result, fill each hole with its piece color, and erase the outside pieces.

    Key insight:
    - Local shape compatibility alone is not enough; the final picture is forced only after considering the whole matching at once.
    """
    outputs, meta = _solve_outputs(grid)
    assert meta["unique_output_count"] == 1, "expected a unique output grid"
    return outputs[0]


# --------------- generator ---------------

_GEN_BG = 0
_GEN_FRAME_DIM = (9, 15)
_GEN_CANVAS_DIM = (20, 30)
_GEN_SLOT_COUNT = (3, 5)
_GEN_REPEAT_SIZE_CHOICES = (2, 3)
_GEN_SHAPE_AREA = (2, 6)
_GEN_OUTER_ATTEMPTS = 400
_GEN_SHAPE_ATTEMPTS = 120
_GEN_LAYOUT_ATTEMPTS = 120
_GEN_PIECE_PLACE_ATTEMPTS = 400


def _random_shape(area: int, max_h: int = 4, max_w: int = 4) -> np.ndarray | None:
    for _ in range(_GEN_SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
        while len(cells) < area and frontier:
            cand = random.choice(tuple(frontier))
            frontier.remove(cand)
            cells.add(cand)
            r, c = cand
            for nxt in _neighbors4(r, c):
                if nxt not in cells:
                    frontier.add(nxt)
        if len(cells) != area:
            continue
        arr = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox(arr)
        arr -= np.array([r0, c0], dtype=int)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h > max_h or w > max_w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[arr[:, 0], arr[:, 1]] = True
        return mask
    return None


def _choose_shapes(n_slots: int) -> tuple[list[np.ndarray], int] | None:
    repeat_size = random.choice([k for k in _GEN_REPEAT_SIZE_CHOICES if k <= n_slots - 1])
    repeat_area = random.randint(*_GEN_SHAPE_AREA)

    repeated = None
    for _ in range(_GEN_SHAPE_ATTEMPTS):
        cand = _random_shape(repeat_area)
        if cand is not None and _variant_count(cand) > 1:
            repeated = cand
            break
    if repeated is None:
        return None

    shapes = [repeated for _ in range(repeat_size)]
    keys = {_canonical_key(repeated)}

    same_area_other = None
    for _ in range(_GEN_SHAPE_ATTEMPTS):
        cand = _random_shape(repeat_area)
        if cand is None:
            continue
        key = _canonical_key(cand)
        if key not in keys:
            same_area_other = cand
            keys.add(key)
            break
    if same_area_other is None:
        return None
    shapes.append(same_area_other)

    while len(shapes) < n_slots:
        area = random.randint(*_GEN_SHAPE_AREA)
        cand = _random_shape(area)
        if cand is None:
            continue
        key = _canonical_key(cand)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(cand)

    random.shuffle(shapes)
    return shapes, repeat_size


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _place_holes(layout_h: int, layout_w: int, shapes: list[np.ndarray]) -> list[dict] | None:
    forbidden = np.zeros((layout_h, layout_w), dtype=bool)
    placed: list[dict] = []
    order = sorted(range(len(shapes)), key=lambda idx: -int(shapes[idx].sum()))
    for idx in order:
        shape = shapes[idx]
        variants = _dihedral_variants(shape)
        success = False
        for _ in range(_GEN_LAYOUT_ATTEMPTS):
            oriented = random.choice(variants)
            sh, sw = oriented.shape
            if sh > layout_h or sw > layout_w:
                continue
            r0 = random.randint(0, layout_h - sh)
            c0 = random.randint(0, layout_w - sw)
            region = forbidden[r0 : r0 + sh, c0 : c0 + sw]
            if np.any(region & oriented):
                continue
            temp = np.zeros((layout_h, layout_w), dtype=bool)
            temp[r0 : r0 + sh, c0 : c0 + sw] = oriented
            forbidden |= _dilate8(temp)
            coords = np.argwhere(temp)
            placed.append(
                {
                    "shape_index": idx,
                    "mask": oriented,
                    "coords": coords,
                    "top_left": (r0, c0),
                    "area": int(oriented.sum()),
                }
            )
            success = True
            break
        if not success:
            return None
    placed.sort(key=lambda slot: slot["top_left"])
    return placed


def _can_place_patch(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    block = occupied[rr0:rr1, cc0:cc1]
    view = np.zeros_like(block, dtype=bool)
    view[r0 - rr0 : r0 - rr0 + h, c0 - cc0 : c0 - cc0 + w] = mask
    return not np.any(block & _dilate8(view))


def _place_pieces_outside(
    canvas: np.ndarray, occupied: np.ndarray, frame_box: BBox, pieces: list[dict]
) -> bool:
    fr0, fc0, fr1, fc1 = frame_box
    for piece in sorted(pieces, key=lambda item: -int(item["mask"].sum())):
        variants = _dihedral_variants(piece["mask"])
        choices = variants[:]
        random.shuffle(choices)
        placed = False
        for oriented in choices:
            h, w = oriented.shape
            for _ in range(_GEN_PIECE_PLACE_ATTEMPTS):
                r0 = random.randint(0, canvas.shape[0] - h)
                c0 = random.randint(0, canvas.shape[1] - w)
                if not (r0 + h - 1 < fr0 or r0 > fr1 or c0 + w - 1 < fc0 or c0 > fc1):
                    continue
                if not _can_place_patch(occupied, oriented, r0, c0):
                    continue
                patch = np.zeros((h, w), dtype=int)
                patch[oriented] = int(piece["color"])
                _paste(canvas, patch, r0, c0, oriented)
                occupied[r0 : r0 + h, c0 : c0 + w] |= oriented
                piece["placed_mask"] = oriented
                piece["placed_top_left"] = (r0, c0)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_slots = random.randint(*_GEN_SLOT_COUNT)
    chosen = _choose_shapes(n_slots)
    if chosen is None:
        return None
    shapes, repeat_size = chosen

    repeated_key = Counter(_canonical_key(shape) for shape in shapes).most_common(1)[0][0]
    repeat_indexes = [idx for idx, shape in enumerate(shapes) if _canonical_key(shape) == repeated_key]
    if len(repeat_indexes) != repeat_size:
        return None

    layout_h = random.randint(*_GEN_FRAME_DIM)
    layout_w = random.randint(*_GEN_FRAME_DIM)
    slots = _place_holes(layout_h, layout_w, shapes)
    if slots is None:
        return None

    frame_h = layout_h + 2
    frame_w = layout_w + 2
    canvas_h = random.randint(max(frame_h + 6, _GEN_CANVAS_DIM[0]), _GEN_CANVAS_DIM[1])
    canvas_w = random.randint(max(frame_w + 6, _GEN_CANVAS_DIM[0]), _GEN_CANVAS_DIM[1])
    if canvas_h > 30 or canvas_w > 30:
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    repeated_color = palette.pop()

    unique_shape_indexes = sorted(set(range(n_slots)) - set(repeat_indexes))
    if len(palette) < len(unique_shape_indexes):
        return None
    color_by_shape_index = {idx: palette[i] for i, idx in enumerate(unique_shape_indexes)}
    for idx in repeat_indexes:
        color_by_shape_index[idx] = repeated_color

    frame_top = random.randint(1, canvas_h - frame_h - 1)
    frame_left = random.randint(1, canvas_w - frame_w - 1)
    frame_box = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    final_grid = np.zeros((canvas_h, canvas_w), dtype=int)
    frame_patch = np.full((frame_h, frame_w), frame_color, dtype=int)
    for slot in slots:
        r0, c0 = slot["top_left"]
        frame_patch[1 + r0 : 1 + r0 + slot["mask"].shape[0], 1 + c0 : 1 + c0 + slot["mask"].shape[1]][slot["mask"]] = 0
    _paste(final_grid, frame_patch, frame_top, frame_left)

    pieces: list[dict] = []
    for slot in slots:
        idx = int(slot["shape_index"])
        color = int(color_by_shape_index[idx])
        coords_abs = slot["coords"] + np.array([frame_top + 1, frame_left + 1], dtype=int)
        final_grid[coords_abs[:, 0], coords_abs[:, 1]] = color
        pieces.append({"mask": shapes[idx], "color": color, "shape_index": idx})

    input_grid = np.array(final_grid, copy=True)
    for slot in slots:
        coords_abs = slot["coords"] + np.array([frame_top + 1, frame_left + 1], dtype=int)
        input_grid[coords_abs[:, 0], coords_abs[:, 1]] = 0

    occupied = input_grid != 0
    if not _place_pieces_outside(input_grid, occupied, frame_box, pieces):
        return None

    try:
        outputs, meta = _solve_outputs(input_grid)
    except AssertionError:
        return None
    if len(outputs) != 1:
        return None
    if not np.array_equal(outputs[0], final_grid):
        return None
    if meta["identity_matchings"] < 2:
        return None
    if max(meta["piece_candidate_counts"]) < 2:
        return None

    hole_shapes = [hole["shape_key"] for hole in meta["holes"]]
    if Counter(hole_shapes).most_common(1)[0][1] < 2:
        return None

    return input_grid, final_grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_OUTER_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        input_grid, final_grid = built
        produced = main(input_grid)
        if not np.array_equal(produced, final_grid):
            continue
        return input_grid
    raise RuntimeError("failed to generate a valid puzzle after repeated attempts")

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PIECE_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _color_components(grid: np.ndarray, allowed_colors: set[int]) -> list[dict]:
    parts = []
    for color in sorted(allowed_colors):
        mask = grid == int(color)
        for coords in _components(mask):
            r0, c0, r1, c1 = _bbox_coords(coords)
            piece_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            piece_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            parts.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": piece_mask,
                }
            )
    return parts


def _apply_orientation(mask: np.ndarray, rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot))
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _orientations(mask: np.ndarray) -> list[tuple[tuple[int, bool], np.ndarray]]:
    out = []
    seen = set()
    for flip in (False, True):
        for rot in range(4):
            transformed = _apply_orientation(mask, rot, flip)
            key = transformed.tobytes() + str(transformed.shape).encode()
            if key in seen:
                continue
            seen.add(key)
            out.append(((rot, flip), transformed))
    return out


def _canonical_key(mask: np.ndarray) -> bytes:
    keys = []
    for _, oriented in _orientations(mask):
        keys.append(oriented.tobytes() + str(oriented.shape).encode())
    return min(keys)


def _mask_from_cells(cells: frozenset[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _normalized_cells(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _generate_shape_pool() -> list[np.ndarray]:
    by_size: dict[int, set[frozenset[tuple[int, int]]]] = {1: {frozenset({(0, 0)})}}
    for size in range(1, 6):
        next_set: set[frozenset[tuple[int, int]]] = set()
        for cells in by_size[size]:
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    grown = set(cells)
                    grown.add((nr, nc))
                    next_set.add(_normalized_cells(grown))
        by_size[size + 1] = next_set

    shapes = []
    for size in range(3, 7):
        for cells in by_size[size]:
            mask = _mask_from_cells(cells)
            h, w = mask.shape
            if h % 2 != 0:
                continue
            top = bool(mask[: h // 2].any())
            bottom = bool(mask[h // 2 :].any())
            if not (top and bottom):
                continue
            variants = _orientations(mask)
            if len(variants) < 8:
                continue
            lower_rows = np.arange(h)[:, None] >= (h // 2)
            lower_count = int((mask & lower_rows).sum())
            if lower_count == 0 or lower_count == int(mask.sum()):
                continue
            shapes.append(mask)
    return shapes


SHAPE_POOL = _generate_shape_pool()


def _find_frame_component(grid: np.ndarray) -> dict:
    grey_comps = _color_components(grid, {GREY})
    if not grey_comps:
        raise ValueError("expected a grey frame")
    return max(grey_comps, key=lambda comp: int(comp["mask"].sum()))


def _extract_slots(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    hole_mask = sub == BLACK
    slots = []
    for coords in _components(hole_mask):
        sr0, sc0, sr1, sc1 = _bbox_coords(coords)
        if sr0 == 0 or sc0 == 0 or sr1 == sub.shape[0] - 1 or sc1 == sub.shape[1] - 1:
            continue
        mask = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
        mask[coords[:, 0] - sr0, coords[:, 1] - sc0] = True
        slots.append(
            {
                "bbox": (r0 + sr0, c0 + sc0, r0 + sr1, c0 + sc1),
                "mask": mask,
            }
        )
    slots.sort(key=lambda slot: (slot["bbox"][0], slot["bbox"][1]))
    return slots


def _match_pieces_to_slots(pieces: list[dict], slots: list[dict]) -> list[tuple[dict, dict, np.ndarray]]:
    slot_candidates = []
    for piece in pieces:
        matches = []
        for slot_idx, slot in enumerate(slots):
            for _, oriented in _orientations(piece["mask"]):
                if oriented.shape == slot["mask"].shape and np.array_equal(oriented, slot["mask"]):
                    matches.append((slot_idx, oriented))
                    break
        if not matches:
            raise ValueError("piece does not match any slot")
        slot_candidates.append(matches)

    order = sorted(range(len(pieces)), key=lambda idx: len(slot_candidates[idx]))
    best: list[tuple[int, int, np.ndarray]] = []
    solutions = 0

    def dfs(pos: int, used_slots: set[int], acc: list[tuple[int, int, np.ndarray]]):
        nonlocal best, solutions
        if solutions > 1:
            return
        if pos == len(order):
            best = list(acc)
            solutions += 1
            return
        piece_idx = order[pos]
        for slot_idx, oriented in slot_candidates[piece_idx]:
            if slot_idx in used_slots:
                continue
            used_slots.add(slot_idx)
            acc.append((piece_idx, slot_idx, oriented))
            dfs(pos + 1, used_slots, acc)
            acc.pop()
            used_slots.remove(slot_idx)

    dfs(0, set(), [])
    if solutions != 1:
        raise ValueError("matching must be unique")

    placements = []
    for piece_idx, slot_idx, oriented in best:
        placements.append((pieces[piece_idx], slots[slot_idx], oriented))
    placements.sort(key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][1]))
    return placements


def _render_output(grid: np.ndarray) -> np.ndarray:
    frame = _find_frame_component(grid)
    slots = _extract_slots(grid, frame["bbox"])
    piece_colors = set(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GREY))
    pieces = _color_components(grid, piece_colors)
    placements = _match_pieces_to_slots(pieces, slots)

    out = np.zeros_like(grid)
    out[frame["coords"][:, 0], frame["coords"][:, 1]] = GREY
    for piece, slot, oriented in placements:
        r0, c0, r1, c1 = slot["bbox"]
        color = int(piece["color"])
        patch = np.full(oriented.shape, color, dtype=int)
        rr = np.arange(r0, r1 + 1)[:, None]
        dest = out[r0 : r1 + 1, c0 : c1 + 1]
        dest[oriented] = patch[oriented]
        lower = rr >= (r0 + (r1 - r0 + 1) // 2)
        lower_mask = oriented & lower
        dest[lower_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate/flip each outside piece into its matching slot, then recolor the placed lower-half pixels red.

    Input structure:
    - A black background contains one large grey frame component with a 1-cell outer border and grey internal walls.
    - Inside the frame are 2-3 disconnected black hole slots; each slot is a 4-connected shape whose tight bbox has even height.
    - Outside the frame are 2-3 small monochrome 4-connected pieces in distinct non-black, non-grey colors.

    Rule:
    - Extract the black slots inside the grey frame.
    - For each outside piece, test all 8 dihedral orientations and match it to the unique slot with exactly the same shape.
    - Remove the outside pieces, place each matched piece into its slot using its original color, and keep the grey frame unchanged.
    - Within each slot bbox, recolor the placed pixels in the lower half rows to red while keeping the upper-half placed pixels in the piece color.

    Key insight:
    - The frame defines both the target slots and the row-based half split, so solving requires matching each outside shape to its slot under rotation/flip and then applying the recolor in slot coordinates.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    return _render_output(g)


def _frame_and_slots(
    layout_h: int, layout_w: int, shapes: list[np.ndarray]
) -> tuple[np.ndarray, list[dict]]:
    inner = np.full((layout_h, layout_w), GREY, dtype=int)
    occupied = np.zeros((layout_h, layout_w), dtype=bool)
    slots = []
    shape_order = list(range(len(shapes)))
    random.shuffle(shape_order)

    for idx in shape_order:
        mask = shapes[idx]
        h, w = mask.shape
        placed = False
        candidates = []
        for r0 in range(1, layout_h - h):
            for c0 in range(1, layout_w - w):
                r1 = r0 + h
                c1 = c0 + w
                pad_r0 = r0 - 1
                pad_c0 = c0 - 1
                pad_r1 = r1 + 1
                pad_c1 = c1 + 1
                if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                    continue
                candidates.append((r0, c0))
        random.shuffle(candidates)
        for r0, c0 in candidates:
            inner[r0 : r0 + h, c0 : c0 + w][mask] = BLACK
            occupied[r0 : r0 + h, c0 : c0 + w][mask] = True
            slots.append(
                {
                    "mask": mask.copy(),
                    "bbox": (r0 + 1, c0 + 1, r0 + h, c0 + w),
                }
            )
            placed = True
            break
        if not placed:
            raise ValueError("failed to place slots")

    frame = np.full((layout_h + 2, layout_w + 2), GREY, dtype=int)
    frame[1:-1, 1:-1] = inner
    return frame, slots


def _sample_shapes(count: int) -> list[np.ndarray]:
    candidates = SHAPE_POOL[:]
    random.shuffle(candidates)
    chosen = []
    seen = set()
    for mask in candidates:
        key = _canonical_key(mask)
        if key in seen:
            continue
        chosen.append(mask)
        seen.add(key)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("insufficient unique shapes")
    return chosen


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    region[patch] = int(color)


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, padding: int = 1) -> bool:
    h, w = mask.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(occupied.shape[0], top + h + padding)
    c1 = min(occupied.shape[1], left + w + padding)
    if occupied[r0:r1, c0:c1].any():
        return False
    return True


def _mark_occupied(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = mask.shape
    occupied[top : top + h, left : left + w][mask] = True


def _build_expected(grid: np.ndarray) -> np.ndarray:
    return _render_output(grid)


def _validate_example(inp: np.ndarray, expected: np.ndarray) -> None:
    out = main(inp)
    if not np.array_equal(out, expected):
        raise ValueError("main(input) mismatch")

    frame = _find_frame_component(inp)
    fr0, fc0, fr1, fc1 = frame["bbox"]
    slots = _extract_slots(inp, frame["bbox"])
    pieces = _color_components(inp, set(int(v) for v in np.unique(inp) if int(v) not in (BLACK, GREY)))
    if len(slots) != len(pieces):
        raise ValueError("piece/slot count mismatch")
    if len(slots) not in (2, 3):
        raise ValueError("expected 2 or 3 slots")

    placements = _match_pieces_to_slots(pieces, slots)
    non_identity = 0
    for piece, slot, oriented in placements:
        pr0, pc0, pr1, pc1 = piece["bbox"]
        if not (pr1 < fr0 - 1 or pr0 > fr1 + 1 or pc1 < fc0 - 1 or pc0 > fc1 + 1):
            raise ValueError("outside piece too close to frame")
        if not np.array_equal(piece["mask"], oriented):
            non_identity += 1
        r0, c0, r1, c1 = slot["bbox"]
        h = r1 - r0 + 1
        if h % 2 != 0:
            raise ValueError("slot bbox height must be even")
        upper_rows = np.arange(h)[:, None] < (h // 2)
        lower_rows = ~upper_rows
        if int((oriented & upper_rows).sum()) == 0 or int((oriented & lower_rows).sum()) == 0:
            raise ValueError("slot must occupy both halves")
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        if RED not in region or int(piece["color"]) not in region:
            raise ValueError("recolor split not visible")
    if non_identity == 0:
        raise ValueError("need at least one rotated/flipped piece")


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.choice([2, 3])
    shapes = _sample_shapes(piece_count)

    for _ in range(80):
        layout_h = random.randint(8, 12)
        layout_w = random.randint(8, 12)
        try:
            frame_patch, slot_specs = _frame_and_slots(layout_h, layout_w, shapes)
            break
        except ValueError:
            continue
    else:
        return None

    colors = list(PIECE_COLORS)
    random.shuffle(colors)
    piece_colors = colors[:piece_count]

    pieces = []
    orientation_choices = []
    for mask in shapes:
        options = _orientations(mask)
        nontrivial = [item for item in options if not np.array_equal(item[1], mask)]
        if not nontrivial:
            return None
        orientation_choices.append((options, nontrivial))

    chosen_non_identity = random.randrange(piece_count)
    for idx, mask in enumerate(shapes):
        options, nontrivial = orientation_choices[idx]
        if idx == chosen_non_identity or random.random() < 0.7:
            _, oriented = random.choice(nontrivial)
        else:
            _, oriented = random.choice(options)
        pieces.append({"mask": oriented, "color": piece_colors[idx]})

    frame_h, frame_w = frame_patch.shape
    piece_area = sum(piece["mask"].size for piece in pieces)
    min_h = max(16, frame_h + 4)
    min_w = max(16, frame_w + 8)
    max_h = min(26, max(min_h, frame_h + 10 + piece_area // 3))
    max_w = min(26, max(min_w, frame_w + 12 + piece_area // 2))
    if min_h > max_h or min_w > max_w:
        return None

    for _ in range(120):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        frame_top = random.randint(1, h - frame_h - 1)
        frame_left = random.randint(1, w - frame_w - 1)
        if not _can_place_mask(occupied, frame_patch != BLACK, frame_top, frame_left, padding=1):
            continue
        grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_patch
        occupied[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = True

        placed_ok = True
        for piece in pieces:
            ph, pw = piece["mask"].shape
            candidates = []
            for top in range(0, h - ph + 1):
                for left in range(0, w - pw + 1):
                    if not _can_place_mask(occupied, piece["mask"], top, left, padding=1):
                        continue
                    candidates.append((top, left))
            random.shuffle(candidates)
            if not candidates:
                placed_ok = False
                break
            top, left = candidates[0]
            _place_patch(grid, piece["mask"], top, left, piece["color"])
            _mark_occupied(occupied, piece["mask"], top, left)
        if not placed_ok:
            continue

        try:
            expected = _build_expected(grid)
            _validate_example(grid, expected)
        except Exception:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")

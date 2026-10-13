from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
GEN_ATTEMPTS = 400
VISIBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _mode_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _bbox_and_mask(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return (int(r0), int(c0), int(r1), int(c1)), mask


def _rotation_variants(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    variants: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (tuple(map(int, rotated.shape)), rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append((int(k), rotated.copy()))
    return variants


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for _, rotated in _rotation_variants(mask):
        keys.append((tuple(map(int, rotated.shape)), rotated.tobytes()))
    return min(keys)


def _find_separator_line(grid: np.ndarray, bg: int) -> tuple[str, int, int] | None:
    candidates: list[tuple[str, int, int]] = []
    h, w = map(int, grid.shape)
    for r in range(h):
        vals = np.unique(grid[r, :])
        if vals.size == 1 and int(vals[0]) != int(bg):
            candidates.append(("row", int(r), int(vals[0])))
    for c in range(w):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != int(bg):
            candidates.append(("col", int(c), int(vals[0])))
    grey_candidates = [cand for cand in candidates if int(cand[2]) == GREY]
    if len(grey_candidates) == 1:
        return grey_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _find_template(grid: np.ndarray, bg: int, sep_color: int | None) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[tuple[int, int], int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid)))):
        if color == int(bg) or (sep_color is not None and color == int(sep_color)):
            continue
        coords = [tuple(map(int, xy)) for xy in np.argwhere(grid == int(color))]
        if len(coords) < 8:
            continue
        bbox, _ = _bbox_and_mask(coords)
        r0, c0, r1, c1 = bbox
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if box.shape[0] < 3 or box.shape[1] < 3:
            continue
        if not (
            np.all(box[0, :] == int(color))
            and np.all(box[-1, :] == int(color))
            and np.all(box[:, 0] == int(color))
            and np.all(box[:, -1] == int(color))
        ):
            continue
        if np.any((box != int(color)) & (box != int(bg))):
            continue
        hole_mask = box == int(bg)
        if not bool(hole_mask.any()):
            continue
        hole_comps = _components(hole_mask)
        if not hole_comps:
            continue
        if any(
            any(
                rr == 0
                or cc == 0
                or rr == box.shape[0] - 1
                or cc == box.shape[1] - 1
                for rr, cc in comp
            )
            for comp in hole_comps
        ):
            continue
        bbox_area = int(box.shape[0] * box.shape[1])
        candidates.append(((bbox_area, len(coords)), int(color), bbox))
    assert candidates, "expected one rectangular template region"
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def _extract_holes(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, r1, c1 = bbox
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    hole_mask = box == int(bg)
    holes: list[dict] = []
    for comp in _components(hole_mask):
        if any(
            rr == 0 or cc == 0 or rr == box.shape[0] - 1 or cc == box.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        local_bbox, local_mask = _bbox_and_mask(comp)
        hr0, hc0, hr1, hc1 = local_bbox
        holes.append(
            {
                "bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "mask": local_mask,
            }
        )
    holes.sort(key=lambda hole: hole["bbox"])
    assert holes, "template must contain holes"
    return holes


def _extract_pieces(
    grid: np.ndarray,
    bg: int,
    template_color: int,
    sep_color: int | None,
) -> list[dict]:
    piece_mask = (grid != int(bg)) & (grid != int(template_color))
    if sep_color is not None:
        piece_mask &= grid != int(sep_color)
    pieces: list[dict] = []
    for comp in _components(piece_mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "pieces must be monochrome"
        bbox, mask = _bbox_and_mask(comp)
        pieces.append(
            {
                "coords": comp,
                "bbox": bbox,
                "mask": mask,
                "color": colors.pop(),
            }
        )
    pieces.sort(key=lambda piece: piece["bbox"])
    assert pieces, "expected loose pieces"
    return pieces


def _match_pieces_to_holes(pieces: list[dict], holes: list[dict]) -> dict[int, tuple[int, np.ndarray]]:
    assert len(pieces) == len(holes), "piece count must equal hole count"
    candidates: list[list[tuple[int, np.ndarray]]] = []
    for piece in pieces:
        piece_candidates: list[tuple[int, np.ndarray]] = []
        for _, rotated in _rotation_variants(piece["mask"]):
            for hi, hole in enumerate(holes):
                if rotated.shape == hole["mask"].shape and np.array_equal(rotated, hole["mask"]):
                    piece_candidates.append((int(hi), rotated.copy()))
        uniq: list[tuple[int, np.ndarray]] = []
        seen_holes: set[int] = set()
        for hi, rotated in piece_candidates:
            if hi in seen_holes:
                continue
            seen_holes.add(hi)
            uniq.append((int(hi), rotated))
        assert uniq, "every piece must match at least one hole"
        candidates.append(uniq)

    order = sorted(range(len(pieces)), key=lambda idx: len(candidates[idx]))
    assignment: dict[int, tuple[int, np.ndarray]] = {}
    used_holes: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        piece_index = order[pos]
        for hole_index, rotated in candidates[piece_index]:
            if hole_index in used_holes:
                continue
            used_holes.add(int(hole_index))
            assignment[piece_index] = (int(hole_index), rotated.copy())
            if dfs(pos + 1):
                return True
            used_holes.remove(int(hole_index))
            assignment.pop(piece_index, None)
        return False

    assert dfs(0), "piece-hole matching must be unique and complete"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each loose monochrome fragment into the uniquely matching hole of the template rectangle.

    Input structure:
    - The background is black.
    - One region is a large solid rectangle of one color whose boundary is intact, but whose interior contains several black 4-connected holes.
    - Another region contains the same number of loose monochrome 4-connected fragments in distinct colors.
    - The two regions are separated either by a clear black gap or by a full grey separator line.

    Rule:
    - Identify the template rectangle and extract each enclosed black hole as a shape.
    - For each loose fragment, try the four right-angle rotations and find the unique hole with exactly the same shape.
    - Fill that hole with the fragment's color in the matching rotation.
    - Clear the original fragment region to black, while leaving the template color and any grey separator unchanged.

    Key insight:
    - The loose pieces do not map by position; they must first be rotation-matched by shape, then assigned one-to-one to the template's holes.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    sep = _find_separator_line(g, bg)
    sep_color = None if sep is None else int(sep[2])
    template_color, template_bbox = _find_template(g, bg, sep_color)
    holes = _extract_holes(g, template_bbox, bg)
    pieces = _extract_pieces(g, bg, template_color, sep_color)
    assignment = _match_pieces_to_holes(pieces, holes)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = int(bg)
    for piece_index, (hole_index, _rotated) in assignment.items():
        hole = holes[hole_index]
        r0, c0, r1, c1 = hole["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[hole["mask"]] = int(pieces[piece_index]["color"])
    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_full_rectangle(mask: np.ndarray) -> bool:
    return int(mask.sum()) == int(mask.shape[0] * mask.shape[1])


def _random_connected_shape() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area_max = h * w - 1
        if area_max < 3:
            continue
        area = random.randint(3, area_max)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(start[0], start[1], h, w)
            if not bool(mask[nr, nc])
        }
        while int(mask.sum()) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            rr, cc = cell
            mask[rr, cc] = True
            for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                if not bool(mask[nr, nc]):
                    frontier.add((int(nr), int(nc)))
        if int(mask.sum()) != area:
            continue
        mask = _trim(mask)
        if min(mask.shape) < 2 or _is_full_rectangle(mask):
            continue
        return mask
    return None


def _shape_grid(count: int) -> tuple[int, int]:
    if count == 2:
        return random.choice([(1, 2), (2, 1)])
    if count in (3, 4):
        return 2, 2
    raise AssertionError("unsupported piece count")


def _build_template_block(shapes: list[np.ndarray], template_color: int) -> tuple[np.ndarray, list[dict]]:
    rows, cols = _shape_grid(len(shapes))
    slot_h = max(int(shape.shape[0]) for shape in shapes) + 1
    slot_w = max(int(shape.shape[1]) for shape in shapes) + 1
    border = 1
    gap = 1
    height = 2 * border + rows * slot_h + (rows - 1) * gap
    width = 2 * border + cols * slot_w + (cols - 1) * gap
    block = np.full((height, width), int(template_color), dtype=int)
    slot_ids = random.sample(range(rows * cols), len(shapes))
    holes: list[dict] = []
    for shape, slot in zip(shapes, slot_ids, strict=False):
        sr, sc = divmod(slot, cols)
        base_r = border + sr * (slot_h + gap)
        base_c = border + sc * (slot_w + gap)
        off_r = random.randint(0, slot_h - int(shape.shape[0]))
        off_c = random.randint(0, slot_w - int(shape.shape[1]))
        top = base_r + off_r
        left = base_c + off_c
        patch = block[top : top + shape.shape[0], left : left + shape.shape[1]]
        patch[shape] = BG
        holes.append(
            {
                "bbox": (int(top), int(left), int(top + shape.shape[0] - 1), int(left + shape.shape[1] - 1)),
                "mask": shape.copy(),
            }
        )
    return block, holes


def _build_piece_block(piece_masks: list[np.ndarray], colors: list[int]) -> tuple[np.ndarray, list[dict]]:
    rows, cols = _shape_grid(len(piece_masks))
    slot_h = max(int(shape.shape[0]) for shape in piece_masks) + 1
    slot_w = max(int(shape.shape[1]) for shape in piece_masks) + 1
    border = 1
    gap = 1
    height = 2 * border + rows * slot_h + (rows - 1) * gap
    width = 2 * border + cols * slot_w + (cols - 1) * gap
    block = np.full((height, width), BG, dtype=int)
    slot_ids = random.sample(range(rows * cols), len(piece_masks))
    entries: list[dict] = []
    for mask, color, slot in zip(piece_masks, colors, slot_ids, strict=False):
        sr, sc = divmod(slot, cols)
        base_r = border + sr * (slot_h + gap)
        base_c = border + sc * (slot_w + gap)
        off_r = random.randint(0, slot_h - int(mask.shape[0]))
        off_c = random.randint(0, slot_w - int(mask.shape[1]))
        top = base_r + off_r
        left = base_c + off_c
        patch = block[top : top + mask.shape[0], left : left + mask.shape[1]]
        assert not bool(np.any(patch[mask] != BG))
        patch[mask] = int(color)
        entries.append(
            {
                "bbox": (int(top), int(left), int(top + mask.shape[0] - 1), int(left + mask.shape[1] - 1)),
                "mask": mask.copy(),
                "color": int(color),
            }
        )
    return block, entries


def _compose_blocks(
    template_block: np.ndarray,
    piece_block: np.ndarray,
    *,
    use_separator: bool,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    horizontal = random.choice([True, False])
    swap_order = random.choice([True, False])
    top_margin = random.randint(0, 1)
    bottom_margin = random.randint(0, 1)
    left_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    mid = 1 if use_separator else random.randint(2, 3)

    th, tw = map(int, template_block.shape)
    ph, pw = map(int, piece_block.shape)

    if horizontal:
        height = max(th, ph) + top_margin + bottom_margin
        width = tw + pw + mid + left_margin + right_margin
    else:
        height = th + ph + mid + top_margin + bottom_margin
        width = max(tw, pw) + left_margin + right_margin
    if height > 30 or width > 30:
        raise ValueError("composed grid exceeds ARC size limit")

    grid = np.full((height, width), BG, dtype=int)
    if horizontal:
        template_r = top_margin + random.randint(0, height - top_margin - bottom_margin - th)
        piece_r = top_margin + random.randint(0, height - top_margin - bottom_margin - ph)
        if not swap_order:
            template_c = left_margin
            divider_c = template_c + tw
            piece_c = divider_c + mid
        else:
            piece_c = left_margin
            divider_c = piece_c + pw
            template_c = divider_c + mid
        if use_separator:
            grid[:, divider_c] = GREY
            if swap_order:
                template_c = divider_c + 1
            else:
                piece_c = divider_c + 1
    else:
        template_c = left_margin + random.randint(0, width - left_margin - right_margin - tw)
        piece_c = left_margin + random.randint(0, width - left_margin - right_margin - pw)
        if not swap_order:
            template_r = top_margin
            divider_r = template_r + th
            piece_r = divider_r + mid
        else:
            piece_r = top_margin
            divider_r = piece_r + ph
            template_r = divider_r + mid
        if use_separator:
            grid[divider_r, :] = GREY
            if swap_order:
                template_r = divider_r + 1
            else:
                piece_r = divider_r + 1

    grid[template_r : template_r + th, template_c : template_c + tw] = template_block
    piece_patch = grid[piece_r : piece_r + ph, piece_c : piece_c + pw]
    piece_patch[piece_block != BG] = piece_block[piece_block != BG]
    return grid, (int(template_r), int(template_c)), (int(piece_r), int(piece_c))


def _direct_expected(
    inp: np.ndarray,
    template_origin: tuple[int, int],
    hole_entries: list[dict],
    piece_origin: tuple[int, int],
    piece_entries: list[dict],
) -> np.ndarray:
    out = inp.copy()
    piece_r0, piece_c0 = piece_origin
    for piece in piece_entries:
        r0, c0, r1, c1 = piece["bbox"]
        patch = out[piece_r0 + r0 : piece_r0 + r1 + 1, piece_c0 + c0 : piece_c0 + c1 + 1]
        patch[piece["mask"]] = BG
    temp_r0, temp_c0 = template_origin
    for hole, piece in zip(hole_entries, piece_entries, strict=False):
        r0, c0, r1, c1 = hole["bbox"]
        patch = out[temp_r0 + r0 : temp_r0 + r1 + 1, temp_c0 + c0 : temp_c0 + c1 + 1]
        patch[hole["mask"]] = int(piece["color"])
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.randint(2, 4)
    shapes: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    while len(shapes) < hole_count:
        shape = _random_connected_shape()
        if shape is None:
            return None
        key = _canonical_rotation_key(shape)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        shapes.append(shape)

    usable_colors = list(VISIBLE_COLORS)
    template_color = random.choice(usable_colors)
    usable_colors.remove(int(template_color))
    piece_colors = random.sample(usable_colors, hole_count)

    display_masks: list[np.ndarray] = []
    forced_rotation_done = False
    for shape in shapes:
        differing = [k for k in range(1, 4) if not np.array_equal(np.rot90(shape, k), shape)]
        if differing and not forced_rotation_done and random.random() < 0.7:
            rot = random.choice(differing)
            forced_rotation_done = True
        else:
            rot = random.randint(0, 3)
            if rot in differing:
                forced_rotation_done = True
        display_masks.append(np.rot90(shape, rot).copy())
    if not forced_rotation_done:
        for i, shape in enumerate(shapes):
            differing = [k for k in range(1, 4) if not np.array_equal(np.rot90(shape, k), shape)]
            if differing:
                display_masks[i] = np.rot90(shape, random.choice(differing)).copy()
                forced_rotation_done = True
                break
    if not forced_rotation_done:
        return None

    template_block, hole_entries = _build_template_block(shapes, template_color)
    piece_block, piece_entries = _build_piece_block(display_masks, piece_colors)
    use_separator = random.choice([True, False])
    try:
        inp, template_origin, piece_origin = _compose_blocks(
            template_block,
            piece_block,
            use_separator=use_separator,
        )
    except ValueError:
        return None
    expected = _direct_expected(inp, template_origin, hole_entries, piece_origin, piece_entries)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        inp, expected = case
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")

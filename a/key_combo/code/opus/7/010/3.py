import itertools
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8

RECT_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)
RECT_LONG_SIDES = (5, 7)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_dims(coords) -> tuple[int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    arr = arr.copy()
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _mask_from_signature(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _coords_dims(sig)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[int(r), int(c)] = True
    return mask


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    base = _mask_from_signature(_normalize_coords(coords))
    variants = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for candidate in (rot, np.fliplr(rot)):
            variants.add(_normalize_coords(np.argwhere(candidate)))
    return min(variants)


def _shape_variants(sig: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _mask_from_signature(sig)
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            variant = _normalize_coords(np.argwhere(candidate))
            if variant not in seen:
                seen.add(variant)
                variants.append(variant)
    return variants


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _components_touch(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    other = {tuple(map(int, xy)) for xy in coords_b}
    for r, c in coords_a:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in other:
                return True
    return False


def _find_frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _components_of_color(grid, GREY):
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        expected = {(r1, c) for c in range(c0, c1 + 1)}
        expected.update((r, c1) for r in range(r0, r1 + 1))
        actual = {tuple(map(int, xy)) for xy in comp}
        if actual == expected and r1 > r0 and c1 > c0:
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected exactly one grey L frame"
    return candidates[0]


def _extract_template_signature(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0:r1, c0:c1]
    comps = _components_of_color(interior, CYAN)
    assert len(comps) == 1, "expected exactly one cyan template inside the frame"
    return _canonical_shape(comps[0])


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for color in RECT_COLORS:
        for comp in _components_of_color(grid, color):
            bbox = _bbox_from_coords(comp)
            r0, c0, r1, c1 = bbox
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if len(comp) == h * w and min(h, w) == 3 and max(h, w) in RECT_LONG_SIDES:
                rects.append({"color": int(color), "coords": comp, "bbox": bbox})
    assert rects, "expected at least one rectangle"
    return rects


def _find_magenta_sprites(grid: np.ndarray) -> list[np.ndarray]:
    sprites = _components_of_color(grid, MAGENTA)
    assert sprites, "expected magenta sprite components"
    return sprites


def _punch_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h == 3:
        center_r = r0 + 1
        for c in range(c0 + 1, c1, 2):
            grid[center_r, c] = BG
    else:
        center_c = c0 + 1
        for r in range(r0 + 1, r1, 2):
            grid[r, center_c] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """Use the template inside the grey L frame to decide which rectangles are punched and which are erased.

    Input structure:
    - Black background with one grey L-shaped frame whose vertical arm is on the right and horizontal arm is on the bottom.
    - The upper-left interior of that frame contains one cyan 4-connected template sprite.
    - Outside the frame are several solid 3x5 or 3x7 rectangles in distinct colors, each touching exactly one magenta sprite.

    Rule:
    - Read the cyan template's normalized shape inside the frame.
    - For each outside rectangle, find the unique magenta sprite that touches it orthogonally and compare that sprite's
      normalized shape to the template up to rotation and reflection.
    - If the sprite matches, erase the sprite and punch alternating black holes along the rectangle's center line on its long axis,
      starting one cell in from a short edge.
    - If the sprite does not match, erase both the rectangle and the sprite. Keep the grey frame and cyan template unchanged.

    Key insight:
    - Shape matching is a logic gate: it does not recolor the target, it chooses between two different operations on the paired rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bbox = _find_frame_bbox(g)
    template_sig = _extract_template_signature(g, frame_bbox)
    rects = _find_rectangles(g)
    sprites = _find_magenta_sprites(g)

    rect_hits = []
    sprite_hit_counts = [0] * len(sprites)
    for rect in rects:
        touching = [idx for idx, sprite in enumerate(sprites) if _components_touch(rect["coords"], sprite)]
        assert len(touching) == 1, "each rectangle must touch exactly one sprite"
        sprite_hit_counts[touching[0]] += 1
        rect_hits.append(touching[0])
    assert all(count == 1 for count in sprite_hit_counts), "each sprite must belong to exactly one rectangle"

    out = g.copy()
    for rect, sprite_idx in zip(rects, rect_hits):
        sprite = sprites[sprite_idx]
        sprite_sig = _canonical_shape(sprite)
        out[sprite[:, 0], sprite[:, 1]] = BG
        if sprite_sig == template_sig:
            _punch_rectangle(out, rect["bbox"])
        else:
            out[rect["coords"][:, 0], rect["coords"][:, 1]] = BG
    return out


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_shape_catalog() -> list[dict]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    catalog = {}
    for area in range(3, 7):
        for combo in itertools.combinations(cells, area):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if not _is_connected_mask(mask):
                continue
            sig = _canonical_shape(np.array(combo, dtype=int))
            if sig in catalog:
                continue
            variants = _shape_variants(sig)
            catalog[sig] = {
                "canonical": sig,
                "area": area,
                "variants": variants,
                "dims": sorted({_coords_dims(v) for v in variants}),
            }
    return sorted(catalog.values(), key=lambda item: (item["area"], item["canonical"]))


SHAPE_CATALOG = _build_shape_catalog()


def _build_template_options() -> list[dict]:
    options = []
    for entry in SHAPE_CATALOG:
        for variant in entry["variants"]:
            dims = _coords_dims(variant)
            hard = []
            for other in SHAPE_CATALOG:
                if other["canonical"] == entry["canonical"] or other["area"] != entry["area"]:
                    continue
                matching_variants = [cand for cand in other["variants"] if _coords_dims(cand) == dims]
                if matching_variants:
                    hard.append({"entry": other, "variants": matching_variants})
            if hard:
                options.append({"template": entry, "variant": variant, "hard_mismatches": hard})
    return options


TEMPLATE_OPTIONS = _build_template_options()


def _render_frame_patch(template_variant: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, dict]:
    th, tw = _coords_dims(template_variant)
    interior_h = random.randint(th + 1, min(6, th + 3))
    interior_w = random.randint(tw + 1, min(6, tw + 3))
    patch = np.zeros((interior_h + 1, interior_w + 1), dtype=int)
    patch[-1, :] = GREY
    patch[:, -1] = GREY
    top = random.randint(0, interior_h - th)
    left = random.randint(0, interior_w - tw)
    template_cells = []
    for r, c in template_variant:
        rr = top + int(r)
        cc = left + int(c)
        patch[rr, cc] = CYAN
        template_cells.append((rr, cc))
    return patch, {"template_coords": tuple(template_cells), "frame_bbox": (0, 0, interior_h, interior_w)}


def _render_pair_patch(
    rect_color: int,
    long_side: int,
    horizontal: bool,
    side: str,
    sprite_variant: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, dict]:
    sh, sw = _coords_dims(sprite_variant)
    if horizontal:
        patch = np.zeros((3 + sh, long_side), dtype=int)
        if side == "top":
            rect_r0, rect_c0 = sh, 0
            sprite_r0, sprite_c0 = 0, random.randint(0, long_side - sw)
        else:
            rect_r0, rect_c0 = 0, 0
            sprite_r0, sprite_c0 = 3, random.randint(0, long_side - sw)
        patch[rect_r0 : rect_r0 + 3, rect_c0 : rect_c0 + long_side] = rect_color
        rect_bbox = (rect_r0, rect_c0, rect_r0 + 2, rect_c0 + long_side - 1)
    else:
        patch = np.zeros((long_side, 3 + sw), dtype=int)
        if side == "left":
            rect_r0, rect_c0 = 0, sw
            sprite_r0, sprite_c0 = random.randint(0, long_side - sh), 0
        else:
            rect_r0, rect_c0 = 0, 0
            sprite_r0, sprite_c0 = random.randint(0, long_side - sh), 3
        patch[rect_r0 : rect_r0 + long_side, rect_c0 : rect_c0 + 3] = rect_color
        rect_bbox = (rect_r0, rect_c0, rect_r0 + long_side - 1, rect_c0 + 2)

    sprite_cells = []
    for r, c in sprite_variant:
        rr = sprite_r0 + int(r)
        cc = sprite_c0 + int(c)
        patch[rr, cc] = MAGENTA
        sprite_cells.append((rr, cc))

    return patch, {"rect_bbox": rect_bbox, "sprite_coords": tuple(sprite_cells)}


def _can_place(mask: np.ndarray, forbidden: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    return not np.any(mask & forbidden[top : top + h, left : left + w])


def _stamp_patch(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = grid[top : top + h, left : left + w]
    view[mask] = patch[mask]
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    forbidden[nr, nc] = True


def _random_location(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    rows = list(range(1, grid.shape[0] - h))
    cols = list(range(1, grid.shape[1] - w))
    if not rows or not cols:
        return None
    positions = [(r, c) for r in rows for c in cols]
    random.shuffle(positions)
    mask = patch != BG
    for top, left in positions:
        if _can_place(mask, forbidden, top, left):
            return top, left
    return None


def _translate_coords(coords, dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r) + dr, int(c) + dc) for r, c in coords)


def _build_expected_output(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for info in pairs:
        sprite_coords = np.array(info["sprite_coords"], dtype=int)
        out[sprite_coords[:, 0], sprite_coords[:, 1]] = BG
        if info["matches"]:
            _punch_rectangle(out, info["rect_bbox"])
        else:
            rect_coords = np.array(info["rect_coords"], dtype=int)
            out[rect_coords[:, 0], rect_coords[:, 1]] = BG
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    option = random.choice(TEMPLATE_OPTIONS)
    template_entry = option["template"]
    template_variant = option["variant"]
    template_dims = _coords_dims(template_variant)
    hard_choices = option["hard_mismatches"]

    grid_h = random.randint(17, 24)
    grid_w = random.randint(17, 24)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)

    frame_patch, frame_meta = _render_frame_patch(template_variant)
    frame_pos = _random_location(grid, forbidden, frame_patch)
    if frame_pos is None:
        return None
    frame_top, frame_left = frame_pos
    _stamp_patch(grid, forbidden, frame_patch, frame_top, frame_left)

    pair_count = random.randint(3, 5)
    match_count = random.randint(1, pair_count - 1)
    flags = [True] * match_count + [False] * (pair_count - match_count)
    random.shuffle(flags)
    rect_colors = random.sample(list(RECT_COLORS), pair_count)

    mismatch_slots = [i for i, flag in enumerate(flags) if not flag]
    hard_slot = random.choice(mismatch_slots)

    same_area_others = [entry for entry in SHAPE_CATALOG if entry["canonical"] != template_entry["canonical"] and entry["area"] == template_entry["area"]]
    all_other_shapes = [entry for entry in SHAPE_CATALOG if entry["canonical"] != template_entry["canonical"]]

    pairs = []
    hard_mismatch_seen = False
    for idx, matches in enumerate(flags):
        if matches:
            variants = template_entry["variants"][:]
            random.shuffle(variants)
            if len(variants) > 1 and idx == flags.index(True):
                preferred = [v for v in variants if v != template_variant]
                sprite_variant = preferred[0] if preferred else variants[0]
            else:
                sprite_variant = variants[0]
        else:
            if idx == hard_slot:
                hard_choice = random.choice(hard_choices)
                sprite_variant = random.choice(hard_choice["variants"])
                hard_mismatch_seen = True
            else:
                if same_area_others and random.random() < 0.7:
                    chosen = random.choice(same_area_others)
                else:
                    chosen = random.choice(all_other_shapes)
                sprite_variant = random.choice(chosen["variants"])

        horizontal = random.random() < 0.5
        long_side = random.choice(RECT_LONG_SIDES)
        side = random.choice(("top", "bottom") if horizontal else ("left", "right"))
        patch, local_meta = _render_pair_patch(rect_colors[idx], long_side, horizontal, side, sprite_variant)
        pos = _random_location(grid, forbidden, patch)
        if pos is None:
            return None
        top, left = pos
        _stamp_patch(grid, forbidden, patch, top, left)

        rect_bbox = local_meta["rect_bbox"]
        rr0, cc0, rr1, cc1 = rect_bbox
        abs_bbox = (top + rr0, left + cc0, top + rr1, left + cc1)
        rect_coords = tuple(
            (r, c)
            for r in range(abs_bbox[0], abs_bbox[2] + 1)
            for c in range(abs_bbox[1], abs_bbox[3] + 1)
        )
        pairs.append(
            {
                "matches": matches,
                "rect_bbox": abs_bbox,
                "rect_coords": rect_coords,
                "sprite_coords": _translate_coords(local_meta["sprite_coords"], top, left),
                "sprite_signature": _canonical_shape(np.array(sprite_variant, dtype=int)),
            }
        )

    if not hard_mismatch_seen:
        return None

    expected = _build_expected_output(grid, pairs)
    meta = {
        "template_signature": _canonical_shape(np.array(template_variant, dtype=int)),
        "frame_bbox": (
            frame_top + frame_meta["frame_bbox"][0],
            frame_left + frame_meta["frame_bbox"][1],
            frame_top + frame_meta["frame_bbox"][2],
            frame_left + frame_meta["frame_bbox"][3],
        ),
        "template_coords": _translate_coords(frame_meta["template_coords"], frame_top, frame_left),
        "pairs": pairs,
        "hard_mismatch": hard_mismatch_seen,
        "template_dims": template_dims,
    }
    return grid, expected, meta


def _validate_case(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    if sum(1 for pair in meta["pairs"] if pair["matches"]) == 0:
        return False
    if sum(1 for pair in meta["pairs"] if not pair["matches"]) == 0:
        return False
    if not meta["hard_mismatch"]:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    frame_bbox = meta["frame_bbox"]
    r0, c0, r1, c1 = frame_bbox
    if not np.array_equal(solved[r0 : r1 + 1, c0 : c1 + 1], inp[r0 : r1 + 1, c0 : c1 + 1]):
        return False

    template_sig = meta["template_signature"]
    for pair in meta["pairs"]:
        sprite = np.array(pair["sprite_coords"], dtype=int)
        if np.any(solved[sprite[:, 0], sprite[:, 1]] != BG):
            return False
        if pair["matches"]:
            if pair["sprite_signature"] != template_sig:
                return False
            rect_coords = np.array(pair["rect_coords"], dtype=int)
            after = solved[rect_coords[:, 0], rect_coords[:, 1]]
            before = inp[rect_coords[:, 0], rect_coords[:, 1]]
            if not np.any(after == BG):
                return False
            if not np.all((after == BG) | (after == before)):
                return False
        else:
            if pair["sprite_signature"] == template_sig:
                return False
            rect = np.array(pair["rect_coords"], dtype=int)
            if np.any(solved[rect[:, 0], rect[:, 1]] != BG):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _generate_case()
        if case is None:
            continue
        inp, expected, meta = case
        if _validate_case(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid ARC instance")

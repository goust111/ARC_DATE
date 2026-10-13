from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
STRUCTURE = Color.GREY
PIECE_COLORS = [c for c in Color.NOT_BLACK if c != STRUCTURE]
GEN_ATTEMPTS = 500


def _d4_transforms(arr: np.ndarray) -> list[np.ndarray]:
    return [
        arr,
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
        np.flipud(arr),
        np.fliplr(arr),
        np.flipud(np.rot90(arr, 1)),
        np.fliplr(np.rot90(arr, 1)),
    ]


def _unique_d4_transforms(arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[tuple[int, np.ndarray]] = []
    for tid, variant in enumerate(_d4_transforms(arr)):
        key = (tuple(map(int, variant.shape)), variant.tobytes())
        if key not in seen:
            seen.add(key)
            out.append((tid, np.asarray(variant, dtype=int)))
    return out


def _mask(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr != BG, dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    base = mask.astype(np.uint8)
    for variant in _d4_transforms(base)[1:]:
        if variant.shape == base.shape and np.array_equal(variant, base):
            return True
    return False


def _piece_signature(sprite: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, sprite.shape)), sprite.tobytes()


def _component_sprite(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    return crop_bbox(grid, bbox)


def _extract_slots(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    slots: list[dict] = []
    slot_mask = np.zeros(grid.shape, dtype=bool)
    grey_components = find_components(
        grid,
        bg=get_bg(grid),
        connectivity=4,
        monochromatic=True,
        colors=[STRUCTURE],
    )
    for comp in grey_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        patch = crop_bbox(grid, comp["bbox"])
        if not np.all(patch[0, :] == STRUCTURE):
            continue
        if not np.all(patch[-1, :] == STRUCTURE):
            continue
        if not np.all(patch[:, 0] == STRUCTURE):
            continue
        if not np.all(patch[:, -1] == STRUCTURE):
            continue
        interior = patch[1:-1, 1:-1]
        if interior.size == 0 or not np.any(interior == STRUCTURE):
            continue
        heights: list[int] = []
        ih, iw = interior.shape
        valid = True
        for col in range(iw):
            height = 0
            while height < ih and interior[ih - 1 - height, col] == STRUCTURE:
                height += 1
            if np.any(interior[: ih - height, col] == STRUCTURE):
                valid = False
                break
            heights.append(height)
        if not valid or min(heights) <= 0:
            continue
        slot_mask[r0 : r1 + 1, c0 : c1 + 1] |= patch == STRUCTURE
        slots.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior_h": ih,
                "interior_w": iw,
                "terrain": tuple(int(v) for v in heights),
            }
        )
    slots.sort(key=lambda slot: (slot["bbox"][0], slot["bbox"][1]))
    return slots, slot_mask


def _extract_pieces(grid: np.ndarray, slot_mask: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    outside = grid.copy()
    outside[slot_mask] = bg
    pieces: list[dict] = []
    for comp in find_components(outside, bg=bg, connectivity=4, monochromatic=False):
        sprite = _component_sprite(outside, comp["bbox"])
        pieces.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": np.asarray(comp["coords"], dtype=int),
                "sprite": np.asarray(sprite, dtype=int),
            }
        )
    pieces.sort(key=lambda piece: (piece["bbox"][0], piece["bbox"][1]))
    return pieces


def _fit_positions(slot: dict, sprite: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = map(int, sprite.shape)
    ih = int(slot["interior_h"])
    iw = int(slot["interior_w"])
    if sh > ih or sw > iw:
        return []

    fg = np.argwhere(sprite != BG)
    if fg.size == 0:
        return []

    bottom_by_col: list[int | None] = []
    for col in range(sw):
        rows = np.flatnonzero(sprite[:, col] != BG)
        bottom_by_col.append(None if rows.size == 0 else int(rows.max()))

    terrain = slot["terrain"]
    fits: list[tuple[int, int]] = []
    for left in range(iw - sw + 1):
        top: int | None = None
        ok = True
        for col, bottom in enumerate(bottom_by_col):
            if bottom is None:
                continue
            candidate = ih - int(terrain[left + col]) - 1 - bottom
            if top is None:
                top = int(candidate)
            elif int(candidate) != top:
                ok = False
                break
        if not ok or top is None or top < 0 or top + sh > ih:
            continue
        for r, c in fg:
            rr = int(top + r)
            cc = int(left + c)
            terrain_top = ih - int(terrain[cc])
            if rr >= terrain_top:
                ok = False
                break
        if ok:
            fits.append((int(top), int(left)))
    return fits


def _all_slot_matches(slots: list[dict], pieces: list[dict]) -> list[list[dict]]:
    per_slot: list[list[dict]] = []
    for slot_index, slot in enumerate(slots):
        slot_matches: list[dict] = []
        for piece_index, piece in enumerate(pieces):
            seen: set[tuple[tuple[int, int], bytes, int, int]] = set()
            for transform_id, variant in _unique_d4_transforms(piece["sprite"]):
                for top, left in _fit_positions(slot, variant):
                    key = (_piece_signature(variant)[0], _piece_signature(variant)[1], top, left)
                    if key in seen:
                        continue
                    seen.add(key)
                    slot_matches.append(
                        {
                            "slot_index": slot_index,
                            "piece_index": piece_index,
                            "transform_id": transform_id,
                            "sprite": variant,
                            "top": int(top),
                            "left": int(left),
                        }
                    )
        per_slot.append(slot_matches)
    return per_slot


def _find_unique_assignment(grid: np.ndarray) -> tuple[list[dict], list[dict], tuple[dict, dict]]:
    slots, slot_mask = _extract_slots(grid)
    assert len(slots) == 2, "expected exactly two slots"
    pieces = _extract_pieces(grid, slot_mask)
    assert len(pieces) == 3, "expected exactly three outside pieces"

    per_slot = _all_slot_matches(slots, pieces)
    assignments: list[tuple[dict, dict]] = []
    for left_match in per_slot[0]:
        for right_match in per_slot[1]:
            if left_match["piece_index"] != right_match["piece_index"]:
                assignments.append((left_match, right_match))
    assert len(assignments) == 1, "slot-piece assignment must be unique"
    return slots, pieces, assignments[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each outer multicolor piece into the unique grey slot whose floor profile it complements after rotation or reflection.

    Input structure:
    - The grid has a black background, two grey rectangular slot frames, and three separated multicolor pieces outside them.
    - Inside each frame, the bottom interior contains grey pillar terrain with a non-flat column-height profile.
    - The three outside pieces are connected, shape-distinct, and not already shown in their fitting orientation.

    Rule:
    - Detect the two grey slots and the three outside pieces.
    - For each slot, enumerate every piece under all rotations and reflections and every horizontal placement that stays inside the slot.
    - Keep the unique global assignment where two different pieces land so that every occupied piece column touches the terrain exactly one cell above it, with no overlap and no gaps; erase those two original pieces and paste the fitted orientations into the slots.
    - Leave the unused distractor piece and all slot terrain unchanged.

    Key insight:
    - The lock is the slot's bottom height profile: only one transformed piece per slot has a lower contour that is column-wise complementary to that terrain.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    slots, pieces, matches = _find_unique_assignment(g)

    out = g.copy()
    for match in matches:
        piece = pieces[int(match["piece_index"])]
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for match in matches:
        slot = slots[int(match["slot_index"])]
        r0, c0, _r1, _c1 = slot["bbox"]
        paste(
            out,
            match["sprite"],
            r0 + 1 + int(match["top"]),
            c0 + 1 + int(match["left"]),
            transparent=bg,
        )
    return out


def _slot_patch(interior_h: int, terrain: tuple[int, ...]) -> np.ndarray:
    interior_w = len(terrain)
    patch = np.full((interior_h + 2, interior_w + 2), BG, dtype=int)
    patch[0, :] = STRUCTURE
    patch[-1, :] = STRUCTURE
    patch[:, 0] = STRUCTURE
    patch[:, -1] = STRUCTURE
    for col, height in enumerate(terrain, start=1):
        patch[interior_h + 1 - height : interior_h + 1, col] = STRUCTURE
    return patch


def _choose_x(max_left: int) -> int:
    if max_left <= 0:
        return 0
    edge_choices = [0, max_left]
    if random.random() < 0.5:
        return random.choice(edge_choices)
    return random.randint(0, max_left)


def _build_mask_from_bottoms(bottoms: tuple[int, ...]) -> np.ndarray | None:
    width = len(bottoms)
    height = max(bottoms) + 1
    mask = np.zeros((height, width), dtype=bool)
    for col, bottom in enumerate(bottoms):
        mask[: bottom + 1, col] = True

    columns = [c for c, bottom in enumerate(bottoms) if bottom >= 2]
    random.shuffle(columns)
    cuts = random.randint(1, max(1, min(3, len(columns))))
    for col in columns[:cuts]:
        mask[0, col] = False

    if not np.any(mask[0, :]):
        tallest = max(range(width), key=lambda c: bottoms[c])
        mask[0, tallest] = True
    if not np.all(mask.any(axis=0)):
        return None
    if not _is_connected(mask):
        return None
    if _has_nontrivial_symmetry(mask):
        return None
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    num_colors = random.randint(2, min(4, len(PIECE_COLORS)))
    palette = random.sample(PIECE_COLORS, num_colors)
    sprite = np.full(mask.shape, BG, dtype=int)
    offset = random.randint(0, num_colors - 1)
    for index, (r, c) in enumerate(coords):
        sprite[r, c] = palette[(3 * int(r) + 5 * int(c) + index + offset) % num_colors]
    return sprite


def _make_piece_from_bottoms(bottoms: tuple[int, ...]) -> np.ndarray | None:
    mask = _build_mask_from_bottoms(bottoms)
    if mask is None:
        return None
    sprite = _colorize_mask(mask)
    if _has_nontrivial_symmetry(_mask(sprite)):
        return None
    return sprite


def _random_terrain(interior_h: int, interior_w: int) -> tuple[int, ...] | None:
    max_height = interior_h - 3
    if max_height < 2:
        return None
    for _ in range(80):
        terrain = tuple(random.randint(1, max_height) for _ in range(interior_w))
        if len(set(terrain)) < 2:
            continue
        if max(terrain) - min(terrain) < 1:
            continue
        return terrain
    return None


def _matched_piece_for_slot(slot: dict, transform_pool: list[int]) -> dict | None:
    iw = int(slot["interior_w"])
    ih = int(slot["interior_h"])
    if iw < 5:
        return None
    width = random.randint(3, min(5, iw - 1))
    left = _choose_x(iw - width)
    segment = slot["terrain"][left : left + width]
    high = max(segment)
    if high + 2 > ih - 1:
        return None
    level = random.randint(high + 2, ih - 1)
    bottoms = tuple(int(level - value) for value in segment)
    sprite = _make_piece_from_bottoms(bottoms)
    if sprite is None:
        return None
    fit_options = _fit_positions(slot, sprite)
    if fit_options != [(ih - level - 1, left)]:
        return None
    display_tid = random.choice(transform_pool)
    display_sprite = np.asarray(_d4_transforms(sprite)[display_tid], dtype=int)
    return {
        "fit_sprite": sprite,
        "display_sprite": display_sprite,
        "expected_top": ih - level - 1,
        "expected_left": left,
    }


def _distractor_piece(slots: list[dict]) -> np.ndarray | None:
    base_slot = random.choice(slots)
    iw = int(base_slot["interior_w"])
    ih = int(base_slot["interior_h"])
    width = random.randint(3, min(5, iw - 1))
    left = _choose_x(iw - width)
    segment = base_slot["terrain"][left : left + width]
    high = max(segment)
    if high + 2 > ih - 1:
        return None
    level = random.randint(high + 2, ih - 1)
    bottoms = [int(level - value) for value in segment]
    idx = random.randrange(width)
    delta_choices = [delta for delta in (-2, -1, 1, 2) if bottoms[idx] + delta >= 1]
    if not delta_choices:
        return None
    bottoms[idx] += random.choice(delta_choices)
    sprite = _make_piece_from_bottoms(tuple(bottoms))
    if sprite is None:
        return None
    return np.asarray(_d4_transforms(sprite)[random.randint(1, 7)], dtype=int)


def _place_on_canvas(canvas: np.ndarray, patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for index in order:
        patch = patches[index]
        try:
            r, c = random_free_location(canvas, patch, transparent=BG, border=1, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=BG)
        placements[index] = (int(r), int(c))
    return placements


def _compose_scene(shape: tuple[int, int], slot_specs: list[dict], piece_sprites: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    slot_patches = [_slot_patch(slot["interior_h"], slot["terrain"]) for slot in slot_specs]
    canvas = np.full(shape, BG, dtype=int)
    slot_places = _place_on_canvas(canvas, slot_patches)
    if slot_places is None:
        return None

    piece_places = _place_on_canvas(canvas, piece_sprites)
    if piece_places is None:
        return None
    return canvas, slot_places, piece_places


def _expected_output(input_grid: np.ndarray, slot_specs: list[dict], slot_places: list[tuple[int, int]], piece_places: list[tuple[int, int]], piece_specs: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for piece_place, spec in zip(piece_places, piece_specs):
        if "fit_sprite" not in spec:
            continue
        pr, pc = piece_place
        sprite = spec["display_sprite"]
        mask = sprite != BG
        sub = out[pr : pr + sprite.shape[0], pc : pc + sprite.shape[1]]
        sub[mask] = BG

    for slot_index, spec in enumerate(piece_specs[:2]):
        fit_sprite = spec["fit_sprite"]
        sr, sc = slot_places[slot_index]
        paste(
            out,
            fit_sprite,
            sr + 1 + int(spec["expected_top"]),
            sc + 1 + int(spec["expected_left"]),
            transparent=BG,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(22, 30), random.randint(22, 30))
    slot_specs: list[dict] = []
    for _ in range(2):
        interior_h = random.randint(5, 7)
        interior_w = random.randint(6, 9)
        terrain = _random_terrain(interior_h, interior_w)
        if terrain is None:
            return None
        slot_specs.append({"interior_h": interior_h, "interior_w": interior_w, "terrain": terrain})
    if slot_specs[0]["terrain"] == slot_specs[1]["terrain"]:
        return None

    first = _matched_piece_for_slot(slot_specs[0], [1, 2, 3])
    second = _matched_piece_for_slot(slot_specs[1], [4, 5, 6, 7])
    distractor = _distractor_piece(slot_specs)
    if first is None or second is None or distractor is None:
        return None

    masks = [_mask(first["fit_sprite"]), _mask(second["fit_sprite"]), _mask(distractor)]
    if any(_has_nontrivial_symmetry(mask) for mask in masks):
        return None
    if len({_piece_signature(mask.astype(np.uint8)) for mask in masks}) < 3:
        return None

    piece_specs = [first, second, {"display_sprite": distractor}]
    piece_sprites = [spec["display_sprite"] for spec in piece_specs]

    composed = _compose_scene(shape, slot_specs, piece_sprites)
    if composed is None:
        return None
    grid, slot_places, piece_places = composed
    expected = _expected_output(grid, slot_specs, slot_places, piece_places, piece_specs)

    try:
        solved = main(grid)
        slots, pieces, matches = _find_unique_assignment(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if len(slots) != 2 or len(pieces) != 3 or len(matches) != 2:
        return None
    chosen = {match["piece_index"] for match in matches}
    if len(chosen) != 2:
        return None
    unused = [idx for idx in range(3) if idx not in chosen]
    if len(unused) != 1:
        return None
    for slot_match in matches:
        slot = slots[slot_match["slot_index"]]
        piece = pieces[slot_match["piece_index"]]
        fit_only = _fit_positions(slot, slot_match["sprite"])
        if fit_only != [(slot_match["top"], slot_match["left"])]:
            return None
        if piece["sprite"].shape == slot_match["sprite"].shape and np.array_equal(piece["sprite"], slot_match["sprite"]):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

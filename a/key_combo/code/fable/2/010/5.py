from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, paste, scale_sprite


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
SPRITE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
COLOR_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 80


def _primitive_mask(mask: np.ndarray) -> np.ndarray:
    cur = np.asarray(mask, dtype=bool)
    while True:
        h, w = cur.shape
        g = math.gcd(h, w)
        reduced = False
        for factor in range(g, 1, -1):
            if h % factor != 0 or w % factor != 0:
                continue
            small = cur[::factor, ::factor]
            if np.array_equal(scale_sprite(small.astype(int), factor).astype(bool), cur):
                cur = small
                reduced = True
                break
        if not reduced:
            return cur


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    primitive = _primitive_mask(mask)
    return primitive.shape, primitive.astype(np.uint8).tobytes()


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _outline_info_from_filled(filled: np.ndarray) -> dict | None:
    filled = np.asarray(filled, dtype=bool)
    eroded = binary_erosion(filled, structure=STRUCT4, border_value=0)
    outline = filled & ~eroded
    interior = filled & ~outline
    if not interior.any():
        return None
    if len(components_mask(outline, connectivity=4)) != 1:
        return None
    if len(components_mask(interior, connectivity=4)) != 1:
        return None
    if not np.array_equal(binary_fill_holes(outline), filled):
        return None
    return {
        "filled": filled,
        "outline": outline,
        "interior": interior,
    }


def _outline_objects(grid: np.ndarray, bg: int) -> list[dict]:
    outlines = []
    for comp in find_components(grid, bg=bg, connectivity=4, colors=[Color.GREY]):
        local = crop_bbox(grid, comp["bbox"]) == Color.GREY
        info = _outline_info_from_filled(binary_fill_holes(local))
        if info is None:
            continue
        if not np.array_equal(info["outline"], local):
            continue
        outlines.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": comp["coords"],
                "outline": info["outline"],
                "interior": info["interior"],
                "filled": info["filled"],
                "patch": crop_bbox(grid, comp["bbox"]),
            }
        )
    return outlines


def _sprite_objects(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), int(Color.GREY)}]
    if not colors:
        return []
    sprites = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False, colors=colors):
        patch = crop_bbox(grid, comp["bbox"])
        mask = patch != bg
        vals = sorted(set(map(int, patch[mask].tolist())))
        if len(vals) != 2:
            continue
        sprites.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": comp["coords"],
                "patch": patch,
                "mask": mask,
                "colors": vals,
            }
        )
    return sprites


def _match_scale(filled: np.ndarray, sprite_mask: np.ndarray) -> int | None:
    fh, fw = filled.shape
    sh, sw = sprite_mask.shape
    if sh == 0 or sw == 0 or fh % sh != 0 or fw % sw != 0:
        return None
    factor_h = fh // sh
    factor_w = fw // sw
    if factor_h != factor_w or factor_h < 1:
        return None
    factor = int(factor_h)
    scaled = scale_sprite(sprite_mask.astype(int), factor).astype(bool)
    if np.array_equal(scaled, filled):
        return factor
    return None


def _candidate_pairs(outlines: list[dict], sprites: list[dict]) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = [[] for _ in outlines]
    for oi, outline in enumerate(outlines):
        for si, sprite in enumerate(sprites):
            factor = _match_scale(outline["filled"], sprite["mask"])
            if factor is not None:
                candidates[oi].append((si, factor))
    return candidates


def _choose_pairs(candidates: list[list[tuple[int, int]]]) -> list[tuple[int, int, int]]:
    best_size = -1
    best: list[list[tuple[int, int, int]]] = []

    def search(oi: int, used: set[int], chosen: list[tuple[int, int, int]]) -> None:
        nonlocal best_size, best
        if len(chosen) + (len(candidates) - oi) < best_size:
            return
        if oi == len(candidates):
            size = len(chosen)
            if size > best_size:
                best_size = size
                best = [chosen.copy()]
            elif size == best_size:
                best.append(chosen.copy())
            return

        search(oi + 1, used, chosen)
        for si, factor in candidates[oi]:
            if si in used:
                continue
            used.add(si)
            chosen.append((oi, si, factor))
            search(oi + 1, used, chosen)
            chosen.pop()
            used.remove(si)

    search(0, set(), [])
    if not best:
        return []
    best.sort()
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each matched gray hollow outline with the enlarged pattern of its shape-matching two-color sprite.

    Input structure
    - The background is black.
    - Some gray 1-cell-thick closed outline objects enclose empty black interiors.
    - Separate small foreground components are solid two-color sprites on the background.

    Rule
    - For each gray outline, treat the outline together with its enclosed area as one filled silhouette.
    - If a sprite's overall occupied shape is exactly that silhouette reduced by an integer factor, enlarge the whole sprite by that factor and align it to the silhouette.
    - Write the enlarged colors only into the outline's interior cells, keep the gray outline unchanged, and erase the matched source sprite.
    - Sprites that match no outline stay in place, and outlines with no matching sprite remain hollow.

    Key insight
    - The small sprite is not just a color key: its full two-color pixel arrangement is a miniature negative whose filled shape identifies the matching outline after cross-scale silhouette comparison.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    out = g.copy()

    outlines = _outline_objects(g, bg)
    sprites = _sprite_objects(g, bg)
    pairs = _choose_pairs(_candidate_pairs(outlines, sprites))

    for oi, si, factor in pairs:
        outline = outlines[oi]
        sprite = sprites[si]
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

        scaled = scale_sprite(sprite["patch"], factor)
        r0, c0, r1, c1 = outline["bbox"]
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[outline["interior"]] = scaled[outline["interior"]]
        view[outline["outline"]] = Color.GREY

    return out


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        area = random.randint(max(3, (h * w + 1) // 2), h * w)

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], (h, w)))
        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], (h, w)):
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        coords = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        if not np.array_equal(binary_fill_holes(mask), mask):
            continue
        return mask
    return None


def _sample_shape(existing_keys: set[tuple[tuple[int, int], bytes]]) -> tuple[np.ndarray, int, dict] | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_connected_mask()
        if mask is None:
            continue
        key = _shape_key(mask)
        if key in existing_keys:
            continue
        factor = random.randint(2, 4)
        filled = scale_sprite(mask.astype(int), factor).astype(bool)
        info = _outline_info_from_filled(filled)
        if info is None:
            continue
        existing_keys.add(key)
        return mask, factor, info
    return None


def _sample_sprite_patch(mask: np.ndarray, colors: tuple[int, int], factor: int, interior: np.ndarray) -> np.ndarray | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for _ in range(COLOR_ATTEMPTS):
        sprite = np.zeros(mask.shape, dtype=int)
        group_a = set(random.sample(cells, random.randint(1, len(cells) - 1)))
        for r, c in cells:
            sprite[r, c] = colors[0] if (r, c) in group_a else colors[1]
        scaled = scale_sprite(sprite, factor)
        visible = set(map(int, scaled[interior].tolist()))
        visible.discard(Color.BLACK)
        if visible == set(colors):
            return sprite
    return None


def _build_matched_pair(existing_keys: set[tuple[tuple[int, int], bytes]]) -> tuple[dict, dict] | None:
    sampled = _sample_shape(existing_keys)
    if sampled is None:
        return None
    mask, factor, info = sampled
    colors = tuple(random.sample(SPRITE_COLORS, 2))
    sprite_patch = _sample_sprite_patch(mask, colors, factor, info["interior"])
    if sprite_patch is None:
        return None

    outline_patch = np.where(info["outline"], Color.GREY, Color.BLACK).astype(int)
    return (
        {
            "kind": "outline",
            "patch": outline_patch,
            "filled": info["filled"],
            "outline": info["outline"],
            "interior": info["interior"],
            "factor": factor,
        },
        {
            "kind": "sprite",
            "patch": sprite_patch,
            "mask": mask,
        },
    )


def _build_unmatched_outline(existing_keys: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    sampled = _sample_shape(existing_keys)
    if sampled is None:
        return None
    _mask, _factor, info = sampled
    return {
        "kind": "outline",
        "patch": np.where(info["outline"], Color.GREY, Color.BLACK).astype(int),
        "filled": info["filled"],
        "outline": info["outline"],
        "interior": info["interior"],
        "factor": None,
    }


def _build_distractor_sprite(existing_keys: set[tuple[tuple[int, int], bytes]]) -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_connected_mask()
        if mask is None:
            continue
        key = _shape_key(mask)
        if key in existing_keys:
            continue
        colors = tuple(random.sample(SPRITE_COLORS, 2))
        sprite = np.zeros(mask.shape, dtype=int)
        cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        for _ in range(COLOR_ATTEMPTS):
            group_a = set(random.sample(cells, random.randint(1, len(cells) - 1)))
            sprite.fill(Color.BLACK)
            for r, c in cells:
                sprite[r, c] = colors[0] if (r, c) in group_a else colors[1]
            if set(map(int, sprite[mask].tolist())) == set(colors):
                existing_keys.add(key)
                return {
                    "kind": "sprite",
                    "patch": sprite.copy(),
                    "mask": mask,
                }
    return None


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah + gap <= br or br + bh + gap <= ar or ac + aw + gap <= bc or bc + bw + gap <= ac


def _place_objects(objects: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    max_h = max(obj["patch"].shape[0] for obj in objects)
    max_w = max(obj["patch"].shape[1] for obj in objects)
    buffered_area = sum((obj["patch"].shape[0] + 1) * (obj["patch"].shape[1] + 1) for obj in objects)
    min_side = max(max_h + 2, max_w + 2, int(math.ceil(math.sqrt(buffered_area))))

    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(min(min_side, 30), 30)
        w = random.randint(min(min_side, 30), 30)
        grid = np.full((h, w), Color.BLACK, dtype=int)
        placed_rects: list[tuple[int, int, int, int]] = []
        order = list(range(len(objects)))
        order.sort(key=lambda i: objects[i]["patch"].shape[0] * objects[i]["patch"].shape[1], reverse=True)

        for idx in order:
            patch = objects[idx]["patch"]
            ph, pw = patch.shape
            candidates = []
            for r in range(h - ph + 1):
                for c in range(w - pw + 1):
                    rect = (r, c, ph, pw)
                    if all(_rect_separated(rect, other, gap=1) for other in placed_rects):
                        candidates.append((r, c))
            if not candidates:
                break
            r, c = random.choice(candidates)
            paste(grid, patch, r, c, transparent=Color.BLACK)
            objects[idx]["top_left"] = (r, c)
            objects[idx]["bbox"] = (r, c, r + ph - 1, c + pw - 1)
            placed_rects.append((r, c, ph, pw))
        else:
            return grid, objects
    return None


def _render_expected(inp: np.ndarray, outlines: list[dict], sprites: list[dict], intended_pairs: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for oi, si in intended_pairs:
        outline = outlines[oi]
        sprite = sprites[si]
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        sprite_view = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        sprite_view[sprite["patch"] != Color.BLACK] = Color.BLACK

        r0, c0, r1, c1 = outline["bbox"]
        scaled = scale_sprite(sprite["patch"], outline["factor"])
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[outline["interior"]] = scaled[outline["interior"]]
        view[outline["outline"]] = Color.GREY
    return out


def _semantic_edges(grid: np.ndarray) -> set[tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]]:
    bg = Color.BLACK
    outlines = _outline_objects(grid, bg)
    sprites = _sprite_objects(grid, bg)
    edges = set()
    for oi, candidates in enumerate(_candidate_pairs(outlines, sprites)):
        for si, factor in candidates:
            edges.add((outlines[oi]["bbox"], sprites[si]["bbox"], factor))
    return edges


def _gen_attempt() -> np.ndarray | None:
    existing_keys: set[tuple[tuple[int, int], bytes]] = set()

    matched_count = 1 if random.random() < 0.5 else 2
    unmatched_count = random.choices([0, 1, 2], weights=[3, 5, 2], k=1)[0]
    distractor_count = random.choices([0, 1, 2], weights=[2, 5, 3], k=1)[0]

    outline_specs: list[dict] = []
    sprite_specs: list[dict] = []
    intended_pairs: list[tuple[int, int]] = []

    for _ in range(matched_count):
        built = _build_matched_pair(existing_keys)
        if built is None:
            return None
        outline, sprite = built
        intended_pairs.append((len(outline_specs), len(sprite_specs)))
        outline_specs.append(outline)
        sprite_specs.append(sprite)

    for _ in range(unmatched_count):
        outline = _build_unmatched_outline(existing_keys)
        if outline is None:
            return None
        outline_specs.append(outline)

    for _ in range(distractor_count):
        sprite = _build_distractor_sprite(existing_keys)
        if sprite is None:
            return None
        sprite_specs.append(sprite)

    objects = [obj.copy() for obj in outline_specs + sprite_specs]
    placed = _place_objects(objects)
    if placed is None:
        return None
    inp, objects = placed

    placed_outlines = objects[: len(outline_specs)]
    placed_sprites = objects[len(outline_specs) :]
    expected = _render_expected(inp, placed_outlines, placed_sprites, intended_pairs)

    expected_edges = {
        (placed_outlines[oi]["bbox"], placed_sprites[si]["bbox"], placed_outlines[oi]["factor"])
        for oi, si in intended_pairs
    }
    if _semantic_edges(inp) != expected_edges:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
